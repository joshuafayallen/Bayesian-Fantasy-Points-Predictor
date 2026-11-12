"""Joint model
QBs are modeled separtely and then the residuals are used to help predict skill player production. 


"""
import arviz as az
import numpy as np
import polars as pl
import pymc as pm
import pytensor.tensor as pt
from ffpred import backtest as B
from ffpred import data as D
from ffpred import scoring as Sc
from ffpred.config import MODELS, PATHS, WINDOW_SEASONS
from ffpred.model import HetPrep, _inverse_gamma_maxent, payload
 
CFG_S = MODELS['het_gp_vegas']
CFG_Q = MODELS['het_gp_qb_vegas']
NAME = 'scratch_joint_qb_resid_cut'      # results folder; change per experiment so runs don't overwrite
UNCUT = 'scratch_joint_qb_resid'         # earlier joint run, for cut-vs-uncut
SEED = 41029041
JITTER = 1e-6
DUD_LINE = 0.05
STARTER_COLS = ('attempts', 'pass_attempts', 'offense_snaps', 'offense_pct', 'trail_pts')
RHAT_Q = ['qb_sigma_obs', 'qb_sigma_skill', 'qb_eta_season', 'qb_ell_season']
RHAT_S = ['lam_rec', 'sigma_skill', 'eta_season', 'ell_season', 'log_cv0', 'delta_cv']
TE_COLS = ['player_id', 'player', 'position', 'team', 'opp_team', 'season', 'week', 'total_fantasy_points']
CALIB = PATHS.results_dir / 'backtest' / 'calib'
SAMPLE_KW = dict(draws=2000, tune=2000, chains=4, nuts_sampler='nutpie', target_accept=0.95, random_seed=SEED)
 
 
#  ============================ helpers (rarely touched) ============================
def fold_frames(season, week, history, team_form, group, window_seasons=WINDOW_SEASONS):
    week = float(week)
    df = D.join_team_form(history, team_form)
    df = df.sort('player_id', 'season', 'week').with_columns(
        pl.col('total_fantasy_points').shift(1)
          .rolling_mean(window_size=8, min_samples=1)
          .over('player_id').alias('trail_pts'))
    cfg = MODELS[group]
    ref = df.filter(D.before(season, week) | ((pl.col('season') == season) & (pl.col('week') == week)))
    df = D.with_primary_position(df, reference=ref).filter(pl.col('position').is_in(list(cfg.positions)))
    train = D.training_window(df, season, week, window_seasons)
    test = D.week_rows(df, season, week).sort('player_id')
    pos_mean = train.group_by('position').agg(pl.col('total_fantasy_points').mean().alias('_pm'))
    fill = lambda f: (f.join(pos_mean, on='position', how='left')
                       .with_columns(pl.col('trail_pts').fill_null(pl.col('_pm'))).drop('_pm'))
    train, test = fill(train), fill(test)
    keep = test.filter(pl.Series(HetPrep(train, cfg).eligible(test)))
    return train, keep, test.height
 
 
def frames(s, w, history, team_form):
    """Train frames sorted as the models see them; test frames sorted by player_id (draw column order)."""
    tr_s, te_s, _ = fold_frames(s, w, history, team_form, 'het_gp_vegas')
    tr_q, te_q, _ = fold_frames(s, w, history, team_form, 'het_gp_qb_vegas')
    key = ('player_id', 'season', 'week')
    return tr_s.sort(*key), te_s.sort('player_id'), tr_q.sort(*key), te_q.sort('player_id')
 
 
def _tg_key(f):
    return f.select(pl.concat_str([pl.col('season').cast(pl.String), pl.col('week').cast(pl.String),
                                   pl.col('team')], separator='|')).to_series().to_list()
 
 
def starter_row(frame_q, frame_s, test=False):
    """Skill row -> row of its team-game's starting QB in frame_q (frame_q.height = no QB)."""
    col = 'trail_pts' if test else next(c for c in STARTER_COLS if c in frame_q.columns)
    kq = _tg_key(frame_q)
    v = frame_q[col].fill_null(-1e9).to_numpy()
    best = {}
    for i, (k, x) in enumerate(zip(kq, v)):
        if k not in best or x > v[best[k]]:
            best[k] = i
    none = frame_q.height
    rows = np.array([best.get(k, none) for k in _tg_key(frame_s)], dtype=int)
    return rows, (rows < none).astype(float), col
 
 
def qb_payload(prep_q, frame):
    return {f'qb_{k}': v for k, v in payload(prep_q.arrays(frame)).items()}
 
 
def summarize(S, frame, top=10, bottom=5):
    summ = frame.select('player', 'position', 'team').with_columns(
        pl.Series('exp', S.mean(0)), pl.Series('p10', np.percentile(S, 10, 0)),
        pl.Series('p90', np.percentile(S, 90, 0)), pl.Series('sd', S.std(0)),
        pl.Series('actual', frame['total_fantasy_points'].to_numpy().astype(float)))
    print(summ.group_by('position').agg(pl.col('exp', 'actual', 'sd').mean()).sort('position'))
    print(summ.sort('exp', descending=True).head(top))
    print(summ.sort('exp').head(bottom))
    return summ
 
 
def stack_corr(S_s, S_q, te_s, te_q, team):
    """Draw correlation of a team's skill players with its starting QB (top projected QB)."""
    qi = te_q.with_row_index().filter(pl.col('team') == team)['index'].to_numpy()
    if qi.size == 0:
        print('no QB row for', team); return
    q = qi[np.argmax(S_q[:, qi].mean(0))]
    for i, nm, p in te_s.with_row_index().filter(pl.col('team') == team) \
                        .select('index', 'player', 'position').iter_rows():
        print(f'{p} {nm:<24} corr with QB {np.corrcoef(S_q[:, q], S_s[:, i])[0, 1]:+.2f}')
 
 
#  ============================ MODEL: QB block (stage 1, production het_gp_qb_vegas) ============================
def build_qb(tr_q):
    pq = HetPrep(tr_q, CFG_Q)
    Aq = pq.arrays(tr_q)
    Xq = pq.grid[:, None]
    coords = {'qb_player': pq.coords['player_id'], 'qb_time': pq.coords['time'],
              'qb_cont_vars': pq.coords['cont_vars'], 'qb_binary_vars': pq.coords['binary_vars']}
 
    with pm.Model(coords=coords) as model:
        q_player = pm.Data('qb_player_data', Aq['player'])
        pm.Data('qb_position_data', Aq['position'])
        q_t      = pm.Data('qb_time_data', Aq['time'])
        q_cont   = pm.Data('qb_cont_data', Aq['cont'])
        q_bi     = pm.Data('qb_bi_data', Aq['bi'])
        q_tf     = pm.Data('qb_team_form_data', Aq['team_form_z'])
        q_otf    = pm.Data('qb_opp_team_form_data', Aq['opp_team_form_z'])
        q_obs    = pm.Data('qb_obs_data', Aq['y_z'])
 
        q_alpha  = pm.Normal('qb_alpha', 0.0, 0.5)
        q_tpeak  = pm.Normal('qb_t_peak', *CFG_Q.t_peak)
        q_height = pt.exp(pm.Normal('qb_log_height', np.log(0.5), 0.4))
        q_w_rise = 1.0 + pm.Gamma('qb_width_rise_raw', alpha=4, beta=1.5)
        q_w_dec  = 1.0 + pm.Gamma('qb_width_decline_raw', alpha=8, beta=1.2)
        q_width  = pt.switch(pq.grid < q_tpeak, q_w_rise, q_w_dec)
        q_curve  = q_height * pt.exp(-0.5 * ((pq.grid - q_tpeak) / q_width) ** 2)
        q_career = q_curve - (pq.t_weights_pos[:, 0] * q_curve).sum()
 
        q_sigma_skill = pm.HalfNormal('qb_sigma_skill', 1.0)
        q_skill = q_sigma_skill * pm.ZeroSumNormal('qb_z_skill', sigma=1.0, dims='qb_player')
 
        q_ell = pm.InverseGamma('qb_ell_season', *pq.season_prior)
        q_eta = pm.HalfNormal('qb_eta_season', 0.5)
        L_q = pt.linalg.cholesky(pm.gp.util.stabilize((q_eta**2 * pm.gp.cov.Matern32(1, ls=q_ell))(Xq), JITTER))
        q_zs = pm.Normal('qb_z_season', 0.0, 1.0, dims=('qb_player', 'qb_time'))
        q_fr = q_zs @ L_q.T
        q_fp = q_fr - ((q_fr * pq.active).sum(1) / pq.n_seasons_player)[:, None]
        q_season = q_fp - (q_fp * pq.active).sum(0) / pq.n_active
 
        q_cont_prior = pm.Normal('qb_cont_prior', 0.0, 0.5, dims='qb_cont_vars')
        q_bi_prior   = pm.Normal('qb_bi_prior', 0.0, 0.5, dims='qb_binary_vars')
        q_tf_prior   = pm.Normal('qb_team_form_prior', 0.0, 0.5)
        q_otf_prior  = pm.Normal('qb_opp_form_prior', 0.0, 0.5)
 
        q_mu = pm.Deterministic('qb_mu',
                                q_alpha + q_career[q_t] + q_skill[q_player] + q_season[q_player, q_t]
                                + pt.dot(q_cont, q_cont_prior) + pt.dot(q_bi, q_bi_prior)
                                + q_tf_prior * q_tf + q_otf_prior * q_otf)
        q_sigma = pm.HalfNormal('qb_sigma_obs', 1.0)
        pm.Normal('qb_y_obs', mu=q_mu, sigma=q_sigma, observed=q_obs)
    return model, pq
 
 
def qb_residuals(idf_q, pq, tr_q):
    """Plug-in standardised residual per training QB row: (y - E[mu]) / E[sigma]."""
    mu_hat  = idf_q.posterior['qb_mu'].mean(('chain', 'draw')).values
    sig_hat = float(idf_q.posterior['qb_sigma_obs'].mean())
    return (pq.arrays(tr_q)['y_z'] - mu_hat) / sig_hat
 
 
#  ============================ MODEL: skill block (stage 2, v3 + QB residual) ============================
def build_skill(tr_s, tr_q, r_q):
    ps = HetPrep(tr_s, CFG_S)
    As = ps.arrays(tr_s)
    Xs = ps.grid[:, None]
    m_ref_pos = (tr_s.group_by('position').agg(pl.col('total_fantasy_points').mean())
                 .join(pl.DataFrame({'position': ps.positions}), on='position', how='right')
                 ['total_fantasy_points'].to_numpy())
    qrow, has, ps.starter_col = starter_row(tr_q, tr_s)
    r_ext = np.append(r_q, 0.0)[qrow]                        # QB residual per skill row (0 if no QB)
 
    with pm.Model(coords=ps.coords) as model:
        # ---------------- QB residual -> receivers (fixed data: the cut) ----------------
        r_d    = pm.Data('r_qb_data', r_ext)
        has_d  = pm.Data('has_qb_data', has)
        lam_rec = pm.Normal('lam_rec', 0.0, 0.3, dims='position')
 
        player = pm.Data('player_data', As['player'])
        pos    = pm.Data('position_data', As['position'])
        t      = pm.Data('time_data', As['time'])
        cont   = pm.Data('cont_data', As['cont'])
        bi     = pm.Data('bi_data', As['bi'])
        tf     = pm.Data('team_form_data', As['team_form_z'])
        otf    = pm.Data('opp_team_form_data', As['opp_team_form_z'])
        obs    = pm.Data('obs_data', As['y_z'])
        trail  = pm.Data('trail_data', tr_s['trail_pts'].to_numpy().astype(float))
        log_mref = pt.as_tensor(np.log(m_ref_pos))
 
        alpha_pos  = pm.Normal('alpha_pos', 0.15, 0.3, dims='position')
        t_peak_mu  = pm.Normal('t_peak_mu', 4.0, 1.25)
        t_peak_off = pm.ZeroSumNormal('t_peak_off', sigma=1, dims='position')
        t_peak     = t_peak_mu + t_peak_off
        log_height_mu  = pm.Normal('log_height_mu', np.log(0.2), 0.4)
        log_height_off = pm.ZeroSumNormal('log_height_off', sigma=0.25, dims='position')
        peak_height    = pt.exp(log_height_mu + log_height_off)
        peak_width     = 1.0 + pm.Gamma('peak_width_raw', alpha=6, beta=1.5)
        curve = peak_height[None, :] * pt.exp(-0.5 * ((ps.grid[:, None] - t_peak[None, :]) / peak_width) ** 2)
        f_career = curve - (ps.t_weights_pos * curve).sum(0)
 
        sigma_skill = pm.HalfNormal('sigma_skill', 0.35)
        z_skill     = pm.Normal('z_skill', 0.0, 1.0, dims='player_id')
        skill_raw   = sigma_skill * z_skill
        skill = skill_raw - ((ps.pos_onehot.T @ skill_raw) / ps.pos_counts)[ps.position_of_player]
 
        ell_season = pm.InverseGamma('ell_season', *_inverse_gamma_maxent(0.8, 2.5))
        eta_season = pm.HalfNormal('eta_season', 0.15)
        L_s = pt.linalg.cholesky(pm.gp.util.stabilize(
            (eta_season**2 * pm.gp.cov.Matern32(1, ls=ell_season))(Xs), JITTER))
        z_season = pm.Normal('z_season', 0.0, 1.0, dims=('player_id', 'time'))
        f_raw = z_season @ L_s.T
        f_p = f_raw - ((f_raw * ps.active).sum(1) / ps.n_seasons_player)[:, None]
        f_season = f_p - (f_p * ps.active).sum(0) / ps.n_active
 
        cont_prior      = pm.Normal('cont_prior', 0.0, 0.08, dims='cont_vars')
        bi_prior        = pm.Normal('bi_prior', 0.0, 0.08, dims='binary_vars')
        team_form_prior = pm.Normal('team_form_prior', 0.0, 0.08)
        opp_form_prior  = pm.Normal('opp_form_prior', 0.0, 0.08)
 
        shock = has_d * (lam_rec[pos] * r_d - 0.5 * lam_rec[pos] ** 2)
        eta = (log_mref[pos] + alpha_pos[pos] + f_career[t, pos] + skill[player] + f_season[player, t]
               + pt.dot(cont, cont_prior) + pt.dot(bi, bi_prior)
               + team_form_prior * tf + opp_form_prior * otf + shock)
        m = pt.exp(eta)
 
        y_h = pt.switch(ps.y_mu + ps.y_sd * obs > DUD_LINE, ps.y_mu + ps.y_sd * obs, 0.0)
        a_psi  = pm.Normal('a_psi', 2.0, 0.5, dims='position')
        w_psi  = pm.Normal('w_psi', 0.0, 0.5, dims='cont_vars')
        wb_psi = pm.Normal('wb_psi', 0.0, 0.5, dims='binary_vars')
        psi = pm.math.sigmoid(a_psi[pos] + pt.dot(cont, w_psi) + pt.dot(bi, wb_psi))
 
        log_cv0 = pm.Normal('log_cv0', np.log(0.6), 0.15, dims='position')
        lvl     = pt.clip(pt.log(pt.maximum(trail, 1.0)) - log_mref[pos], -1.5, 1.5)
        delta   = pm.Normal('delta_cv', -0.3, 0.2, dims='position')
        cv_raw  = pt.exp(log_cv0[pos] + delta[pos] * lvl)
        cv      = cv_raw / pt.sqrt(1.0 + (cv_raw / 1.2) ** 2)
        pm.HurdleGamma('y_obs', psi=psi, mu=m, sigma=cv * m, observed=y_h)
    return model, ps
 
 
#  ============================ fit / predict ============================
def fit(tr_s, tr_q, **kw):
    """Stage 1 (QB), plug-in residuals, stage 2 (skill). Returns both fits."""
    kw = {**SAMPLE_KW, **kw}
    mq, pq = build_qb(tr_q)
    with mq:
        idf_q = pm.sample(**kw)
    r_q = qb_residuals(idf_q, pq, tr_q)
    ms, ps = build_skill(tr_s, tr_q, r_q)
    with ms:
        idf_s = pm.sample(**kw)
    return (mq, idf_q, pq), (ms, idf_s, ps), r_q
 
 
def predict(qfit, sfit, te_s, te_q, seed=SEED):
    """Correlated predictive draws in points. Row i of S_s and S_q share the same QB eps.
    The two posteriors are independent (cut), so pairing draw i of each is valid."""
    (mq, idf_q, pq), (ms, idf_s, ps) = qfit, sfit
    n_s, n_q = te_s.height, te_q.height
    qcol, has_te, _ = starter_row(te_q, te_s, test=True)
 
    with mq:
        pm.set_data(qb_payload(pq, te_q))
        pp_q = pm.sample_posterior_predictive(idf_q, var_names=['qb_mu'], random_seed=seed + 1, progressbar=False)
    with ms:
        pm.set_data({**payload(ps.arrays(te_s)),
                     'trail_data': te_s['trail_pts'].to_numpy().astype(float),
                     'r_qb_data': np.zeros(n_s), 'has_qb_data': np.zeros(n_s)})       # shock off: added below
        pp_s = pm.sample_posterior_predictive(idf_s, var_names=['y_obs'], random_seed=seed + 2, progressbar=False)
 
    mu_q = pp_q.posterior_predictive['qb_mu'].values.reshape(-1, n_q)            # z units
    sig  = idf_q.posterior['qb_sigma_obs'].values.reshape(-1, 1)
    eps  = np.random.default_rng(seed + 3).standard_normal(mu_q.shape)
    Sq_z = mu_q + sig * eps                                                       # QB draws, z units
    S_q  = Sq_z * pq.y_sd + pq.y_mu

    # QB surprise on the training-r scale: deviation of the draw from its predictive mean, / E[sigma]
    # (includes mean uncertainty, which receivers react to in reality; eps alone missed it)
    e_q = (Sq_z - Sq_z.mean(0)) / sig.mean()
    v_q = e_q.var(0)                                                              # >1: the extra mean uncertainty

    S0  = pp_s.posterior_predictive['y_obs'].values.reshape(-1, n_s)
    lam = idf_s.posterior['lam_rec'].values.reshape(-1, len(ps.positions))[:, ps.arrays(te_s)['position']]
    n = min(len(S0), len(S_q))
    qc  = np.minimum(qcol, n_q - 1)
    on  = has_te[None, :] > 0
    e_s = np.where(on, e_q[:n, qc], 0.0)
    v_s = np.where(on, v_q[qc][None, :], 0.0)
    S_s = S0[:n] * np.exp(np.where(on, lam[:n] * e_s - 0.5 * lam[:n] ** 2 * v_s, 0.0))   # mean-preserving
    return S_s, S_q[:n], has_te
 
 
def diagnostics(idf_q, idf_s, ps):
    rq = az.summary(idf_q, var_names=RHAT_Q, kind='diagnostics')['r_hat'].max()
    rs = az.summary(idf_s, var_names=RHAT_S, kind='diagnostics')['r_hat'].max()
    lam = idf_s.posterior['lam_rec'].mean(('chain', 'draw')).values
    return dict(divergences=int(idf_q.sample_stats['diverging'].sum() + idf_s.sample_stats['diverging'].sum()),
                rhat_max=float(max(rq, rs)),
                **{f'lam_rec_{p}': float(v) for p, v in zip(ps.positions, lam)})
 
 
#  ============================ data (run once) ============================
history, team_form = D.load_processed(), D.load_team_form()
s, w = 2025, 9
tr_s, te_s, tr_q, te_q = frames(s, w, history, team_form)
 
 
#  ============================ fit (both stages) ============================
qfit, sfit, r_q = fit(tr_s, tr_q)
(mq, idf_q, pq), (ms, idf_s, ps) = qfit, sfit
print('starter col:', ps.starter_col, '  QB residual mean/sd:', r_q.mean().round(3), r_q.std().round(3))  # ~0 / ~1
print(az.summary(idf_q, var_names=RHAT_Q, kind='diagnostics'))
print(az.summary(idf_s, var_names=RHAT_S, kind='diagnostics'))
print(diagnostics(idf_q, idf_s, ps))
 
 
#  ============================ trace / ESS plots ============================
az.rcParams['plot.max_subplots'] = 60
for m_, idf_ in ((mq, idf_q), (ms, idf_s)):
    small = [rv.name for rv in m_.free_RVs if rv.size.eval() <= 10]
    az.plot_trace(idf_, var_names=small)
    az.plot_ess_evolution(idf_, var_names=small)
 
 
#  ============================ predict the test week ============================
S_s, S_q, has_te = predict(qfit, sfit, te_s, te_q)
print('has_qb', has_te.mean().round(2), '  max draw', round(float(max(S_s.max(), S_q.max())), 1))
summ_s = summarize(S_s, te_s)
summ_q = summarize(S_q, te_q)
stack_corr(S_s, S_q, te_s, te_q, 'KC')
 
 
#  ============================ score this week vs truth ============================
for label, S, te in (('skill', S_s, te_s), ('qb', S_q, te_q)):
    sc = Sc.score(te['total_fantasy_points'].to_numpy().astype(float), S); sc.pop('pit_hist')
    print(label, {k: round(float(v), 3) for k, v in sc.items() if np.isscalar(v)})
    with pl.Config(tbl_rows=-1, float_precision=2):
        print(Sc.calibration_table(te, S, by_position=True))
 
 
#  ============================ BACKTEST: 12 folds (uses the MODEL cells as they stand) ============================
OUT = PATHS.results_dir / NAME
OUT.mkdir(parents=True, exist_ok=True)
rows, tiers = [], []
 
for s_bt in (2021,2022,2023, 2024, 2025):
    for w_bt in (1,2,3,4,5,6,7, 8, 9, 10,11, 12,13,14,15):
        tr_s_bt, te_s_bt, tr_q_bt, te_q_bt = frames(s_bt, w_bt, history, team_form)
        qf, sf, _ = fit(tr_s_bt, tr_q_bt, progressbar=False)
        Ss, Sq, has = predict(qf, sf, te_s_bt, te_q_bt)
 
        diag = dict(**diagnostics(qf[1], sf[1], sf[2]),
                    has_qb_test=float(has.mean()), max_draw=float(max(Ss.max(), Sq.max())))
        for label, S, te in (('skill', Ss, te_s_bt), ('qb', Sq, te_q_bt)):
            sc = Sc.score(te['total_fantasy_points'].to_numpy().astype(float), S); sc.pop('pit_hist')
            rows.append(dict(model=f'{NAME}_{label}', season=s_bt, week=float(w_bt),
                             eligibility='model', **sc, **diag))
            tiers.append(Sc.calibration_table(te, S, by_position=True).rename({'group': 'pos'})
                         .with_columns(pl.lit(f'{NAME}_{label}').alias('model'),
                                       pl.lit(s_bt).alias('season'), pl.lit(float(w_bt)).alias('week')))
 
        tag = f'{s_bt}_{w_bt:02d}'
        np.save(OUT / f'S_{tag}.npy', np.concatenate([Sq, Ss], axis=1).astype(np.float32))
        pl.concat([te_q_bt, te_s_bt], how='diagonal_relaxed').select(TE_COLS).write_parquet(OUT / f'te_{tag}.parquet')
        pl.DataFrame(rows).write_parquet(OUT / 'folds.parquet')
        pl.concat(tiers).write_parquet(OUT / 'tiers.parquet')
        print(s_bt, w_bt, f"crps skill {rows[-2]['crps']:.3f} qb {rows[-1]['crps']:.3f}  "
                          f"div {diag['divergences']}  rhat {diag['rhat_max']:.3f}  max {diag['max_draw']:.0f}  "
                          f"has_qb {diag['has_qb_test']:.2f}  "
                          + '  '.join(f"{p} {diag[f'lam_rec_{p}']:+.2f}" for p in sf[2].positions))
        del qf, sf
 
 
#  ============================ BACKTEST: compare with production and the uncut run ============================
PAIRS = {f'{NAME}_skill': 'het_gp_vegas', f'{NAME}_qb': 'het_gp_qb_vegas'}
folds = pl.read_parquet(OUT / 'folds.parquet')
tiers_df = pl.read_parquet(OUT / 'tiers.parquet')
prod = list(PAIRS.values())
folds_all = pl.concat([B.load_run(CALIB)[0].filter(pl.col('model').is_in(prod)), folds,
                       pl.read_parquet(PATHS.results_dir / UNCUT / 'folds.parquet')], how='diagonal_relaxed')
tiers_all = pl.concat([B.load_tiers(CALIB).filter(pl.col('model').is_in(prod)), tiers_df], how='diagonal_relaxed')

folds_all['model'].unique()

for new, base in PAIRS.items():
    for metric in ('crps', 'mae', 'rmse'):
        d = B.paired_diff(folds_all, new, base, metric)                  # negative = new better
        print(f"{new:<34} vs {base:<16} {metric:<5} {d['mean_diff']:+.3f} ± {d['se']:.3f}  "
              f"better {d['folds_a_better']}/{d['n_folds']}")
for label in ('skill', 'qb'):                                           # QB should improve, skill should not move
    d = B.paired_diff(folds_all, f'{NAME}_{label}', f'{UNCUT}_{label}', 'crps')
    print(f"cut vs uncut {label:<5} crps {d['mean_diff']:+.3f} ± {d['se']:.3f}  "
          f"better {d['folds_a_better']}/{d['n_folds']}")
 
with pl.Config(tbl_cols=-1, tbl_rows=-1, float_precision=2):
    print(B.tier_summary(tiers_all)
          .with_columns((pl.col('mean_actual') - pl.col('mean_expected')).alias('bias'),
                        (pl.col('pred_sd') / pl.col('resid_sd')).alias('sd_ratio'))
          .select('model', 'tier', 'n', 'bias', 'sd_ratio', 'below_p10', 'above_p90'))
    print(folds.filter(pl.col('model').str.ends_with('_skill'))
          .select('season', 'week', 'crps', 'divergences', 'rhat_max', 'max_draw', pl.col('^lam_rec_.*$')))
 
 
#  ============================ BACKTEST: blow-up check ============================
for p in sorted(OUT.glob('S_*.npy')):
    S = np.load(p).astype('float64')
    te = pl.read_parquet(OUT / p.name.replace('S_', 'te_').replace('.npy', '.parquet'))
    bad = np.flatnonzero(S.max(0) > 150)
    if bad.size:
        print(p.name, f'max {S.max():.3g}')
        print(te[bad].select('player', 'position', 'team').with_columns(
            pl.Series('max', S.max(0)[bad]), pl.Series('n_draws_gt150', (S[:, bad] > 150).sum(0))))
 

from scipy import stats
rng = np.random.default_rng(11)

def own_qb_model_corr(S_s, S_q, te_s, te_q):
    """Mean draw correlation (normal scores) of each receiver with his team's starting QB, by position."""
    qcol, has, _ = starter_row(te_q, te_s, test=True)
    zq = stats.norm.ppf((S_q.argsort(0).argsort(0) + 1) / (len(S_q) + 1))
    zs = stats.norm.ppf((S_s.argsort(0).argsort(0) + 1) / (len(S_s) + 1))
    c = np.array([np.corrcoef(zq[:, qcol[j]], zs[:, j])[0, 1] if has[j] else np.nan for j in range(te_s.height)])
    return (pl.DataFrame({'position': te_s['position'], 'corr': c}).drop_nans()
              .group_by('position').agg(pl.col('corr').mean()).sort('position'))

S_s_new, S_q_new, _ = predict(qfit, sfit, te_s, te_q)          # the new predict you just defined

# how much extra QB uncertainty the old version ignored
(mq, idf_q, pq) = qfit
with mq:
    pm.set_data(qb_payload(pq, te_q))
    mu_q = pm.sample_posterior_predictive(idf_q, var_names=['qb_mu'], random_seed=SEED + 1,
                                          progressbar=False).posterior_predictive['qb_mu'].values.reshape(-1, te_q.height)
sig = float(idf_q.posterior['qb_sigma_obs'].mean())
v = 1 + mu_q.var(0) / sig**2
print('QB surprise variance / noise variance: median', np.median(v).round(2), ' range', v.min().round(2), '-', v.max().round(2))

with pl.Config(float_precision=3):
    print('new predict, model corr with own QB (2025 wk9):')
    print(own_qb_model_corr(S_s_new, S_q_new, te_s, te_q))
print('old predict, 12 folds: WR 0.177  TE 0.147  RB 0.051   |   actual: WR 0.238  TE 0.184  RB 0.003')


def corr_table(df, by):
    return (df.group_by(by)
              .agg(pl.len().alias('n'),
                   pl.corr('z_qb', 'z_rec').alias('actual_corr'),
                   pl.col('model_corr').mean().alias('model_corr'))
              .with_columns((1 / (pl.col('n') - 3).sqrt()).alias('se_actual'))      # Fisher-z approx
              .sort(by))

with pl.Config(tbl_rows=-1, float_precision=3):
    print(corr_table(P.filter((pl.col('side') == 'own') & (pl.col('position') != 'RB')), ['season', 'position']))


pairs = []
for s, w in FOLDS:
    S, te = load(s, w)
    y = te['total_fantasy_points'].to_numpy().astype(float)
    z_act, z_draw = normal_scores(S, y)
    qb, proj = starters(te, S), S.mean(0)
    for c, pos, team in te.filter(pl.col('position').is_in(['WR', 'TE'])).select('col', 'position', 'team').iter_rows():
        if team in qb:
            q = qb[team]
            tier = '<6' if proj[c] < 6 else '6-12' if proj[c] < 12 else '12+'
            pairs.append(dict(position=pos, tier=tier, z_qb=z_act[q], z_rec=z_act[c],
                              model_corr=float(np.corrcoef(z_draw[:, q], z_draw[:, c])[0, 1])))
with pl.Config(tbl_rows=-1, float_precision=3):
    print(corr_table(pl.DataFrame(pairs), ['tier']).with_columns(
        (pl.col('actual_corr') - pl.col('model_corr')).alias('gap')))