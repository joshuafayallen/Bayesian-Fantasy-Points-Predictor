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
NAME = 'scratch_joint_qb_teamseason'     # results folder
BASE = 'scratch_joint_qb_resid_cut'      # cut model without tau, for tau-vs-no-tau
SEED = 41029041
JITTER = 1e-6
DUD_LINE = 0.05
STARTER_COLS = ('attempts', 'pass_attempts', 'offense_snaps', 'offense_pct', 'trail_pts')
RHAT_Q = ['qb_sigma_obs', 'qb_sigma_skill', 'qb_eta_season', 'qb_ell_season']
RHAT_J = ['kappa_q', 'kappa_s', 'lam_rec', 'qb_sigma_obs', 'sigma_skill', 'eta_season', 'ell_season',
          'log_cv0', 'delta_cv']
TE_COLS = ['player_id', 'player', 'position', 'team', 'opp_team', 'season', 'week', 'total_fantasy_points']
CALIB = PATHS.results_dir / 'backtest' / 'calib'
SAMPLE_KW = dict(draws=1000, tune=1000, chains=4, nuts_sampler='nutpie', target_accept=0.95, random_seed=SEED)
 
 
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
    tr_s, te_s, _ = fold_frames(s, w, history, team_form, 'het_gp_vegas')
    tr_q, te_q, _ = fold_frames(s, w, history, team_form, 'het_gp_qb_vegas')
    key = ('player_id', 'season', 'week')
    return tr_s.sort(*key), te_s.sort('player_id'), tr_q.sort(*key), te_q.sort('player_id')
 
 
def _tg_key(f):
    return f.select(pl.concat_str([pl.col('season').cast(pl.String), pl.col('week').cast(pl.String),
                                   pl.col('team')], separator='|')).to_series().to_list()
 
 
def _ts_key(f):
    return f.select(pl.concat_str([pl.col('season').cast(pl.String), pl.col('team')], separator='|')) \
            .to_series().to_list()
 
 
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
 
 
def ts_index(tr_s, tr_q, te_s, te_q):
    """One team-season level per season|team, over train and test rows of both blocks."""
    keys = {'s_tr': _ts_key(tr_s), 'q_tr': _ts_key(tr_q), 's_te': _ts_key(te_s), 'q_te': _ts_key(te_q)}
    levels = list(dict.fromkeys(keys['s_tr'] + keys['q_tr'] + keys['s_te'] + keys['q_te']))
    lut = {k: i for i, k in enumerate(levels)}
    return levels, {k: np.array([lut[x] for x in v], dtype=int) for k, v in keys.items()}
 
 
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
    qi = te_q.with_row_index().filter(pl.col('team') == team)['index'].to_numpy()
    if qi.size == 0:
        print('no QB row for', team); return
    q = qi[np.argmax(S_q[:, qi].mean(0))]
    for i, nm, p in te_s.with_row_index().filter(pl.col('team') == team) \
                        .select('index', 'player', 'position').iter_rows():
        print(f'{p} {nm:<24} corr with QB {np.corrcoef(S_q[:, q], S_s[:, i])[0, 1]:+.2f}')
 
 
#  ============================ QB block (shared by stage A and stage B) ============================
def qb_block(pq, Aq, tau_term=0.0):
    """Production het_gp_qb_vegas inside the current pm.Model. tau_term is added to the QB mean (z units)."""
    Xq = pq.grid[:, None]
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
                            + q_tf_prior * q_tf + q_otf_prior * q_otf + tau_term)
    q_sigma = pm.HalfNormal('qb_sigma_obs', 1.0)
    pm.Normal('qb_y_obs', mu=q_mu, sigma=q_sigma, observed=q_obs)
 
 
def qb_coords(pq):
    return {'qb_player': pq.coords['player_id'], 'qb_time': pq.coords['time'],
            'qb_cont_vars': pq.coords['cont_vars'], 'qb_binary_vars': pq.coords['binary_vars']}
 
 
#  ============================ stage A: QB alone -> game residual r ============================
def build_qb(tr_q):
    pq = HetPrep(tr_q, CFG_Q)
    with pm.Model(coords=qb_coords(pq)) as model:
        qb_block(pq, pq.arrays(tr_q))
    return model, pq
 
 
def qb_residuals(idf_q, pq, tr_q):
    mu_hat  = idf_q.posterior['qb_mu'].mean(('chain', 'draw')).values
    sig_hat = float(idf_q.posterior['qb_sigma_obs'].mean())
    return (pq.arrays(tr_q)['y_z'] - mu_hat) / sig_hat
 
 
#  ============================ stage B: joint model with tau (tweak here) ============================
def build_joint(tr_s, tr_q, te_s, te_q, r_q):
    ps, pq = HetPrep(tr_s, CFG_S), HetPrep(tr_q, CFG_Q)
    As, Aq = ps.arrays(tr_s), pq.arrays(tr_q)
    Xs = ps.grid[:, None]
    m_ref_pos = (tr_s.group_by('position').agg(pl.col('total_fantasy_points').mean())
                 .join(pl.DataFrame({'position': ps.positions}), on='position', how='right')
                 ['total_fantasy_points'].to_numpy())
    qrow, has, ps.starter_col = starter_row(tr_q, tr_s)
    r_row = np.append(r_q, 0.0)[qrow]
    ts_levels, ts = ts_index(tr_s, tr_q, te_s, te_q)
    ps.ts_test, pq.ts_test = ts['s_te'], ts['q_te']
 
    with pm.Model(coords={**ps.coords, **qb_coords(pq), 'team_season': ts_levels}) as model:
        # ---------------- shared team-season offence level ----------------
        tau     = pm.Normal('tau', 0.0, 1.0, dims='team_season')
        kappa_q = pm.HalfNormal('kappa_q', 0.3)                       # QB z units per tau sd
        kappa_s = pm.Normal('kappa_s', 0.0, 0.1, dims='position')    # skill log scale per tau sd
        ts_q = pm.Data('qb_ts_data', ts['q_tr'])
        ts_s = pm.Data('ts_data', ts['s_tr'])
 
        # ---------------- QB block (production + tau) ----------------
        qb_block(pq, Aq, tau_term=kappa_q * tau[ts_q])
 
        # ---------------- game-level QB link (cut: r is data from stage A) ----------------
        r_d    = pm.Data('r_qb_data', r_row)
        has_d  = pm.Data('has_qb_data', has)
        lam_rec = pm.Normal('lam_rec', 0.0, 0.3, dims='position')
 
        # ---------------- skill block (v3) ----------------
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
 
        game  = has_d * (lam_rec[pos] * r_d - 0.5 * lam_rec[pos] ** 2)
        level = kappa_s[pos] * tau[ts_s] - 0.5 * kappa_s[pos] ** 2
        eta = (log_mref[pos] + alpha_pos[pos] + f_career[t, pos] + skill[player] + f_season[player, t]
               + pt.dot(cont, cont_prior) + pt.dot(bi, bi_prior)
               + team_form_prior * tf + opp_form_prior * otf + game + level)
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
    return model, ps, pq
 
 
#  ============================ fit / predict ============================
def fit(tr_s, tr_q, te_s, te_q, **kw):
    kw = {**SAMPLE_KW, **kw}
    mq, pq_a = build_qb(tr_q)                                   # stage A
    with mq:
        idf_a = pm.sample(**kw)
    r_q = qb_residuals(idf_a, pq_a, tr_q)
    mj, ps, pq = build_joint(tr_s, tr_q, te_s, te_q, r_q)      # stage B
    with mj:
        idf = pm.sample(**kw)
    return mj, idf, ps, pq, r_q
 
 
def predict(mj, idf, ps, pq, te_s, te_q, seed=SEED):
    """Same posterior draw -> same tau for a team's QB and receivers; plus a shared QB eps per game."""
    n_s, n_q = te_s.height, te_q.height
    qcol, has_te, _ = starter_row(te_q, te_s, test=True)
    with mj:
        pm.set_data({**payload(ps.arrays(te_s)),
                     'trail_data': te_s['trail_pts'].to_numpy().astype(float),
                     'ts_data': ps.ts_test, 'qb_ts_data': pq.ts_test,
                     'r_qb_data': np.zeros(n_s), 'has_qb_data': np.zeros(n_s),     # game link added below
                     **qb_payload(pq, te_q)})
        pp = pm.sample_posterior_predictive(idf, var_names=['y_obs', 'qb_mu'], random_seed=seed + 1,
                                            progressbar=False)
    mu_q = pp.posterior_predictive['qb_mu'].values.reshape(-1, n_q)            # includes kappa_q * tau
    sig  = idf.posterior['qb_sigma_obs'].values.reshape(-1, 1)
    eps  = np.random.default_rng(seed + 3).standard_normal(mu_q.shape)
    Sq_z = mu_q + sig * eps
    S_q  = Sq_z * pq.y_sd + pq.y_mu
 
    # game-level link: the QB's within-game surprise (eps). tau already makes the mean errors co-move,
    # so the game link uses eps on the training-r scale (sig / E[sig] ~ 1), not the whole deviation.
    e_q = sig * eps / sig.mean()
    S0  = pp.posterior_predictive['y_obs'].values.reshape(-1, n_s)            # includes the tau level
    lam = idf.posterior['lam_rec'].values.reshape(-1, len(ps.positions))[:, ps.arrays(te_s)['position']]
    qc  = np.minimum(qcol, n_q - 1)
    on  = has_te[None, :] > 0
    e_s = np.where(on, e_q[:, qc], 0.0)
    v_s = np.where(on, (e_q.var(0))[qc][None, :], 0.0)
    S_s = S0 * np.exp(np.where(on, lam * e_s - 0.5 * lam ** 2 * v_s, 0.0))
    return S_s, S_q, has_te
 
 
def diagnostics(idf, ps, S_s, S_q, has_te):
    lam = idf.posterior['lam_rec'].mean(('chain', 'draw')).values
    ks  = idf.posterior['kappa_s'].mean(('chain', 'draw')).values
    return dict(divergences=int(idf.sample_stats['diverging'].sum()),
                rhat_max=float(az.summary(idf, var_names=RHAT_J, kind='diagnostics')['r_hat'].max()),
                kappa_q=float(idf.posterior['kappa_q'].mean()),
                has_qb_test=float(has_te.mean()), max_draw=float(max(S_s.max(), S_q.max())),
                **{f'kappa_s_{p}': float(v) for p, v in zip(ps.positions, ks)},
                **{f'lam_rec_{p}': float(v) for p, v in zip(ps.positions, lam)})
 
 
#  ============================ data (run once) ============================
history, team_form = D.load_processed(), D.load_team_form()
s, w = 2025, 9
tr_s, te_s, tr_q, te_q = frames(s, w, history, team_form)
 
 
#  ============================ fit (stage A + stage B) ============================
mj, idata, ps, pq, r_q = fit(tr_s, tr_q, te_s, te_q)
print(az.summary(idata, var_names=RHAT_J, kind='diagnostics'))
print('kappa_q', float(idata.posterior['kappa_q'].mean()).__round__(3),
      ' kappa_s', dict(zip(ps.positions, idata.posterior['kappa_s'].mean(('chain', 'draw')).values.round(3))),
      ' lam_rec', dict(zip(ps.positions, idata.posterior['lam_rec'].mean(('chain', 'draw')).values.round(3))))
post = idata.posterior                              # ridge check: kappa_q vs QB season-GP scale, per chain
for v in ('kappa_q', 'qb_eta_season', 'qb_sigma_obs'):
    print(f'{v:14s} per chain', post[v].mean('draw').values.round(3))
 
 
#  ============================ trace / ESS plots ============================
small = [rv.name for rv in mj.free_RVs if rv.size.eval() <= 10]
az.rcParams['plot.max_subplots'] = 60
az.plot_trace(idata, var_names=small)
az.plot_ess_evolution(idata, var_names=small)
 
 
#  ============================ predict the test week ============================
S_s, S_q, has_te = predict(mj, idata, ps, pq, te_s, te_q)
print(diagnostics(idata, ps, S_s, S_q, has_te))
summ_s = summarize(S_s, te_s)
summ_q = summarize(S_q, te_q)
stack_corr(S_s, S_q, te_s, te_q, 'KC')
 
 
#  ============================ BACKTEST: 12 folds ============================
OUT = PATHS.results_dir / NAME
OUT.mkdir(parents=True, exist_ok=True)
rows, tiers = [], []
 
for s_bt in (2023, 2024, 2025):
    for w_bt in (4, 8, 12, 16):
        tr_s_bt, te_s_bt, tr_q_bt, te_q_bt = frames(s_bt, w_bt, history, team_form)
        mdl, idf, ps_bt, pq_bt, _ = fit(tr_s_bt, tr_q_bt, te_s_bt, te_q_bt, progressbar=False)
        Ss, Sq, has = predict(mdl, idf, ps_bt, pq_bt, te_s_bt, te_q_bt)
        diag = diagnostics(idf, ps_bt, Ss, Sq, has)
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
                          f"kappa_q {diag['kappa_q']:.2f}  "
                          + '  '.join(f"{p} k{diag[f'kappa_s_{p}']:+.2f}/l{diag[f'lam_rec_{p}']:+.2f}"
                                      for p in ps_bt.positions))
        del idf, mdl
 
 
#  ============================ BACKTEST: compare with production and the no-tau cut model ============================
OUT = PATHS.results_dir / NAME
PAIRS = {f'{NAME}_skill': 'het_gp_vegas', f'{NAME}_qb': 'het_gp_qb_vegas'}
folds = pl.read_parquet(OUT / 'folds.parquet')
tiers_df = pl.read_parquet(OUT / 'tiers.parquet')
prod = list(PAIRS.values())
base_fp = PATHS.results_dir / BASE / 'folds.parquet'
extra = [pl.read_parquet(base_fp)] if base_fp.exists() else []
folds_all = pl.concat([B.load_run(CALIB)[0].filter(pl.col('model').is_in(prod)), folds, *extra],
                      how='diagonal_relaxed')
tiers_all = pl.concat([B.load_tiers(CALIB).filter(pl.col('model').is_in(prod)), tiers_df], how='diagonal_relaxed')
 
for new, base in PAIRS.items():
    for metric in ('crps', 'mae', 'rmse'):
        d = B.paired_diff(folds_all, new, base, metric)
        if d is None:
            print('no paired folds for', new, 'vs', base); continue
        print(f"{new:<38} vs {base:<16} {metric:<5} {d['mean_diff']:+.3f} ± {d['se']:.3f}  "
              f"better {d['folds_a_better']}/{d['n_folds']}")
for label in ('skill', 'qb') if extra else ():
    d = B.paired_diff(folds_all, f'{NAME}_{label}', f'{BASE}_{label}', 'crps')
    print(f"tau vs no-tau {label:<5} crps {d['mean_diff']:+.3f} ± {d['se']:.3f}  "
          f"better {d['folds_a_better']}/{d['n_folds']}")
 
with pl.Config(tbl_cols=-1, tbl_rows=-1, float_precision=2):
    print(B.tier_summary(tiers_all)
          .with_columns((pl.col('mean_actual') - pl.col('mean_expected')).alias('bias'),
                        (pl.col('pred_sd') / pl.col('resid_sd')).alias('sd_ratio'))
          .select('model', 'tier', 'n', 'bias', 'sd_ratio', 'below_p10', 'above_p90'))
    print(folds.filter(pl.col('model').str.ends_with('_skill'))
          .select('season', 'week', 'crps', 'divergences', 'rhat_max', 'max_draw', 'kappa_q',
                  pl.col('^kappa_s_.*$'), pl.col('^lam_rec_.*$')))
