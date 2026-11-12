"""Joint model where both skill and QB's can move together
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
NAME = 'scratch_joint_qb_resid'          # results folder; change per experiment so runs don't overwrite
SEED = 41029041
JITTER = 1e-6
DUD_LINE = 0.05                          # v3 used 0.05 (1.0 / 3.0 were tested and were worse)
STARTER_COLS = ('attempts', 'pass_attempts', 'offense_snaps', 'offense_pct', 'trail_pts')
RHAT_VARS = ['lam_rec', 'sigma_skill', 'eta_season', 'ell_season', 'qb_sigma_obs', 'log_cv0', 'delta_cv']
TE_COLS = ['player_id', 'player', 'position', 'team', 'opp_team', 'season', 'week', 'total_fantasy_points']
CALIB = PATHS.results_dir / 'backtest' / 'calib'
 
 

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
    return tr_s, te_s.sort('player_id'), tr_q, te_q.sort('player_id')
 
 
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
 
 
def predict(mdl, idf, ps, pq, te_s, te_q, seed=SEED):
    """Joint predictive draws in points. Row i of S_s and S_q = same posterior draw + same QB eps."""
    n_s, n_q = te_s.height, te_q.height
    qcol, has_te, _ = starter_row(te_q, te_s, test=True)
    with mdl:
        pm.set_data({**payload(ps.arrays(te_s)),
                     'trail_data': te_s['trail_pts'].to_numpy().astype(float),
                     **qb_payload(pq, te_q),
                     'qrow_data': np.full(n_s, n_q, dtype=int),        # shock off: added below
                     'has_qb_data': np.zeros(n_s)})
        pp = pm.sample_posterior_predictive(idf, var_names=['y_obs', 'qb_mu'],
                                            random_seed=seed + 1, progressbar=False)
    S0   = pp.posterior_predictive['y_obs'].values.reshape(-1, n_s)
    mu_q = pp.posterior_predictive['qb_mu'].values.reshape(-1, n_q)
    sig  = idf.posterior['qb_sigma_obs'].values.reshape(-1, 1)
    eps  = np.random.default_rng(seed + 2).standard_normal(mu_q.shape)
    S_q  = (mu_q + sig * eps) * pq.y_sd + pq.y_mu
    lam = idf.posterior['lam_rec'].values.reshape(-1, len(ps.positions))[:, ps.arrays(te_s)['position']]
    e_s = np.where(has_te[None, :] > 0, eps[:, np.minimum(qcol, n_q - 1)], 0.0)
    S_s = S0 * np.exp(np.where(has_te[None, :] > 0, lam * e_s - 0.5 * lam**2, 0.0))
    return S_s, S_q, has_te
 
 
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
 

def build_model(tr_s, tr_q, prior_check=False):
    tr_s = tr_s.sort('player_id', 'season', 'week')
    tr_q = tr_q.sort('player_id', 'season', 'week')
    ps, pq = HetPrep(tr_s, CFG_S), HetPrep(tr_q, CFG_Q)
    As, Aq = ps.arrays(tr_s), pq.arrays(tr_q)
    Xs, Xq = ps.grid[:, None], pq.grid[:, None]
    m_ref_pos = (tr_s.group_by('position').agg(pl.col('total_fantasy_points').mean())
                 .join(pl.DataFrame({'position': ps.positions}), on='position', how='right')
                 ['total_fantasy_points'].to_numpy())
    qrow, has, ps.starter_col = starter_row(tr_q, tr_s)
 
    coords = {**ps.coords,
              'qb_player': pq.coords['player_id'], 'qb_time': pq.coords['time'],
              'qb_cont_vars': pq.coords['cont_vars'], 'qb_binary_vars': pq.coords['binary_vars']}
 
    with pm.Model(coords=coords) as model:
        # ================= QB BLOCK (production het_gp_qb_vegas) =================
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
 
        # ================= QB RESIDUAL -> RECEIVERS =================
        r_qb  = (q_obs - q_mu) / q_sigma                 # wrap in pt.stop_gradient(...) to cut feedback
        r_ext = pt.concatenate([r_qb, pt.zeros((1,))])
        qrow_d = pm.Data('qrow_data', qrow)
        has_d  = pm.Data('has_qb_data', has)
        lam_rec = pm.Normal('lam_rec', 0.0, 0.3, dims='position')
 
        # ================= SKILL BLOCK (scratch_hurdle_v3) =================
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
 
        shock = has_d * (lam_rec[pos] * r_ext[qrow_d] - 0.5 * lam_rec[pos] ** 2)
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
 
        if prior_check:
            pm.Deterministic('m_pts', m)
    return model, ps, pq
 

history, team_form = D.load_processed(), D.load_team_form()
s, w = 2025, 9
tr_s, te_s, tr_q, te_q = frames(s, w, history, team_form)

model, ps, pq = build_model(tr_s, tr_q)
with model:
    idata = pm.sample(
                      target_accept=0.95, random_seed=SEED)
print('divergences', int(idata.sample_stats['diverging'].sum()), '  starter col:', ps.starter_col)
print(az.summary(idata, var_names=RHAT_VARS, kind='diagnostics'))
print(dict(zip(ps.positions, idata.posterior['lam_rec'].mean(('chain', 'draw')).values.round(3))))

small = [rv.name for rv in model.free_RVs if rv.size.eval() <= 10]
az.rcParams['plot.max_subplots'] = 60
az.plot_trace(idata, var_names=small)
az.plot_ess_evolution(idata, var_names=small)

S_s, S_q, has_te = predict(model, idata, ps, pq, te_s, te_q)
print('has_qb', has_te.mean().round(2), '  max draw', round(float(max(S_s.max(), S_q.max())), 1))
summ_s = summarize(S_s, te_s)
summ_q = summarize(S_q, te_q)
stack_corr(S_s, S_q, te_s, te_q, 'KC')
 

for label, S, te in (('skill', S_s, te_s), ('qb', S_q, te_q)):
    sc = Sc.score(te['total_fantasy_points'].to_numpy().astype(float), S); sc.pop('pit_hist')
    print(label, {k: round(float(v), 3) for k, v in sc.items() if np.isscalar(v)})
    with pl.Config(tbl_rows=-1, float_precision=2):
        print(Sc.calibration_table(te, S, by_position=True))



OUT = PATHS.results_dir / NAME
OUT.mkdir(parents=True, exist_ok=True)
rows, tiers = [], []
 
for s_bt in (2023, 2024, 2025):
    for w_bt in (4, 8, 12, 16):
        tr_s_bt, te_s_bt, tr_q_bt, te_q_bt = frames(s_bt, w_bt, history, team_form)
        mdl, ps_bt, pq_bt = build_model(tr_s_bt, tr_q_bt)
        with mdl:
            idf = pm.sample(draws=1000, tune=1000, chains=4, nuts_sampler='nutpie',
                            target_accept=0.95, random_seed=SEED, progressbar=False)
        Ss, Sq, has = predict(mdl, idf, ps_bt, pq_bt, te_s_bt, te_q_bt)
 
        lam = idf.posterior['lam_rec'].mean(('chain', 'draw')).values
        diag = dict(divergences=int(idf.sample_stats['diverging'].sum()),
                    rhat_max=float(az.summary(idf, var_names=RHAT_VARS, kind='diagnostics')['r_hat'].max()),
                    has_qb_test=float(has.mean()), max_draw=float(max(Ss.max(), Sq.max())),
                    **{f'lam_rec_{p}': float(v) for p, v in zip(ps_bt.positions, lam)})
        for label, S, te in (('skill', Ss, te_s_bt), ('qb', Sq, te_q_bt)):
            sc = Sc.score(te['total_fantasy_points'].to_numpy().astype(float), S); sc.pop('pit_hist')
            rows.append(dict(model=f'{NAME}_{label}', season=s_bt, week=float(w_bt),
                             eligibility='model', **sc, **diag))
            tiers.append(Sc.calibration_table(te, S, by_position=True).rename({'group': 'pos'})
                         .with_columns(pl.lit(f'{NAME}_{label}').alias('model'),
                                       pl.lit(s_bt).alias('season'), pl.lit(float(w_bt)).alias('week')))
 
        # checkpoint every fold; row i of QB and skill draws = same posterior draw + same QB eps
        tag = f'{s_bt}_{w_bt:02d}'
        np.save(OUT / f'S_{tag}.npy', np.concatenate([Sq, Ss], axis=1).astype(np.float32))
        pl.concat([te_q_bt, te_s_bt], how='diagonal_relaxed').select(TE_COLS).write_parquet(OUT / f'te_{tag}.parquet')
        pl.DataFrame(rows).write_parquet(OUT / 'folds.parquet')
        pl.concat(tiers).write_parquet(OUT / 'tiers.parquet')
        print(s_bt, w_bt, f"crps skill {rows[-2]['crps']:.3f} qb {rows[-1]['crps']:.3f}  "
                          f"div {diag['divergences']}  rhat {diag['rhat_max']:.3f}  max {diag['max_draw']:.0f}  "
                          f"has_qb {diag['has_qb_test']:.2f}  "
                          + '  '.join(f"{p} {diag[f'lam_rec_{p}']:+.2f}" for p in ps_bt.positions))
        del idf, mdl
 
 

PAIRS = {f'{NAME}_skill': 'het_gp_vegas', f'{NAME}_qb': 'het_gp_qb_vegas'}
folds = pl.read_parquet(OUT / 'folds.parquet')          # reads the saved run, so this cell works on its own
tiers_df = pl.read_parquet(OUT / 'tiers.parquet')
prod = list(PAIRS.values())
folds_all = pl.concat([B.load_run(CALIB)[0].filter(pl.col('model').is_in(prod)), folds], how='diagonal_relaxed')
tiers_all = pl.concat([B.load_tiers(CALIB).filter(pl.col('model').is_in(prod)), tiers_df], how='diagonal_relaxed')
 
for new, base in PAIRS.items():
    for metric in ('crps', 'mae', 'rmse'):
        d = B.paired_diff(folds_all, new, base, metric)                  # negative = new better
        print(f"{new:<30} vs {base:<16} {metric:<5} {d['mean_diff']:+.3f} ± {d['se']:.3f}  "
              f"better {d['folds_a_better']}/{d['n_folds']}")
 
with pl.Config(tbl_cols=-1, tbl_rows=-1, float_precision=2):
    print(B.tier_summary(tiers_all)
          .with_columns((pl.col('mean_actual') - pl.col('mean_expected')).alias('bias'),
                        (pl.col('pred_sd') / pl.col('resid_sd')).alias('sd_ratio'))
          .select('model', 'tier', 'n', 'bias', 'sd_ratio', 'below_p10', 'above_p90'))
    print(folds.filter(pl.col('model').str.ends_with('_skill'))
          .select('season', 'week', 'crps', 'divergences', 'rhat_max', 'max_draw', pl.col('^lam_rec_.*$')))
 
 

for p in sorted(OUT.glob('S_*.npy')):
    S = np.load(p).astype('float64')
    te = pl.read_parquet(OUT / p.name.replace('S_', 'te_').replace('.npy', '.parquet'))
    bad = np.flatnonzero(S.max(0) > 150)                # ~1000 draws -> stuck chain; all -> bad covariate row
    if bad.size:
        print(p.name, f'max {S.max():.3g}')
        print(te[bad].select('player', 'position', 'team').with_columns(
            pl.Series('max', S.max(0)[bad]), pl.Series('n_draws_gt150', (S[:, bad] > 150).sum(0))))
