import arviz as az
import numpy as np
import polars as pl
import pymc as pm
import pytensor.tensor as pt
# pyrefly: ignore [missing-import]
from ffpred import backtest as B
from ffpred import data as D
from ffpred import scoring as Sc
from ffpred.config import MODELS, PATHS, WINDOW_SEASONS
from ffpred.forecast import WeekForecast

# pyrefly: ignore [missing-import]
from ffpred.model import HetPrep, _inverse_gamma_maxent, payload

CFG = MODELS['het_gp_vegas']

SEED = 41029041
JITTER = 1e-6

def _tg_key(f):
    return f.select(pl.concat_str([pl.col('season').cast(pl.String), pl.col('week').cast(pl.String),
                                   pl.col('team')], separator='|')).to_series()



def fold_frames(season, week, history, team_form, window_seasons=WINDOW_SEASONS, group='het_gp_vegas'):
    week = float(week)
    df = D.join_team_form(history, team_form)
    # trailing mean of the previous 8 games; shift(1) keeps the current game out;
    # computed on the full history so the window crosses seasons
    df = df.sort('player_id', 'season', 'week').with_columns(
        pl.col('total_fantasy_points').shift(1)
          .rolling_mean(window_size=6, min_samples=1)      
          .over('player_id').alias('trail_pts'))
    cfg = MODELS[group]
    ref = df.filter(D.before(season, week) | ((pl.col('season') == season) & (pl.col('week') == week)))
    df = D.with_primary_position(df, reference=ref).filter(pl.col('position').is_in(list(cfg.positions)))
    train = D.training_window(df, season, week, window_seasons)
    test = D.week_rows(df, season, week).sort('player_id')
    # no history (rookies, first game) -> position training mean
    pos_mean = train.group_by('position').agg(pl.col('total_fantasy_points').mean().alias('_pm'))
    fill = lambda f: (f.join(pos_mean, on='position', how='left')
                       .with_columns(pl.col('trail_pts').fill_null(pl.col('_pm'))).drop('_pm'))
    train, test = fill(train), fill(test)
    keep = test.filter(pl.Series(HetPrep(train, cfg).eligible(test)))
    return train, keep, test.height

def floor1(x):
    """Smooth floor at 1 point: ~x above ~5, never below 1."""
    return 1.0 + 2.0 * pm.math.log1pexp((x - 1.0) / 2.0)



def build_model(train, test, prior_check=False):
    """test must be the frame you will predict, in the same row order (e.g. sorted by player_id)."""
    train = train.sort('player_id', 'season', 'week')
    prep = HetPrep(train, CFG)
    arr = prep.arrays(train)
    X = prep.grid[:, None]
    m_ref_pos = (train.group_by('position').agg(pl.col('total_fantasy_points').mean())
                 .join(pl.DataFrame({'position': prep.positions}), on='position', how='right')
                 ['total_fantasy_points'].to_numpy())

    # NEW: team-game index over train AND test team-games (test slots carry no data)
    k_tr, k_te = _tg_key(train), _tg_key(test)
    tg_levels = pl.concat([k_tr, k_te]).unique(maintain_order=True).to_list()
    lut = {k: i for i, k in enumerate(tg_levels)}
    tg_tr = np.array([lut[k] for k in k_tr.to_list()])
    prep.tg_test = np.array([lut[k] for k in k_te.to_list()])          # used in pm.set_data

    with pm.Model(coords={**prep.coords, 'team_game': tg_levels}) as model:
        player = pm.Data('player_data', arr['player'])
        pos    = pm.Data('position_data', arr['position'])
        t      = pm.Data('time_data', arr['time'])
        cont   = pm.Data('cont_data', arr['cont'])
        bi     = pm.Data('bi_data', arr['bi'])
        tf     = pm.Data('team_form_data', arr['team_form_z'])
        otf    = pm.Data('opp_team_form_data', arr['opp_team_form_z'])
        obs    = pm.Data('obs_data', arr['y_z'])
        trail  = pm.Data('trail_data', train['trail_pts'].to_numpy().astype(float))
        tg     = pm.Data('tg_data', tg_tr)                                  # NEW

        log_mref = pt.as_tensor(np.log(m_ref_pos))

        # ======================= MEAN (log link) =======================
        alpha_pos = pm.Normal('alpha_pos', 0.15, 0.3, dims='position')

        t_peak_mu  = pm.Normal('t_peak_mu', 4.0, 1.25)
        t_peak_off = pm.ZeroSumNormal('t_peak_off', sigma=1, dims='position')
        t_peak     = pm.Deterministic('t_peak', t_peak_mu + t_peak_off, dims='position')
        log_height_mu  = pm.Normal('log_height_mu', np.log(0.2), 0.4)
        log_height_off = pm.ZeroSumNormal('log_height_off', sigma=0.25, dims='position')
        peak_height    = pm.Deterministic('peak_height', pt.exp(log_height_mu + log_height_off), dims='position')
        peak_width_raw = pm.Gamma('peak_width_raw', alpha=6, beta=1.5)
        peak_width     = pm.Deterministic('peak_width', 1.0 + peak_width_raw)
        curve = peak_height[None, :] * pt.exp(
            -0.5 * ((prep.grid[:, None] - t_peak[None, :]) / peak_width) ** 2)
        f_career = pm.Deterministic(
            'f_career', curve - (prep.t_weights_pos * curve).sum(0), dims=('time', 'position'))

        sigma_skill = pm.HalfNormal('sigma_skill', 0.35)
        z_skill     = pm.Normal('z_skill', 0.0, 1.0, dims='player_id')
        skill_raw   = sigma_skill * z_skill
        pos_mean_skill = (prep.pos_onehot.T @ skill_raw) / prep.pos_counts
        skill = pm.Deterministic('skill', skill_raw - pos_mean_skill[prep.position_of_player], dims='player_id')

        ell_season = pm.InverseGamma('ell_season', *_inverse_gamma_maxent(0.8, 2.5))
        eta_season = pm.HalfNormal('eta_season', 0.15)
        K_season = (eta_season**2 * pm.gp.cov.Matern32(1, ls=ell_season))(X)
        L_season = pt.linalg.cholesky(pm.gp.util.stabilize(K_season, JITTER))
        z_season = pm.Normal('z_season', 0.0, 1.0, dims=('player_id', 'time'))
        f_season_raw = z_season @ L_season.T
        f_season_p = f_season_raw - ((f_season_raw * prep.active).sum(1) / prep.n_seasons_player)[:, None]
        f_season = pm.Deterministic(
            'f_season', f_season_p - (f_season_p * prep.active).sum(0) / prep.n_active,
            dims=('player_id', 'time'))

        cont_prior      = pm.Normal('cont_prior', 0.0, 0.08, dims='cont_vars')
        bi_prior        = pm.Normal('bi_prior', 0.0, 0.08, dims='binary_vars')
        team_form_prior = pm.Normal('team_form_prior', 0.0, 0.08)
        opp_form_prior  = pm.Normal('opp_form_prior', 0.0, 0.08)

        # NEW: shared team-game shock, per-position loading, mean-preserving
        sigma_tg = pm.HalfNormal('sigma_tg', 0.15, dims='position')
        z_tg     = pm.Normal('z_tg', 0.0, 1.0, dims='team_game')
        shock    = sigma_tg[pos] * z_tg[tg] - 0.5 * sigma_tg[pos] ** 2

        eta = (log_mref[pos] + alpha_pos[pos] + f_career[t, pos] + skill[player] + f_season[player, t]
               + pt.dot(cont, cont_prior) + pt.dot(bi, bi_prior)
               + team_form_prior * tf + opp_form_prior * otf
               + shock)                                                     # NEW
        m = pt.exp(eta)

        # ======================= HURDLE-GAMMA (as v3) =======================
        y_pts = prep.y_mu + prep.y_sd * obs
        y_h   = pt.switch(y_pts > 1.0, y_pts, 0.0)

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
            pm.Deterministic('psi_det', psi)
    return model, prep


history, team_form = D.load_processed(), D.load_team_form()
train, test, _ = fold_frames(2025, 9, history, team_form)
model, prep = build_model(train, test)


with model: 
    idata = pm.sample_prior_predictive(random_seed=SEED)

print('alpha_pos mean:', round(idata.prior['alpha_pos'].mean().item(), 3))   # ~0.15



az.plot_ppc_dist(
    idata, 
    group = 'prior_predictive', 
    num_samples=50,
    visuals = {'observed_dist': True}
)

pp, p = idata.prior_predictive, idata.prior
q = lambda v: np.percentile(np.asarray(v).ravel(), [50, 90, 99, 99.9]).round(1)
y, m = np.asarray(pp['y_obs']).ravel(), np.asarray(p['m_pts']).ravel()
print('m_pts      ', q(m))
print('y_obs      ', q(y), ' share > 60:', round(float((y > 60).mean()), 4),
      ' dud share:', round(float((y == 0).mean()), 3))
print('y/m (y>0)  ', q(y[y > 0] / m[y > 0]))
print('share>60 when m_pts < 30:', round(float((y[m < 30] > 60).mean()), 4))

tr_sorted = train.sort('player_id', 'season', 'week')
m_ref_pos = (tr_sorted.group_by('position').agg(pl.col('total_fantasy_points').mean())
             .join(pl.DataFrame({'position': prep.positions}), on='position', how='right')
             ['total_fantasy_points'].to_numpy())
pos_idx = prep.arrays(tr_sorted)['position']
lvl_np = np.clip(np.log(np.maximum(tr_sorted['trail_pts'].to_numpy(), 1.0)) - np.log(m_ref_pos)[pos_idx], -2, 2)
yy_all = np.asarray(pp['y_obs'])
for lo, hi in [(-2, -1), (-1, 0), (0, 1), (1, 2)]:
    k = (lvl_np >= lo) & (lvl_np < hi)
    yy = yy_all[..., k].ravel()
    print(f'lvl {lo:+d}..{hi:+d}  n={k.sum():5d}  p99={np.percentile(yy, 99):6.1f}  '
          f'p99.9={np.percentile(yy, 99.9):6.1f}  share>60={np.mean(yy > 60):.4f}')


pr = idata.prior
cont_np = prep.arrays(tr_sorted)['cont']                         # (rows, 13)
bi_np   = prep.arrays(tr_sorted)['bi']
beta    = pr['cont_prior'].stack(s=('chain', 'draw')).values      # (13, S)
bb      = pr['bi_prior'].stack(s=('chain', 'draw')).values
sk      = pr['sigma_skill'].stack(s=('chain', 'draw')).values
rows    = np.random.default_rng(0).choice(cont_np.shape[0], 2000, replace=False)
sd = lambda a: round(float(np.std(a)), 2)
arr_tr = prep.arrays(tr_sorted)
pl_i, t_i, ps_i = arr_tr['player'], arr_tr['time'], arr_tr['position']
S = lambda v: pr[v].stack(s=('chain', 'draw')).values
fs = S('f_season')[pl_i[rows], t_i[rows], :]                     # (rows, S)
fc = S('f_career')[t_i[rows], ps_i[rows], :]
tfp, ofp = S('team_form_prior'), S('opp_form_prior')
tf_np, otf_np = arr_tr['team_form_z'][rows], arr_tr['opp_team_form_z'][rows]
eta_tot = np.log(np.asarray(p['m_pts']).reshape(-1, len(lvl_np)).T[rows]) - np.log(m_ref_pos)[ps_i[rows]][:, None]
print('f_season   sd:', sd(fs), '  p99.9 |.|:', round(float(np.percentile(np.abs(fs), 99.9)), 2))
print('f_career   sd:', sd(fc))
print('team form  sd:', sd(tf_np[:, None] * tfp + otf_np[:, None] * ofp))
print('alpha      sd:', sd(S('alpha_pos')))
print('TOTAL eta  sd:', sd(eta_tot), '  p99:', round(float(np.percentile(eta_tot, 99)), 2))
print('max |team_form_z|, |opp_form_z|:', np.abs(arr_tr['team_form_z']).max().round(1),
      np.abs(arr_tr['opp_team_form_z']).max().round(1))

with model:
    idata.update(
        pm.sample(random_seed=SEED)
    )

    idata.update(
        pm.sample_posterior_predictive(idata, random_seed=SEED)
    )
    pm.stats.compute_log_likelihood(idata)



az.plot_trace(
    idata,
    var_names=[rv.name for rv in model.free_RVs if rv.size.eval() <= 10]
)
az.plot_ess_evolution(
    idata,
    var_names=[rv.name for rv in model.free_RVs if rv.size.eval() <= 10]
)

az.plot_energy(idata)

post = idata.posterior
for v in ['ell_season', 'eta_season', 'sigma_skill', ]:
    print(f'{v:12s} per-chain means:', post[v].mean('draw').values.round(3))
print('corr(ell_season, eta_season):',
      np.corrcoef(post['ell_season'].values.ravel(), post['eta_season'].values.ravel())[0, 1].round(2))
print('divergences:', int(idata.sample_stats['diverging'].sum()))
print(az.summary(idata, var_names=['ell_season', 'eta_season', 'sigma_skill', ], kind='diagnostics'))

test = test.sort('player_id')
with model:
    pm.set_data({**payload(prep.arrays(test)),
                'trail_data': test['trail_pts'].to_numpy().astype(float)})
    ppc = pm.sample_posterior_predictive(idata, var_names=['y_obs'], random_seed=SEED + 1)
S_new = ppc.posterior_predictive['y_obs'].values.reshape(-1, test.height)
y = test['total_fantasy_points'].to_numpy().astype(float)
print(S_new.shape, 'all finite:', np.isfinite(S_new).all())


summ = (test.select('player', 'position')
        .with_columns(pl.Series('exp', S_new.mean(0)),
                      pl.Series('p10', np.percentile(S_new, 10, 0)),
                      pl.Series('p90', np.percentile(S_new, 90, 0)),
                      pl.Series('sd',  S_new.std(0)),
                      pl.Series('actual', y)))
print(summ.group_by('position').agg(pl.col('exp', 'actual', 'sd').mean()).sort('position'))
print(summ.sort('exp', descending=True).head(10))
print(summ.sort('exp').head(5))

print('divergences:', int(idata.sample_stats['diverging'].sum()))
print(az.summary(idata, var_names=['delta_cv', 'log_cv0', 'sigma_skill', 'eta_season', 'ell_season'], kind='stats'))

y_tr = tr_sorted['total_fantasy_points'].to_numpy()
lvl_tr = lvl_np                                   # from the earlier by-level check
for lo, hi in [(-2, -1), (-1, 0), (0, 1), (1, 2)]:
    k = (lvl_tr >= lo) & (lvl_tr < hi)
    yk = y_tr[k]; pos_ = yk[yk > 0.05]
    print(f'lvl {lo:+d}..{hi:+d}  dud {np.mean(yk <= 0.05):.3f}  near-dud (0,3] {np.mean((yk > 0.05) & (yk <= 3)):.3f}  '
          f'CV of positives {pos_.std() / pos_.mean():.2f}  CV of y>3 {yk[yk > 3].std() / yk[yk > 3].mean():.2f}')

prod = WeekForecast.load(PATHS.forecasts_dir / 'nuts_2025_wk09')
pf = prod.frame.with_row_index('_c').filter(pl.col('position').is_in(list(CFG.positions)))
common = test.with_row_index('_n').join(pf.select('player_id', '_c'), on='player_id', how='inner')
i_new, i_prod = common['_n'].to_numpy(), common['_c'].to_numpy()
yc = y[i_new]

res = pl.DataFrame([{'model': 'scratch_hurdle_gamma', **{k: v for k, v in Sc.score(yc, S_new[:, i_new]).items() if k != 'pit_hist'}},
                    {'model': 'het_gp_vegas', **{k: v for k, v in Sc.score(yc, prod.draws[:, i_prod]).items() if k != 'pit_hist'}}])
print(f'{len(common)} common rows')
print(res.select('model', 'mae', 'crps', 'spearman', 'below_p10', 'above_p90', 'pred_sd', 'resid_sd'))

ct = test[i_new]
print('scratch:\n', Sc.calibration_table(ct, S_new[:, i_new]))
print('production:\n', Sc.calibration_table(ct, prod.draws[:, i_prod]))

cv_ = [str(v) for v in idata.posterior['cont_vars'].values]
b   = idata.posterior['cont_prior'].mean(('chain', 'draw')).values
print(dict(zip(cv_, b.round(3))))                                       # vegas: ~0.05-0.10?
print('delta_cv', idata.posterior['delta_cv'].mean(('chain', 'draw')).values.round(2),
      ' sigma_skill', round(float(idata.posterior['sigma_skill'].mean()), 3))


with model:
    pm.set_data({**payload(prep.arrays(test)), 'trail_data': test['trail_pts'].to_numpy().astype(float)})
    pe = pm.sample_posterior_predictive(idata, var_names=['eta'], random_seed=1)
e = pe.posterior_predictive['eta'].values.reshape(-1, test.height)
ratio = np.exp(e).mean(0) / np.exp(e.mean(0))
top = np.argsort(-S_new.mean(0))[:10]
print('mean ratio, all:', ratio.mean().round(3), '  top 10:', ratio[top].round(3))


def gamma_center(post):
    """Average gamma, whichever parameterisation the model used."""
    for v in ('gamma_mu', 'gamma_bar'):
        if v in post:
            return float(post[v].mean())
    return float(post['gamma'].mean())          # fallback: mean over positions


NAME = 'scratch_joint_qb_resid'
OUT = PATHS.results_dir / NAME
OUT.mkdir(parents=True, exist_ok=True)
fold_rows, tier_rows = [], []
for s in (2023, 2024, 2025):
    for w in (4, 8, 12, 16):
        tr, te, _ = fold_frames(s, w, history, team_form)
        te = te.sort('player_id')                     # sort BEFORE build_model; the tg index follows this order
        mdl, pr = build_model(tr, te)
        with mdl:
            idf = pm.sample(draws=1000, tune=1000, chains=4, nuts_sampler='nutpie',
                            target_accept=0.95, random_seed=SEED, progressbar=False)
            pm.set_data({**payload(pr.arrays(te)),
                         'trail_data': te['trail_pts'].to_numpy().astype(float),
                         'tg_data': pr.tg_test})
            pp = pm.sample_posterior_predictive(idf, var_names=['y_obs'],
                                                random_seed=SEED + 1, progressbar=False)

        S  = pp.posterior_predictive['y_obs'].values.reshape(-1, te.height)   # already in points
        yv = te['total_fantasy_points'].to_numpy().astype(float)
        sc = Sc.score(yv, S); sc.pop('pit_hist')
        post = idf.posterior

        fold_rows.append(dict(
            model=NAME, season=s, week=float(w), eligibility='model', **sc,
            divergences=int(idf.sample_stats['diverging'].sum()),
            rhat_max=float(az.summary(idf, var_names=['ell_season', 'eta_season', 'sigma_skill',
                                                      'a_psi', 'log_cv0', 'delta_cv'],
                                      kind='diagnostics')['r_hat'].max()),
            dud_pred=float((S <= 0.05).mean()), dud_actual=float((yv <= 0.05).mean()),
            sigma_skill=float(post['sigma_skill'].mean()),
            **{f'sigma_tg_{p}': float(v) for p, v in
               zip(pr.positions, post['sigma_tg'].mean(('chain', 'draw')).values)},
        ))
        tier_rows.append(Sc.calibration_table(te, S, by_position=True).rename({'group': 'pos'})
                         .with_columns(pl.lit(NAME).alias('model'), pl.lit(s).alias('season'),
                                       pl.lit(float(w)).alias('week')))

        pl.DataFrame(fold_rows).write_parquet(OUT / 'folds.parquet')        # checkpoint every fold
        pl.concat(tier_rows).write_parquet(OUT / 'tiers.parquet')
        r = fold_rows[-1]
        print(s, w, f"crps {r['crps']:.3f}  div {r['divergences']}  rhat {r['rhat_max']:.3f}  "
                    f"dud {r['dud_pred']:.3f}/{r['dud_actual']:.3f}  sigma_skill {r['sigma_skill']:.3f}")
        del idf, pp, mdl




RUNS = ['scratch_hurdle', 'scratch_hurdle_v3', 'scratch_hurdle_larger_dud','scratch_hurdle_team_shocks']     # folder names under results/
CALIB = PATHS.results_dir / 'backtest' / 'calib'

prod_folds = B.load_run(CALIB)[0].filter(pl.col('model') == 'het_gp_vegas')
prod_tiers = B.load_tiers(CALIB).filter(pl.col('model') == 'het_gp_vegas')
folds = pl.concat([prod_folds] + [pl.read_parquet(PATHS.results_dir / r / 'folds.parquet') for r in RUNS],
                  how='diagonal_relaxed')
tiers = pl.concat([prod_tiers] + [pl.read_parquet(PATHS.results_dir / r / 'tiers.parquet') for r in RUNS],
                  how='diagonal_relaxed')

# ---- 1. paired CRPS / MAE ------------------------------------------------------------
def paired(a, b):
    out = {}
    for metric in ('crps', 'mae'):
        d = B.paired_diff(folds, a, b, metric)
        out[metric] = f"{d['mean_diff']:+.3f} ± {d['se']:.3f}  ({d['folds_a_better']}/{d['n_folds']})"
    return out

print('negative = first model better')
for a, b in [('scratch_hurdle_larger_dud', 'het_gp_vegas'),
             ('scratch_hurdle_larger_dud', 'scratch_hurdle_v3'),
             ('scratch_hurdle_v3',   'het_gp_vegas')]:
    p = paired(a, b)
    print(f'{a:22s} vs {b:20s}  CRPS {p["crps"]}   MAE {p["mae"]}')

# ---- 2. tier table: bias and spread ratio ---------------------------------------------
with pl.Config(tbl_cols=-1, tbl_rows=-1, float_precision=2):
    print(B.tier_summary(tiers)
          .with_columns((pl.col('mean_actual') - pl.col('mean_expected')).alias('bias'),
                        (pl.col('pred_sd') / pl.col('resid_sd')).alias('sd_ratio'))
          .select('model', 'tier', 'n', 'bias', 'sd_ratio', 'below_p10', 'above_p90'))

# ---- 3. tier table by position (hurdle runs; production too if it has a position column) ----
pos_col = 'pos' if 'pos' in tiers.columns else 'group'
by_pos = (tiers.filter(pl.col(pos_col).is_in(['RB', 'TE', 'WR']))
          .group_by('model', pos_col, 'tier')
          .agg(pl.col('n').sum(),
               ((pl.col('mean_actual') - pl.col('mean_expected')) * pl.col('n')).sum().alias('_b'),
               (pl.col('pred_sd') * pl.col('n')).sum().alias('_p'),
               (pl.col('resid_sd') ** 2 * pl.col('n')).sum().alias('_r'),
               (pl.col('below_p10') * pl.col('n')).sum().alias('_lo'),
               (pl.col('above_p90') * pl.col('n')).sum().alias('_hi'))
          .with_columns((pl.col('_b') / pl.col('n')).alias('bias'),
                        ((pl.col('_p') / pl.col('n')) / (pl.col('_r') / pl.col('n')).sqrt()).alias('sd_ratio'),
                        (pl.col('_lo') / pl.col('n')).alias('below_p10'),
                        (pl.col('_hi') / pl.col('n')).alias('above_p90'))
          .drop('_b', '_p', '_r', '_lo', '_hi')
          .sort('model', pos_col, 'tier'))
with pl.Config(tbl_cols=-1, tbl_rows=-1, float_precision=2):
    print(by_pos)

# ---- 4. dud3 per-fold diagnostics ------------------------------------------------------
d3 = pl.read_parquet(PATHS.results_dir / 'scratch_hurdle_larger_dud' / 'folds.parquet')
keep = [c for c in ['season', 'week', 'crps', 'divergences', 'rhat_max', 'dud_pred', 'dud_actual',
                    'sigma_skill', 'delta_cv_RB', 'delta_cv_TE', 'delta_cv_WR'] if c in d3.columns]
with pl.Config(tbl_cols=-1, float_precision=3):
    print(d3.select(keep))

y = tr_sorted['total_fantasy_points'].to_numpy()
for lo, hi in [(-2, -1), (-1, 0), (0, 1), (1, 2)]:
    k = (lvl_np >= lo) & (lvl_np < hi); yk = y[k]
    a, b = (yk > 0.05) & (yk <= 1), (yk > 1) & (yk <= 3)
    print(f'lvl {lo:+d}..{hi:+d}  share (0,1] {a.mean():.3f} mean {yk[a].mean():.2f}   '
          f'share (1,3] {b.mean():.3f} mean {yk[b].mean():.2f}   '
          f'CV y>1 {yk[yk > 1].std() / yk[yk > 1].mean():.2f}   CV y>3 {yk[yk > 3].std() / yk[yk > 3].mean():.2f}')

np.save(OUT / f'S_{s}_{int(w):02d}.npy', S.astype(np.float32))          # ~4 MB per fold
te.select('player_id', 'player', 'position', 'team', 'total_fantasy_points') \
    .write_parquet(OUT / f'te_{s}_{int(w):02d}.parquet')

import itertools
RUN   = 'scratch_hurdle_v3'
OUT   = PATHS.results_dir / RUN
FOLDS = [(s, w) for s in (2023, 2024, 2025) for w in (4, 8, 12, 16)]
N_ROSTERS, ALPHA, SEED = 200, 0.10, 7
# roster: 4 RB, 5 WR, 2 TE drawn from fantasy-relevant pools; lineup: 2 RB, 2 WR, 1 TE, 1 FLEX
POOL   = {'RB': 36, 'WR': 48, 'TE': 18}       # top-N by projection at each position
ROSTER = {'RB': 4, 'WR': 5, 'TE': 2}


def as_is(S, rng):
    return S

def flatten(S, rng):
    """Same means and same shape per player, but every player rescaled to one common sd."""
    mu, sd = S.mean(0), S.std(0)
    return mu + (S - mu) * (np.median(sd) / np.maximum(sd, 1e-6))

def normal_flat(S, rng):
    """Production-like: same means, symmetric Normal draws, one common sd for everyone."""
    mu, sd = S.mean(0), S.std(0)
    return mu + np.median(sd) * rng.standard_normal(S.shape)


# ---------------- lineup machinery ----------------
def lineups(r):
    out = []
    for rb in itertools.combinations(r['RB'], 2):
        for wr in itertools.combinations(r['WR'], 2):
            for te in itertools.combinations(r['TE'], 1):
                core = set(rb) | set(wr) | set(te)
                for f in [i for i in r['RB'] + r['WR'] + r['TE'] if i not in core]:
                    out.append(list(rb) + list(wr) + list(te) + [f])
    return out

def cvar(T, alpha=ALPHA):
    k = max(1, int(alpha * T.shape[1]))
    return np.sort(T, axis=1)[:, :k].mean(1)


def run_diag(transform):
    rows = []
    for s, w in FOLDS:
        S_raw = np.load(OUT / 'S_2023_04.npy').astype(np.float64)
        te    = pl.read_parquet(OUT / 'te_2023_04.parquet')
        y     = te['total_fantasy_points'].to_numpy()
        S     = transform(S_raw, np.random.default_rng(SEED + 1000 * s + w))
        proj  = S_raw.mean(0)                                   # pools from the real means
        pos   = te['position'].to_numpy()
        pools = {p: (ix := np.flatnonzero(pos == p))[np.argsort(-proj[ix])][:POOL[p]] for p in POOL}
        rng   = np.random.default_rng(SEED + 1000 * s + w)      # same rosters for every transform
        for _ in range(N_ROSTERS):
            r = {p: list(rng.choice(pools[p], ROSTER[p], replace=False)) for p in ROSTER}
            L = lineups(r)
            T = np.stack([S[:, l].sum(1) for l in L])
            E, SD, CV = T.mean(1), T.std(1), cvar(T)
            target = np.percentile(T[E.argmax()], 75)
            P_up = (T >= target).mean(1)
            be, bc, bu = E.argmax(), CV.argmax(), P_up.argmax()
            top5 = np.argsort(-E)[:5]
            actual = np.array([y[l].sum() for l in L])
            rows.append(dict(season=s, week=w,
                             sd_best=SD[be], sd_spread_top5=SD[top5].max() - SD[top5].min(),
                             corr_E_cvar=np.corrcoef(E, CV)[0, 1],
                             same_cvar=bc == be, same_upside=bu == be,
                             pts_cvar_minus_E=actual[bc] - actual[be],
                             pts_upside_minus_E=actual[bu] - actual[be],
                             pit_best=(T[be] < actual[be]).mean()))
    return pl.DataFrame(rows)


def summarise(df, label):
    pit = df['pit_best'].to_numpy()
    return dict(
        draws=label, rosters=df.height,
        sd_best=df['sd_best'].mean(),
        sd_spread_top5=df['sd_spread_top5'].mean(),
        corr_E_cvar=df['corr_E_cvar'].mean(),
        same_cvar=df['same_cvar'].mean(), same_upside=df['same_upside'].mean(),
        pts_cvar_minus_E=df['pts_cvar_minus_E'].mean(),
        pts_upside_minus_E=df['pts_upside_minus_E'].mean(),
        cov50=np.mean((pit > 0.25) & (pit < 0.75)),
        cov80=np.mean((pit > 0.10) & (pit < 0.90)),
        cov90=np.mean((pit > 0.05) & (pit < 0.95)))


results = {name: run_diag(fn) for name, fn in
           [('v3 as is', as_is), ('v3 flat sd', flatten), ('normal flat sd', normal_flat)]}
with pl.Config(tbl_cols=-1, float_precision=3):
    print(pl.DataFrame([summarise(df, k) for k, df in results.items()]))
