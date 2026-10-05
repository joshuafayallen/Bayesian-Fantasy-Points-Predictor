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

def fold_frames(season, week, history, team_form, window_seasons = WINDOW_SEASONS, group ='het_gp_vegas'):
    week = float(week)
    df = D.join_team_form(history, team_form)
    cfg = MODELS[group]
    week = float(week)
    ref = df.filter(D.before(season, week) | ((pl.col('season') == season) & (pl.col('week') == week)))
    df = D.with_primary_position(df, reference=ref).filter(pl.col('position').is_in(list(cfg.positions)))
    train = D.training_window(df, season, week, window_seasons)
    test = D.week_rows(df, season, week).sort('player_id')
    keep = test.filter(pl.Series(HetPrep(train, cfg).eligible(test)))
    return train, keep, test.height

def floor1(x):
    """Smooth floor at 1 point: ~x above ~5, never below 1."""
    return 1.0 + 2.0 * pm.math.log1pexp((x - 1.0) / 2.0)



def build_model(train): 
    train = train.sort('player_id', 'season', 'week')
    prep = HetPrep(train, CFG)
    arr = prep.arrays(train)
    X = prep.grid[:, None]
    m_ref_pos = (train.group_by('position').agg(pl.col('total_fantasy_points').mean())
                 .join(pl.DataFrame({'position': prep.positions}), on='position', how='right')
                 ['total_fantasy_points'].to_numpy())

    with pm.Model(coords = prep.coords) as model:
        player = pm.Data('player_data', arr['player'])
        pos    = pm.Data('position_data', arr['position'])
        t      = pm.Data('time_data', arr['time'])          # tenure index
        cont   = pm.Data('cont_data', arr['cont'])          # 13 standardized covariates
        bi     = pm.Data('bi_data', arr['bi'])              # 5 centered binaries
        tf     = pm.Data('team_form_data', arr['team_form_z'])
        otf    = pm.Data('opp_team_form_data', arr['opp_team_form_z'])
        obs    = pm.Data('obs_data', arr['y_z'])

        alpha = pm.Normal('alpha_pos', 0.0, 0.5, dims='position')
 
    # --- 1. aging curve -----------------------------------------------------
    # parametric peak encodes "best years = late rookie deal into 2nd contract".
    # ASSUMES tenure = 1 in the rookie season -> peak prior covers ~years 3-7.
    # Shift t_peak's mean down by 1 if your tenure is 0-indexed.
        t_peak_mu = pm.Normal('t_peak_mu', 4.0, 1.25)
        t_peak_off = pm.ZeroSumNormal('t_peak_off', sigma=1, dims='position')
        t_peak = pm.Deterministic('t_peak', t_peak_mu + t_peak_off, dims='position')

        # peak height: shared mean on the log scale + zero-sum offsets (~25% spread)
        log_height_mu = pm.Normal('log_height_mu', np.log(0.5), 0.4)
        log_height_off = pm.ZeroSumNormal('log_height_off', sigma=0.25, dims='position')
        peak_height = pm.Deterministic('peak_height', pt.exp(log_height_mu + log_height_off), dims='position')

        # width shared, floored at one season
        peak_width_raw = pm.Gamma('peak_width_raw', alpha=6, beta=1.5)
        peak_width = pm.Deterministic('peak_width', 1.0 + peak_width_raw)

        curve = peak_height[None, :] * pt.exp(
            -0.5 * ((prep.grid[:, None] - t_peak[None, :]) / peak_width) ** 2
        )                                                                        # (tenure, position)
        f_career = pm.Deterministic(
            'f_career', curve - (prep.t_weights_pos * curve).sum(0), dims=('time', 'position')
        )
    
        # --- 2. player skill: career-long level, same spread for every position ---
        sigma_skill = pm.HalfNormal('sigma_skill', 1.0)
        z_skill = pm.Normal('z_skill',  mu = 0.0, sigma = 1.0 ,dims='player_id')
        skill_raw = sigma_skill * z_skill
        pos_mean_skill = (prep.pos_onehot.T @ skill_raw) / prep.pos_counts               # (n_positions,)
        skill = pm.Deterministic('skill', skill_raw - pos_mean_skill[prep.position_of_player], dims='player_id')
    
        # --- 3. season level: per-player GP over tenure, shared kernel -----------
        ell_season = pm.InverseGamma('ell_season', *_inverse_gamma_maxent(0.8, 2.5))
        eta_season = pm.HalfNormal('eta_season', 0.5)
        K_season = (eta_season**2 * pm.gp.cov.Matern32(1, ls=ell_season))(X)
        L_season = pt.linalg.cholesky(pm.gp.util.stabilize(K_season, JITTER))
        z_season = pm.Normal('z_season', 0.0, 1.0, dims=('player_id', 'time'))
        # (player, tenure) -- large; drop the Deterministic if memory gets tight and
        # rebuild it post hoc from z_season, ell_season, eta_season
        f_season_raw = z_season @ L_season.T
        f_season_p = f_season_raw - ((f_season_raw * prep.active).sum(1) / prep.n_seasons_player)[:, None]   # mean-zero per player
        f_season = pm.Deterministic(
            'f_season',
            f_season_p - (f_season_p * prep.active).sum(0) / prep.n_active,                                  # mean-zero per tenure
            dims=('player_id', 'time'),
        )
        

        cont_prior = pm.Normal('cont_prior', 0.0, 0.5, dims='cont_vars')
        bi_prior   = pm.Normal('bi_prior', 0.0, 0.5, dims='binary_vars')
        team_form_prior = pm.Normal('team_form_prior', 0.0, 0.5)
        opp_form_prior  = pm.Normal('opp_form_prior', 0.0, 0.5)

        mu = pm.Deterministic('mu', alpha[pos] + f_career[t, pos] + skill[player] + f_season[player, t]
                              + pt.dot(cont, cont_prior) + pt.dot(bi, bi_prior)
                              + team_form_prior * tf + opp_form_prior * otf)

        # ---------------- noise: production + gamma (the only change) ----------------
        # testing a parameter too see if we can slow the shrinkage on elite players 
        # the model tends to be a touch pessimistic for really good players
        log_sigma0 = pm.Normal('log_sigma0', 0.0, 0.5, dims='position')
        ell_noise  = pm.InverseGamma('ell_noise', *prep.noise_prior)
        eta_noise  = pm.HalfNormal('eta_noise', 0.15)
        gp_noise   = pm.gp.Latent(cov_func=eta_noise**2 * pm.gp.cov.Matern52(1, ls=ell_noise))
        g_raw   = gp_noise.prior('g_noise_raw', X=X, jitter=JITTER, dims='time')
        g_noise = pm.Deterministic('g_noise', g_raw - pt.dot(prep.t_weights, g_raw), dims='time')

        gamma_mu  = pm.Normal('gamma_mu', 0.5, 0.12)
        gamma_off = pm.ZeroSumNormal('gamma_off', sigma=0.1, dims='position')
        gamma     = pm.Deterministic('gamma', gamma_mu + gamma_off, dims='position')

        mu_pts    = prep.y_mu + prep.y_sd * mu
        log_ratio = pt.clip(pt.log(floor1(mu_pts)) - pt.as_tensor(np.log(m_ref_pos))[pos], -2.0, 2.0)
        log_sigma = log_sigma0[pos] + g_noise[t] + gamma[pos] * log_ratio
        pm.Normal('y_obs', mu=mu, sigma=pt.exp(log_sigma), observed=obs)
        return model, prep


history, team_form = D.load_processed(), D.load_team_form()
train, test, _ = fold_frames(2025, 9, history, team_form)
model, prep = build_model(train)


with model: 
    idata = pm.sample_prior_predictive(random_seed=SEED)


az.plot_ppc_dist(
    idata, 
    group = 'prior_predictive', 
    visuals = {'observed_dist': True}
)

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
for v in ['ell_season', 'eta_season', 'sigma_skill', 'gamma_bar']:
    print(f'{v:12s} per-chain means:', post[v].mean('draw').values.round(3))
print('corr(ell_season, eta_season):',
      np.corrcoef(post['ell_season'].values.ravel(), post['eta_season'].values.ravel())[0, 1].round(2))
print('divergences:', int(idata.sample_stats['diverging'].sum()))
print(az.summary(idata, var_names=['ell_season', 'eta_season', 'sigma_skill', 'gamma_bar'], kind='diagnostics'))

test = test.sort('player_id')
with model:
    pm.set_data(payload(prep.arrays(test)))       # swaps in week-9 rows; obs set to zeros
    ppc = pm.sample_posterior_predictive(idata, var_names=['y_obs'], random_seed=SEED + 1)
S_new = ppc.posterior_predictive['y_obs'].values.reshape(-1, test.height) * prep.y_sd + prep.y_mu
y = test['total_fantasy_points'].to_numpy().astype(float)
print(S_new.shape, 'all finite:', np.isfinite(S_new).all())


summ = (test.select('player', 'position')
        .with_columns(pl.Series('exp', S_new.mean(0)),
                      pl.Series('p10', np.percentile(S_new, 10, 0)),
                      pl.Series('p90', np.percentile(S_new, 90, 0)),
                      pl.Series('sd',  S_new.std(0)),
                      pl.Series('actual', y)))
print(summ.group_by('position').agg(pl.col('exp', 'actual', 'sd').mean()).sort('position'))
print(summ.sort('exp', descending=True).head(15))
print(summ.sort('exp').head(5))


prod = WeekForecast.load(PATHS.forecasts_dir / 'nuts_2025_wk09')
pf = prod.frame.with_row_index('_c').filter(pl.col('position').is_in(list(CFG.positions)))
common = test.with_row_index('_n').join(pf.select('player_id', '_c'), on='player_id', how='inner')
i_new, i_prod = common['_n'].to_numpy(), common['_c'].to_numpy()
yc = y[i_new]

res = pl.DataFrame([{'model': 'scratch_gamma', **{k: v for k, v in Sc.score(yc, S_new[:, i_new]).items() if k != 'pit_hist'}},
                    {'model': 'het_gp_vegas', **{k: v for k, v in Sc.score(yc, prod.draws[:, i_prod]).items() if k != 'pit_hist'}}])
print(f'{len(common)} common rows')
print(res.select('model', 'mae', 'crps', 'spearman', 'cov80', 'below_p10', 'above_p90', 'pred_sd', 'resid_sd'))

ct = test[i_new]
print('scratch:\n', Sc.calibration_table(ct, S_new[:, i_new]))
print('production:\n', Sc.calibration_table(ct, prod.draws[:, i_prod]))

az.summary(idata, var_names=['gamma_bar', 'gamma_tau', 'gamma'], kind='stats')


CALIB = PATHS.results_dir / 'backtest' / 'calib'
NAME = 'scratch_gamma_v2'
OUT = PATHS.results_dir / NAME
OUT.mkdir(parents=True, exist_ok=True)
fold_rows, tier_rows = [], []

def gamma_center(post):
    """Average gamma, whichever parameterisation the model used."""
    for v in ('gamma_mu', 'gamma_bar'):
        if v in post:
            return float(post[v].mean())
    return float(post['gamma'].mean())          # fallback: mean over positions

fold_rows, tier_rows = [], []
for s in (2023, 2024, 2025):
    for w in (4, 8, 12, 16):
        tr, te, _ = fold_frames(s, w, history, team_form)
        te = te.sort('player_id')
        mdl, pr = build_model(tr)
        with mdl:
            idf = pm.sample(draws=1000, tune=1000, chains=4, nuts_sampler='nutpie',
                            target_accept=0.95, random_seed=SEED, progressbar=False)
            pm.set_data(payload(pr.arrays(te)))
            pp = pm.sample_posterior_predictive(idf, var_names=['y_obs'], random_seed=SEED + 1, progressbar=False)
        S = pp.posterior_predictive['y_obs'].values.reshape(-1, te.height) * pr.y_sd + pr.y_mu
        yv = te['total_fantasy_points'].to_numpy().astype(float)
        sc = Sc.score(yv, S); sc.pop('pit_hist')
        post = idf.posterior
        key = dict(model=NAME, season=s, week=float(w), eligibility='model')
        fold_rows.append({**key, **sc,
                          'divergences': int(idf.sample_stats['diverging'].sum()),
                          'gamma_center': gamma_center(post),
                          'rhat_max': float(az.summary(idf, var_names=['ell_season', 'eta_season', 'gamma'],
                                                       kind='diagnostics')['r_hat'].max())})
        tier_rows.append(Sc.calibration_table(te, S, by_position=True).rename({'group': 'pos'})
                         .with_columns(pl.lit(NAME).alias('model'), pl.lit(s).alias('season'),
                                       pl.lit(float(w)).alias('week')))
        pl.DataFrame(fold_rows).write_parquet(OUT / 'folds.parquet')        # checkpoint every fold
        pl.concat(tier_rows).write_parquet(OUT / 'tiers.parquet')
        r = fold_rows[-1]
        print(s, w, f"crps {r['crps']:.3f}  div {r['divergences']}  gamma {r['gamma_center']:.2f}  rhat_max {r['rhat_max']:.3f}")
        del idf, pp, mdl



prod_folds = B.load_run(CALIB)[0].filter(pl.col('model') == 'het_gp_vegas')
prod_tiers = B.load_tiers(CALIB).filter(pl.col('model') == 'het_gp_vegas')
folds = pl.concat([prod_folds, pl.DataFrame(fold_rows)], how='diagonal_relaxed')
tiers = pl.concat([prod_tiers, pl.concat(tier_rows)], how='diagonal_relaxed')

print(B.paired_diff(folds, 'scratch_gamma_v2', 'het_gp_vegas', 'crps'))   # negative = scratch better
print(B.paired_diff(folds, 'scratch_gamma_v2', 'het_gp_vegas', 'mae'))
with pl.Config(tbl_cols=-1):
    print(B.tier_summary(tiers))


pl.DataFrame(fold_rows).write_parquet(PATHS.results_dir / 'scratch_gamma_folds.parquet')
pl.concat(tier_rows).write_parquet(PATHS.results_dir / 'scratch_gamma_tiers.parquet')


s, w = 2025, 16
tr, te, _ = fold_frames(s, w, history, team_form)
mdl, pr = build_model(tr)
with mdl:
    idf = pm.sample(draws=1000, tune=1000, chains=4, nuts_sampler='nutpie',
                    target_accept=0.95, random_seed=SEED, progressbar=False)

post = idf.posterior
mine = pl.DataFrame({'var': [str(v) for v in post['cont_vars'].values],
                     'gamma_model': post['cont_prior'].mean(('chain', 'draw')).values})
prod_c = (B.load_run(CALIB)[2]
          .filter((pl.col('model') == 'het_gp_vegas') & (pl.col('season') == s) & (pl.col('week') == float(w)))
          .select('var', pl.col('coef_mean').alias('production')))
cmp = mine.join(prod_c, on='var').with_columns((pl.col('gamma_model') / pl.col('production')).alias('ratio'))
print(cmp.sort(pl.col('production').abs(), descending=True))
print(az.summary(idf, var_names=['gamma', 'alpha_pos', 'sigma_skill', 'eta_season'], kind='stats'))