import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import polars.selectors as cs
import preliz as pz
import pymc as pm
import pytensor.tensor as pt

import models 

SEED = 41029041
np.random.seed(SEED)

DATA_PATH = 'processed-data/ff-processed.parquet'
LAST_SEASON = 2025


def sample_check_coords(frame, col, n=5, seed=None):
    """n random category labels from `frame[col]`, for spot-checking ESS at
    specific coords (a handful of players/teams/etc.) rather than every
    one -- az.plot_ess_evolution over the full dim is unreadable and slow."""
    return frame[col].unique().sample(n, seed=seed).to_list()


def fit_and_diagnose(model, name, target_accept=0.99, prior_check_only=True, samps = 50, save = False, **sample_kwargs, ):
    """Standard workflow shared by all three models: prior predictive check
    -> NUTS -> posterior predictive + log-likelihood -> save to
    model-nc/ff_{name}.nc. Returns the idata. Deliberately does NOT call
    az.plot_ess_evolution here -- run those separately, after the fact,
    against the saved idata (model-nc/ff_{name}.nc) rather than inline with
    fitting, so a too-many-subplots error doesn't cost a refit.

    prior_check_only=True stops right after the prior predictive check and
    returns (idata, ppc_fig) WITHOUT running NUTS -- use this to look at
    the prior predictive plot (and hang onto the figure object -- save it,
    close it, whatever) before committing to a real, possibly slow,
    sample() call. Same spirit as the ESS-evolution note above, just at
    the other end of the function. Default (False) is unchanged from
    before: single idata returned, NUTS + posterior predictive always run,
    existing call sites (id_ar = fit_and_diagnose(...)) don't need to
    change."""
    with model:
        idata = pm.sample_prior_predictive(random_seed=SEED)
    ppc_fig = az.plot_ppc_dist(idata, group='prior_predictive', visuals={'observed_dist': True}, num_samples=samps)
    if prior_check_only:
        return idata, ppc_fig

    with model:
        idata.update(pm.sample(random_seed=SEED, target_accept=target_accept, **sample_kwargs))

    with model:
        pm.compute_log_likelihood(idata)
        pm.stats.compute_log_prior(idata)
        idata.update(pm.sample_posterior_predictive(idata))
    if save:
        idata.to_netcdf(f'model-nc/ff_{name}.nc')
    return idata


# ============================================================================
# Data prep -- full-history-specific (one "fold" covering everything up to
# the current in-progress week of LAST_SEASON). Uses src/models.py's
# generic scaler/encoder helpers so this prep is provably the same
# arithmetic backtest_harness.py's per-fold prep uses, just run once over
# all of history instead of once per fold.
# ============================================================================

raw_data = pl.read_parquet(DATA_PATH)

primary_pos = (
    raw_data.group_by('player_id', 'position').agg(pl.len().alias('n'))
    .sort(['player_id', 'n', 'position'], descending=[False, True, False])   # most games; ties broken alphabetically
    .group_by('player_id', maintain_order=True).first()
    .select('player_id', pl.col('position').alias('primary_position'))
)

raw_data = (
    raw_data.join(primary_pos, on='player_id', how='left')
    .with_columns(pl.col('primary_position').alias('position'))
    .drop('primary_position')
)
# team_form: filtered ("through week i") team-strength signal from the
# standalone Bradley-Terry-style state-space model in
# sandbox/team_strength_ssm.py -- a batched local-level Kalman filter over
# real score margins, fit separately and joined in here as a precomputed
# feature. Deliberately kept OUT of std_these/cont_vars below: that list
# feeds every model's shared cont_priors design matrix, but team_mod's
# whole point (see src/models.py's build_team_mod docstring) is testing
# whether the player's own team's quality matters at all -- silently
# adding it to ar_mod/hs_version too would defeat that ablation.
# Left-joined + fill_null(0.0) rather than inner-joined so a schedule
# mismatch shows up as a printed warning instead of silently dropping rows.
team_form = (pl.read_parquet('processed-data/team_form.parquet')
            .sort(['team', 'season','week'])
            .with_columns(pl.col('team_form').shift(1).over(['week', 'season'])).fill_null(0))
add_team_form = raw_data.select(pl.exclude("^.*_right$")).join(team_form, on=['season', 'week', 'team'], how='left')
n_missing_form = add_team_form['team_form'].null_count()
if n_missing_form:
    print(f"warning: {n_missing_form} rows have no team_form match -- filling with 0.0")
add_team_form = add_team_form.with_columns(pl.col('team_form').fill_null(0.0))
# same signal, keyed on the player's OPPONENT this week -- lets team_mod
# estimate separate coefficients for "my team's strength" vs "their
# defense's strength" instead of assuming the two effects are equal and
# opposite. Rename before the join since both sides otherwise collide on
# 'team'/'team_form'.
opp_team_form = team_form.rename({'team': 'opp_team', 'team_form': 'opp_team_form'})
add_opp_form = add_team_form.join(opp_team_form, on=['season', 'week', 'opp_team'], how='left')
n_missing_opp_form = add_opp_form['opp_team_form'].null_count()
if n_missing_opp_form:
    print(f"warning: {n_missing_opp_form} rows have no opp_team_form match -- filling with 0.0")
add_opp_form = add_opp_form.with_columns(pl.col('opp_team_form').fill_null(0.0))

shrunk_estimates = pl.read_csv('processed-data/shrinkage_features.csv').select('player_id', 'season','week', cs.ends_with('shrunk'))
join_shrink = (
    add_opp_form
    .join(shrunk_estimates, on = ['player_id', 'season', 'week'])
)

# fit on everything up to (not including) the last, still-in-progress week
# of LAST_SEASON. See src/backtest_harness.py for genuine held-out
# accuracy across many such cutoffs -- this file's `train` is just the
# single "fit on everything we have" slice used for diagnostics.
lw = join_shrink.filter(pl.col('season') == LAST_SEASON)['week'].max()
train = join_shrink.filter(
    (pl.col('season') < LAST_SEASON) | ((pl.col('season') == LAST_SEASON) & (pl.col('week') < lw))
)


std_these = ['spread_line',
            'roll_avg_points_allowed',
            'lag_yards_per_rush_attempt',
            'lag_yards_per_target',
            'lag_yards_per_pass_attempt',
            'lag_avg_depth_of_target',
            'lag_target_share',
            'rush_share',
            'temp', 'wind', 'games_missed_ytd',
            'lag_offense_pct', 'lag_st_pct']
binary_vars = ['is_grass', 'is_indoors', 'is_home', 'div_game', 'era']
idx_cols = ['player_id', 'position', 'team', 'week', 'tenure', 'player_season',
            'team_season', 'opp_team', 'season', 'games_missed_ytd']




train_skill = train.filter(pl.col('position') != 'QB')
train_skill_cont = train_skill.select(cs.ends_with('_z'))
train_bi_vars = train_skill.select(binary_vars)


tf_mu, tf_sd = train_skill['team_form'].mean(), train_skill['team_form'].std()
train_skill = train_skill.with_columns(
    ((pl.col('team_form') - tf_mu) / tf_sd).alias('team_form_std'),
    ((pl.col('opp_team_form') - tf_mu) / tf_sd).alias('opp_team_form_std'),   # same scaler, on purpose
)

skill_std = [c for c in std_these if c != 'lag_yards_per_pass_attempt']
scalers = models.fit_scalers(df = train_skill, cols = skill_std)
idxs = models.make_indices(train_skill, idx_cols)
train_skill = models.apply_scalers(train_skill, scalers)
train_skill = models.encode(train_skill, cols = idx_cols, enums = idxs)

train_skill_cont = train_skill.select(cs.ends_with('_z'))
train_bi_vars = train_skill.select(binary_vars)

coords_skill = {col: enum.categories.to_list() for col, enum in idxs.items()}

tenure_grid_vals = np.array(coords_skill['tenure'], dtype = 'float64')
coords_skill['cont_vars'] = train_skill_cont.columns
coords_skill['binary_vars'] = train_bi_vars.columns

player_pos_frame = (
    train_skill.group_by(['player_id_idx'])
    .agg(pl.col('position_idx').first(), pl.col('position_idx').n_unique().alias('n_pos'))
    .sort('player_id_idx')
)



assert (player_pos_frame['n_pos'] == 1).all(), "a skill player is listed at more than one position"
position_of_player_skill = player_pos_frame['position_idx'].to_numpy()
n_pos_skill = len(coords_skill['position'])
n_players_skill = len(coords_skill['player_id'])
n_ten_skill = len(coords_skill['tenure'])

d_skill = {
    'player': train_skill['player_id_idx'].to_numpy().squeeze(),
    'position': train_skill['position_idx'].to_numpy().squeeze(),
    'team': train_skill['team_idx'].to_numpy().squeeze(),
    'tenure': train_skill['tenure_idx'].to_numpy().squeeze(),
    'opp_team': train_skill['opp_team_idx'].to_numpy().squeeze(),
    'season': train_skill['season_idx'].to_numpy().squeeze(),
    'week': train_skill['week_idx'].to_numpy().squeeze(),
    'player_season': train_skill['player_season_idx'].to_numpy().squeeze(),
    'cont': train_skill_cont.to_numpy(),
    'bi': train_bi_vars.to_numpy(),
    'y': train_skill['total_fantasy_points'].to_numpy(),
    'games_missed_z': train_skill['games_missed_ytd_z'].to_numpy(),
    'snap_pct_z': train_skill['lag_offense_pct_z'].to_numpy(),
    'team_form_z': train_skill['team_form_std'].to_numpy(),
    'opp_team_form_z': train_skill['opp_team_form_std'].to_numpy(),
    'target_share_z': train_skill['target_share'].to_numpy(), 
    'rush_share_z': train_skill['rush_share_z'].to_numpy()
}

# probably a touch vague 


def base_data(D):
    """The pm.Data nodes needed by every model below. Not every model uses
    every key as a *used* input (e.g. team_data/season_data are unused by
    ar_mod/hs_version/team_mod's mu, only by the retired team_season_mod)
    -- unused pm.Data nodes are harmless, and one definition here is much
    less likely to drift than several near-identical copies.
 
    `D` must be a dict with the arrays produced by fold_arrays() (or an
    equivalent construction): player, position, team, tenure, cont, bi, y,
    opp_team, season, week, player_season, games_missed_z, snap_pct_z."""
    return {
        'player_data': pm.Data('player_data', D['player']),
        'position_data': pm.Data('position_data', D['position']),
        'team_data': pm.Data('team_data', D['team']),
        'tenure_data': pm.Data('tenure_data', D['tenure']),
        'cont_data': pm.Data('cont_data', np.ascontiguousarray(D['cont'])),
        'bi_data': pm.Data('bi_data', np.ascontiguousarray(D['bi'])),
        'obs_data': pm.Data('obs_data', D['y']),
        'opp_team_data': pm.Data('opp_team_data', D['opp_team']),
        'season_data': pm.Data('season_data', D['season']),
        'week_data': pm.Data('week_data', D['week']),
        'player_season_data': pm.Data('player_season_data', D['player_season']),
        'games_missed_data': pm.Data('games_missed_data', D['games_missed_z']),
        'snap_pct_data': pm.Data('snap_pct_data', D['snap_pct_z']),
        #'global_time_data': pm.Data('global_time_data', global_time(D['season'], D['week'],season_offsets if season_offsets is not None else D['week'])),
        'target_share_data': pm.Data("target_share_data", D['target_share_z']), 
        'rush_share_data': pm.Data('rush_share_data', D['rush_share_z']),
        'team_form_data': pm.Data('team_form_data', D['team_form_z']),
        'opp_team_form_data': pm.Data('opp_team_form_data', D['opp_team_form_z'])
    }





 

y_mu, y_sd = d_skill['y'].mean(), d_skill['y'].std()
d_skill_z = {**d_skill, 'y': (d_skill['y'] - y_mu) / y_sd}

X_ten = tenure_grid_vals[:, None]
JITTER = 1e-6   # same role as WhiteNoise(1e-6) in the PyMC example
 
# how many seasons a player's "level" persists: ~1-4 seasons
season_prior = pz.InverseGamma()
pz.maxent(season_prior, lower=1.5, upper=3, plot=False)     # was 1-4

noise_prior = pz.InverseGamma()
pz.maxent(noise_prior, lower=4, upper=8, plot=False)        
n_pos_skill = len(coords_skill['position'])
position_of_player_skill, _ = models.player_position_map(
    train_skill, 'player_id', 'position', idxs, n_pos_skill
)

pos_onehot = np.eye(n_pos_skill)[position_of_player_skill]     # (n_players, n_positions)
pos_counts = np.maximum(pos_onehot.sum(0), 1.0)
active = np.zeros((len(coords_skill['player_id']), len(coords_skill['tenure'])))
active[d_skill['player'], d_skill['tenure']] = 1.0
n_active = np.maximum(active.sum(0), 1.0) 



### now lets add the player intercepts 
career_prior, _ = pz.maxent(pz.InverseGamma(), lower=4, upper=10,)

ten_counts_pos = np.zeros((n_ten_skill, n_pos_skill))
np.add.at(ten_counts_pos, (d_skill['tenure'], d_skill['position']), 1)
ten_weights_pos = ten_counts_pos / ten_counts_pos.sum(0, keepdims=True)

pos_onehot = np.eye(n_pos_skill)[position_of_player_skill]
pos_counts = np.maximum(pos_onehot.sum(0), 1.0)
 
# active (player, tenure) cells for centering f_season
active = np.zeros((n_players_skill, n_ten_skill))
active[d_skill['player'], d_skill['tenure']] = 1.0
n_active = np.maximum(active.sum(0), 1.0)            # active players per tenure
n_seasons_player = np.maximum(active.sum(1), 1.0)  

ten_weights = np.bincount(d_skill['tenure'], minlength=n_ten_skill).astype(float)
ten_weights /= ten_weights.sum()
with pm.Model(coords=coords_skill) as add_player_intercepts:
    dat = base_data(d_skill_z)
    player = dat['player_data']
    ten = dat['tenure_data']
    pos = dat['position_data']
    
 
    # --- position baseline --------------------------------------------------
    # renamed to plot power scaling curves
    alpha_pos = pm.Normal('alpha_pos', 0.0, 0.5, dims='position')
 
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
        -0.5 * ((tenure_grid_vals[:, None] - t_peak[None, :]) / peak_width) ** 2
    )                                                                        # (tenure, position)
    f_career = pm.Deterministic(
        'f_career', curve - (ten_weights_pos * curve).sum(0), dims=('tenure', 'position')
    )
 
    # --- 2. player skill: career-long level, same spread for every position ---
    sigma_skill = pm.HalfNormal('sigma_skill', 1.0)
    z_skill = pm.Normal('z_skill',  mu = 0.0, sigma = 1.0 ,dims='player_id')
    skill_raw = sigma_skill * z_skill
    pos_mean_skill = (pos_onehot.T @ skill_raw) / pos_counts               # (n_positions,)
    skill = pm.Deterministic('skill', skill_raw - pos_mean_skill[position_of_player_skill], dims='player_id')
 
    # --- 3. season level: per-player GP over tenure, shared kernel -----------
    ell_season = pm.InverseGamma('ell_season', season_prior.alpha, season_prior.beta)
    eta_season = pm.HalfNormal('eta_season', 0.5)
    K_season = (eta_season**2 * pm.gp.cov.Matern32(1, ls=ell_season))(X_ten)
    L_season = pt.linalg.cholesky(pm.gp.util.stabilize(K_season, JITTER))
    z_season = pm.Normal('z_season', 0.0, 1.0, dims=('player_id', 'tenure'))
    # (player, tenure) -- large; drop the Deterministic if memory gets tight and
    # rebuild it post hoc from z_season, ell_season, eta_season
    f_season_raw = z_season @ L_season.T
    f_season_p = f_season_raw - ((f_season_raw * active).sum(1) / n_seasons_player)[:, None]   # mean-zero per player
    f_season = pm.Deterministic(
        'f_season',
        f_season_p - (f_season_p * active).sum(0) / n_active,                                  # mean-zero per tenure
        dims=('player_id', 'tenure'),
    )
    player_level = skill[player] + f_season[player, ten]
    player_level = skill[player] + f_season[player, ten]
    cont_prior = pm.Normal('cont_prior', 0.0, 0.5, dims = 'cont_vars')
    bi_prior = pm.Normal('bi_prior', 0.0, 0.5, dims = 'binary_vars')
    team_form_data = dat['team_form_data']
    opp_team_form_data = dat['opp_team_form_data']

    opp_form_prior = pm.Normal('opp_form_prior', 0, 0.5)
    team_form_prior = pm.Normal('team_form_prior', 0, 0.5)



    mu = (alpha_pos[pos] 
            + f_career[ten, pos] 
            + player_level 
            + pm.math.dot(dat['bi_data'], bi_prior)
            + pm.math.dot(dat['cont_data'], cont_prior)
            + opp_team_form_data * opp_form_prior
            + team_form_data * team_form_prior
            )

    # --- 4. latent log-variance (heteroskedastic white noise) ----------------
    log_sigma0 = pm.Normal('log_sigma0', 0.0, 0.5, dims='position')
    ell_noise = pm.InverseGamma('ell_noise', noise_prior.alpha, noise_prior.beta)
    eta_noise = pm.HalfNormal('eta_noise', 0.15)
    gp_noise = pm.gp.Latent(cov_func=eta_noise**2 * pm.gp.cov.Matern52(1, ls=ell_noise))
    g_noise_raw = gp_noise.prior('g_noise_raw', X=X_ten, jitter=JITTER, dims='tenure')
    g_noise = pm.Deterministic('g_noise', g_noise_raw - pt.dot(ten_weights, g_noise_raw), dims='tenure')
 
    log_sigma = log_sigma0[pos] + g_noise[ten] 
 
    y_obs = pm.Normal('y_obs', mu=mu, sigma=pt.exp(log_sigma), observed=dat['obs_data'])

    idata_int = pm.sample_prior_predictive(random_seed=SEED, draws = 200)

fig,ax = plt.subplots()
az.plot_ppc_dist(
    idata_int, 
    group = 'prior_predictive', 
    visuals={"observed_dist": True}
)


with add_player_intercepts:
    idata_int.update(
        pm.sample(random_seed=SEED, target_accept = 0.95)
    )
    idata_int.update(
        pm.sample_posterior_predictive(
            idata_int, 
            compile_kwargs={'mode': 'NUMBA'}
        )
    )

    pm.stats.compute_log_likelihood(idata_int, compile_kwargs={'mode': 'NUMBA'})
    pm.stats.compute_log_prior(idata_int, compile_kwargs={'mode': 'NUMBA'})



fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True, layout='constrained')
f_career_posterior = az.extract(idata_int, group='posterior', var_names=['f_career']) * y_sd
for ax, p in zip(axes, ['RB', 'WR', 'TE']):
    ax.plot(tenure_grid_vals, f_career_posterior.sel(position=p).isel(sample=slice(0, 30)),
            color='#70133A', alpha=0.3)
    ax.set(title=p, xlabel='Tenure')
axes[0].set_ylabel('Career effect (points)')
plt.suptitle('Prior: aging curve')

def post_mean(v):
    return post[v].mean(('chain', 'draw')).values

def draws(v, **isel):
    """Posterior draws of one slice of v, samples on the last axis."""
    da = post[v].isel(**isel) if isel else post[v]
    return da.stack(sample=('chain', 'draw')).transpose(..., 'sample').values

post = idata_int.posterior
rng_s = np.random.default_rng(SEED)
 
alpha_m = post_mean('alpha')              # (position,)
f_career_m = post_mean('f_career')        # (tenure, position)
skill_m = post_mean('skill')              # (player,)
f_season_m = post_mean('f_season')        # (player, tenure)
cont_m = post_mean('cont_prior')
bi_m = post_mean('bi_prior')
team_form_m = post_mean('team_form_prior')
opp_team_form_m = post_mean('opp_form_prior')


pl_i = np.asarray(d_skill['player'])
ten_i = np.asarray(d_skill['tenure'])
pos_i = np.asarray(d_skill['position'])
n_ten = len(coords_skill['tenure'])
pos_names = np.array(coords_skill['position'])
player_names = np.array(coords_skill['player_id'])
cont_data= np.asarray(d_skill['cont'])
bi_data = np.asarray(d_skill['bi'])
team_form = np.asarray(d_skill['team_form_z'])
opp_form = np.asarray(d_skill['opp_team_form_z'])
#  0. persistence: year-to-year correlation implied by the season kernel
def matern32_corr(d, ell):
    r = np.sqrt(3.0) * d / ell
    return (1 + r) * np.exp(-r)
 
ell_s = post['ell_season'].values.ravel()
for d in (1, 2, 3):
    rho = matern32_corr(d, ell_s)
    print(f"season level correlation {d} season(s) apart: "
          f"{rho.mean():.2f}  (90%: {np.percentile(rho, 5):.2f}-{np.percentile(rho, 95):.2f})")
 
# 1. variance split, in fantasy points
sd_skill = post['sigma_skill'].values.ravel() * y_sd
sd_season = post['eta_season'].values.ravel() * y_sd
share_skill = sd_skill**2 / (sd_skill**2 + sd_season**2)
print(f"sd of career-long skill:   {sd_skill.mean():.2f} pts")
print(f"sd of season level:        {sd_season.mean():.2f} pts")
print(f"share of player-season variation that is career-long: {share_skill.mean():.0%} "
      f"(90%: {np.percentile(share_skill, 5):.0%}-{np.percentile(share_skill, 95):.0%})")
weekly = np.exp(post['log_sigma0']).mean(('chain', 'draw')).values * y_sd
for p, s in zip(pos_names, weekly):
    print(f"baseline weekly sd {p}: {s:.2f} pts")
 
#  2. one row per active player-season
x_eff_z = cont_data @ cont_m + bi_data @ bi_m + team_form * team_form_m + opp_form * opp_team_form_m
base_z = alpha_m[pos_i] + f_career_m[ten_i, pos_i] + skill_m[pl_i] + x_eff_z 
season_df = (
    pl.DataFrame({'player': pl_i, 'tenure': ten_i, 'pos': pos_i,
                  'y': np.asarray(d_skill['y'], float),
                  'dev_z': np.asarray(d_skill_z['y'], float) - base_z,
                  'x_eff': x_eff_z * y_sd})                          # points
    .group_by('player', 'tenure', 'pos')
    .agg(pl.len().alias('games'), pl.col('y').mean().alias('ppg'),
         pl.col('dev_z').mean().alias('raw_dev'), pl.col('x_eff').mean().alias('x_eff'))
)
season_df = season_df.with_columns(
    pl.Series('f_season', f_season_m[season_df['player'].to_numpy(), season_df['tenure'].to_numpy()])
)
print(season_df.describe())
 

fig, ax = plt.subplots(figsize=(6, 5), layout='constrained')
sc = ax.scatter(season_df['raw_dev'] * y_sd, season_df['f_season'] * y_sd,
                c=season_df['games'].clip(upper_bound=17), cmap='viridis', s=6, alpha=0.5)
lim = np.percentile(season_df['raw_dev'] * y_sd, [1, 99])
ax.plot(lim, lim, 'k--', lw=1)
ax.set(xlabel='raw season deviation (pts)', ylabel='f_season (pts)',
       title='Shrinkage of season level (dashed = no shrinkage)')
fig.colorbar(sc, label='games played')
 
shrink = (
    season_df.with_columns(
        pl.when(pl.col('games') <= 4).then(pl.lit('1-4'))
        .when(pl.col('games') <= 10).then(pl.lit('5-10'))
        .otherwise(pl.lit('11+')).alias('games_bucket'))
    .group_by('games_bucket')
    .agg(pl.len().alias('n_seasons'),
         (pl.cov('raw_dev', 'f_season') / pl.col('raw_dev').var()).alias('slope'))
    .sort('games_bucket')
)
print("slope of f_season on raw deviation (1 = no shrinkage, 0 = full):")
print(shrink)
 

yoy = season_df.join(
    season_df.with_columns((pl.col('tenure') - 1).alias('tenure')),
    on=['player', 'tenure'], suffix='_next',
)
print(f"consecutive player-seasons: {yoy.height}")
print(f"corr of f_season, year t vs t+1:        {np.corrcoef(yoy['f_season'], yoy['f_season_next'])[0, 1]:.2f}")
print(f"corr of raw deviation, year t vs t+1:   {np.corrcoef(yoy['raw_dev'], yoy['raw_dev_next'])[0, 1]:.2f}")
print(f"kernel-implied 1-season correlation:    {matern32_corr(1, ell_s).mean():.2f}")
 

long_players = season_df.group_by('player').agg(pl.len().alias('n')).filter(pl.col('n') >= 4)['player'].to_numpy()
featured = ["Christian McCaffrey", "Travis Kelce", "Ja'Marr Chase"]

def find_player(name):
    hits = np.where(np.char.lower(player_labels.astype(str)) == name.lower())[0]
    if len(hits) == 0:
        close = [l for l in player_labels if name.split()[-1].lower() in str(l).lower()]
        raise KeyError(f"{name!r} not found; similar: {close[:5]}")
    return int(hits[0])

pinned = [find_player(n) for n in featured]

# one random long-career player per position, excluding the pinned ones
fill = []
for pos_idx in range(len(pos_names)):
    pool = [p for p in long_players
            if position_of_player_skill[p] == pos_idx and p not in pinned]
    fill.append(int(rng_s.choice(pool)))

pick = pinned + fill
name_col = next(c for c in ['player']
                if c in train_skill.columns)

id_to_name = dict(
    train_skill.select('player_id', 'player')
    .unique(subset='player_id', keep='last')      # latest spelling if a name changed
    .iter_rows()
)
player_labels = np.array([id_to_name.get(pid, pid) for pid in coords_skill['player_id']])
fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True, layout='constrained')
for ax, p in zip(axes.ravel(), pick):
    pos_p = int(position_of_player_skill[p])
    obs_p = season_df.filter(pl.col('player') == p).sort('tenure')
    t_lo, t_hi = obs_p['tenure'].min(), min(obs_p['tenure'].max() + 1, n_ten - 1)  # +1 = next season
    t_rng = np.arange(t_lo, t_hi + 1)
    x_by_t = dict(zip(obs_p['tenure'].to_list(), obs_p['x_eff'].to_list()))
    last_x = obs_p['x_eff'][-1]
    x_add = np.array([x_by_t.get(t, last_x) for t in t_rng])     # points; next season = last season's usage

    level = (draws('alpha', position=pos_p)[None, :]
             + draws('f_career', position=pos_p)[t_rng]
             + draws('skill', player_id=p)[None, :]
             + draws('f_season', player_id=p)[t_rng]) * y_sd + y_mu + x_add[:, None]          # (tenure, samples)
    lo, hi = np.percentile(level, [5, 95], axis=1)
    x = np.asarray(coords_skill['tenure'], float)[t_rng]
    ax.fill_between(x, lo, hi, color='#70133A', alpha=0.25)
    ax.plot(x, level.mean(1), color='#70133A')
    ax.scatter(np.asarray(coords_skill['tenure'], float)[obs_p['tenure'].to_numpy()], obs_p['ppg'],
               s=obs_p['games'] * 6, color='black', zorder=3)
    ax.set(title=f"{player_labels[p]} ({pos_names[pos_p]})", xlabel='Tenure')
for ax in axes[:, 0]:
    ax.set_ylabel('Points per game')
plt.suptitle('Latent level (band = 90%) vs observed season average (dot size = games); last point = next season')




az.plot_ess_evolution(
    idata_int, 
    var_names=[rv.name for rv in add_player_intercepts.free_RVs if rv.size.eval() <= 10]
)

post = idata_int.posterior
s = np.log(post['sigma_skill'].values.ravel())
for v in ['eta_season', 'ell_season']:
    print(v, np.corrcoef(s, np.log(post[v].values.ravel()))[0, 1].round(2))

az.plot_ess_evolution(
    idata_int, 
    var_names=['eta', 'log'], 
    filter_vars='like'
)
az.plot_trace(idata_int, var_names=['eta_career_pos', 'ell_pos'])
dev_pts = (idata_int.posterior['f_career'] - idata_int.posterior['f_career'].mean('position')) * y_sd
print(f"largest position deviation: {float(np.abs(dev_pts.mean(('chain', 'draw'))).max()):.2f} pts")


print(np.corrcoef(np.log(post['eta_noise'].values.ravel()),
                  np.log(post['ell_noise'].values.ravel()))[0, 1].round(2))

az.summary(idata_int, var_names=['t_peak', 't_peak_off', 'peak_height', 'log_height_off'])


az.plot_energy(idata_int)



idata_int.to_netcdf('model-nc/ff_skill_pos_model.nc')


idata_int = az.from_netcdf('model-nc/ff_skill_pos_model.nc')

hyper_priors = ['t_peak_mu', 't_peak_off', 'log_height_mu', 'log_height_off', 'peak_width_raw',
                'sigma_skill', 'eta_season', 'ell_season', 'log_sigma0', 'eta_noise', 'ell_noise']

if 'log_prior' in idata_int.children:
    del idata_int['log_prior']
with add_player_intercepts:
    pm.compute_log_prior(idata_int, var_names=hyper_priors)

print(list(idata_int['log_prior'].data_vars))

az.plot_psense_dist(
    idata_int,
    var_names=['t_peak', 'peak_height', 'peak_width', 'sigma_skill', 'eta_season', 'ell_season', 'eta_noise'],
    visuals = {'dist': False}
)
