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

## effectively just trying to handle taysom hill 
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
            #'lag_yards_per_target',
            'lag_yards_per_pass_attempt',
            'lag_avg_depth_of_target',
            #'lag_target_share',
            'rush_share',
            'temp', 'wind', 'games_missed_ytd',
            'lag_offense_pct']
binary_vars = ['is_grass', 'is_indoors', 'is_home', 'div_game', 'era']
idx_cols = ['player_id', 'position', 'team', 'week', 'age', 'player_season',
            'team_season', 'opp_team', 'season', 'games_missed_ytd']


train_qb = train.filter(pl.col('position') == 'QB')
train_qb_cont = train_qb.select(cs.ends_with('_z'))
train_bi_vars = train_qb.select(binary_vars)


tf_mu, tf_sd = train_qb['team_form'].mean(), train_qb['team_form'].std()
train_qb = train_qb.with_columns(
    ((pl.col('team_form') - tf_mu) / tf_sd).alias('team_form_std'),
    ((pl.col('opp_team_form') - tf_mu) / tf_sd).alias('opp_team_form_std'),   # same scaler, on purpose
)


scalers = models.fit_scalers(df = train_qb, cols = std_these)
idxs = models.make_indices(train_qb, idx_cols)
train_qb = models.apply_scalers(train_qb, scalers)
train_qb = models.encode(train_qb, cols = idx_cols, enums = idxs)

train_qb_cont = train_qb.select(cs.ends_with('_z'))
train_bi_vars = train_qb.select(binary_vars)

coords_qb = {col: enum.categories.to_list() for col, enum in idxs.items()}

tenure_grid_vals = np.array(coords_qb['age'], dtype = 'float64')
coords_qb['cont_vars'] = train_qb_cont.columns
coords_qb['binary_vars'] = train_bi_vars.columns

player_pos_frame = (
    train_qb.group_by(['player_id_idx'])
    .agg(pl.col('position_idx').first(), pl.col('position_idx').n_unique().alias('n_pos'))
    .sort('player_id_idx')
)



assert (player_pos_frame['n_pos'] == 1).all(), "a skill player is listed at more than one position"
position_of_player_skill = player_pos_frame['position_idx'].to_numpy()
n_pos_skill = len(coords_qb['position'])
n_players_skill = len(coords_qb['player_id'])
n_ten_skill = len(coords_qb['age'])

d_qb = {
    'player': train_qb['player_id_idx'].to_numpy().squeeze(),
    'position': train_qb['position_idx'].to_numpy().squeeze(),
    'team': train_qb['team_idx'].to_numpy().squeeze(),
    'tenure': train_qb['age_idx'].to_numpy().squeeze(),
    'opp_team': train_qb['opp_team_idx'].to_numpy().squeeze(),
    'season': train_qb['season_idx'].to_numpy().squeeze(),
    'week': train_qb['week_idx'].to_numpy().squeeze(),
    'player_season': train_qb['player_season_idx'].to_numpy().squeeze(),
    'cont': train_qb_cont.to_numpy(),
    'bi': train_bi_vars.to_numpy(),
    'y': train_qb['total_fantasy_points'].to_numpy(),
    'games_missed_z': train_qb['games_missed_ytd_z'].to_numpy(),
    'snap_pct_z': train_qb['lag_offense_pct_z'].to_numpy(),
    'team_form_z': train_qb['team_form_std'].to_numpy(),
    'opp_team_form_z': train_qb['opp_team_form_std'].to_numpy(),
    'target_share_z': train_qb['target_share'].to_numpy(), 
    'rush_share_z': train_qb['rush_share_z'].to_numpy()
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


n_players_qb = len(coords_qb['player_id'])
n_ten_qb = len(coords_qb['age'])

y_mu, y_sd = d_qb['y'].mean(), d_qb['y'].std()
d_qb_z = {**d_qb, 'y': (d_qb['y'] - y_mu) / y_sd}
 
X_ten = tenure_grid_vals[:, None]
JITTER = 1e-6   # same role as WhiteNoise(1e-6) in the PyMC example
 
# how long a within-career swing persists: ~1.5-3 seasons
season_prior = pz.InverseGamma()
pz.maxent(season_prior, lower=1, upper=4,)
 

 
# active (player, tenure) cells for centering f_season
active = np.zeros((n_players_qb, n_ten_qb))
active[d_qb['player'], d_qb['tenure']] = 1.0
n_active = np.maximum(active.sum(0), 1.0)            # active QBs per tenure
n_seasons_player = np.maximum(active.sum(1), 1.0)    # active seasons per QB
 
# share of QB weeks at each tenure: centering weights for the aging bump + noise GP
ten_weights = np.bincount(d_qb['tenure'], minlength=n_ten_qb).astype(float)
ten_weights /= ten_weights.sum()



with pm.Model(coords=coords_qb) as qb_model:
    dat = base_data(d_qb_z)
    player = dat['player_data']
    ten = dat['tenure_data']
 
    # --- QB baseline ---------------------------------------------------------
    # named alpha_qb: ArviZ's power-scaling plots use an 'alpha' coordinate
    alpha_qb = pm.Normal('alpha_qb', 0.0, 0.5)
 
    # --- 1. aging curve --------------------------------------------------------
    # QBs peak later and stay near peak longer than skill players. Age
    # 0-indexed (youngest = 0), so Normal(28,5, 3) centers the peak around the
    # their second contract. 
    t_peak = pm.Normal('t_peak', 28.5, 3.0)
 
    # peak height on the log scale: ~0.25-1 sd above the average QB week
    log_height = pm.Normal('log_height', np.log(0.5), 0.4)
    peak_height = pm.Deterministic('peak_height', pt.exp(log_height))
 
    # width floored at one season; Gamma(8, 1.5) -> ~5.3, so a ~6-season prime
    width_rise_raw = pm.Gamma('width_rise_raw', alpha=4, beta=1.5)       # ~2.7 -> rise width ~3.7 years
    width_decline_raw = pm.Gamma('width_decline_raw', alpha=8, beta=1.2)  # ~6.7 -> decline width ~7.7 years
    width_rise = pm.Deterministic('width_rise', 1.0 + width_rise_raw)
    width_decline = pm.Deterministic('width_decline', 1.0 + width_decline_raw)

    w = pt.switch(tenure_grid_vals < t_peak, width_rise, width_decline)
    curve = peak_height * pt.exp(-0.5 * ((tenure_grid_vals - t_peak) / w) ** 2)
    f_career = pm.Deterministic('f_career', curve - pt.dot(ten_weights, curve), dims='age')
 
    # --- 2. QB skill: career-long level, mean-zero across QBs -------------------
    # HalfNormal(0.75): fewer players than the skill model, so a tighter
    # scale keeps the prior predictive from wandering
    sigma_skill = pm.HalfNormal('sigma_skill', 1.0)
    z_skill = pm.ZeroSumNormal('z_skill', sigma=1.0, dims='player_id')
    skill = pm.Deterministic('skill', sigma_skill * z_skill, dims='player_id')
 
    # --- 3. season level: per-QB GP over tenure, shared kernel -------------------
    ell_season = pm.InverseGamma('ell_season', season_prior.alpha, season_prior.beta)
    eta_season = pm.HalfNormal('eta_season', 0.5)
    K_season = (eta_season**2 * pm.gp.cov.Matern32(1, ls=ell_season))(X_ten)
    L_season = pt.linalg.cholesky(pm.gp.util.stabilize(K_season, JITTER))
    z_season = pm.Normal('z_season', 0.0, 1.0, dims=('player_id', 'age'))
    f_season_raw = z_season @ L_season.T
    f_season_p = f_season_raw - ((f_season_raw * active).sum(1) / n_seasons_player)[:, None]   # mean-zero per QB
    f_season = pm.Deterministic(
        'f_season',
        f_season_p - (f_season_p * active).sum(0) / n_active,                                  # mean-zero per tenure
        dims=('player_id', 'age'),
    )
    player_level = skill[player] + f_season[player, ten]
 
    cont_prior = pm.Normal('cont_prior', 0.0, 0.5, dims='cont_vars')
    bi_prior = pm.Normal('bi_prior', 0.0, 0.5, dims='binary_vars')
    opp_form_prior = pm.Normal('opp_form_prior', 0, 0.5)
    team_form_prior = pm.Normal('team_form_prior', 0, 0.5)
 
    mu = (alpha_qb
        + f_career[ten]
        + player_level
        + pm.math.dot(dat['bi_data'], bi_prior)
        + pm.math.dot(dat['cont_data'], cont_prior)
        + dat['opp_team_form_data'] * opp_form_prior
        + dat['team_form_data'] * team_form_prior
        )
 
    # --- 4. latent log-variance (heteroskedastic white noise) ----------------
    sigma_obs = pm.HalfNormal('sigma_obs', 1)
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma_obs, observed=dat['obs_data'])
 
    idata_qb = pm.sample_prior_predictive(random_seed=SEED, draws=200)


fig, ax = plt.subplots()
az.plot_ppc_dist(
    idata_qb, 
    group = 'prior_predictive', 
    visuals={'observed_dist': True}
)






fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
f_career_prior = az.extract(idata_qb, group='prior', var_names=['f_career']) * y_sd
ax.plot(tenure_grid_vals, f_career_prior.isel(sample=slice(0, 40)), color='#70133A', alpha=0.3)
ax.axhline(0, color='black', lw=0.8, ls='--')
ax.set(xlabel='Tenure', ylabel='Points vs. average QB week', title='Prior: QB aging curve')
 

with qb_model:
    idata_qb.update(
        pm.sample(random_seed=SEED, target_accept = 0.95)
    )
    idata_qb.update(
        pm.sample_posterior_predictive(idata_qb)
    )

    pm.stats.compute_log_likelihood(idata_qb)
    pm.stats.compute_log_prior(idata_qb)


fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
f_career_posterior = az.extract(idata_qb, group='posterior', var_names=['f_career']) * y_sd
ax.plot(tenure_grid_vals, f_career_posterior.isel(sample=slice(0, 40)), color='#70133A', alpha=0.3)
ax.axhline(0, color='black', lw=0.8, ls='--')
ax.set(xlabel='Tenure', ylabel='Points vs. average QB week', title='Posterior: QB aging curve')


def post_mean(v):
    return post[v].mean(('chain', 'draw')).values
 
 
def draws(v, **isel):
    """Posterior draws of one slice of v, samples on the last axis."""
    da = post[v].isel(**isel) if isel else post[v]
    return da.stack(sample=('chain', 'draw')).transpose(..., 'sample').values
 
 
post = idata_qb.posterior
rng_s = np.random.default_rng(SEED)
 
alpha_m = float(post_mean('alpha_qb'))    # scalar
f_career_m = post_mean('f_career')        # (tenure,)
skill_m = post_mean('skill')              # (player,)
f_season_m = post_mean('f_season')        # (player, tenure)
cont_m = post_mean('cont_prior')
bi_m = post_mean('bi_prior')
team_form_m = post_mean('team_form_prior')
opp_team_form_m = post_mean('opp_form_prior')
 
pl_i = np.asarray(d_qb['player'])
ten_i = np.asarray(d_qb['tenure'])
n_ten = len(coords_qb['age'])
player_names = np.array(coords_qb['player_id'])
cont_data = np.asarray(d_qb['cont'])
bi_data = np.asarray(d_qb['bi'])
team_form = np.asarray(d_qb['team_form_z'])
opp_form = np.asarray(d_qb['opp_team_form_z'])
 
print(f"QB prime: peak at tenure {float(post['t_peak'].mean()):.1f}, "
      f"height {float(post['peak_height'].mean()) * y_sd:.1f} pts, width {float(post['peak_width'].mean()):.1f} seasons")
 
 
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
print(f"share of QB-season variation that is career-long: {share_skill.mean():.0%} "
      f"(90%: {np.percentile(share_skill, 5):.0%}-{np.percentile(share_skill, 95):.0%})")

 
#  2. one row per active QB-season
x_eff_z = cont_data @ cont_m + bi_data @ bi_m + team_form * team_form_m + opp_form * opp_team_form_m
base_z = alpha_m + f_career_m[ten_i] + skill_m[pl_i] + x_eff_z
season_df = (
    pl.DataFrame({'player': pl_i, 'tenure': ten_i,
                  'y': np.asarray(d_qb['y'], float),
                  'dev_z': np.asarray(d_qb_z['y'], float) - base_z,
                  'x_eff': x_eff_z * y_sd})                          # points
    .group_by('player', 'tenure')
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
print(f"consecutive QB-seasons: {yoy.height}")
print(f"corr of f_season, year t vs t+1:        {np.corrcoef(yoy['f_season'], yoy['f_season_next'])[0, 1]:.2f}")
print(f"corr of raw deviation, year t vs t+1:   {np.corrcoef(yoy['raw_dev'], yoy['raw_dev_next'])[0, 1]:.2f}")
print(f"kernel-implied 1-season correlation:    {matern32_corr(1, ell_s).mean():.2f}")
 
# names first -- find_player reads player_labels
id_to_name = dict(
    train_qb.select('player_id', 'player')
    .unique(subset='player_id', keep='last')      # latest spelling if a name changed
    .iter_rows()
)
player_labels = np.array([id_to_name.get(pid, pid) for pid in coords_qb['player_id']])
 
long_players = season_df.group_by('player').agg(pl.len().alias('n')).filter(pl.col('n') >= 4)['player'].to_numpy()
featured = ["Patrick Mahomes", "Josh Allen", "Tom Brady", 'Drew Brees']   # QB-only data, so no clash with the other Josh Allen
 
 
def find_player(name):
    hits = np.where(np.char.lower(player_labels.astype(str)) == name.lower())[0]
    if len(hits) == 0:
        close = [l for l in player_labels if name.split()[-1].lower() in str(l).lower()]
        raise KeyError(f"{name!r} not found; similar: {close[:5]}")
    return int(hits[0])
 
 
pinned = [find_player(n) for n in featured]
 
# three random long-career QBs, excluding the pinned ones
pool = [p for p in long_players if p not in pinned]
fill = [int(p) for p in rng_s.choice(pool, size=min(6 - len(pinned), len(pool)), replace=False)]
pick = pinned + fill
 
fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True, layout='constrained')
for ax, p in zip(axes.ravel(), pick):
    obs_p = season_df.filter(pl.col('player') == p).sort('tenure')
    t_lo, t_hi = obs_p['tenure'].min(), min(obs_p['tenure'].max() + 1, n_ten - 1)  # +1 = next season
    t_rng = np.arange(t_lo, t_hi + 1)
    x_by_t = dict(zip(obs_p['tenure'].to_list(), obs_p['x_eff'].to_list()))
    last_x = obs_p['x_eff'][-1]
    x_add = np.array([x_by_t.get(t, last_x) for t in t_rng])     # points; next season = last season's usage
 
    level = (draws('alpha_qb')[None, :]
             + draws('f_career')[t_rng]
             + draws('skill', player_id=p)[None, :]
             + draws('f_season', player_id=p)[t_rng]) * y_sd + y_mu + x_add[:, None]          # (tenure, samples)
    lo, hi = np.percentile(level, [5, 95], axis=1)
    x = tenure_grid_vals[t_rng]
    ax.fill_between(x, lo, hi, color='#70133A', alpha=0.25)
    ax.plot(x, level.mean(1), color='#70133A')
    ax.scatter(tenure_grid_vals[obs_p['tenure'].to_numpy()], obs_p['ppg'],
               s=obs_p['games'] * 6, color='black', zorder=3)
    ax.set(title=str(player_labels[p]), xlabel='Tenure')
for ax in axes[:, 0]:
    ax.set_ylabel('Points per game')
plt.suptitle('QB latent level (band = 90%) vs observed season average (dot size = games); last point = next season')




az.plot_ess_evolution(
    idata_qb, 
    var_names = [rv.name for rv in qb_model.free_RVs if rv.size.eval() <= 10]
)
prior_vars = ['alpha_qb', 't_peak', 'log_height', 'width_rise_raw', 'width_decline_raw',
              'sigma_skill', 'ell_season', 'eta_season', 'sigma_obs']

psense_vars = ['t_peak', 'peak_height', 'width_rise', 'width_decline',
               'sigma_skill', 'eta_season', 'ell_season', 'alpha_qb']

if 'log_prior' in idata_qb.children:
    del idata_qb['log_prior']
with qb_model:
    pm.compute_log_prior(idata_qb, var_names=prior_vars)


az.plot_psense_dist(
    idata_qb,
    var_names=psense_vars,
    visuals = {'dist': False}
)
