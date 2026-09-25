
import arviz as az
import numpy as np
import polars as pl
import polars.selectors as cs
import pymc as pm

import models

SEED = 41029041
np.random.seed(SEED)

DATA_PATH = 'processed-data/ff-processed.parquet'
LAST_SEASON = 2025


# ============================================================================
# Data prep -- full-history-specific (one "fold" covering everything up to
# the current in-progress week of LAST_SEASON). Uses src/models.py's
# generic scaler/encoder helpers so this prep is provably the same
# arithmetic backtest_harness.py's per-fold prep uses, just run once over
# all of history instead of once per fold.
# ============================================================================

raw_data = pl.read_parquet(DATA_PATH)

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
team_form = pl.read_parquet('processed-data/team_form.parquet')
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

scalers = models.fit_scalers(df=train, cols=std_these)
idxs = models.make_indices(raw_data, idx_cols)
train = models.apply_scalers(train, scalers)
train = models.encode(train, cols=idx_cols, enums=idxs)

# standardized separately from std_these/apply_scalers on purpose -- that
# helper always suffixes '_z', and cont_dat_train below sweeps up every
# '_z' column automatically. Naming this '_std' instead keeps it out of
# that sweep, so it only appears where a model explicitly asks for it
# (team_mod, via D['team_form_z']/D['opp_team_form_z'] below) rather than
# silently entering ar_mod/hs_version's shared design matrix too.
_team_form_mu = train['team_form'].mean()
_team_form_sd = train['team_form'].std()
train = train.with_columns(
    ((pl.col('team_form') - _team_form_mu) / _team_form_sd).alias('team_form_std'),
    # same scaler as team_form (not its own mean/sd) -- it's the same
    # underlying quantity just re-keyed on opp_team, so sharing the scaler
    # keeps beta_team_form and beta_opp_team_form directly comparable in
    # magnitude rather than each being in a slightly different unit.
    ((pl.col('opp_team_form') - _team_form_mu) / _team_form_sd).alias('opp_team_form_std'),
)

coords = {col: enum.categories.to_list() for col, enum in idxs.items()}

cont_dat_train = train.select(cs.ends_with('_z'))
coords['cont_vars'] = cont_dat_train.columns

binary_train = train.select(binary_vars)
coords['binary_vars'] = binary_train.columns

weeks_per_season = (
    raw_data.group_by('season').agg(pl.col('week').max().alias('max_week'))
    .sort('season')['max_week'].cast(pl.Int64).to_numpy()
)
n_seasons = len(coords['season'])
assert len(weeks_per_season) == n_seasons
season_offsets, T, boundary_cols = models.build_ar_calendar(weeks_per_season)

# player -> position lookup, for the within-position centering of
# player_skill in models.add_player_skill(): positions_mu[position] (or
# ab_position's intercept for the LKJ models) and the average starting
# level of the player_skill random walk within that position are
# otherwise redundant, which tanks the position-level parameters' ESS.
n_positions = len(coords['position'])
position_of_player, position_group_counts = models.player_position_map(
    raw_data, 'player_id', 'position', idxs, n_positions
)
assert len(position_of_player) == len(coords['player_id']), (
    f"position_of_player covers {len(position_of_player)} players, "
    f"expected {len(coords['player'])} -- check idx_cols/coords alignment"
)

# standardized tenure, one value per row -- the LKJ models' linear
# age/tenure effect (varying by position) acts on this directly instead
# of a per-position HSGP curve.
tenure_grid_vals = np.array(coords['tenure'], dtype='float64')
tenure_z_row_vals = (
    train['tenure_idx'].to_numpy().astype('float64') - tenure_grid_vals.mean()
) / tenure_grid_vals.std()

# week-within-season grid for the recent_form components -- coords['week']
# is already just the 18 distinct week-of-season values, so this doubles
# as both the AR/VAR time axis and the HSGP GP input with no extra
# derivation.
week_grid_vals = np.array(coords['week'], dtype='float64')

D = {
    'player': train['player_id_idx'].to_numpy().squeeze(),
    'position': train['position_idx'].to_numpy().squeeze(),
    'team': train['team_idx'].to_numpy().squeeze(),
    'tenure': train['tenure_idx'].to_numpy().squeeze(),
    'opp_team': train['opp_team_idx'].to_numpy().squeeze(),
    'season': train['season_idx'].to_numpy().squeeze(),
    'week': train['week_idx'].to_numpy().squeeze(),
    'player_season': train['player_season_idx'].to_numpy().squeeze(),
    'cont': cont_dat_train.to_numpy(),
    'bi': binary_train.to_numpy(),
    'y': train['total_fantasy_points'].to_numpy(),
    'tenure_z': tenure_z_row_vals,
    'games_missed_z': train['games_missed_ytd_z'].to_numpy(),
    'snap_pct_z': train['lag_offense_pct_z'].to_numpy(),
    'team_form_z': train['team_form_std'].to_numpy(),
    'opp_team_form_z': train['opp_team_form_std'].to_numpy(),
    'target_share_z': train['target_share'].to_numpy(), 
    'rush_share_z': train['rush_share_z'].to_numpy()
}


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




team_mod = models.build_team_mod(D, coords, season_offsets, T, boundary_cols,
                                  position_of_player, position_group_counts,)



id_team = fit_and_diagnose(team_mod, 'team')
### probably need to bump the sampling up
id_team = fit_and_diagnose(team_mod, 'team_mod_sans_ar', prior_check_only=False, save = True, tune = 1000, draws = 1000)

check_players = sample_check_coords(train, 'player_id')
check_tenure = pl.from_records(coords['tenure']).sample(3)['column_0']

az.plot_ess_evolution(id_team, var_names=['player_skill'],
                       coords={'player_id': check_players, 'tenure': check_tenure})
az.plot_ess_evolution(id_team, var_names=['ab_position'])

id_team.sample_stats['diverging'].sum().item()

check = az.psense_summary(id_team)

check_pl = pl.from_pandas(
    check.reset_index()
)



az.rcParams['plot.max_subplots'] = 60

az.plot_ess_evolution(
    id_team,
    var_names=[rv.name for rv in team_mod.free_RVs if rv.size.eval() <= 10]
)
az.plot_ess_evolution(id_team, var_names=['beta_team_form', 'beta_opp_team_form'])


# ============================================================================
# Model comparison
# ============================================================================
idata_ar = az.from_netcdf('model-nc/ff_ar.nc')
id_hs = az.from_netcdf('model-nc/ff_hs.nc')
id_team = az.from_netcdf('model-nc/ff_team.nc')


## one run of the comparisions had observations in the log likelihood that prevented comparision
for name, idata in {'ar_mod': idata_ar, 'hs_version': id_hs, 'team_mod': id_team}.items():
    ll = idata.log_likelihood['y_obs'].values
    n_nan = np.isnan(ll).sum()
    n_inf = np.isinf(ll).sum()
    print(f"{name}: shape={ll.shape} nan={n_nan} inf={n_inf}")
    if n_nan or n_inf:
        bad_obs = np.where(np.isnan(ll).any(axis=(0, 1)) | np.isinf(ll).any(axis=(0, 1)))[0]
        print(f"  {len(bad_obs)} distinct observations affected, e.g. indices: {bad_obs[:10]}")

az.compare({
    'ar_mod': idata_ar,
    'hs_version': id_hs,
    'team_mod': id_team,

})

for name, idata in {'ar_mod': idata_ar, 'hs_version': id_hs,
                     'team_mod': id_team, }.items():
    print(f"\n{name}")
    print(az.loo(idata, pointwise=True))
