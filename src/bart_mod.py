

from numpy.matlib import rand
import pymc as pm 
import pytensor.tensor as pt
import polars as pl
import polars.selectors as cs
import pymc_bart as pmb
import numpy as np 
import arviz as az 
import preliz as pz 
import matplotlib.pyplot as plt 
from models import add_player_skill


SEED = 41029041
np.random.seed(SEED)


DATA_PATH = 'processed-data/ff-processed.parquet'
LAST_SEASON = 2025


# ============================================================================
# Data prep -- full-history-specific (one "fold" covering everything up to
# the current in-progress week of LAST_SEASON). Uses src/py's
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
# whole point (see src/py's build_team_mod docstring) is testing
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

cont_vars = ['spread_line',
            'roll_avg_points_allowed',
            'lag_yards_per_rush_attempt_shrunk',
            'lag_yards_per_target_shrunk',
            'lag_yards_per_pass_attempt_shrunk',
            'lag_target_share',
            'temp', 'wind', 'games_missed_ytd',
            'lag_offense_pct', 'lag_st_pct', 'tenure', 'position', 
            'opp_team_form', 'team_form']
binary_vars = ['is_grass', 'is_indoors', 'is_home', 'div_game', 'era']

def standardize(series):
    return (pl.col(series) - pl.col(series).mean()/pl.col(series).std())

x_train = (
    train.select(
        pl.col(cont_vars + cont_vars)
    )
    .to_dummies('position')

)

unique_players = train['player_id'].sort().unique().to_list()
make_player_idxs = (
    train
    .with_columns(pl.col("player_id").cast(pl.Enum(unique_players)).to_physical().alias('player_idx'))
)

coords = {'player': unique_players}

y_data = train['total_fantasy_points']

def sample_check_coords(frame, col, n=5, seed=None):
    """n random category labels from `frame[col]`, for spot-checking ESS at
    specific coords (a handful of players/teams/etc.) rather than every
    one -- az.plot_ess_evolution over the full dim is unreadable and slow."""
    return frame[col].unique().sample(n, seed=seed).to_list()


def fit_and_diagnose(model, name, target_accept=0.99, prior_check_only=True, samps = 50, **sample_kwargs):
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
        idata.update(pm.sample_posterior_predictive(idata))

    idata.to_netcdf(f'model-nc/ff_{name}.nc')
    return idata







with pm.Model(coords = coords) as bart_mod: 

    covariates_data = pm.Data(
        'covariates', 
        x_train.to_numpy()
    )
    player_data = pm.Data(
        'player_data', 
        make_player_idxs['player_idx'].to_numpy().squeeze()
    )

    bart_mu = pmb.BART('bart_mu', X = covariates_data, Y = y_data, m = 50)

    sigma_player = pm.HalfNormal('sigma_player', 2.0)
    offset = pm.Normal('offset', 0,1, dims = 'player')
    intercepts = pm.Deterministic('intercepts', sigma_player * offset, dims = 'player')


    mu = intercepts[player_data] + bart_mu

    sigma_obs = pm.HalfNormal('sigma_obs', sigma = 3.0)

    pm.StudentT(
        'y_obs', 
        mu = mu, 
        sigma = sigma_obs, 
        nu = 6, 
        observed = y_data
    )


with bart_mod:
    idata_bart = pm.sample_prior_predictive(random_seed=SEED)

az.plot_ppc_dist(
    idata_bart, 
    group = 'prior_predictive', 
    visuals={'observed_dist': True}
)


with bart_mod: 
    idata_bart.update(
        pm.sample(random_seed = SEED)
    )
    

az.plot_ess_evolution(
    idata_bart, 
    var_names=[rv.name for rv in bart_mod.free_RVs if rv.size.eval() <= 10]
)
## this looks kind of fine to be honest. The only way to quibble with it is the r-hats but that is fixable via target accept and more samples 
az.plot_convergence_dist(idata_bart)
az.plot_rank_dist(idata_bart)



with bart_mod:
    pm.compute_log_likelihood(idata_bart, compile_kwargs={'mode': 'NUMBA'})
    idata_bart.update(
        pm.sample_posterior_predictive(idata_bart, compile_kwargs={'mode': 'NUMBA'})
    )


idata_bart.to_netcdf("model-nc/ff-bart.nc")


cont_vars = ['spread_line',
            'roll_avg_points_allowed',
            'lag_yards_per_rush_attempt',
            'lag_yards_per_target',
            'lag_yards_per_pass_attempt',
            'lag_target_share',
            'temp', 'wind', 'games_missed_ytd',
            'lag_offense_pct', 'lag_st_pct', 'tenure', 'position', 
            'opp_team_form', 'team_form']
binary_vars = ['is_grass', 'is_indoors', 'is_home', 'div_game', 'era']

def standardize(series):
    return (pl.col(series) - pl.col(series).mean()/pl.col(series).std())

x_train2 = (
    train.select(
        pl.col(cont_vars + cont_vars)
    )
    .to_dummies('position')

)



with pm.Model(coords = coords) as bart_mod2: 

    covariates_data = pm.Data(
        'covariates', 
        x_train2.to_numpy()
    )
    player_data = pm.Data(
        'player_data', 
        make_player_idxs['player_idx'].to_numpy().squeeze()
    )

    bart_mu = pmb.BART('bart_mu', X = covariates_data, Y = y_data, m = 50)

    sigma_player = pm.HalfNormal('sigma_player', 2.0)
    offset = pm.Normal('offset', 0,1, dims = 'player')
    intercepts = pm.Deterministic('intercepts', sigma_player * offset, dims = 'player')


    mu = intercepts[player_data] + bart_mu

    sigma_obs = pm.HalfNormal('sigma_obs', sigma = 3.0)

    pm.StudentT(
        'y_obs', 
        mu = mu, 
        sigma = sigma_obs, 
        nu = 6, 
        observed = y_data
    )


with bart_mod2:
    id_bart2 = pm.sample_prior_predictive()


az.plot_ppc_dist(
    id_bart2, 
    group = 'prior_predictive', 
    visuals={'observed_dist': True}
)

with bart_mod2:
    id_bart2.update(
        pm.sample(random_seed=SEED)
    )


az.plot_convergence_dist(id_bart2)
az.plot_ess_evolution(
    id_bart2, 
    var_names=[rv.name for rv in bart_mod2.free_RVs if rv.size.eval() <= 5]
)

with bart_mod2:
    id_bart2.update(
        pm.sample_posterior_predictive(id_bart2)
    )

    pm.compute_log_likelihood(id_bart2)


az.compare(
    {'shrunk features': idata_bart, 'regular features': id_bart2}
)





cont_vars = ['spread_line',
            'roll_avg_points_allowed',
            'lag_yards_per_rush_attempt',
            'lag_yards_per_target',
            'lag_yards_per_pass_attempt',
            'lag_target_share',
            'lag_avg_depth_of_target', 
            'temp', 'wind', 'games_missed_ytd',
            'lag_offense_pct', 'lag_st_pct', 
            'opp_team_form', 'team_form']
binary_vars = ['is_grass', 'is_indoors', 'is_home', 'div_game', 'era']


x_train3 = (
    train.select(
        cont_vars + binary_vars
    )
)

unique_tenure = train['tenure'].sort().unique().to_list()
unique_positions = train['position'].sort().unique().to_list()
unique_players = train['player_id'].sort().unique().to_list()


make_idxs = (
    train
    .with_columns(
        pl.col('player_id').cast(pl.Enum(unique_players)).to_physical().alias("player_idx"), 
        pl.col("position").cast(pl.Enum(unique_positions)).to_physical().alias('position_idx'),
        pl.col('tenure').cast(pl.Enum([str(t) for t in unique_tenure])).to_physical().alias('tenure_idx')
    )
)

coords['positions'] = unique_positions
coords['tenure'] = unique_tenure
coords['player_id'] = unique_players

player_position_map = (
    make_idxs
    .group_by('player_idx')
    .agg(pl.col('position_idx').first())
    .sort('player_idx')
)
position_of_player = player_position_map['position_idx'].to_numpy().astype(int)
position_group_counts = np.bincount(position_of_player, minlength=len(unique_positions)).astype(float)


with pm.Model(coords = coords) as bart_mod3: 

    covariates_data = pm.Data(
        'covariates', 
        x_train3.to_numpy()
    )
    player_data = pm.Data(
        'player_data', 
        make_idxs['player_idx'].to_numpy().squeeze()
    )
    tenure_data = pm.Data(
        'tenure_data', 
        make_idxs['tenure_idx'].to_numpy().squeeze()
    )

    postion_data = pm.Data(
        'position_data', 
        make_idxs['position_idx'].to_numpy().squeeze()
    )

    bart_mu = pmb.BART('bart_mu', X = covariates_data, Y = y_data, m = 50)

    player_skill = add_player_skill(
        coords, position_of_player, position_group_counts, 
        n_positions = len(coords['positions']), total_scale=None
    )
    f_career= player_skill[player_data, tenure_data]


    mu = f_career + bart_mu

    sigma_obs = pm.HalfNormal('sigma_obs', sigma = 3.0)

    pm.StudentT(
        'y_obs', 
        mu = mu, 
        sigma = sigma_obs, 
        nu = 6, 
        observed = y_data
    )


with bart_mod3: 
    id_bart3 = pm.sample_prior_predictive()


az.plot_ppc_dist(
    id_bart3, 
    group = 'prior_predictive', 
    visuals = {'observed_dist': True}
)

with bart_mod3: 
    id_bart3.update(
        pm.sample(random_seed=SEED, tune = 1500, draws = 1500, target_accept = 0.95)
    )


az.plot_convergence_dist(id_bart3)

def sample_check_coords(frame, col, n=5, seed=None):
    """n random category labels from `frame[col]`, for spot-checking ESS at
    specific coords (a handful of players/teams/etc.) rather than every
    one -- az.plot_ess_evolution over the full dim is unreadable and slow."""
    return frame[col].unique().sample(n, seed=seed).to_list()

with bart_mod3: 
    pm.compute_log_likelihood(id_bart3)
    id_bart3.update(
        pm.sample_posterior_predictive(id_bart3)
    )


check_players = sample_check_coords(train, 'player')
check_tenure = pl.from_records(coords['tenure']).sample(3)['column_0']

az.plot_ess_evolution(id_bart3, var_names=['player_skill'],
                       coords={'player': check_players, 'tenure': check_tenure})

check_players

az.compare(
    {'player intercepts': id_bart2, 'random walk': id_bart3}
)

with bart_mod3: 
    id_bart3.update(
        pm.sample_posterior_predictive(id_bart3)
    )


az.plot_ppc_dist(
    id_bart3
)


id_bart3.to_netcdf('model-nc/ff_bart.nc')
