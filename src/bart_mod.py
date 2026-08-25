

import pymc as pm 
import pytensor.tensor as pt
import polars as pl
import polars.selectors as cs
import pymc_bart as pmb
import numpy as np 
import arviz as az 
import preliz as pz 
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotnine as gg

SEED = 41029041
np.random.seed(SEED)


def fit_scalers(df, cols):
    stats = df.select(
        [pl.col(c).mean().alias(f"{c}__mu") for c in cols] +
        [pl.col(c).std().alias(f"{c}__sd")  for c in cols]
    ).row(0, named=True)
    return {c: (stats[f"{c}__mu"], stats[f"{c}__sd"]) for c in cols}

def apply_scalers(frame, scalers):
    return frame.with_columns([
        ((pl.col(c) - mu) / sd).alias(f"{c}_z") for c, (mu, sd) in scalers.items()
    ])

def make_indices(df, cols):
    # Return a dict mapping each column name to its Enum dtype
    return {
        col: pl.Enum(df[col].unique().sort().cast(pl.String).to_list())
        for col in cols
    }

def encode(df, cols, enums):
    # enums is the dict returned by make_indices
    return df.with_columns([
        pl.col(col).cast(pl.String).cast(enums[col]).to_physical().cast(pl.Int32).alias(f"{col}_idx")
        for col in cols
    ])

raw_data= (
    pl.read_parquet('processed-data/ff-processed.parquet')
    .to_dummies('position')
)

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

LAST_SEASON = 2025 

lw = (
    add_opp_form
    .filter(
        (pl.col('season') == LAST_SEASON)
    )['week'].max()
)

train = (
    add_opp_form.filter(
        (pl.col('season') < LAST_SEASON) | ((pl.col('season') == LAST_SEASON) &
        (pl.col('week') < lw))
    )
)


test  = (
    add_opp_form
    .filter((pl.col('season') == LAST_SEASON) & (pl.col('week') == lw))
)
std_these = ['spread_line', 'roll_avg_points_allowed', 'lag_yards_per_rush_attempt', 'lag_target_share', 'lag_yards_per_target', 'lag_yards_per_pass_attempt', 'temp', 'wind', 'tenure', 'games_missed_ytd',  'opp_team_form']

binary_vars = ['is_grass', 'is_indoors', 'is_home', 'div_game', 'era']

idx_cols = ['player',  'player_season', 'opp_team', 'season', 'week']
idxs = make_indices(raw_data, idx_cols)
train = encode(train, cols = idx_cols, enums = idxs)
test = encode(test,cols = idx_cols ,enums = idxs)

scalers = fit_scalers(df = train, cols = std_these)
train = apply_scalers(train, scalers)
test  = apply_scalers(test,  scalers) 

train.glimpse()

x_train = (
    train.select(
        cs.ends_with('_z'), 
        cs.starts_with('team'), 
        cs.starts_with('position'),
        pl.col(binary_vars)
    ).drop('team_season', 'team')
)

x_train.glimpse()

coords = {col: enum.categories.to_list() for col, enum in idxs.items()}
weeks_per_season = (
    raw_data
    .group_by('season')
    .agg(
        pl.col('week').max().alias('max_week')
    )
    .sort("season")
    ['max_week']
    .cast(pl.Int64)
    .to_numpy()
)

n_seasons = len(coords['season'])
assert len(weeks_per_season) == n_seasons
season_offsets = np.concatenate([[0], np.cumsum(weeks_per_season)])  
T = season_offsets[-1]  
season_boundary    = np.zeros(T, dtype=bool)
boundary_step_idx  = season_offsets[1:-1]    
season_boundary[boundary_step_idx] = True
boundary_cols = boundary_step_idx - 1 

coords['global_step'] = list(range(T - 1))
coords['global_step_full'] = list(range(T))
coords['season_gap'] = list(range(n_seasons - 1))

with pm.Model(coords = coords) as bart_mod: 

    player_data = pm.Data(
        'player_data',
        train['player_idx'].to_numpy().squeeze()
    )
    
    x_data = pm.Data(
        'x_data', 
        x_train.to_numpy()
    )

    opp_team_data = pm.Data(
        'opp_team_data',
        train['opp_team_idx'].to_numpy().squeeze()

    )
    season_offset_lookup = season_offsets[:-1]
    global_time_data = pm.Data(
        'global_time_data', 
        (season_offset_lookup[train['season_idx'].to_numpy().squeeze()] + train['week_idx'].to_numpy().squeeze()).astype('int32')
    )

    season_data  = pm.Data(
        'season_data', 
        train['season_idx'].to_numpy().squeeze()
    )

    obs_data = pm.Data(
        'obs_data', 
        train['total_fantasy_points'].to_numpy()
    )

    player_sigma = pm.HalfStudentT('player_sigma', nu = 4, sigma = 1.5)
    player_mu = pm.Normal('player_mu', 0, 1 , dims = 'player')

    player_intercepts = pm.Deterministic(
        'player_intercepts',
        player_mu *  player_sigma, 
        dims = 'player'
    )


    mu_bart = pmb.BART(
        'mu_bart', X = x_data, Y = obs_data, m = 50
    )

    opp_rho          = pm.Beta('opp_rho', 4, 2)
    opp_sigma_theta  = pm.HalfNormal('opp_sigma_theta', 1.0)
    opp_sigma_season = pm.HalfNormal('opp_sigma_season', 1.0)


    opp_theta_init = pm.Normal('opp_theta_init', 0,  1, dims = 'opp_team')
    z_theta = pm.Normal('z_theta', 0, 1, dims = ('opp_team', 'global_step'))
    z_eta = pm.Normal('z_eta', 0,1, dims = ("opp_team", 'season_gap'))

    innov = opp_sigma_theta * z_theta
    innov = pt.set_subtensor(
        innov[:, boundary_cols],
        innov[:, boundary_cols] + opp_sigma_season * z_eta
    )

    tt = np.arange(T - 1)
    lag = tt[:, None] - tt[None, :]
    rho_powers = (lag >= 0).astype('float64') * (opp_rho**np.clip(lag, 0, None))

    innov_path = pt.tensordot(innov, rho_powers, axes = [[-1], [-1]])
    init_path = opp_theta_init[:, None] * (opp_rho**np.arange(1, T))[None, :]
    theta_full = pt.concatenate([opp_theta_init[:, None], init_path + innov_path], axis = 1)

    opp_ar = pm.Deterministic(
        'opp_ar',
        theta_full - pt.mean(theta_full, axis = 0, keepdims=True),
        dims = ('opp_team', 'global_step_full')

    )

    f_opp = opp_ar[opp_team_data, global_time_data]

    f_opp = opp_ar[opp_team_data, global_time_data]

    mu = pm.Deterministic(
        'mu', 
        mu_bart +  player_intercepts[player_data]  + f_opp  )
        

    sigma_obs = pm.HalfNormal('sigma_obs', 1.5)
    #nu = pm.Gamma('nu', alpha = 2, beta= 0.1)
    

    pm.StudentT(
        'y_obs', 
        mu = mu, 
        sigma = sigma_obs, 
        nu = 6, 
        observed = obs_data
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
        pm.sample(random_seed = SEED, target_accept = 0.95,  step=[pmb.PGBART([mu_bart], num_particles=40, batch=(0.1, 0.3))],
    ))
    

az.plot_ess_evolution(
    idata_bart, 
    var_names=[rv.name for rv in bart_mod.free_RVs if rv.size.eval() <= 10]
)


az.summary(
    idata_bart,
    var_names=[rv.name for rv in bart_mod.free_RVs if rv.size.eval() <= 10], 
    kind = 'diagnostics'
)


check_players = train['player'].unique().sample(5).to_list()
check_tenure = pl.from_records(coords['tenure']).sample(3)['column_0']

az.plot_ess_evolution(
    idata_bart, 
    var_names=['player_skill'], 
    coords = {'player': check_players, 'tenure': check_tenure}
)

check_opp = train['opp_team'].unique().sample(5).to_list()
check_global_step = pl.from_records(coords['global_step_full']).sample(5)['column_0']
az.plot_ess_evolution(
    idata_bart, 
    var_names=['opp_ar'], 
    coords = {'opp_team': check_opp, 
            'global_step_full': check_global_step}
)

az.plot_ess_evolution(
    idata_bart, 
    var_names = ['opp_theta_init', 'opp_rho', 'opp_sigma_theta']
)




with bart_mod:
    pm.compute_log_likelihood(idata_bart, compile_kwargs={'mode': 'NUMBA'})
    idata_bart.update(
        pm.sample_posterior_predictive(idata_bart, compile_kwargs={'mode': 'NUMBA'})
    )


idata_bart.to_netcdf("model-nc/ff-bart.nc")
