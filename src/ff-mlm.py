
from arviz_base import rcparams
import pymc as pm 
import pytensor.tensor as pt
import polars as pl
import polars.selectors as cs
import tidydraws as td 
import numpy as np 
import arviz as az 
import preliz as pz 
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotnine as gg
from scipy.stats import gaussian_kde, skew, spearmanr
from dataclasses import dataclass, field
from typing import Callable, Sequence
SEED = 41029041
np.random.seed(SEED)


raw_data= (
    pl.read_parquet('processed-data/ff-processed.parquet')
)

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

LAST_SEASON = 2025 

lw = (
    raw_data
    .filter(
        (pl.col('season') == LAST_SEASON)
    )['week'].max()
)

train = (
    raw_data.filter(
        (pl.col('season') < LAST_SEASON) | ((pl.col('season') == LAST_SEASON) &
        (pl.col('week') < lw))
    )
)

test  = (
    raw_data
    .filter((pl.col('season') == LAST_SEASON) & (pl.col('week') == lw))
)

std_these = ['spread_line', 'roll_avg_points_allowed', 'lag_yards_per_rush_attempt', 'lag_target_share', 'lag_yards_per_target', 'lag_yards_per_pass_attempt', 'temp', 'wind']

binary_vars = ['is_grass', 'is_indoors', 'is_home', 'div_game', 'era']
idx_cols = ['player', 'position', 'team', 'week', 'tenure', 'player_season', 'team_season']

scalers = fit_scalers(df = train, cols = std_these)

idxs = make_indices(raw_data, idx_cols)

train = apply_scalers(train, scalers)
test  = apply_scalers(test,  scalers)  

train = encode(train, cols = idx_cols, enums = idxs)
test = encode(test,cols = idx_cols ,enums = idxs)

coords = {col: enum.categories.to_list() for col, enum in idxs.items()}


cont_dat_train = (
    train.select(
        cs.ends_with('_z')
    )
)

coords['cont_vars'] = cont_dat_train.columns

binary_train = (
    train.select(binary_vars)
)

coords['binary_vars'] = binary_train.columns



games_form, _ = pz.maxent(pz.InverseGamma(), lower = 5, upper = 12, mass = 0.9)

games_m, games_c = pm.gp.hsgp_approx.approx_hsgp_hyperparams(
    x_range = [
        train['week_idx'].min(), train['week_idx'].max()
    ], 
    lengthscale_range=[5,12], 
    cov_func='matern52'
)


unique_weeks = idxs['week'].categories.cast(pl.Float64).cast(pl.Int64).to_numpy()

with pm.Model(coords = coords) as ff_mlm:
    teams_data = pm.Data(
        'teams_data', 
        train['team_idx'].to_numpy().squeeze()
    )

    player_data = pm.Data(
        'player_data', 
        train['player_idx'].to_numpy().squeeze()
    )

    position_data = pm.Data(
        'position_data', 
        train['position_idx'].to_numpy().squeeze()
    )
    obs_data = pm.Data(
        'obs_data', 
        train['total_fantasy_points'].to_numpy()
    )

    cont_data = pm.Data(
        'cont_data', 
        cont_dat_train.to_numpy()
    )
    bi_data = pm.Data(
        'bi_data', 
        binary_train.to_numpy()
    )
    tenure_data = pm.Data(
        'tenure_data', 
        train['tenure_idx'].to_numpy().squeeze()
    )
    games_data = pm.Data(
        'games_data', 
        train['week_idx'].to_numpy().squeeze()
    )

    week_grid = pm.Data('week_grid', unique_weeks, dims = 'week')
    

    team_sigma = pm.HalfStudentT('team_sigma', nu =4 , sigma  = 3)
    #team_mu = pm.Normal('team_mu', 0, 1, dims = 'team')
    team_intercepts = pm.ZeroSumNormal('team_intercepts', team_sigma, dims = 'team')


    positions_sigma = pm.HalfCauchy('positions_sigma', 1.0)
    positions_mu = pm.Normal('positions_mu', 0, 1, dims = 'position')
    positions_intercepts = pm.Deterministic('position_intercepts', positions_mu * positions_sigma, dims = 'position')

    intercepts = pm.Deterministic(
        'intercepts', 
        #player_intercepts[player_data]
        team_intercepts[teams_data]
        + positions_intercepts[position_data]
    )

    cont_priors = pm.Normal('cont_priors', 0, 1, dims = 'cont_vars')
    bi_priors = pm.Normal('bi_priors', 0, 1.5, dims = 'binary_vars')
    


    init_sigma = pm.HalfNormal('init_sigma', 1.0)
    rw_sigma = pm.HalfNormal('rw_sigma', 0.5)
    z_innov = pm.Normal('z_innov', 0,1, dims = ('player', 'tenure'))

    level0 = init_sigma * z_innov[:, 0]
    steps = rw_sigma * z_innov[:, 1:]
    
    player_skill = pm.Deterministic(
        'player_skill',
    pt.concatenate([level0[:, None], level0[:, None] + pt.cumsum(steps, axis=1)], axis=1),
    dims=('player', 'tenure'))
    f_career= player_skill[player_data, tenure_data]

    games_sigma = pm.HalfNormal('games_sigma', 1)
    games_prior = pm.InverseGamma(
        'games_prior', 
        games_form.alpha, 
        games_form.beta
    )

    cov_game = games_sigma**2 * pm.gp.cov.Matern52(input_dim=1, ls = games_prior)
    gp_game = pm.gp.HSGP(
        m =  [games_m], c = games_c, cov_func=cov_game ,drop_first=True
    )
    game_gp = gp_game.prior(
        'game_gp', week_grid[:, None], dims = 'week'
    )

    f_game = game_gp[games_data]


    mu = (
        intercepts
        + pm.math.dot(cont_data, cont_priors)
        + pm.math.dot(bi_data, bi_priors)
        + f_career
        + f_game
    )

    sigma_obs = pm.HalfNormal('sigma_obs',4.0)
    #nu = pm.Gamma('nu', alpha = 2, beta = 0.1)

    pm.StudentT(
        'y_obs',   
        nu = 4, 
        mu = mu, 
        sigma = sigma_obs, 
        observed= obs_data
    )

with ff_mlm:
    idata = pm.sample_prior_predictive()



with ff_mlm:
    idata.update(
        pm.sample(seed = SEED)
    )

az.plot_ess_evolution(
    idata, 
    var_names=[rv.name for rv in ff_mlm.free_RVs if rv.size.eval() <= 10]
)


check_player = train['player'].unique().sample(5).to_list()
check_tenure = train['tenure'].unique().sample(2).cast(pl.Float64).cast(pl.String).to_list()





az.plot_ess_evolution(
    idata, 
    var_names=['player_skill'], 
    coords = {'player': check_player, 
                'tenure': ['1', '3', '4']}
)

az.plot_ess_evolution(
    idata, 
    var_names=['team_intercepts']
)
with ff_mlm:
    idata.update(
        pm.sample_posterior_predictive(idata)
    )
    pm.compute_log_likelihood(idata)


idata.to_netcdf('model-nc/ff_mlm.nc')

