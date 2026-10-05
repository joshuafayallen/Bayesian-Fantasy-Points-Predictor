
from plotnine.positions import position
from arviz_base import rcparams
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
)


raw_data.glimpse()

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
std_these = ['spread_line', 'roll_avg_points_allowed', 'lag_yards_per_rush_attempt', 'lag_target_share', 'lag_yards_per_target', 'lag_yards_per_pass_attempt', 'temp', 'wind', 'tenure']

binary_vars = ['is_grass', 'is_indoors', 'is_home', 'div_game', 'era']

idx_cols = ['player',  'player_season', 'position', 'opp_team', 'season']
idxs = make_indices(raw_data, idx_cols)
train = encode(train, cols = idx_cols, enums = idxs)
test = encode(test,cols = idx_cols ,enums = idxs)


scalers = fit_scalers(df = train, cols = std_these)
train = apply_scalers(train, scalers)
test  = apply_scalers(test,  scalers) 


x_train = (
    train.select(
        cs.ends_with('_z'), 
        cs.starts_with('team'), 
        pl.col(binary_vars)
    ).drop('team_season', 'team')
)

coords = {col: enum.categories.to_list() for col, enum in idxs.items()}

with pm.Model(coords = coords) as bart_mod: 

    player_data = pm.Data(
        'player_data',
        train['player_idx'].to_numpy().squeeze()
    )
    player_season_data = pm.Data(
        'player_season_data',
        train['player_season_idx'].to_numpy().squeeze()
    )

    x_data = pm.Data(
        'x_data', 
        x_train.to_numpy()
    )

    opp_team_data = pm.Data(
        'opp_team_data',
        train['opp_team_idx'].to_numpy().squeeze()

    )

    position_data = pm.Data(
        'position_data',
        train['position_idx'].to_numpy().squeeze()
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
    player_season_sigma = pm.HalfStudentT("player_season_sigma", nu=4, sigma=1.0)
    player_season_mu = pm.Normal("player_season_mu", 0, 1, dims="player_season")
    player_season_intercept = pm.Deterministic(
        "player_season_intercept",player_season_mu * player_season_sigma, dims="player_season"
    )

    

    mu_bart = pmb.BART(
        'mu_bart', X = x_data, Y = obs_data, m = 50, num_particles = 20, batch = (0.1, 1.0)
    )

    opp_rho = pm.Beta('opp_rho', 4, 2)
    opp_sigma = pm.HalfNormal('opp_sigma' , 1.0)


    opp_strength = pm.AR(
        'opp_ar', 
        rho = opp_rho,
        sigma = opp_sigma, 
        init_dist = pm.Normal.dist(0, 1.0), 
        constant= False, 
        ar_order= 1, 
        dims = ('opp_team', 'season')
    )
    f_opp = opp_strength[opp_team_data, season_data]

    mu = pm.Deterministic(
        'mu', 
        mu_bart +  player_intercepts[player_data]  + f_opp  + player_season_intercept[player_season_data])
        

    sigma_obs = pm.HalfNormal('sigma_obs', 5.0, dims = 'position')
    nu = pm.Gamma('nu', alpha = 2, beta= 0.1)
    

    pm.StudentT(
        'y_obs', 
        mu = mu, 
        sigma = sigma_obs[position_data], 
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
        pm.sample(random_seed = SEED, target_accept = 0.95)
    )



az.plot_ess_evolution(
    idata_bart, 
    var_names=[rv.name for rv in bart_mod.free_RVs if rv.size.eval() <= 5]
)




r_hat = az.rhat(
    idata_bart, 
    var_names=['player_sigma', 'player_season_sigma']
)
print(r_hat)


idata_bart.posterior["player_sigma"].mean(dim="draw").data

az.plot_trace(idata_bart, var_names=['player_sigma'])

with bart_mod:
    pm.compute_log_likelihood(idata_bart, compile_kwargs={'mode': 'NUMBA'})
    idata_bart.update(
        pm.sample_posterior_predictive(idata_bart, compile_kwargs={'mode': 'NUMBA'})
    )


az.plot_loo_pit(
    idata_bart, 
    var_names=['y_obs']
)

loow_bart = az.loo(idata_bart, pointwise=True)

fig, ax = plt.subplots()

az.plot_khat(loow_bart, )
