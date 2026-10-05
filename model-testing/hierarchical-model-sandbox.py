
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
    .fill_nan(0)
    .with_columns(
        pl.when(pl.col('season') >= 2018)
        .then(1)
        .otherwise(0)
        .alias('era'), 
        pl.concat_str(
            [
                pl.col('player_id'), 
                pl.lit('_'), 
                pl.col('season')
            ]
        ).alias("player_season"), 
        pl.concat_str(
            [pl.col('team'), 
            pl.lit('_'), 
            pl.col('season')]
        )
        .alias('team_season')
        
    )
    .rename({'full_name' : 'player'})
    .with_columns(
        pl.when(
            pl.col('team') == 'SL'
        )
        .then(pl.lit('LA'))
        .when(pl.col('team') == 'ARZ')
        .then(pl.lit('ARI'))
        .when(pl.col('team') == 'OAK')
        .then(pl.lit('LV'))
        .when(pl.col('team') == 'SD')
        .then(pl.lit('LAC'))
        .when(pl.col('team') == 'BLT' )
        .then(pl.lit('BAL'))
        .when(pl.col('team') == 'CLV')
        .then(pl.lit('CLE'))
        
        .when(pl.col('team') == 'HST')
        .then(pl.lit('HOU'))
        .otherwise(pl.col('team'))
        .alias('team_clean')
    )
    .drop('team')
    .rename({'team_clean': 'team'})
)

len(raw_data['team'].unique())



add_lags = (
    raw_data
    .with_columns(
        pl.col('yards_per_rush_attempt', 'target_share', 'yards_per_target', 'yards_per_pass_attempt').shift(1).name.prefix('lag_')
    )
    .fill_null(0)
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
    add_lags
    .filter(
        (pl.col('season') == LAST_SEASON)
    )['week'].max()
)

train = (
    add_lags.filter(
        (pl.col('season') < LAST_SEASON) | ((pl.col('season') == LAST_SEASON) &
        (pl.col('week') < lw))
    )
)

test  = (
    add_lags
    .filter((pl.col('season') == LAST_SEASON) & (pl.col('week') == lw))
)

std_these = ['spread_line', 'roll_avg_points_allowed', 'lag_yards_per_rush_attempt', 'lag_target_share', 'lag_yards_per_target', 'lag_yards_per_pass_attempt', 'temp', 'wind']

binary_vars = ['is_grass', 'is_indoors', 'is_home', 'div_game', 'era']
idx_cols = ['player', 'position', 'team', 'week', 'tenure', 'player_season', 'team_season']

scalers = fit_scalers(df = train, cols = std_these)

idxs = make_indices(add_lags, idx_cols)

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
with pm.Model(coords=coords) as gp_check:
    week_grid = pm.Data('tenure_grid', unique_weeks, dims = 'week')
    ell = pm.InverseGamma('ell', alpha=games_form.alpha, beta=games_form.beta)
    eta = pm.HalfNormal('eta', 1)  # or whatever amplitude prior you're using
    cov = eta**2 * pm.gp.cov.Matern52(1, ls=ell)
    gp = pm.gp.HSGP(m=[games_m], c=games_c, cov_func=cov, drop_first=True, parametrization='centered')
    f = gp.prior('f', X=week_grid[:, None], dims='week')
    prior = pm.sample_prior_predictive(draws=50, random_seed=0)

fig, ax = plt.subplots()
for i in range(50):
    plt.plot(unique_weeks, prior.prior['f'].values[0, i], alpha=0.3, color='C0')

age_m, age_c = pm.gp.hsgp_approx.approx_hsgp_hyperparams(
    x_range = [
        train['tenure_idx'].min(), train['tenure_idx'].max() 
    ], 
    lengthscale_range= [8,18], 
    cov_func= 'matern52'
)

unique_tenure = idxs['tenure'].categories.cast(pl.Int64).to_numpy()
age_curve, _  = pz.maxent(pz.InverseGamma(), 8, 18, mass = 0.9)
with pm.Model(coords=coords) as gp_check:
    tenure_grid = pm.Data('tenure_grid', unique_tenure, dims = 'tenure')
    ell = pm.InverseGamma('ell', alpha=age_curve.alpha, beta=age_curve.beta)
    eta = pm.HalfNormal('eta', 1)  # or whatever amplitude prior you're using
    cov = eta**2 * pm.gp.cov.Matern52(1, ls=ell)
    gp = pm.gp.HSGP(m=[age_m], c=age_c, cov_func=cov, drop_first=True, parametrization='centered')
    f = gp.prior('f', X=tenure_grid[:, None], dims='tenure')
    prior = pm.sample_prior_predictive(draws=50, random_seed=0)

fig,ax = plt.subplots()
for i in range(50):
    plt.plot(unique_tenure, prior.prior['f'].values[0, i], alpha=0.3, color='C0')

unique_games = train['week_idx'].unique().sort().to_list()
unique_age = train['tenure_idx'].unique().sort().to_list()


sns.kdeplot(
    data = train, x = 'total_fantasy_points', hue = 'position'
)

check = (
    train.filter(
        pl.col('total_fantasy_points') < 0
    )
)


with pm.Model(coords = coords) as ff_intercepts_only: 

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

    positions_sigma = pm.HalfStudentT('positions_sigma', nu = 4, sigma = 1)
    positions_mu = pm.Normal('positions_mu', 0, 1)

    positions_intercepts = pm.Normal('positions_intercepts', positions_mu, positions_sigma, dims = 'position')

    player_sigma = pm.HalfStudentT('player_sigma', nu = 4, sigma =  1) 
    player_mu = pm.Normal('player_mu', 0, 0.5)

    player_intercepts = pm.Normal('player_intercepts', player_mu, player_sigma, dims = 'player')

    team_sigma = pm.HalfCauchy('team_sigma', 1)
    team_mu = pm.Normal('team_mu', 0, 0.5) 

    team_intercepts = pm.Normal('team_intercepts', team_mu, team_sigma, dims = 'team')

    intercepts = (
        positions_intercepts[position_data]
        + player_intercepts[player_data]
        + team_intercepts[teams_data]
    )

    skew_a = pm.Gamma('skew_a', 3, 0.4)
    skew_b = pm.Gamma('skew_b' , 3, 0.4)
    sigma_obs = pm.HalfNormal('sigma_obs', 1)

    pm.SkewStudentT(
        'y_obs', 
        a = skew_a,
        b = skew_b,
        mu = intercepts, 
        sigma = sigma_obs, 
        observed= obs_data
    )



with ff_intercepts_only:
    idata_int = pm.sample_prior_predictive()

az.plot_ppc_dist(
    idata_int, 
    group = 'prior_predictive', 
    kind = 'ecdf'
)


def check_prior(idata, var="y_obs"):
    y = idata["prior_predictive"][var].values
    return {"min": float(y.min()), "max": float(y.max()),
            "mean": float(y.mean()), "std": float(y.std())}

check_prior(idata_int)

with ff_intercepts_only: 
    idata_int.update(
        pm.sample(random_seed=SEED)
    )

check = (train.unique('player').sample(4)['player'].to_list())

az.plot_pair(
    idata_int, 
    var_names=['player_sigma', 'player_intercepts'], 
    coords = {'player': check}, 
    visuals= {'divergence': True}
)

with pm.Model(coords = coords) as add_cont_vars:
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

    positions_sigma = pm.HalfStudentT('positions_sigma', nu = 4, sigma = 1)
    positions_mu = pm.Normal('positions_mu', 0, 1)

    positions_intercepts = pm.Normal('positions_intercepts', positions_mu, positions_sigma, dims = 'position')

    player_sigma = pm.HalfStudentT('player_sigma', nu = 4, sigma =  1) 
    player_mu = pm.Normal('player_mu', 0, 1)

    player_intercepts = pm.Normal('player_intercepts', player_mu, player_sigma, dims = 'player')

    team_sigma = pm.HalfCauchy('team_sigma', 1)
    team_mu = pm.Normal('team_mu', 0, 0.5) 

    team_intercepts = pm.Normal('team_intercepts', team_mu, team_sigma, dims = 'team')
    intercepts = pm.Deterministic(
        'intercepts', 
        positions_intercepts[position_data]
        + player_intercepts[player_data]
        + team_intercepts[teams_data]
    )

    cont_priors = pm.Normal('cont_priors', 0, 1, dims = 'cont_vars')

    mu = pm.Deterministic(
        'mu', 
        intercepts
        + pm.math.dot(cont_data, cont_priors)
    )

    skew_a = pm.Gamma('skew_a', 3, 0.4)
    skew_b = pm.Gamma('skew_b' , 3, 0.4)
    sigma_obs = pm.HalfNormal('sigma_obs', 1)

    pm.SkewStudentT(
        'y_obs', 
        a = skew_a,
        b = skew_b,
        mu = mu, 
        sigma = sigma_obs, 
        observed= obs_data
    )


with add_cont_vars:
    idata_cov1 = pm.sample_prior_predictive()


check_prior(idata_cov1)

az.plot_ppc_dist(
    idata_cov1, 
    kind = 'ecdf',
    group = 'prior_predictive',
    visuals = {'observed_dist': True}
)


with add_cont_vars: 
    idata_cov1.update(
        pm.sample(random_seed=SEED)
    )


idata_cov1.sample_stats['diverging'].sum().item()

az.plot_trace_dist(
    idata_cov1
)


with pm.Model(coords = coords) as add_bi_vars:
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


    player_sigma = pm.HalfStudentT('player_sigma', nu = 4, sigma =  2.5) 
    player_mu = pm.Normal('player_mu', 0 , 1.0, dims = 'player')

    player_intercepts = pm.Deterministic(
        'player_intercepts', 
        player_sigma * player_mu, 
        dims = 'player'
    )

    team_sigma = pm.HalfStudentT('team_sigma', nu = 4, sigma  = 2.5)
    team_intercepts = pm.ZeroSumNormal('team_intercepts', team_sigma, dims = 'team')

    intercepts = pm.Deterministic(
        'intercepts', 
        player_intercepts[player_data]
        + team_intercepts[teams_data]
    )

    cont_priors = pm.Normal('cont_priors', 0, 1, dims = 'cont_vars')
    bi_priors = pm.Normal('bi_priors', 0, 1.5, dims = 'binary_vars')

    mu = pm.Deterministic(
        'mu', 
        intercepts
        + pm.math.dot(cont_data, cont_priors)
        + pm.math.dot(bi_data, bi_priors)
    )

    #FLOOR = 1.05
    #s = pm.Deterministic('skew_sum', 2*FLOOR + pm.Gamma('skew_sum_raw', mu=12, sigma=3, dims='position'))
    #f = pm.Beta('skew_frac', 2, 2, dims='position')
    #half_range = s/2 - FLOOR
    #skew_a = pm.Deterministic('skew_a', s/2 + (2*f-1)*half_range, dims='position')
    #skew_b = pm.Deterministic('skew_b', s/2 - (2*f-1)*half_range, dims='position')
    skew_a= pm.Gamma('skew_a', mu = 10, sigma = 3, dims = 'position')
    skew_b = pm.Gamma('skew_b',  mu = 10, sigma = 3, dims = 'position')
    sigma_obs = pm.HalfNormal('sigma_obs', 3.0)

    pm.SkewStudentT(
        'y_obs', 
        a = skew_a[position_data], 
        b = skew_b[position_data], 
        nu = 4, 
        mu = mu, 
        sigma = sigma_obs, 
        observed= obs_data
    )

with add_bi_vars: 
    idata_cov2 = pm.sample_prior_predictive()


az.plot_ppc_dist(
    idata_cov2, 
    kind = 'ecdf',
    group = 'prior_predictive',
    visuals = {'observed_dist': True}
)

check_prior(idata_cov2)


with add_bi_vars: 
    idata_cov2.update(
        pm.sample(random_seed=SEED)
    )


az.plot_ess_evolution(
    idata_cov2, 
    var_names=[rv.name for rv in add_bi_vars.free_RVs if rv.size.eval() <= 4]
)


check = pl.from_pandas(az.summary(idata_cov2, kind = 'diagnostics', var_names=['player_intercepts', 'team_intercepts']).reset_index())


idata_cov2.sample_stats['diverging'].sum().item()

az.plot_energy(idata_cov2)

test_players=  train['player'].unique().sample(5).to_list()

az.plot_pair(
    idata_cov2,
    var_names=['team_intercepts'],
    coords = {'team': ['SF', 'ATL', 'SEA']}
)

with add_bi_vars:
    idata_cov2.update(
        pm.sample_posterior_predictive(idata_cov2)
    )

az.plot_ppc_dist(idata_cov2, kind = 'ecdf')


Y_VAR     = "y_obs"                    # name of your likelihood RV
OBS_COL   = "total_fantasy_points"     # observed target column in `train`
POS_COL   = "position"                 # position label column in `train`
POSITIONS = ["QB", "RB", "WR", "TE"]
 
 
def get_pp_array(ppc, y_var=Y_VAR):
    """Return posterior-predictive as (n_obs, n_samples), aligned to data row order."""
    pp = ppc.posterior_predictive[y_var]                 # dims: chain, draw, <obs>
    obs_dim = [d for d in pp.dims if d not in ("chain", "draw")][0]
    return pp.stack(s=("chain", "draw")).transpose(obs_dim, "s").values
 
 
def ppc_by_position(model, idata, train):
    # 1. sample posterior predictive IN-SAMPLE (on the training obs)
    with model:
        ppc = pm.sample_posterior_predictive(idata, var_names=[Y_VAR],
                                             progressbar=False)
    pp_arr = get_pp_array(ppc)                            # (n_obs, n_samples)
 
    observed  = train[OBS_COL].to_numpy()                # row-aligned to obs axis
    positions = train[POS_COL].to_numpy()
    assert pp_arr.shape[0] == len(train), "row misalignment: PP obs != train rows"
 
    # 2. per-position skew / mean comparison
    print(f"{'pos':4s}{'obs_mean':>9s}{'pp_mean':>9s}{'obs_skew':>9s}{'pp_skew':>9s}")
    for p in POSITIONS:
        m = positions == p
        obs_p, pp_p = observed[m], pp_arr[m].reshape(-1)
        print(f"{p:4s}{obs_p.mean():>9.1f}{pp_p.mean():>9.1f}"
              f"{skew(obs_p):>9.2f}{skew(np.clip(pp_p, -20, 80)):>9.2f}")
 
    # 3. density overlay (judge fit on the DENSITY, never a CDF)
    grid = np.linspace(-10, 50, 400)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, p in zip(axes.ravel(), POSITIONS):
        m = positions == p
        ax.fill_between(grid, gaussian_kde(observed[m])(grid), color="0.82", label="observed")
        ax.plot(grid, gaussian_kde(np.clip(pp_arr[m].reshape(-1), -20, 80))(grid),
                "C0", lw=1.8, label="posterior predictive")
        ax.set_title(p); ax.set_xlabel(OBS_COL); ax.set_yticks([])
        ax.set_xlim(-10, 50); ax.legend(fontsize=8)
    fig.suptitle("Per-position posterior predictive vs observed (training set)")
    fig.tight_layout()


ppc_by_position(add_bi_vars, idata_cov2, train)


with pm.Model(coords = coords) as add_time_vars:
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
    player_seasons_data = pm.Data(
        'seasons_data', 
        train['player_season_idx'].to_numpy().squeeze()
    )
    


    player_sigma = pm.HalfStudentT('player_sigma', nu = 4, sigma =  2.5) 
    player_mu = pm.Normal('player_mu', 0 , 1.0, dims = 'player')

    player_intercepts = pm.Deterministic(
        'player_intercepts', 
        player_sigma * player_mu, 
        dims = 'player'
    )

    #team_sigma = pm.HalfStudentT('team_sigma', nu = 4, sigma  = 2.5)
    #team_intercepts = pm.ZeroSumNormal('team_intercepts', team_sigma, dims = 'team')

    player_season_sigma = pm.HalfStudentT('player_season_sigma', nu = 4, sigma = 2.5)
    player_season_mu = pm.Normal('player_season_mu', 0, 1, dims = 'player_season')

    player_season_intercept = pm.Deterministic(
        'player_season_intercept', 
        player_season_sigma * player_season_mu, 
        dims = 'player_season'
    )

    intercepts = pm.Deterministic(
        'intercepts', 
        player_intercepts[player_data]
        # + team_intercepts[teams_data]
        + player_season_intercept[player_seasons_data]
    )

    cont_priors = pm.Normal('cont_priors', 0, 1, dims = 'cont_vars')
    bi_priors = pm.Normal('bi_priors', 0, 1.5, dims = 'binary_vars')



    mu = pm.Deterministic(
        'mu', 
        intercepts
        + pm.math.dot(cont_data, cont_priors)
        + pm.math.dot(bi_data, bi_priors)
    )

    FLOOR = 1.05
    s = pm.Deterministic('skew_sum', 2*FLOOR + pm.Gamma('skew_sum_raw', mu=12, sigma=3, dims='position'))
    f = pm.Beta('skew_frac', 2, 2, dims='position')
    half_range = s/2 - FLOOR
    skew_a = pm.Deterministic('skew_a', s/2 + (2*f-1)*half_range, dims='position')
    skew_b = pm.Deterministic('skew_b', s/2 - (2*f-1)*half_range, dims='position')
    #skew_a= pm.Gamma('skew_a', mu = 10, sigma = 3, dims = 'position')
    #skew_b = pm.Gamma('skew_b',  mu = 10, sigma = 3, dims = 'position')
    sigma_obs = pm.HalfNormal('sigma_obs', 5.0)

    pm.SkewStudentT(
        'y_obs', 
        a = skew_a[position_data], 
        b = skew_b[position_data], 
        nu = 4, 
        mu = mu, 
        sigma = sigma_obs, 
        observed= obs_data
    )

with add_time_vars:
    idata_time = pm.sample_prior_predictive()

az.plot_ppc_dist(
    idata_time, 
    group = 'prior_predictive', 
    visuals={'observed_dist': True}
)


with add_time_vars:
    idata_time.update(
        pm.sample(random_seed=SEED)
    )


az.summary(
    idata_time, var_names=['sigma', 'skew_a', 'skew_b'], filter_vars='like'
)



check = (
    pl.from_pandas(
        az.summary(idata_time, kind = 'diagnostics', var_names=['player_intercepts', 'player_season_intercept'])
        .reset_index())
)

diag = (
    check
    .group_by('index')
    .agg(
        pl.col('ess_bulk', 'ess_tail').min(), 
        pl.col("r_hat").cast(pl.Float64).max()
    )
    .filter(
        (pl.col('ess_bulk') <600) | (pl.col("ess_tail") < 600) | (pl.col('r_hat') >= 1.01)
    )
)

with add_time_vars: 
    ppc = pm.sample_posterior_predictive(idata_time)


observed  = train[OBS_COL].to_numpy()    # OBS_COL = "total_fantasy_points"
positions = train[POS_COL].to_numpy()    # POS_COL = "position"
pp = ppc.posterior_predictive["y_obs"]
obs_dim = [d for d in pp.dims if d not in ("chain", "draw")][0]
pp_arr = pp.stack(s=("chain", "draw")).transpose(obs_dim, "s").values 

for p in ["QB", "RB", "WR", "TE"]:
    m = positions == p
    print(p, "obs_sd:", round(float(observed[m].std()), 2),
            "pp_sd:",  round(float(pp_arr[m].reshape(-1).std()), 2))

resid = observed - pp_arr.mean(axis=1)    


resid.mean(), resid.max(), resid.min(), resid.std()

print('mean:', resid.mean(), 'max:', resid.max(), 'std:', resid.std() )
df = (train.select(["player_season", "week"])       # or unit / whatever your key is
        .with_columns(pl.Series("resid", resid))
        .sort(["player_season", "week"])
        .with_columns(pl.col("resid").shift(1).over("player_season").alias("resid_lag"))
        .drop_nulls("resid_lag"))

r = np.corrcoef(df["resid"].to_numpy(), df["resid_lag"].to_numpy())[0, 1]
print("within-player-season lag-1 residual autocorrelation:", round(float(r), 3))


def convergence_report(idata, scalar_vars=None, ess_target=400, rhat_target=1.01,
                       exclude_dims=("obs",), n_worst=6):
    """Print a convergence report. Returns a dict of the headline numbers.
 
    scalar_vars  : names to show in the detailed table (e.g. your sigmas/betas)
    exclude_dims : dims whose variables are skipped (obs-length Deterministics)
    """
    post = idata.posterior
    ss = idata.sample_stats if "sample_stats" in idata else None
 
    # --- 1. sampler-level pathologies -------------------------------------
    print("=" * 62); print("SAMPLER DIAGNOSTICS"); print("=" * 62)
    if ss is not None and "diverging" in ss:
        div, n = int(ss["diverging"].sum()), int(ss["diverging"].size)
        flag = "   <-- INVESTIGATE" if div else ""
        print(f"  divergences      : {div} / {n}  ({100*div/n:.2f}%){flag}")
    if ss is not None and "tree_depth" in ss:
        td = ss["tree_depth"].values
        mx = int(td.max()); sat = float((td >= mx).mean())
        flag = "   <-- SATURATING" if sat > 0.05 else ""
        print(f"  max tree_depth   : {mx}  (hit in {100*sat:.1f}% of draws){flag}")
    if ss is not None and "energy" in ss:
        try:
            # BFMI from the energy trace directly (version-proof):
            #   BFMI = sum((E_t - E_{t-1})^2) / ((N-1) * var(E))
            E = np.asarray(ss["energy"].values, dtype=float)
            E = E if E.ndim == 2 else E[None, :]
            d = np.diff(E, axis=1)
            b = (d ** 2).sum(axis=1) / ((E.shape[1] - 1) * E.var(axis=1, ddof=1))
            flag = "   <-- LOW" if np.nanmin(b) < 0.3 else ""
            print(f"  BFMI per chain   : {np.round(b, 2)}  (<0.3 => poor exploration){flag}")
        except Exception as e:
            print(f"  BFMI             : unavailable ({type(e).__name__})")
    if ss is not None and "acceptance_rate" in ss:
        print(f"  mean accept rate : {float(ss['acceptance_rate'].mean()):.3f}")
 
    # --- 2. aggregate r_hat / ESS over every parameter ---------------------
    keep = [v for v in post.data_vars
            if not any(d in exclude_dims for d in post[v].dims)]
    rh = az.rhat(idata, var_names=keep)
    eb = az.ess(idata, var_names=keep, method="bulk")
    et = az.ess(idata, var_names=keep, method="tail")
 
    def flat(ds):
        return {v: np.atleast_1d(ds[v].values.reshape(-1).astype(float))
                for v in ds.data_vars}
    RH, EB, ET = flat(rh), flat(eb), flat(et)
    a_rh = np.concatenate(list(RH.values()))
    a_eb = np.concatenate(list(EB.values()))
    a_et = np.concatenate(list(ET.values()))
 
    print("\n" + "=" * 62); print("CONVERGENCE ACROSS ALL PARAMETERS"); print("=" * 62)
    print(f"  parameters checked : {a_rh.size}")
    print(f"  max r_hat          : {np.nanmax(a_rh):.3f}"
          f"   (# > {rhat_target}: {int(np.nansum(a_rh > rhat_target))})")
    print(f"  min ess_bulk       : {np.nanmin(a_eb):.0f}"
          f"   (# < {ess_target}: {int(np.nansum(a_eb < ess_target))})")
    print(f"  min ess_tail       : {np.nanmin(a_et):.0f}"
          f"   (# < {ess_target}: {int(np.nansum(a_et < ess_target))})")
 
    print(f"\n  worst variables (by max r_hat):")
    for v in sorted(RH, key=lambda v: -np.nanmax(RH[v]))[:n_worst]:
        print(f"    {v:26s} r_hat={np.nanmax(RH[v]):.3f}  "
              f"ess_bulk_min={np.nanmin(EB[v]):.0f}")
 
    # --- 3. detailed table for the scalars you care about ------------------
    if scalar_vars:
        present = [v for v in scalar_vars if v in post.data_vars]
        if present:
            print("\n" + "=" * 62); print("KEY PARAMETERS"); print("=" * 62)
            print(az.summary(idata, var_names=present).to_string())
 
    return dict(divergences=(int(ss["diverging"].sum())
                             if ss is not None and "diverging" in ss else None),
                max_rhat=float(np.nanmax(a_rh)),
                min_ess_bulk=float(np.nanmin(a_eb)),
                min_ess_tail=float(np.nanmin(a_et)))


res = convergence_report(
    idata_time,
    scalar_vars=["player_sigma", "player_season_sigma", "sigma_obs",
                    "skew_sum_raw", "skew_frac", "skew_a", "skew_b",
                    "cont_priors", "bi_priors"]
)

def rmse(pred, actual):
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(actual)) ** 2)))
 
 
def mae(pred, actual):
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(actual))))
 
 
def rmse_report(train, pp_arr, obs_col="total_fantasy_points",
                pos_col="position", unit_col="player_season",
                positions=("QB", "RB", "WR", "TE")):
    """train: the frame you fit on (row order matching pp_arr's obs axis)
       pp_arr: (n_obs, n_samples) posterior predictive from ppc_check.get_pp_array"""
    y = train[obs_col].to_numpy()
    pred = pp_arr.mean(axis=1)                      # posterior predictive mean
    pos = train[pos_col].to_numpy()
 
    # --- baselines ------------------------------------------------------
    b_global = np.full(len(train), y.mean())
    b_pos    = train.select(pl.col(obs_col).mean().over(pos_col)).to_numpy().ravel()
    b_unit   = train.select(pl.col(obs_col).mean().over(unit_col)).to_numpy().ravel()
 
    print(f"{'predictor':26s}{'RMSE':>8s}{'MAE':>8s}")
    print(f"{'global mean':26s}{rmse(b_global,y):>8.2f}{mae(b_global,y):>8.2f}")
    print(f"{'position mean':26s}{rmse(b_pos,y):>8.2f}{mae(b_pos,y):>8.2f}")
    print(f"{'player-season mean':26s}{rmse(b_unit,y):>8.2f}{mae(b_unit,y):>8.2f}")
    print(f"{'MODEL':26s}{rmse(pred,y):>8.2f}{mae(pred,y):>8.2f}")
    gain = 100 * (1 - rmse(pred, y) / rmse(b_unit, y))
    print(f"\n  model vs player-season-mean baseline: {gain:+.1f}% RMSE")
 
    # --- per position ---------------------------------------------------
    print(f"\n{'pos':5s}{'n':>7s}{'obs_sd':>8s}{'base_RMSE':>11s}{'model_RMSE':>12s}{'gain':>8s}")
    for p in positions:
        m = pos == p
        br, mr = rmse(b_unit[m], y[m]), rmse(pred[m], y[m])
        print(f"{p:5s}{m.sum():>7d}{y[m].std():>8.2f}{br:>11.2f}{mr:>12.2f}"
              f"{100*(1-mr/br):>7.1f}%")
 
    return dict(model_rmse=rmse(pred, y), model_mae=mae(pred, y),
                baseline_rmse=rmse(b_unit, y))


rmse_report(train, pp_arr)

def build_model(D, coords, observed=None):
    """Your add_time_vars model, parameterized for the backtest harness.

    Called twice per fold: observed=D['y'] to fit, observed=None to predict.
    Covariate widths come from coords['cont_vars'] / coords['binary_vars'],
    so adding age/tenure to the ModelSpec flows through with no edit here.
    """
    with pm.Model(coords=coords) as m:
        player_data         = pm.Data('player_data',   D['player'])
        player_seasons_data = pm.Data('seasons_data',  D['player_season'])
        position_data       = pm.Data('position_data', D['position'])

        # --- player: career latent ability (non-centred) ---
        player_sigma = pm.HalfStudentT('player_sigma', nu=4, sigma=2.5)
        player_mu    = pm.Normal('player_mu', 0, 1.0, dims='player')
        player_intercepts = pm.Deterministic(
            'player_intercepts', player_sigma * player_mu, dims='player')

        # --- player-season: that year's realisation (non-centred) ---
        player_season_sigma = pm.HalfStudentT('player_season_sigma', nu=4, sigma=2.5)
        player_season_mu    = pm.Normal('player_season_mu', 0, 1, dims='player_season')
        player_season_intercept = pm.Deterministic(
            'player_season_intercept',
            player_season_sigma * player_season_mu, dims='player_season')

        intercepts = pm.Deterministic(
            'intercepts',
            player_intercepts[player_data]
            + player_season_intercept[player_seasons_data])

        # --- covariates: guarded so an empty spec doesn't break ---
        mu_ = intercepts
        if len(coords.get('cont_vars', [])):
            cont_data   = pm.Data('cont_data', D['cont'])
            cont_priors = pm.Normal('cont_priors', 0, 1, dims='cont_vars')
            mu_ = mu_ + pm.math.dot(cont_data, cont_priors)
        if len(coords.get('binary_vars', [])):
            bi_data   = pm.Data('bi_data', D['bi'])
            bi_priors = pm.Normal('bi_priors', 0, 1.5, dims='binary_vars')
            mu_ = mu_ + pm.math.dot(bi_data, bi_priors)
        mu = pm.Deterministic('mu', mu_)

        # --- position-indexed skew, bounded so both shapes stay > FLOOR ---
        FLOOR = 1.05
        s = pm.Deterministic(
            'skew_sum',
            2*FLOOR + pm.Gamma('skew_sum_raw', mu=12, sigma=3, dims='position'))
        f = pm.Beta('skew_frac', 2, 2, dims='position')
        half_range = s/2 - FLOOR
        skew_a = pm.Deterministic('skew_a', s/2 + (2*f-1)*half_range, dims='position')
        skew_b = pm.Deterministic('skew_b', s/2 - (2*f-1)*half_range, dims='position')

        sigma_obs = pm.HalfNormal('sigma_obs', 5.0)
        pm.SkewStudentT(
            'y_obs',
            a=skew_a[position_data],
            b=skew_b[position_data],
            nu=4,
            mu=mu,
            sigma=sigma_obs,
            observed=observed)      # <-- array to fit, None to predict
    return m

@dataclass
class ModelSpec:
    name: str
    cont_cols: Sequence[str]          # standardized (train-only)
    bi_cols: Sequence[str]            # left as 0/1
    build: Callable                   # (D, coords, observed) -> pm.Model
    likelihood: str = "skew_student_t"
    advi_n: int = 15000
    draws: int = 600


# adding age/tenure is now a one-line change:
BASE = ModelSpec("base",
                 cont_cols=std_these,
                 bi_cols=binary_vars,
                 build=build_model)

IDX_COLS = ["player", "player_season", "position"]



def _arrays(frame, spec, enums, scal):
    idx = lambda c: frame[c].cast(enums[c]).to_physical().cast(pl.Int32).to_numpy()
    cont = (np.column_stack([((frame[c] - scal[c][0]) / scal[c][1]).to_numpy()
                             for c in spec.cont_cols])
            if spec.cont_cols else np.zeros((len(frame), 0)))
    bi = (frame.select(spec.bi_cols).to_numpy().astype(float)
          if spec.bi_cols else np.zeros((len(frame), 0)))
    return dict(player=idx("player"), player_season=idx("player_season"),
                position=idx("position"), cont=cont, bi=bi,
                y=frame["total_fantasy_points"].to_numpy())

def _fit_encoders(train, spec):
    labels = {c: train[c].unique().sort().to_list() for c in IDX_COLS}   # keep these
    enums  = {c: pl.Enum(labels[c]) for c in IDX_COLS}
    scal   = {c: (train[c].mean(), train[c].std() or 1.0) for c in spec.cont_cols}
    coords = {**labels,                                   # lists, not Series
              "cont_vars": list(spec.cont_cols),
              "binary_vars": list(spec.bi_cols)}
    return enums, scal, coords, labels

def fit_predict(spec, train, test, seed=0):
    """Fit on train, return (kept_test_rows, S[draws, n]). Cold starts dropped."""
    enums, scal, coords, labels = _fit_encoders(train, spec)
    Dtr = _arrays(train, spec, enums, scal)
    with spec.build(Dtr, coords, observed=Dtr["y"]):
        idata = pm.fit(n=spec.advi_n, method="advi", random_seed=seed,
                       progressbar=False).sample(spec.draws)
    
    probe = test.with_columns([
        pl.col(c).cast(enums[c], strict = False).alias(f"_{c}_chk") for c in IDX_COLS])
    keep = probe.drop_nulls([f"_{c}_chk" for c in IDX_COLS]).drop(
        [f"_{c}_chk" for c in IDX_COLS])
    n_dropped = len(test) - len(keep)
    if len(keep) == 0:
        return None
    Dte = _arrays(keep, spec, enums, scal)
    with spec.build(Dte, coords, observed=None):
        ppc = pm.sample_posterior_predictive(idata, var_names=["y_obs"],
                                             random_seed=seed + 1, progressbar=False)
    S = ppc.posterior_predictive["y_obs"].values.reshape(-1, len(Dte["y"]))
    return keep, S


# ============================================================
# 3. SCORING — model-agnostic; takes (y, S) and never changes
# ============================================================
def crps(S, y, max_pairs=200, seed=0):
    """E|X-y| - 0.5 E|X-X'|. Lower better. Rewards accuracy AND calibration."""
    t1 = np.abs(S - y[None, :]).mean(0)
    i = np.random.default_rng(seed).choice(S.shape[0], size=(2, min(max_pairs, S.shape[0])))
    t2 = np.abs(S[i[0]] - S[i[1]]).mean(0)
    return float((t1 - 0.5 * t2).mean())

def score(y, S, positions=None, baseline=None, levels=(0.5, 0.8, 0.9)):
    """All metrics for one fold. Pure function of predictions vs actuals."""
    pred = S.mean(0)
    out = dict(n=len(y),
               rmse=float(np.sqrt(((pred - y) ** 2).mean())),
               mae=float(np.abs(pred - y).mean()),
               crps=crps(S, y),
               spearman=float(spearmanr(pred, y).statistic))
    for L in levels:
        lo, hi = np.percentile(S, [100*(1-L)/2, 100*(1+L)/2], axis=0)
        out[f"cov{int(L*100)}"] = float(((y >= lo) & (y <= hi)).mean())
    if baseline is not None:
        out["rmse_base"] = float(np.sqrt(((baseline - y) ** 2).mean()))
        out["spearman_base"] = float(spearmanr(baseline, y).statistic)
    return out

def past_only_baseline(keep, train, key="player_season"):
    b = train.group_by(key).agg(pl.col("total_fantasy_points").mean().alias("_pm"))
    return (keep.join(b, on=key, how="left")["_pm"]
                .fill_null(float(train["total_fantasy_points"].mean())).to_numpy())


# ============================================================
# 4. BACKTEST DRIVER — walks folds, scores any spec
# ============================================================
def backtest(data, folds, spec, levels=(0.5, 0.8, 0.9), verbose=True):
    rows = []
    for (S_, W_) in folds:
        train = data.filter((pl.col("season") < S_) |
                            ((pl.col("season") == S_) & (pl.col("week") < W_)))
        test = data.filter((pl.col("season") == S_) & (pl.col("week") == W_))
        if len(test) == 0 or len(train) < 200:
            continue
        r = fit_predict(spec, train, test)
        if r is None:
            continue
        keep, S = r
        y = keep["total_fantasy_points"].to_numpy()
        m = score(y, S, baseline=past_only_baseline(keep, train), levels=levels)
        m.update(season=S_, week=W_, spec=spec.name)
        rows.append(m)
        if verbose:
            print(f"  {spec.name} {S_} wk{W_:>2}: n={m['n']:>3} "
                  f"RMSE={m['rmse']:.2f} (base {m['rmse_base']:.2f}) "
                  f"CRPS={m['crps']:.2f} rho={m['spearman']:.2f}")
    return pl.DataFrame(rows)


def summarize(df, levels=(0.5, 0.8, 0.9)):
    w = df["n"].to_numpy()
    wm = lambda c: float(np.average(df[c].to_numpy(), weights=w))
    name = df["spec"][0]
    print(f"\n=== {name}  ({len(df)} folds, n={int(w.sum())}) ===")
    print(f"  RMSE      {wm('rmse'):.3f}   baseline {wm('rmse_base'):.3f}")
    print(f"  CRPS      {wm('crps'):.3f}")
    print(f"  Spearman  {wm('spearman'):.3f}   baseline {wm('spearman_base'):.3f}")
    for L in levels:
        print(f"  cov{int(L*100)}     {wm(f'cov{int(L*100)}'):.1%}  (target {L:.0%})")
    return {c: wm(c) for c in ["rmse", "crps", "spearman"]}


def compare(data, folds, specs):
    """Score several specs on the SAME folds — this is how you decide whether
    age/tenure earn their place."""
    results = {}
    for sp in specs:
        df = backtest(data, folds, sp, verbose=False)
        results[sp.name] = summarize(df)
    print("\n=== HEAD-TO-HEAD ===")
    base = list(results)[0]
    for k, v in results.items():
        d_crps = 100 * (1 - v["crps"] / results[base]["crps"])
        print(f"  {k:12s} CRPS={v['crps']:.3f} ({d_crps:+.1f}% vs {base})  "
              f"RMSE={v['rmse']:.3f}  rho={v['spearman']:.3f}")
    return results


S_, W_ = 2023, 15
train = add_lags.filter((pl.col("season") < S_) | ((pl.col("season") == S_) & (pl.col("week") < W_)))
test  = add_lags.filter((pl.col("season") == S_) & (pl.col("week") == W_))

r = fit_predict(BASE, train, test)
keep, S = r
y = keep["total_fantasy_points"].to_numpy()
print("dropped cold starts:", len(test) - len(keep), "of", len(test))
print(score(y, S, baseline=past_only_baseline(keep, train)))

folds = [(2023, w) for w in range(6, 19)]     # skip early weeks: thin training data
df = backtest(train, folds, BASE)
summarize(df)
unique_tenure = idxs['tenure'].categories.cast(pl.Int64).to_numpy()
unique_weeks = idxs['week'].categories.cast(pl.Float64).cast(pl.Int64).to_numpy()

with pm.Model(coords = coords) as hsgp_version:
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

    tenure_grid = pm.Data('tenure_grid', unique_tenure, dims = 'tenure')
    week_grid = pm.Data('week_grid', unique_weeks, dims = 'week')
    

    player_sigma = pm.HalfStudentT('player_sigma', nu = 4, sigma =  2.5) 
    player_mu = pm.Normal('player_mu', 0 , 1.0, dims = 'player')

    player_intercepts = pm.Deterministic(
        'player_intercepts', 
        player_sigma * player_mu, 
        dims = 'player'
    )

    team_sigma = pm.HalfStudentT('team_sigma', nu = 4, sigma  = 2.5)
    team_mu = pm.Normal('team_mu', 0 ,1, dims = 'team')
    team_intercepts = pm.Deterministic('team_intercepts',
                    team_sigma * team_mu, dims = 'team')

    positions_sigma = pm.HalfCauchy('positions_sigma', 1.0)
    
    positions_intercepts = pm.ZeroSumNormal(
        'positions_intercepts', 
        positions_sigma, 
        dims = 'position'
    )

    intercepts = pm.Deterministic(
        'intercepts', 
        player_intercepts[player_data]
        + team_intercepts[teams_data]
        + positions_intercepts[position_data]
    )

    cont_priors = pm.Normal('cont_priors', 0, 1, dims = 'cont_vars')
    bi_priors = pm.Normal('bi_priors', 0, 1.5, dims = 'binary_vars')
    


    tenure_sigma = pm.HalfNormal('tenure_sigma', 1) # effecetively two points over a rookie contract is how i am thinking of this prior 
    tenure_prior = pm.InverseGamma(
        'tenure_prior', 
        alpha = age_curve.alpha, 
        beta = age_curve.beta
    )

    cov_tenure = tenure_sigma**2 * pm.gp.cov.Matern52(input_dim=1, ls = tenure_prior)
    gp_tenure = pm.gp.HSGP(m = [age_m],  c = age_c, cov_func=cov_tenure, parametrization='centered', drop_first=True)

    tenure_gp = gp_tenure.prior(
        'tenure_gp', X = tenure_grid[:, None], dims = 'tenure'
    )
    f_tenure = tenure_gp[tenure_data]

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


    mu = pm.Deterministic(
        'mu', 
        intercepts
        + pm.math.dot(cont_data, cont_priors)
        + pm.math.dot(bi_data, bi_priors)
        + f_tenure
        + f_game
    )

    #FLOOR = 2.5   # comfortably above the variance-existence threshold of 1
    # = pm.Deterministic('shape_sum', 2*FLOOR + pm.Gamma('shape_sum_raw', #u=15, sigma=5, ), dims = 'position')
    # = pm.Beta('shape_split', 2, 2,)
    #alf_range = s/2 - FLOOR
    #skew_a = pm.Deterministic('skew_a', s/2 + (2*f-1)*half_range, dims = 'position')
    #skew_b = pm.Deterministic('skew_b', s/2 - (2*f-1)*half_range, dims = 'position')
    sigma_obs = pm.HalfNormal('sigma_obs',4.0)

    pm.StudentT(
        'y_obs',   
        nu = 4, 
        mu = mu, 
        sigma = sigma_obs, 
        observed= obs_data
    )


with hsgp_version:
    idata_hsgp = pm.sample_prior_predictive()


az.plot_ppc_dist(
    idata_hsgp, 
    group = 'prior_predictive', 
    visuals = {'observed_dist': True}
)

with hsgp_version: 
    idata_hsgp.update(
        pm.sample(random_seed=SEED)
    )

az.plot_ess_evolution(
    idata_hsgp, 
    var_names=['positions_intercepts']
)

check_player = train['player'].unique().sample(5).to_list()

az.plot_ess_evolution(
    idata_hsgp, 
    var_names=['player_intercepts'], 
    coords = {'player': check_player}
)



az.plot_ess_evolution(
    idata_hsgp, 
    var_names=['team_intercepts']
)




az.plot_ess_evolution(
    idata_hsgp, 
    var_names=['tenure_sigma', 'games_sigma'], 
    filter_vars='like'
)

az.plot_ess_evolution(
    idata_hsgp, 
    var_names=['tenure_gp', 'game_gp']
)


with hsgp_version: 
    pm.compute_log_likelihood(
        idata_hsgp
    )


with pm.Model(coords=coords) as rw_check:
    init_sigma_ps = pm.HalfNormal('init_sigma_ps', 0.5)
    rw_sigma_ps = pm.HalfNormal('rw_sigma_ps', 0.15)

    z = pm.Normal('z', 0, 1, dims='week')
    level0 = init_sigma_ps * z[0]
    steps = rw_sigma_ps * z[1:]
    traj = pm.Deterministic(
        'traj', pt.concatenate([[level0], level0 + pt.cumsum(steps)]), dims='week'
    )
    prior = pm.sample_prior_predictive(draws=50, random_seed=0)

for i in range(50):
    plt.plot(coords['week'], prior.prior['traj'].values[0, i], alpha=0.3, color='C0')


with pm.Model(coords = coords) as random_walk_version:
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

    #tenure_grid = pm.Data('tenure_grid', unique_tenure, dims = 'tenure')
    week_grid = pm.Data('week_grid', unique_weeks, dims = 'week')
    

    team_sigma = pm.HalfStudentT('team_sigma', nu =4 , sigma  = 3)
    team_mu = pm.Normal('team_mu', 0, 1, dims = 'team')
    team_intercepts = pm.Deterministic('team_intercepts',
                    team_mu * team_sigma, dims = 'team')


    positions_sigma = pm.HalfCauchy('positions_sigma', 1.0)
    positions_mu = pm.Normal('positions_mu', 3, 2)
    positions_intercepts = pm.Normal('position_intercepts', positions_mu ,positions_sigma, dims = 'position')

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


    mu = pm.Deterministic(
        'mu', 
        intercepts
        + pm.math.dot(cont_data, cont_priors)
        + pm.math.dot(bi_data, bi_priors)
        + f_career
        + f_game
    )

    sigma_obs = pm.HalfNormal('sigma_obs',4.0)

    pm.StudentT(
        'y_obs',   
        nu = 4, 
        mu = mu, 
        sigma = sigma_obs, 
        observed= obs_data
    )


with random_walk_version: 
    idata_walk = pm.sample_prior_predictive()


az.plot_ppc_dist(
    idata_walk, 
    group = 'prior_predictive', 
    visuals = {'observed_dist': True}
)


with random_walk_version:
    idata_walk.update(
        pm.sample(random_seed=SEED)
    )


az.plot_ess_evolution(
    idata_walk, 
    var_names=[rv.name for rv in random_walk_version.free_RVs if rv.size.eval() <= 10]
)


check_player = train['player'].unique().sample(5).to_list()
check_tenure = train['tenure_idx'].unique().sample(2).cast(pl.String).to_list()

az.plot_ess_evolution(
    idata_walk, 
    var_names=['player_skill'], 
    coords = {'player': check_player, 
                'tenure': check_tenure}
)


az.plot_ess_evolution(
    idata_walk, 
    var_names=['team_intercepts']
)

az.plot_ess_evolution(
    idata_walk, 
    var_names ='position_intercepts'
)


len(coords['team'])

with random_walk_version: 
    pm.compute_log_likelihood(idata_walk)



ppc_by_position(
    random_walk_version, idata_walk, train
)


with pm.Model(coords = coords) as rand_week:
    teams_data = pm.Data(
        'teams_data', 
        train['team_season_idx'].to_numpy().squeeze()
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
    

    tenure_grid = pm.Data('tenure_grid', unique_tenure, dims = 'tenure')
    week_grid = pm.Data('week_grid', unique_weeks, dims = 'week')
    

    init_sigma = pm.HalfNormal('init_sigma', 1.0)
    rw_sigma = pm.HalfNormal('rw_sigma', 0.5)
    z_innov = pm.Normal('z_innov', 0,1, dims = ('team_season', 'week'))

    level0 = init_sigma * z_innov[:, 0]
    steps = rw_sigma * z_innov[:, 1:]
    
    team_skill = pm.Deterministic(
        'team_skill',
    pt.concatenate([level0[:, None], level0[:, None] + pt.cumsum(steps, axis=1)], axis=1),
    dims=('team_season', 'week'))

    f_team= team_skill[teams_data, games_data]


    positions_sigma = pm.HalfCauchy('positions_sigma', 1.0)
    #positions_mu = pm.Normal('positions_mu', 3, 2)
    positions_intercepts = pm.ZeroSumNormal('position_intercepts', positions_sigma, dims = 'position')
    player_sigma = pm.HalfStudentT('player_sigma', nu = 4, sigma =  2.5) 
    player_mu = pm.Normal('player_mu', 0 , 1.0, dims = 'player')

    player_intercepts = pm.Deterministic(
        'player_intercepts', 
        player_sigma * player_mu, 
        dims = 'player'
    )

    intercepts = pm.Deterministic(
        'intercepts', 
        player_intercepts[player_data]
        #+ team_intercepts[teams_data]
        + positions_intercepts[position_data]
    )

    cont_priors = pm.Normal('cont_priors', 0, 1, dims = 'cont_vars')
    bi_priors = pm.Normal('bi_priors', 0, 1.5, dims = 'binary_vars')
    



    tenure_sigma = pm.HalfNormal('tenure_sigma', 1) # effecetively two points over a rookie contract is how i am thinking of this prior 
    tenure_prior = pm.InverseGamma(
        'tenure_prior', 
        alpha = age_curve.alpha, 
        beta = age_curve.beta
    )

    cov_tenure = tenure_sigma**2 * pm.gp.cov.Matern52(input_dim=1, ls = tenure_prior)
    gp_tenure = pm.gp.HSGP(m = [age_m],  c = age_c, cov_func=cov_tenure, parametrization='centered', drop_first=True)

    tenure_gp = gp_tenure.prior(
        'tenure_gp', X = tenure_grid[:, None], dims = 'tenure'
    )
    f_tenure = tenure_gp[tenure_data]




    mu = pm.Deterministic(
        'mu', 
        intercepts
        + pm.math.dot(cont_data, cont_priors)
        + pm.math.dot(bi_data, bi_priors)
        + f_tenure 
        + f_team
    )

    sigma_obs = pm.HalfNormal('sigma_obs',4.0)

    pm.StudentT(
        'y_obs',   
        nu = 4, 
        mu = mu, 
        sigma = sigma_obs, 
        observed= obs_data
    )


with rand_week: 
    idata_week = pm.sample_prior_predictive()


az.plot_ppc_dist(
    idata_week, 
    group = 'prior_predictive', 
    visuals = {'observed_dist': True}
)

with rand_week:
    idata_week.update(
        pm.sample(random_seed=SEED)
    )



az.plot_ess_evolution(
    idata_week, 
    var_names=[rv.name for rv in rand_week.free_RVs if rv.size.eval() <= 10]
)


check_player = train['player'].unique().sample(5).to_list()
check_tenure = train['week_idx'].unique().sample(2).cast(pl.Float64).cast(pl.String).to_list()

check_player_season = train['player_season'].unique().sample(5).to_list()

az.plot_ess_evolution(
    idata_week, 
    var_names=['player_skill'], 
    coords = {'player_season': check_player_season, 
                'week': check_tenure}
)


az.plot_ess_evolution(
    idata_week, 
    var_names ='position_intercepts'
)


with rand_week: 
    pm.compute_log_likelihood(idata_week)


az.compare(
    {'Just HSGP': idata_hsgp, 'Player Random Walk': idata_walk, 'Team Random Walk': idata_week}
)

with random_walk_version: 
    idata_walk.update(
        pm.sample_posterior_predictive(idata_walk, compile_kwargs={'mode': 'NUMBA'})
    )

az.plot_ppc_dist(
    idata_walk, 
    kind = 'ecdf'
)

loow_rw = az.loo(idata_walk, pointwise=True)

fig, ax = plt.subplots()

az.plot_khat(loow_rw, )

k = loow_rw.pareto_k.values
bad_idx = np.where(k > 0.7)[0]
print(f"{len(bad_idx)} / {len(k)} obs with k > 0.7")

names = pl.read_parquet('processed-data/ff-processed.parquet').select(
    ['player_id', 'full_name']
).unique()

bad_rows = (
    train[bad_idx]                      # polars: train[bad_idx] or train.gather(bad_idx)
    .join(names, on='player_id', how='left')
    .select(['full_name', 'season', 'week', 'position', 'total_fantasy_points'])
    .sort('total_fantasy_points', descending=True)
)

check = bad_rows['full_name'][0]

raw_data.filter(pl.col("player") == check)

np.where(k > 0.7)

with pm.Model(coords = coords) as skew_walk:
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

    #tenure_grid = pm.Data('tenure_grid', unique_tenure, dims = 'tenure')
    week_grid = pm.Data('week_grid', unique_weeks, dims = 'week')
    

    team_sigma = pm.HalfStudentT('team_sigma', nu =4 , sigma  = 3)
    #team_mu = pm.Normal('team_mu', 2, 1)
    team_intercepts = pm.ZeroSumNormal('team_intercepts',
                    #team_mu,
                    team_sigma, dims = 'team')


    positions_sigma = pm.HalfCauchy('positions_sigma', 1.0)
    positions_mu = pm.Normal('positions_mu', 4, 2.5, )
    positions_intercepts = pm.Normal('position_intercepts', positions_mu, positions_sigma, dims = 'position')

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


    mu = pm.Deterministic(
        'mu', 
        intercepts
        + pm.math.dot(cont_data, cont_priors)
        + pm.math.dot(bi_data, bi_priors)
        + f_career
        + f_game
    )
    FLOOR = 2.5   # comfortably above the variance-existence threshold of 1
    s = pm.Deterministic('shape_sum', 2*FLOOR + pm.Gamma('shape_sum_raw', mu=15, sigma=5, dims = 'position' ))
    shape_split = pm.Beta('shape_split', 2, 2, dims = 'position')
    half_range = s/2 - FLOOR
    skew_a = pm.Deterministic('skew_a', s/2 + (2*shape_split-1)*half_range,dims = 'position' )
    skew_b = pm.Deterministic('skew_b', s/2 - (2*shape_split-1)*half_range,dims = 'position')

    sigma_obs = pm.HalfNormal('sigma_obs',4.0)

    pm.SkewStudentT(
        'y_obs', 
        a = skew_a[position_data], 
        b = skew_b[position_data],   
        nu = 4, 
        mu = mu, 
        sigma = sigma_obs, 
        observed= obs_data
    )


with skew_walk: 
    idata_skew_walk = pm.sample_prior_predictive()


az.plot_ppc_dist(
    idata_skew_walk, 
    group = 'prior_predictive', 
    visuals = {"observed_dist": True}
)


with skew_walk:
    idata_skew_walk.update(
        pm.sample(random_seed=SEED)
    )
az.plot_ess_evolution(
    idata_skew_walk, 
    var_names=[rv.name for rv in skew_walk.free_RVs if rv.size.eval() <= 10]
)


check_player = train['player'].unique().sample(5).to_list()
check_tenure = train['tenure'].unique().sample(2).cast(pl.Float64).cast(pl.String).to_list()



check_player_season = train['player_season'].unique().sample(5).to_list()

az.plot_ess_evolution(
    idata_skew_walk, 
    var_names=['player_skill'], 
    coords = {'player': check_player, 
                'tenure': ['1', '3', '4']}
)


az.plot_ess_evolution(
    idata_skew_walk, 
    var_names =['position_intercepts', 'skew_a', 'skew_b']
)
az.plot_ess_evolution(
    idata_skew_walk, 
    var_names ='team_intercepts'
)
