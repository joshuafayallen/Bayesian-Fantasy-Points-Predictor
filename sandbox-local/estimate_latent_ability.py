import numpy as np
import polars as pl
import pymc as pm
import pytensor.tensor as pt
import arviz as az
from pymc_extras.statespace import structural as st
import matplotlib.pyplot as plt
import tidydraws as td


dat = pl.read_parquet('processed-data/ff-processed.parquet').unique(subset = 'game_id')

home = dat.select(
        pl.col("season"), pl.col("week"),
        pl.col("home_team").alias("team"),
        (pl.col("home_score") - pl.col("away_score")).cast(pl.Float64).alias("margin"),
        (pl.col("home_rest") - pl.col("away_rest")).cast(pl.Float64).alias("rest_diff"),
        pl.lit(1.0).alias("is_home"),
    )
away = dat.select(
        pl.col("season"), pl.col("week"),
        pl.col("away_team").alias("team"),
        (pl.col("away_score") - pl.col("home_score")).cast(pl.Float64).alias("margin"),
        (pl.col("away_rest") - pl.col("home_rest")).cast(pl.Float64).alias("rest_diff"),
        pl.lit(0.0).alias("is_home"),
    )


long = pl.concat([home, away]).sort(['season', 'week', 'team'])
x = long.select(pl.col('is_home', 'rest_diff')).to_numpy()
y = long['margin'].to_numpy()
beta, *_ = np.linalg.lstsq(x,y, rcond = None)
fitted = x @ beta
add_fit = long.with_columns(
    pl.Series('adjusted_margin', y - fitted)
)

global_week = (
    add_fit
    .select(pl.col('season', 'week'))
    .unique()
    .sort(['season', 'week'])
    .with_row_index('global_week')
)

add_global_week = (
    add_fit
    .join(global_week, on = ['season', 'week'])
)


teams = sorted(add_global_week['team'].unique().to_list())
n_week = global_week.height
wide = np.full((n_week, len(teams)), np.nan)
team_pos = {t: i for i, t in enumerate(teams)}


for row in add_global_week.iter_rows(named = True):
    wide[row["global_week"], team_pos[row["team"]]] = row["adjusted_margin"]



ss_mod = (
    st.LevelTrend(order=1, innovations_order=1, observed_state_names=teams)
    + st.MeasurementError(observed_state_names=teams)
).build()

with pm.Model(coords=ss_mod.coords) as team_strength_mod:
    pm.Normal('initial_level_trend', 0, 10, dims=ss_mod.param_dims['initial_level_trend'])
    pm.HalfNormal('sigma_level_trend', 3.0, dims=ss_mod.param_dims['sigma_level_trend'])
    pm.Exponential(
    'sigma_MeasurementError', 1.0,
    dims=ss_mod.param_dims['sigma_MeasurementError'],)
    pm.Deterministic('P0', pt.eye(ss_mod.k_states) * 10.0, dims=ss_mod.param_dims['P0'])
    ss_mod.build_statespace_graph(wide, missing_fill_value=-999.0, save_kalman_filter_outputs_in_idata=True)

    id_test = pm.sample()
    pm.stats.compute_log_likelihood(id_test)


az.plot_ess_evolution(id_test, var_names=['sigma_level_trend'])
az.plot_ess_evolution(id_test, var_names=['sigma_MeasurementError'])

az.summary(id_test, var_names = ['sigma_level_trend', 'initial_level_trend'])


teams = sorted(add_global_week['team'].unique().to_list())  # same canonical list used everywhere
team_pos = {t: i for i, t in enumerate(teams)}

# sanity check before trusting the .sel() below -- confirms the exact label
# format actually present in this idata rather than assuming it
state_labels = id_test.posterior['filtered_states'].coords['state'].values
assert f'level[SF]' in state_labels, state_labels[:5]  # first few, for eyeballing if it fails

team = 'SF'  # swap for whichever team had the clearest story in your data

filtered = id_test.posterior['filtered_states'].sel(state=f'level[{team}]')
smoothed = id_test.posterior['smoothed_states'].sel(state=f'level[{team}]')

fig, ax = plt.subplots(figsize=(10, 4))
filtered.mean(('chain', 'draw')).plot(ax=ax, label='filtered (through week i)')
smoothed.mean(('chain', 'draw')).plot(ax=ax, label='smoothed (all data)')
ax.axhline(0, color='grey', linewidth=0.5)
ax.set_title(f'{team} team strength: filtered vs smoothed')
ax.legend()
plt.show()


az.summary(id_test, var_names=['sigma_level_trend', 'sigma_MeasurementError'])
 
id = az.from_netcdf('model-nc/team_ability.nc')
draws = td.parameter_draws(id, 'filtered_states')

summarized = (
    draws.group_by(
        ['state', 'time']
    )
    .agg(
        pl.col('filtered_states').mean().alias("team_form")
    )
    .with_columns(
        pl.col('state').str.extract(r"level\[(.*)\]").alias('team')
    )
    .rename({'time': 'global_week'})
)

add_form  = (
    summarized
    .join(add_global_week, on  = ['global_week', 'team'])
)



# keep only what ff_ar.py actually needs -- margin/rest_diff/is_home/
# adjusted_margin/state are leftovers from this script's own intermediate
# tables, not something the join in ff_ar.py should be carrying around.
team_form_out = add_form.select('season', 'week', 'team', 'team_form')

assert team_form_out.height == team_form_out.select('season', 'week', 'team').unique().height, (
    "team_form has duplicate (season, week, team) keys -- check the joins above"
)

def state_space_predictive_draws(
    idata,
    games,
    teams,
    beta,
    n_draws=2000,
    seed=42,
):
    rng = np.random.default_rng(seed)

    posterior = idata.posterior

    states = posterior["filtered_states"]

    # Keep only the level states
    states = states.sel(
        state=[f"level[{t}]" for t in teams]
    )

    states = states.stack(sample=("chain", "draw"))

    sample_idx = rng.choice(
        states.sizes["sample"],
        size=n_draws,
        replace=False,
    )

    states = states.isel(sample=sample_idx)

    # Map team -> state index
    team_idx = {team: i for i, team in enumerate(teams)}

    sigma = (
        posterior["sigma_MeasurementError"]
        .stack(sample=("chain", "draw"))
        .isel(sample=sample_idx)
        .values
        .reshape(-1)
    )

    hfa = beta[0]
    beta_rest = beta[1]

    predictions = []

    for row in games.iter_rows(named=True):

        # We need ability from the PREVIOUS week
        t = row["global_week"] - 1

        if t < 0:
            continue

        home_i = team_idx[row["home_team"]]
        away_i = team_idx[row["away_team"]]

        home_ability = states.isel(
            state=home_i,
            time=t,
        ).values

        away_ability = states.isel(
            state=away_i,
            time=t,
        ).values

        mean = (
            home_ability
            - away_ability
            + hfa
            + beta_rest * row["rest_diff"]
        )

        pred = mean + rng.normal(
            0,
            sigma,
            size=n_draws,
        )

        predictions.append(
            {
                "game_id": row["game_id"],
                "season": row["season"],
                "week": row["week"],
                "global_week": row["global_week"],
                "actual": row["margin"],
                "predictions": pred,
            }
        )

    return predictions

team_form_out.write_parquet('processed-data/team_form.parquet')

id_test.to_netcdf('model-nc/team_ability.nc')


games = (
    dat
    .select(
        'game_id', 'season', 'week', 'home_team', 'away_team',
        pl.col('home_score', 'away_score').cast(pl.Float64),
        ((pl.col('home_rest').cast(pl.Float64) - pl.col('away_rest').cast(pl.Float64))
         .clip(-7, 7) / 7).alias('rest_diff'),
    )
    .sort(['season', 'week', 'game_id'])
)
 
global_week = (
    games
    .select(pl.col('season', 'week'))
    .unique()
    .sort(['season', 'week'])
    .with_row_index('global_week')
)
 
add_global_week = games.join(global_week, on = ['season', 'week'])
 
 
teams = sorted(add_global_week['home_team'].unique().to_list())
seasons = sorted(add_global_week['season'].unique().to_list())
team_pos = {t: i for i, t in enumerate(teams)}
season_pos = {s: i for i, s in enumerate(seasons)}
n_week = global_week.height
n_team = len(teams)
 
t_idx = add_global_week['global_week'].to_numpy()
home_idx = add_global_week['home_team'].replace_strict(team_pos).to_numpy()
away_idx = add_global_week['away_team'].replace_strict(team_pos).to_numpy()
season_idx = add_global_week['season'].replace_strict(season_pos).to_numpy()
rest = add_global_week['rest_diff'].to_numpy()
y = add_global_week.select('home_score', 'away_score').to_numpy()
 
# (first, last+1) global week of each season -- where the offseason reversion happens
season_of_week = global_week['season'].replace_strict(season_pos).to_numpy()
starts = np.flatnonzero(np.r_[True, np.diff(season_of_week) != 0])
season_bounds = list(zip(starts, np.r_[starts[1:], n_week]))
 
 
coords = {
    'team': teams, 'season': seasons, 'comp': ['off', 'def'],
    'global_week': np.arange(n_week), 'game': add_global_week['game_id'].to_numpy(),
    'side': ['home', 'away'],
}
 
with pm.Model(coords = coords) as team_strength_mod:
    # scoring environment: league scoring level + home field, per season, partially pooled
    mu_bar = pm.Normal('mu_bar', 22, 5)
    mu_sd = pm.HalfNormal('mu_sd', 2)
    mu = pm.Deterministic('mu', mu_bar + mu_sd * pm.Normal('mu_z', 0, 1, dims = 'season'), dims = 'season')
    hfa_bar = pm.Normal('hfa_bar', 2, 2)
    hfa_sd = pm.HalfNormal('hfa_sd', 1)
    hfa = pm.Deterministic('hfa', hfa_bar + hfa_sd * pm.Normal('hfa_z', 0, 1, dims = 'season'), dims = 'season')
    beta_rest = pm.Normal('beta_rest', 0, 2)   # margin pts per extra week of rest
 
    # dynamics: pooled across teams, separate for offense and defense
    init_sd = pm.HalfNormal('init_sd', 5, dims = 'comp')
    sigma_week = pm.HalfNormal('sigma_week', 1.0, dims = 'comp')       # in-season drift
    sigma_season = pm.HalfNormal('sigma_season', 3.0, dims = 'comp')   # offseason shock
    rho = pm.Beta('rho', 4, 3, dims = 'comp')                          # carryover year to year
 
    # non-centered latent paths: random walk within a season,
    # shrink toward league average (rho) at each season start
    z_init = pm.Normal('z_init', 0, 1, dims = ('comp', 'team'))
    z_season = pm.Normal('z_season', 0, 1, shape = (len(seasons) - 1, 2, n_team))
    z_week = pm.Normal('z_week', 0, 1, shape = (n_week, 2, n_team))
 
    pieces = []
    for i, (t0, t1) in enumerate(season_bounds):
        if i == 0:
            start = init_sd[:, None] * z_init
        else:
            start = rho[:, None] * pieces[-1][-1] + sigma_season[:, None] * z_season[i - 1]
        steps = pt.cumsum(sigma_week[None, :, None] * z_week[t0 + 1:t1], axis = 0)
        pieces.append(start[None] + pt.concatenate([pt.zeros((1, 2, n_team)), steps], axis = 0))
 
    raw = pt.concatenate(pieces, axis = 0)
    # sum-to-zero each week: only differences between teams are identified
    ability = pm.Deterministic('ability', raw - raw.mean(axis = -1, keepdims = True),
                               dims = ('global_week', 'comp', 'team'))
    off, dfn = ability[:, 0, :], ability[:, 1, :]
    pm.Deterministic('team_strength', off + dfn, dims = ('global_week', 'team'))   # margin scale
 
    # each game: home pts ~ off_home - def_away, away pts ~ off_away - def_home
    shift = 0.5 * (hfa[season_idx] + beta_rest * rest)
    mu_home = mu[season_idx] + shift + off[t_idx, home_idx] - dfn[t_idx, away_idx]
    mu_away = mu[season_idx] - shift + off[t_idx, away_idx] - dfn[t_idx, home_idx]
 
    # the two scores in a game are correlated (pace, weather, garbage time)
    sigma_obs = pm.HalfNormal('sigma_obs', 12)
    rho_obs = pm.Uniform('rho_obs', -0.5, 0.8)
    cov = sigma_obs**2 * pt.stack([pt.stack([1.0, rho_obs]), pt.stack([rho_obs, 1.0])])
    pm.MvNormal('scores', mu = pt.stack([mu_home, mu_away], axis = 1), cov = cov,
                observed = y, dims = ('game', 'side'))
 
    id_bt = pm.sample(nuts_sampler = 'nutpie')


with team_strength_mod:
    pm.stats.compute_log_likelihood(id_bt)

az.compare(
    {'State Space': id_test, 'BT style comparision': id_bt}
)

from evaluate_team_models import evaluate
summary, rows = evaluate(id_test, id_bt)
