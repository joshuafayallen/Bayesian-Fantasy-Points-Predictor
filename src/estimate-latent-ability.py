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



# keep only what ff-ar.py actually needs -- margin/rest_diff/is_home/
# adjusted_margin/state are leftovers from this script's own intermediate
# tables, not something the join in ff-ar.py should be carrying around.
team_form_out = add_form.select('season', 'week', 'team', 'team_form')

assert team_form_out.height == team_form_out.select('season', 'week', 'team').unique().height, (
    "team_form has duplicate (season, week, team) keys -- check the joins above"
)



team_form_out.write_parquet('processed-data/team_form.parquet')

id_test.to_netcdf('model-nc/team_ability.nc')
