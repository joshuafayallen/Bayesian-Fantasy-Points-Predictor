"""
Refit ar_mod from the current src/models.py on real data, saved to a
SEPARATE file from the ground truth -- so it can be diffed against
model-nc/ff_ar.nc with tests/diagnostics_report.py without touching the
ground truth file itself.

This mirrors ff_ar.py's data-prep section exactly (same scalers, same
encode/idx_array calls, same D/coords construction) and calls
models.build_ar_mod the same way ff_ar.py does, but skips prior-predictive
and posterior-predictive/log-likelihood sampling (fit_and_diagnose's extra
steps) since diagnostics_report.py only needs the `posterior` and
`sample_stats` groups -- this keeps the refit itself faster without
changing anything about the NUTS run being diagnosed.

Usage:
    python tests/refit_ar_and_compare.py
    # then, once it finishes:
    python tests/diagnostics_report.py model-nc/ff_ar.nc model-nc/ff_ar_new.nc

Runs PyMC's default NUTS sampler (draws=1000, tune=1000, chains=4,
target_accept=0.95 -- same defaults ff_ar.py's fit_and_diagnose uses) so
the comparison is apples-to-apples with the ground truth run. On real
hardware with a working BLAS this should take low-single-digit minutes,
not the ~30 min/200 draws it took in my sandbox (no BLAS, 2 cores).
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
import polars as pl
import polars.selectors as cs
import pymc as pm
import models

SEED = 41029041
np.random.seed(SEED)
DATA_PATH = 'processed-data/ff-processed.parquet'
LAST_SEASON = 2025
OUT_PATH = 'model-nc/ff_ar_new.nc'

raw_data = pl.read_parquet(DATA_PATH)
team_form = pl.read_parquet('processed-data/team_form.parquet')
add_team_form = raw_data.select(pl.exclude("^.*_right$")).join(team_form, on=['season', 'week', 'team'], how='left')
add_team_form = add_team_form.with_columns(pl.col('team_form').fill_null(0.0))
opp_team_form = team_form.rename({'team': 'opp_team', 'team_form': 'opp_team_form'})
add_opp_form = add_team_form.join(opp_team_form, on=['season', 'week', 'opp_team'], how='left')
add_opp_form = add_opp_form.with_columns(pl.col('opp_team_form').fill_null(0.0))

lw = add_opp_form.filter(pl.col('season') == LAST_SEASON)['week'].max()
train = add_opp_form.filter(
    (pl.col('season') < LAST_SEASON) | ((pl.col('season') == LAST_SEASON) & (pl.col('week') < lw))
)

std_these = ['spread_line', 'roll_avg_points_allowed', 'lag_yards_per_rush_attempt',
             'lag_target_share', 'lag_yards_per_target', 'lag_yards_per_pass_attempt',
             'temp', 'wind', 'games_missed_ytd', 'lag_offense_pct', 'lag_st_pct']
binary_vars = ['is_grass', 'is_indoors', 'is_home', 'div_game', 'era']
idx_cols = ['player', 'position', 'team', 'week', 'tenure', 'player_season',
            'team_season', 'opp_team', 'season', 'games_missed_ytd']

scalers = models.fit_scalers(df=train, cols=std_these)
idxs = models.make_indices(raw_data, idx_cols)
train = models.apply_scalers(train, scalers)
train = models.encode(train, cols=idx_cols, enums=idxs)

_team_form_mu = train['team_form'].mean()
_team_form_sd = train['team_form'].std()
train = train.with_columns(
    ((pl.col('team_form') - _team_form_mu) / _team_form_sd).alias('team_form_std'),
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
season_offsets, T, boundary_cols = models.build_ar_calendar(weeks_per_season)

n_positions = len(coords['position'])
position_of_player, position_group_counts = models.player_position_map(
    raw_data, 'player', 'position', idxs, n_positions
)

tenure_grid_vals = np.array(coords['tenure'], dtype='float64')
tenure_z_row_vals = (
    train['tenure_idx'].to_numpy().astype('float64') - tenure_grid_vals.mean()
) / tenure_grid_vals.std()

D = dict(
    player=train['player_idx'].to_numpy().squeeze(),
    position=train['position_idx'].to_numpy().squeeze(),
    team=train['team_idx'].to_numpy().squeeze(),
    tenure=train['tenure_idx'].to_numpy().squeeze(),
    opp_team=train['opp_team_idx'].to_numpy().squeeze(),
    season=train['season_idx'].to_numpy().squeeze(),
    week=train['week_idx'].to_numpy().squeeze(),
    player_season=train['player_season_idx'].to_numpy().squeeze(),
    cont=cont_dat_train.to_numpy(),
    bi=binary_train.to_numpy(),
    y=train['total_fantasy_points'].to_numpy(),
    tenure_z=tenure_z_row_vals,
    games_missed_z=train['games_missed_ytd_z'].to_numpy(),
    snap_pct_z=train['lag_offense_pct_z'].to_numpy(),
    team_form_z=train['team_form_std'].to_numpy(),
    opp_team_form_z=train['opp_team_form_std'].to_numpy(),
)

ar_mod = models.build_ar_mod(D, coords, season_offsets, T, boundary_cols,
                              position_of_player, position_group_counts)
print("model built, sampling...")

with ar_mod:
    idata = pm.sample(random_seed=SEED, target_accept=0.95)

idata.to_netcdf(OUT_PATH)
print(f"saved to {OUT_PATH}")
