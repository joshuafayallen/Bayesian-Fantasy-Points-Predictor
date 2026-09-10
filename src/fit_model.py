"""
CLI driver for fitting one of src/models.py's full-history model
architectures (ar/hs/team/r2d2) and saving the result to a .nc file.

This replaces the copy-pasted "hardcode std_these to the shrunk lag
columns" data prep that used to live at the top of src/ff_ar.py and
src/r2d2_mod.py with a single --shrunk flag:

    python src/fit_model.py team --shrunk            # shrunk lag features
    python src/fit_model.py team                      # plain lags (default)
    python src/fit_model.py r2d2 --shrunk --draws 500 --tune 500 --chains 4

--shrunk=True swaps exactly the three lag-rate columns that
processed-data/shrinkage_features.csv has shrunk variants for
(lag_yards_per_rush_attempt, lag_yards_per_target, lag_yards_per_pass_attempt)
for their `_shrunk` counterparts, joined on (player_id, season, week).
Every other std_these column (including lag_target_share, which has no
shrunk variant) is unchanged. This matches src/backtest_harness.py's
STD_COLS when --shrunk is NOT passed, so the plain-lags fit here is
directly comparable to the backtest harness's folds; the --shrunk fit is
the deliberately-separate "does shrinkage help" experiment that was
previously bundled into r2d2_mod.py's data prep (see that file's history)
and is now available for ANY of the four architectures, not just r2d2.

The output idata carries idata.attrs['shrunk'] (bool) and
idata.attrs['model'] (str) so a saved .nc file is self-describing --
you can tell which feature set/architecture produced it without having to
remember the filename convention or re-run this script.
"""
import argparse
import datetime
import os

import arviz as az
import numpy as np
import polars as pl
import polars.selectors as cs
import pymc as pm

import models

SEED = 41029041
DATA_PATH = 'processed-data/ff-processed.parquet'
TEAM_FORM_PATH = 'processed-data/team_form.parquet'
LAST_SEASON = 2025
OUT_DIR = 'model-nc'

# matches src/backtest_harness.py's STD_COLS exactly -- the plain-lags
# (--shrunk not passed) fit here is meant to be directly comparable to the
# backtest harness's folds. Which three of these get swapped for their
# `_shrunk` counterpart when --shrunk is passed lives in
# models.SHRUNK_SWAP_COLS (shared with backtest_harness.py).
STD_COLS = ['spread_line', 'roll_avg_points_allowed', 'lag_yards_per_rush_attempt',
            'lag_target_share', 'lag_yards_per_target', 'lag_yards_per_pass_attempt',
            'temp', 'wind', 'games_missed_ytd', 'lag_offense_pct', 'lag_st_pct']
BINARY_VARS = ['is_grass', 'is_indoors', 'is_home', 'div_game', 'era']
IDX_COLS = ['player', 'position', 'team', 'week', 'tenure', 'player_season',
            'team_season', 'opp_team', 'season', 'games_missed_ytd']

BUILDERS = {
    'ar': lambda D, coords, season_offsets, T, boundary_cols, pos_of_player, pos_counts, week_grid_vals:
        models.build_ar_mod(D, coords, season_offsets, T, boundary_cols, pos_of_player, pos_counts),
    'hs': lambda D, coords, season_offsets, T, boundary_cols, pos_of_player, pos_counts, week_grid_vals:
        models.build_hs_version(D, coords, season_offsets, T, boundary_cols, pos_of_player, pos_counts, week_grid_vals),
    'team': lambda D, coords, season_offsets, T, boundary_cols, pos_of_player, pos_counts, week_grid_vals:
        models.build_team_mod(D, coords, season_offsets, T, boundary_cols, pos_of_player, pos_counts),
    'r2d2': lambda D, coords, season_offsets, T, boundary_cols, pos_of_player, pos_counts, week_grid_vals:
        models.build_r2d2_mod(D, coords, season_offsets, T, boundary_cols, pos_of_player, pos_counts),
}


# ============================================================================
# Data prep -- full-history, one "fold" covering everything up to the
# current in-progress week of LAST_SEASON. Same arithmetic as
# src/backtest_harness.py's per-fold prep (via src/models.py's shared
# scaler/encoder helpers), just run once over all of history.
# ============================================================================

def prep_data(shrunk, last_season=LAST_SEASON):
    raw_data = pl.read_parquet(DATA_PATH)

    # team_form/opp_team_form -- only team_mod/r2d2_mod actually read these
    # (via D['team_form_z']/D['opp_team_form_z']), but computing them is
    # cheap and harmless for ar/hs, and keeps this prep function usable for
    # all four architectures uniformly.
    team_form = pl.read_parquet(TEAM_FORM_PATH)
    add_team_form = raw_data.select(pl.exclude("^.*_right$")).join(
        team_form, on=['season', 'week', 'team'], how='left')
    n_missing_form = add_team_form['team_form'].null_count()
    if n_missing_form:
        print(f"warning: {n_missing_form} rows have no team_form match -- filling with 0.0")
    add_team_form = add_team_form.with_columns(pl.col('team_form').fill_null(0.0))

    opp_team_form = team_form.rename({'team': 'opp_team', 'team_form': 'opp_team_form'})
    add_opp_form = add_team_form.join(opp_team_form, on=['season', 'week', 'opp_team'], how='left')
    n_missing_opp_form = add_opp_form['opp_team_form'].null_count()
    if n_missing_opp_form:
        print(f"warning: {n_missing_opp_form} rows have no opp_team_form match -- filling with 0.0")
    add_opp_form = add_opp_form.with_columns(pl.col('opp_team_form').fill_null(0.0))

    std_these = list(STD_COLS)
    if shrunk:
        # models.apply_shrunk_features left-joins processed-data/
        # shrinkage_features.csv and overwrites SHRUNK_SWAP's columns'
        # VALUES in place (falling back to the plain value where
        # shrinkage_features.csv doesn't cover a row) -- column NAMES are
        # unchanged, so coords['cont_vars'] stays identical in shape/labels
        # between shrunk and non-shrunk fits and only
        # idata.attrs['shrunk'] distinguishes them. Shared with
        # src/backtest_harness.py's --shrunk path so the swap logic can't
        # drift between the two.
        add_opp_form = models.apply_shrunk_features(add_opp_form)

    lw = add_opp_form.filter(pl.col('season') == last_season)['week'].max()
    train = add_opp_form.filter(
        (pl.col('season') < last_season) | ((pl.col('season') == last_season) & (pl.col('week') < lw))
    )

    scalers = models.fit_scalers(df=train, cols=std_these)
    idxs = models.make_indices(raw_data, IDX_COLS)
    train = models.apply_scalers(train, scalers)
    train = models.encode(train, cols=IDX_COLS, enums=idxs)

    _team_form_mu = train['team_form'].mean()
    _team_form_sd = train['team_form'].std()
    train = train.with_columns(
        ((pl.col('team_form') - _team_form_mu) / _team_form_sd).alias('team_form_std'),
        ((pl.col('opp_team_form') - _team_form_mu) / _team_form_sd).alias('opp_team_form_std'),
    )

    coords = {col: enum.categories.to_list() for col, enum in idxs.items()}

    cont_dat_train = train.select(cs.ends_with('_z'))
    coords['cont_vars'] = cont_dat_train.columns

    binary_train = train.select(BINARY_VARS)
    coords['binary_vars'] = binary_train.columns

    weeks_per_season = (
        raw_data.group_by('season').agg(pl.col('week').max().alias('max_week'))
        .sort('season')['max_week'].cast(pl.Int64).to_numpy()
    )
    n_seasons = len(coords['season'])
    assert len(weeks_per_season) == n_seasons
    season_offsets, T, boundary_cols = models.build_ar_calendar(weeks_per_season)

    n_positions = len(coords['position'])
    position_of_player, position_group_counts = models.player_position_map(
        raw_data, 'player', 'position', idxs, n_positions
    )
    assert len(position_of_player) == len(coords['player']), (
        f"position_of_player covers {len(position_of_player)} players, "
        f"expected {len(coords['player'])} -- check idx_cols/coords alignment"
    )

    tenure_grid_vals = np.array(coords['tenure'], dtype='float64')
    tenure_z_row_vals = (
        train['tenure_idx'].to_numpy().astype('float64') - tenure_grid_vals.mean()
    ) / tenure_grid_vals.std()

    week_grid_vals = np.array(coords['week'], dtype='float64')

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

    return dict(D=D, coords=coords, season_offsets=season_offsets, T=T, boundary_cols=boundary_cols,
                position_of_player=position_of_player, position_group_counts=position_group_counts,
                week_grid_vals=week_grid_vals, train=train)


# ------------------------------------------------------------------ fitting

def fit_and_save(model, model_name, shrunk, out_path, seed=SEED, target_accept=0.99, **sample_kwargs):
    """Prior predictive -> NUTS -> posterior predictive + log-likelihood ->
    save to out_path, with idata.attrs recording exactly what produced this
    file (model architecture, shrunk/plain features, when, seed)."""
    with model:
        idata = pm.sample_prior_predictive(random_seed=seed)

    with model:
        idata.update(pm.sample(random_seed=seed, target_accept=target_accept, **sample_kwargs))

    with model:
        pm.compute_log_likelihood(idata)
        idata.update(pm.sample_posterior_predictive(idata))

    # netCDF4 attributes can't be a Python/numpy bool (b1) -- setncattr
    # only accepts S1/i1/u1/i2/u2/i4/u4/i8/u8/f4/f8 -- so store shrunk as
    # a plain int (0/1) instead. compare_fits.py's label_for() and
    # backtest_harness.py both just need it to be truthy/falsy, so this
    # doesn't change how either reads it back.
    idata.attrs['shrunk'] = int(shrunk)
    idata.attrs['model'] = model_name
    idata.attrs['fit_datetime'] = datetime.datetime.now().isoformat()
    idata.attrs['seed'] = int(seed)

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    idata.to_netcdf(out_path)
    print(f"saved {out_path}  (model={model_name}, shrunk={shrunk})")
    return idata


def default_out_path(model_name, shrunk):
    suffix = 'shrunk' if shrunk else 'plain'
    return os.path.join(OUT_DIR, f'ff_{model_name}_{suffix}.nc')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('model', choices=list(BUILDERS.keys()))
    p.add_argument('--shrunk', action='store_true',
                    help="fit with the shrinkage_features.csv-derived shrunk lag columns instead of the plain lags")
    p.add_argument('--last-season', type=int, default=LAST_SEASON)
    p.add_argument('--draws', type=int, default=500)
    p.add_argument('--tune', type=int, default=500)
    p.add_argument('--chains', type=int, default=4)
    p.add_argument('--target-accept', type=float, default=0.9)
    p.add_argument('--seed', type=int, default=SEED)
    p.add_argument('--out', type=str, default=None,
                    help=f"defaults to {OUT_DIR}/ff_<model>_<shrunk|plain>.nc")
    args = p.parse_args()

    np.random.seed(args.seed)

    prep = prep_data(shrunk=args.shrunk, last_season=args.last_season)
    build = BUILDERS[args.model]
    model = build(prep['D'], prep['coords'], prep['season_offsets'], prep['T'], prep['boundary_cols'],
                  prep['position_of_player'], prep['position_group_counts'], prep['week_grid_vals'])

    out_path = args.out or default_out_path(args.model, args.shrunk)
    fit_and_save(model, args.model, args.shrunk, out_path,
                 seed=args.seed, target_accept=args.target_accept,
                 draws=args.draws, tune=args.tune, chains=args.chains, cores=args.chains)


if __name__ == '__main__':
    main()
