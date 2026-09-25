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
BART_M = 50  # number of trees -- overridden by --bart-m in main() before BUILDERS['bart'] is called
 
# STD_COLS/BINARY_VARS now live in models.py as the single shared copy --
# see that module's comment for why (fit_model.py's old local STD_COLS
# and backtest_harness.py's had already drifted by one column despite
# both claiming to match). Which columns get swapped-in-place vs.
# kept-alongside-raw when --shrunk is passed lives in
# models.SHRUNK_SWAP_COLS/models.SHRUNK_ADD_COLS (also shared with
# backtest_harness.py).
IDX_COLS = ['player_id', 'position', 'team', 'week', 'tenure', 'player_season',
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
    'bart': lambda D, coords, season_offsets, T, boundary_cols, pos_of_player, pos_counts, week_grid_vals:
        models.build_bart_mod(D, coords, pos_of_player, pos_counts, m=BART_M)
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
 
    std_these = list(models.STD_COLS)
    if shrunk:
        add_opp_form = models.apply_shrunk_features(add_opp_form)
 
    lw = add_opp_form.filter(pl.col('season') == last_season)['week'].max()
    train = add_opp_form.filter(
        (pl.col('season') < last_season) | ((pl.col('season') == last_season) & (pl.col('week') < lw))
    )

    scalers = models.fit_scalers(df=train, cols=std_these)
    idxs = models.make_indices(raw_data, models.IDX_COLS)
    team_form_scaler = models.fit_team_form_scaler(train)

    D = models.frame_to_D(train, scalers, idxs, team_form_scaler=team_form_scaler, idx_cols=models.IDX_COLS)

    coords = {col: enum.categories.to_list() for col, enum in idxs.items()}

    coords['cont_vars'] = [f'{c}_z' for c in scalers]
    coords['binary_vars'] = list(models.BINARY_VARS)

    weeks_per_season = (
        raw_data.group_by('season').agg(pl.col('week').max().alias('max_week'))
        .sort('season')['max_week'].cast(pl.Int64).to_numpy()
    )
    n_seasons = len(coords['season'])
    assert len(weeks_per_season) == n_seasons
    season_offsets, T, boundary_cols = models.build_ar_calendar(weeks_per_season)
 
    n_positions = len(coords['position'])
    position_of_player, position_group_counts = models.player_position_map(
        raw_data, 'player_id', 'position', idxs, n_positions
    )
    assert len(position_of_player) == len(coords['player_id']), (
        f"position_of_player covers {len(position_of_player)} players, "
        f"expected {len(coords['player_id'])} -- check idx_cols/coords alignment"
    )

    week_grid_vals = np.array(coords['week'], dtype='float64')
 
    return {'D': D, 'coords': coords, 'season_offsets': season_offsets, 'T': T, 'boundary_cols': boundary_cols,
                'position_of_player': position_of_player, 'position_group_counts': position_group_counts,
                'week_grid_vals': week_grid_vals, 'train': train,
    
                'scalers': scalers, 'idxs': idxs, 'team_form_scaler': team_form_scaler}
 
 
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
        pm.compute_log_prior(idata)
        idata.update(pm.sample_posterior_predictive(idata))
 

    idata.attrs['shrunk'] = int(shrunk)
    idata.attrs['model'] = model_name
    idata.attrs['fit_datetime'] = datetime.datetime.now().isoformat()  # noqa: DTZ005
    idata.attrs['seed'] = int(seed)
 
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    idata.to_netcdf(out_path)
    print(f"saved {out_path}  (model={model_name}, shrunk={shrunk})")
    return idata
 
 
def default_out_path(model_name, shrunk):
    suffix = 'shrunk' if shrunk else 'plain'
    return os.path.join(OUT_DIR, f'ff_{model_name}_{suffix}.nc')
 
 
def main():
    global BART_M
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
    p.add_argument('--bart-m', type=int, default=BART_M,
                    help="number of trees for model=bart's pymc_bart.BART ensemble "
                         "(ignored for every other model)")
    p.add_argument('--out', type=str, default=None,
                    help=f"defaults to {OUT_DIR}/ff_<model>_<shrunk|plain>.nc")
    args = p.parse_args()
 
    BART_M = args.bart_m
 
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