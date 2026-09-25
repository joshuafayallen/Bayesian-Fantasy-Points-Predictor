"""
Shared PyMC model-building plumbing for the fantasy-points predictor.

WHY THIS FILE EXISTS
---------------------
Before this module existed, src/ff_ar.py (the full-history "fit and
diagnose" script) and src/backtest_harness.py (the windowed walk-forward
backtest) each defined their own copies of every model-building function
(add_player_skill, add_strength_ar, add_recent_form_ar/hsgp,
add_role_mixture, add_lkj_position_intercept, ...) and the three model
architectures built from them (ar_mod, hs_version, team_mod). The two
copies were ported "by hand" and had already drifted at least once (see
git history / backtest_harness.py's old module docstring, which had an
explicit "IMPORTANT: this file has no way to detect drift from
src/ff_ar.py automatically" warning).

This module is now the single source of truth for that model-building
code. src/ff_ar.py and src/backtest_harness.py both import from here.
Nothing in this module reads data from disk, does any Polars-frame-level
data prep tied to a *specific* run, or has module-level state -- every
function takes exactly the coords/arrays it needs as arguments and
returns a value (a pm.Model, a pm.Data node, a dict of arrays). That is
what makes it safe to call once from ff_ar.py's single full-history
"fold" and once per fold from backtest_harness.py's walk-forward loop.

SOURCE OF TRUTH
----------------
Every function here is ported from src/ff_ar.py's inline model-building
code, which remains the canonical description of "what the model is."
When ff_ar.py's models change, this file should change to match, and
callers (ff_ar.py and backtest_harness.py) should not need to change at
all beyond picking up the new coords/data plumbing requires.
"""

import numpy as np
import polars as pl
import pymc as pm
import pymc_bart as pmb
import pymc_extras as pmx
import pytensor
import pytensor.tensor as pt

# Single source of truth for the continuous/binary feature lists every
# model architecture uses. src/fit_model.py's full-history fit and
# src/backtest_harness.py's walk-forward folds used to each hand-copy
# these (STD_COLS in particular had drifted: fit_model.py's copy carried
# an extra 'lag_avg_depth_of_target' column backtest_harness.py's copy
# didn't, so the two were silently NOT directly comparable despite
# fit_model.py's own docstring claiming they matched). Both files now
# import these from here instead.
STD_COLS = ['spread_line', 'roll_avg_points_allowed', 'lag_yards_per_rush_attempt',
            'lag_target_share', 'lag_yards_per_target', 'lag_yards_per_pass_attempt',
            'temp', 'wind', 'games_missed_ytd', 'lag_offense_pct', 'lag_st_pct',
            'lag_avg_depth_of_target', 'lag_rush_share']
BINARY_VARS = ['is_grass', 'is_indoors', 'is_home', 'div_game', 'era']
IDX_COLS = ['player_id', 'position', 'team', 'week', 'tenure', 'player_season',
            'team_season', 'opp_team', 'season', 'games_missed_ytd']

# bart_mod3's raw (unstandardized) covariate list -- STD_COLS plus
# team_form/opp_team_form as RAW features (unlike ar_mod/hs_version's
# cont_vars, which deliberately excludes team_form/opp_team_form so they
# stay a clean ablation against team_mod's separate beta_team_form/
# beta_opp_team_form coefficients -- BART has no such ablation concern).
# BINARY_VARS is concatenated on separately by each caller building the
# actual bart_cont array (see src/backtest_harness.py's Fold.arrays and
# src/fit_model.py's prep_data), same as this list previously lived only
# in src/backtest_harness.py.
BART_CONT_VARS = STD_COLS + ['team_form', 'opp_team_form']
 
# bart_mod needs none of the AR-calendar/opp_team/season/week/
# player_season machinery every other architecture shares -- just
# player/position/tenure, for add_player_skill. player_id (not a display
# name) -- see SHRUNK_SWAP_COLS-adjacent note in fit_model.py/
# backtest_harness.py: player names aren't unique (e.g. more than one
# real "Josh Allen" can appear across positions/teams), so indexing on
# name would silently conflate distinct careers.
BART_IDX_COLS = ['player_id', 'position', 'tenure']

SHRINKAGE_PATH = 'processed-data/shrinkage_features.csv'

SHRUNK_SWAP_COLS = ['lag_yards_per_rush_attempt', 'lag_yards_per_target', 'lag_yards_per_pass_attempt', 'lag_avg_depth_of_target']

## Just in case we want to trial another shrunk featrue
SHRUNK_ADD_COLS = []

def apply_shrunk_features(df, swap_cols=SHRUNK_SWAP_COLS, path=SHRINKAGE_PATH, add_cols = SHRUNK_ADD_COLS):
    """Left-join processed-data/shrinkage_features.csv onto `df` (keyed on
    player_id/season/week) and:
      - for each column in `swap_cols`: overwrite its VALUE in place with
        its `_shrunk` counterpart, falling back to the original (unshrunk)
        value via pl.coalesce for any row shrinkage_features.csv doesn't
        cover (rather than dropping it or leaving a null). Column NAME is
        unchanged, so a caller's existing std_these/STD_COLS list works
        unmodified for these columns whether or not shrinkage was
        applied.
      - for each column in `add_cols`: leave the original column
        untouched and ADD a new `{col}_shrunk` column (same coalesce-to-
        raw fallback for uncovered rows), so a caller sees BOTH the raw
        and shrunk signal as separate features rather than one replacing
        the other.
 
    Left-joined + coalesce instead of inner-joined so a coverage gap
    shows up as a printed warning and a graceful fallback, not silently
    dropped rows -- an earlier version of this join (in what's now
    src/r2d2_mod.py) used a bare inner join, which is exactly the bug
    this fixes."""

    all_cols = swap_cols + add_cols
    
    shrunk_estimates = pl.read_csv(path).select(
        'player_id', 'season', 'week', *[f'{c}_shrunk' for c in all_cols])
    n0 = df.height
    out = df.join(shrunk_estimates, on=['player_id', 'season', 'week'], how='left')
    assert out.height == n0, f"shrinkage_features join fanned out rows: {n0} -> {out.height}"
    n_missing = out[f'{swap_cols[0]}_shrunk'].null_count()
    if n_missing:
        print(f"warning: {n_missing} rows have no shrinkage_features match -- "
              f"falling back to the plain (unshrunk) value for those rows")
    for col in swap_cols:
        out = out.with_columns(pl.coalesce([f'{col}_shrunk', col]).alias(col))
    for col in add_cols:
        out = out.with_columns(pl.coalesce([f'{col}_shrunk', col]).alias(f'{col}_shrunk'))
    return out.drop([f'{col}_shrunk' for col in swap_cols])
 
 
# ============================================================================
# Data prep -- scalers, categorical encoders, arrays
# ============================================================================
# Two parallel representations are kept on purpose:
#   * the polars/frame-level helpers (fit_scalers, apply_scalers, encode)
#     operate on a whole training frame at once and add `_z`/`_idx` columns
#     -- this is ff_ar.py's original style, convenient for a single
#     full-history fit where "the frame" and "the model's training data"
#     are the same object throughout the file.
#   * the numpy-array helpers (cont_array, bi_array, idx_array) build the
#     flat arrays a pm.Data node actually wants -- this is
#     backtest_harness.py's original style, convenient when a fold's
#     train/test frames need to be turned into arrays independently using
#     encoders fit once on `train`.
# Both compute the exact same standardization ((x - mu) / sd) and the same
# categorical -> contiguous-int encoding; pick whichever fits the calling
# code.
 
def fit_scalers(df, cols):
    """{col: (mean, sd)} fit on `df`. sd is floored to 1.0 when a column is
    constant in this slice (e.g. a windowed fold with no variance in some
    covariate) so apply_scalers/cont_array never divides by zero."""
    stats = df.select(
        [pl.col(c).mean().alias(f"{c}__mu") for c in cols] +
        [pl.col(c).std().alias(f"{c}__sd") for c in cols]
    ).row(0, named=True)
    return {c: (stats[f"{c}__mu"], stats[f"{c}__sd"] or 1.0) for c in cols}

def fit_team_form_scaler(train):
    mu = train['team_form'].mean()
    sd = train['team_form'].std()
    return mu, sd 

def apply_team_form_scaler(frame, fit_team_form_scaler):
    mu, sd = fit_team_form_scaler
    return frame.with_columns(
        ((pl.col('team_form') - mu)/sd).alias('team_form_std'),
        ((pl.col('opp_team_form') - mu)/sd).alias('opp_team_form_std')
    )

def apply_scalers(frame, scalers):
    """Add a `{col}_z` column for every entry in `scalers`."""
    return frame.with_columns([
        ((pl.col(c) - mu) / sd).alias(f"{c}_z") for c, (mu, sd) in scalers.items()
    ])
 
 
def cont_array(frame, scalers):
    """Same standardization as apply_scalers, as a single (n_rows, n_cols)
    numpy array in `scalers`' key order -- for building a cont_data pm.Data
    node directly instead of going through apply_scalers + cs.ends_with('_z')."""
    cols = list(scalers.keys())
    if not cols:
        return np.zeros((len(frame), 0))
    return np.column_stack([((frame[c] - mu) / sd).to_numpy() for c, (mu, sd) in scalers.items()])
 
 
def bi_array(frame, binary_vars):
    return frame.select(binary_vars).to_numpy().astype(float)
 
 
def make_enums(df, cols):
    """{col: pl.Enum(sorted unique categories)}, fit on `df`."""
    return {
        col: pl.Enum(df[col].unique().sort().cast(pl.String).to_list())
        for col in cols
    }
 
 
# kept as an alias -- ff_ar.py originally called this make_indices.
make_indices = make_enums
 
 
def encode(df, cols, enums):
    """Add a `{col}_idx` Int32 column for every entry in `enums`, using each
    column's already-fit pl.Enum categorical encoding."""
    return df.with_columns([
        pl.col(col).cast(pl.String).cast(enums[col]).to_physical().cast(pl.Int32).alias(f"{col}_idx")
        for col in cols
    ])
 
 
def idx_array(frame, col, enums):
    """Same encoding as encode(), as a single numpy int array for one column
    -- for arrays that were encoded with a *different* fold's enums than the
    frame they're being applied to (e.g. scoring a test frame against
    train-fit enums)."""
    return (frame[col].cast(pl.String).cast(enums[col]).to_physical()
            .cast(pl.Int32).to_numpy())
 
 
# ============================================================================
# Calendar helpers -- the global weekly AR timeline shared by every model's
# opp_ar (and, for team_season_mod, the coarser team/season timeline)
# ============================================================================
 
def build_ar_calendar(weeks_per_season):
    """Given the number of weeks present for each season (in season order),
    return (season_offsets, T, boundary_cols):
      season_offsets -- cumulative week count, one extra leading 0. Season i
                         occupies global steps season_offsets[i]:season_offsets[i+1].
      T              -- total number of global weekly steps.
      boundary_cols  -- innovation-array column indices (into a (group, T-1)
                         innovation matrix) corresponding to the season-jump
                         steps: the first real week of every season but the
                         first. Used by add_strength_ar's season_jump term.
 
    Shared math behind ff_ar.py's top-level season_offsets/T/boundary_cols
    derivation and backtest_harness.py's per-fold ar_calendar()."""
    weeks_per_season = np.asarray(weeks_per_season, dtype='int64')
    season_offsets = np.concatenate([[0], np.cumsum(weeks_per_season)]).astype(int)
    T = int(season_offsets[-1])
    boundary_step_idx = season_offsets[1:-1]  # first real week of every season but the first
    boundary_cols = boundary_step_idx - 1
    return season_offsets, T, boundary_cols
 
 
def global_time(season_idx, week_idx, season_offsets):
    """Global weekly step for each row, given per-row season/week indices
    and the season_offsets from build_ar_calendar."""
    return (season_offsets[:-1][season_idx] + week_idx).astype('int32')
 
 
# ============================================================================
# Player/position lookup -- for add_player_skill's within-position centering
# ============================================================================
 
def player_position_map(frame, col_player, col_position, enums, n_positions):
    """player_idx -> position_idx lookup (`position_of_player`, one entry per
    player) plus per-position player counts (`position_group_counts`),
    built from `frame`'s own (player, position) pairs and `enums`.
 
    Every player must map to exactly one position in `frame` (a player who
    switched position mid-history would violate add_player_skill's
    within-position centering assumption) -- this is not defended against
    here; `.unique(subset=[col_player], keep='first')` silently takes
    whichever row comes first, matching the historical behavior of both
    ff_ar.py and backtest_harness.py."""
    n_players = len(enums[col_player].categories)
    lut = frame.select([col_player, col_position]).unique(subset=[col_player], keep='first')
    parr = idx_array(lut, col_player, enums)
    posarr = idx_array(lut, col_position, enums)
    out = np.zeros(n_players, dtype=int)
    out[parr] = posarr
    counts = np.bincount(out, minlength=n_positions).astype('float64')
    return out, counts


def frame_to_D(frame, scalers, idxs, team_form_scaler = None, idx_cols = None, y_col = 'total_fantasy_points'):
    """Build the dicts that exact moedls expect from a frame that already has lag features computed. Can be passed to training data
    or forecasting data
    
    """

    idx_cols = idx_cols or IDX_COLS
    frame = apply_scalers(frame, scalers)
    frame = encode(frame, cols = idx_cols, enums = idxs)
    D = {
        'player': frame['player_id_idx'].to_numpy().squeeze(),
        'position': frame['position_idx'].to_numpy().squeeze(),
        'team': frame['team_idx'].to_numpy().squeeze(),
        'tenure': frame['tenure_idx'].to_numpy().squeeze(),
        'opp_team': frame['opp_team_idx'].to_numpy().squeeze(),
        'season': frame['season_idx'].to_numpy().squeeze(),
        'week': frame['week_idx'].to_numpy().squeeze(),
        'player_season': frame['player_season_idx'].to_numpy().squeeze(),
        'cont': cont_array(frame, scalers),
        'bi': bi_array(frame, BINARY_VARS),
        'bart_cont': frame.select(BART_CONT_VARS + BINARY_VARS).to_numpy().astype('float64'),
        'y': frame[y_col].to_numpy(),
        'games_missed_z': frame['games_missed_ytd_z'].to_numpy(),
        'snap_pct_z': frame['lag_offense_pct_z'].to_numpy(),
        'target_share_z': frame['lag_target_share_z'].to_numpy(),
        'rush_share_z': frame['lag_rush_share_z'].to_numpy()}

    # tenure_z -- same standardization as fit_model.py's old inline
    # tenure_grid_vals block, just using idxs['tenure']'s categories
    # (the training-time tenure grid) instead of a caller-supplied coords
    # dict, so this needs nothing beyond what's already passed in.
    tenure_grid_vals = np.array(idxs['tenure'].categories, dtype='float64')
    D['tenure_z'] = (
        frame['tenure_idx'].to_numpy().astype('float64') - tenure_grid_vals.mean()
    ) / tenure_grid_vals.std()

    if team_form_scaler is not None:
        frame = apply_team_form_scaler(frame, team_form_scaler)
        D['team_form_z'] = frame['team_form_std'].to_numpy()
        D['opp_team_form_z'] = frame['opp_team_form_std'].to_numpy()
 
    return D


# ============================================================================
# pm.Data plumbing -- must be called inside an active `with pm.Model(...):`
# ============================================================================
 
def base_data(D, season_offsets = None):
    """The pm.Data nodes needed by every model below. Not every model uses
    every key as a *used* input (e.g. team_data/season_data are unused by
    ar_mod/hs_version/team_mod's mu, only by the retired team_season_mod)
    -- unused pm.Data nodes are harmless, and one definition here is much
    less likely to drift than several near-identical copies.
 
    `D` must be a dict with the arrays produced by fold_arrays() (or an
    equivalent construction): player, position, team, tenure, cont, bi, y,
    opp_team, season, week, player_season, games_missed_z, snap_pct_z."""
    return {
        'player_data': pm.Data('player_data', D['player']),
        'position_data': pm.Data('position_data', D['position']),
        'team_data': pm.Data('team_data', D['team']),
        'tenure_data': pm.Data('tenure_data', D['tenure']),
        'cont_data': pm.Data('cont_data', np.ascontiguousarray(D['cont'])),
        'bi_data': pm.Data('bi_data', np.ascontiguousarray(D['bi'])),
        'obs_data': pm.Data('obs_data', D['y']),
        'opp_team_data': pm.Data('opp_team_data', D['opp_team']),
        'season_data': pm.Data('season_data', D['season']),
        'week_data': pm.Data('week_data', D['week']),
        'player_season_data': pm.Data('player_season_data', D['player_season']),
        'games_missed_data': pm.Data('games_missed_data', D['games_missed_z']),
        'snap_pct_data': pm.Data('snap_pct_data', D['snap_pct_z']),
        'global_time_data': pm.Data('global_time_data', global_time(D['season'], D['week'],season_offsets if season_offsets is not None else D['week'])),
        'target_share_data': pm.Data("target_share_data", D['target_share_z']), 
        'rush_share_data': pm.Data('rush_share_data', D['rush_share_z'])
    }


def predict_payload(model_name, D, season_offsets):
    """{pm_data_name: array} for pm.set_data(), from a frame_to_D()-built
    D (new/prediction rows, encoded and standardized with TRAINING-time
    scalers/enums -- see frame_to_D). Maps D's keys to exactly the
    pm.Data names each build_* function below registers.
 
    KEEP THIS IN SYNC with base_data() above and with whatever extra
    pm.Data nodes build_hs_version()/build_team_mod()/build_r2d2_mod()/
    build_bart_mod() create beyond base_data() -- a mismatch here (wrong
    name, or a pm.Data node this function forgets to include) silently
    leaves that node at its TRAINING-time value/length instead of
    updating it, which is exactly the bug forecast_roster.py's old
    hand-written pm.set_data dict had (a 'teams_data'/'team_data' typo, a
    'games_data' that doesn't exist, and several required keys -- 
    opp_team_data, season_data, global_time_data, team_form_data among
    them -- missing entirely).
 
    `season_offsets` must be the SAME array build_ar_calendar produced at
    training time (not recomputed from the prediction frame alone -- a
    single week's data can't determine season boundaries)."""
    if model_name == 'bart':
        return {
            'bart_cont_data': np.ascontiguousarray(D['bart_cont']),
            'player_data': D['player'],
            'tenure_data': D['tenure'],
            'obs_data': D['y'],
        }
 
    payload = {
        'player_data': D['player'],
        'position_data': D['position'],
        'team_data': D['team'],
        'tenure_data': D['tenure'],
        'cont_data': np.ascontiguousarray(D['cont']),
        'bi_data': np.ascontiguousarray(D['bi']),
        'obs_data': D['y'],
        'opp_team_data': D['opp_team'],
        'season_data': D['season'],
        'week_data': D['week'],
        'player_season_data': D['player_season'],
        'games_missed_data': D['games_missed_z'],
        'snap_pct_data': D['snap_pct_z'],
        'target_share_data': D['target_share_z'], 
        'rush_share_data': D['rush_share_z'],
        'global_time_data': global_time(D['season'], D['week'], season_offsets),
    }
    if model_name in ('hs', 'team', 'r2d2'):
        # build_hs_version/build_team_mod/build_r2d2_mod each additionally
        # register their own tenure_z_data (base_data doesn't -- only
        # these three use a continuous tenure_z rather than the
        # integer tenure_data index).
        payload['tenure_z_data'] = D['tenure_z']
    if model_name in ('team', 'r2d2'):
        # build_team_mod/build_r2d2_mod additionally register
        # team_form_data/opp_team_form_data -- requires D to have come
        # from a frame_to_D(..., team_form_scaler=...) call.
        payload['team_form_data'] = D['team_form_z']
        payload['opp_team_form_data'] = D['opp_team_form_z']
    return payload
# ============================================================================
# Shared model-building blocks
# ============================================================================
# Every function below must be called inside an active `with pm.Model(...):`
# context.
 
def add_player_skill(coords, position_of_player, position_group_counts, n_positions, total_scale=None):
    """The random-walk-over-tenure player_skill term, hard-centered to zero
    mean within each position at every tenure level. Without this
    centering, positions_mu/ab_position and the average starting level of
    player_skill within a position are redundant, which tanks the
    position-level parameters' ESS.
 
    total_scale=None (default): init_sigma/rw_sigma are independent
    HalfNormal priors, as used by ar_mod/hs_version/team_mod.
    total_scale=<scalar tensor>: init_sigma/rw_sigma are instead a Beta-
    distributed split of a caller-supplied total variance budget (used by
    r2d2_mod's build_r2d2_mod) -- e.g. a scale derived from an R2D2M2CP-
    style decomposition. init_sigma^2 + rw_sigma^2 == total_scale^2 either
    way, just with the split itself estimated (career_shape) instead of
    the two amplitudes being independently free."""
    n_tenure = len(coords['tenure'])
    if total_scale is None:
        init_sigma = pm.HalfNormal('init_sigma', 4.0)
        rw_sigma = pm.HalfNormal('rw_sigma', 0.5)
    else:
        career_shape = pm.Beta('career_shape', 2, 2)
        init_sigma = total_scale * pt.sqrt(career_shape)
        rw_sigma = total_scale * pt.sqrt(1 - career_shape)
    z_innov = pm.Normal('z_innov', 0, 1, dims=('player_id', 'tenure'))
    level0 = init_sigma * z_innov[:, 0]
    steps = rw_sigma * z_innov[:, 1:]  # (player, n_tenure - 1)
 
    # player_skill_raw via pytensor.scan instead of
    # concatenate([level0[:, None], level0[:, None] + cumsum(steps, axis=1)])
    # -- that concatenate (combining level0 with a cumsum-of-a-Mul-derived
    # block) is a known trigger for a pytensor graph-rewrite bug ("BUG IN
    # FGRAPH.REPLACE OR A LISTENER", local_add_of_sparse_write). scan builds
    # the recursion step-by-step, so there's no concatenate-of-a-computed-
    # block for pytensor's canonicalizer to rewrite into a sparse write.
    # Verified numerically identical to the cumsum-based construction in a
    # standalone numpy script.
    def _tenure_step(step_t, prev):
        return prev + step_t
 
    steps_seq = steps.T  # (n_tenure - 1, player)
    path, _ = pytensor.scan(
        fn=_tenure_step,
        sequences=[steps_seq],
        outputs_info=[level0],
        strict=True,
    )  # (n_tenure - 1, player) -- tenure steps 1..n_tenure-1
 
    player_skill_raw = pt.concatenate([level0[None, :], path], axis=0).T  # (player, n_tenure)
 
    position_of_player_data = pm.Data('position_of_player', position_of_player, dims='player_id')
 
    # one-hot (position x player) matmul instead of
    # inc_subtensor(group_sum[position_of_player_data, :], player_skill_raw)
    # -- same local_add_of_sparse_write trigger family as above.
    # position_of_player is a plain numpy array (not pm.Data), known at
    # graph-build time, so the one-hot matrix is a static constant -- one
    # dense matmul, not a meaningfully different computation.
    one_hot_position = np.zeros((n_positions, len(position_of_player)))
    one_hot_position[position_of_player, np.arange(len(position_of_player))] = 1.0
    group_sum = pt.dot(one_hot_position, player_skill_raw)
    group_mean = group_sum / position_group_counts[:, None]
 
    return pm.Deterministic(
        'player_skill',
        player_skill_raw - group_mean[position_of_player_data, :],
        dims=('player_id', 'tenure'),
    )
 
 
def add_strength_ar(prefix, group_dim, group_idx_data, global_time_data, T, boundary_cols, total_scale=None):
    """Continuous non-centered AR(1)-with-season-jumps strength curve over
    `group_dim` (e.g. 'opp_team') across the full weekly timeline. Used for
    opp_ar in every model.
 
    total_scale=None (default): sigma_theta/sigma_season are independent
    HalfNormal priors, as used by ar_mod/hs_version/team_mod.
    total_scale=<scalar tensor>: sigma_theta/sigma_season are instead a
    Beta-distributed split of a caller-supplied total variance budget
    (used by r2d2_mod's build_r2d2_mod)."""
    rho = pm.Beta(f'{prefix}_rho', 2, 2)
    if total_scale is None:
        sigma_theta  = pm.HalfNormal(f'{prefix}_sigma_theta', 1.0)
        sigma_season = pm.HalfNormal(f'{prefix}_sigma_season', 1.0)
    else:
        ar_shape = pm.Beta(f'{prefix}_shape', 2, 2)
        sigma_theta  = total_scale * pt.sqrt(ar_shape)
        sigma_season = total_scale * pt.sqrt(1 - ar_shape)
    theta_init   = pm.Normal(f'{prefix}_theta_init', 0, 1, dims=group_dim)
    z_theta = pm.Normal(f'{prefix}_z_theta', 0, 1, dims=(group_dim, 'global_step'))
    z_eta   = pm.Normal(f'{prefix}_z_eta', 0, 1, dims=(group_dim, 'season_gap'))
 
    # boundary-column injection as a dense scatter matmul instead of
    # innov[:, boundary_cols] += ... -- pt.set_subtensor/AdvancedIncSubtensor
    # immediately followed by an Add downstream is a known trigger for the
    # same FGRAPH.REPLACE / local_add_of_sparse_write bug. scatter_mat is a
    # small fixed 0/1 constant, so this is one extra small matmul.
    # `if len(boundary_cols)` guards a single-season slice (a short
    # backtest fold's train window can legitimately have no season
    # boundary at all) -- harmless no-op for the usual multi-season case.
    if len(boundary_cols):
        n_gaps = len(boundary_cols)
        scatter_mat = np.zeros((n_gaps, T - 1))
        scatter_mat[np.arange(n_gaps), boundary_cols] = 1.0
        season_jump = sigma_season * pt.dot(z_eta, scatter_mat)  # (group_dim, T-1)
        innov = sigma_theta * z_theta + season_jump
    else:
        innov = sigma_theta * z_theta
 
    # theta_full via pytensor.scan instead of the closed-form rho-power-
    # matrix trick -- rho_powers (a large static triangular, mostly-zero
    # constant) kept triggering the same FGRAPH.REPLACE crash regardless of
    # how the surrounding code was written, since pytensor's own optimizer
    # was rewriting the resulting tensordot internally. scan avoids the
    # triangular constant entirely.
    #
    # recursion: theta_full[:, 0] = theta_init
    #            theta_full[:, t] = rho * theta_full[:, t-1] + innov[:, t-1]
    # (verified against the closed-form construction in a standalone numpy
    # script -- identical to float precision.)
    def _ar_step(innov_t, theta_prev, rho):
        return rho * theta_prev + innov_t
 
    innov_seq = innov.T  # (T-1, group_dim) -- scan iterates over axis 0
 
    theta_path, _ = pytensor.scan(
        fn=_ar_step,
        sequences=[innov_seq],
        outputs_info=[theta_init],
        non_sequences=[rho],
        strict=True,
    )  # (T-1, group_dim) -- steps t=1..T-1
 
    theta_full = pt.concatenate([theta_init[None, :], theta_path], axis=0).T  # (group_dim, T)
 
    strength = pm.Deterministic(
        f'{prefix}_ar',
        theta_full - pt.mean(theta_full, axis=0, keepdims=True),
        dims=(group_dim, 'global_step_full'),
    )
    return strength[group_idx_data, global_time_data]
 
 
def add_team_season_ar(team_data, season_data):
    """Team quality as a season-level AR(1) -- persistent/geometrically-
    decaying like opp_ar's rho, but held constant across every week within
    a season rather than reshuffling weekly. Used only by the retired
    team_season_mod (see build_team_season_mod) -- neither ar_mod,
    hs_version nor team_mod use this."""
    team_rho = pm.Beta('team_rho', 4, 2)
    team_sigma = pm.HalfNormal('team_sigma', 1.0)
    team_season_raw = pm.AR(
        'team_season_ar_raw', rho=team_rho, sigma=team_sigma,
        init_dist=pm.Normal.dist(0, 1.0), constant=False, ar_order=1,
        dims=('team', 'season'),
    )
    team_season_ar = pm.Deterministic(
        'team_season_ar',
        team_season_raw - pt.mean(team_season_raw, axis=0, keepdims=True),
        dims=('team', 'season'),
    )
    return team_season_ar[team_data, season_data]
 
 
def add_recent_form_ar(player_season_data):
    """Per-player-season 'recent form' offset -- simplified from an
    original within-season AR(1) over `week` after that version's
    form_sigma posterior piled at the zero boundary (essentially no
    week-to-week evolution beyond a persistent per-season shift). Used by
    ar_mod."""
    form_sigma = pm.HalfNormal('form_sigma', 2.0)
    form_z = pm.Normal('form_z', 0, 1, dims='player_season')
    form = pm.Deterministic('recent_form', form_sigma * form_z, dims='player_season')
    return form[player_season_data]
 
 
def add_recent_form_hsgp(week_grid_vals, player_season_data, week_data, m=6, c=1.5):
    """HSGP analogue of add_recent_form_ar: same role (the within-season
    wobble around player_skill's season-level average), different
    generative story -- a smooth nonparametric curve instead of a
    first-order Markov process. Used by hs_version.
 
    Hierarchical/shared-basis HSGP: the eigenbasis (and its lengthscale/
    amplitude hyperparameters) is shared across every player_season -- one
    GP, not thousands of independent ones -- but each player_season gets
    its own partially-pooled coefficients on that shared basis via
    `beta_raw`."""
    ell = pm.InverseGamma('form_ell', alpha=3, beta=6)  # weeks -- a few-week correlation range
    eta = pm.HalfNormal('form_eta', 1.0)
    cov = eta ** 2 * pm.gp.cov.Matern52(1, ls=ell)
    gp = pm.gp.HSGP(m=[m], c=c, cov_func=cov, drop_first=False)
    phi, sqrt_psd = gp.prior_linearized(X=week_grid_vals[:, None])  # phi: (week, m)
 
    beta_raw = pm.Normal('form_beta_raw', 0, 1, dims=('player_season', 'form_basis'))
    form_curve = pt.dot(beta_raw * sqrt_psd[None, :], phi.T)  # (player_season, week)
 
    # row mean via a matmul against a static ones-vector instead of
    # pt.mean(form_curve, axis=1, keepdims=True) -- pt.mean(keepdims=True)'s
    # internal Sum -> Mul(1/n) -> ExpandDims graph is the same
    # FGRAPH.REPLACE/local_add_of_sparse_write trigger family as
    # add_strength_ar/add_player_skill above. n_week_form is a small fixed
    # constant, so this is one cheap matmul.
    n_week_form = len(week_grid_vals)
    row_mean = pt.dot(form_curve, pt.ones((n_week_form, 1))) / n_week_form  # (player_season, 1)
    form = pm.Deterministic(
        'recent_form',
        form_curve - row_mean,
        dims=('player_season', 'week'),
    )
    return form[player_season_data, week_data]
 
 
def add_flat_position_intercept(position_data):
    """Flat per-position baseline scoring level -- no age/tenure slope, no
    correlation across positions. Used by ar_mod."""
    positions_mu = pm.Normal('positions_mu', 6, 3, dims='position')
    return pm.Deterministic('intercepts', positions_mu[position_data])
 
 
def add_lkj_position_intercept(position_data, tenure_z_data, eta=6):
    """Correlated varying effects across position: baseline scoring level
    and a linear tenure/age slope, drawn jointly per position via an
    LKJ-correlated covariance -- McElreath's "varying slopes" cafes
    pattern. Partially pools both quantities across positions and lets the
    model pick up genuine correlation between them (e.g. if
    higher-baseline positions also tend to decline faster/slower). Used by
    hs_version and team_mod.
 
    Centered (not non-centered): non-centering through pos_chol_cov made
    ESS worse across the board. Each position has thousands of
    player-weeks pinning down its (intercept, slope) pair, so this isn't a
    data-starved-group funnel where non-centering helps -- it's stuck the
    other way.
 
    Caller is responsible for creating tenure_z_data (a pm.Data node with
    dims aligned to the model's rows) and passing it in -- this function
    only builds the LKJ structure and the resulting intercepts, matching
    the every-other-block-takes-its-data-as-an-argument convention used
    throughout this module."""
    pos_chol, pos_corr, pos_sigmas = pm.LKJCholeskyCov(
        'pos_chol_cov', n=2, eta=eta,
        sd_dist=pm.Exponential.dist(1.0),
        compute_corr=True,
    )
    intercept = pm.Normal('intercept', 6, 3)
    beta = pm.Normal('beta', 0, 1)
    ab_position = pm.MvNormal(
        'ab_position',
        mu=pt.stack([intercept, beta]),
        chol=pos_chol,
        dims=('position', 'pos_param'),
    )
    pm.Deterministic('positions_mu', ab_position[:, 0], dims='position')
    pm.Deterministic('age_slope', ab_position[:, 1], dims='position')
    return pm.Deterministic(
        'intercepts',
        ab_position[position_data, 0] + ab_position[position_data, 1] * tenure_z_data,
    )
 
 
def add_role_mixture(mu_full, sigma_full, position_data, games_missed_data, snap_pct_data,  target_share_data, rush_share_data, obs_data,):
    """Two-component mixture for the observation layer: when a player
    barely plays, matchup quality is close to irrelevant to how little
    they score. 'full' is the existing linear predictor, unchanged.
    'limited' deliberately doesn't see opp_ar/recent_form/cont_priors.
 
    p_full's logit is driven by two lagged, pre-game role signals instead
    of a flat per-position rate:
      - games_missed_data: has this player recently been absent (injury-
        return risk)
      - snap_pct_data: trailing snap share when they DID play -- catches
        healthy-but-limited-role cases (committee backs, WR4s) that
        games_missed_ytd can't see.
    Both must already be lagged in data prep -- never this week's actual
    snap count, which would leak the outcome into its own predictor. Used
    by every current model (ar_mod, hs_version, team_mod)."""
    mu_limited = pm.HalfNormal('mu_limited', 0.75, dims='position')
    sigma_limited = pm.HalfNormal('sigma_limited', 0.75, dims='position')
 
    role_intercept = pm.Normal('role_intercept', 0, 1.5, dims='position')
    role_beta_missed = pm.Normal('role_beta_missed', 0, 1, dims='position')
    role_beta_snap = pm.Normal('role_beta_snap', 0, 1, dims='position')
    role_beta_target_share = pm.Normal('role_target_share', 0,1, dims = 'position')
    role_beta_rush_share = pm.Normal('role_rush_share', 0, 1, dims = 'position')
 
    logit_p_full = (
        role_intercept[position_data]
        + role_beta_missed[position_data] * games_missed_data
        + role_beta_snap[position_data] * snap_pct_data
        + role_beta_target_share[position_data] * target_share_data
        + role_beta_rush_share[position_data] * rush_share_data
    )
    # NOT wrapped in pm.Deterministic -- this is one value per row (tens of
    # thousands+), and idata files are already large; tracking a full
    # per-row vector every draw would make that meaningfully worse for a
    # quantity that's cheap to recompute post-hoc from role_intercept/
    # role_beta_* if needed.
    p_full = pm.math.sigmoid(logit_p_full)
    w = pt.stack([1 - p_full, p_full], axis=-1)  # (n_obs, 2)
 
    comp_limited = pm.StudentT.dist(nu=4, mu=mu_limited[position_data], sigma=sigma_limited[position_data])
    comp_full = pm.StudentT.dist(nu=4, mu=mu_full, sigma=sigma_full)
 
    return pm.Mixture('y_obs', w=w, comp_dists=[comp_limited, comp_full], observed=obs_data)
 
 
# ============================================================================
# Full model builders
# ============================================================================
# Each of these takes exactly the data/coords a fold (or the full-history
# "single fold") has prepared, and returns a ready-to-sample pm.Model.
# `coords` must already include every dim these models' shared blocks need:
# player, position, team, tenure, opp_team, season, week, player_season,
# cont_vars, binary_vars -- plus 'pos_param', 'global_step',
# 'global_step_full', 'season_gap', and (hs_version/team_mod only)
# 'form_basis', built here from `T`/`boundary_cols`/`n_seasons` so callers
# don't have to duplicate that bookkeeping.
 
def _extend_coords(coords, T, n_seasons, pos_param=False):
    coords = dict(coords)
    coords['global_step'] = list(range(T - 1))
    coords['global_step_full'] = list(range(T))
    coords['season_gap'] = list(range(max(n_seasons - 1, 0)))
    if pos_param:
        coords['pos_param'] = ['positions_mu', 'age_slope']
    return coords
 
 
def build_ar_mod(D, coords, season_offsets, T, boundary_cols,
                  position_of_player, position_group_counts):
    """ar_mod -- flat per-position intercept + opponent-strength AR(1) +
    a flat per-player_season recent-form offset."""
    n_positions = len(coords['position'])
    n_seasons = len(coords['season'])
    coords = _extend_coords(coords, T, n_seasons)
 
    with pm.Model(coords=coords) as model:
        dat = base_data(D, season_offsets)
        intercepts = add_flat_position_intercept(dat['position_data'])
        cont_priors = pm.Normal('cont_priors', 0, 1, dims='cont_vars')
        bi_priors = pm.Normal('bi_priors', 0, 1.5, dims='binary_vars')
 
        player_skill = add_player_skill(coords, position_of_player, position_group_counts, n_positions)
        f_career = player_skill[dat['player_data'], dat['tenure_data']]
        f_opp = add_strength_ar('opp', 'opp_team', dat['opp_team_data'], dat['global_time_data'], T, boundary_cols)
        f_form = add_recent_form_ar(dat['player_season_data'])
 
        mu = intercepts + pm.math.dot(dat['cont_data'], cont_priors) + pm.math.dot(dat['bi_data'], bi_priors) \
            + f_career + f_opp + f_form
 
        sigma_obs = pm.HalfNormal('sigma_obs', 3.0, dims='position')
        add_role_mixture(mu, sigma_obs, dat['position_data'],
                          dat['games_missed_data'], dat['snap_pct_data'],dat['target_share_data'], dat['rush_share_data'] , dat['obs_data'])
    return model
 
 
def build_hs_version(D, coords, season_offsets, T, boundary_cols,
                      position_of_player, position_group_counts, week_grid_vals):
    """hs_version -- LKJ-correlated position intercept + age slope +
    opponent-strength AR(1) + an HSGP within-season recent-form curve.
    Statistically tied with ar_mod on held-out CRPS/RMSE; kept since it's
    not clearly worse and calibrates slightly better."""
    n_positions = len(coords['position'])
    n_seasons = len(coords['season'])
    coords = _extend_coords(coords, T, n_seasons, pos_param=True)
    coords.setdefault('form_basis', list(range(6)))
 
    with pm.Model(coords=coords) as model:
        dat = base_data(D, season_offsets)
        tenure_z_data = pm.Data('tenure_z_data', D['tenure_z'])
        intercepts = add_lkj_position_intercept(dat['position_data'], tenure_z_data)
 
        cont_priors = pm.Normal('cont_priors', 0, 1, dims='cont_vars')
        bi_priors = pm.Normal('bi_priors', 0, 1.5, dims='binary_vars')
 
        player_skill = add_player_skill(coords, position_of_player, position_group_counts, n_positions)
        f_career = player_skill[dat['player_data'], dat['tenure_data']]
        f_opp = add_strength_ar('opp', 'opp_team', dat['opp_team_data'], dat['global_time_data'], T, boundary_cols)
        f_form = add_recent_form_hsgp(week_grid_vals, dat['player_season_data'], dat['week_data'])
 
        mu = intercepts + pm.math.dot(dat['cont_data'], cont_priors) + pm.math.dot(dat['bi_data'], bi_priors) \
            + f_career + f_opp + f_form
 
        sigma_obs = pm.HalfNormal('sigma_obs', 2.5, dims='position')
        add_role_mixture(mu, sigma_obs, dat['position_data'],
                          dat['games_missed_data'], dat['snap_pct_data'],dat['target_share_data'], dat['rush_share_data'] , dat['obs_data'])
    return model
 
 
def build_team_mod(D, coords, season_offsets, T, boundary_cols,
                    position_of_player, position_group_counts):
    """team_mod -- hs_version's LKJ intercept + opponent-strength AR(1)
    (no recent-form term) + two regression coefficients (beta_team_form,
    beta_opp_team_form) on a precomputed team-strength feature (see
    src/estimate-latent-ability.py). Own-team and opponent-team strength
    are two separate coefficients, not a single differenced term -- there
    's no reason to assume a player's own offense helping them and the
    opponent's defense hurting them are equal-and-opposite effects on
    THEIR fantasy points specifically.
 
    Requires D['team_form_z'] / D['opp_team_form_z']."""
    n_positions = len(coords['position'])
    n_seasons = len(coords['season'])
    coords = _extend_coords(coords, T, n_seasons, pos_param=True)
 
    with pm.Model(coords=coords) as model:
        dat = base_data(D, season_offsets)
        tenure_z_data = pm.Data('tenure_z_data', D['tenure_z'])
        intercepts = add_lkj_position_intercept(dat['position_data'], tenure_z_data, eta=4)
 
        cont_priors = pm.Normal('cont_priors', 0, 1, dims='cont_vars')
        bi_priors = pm.Normal('bi_priors', 0, 1.5, dims='binary_vars')
 
        player_skill = add_player_skill(coords, position_of_player, position_group_counts, n_positions)
        f_career = player_skill[dat['player_data'], dat['tenure_data']]
        #f_opp = add_strength_ar('opp', 'opp_team', dat['opp_team_data'], dat['global_time_data'], T, boundary_cols)
 
        team_form_data = pm.Data('team_form_data', D['team_form_z'])
        opp_team_form_data = pm.Data('opp_team_form_data', D['opp_team_form_z'])
        beta_team_form = pm.Normal('beta_team_form', 0, 1)
        beta_opp_team_form = pm.Normal('beta_opp_team_form', 0, 1)
        f_team_form = beta_team_form * team_form_data + beta_opp_team_form * opp_team_form_data
 
        mu = intercepts + pm.math.dot(dat['cont_data'], cont_priors) + pm.math.dot(dat['bi_data'], bi_priors) \
            + f_career  + f_team_form
 
        sigma_obs = pm.HalfNormal('sigma_obs', 0.75)
        add_role_mixture(mu, sigma_obs, dat['position_data'],
                          dat['games_missed_data'], dat['snap_pct_data'],dat['target_share_data'], dat['rush_share_data'] , dat['obs_data'])
    return model
 
 
def build_r2d2_mod(D, coords, season_offsets, T, boundary_cols,
                    position_of_player, position_group_counts):
    """r2d2 -- same features/structure as team_mod (LKJ intercept +
    opponent-strength AR(1) + team_form/opp_team_form regression), but with
    a more defensible/empirically-informed prior structure:
 
    - linear_effects (the 11 continuous + 5 binary covariates) and
      team_form_effects (beta_team_form/beta_opp_team_form) go through
      pmx.R2D2M2CP, which puts a prior directly on variance-explained (r2)
      for that block and decomposes it across the block's coefficients via
      an internal Dirichlet. Both are genuinely linear-in-coefficients
      blocks, which is exactly what R2D2M2CP is for.
    - career (player_skill's random walk) and opp (opp_team's AR(1)) are
      NOT linear-coefficient blocks -- they're structurally different
      (a random walk, an AR(1) process), and an earlier attempt to fold
      them into R2D2M2CP's shared-Dirichlet-variance-budget alongside
      linear/team_form produced a real, confirmed ridge (r_hat 1.03-1.07
      on R2/block_share, resolved via az.plot_pair showing a genuine
      negative-correlation ridge between block_share components -- a
      multiplicative tie between two simplex/positive-valued parents).
      Instead they get independent, informatively-centered Gamma total-
      scale priors, split internally into their two sub-components
      (init_sigma/rw_sigma for career, sigma_theta/sigma_season for opp)
      via add_player_skill/add_strength_ar's total_scale= argument. The
      Gamma centers below (career~2.6, opp~1.8) are the values those two
      terms converged to cleanly when *not* forced to share a budget with
      anything else.
 
    r2/r2_std for the two R2D2M2CP blocks are set from the baseline
    model's pooled backtest R^2 (~0.30-0.37 across folds; see
    results/backtest_walkforward.csv) as a pilot-data-informed prior on
    how much of the outcome variance a purely-linear signal plausibly
    explains -- not a claim that this exact fold will hit that number.
 
    Requires D['team_form_z'] / D['opp_team_form_z'], same as team_mod."""
    n_positions = len(coords['position'])
    n_seasons = len(coords['season'])
    coords = _extend_coords(coords, T, n_seasons, pos_param=True)
    coords.setdefault('linear_vars', coords['cont_vars'] + coords['binary_vars'])
    coords.setdefault('team_form_vars', ['team_form', 'opp_team_form'])
 
    with pm.Model(coords=coords) as model:
        dat = base_data(D, season_offsets)
        tenure_z_data = pm.Data('tenure_z_data', D['tenure_z'])
        intercepts = add_lkj_position_intercept(dat['position_data'], tenure_z_data, eta=4)
 
        # -- linear_effects: R2D2M2CP over the 11 cont + 5 binary covariates
        design = pt.concatenate([dat['cont_data'], dat['bi_data']], axis=1)
        input_sigma = np.concatenate([
            np.asarray(D['cont']).std(axis=0),
            np.asarray(D['bi']).std(axis=0),
        ])
        # floor any zero-variance column to 1.0 instead of feeding
        # R2D2M2CP a literal 0 -- same convention as fit_scalers' sd
        # flooring. A windowed backtest fold can genuinely have a constant
        # binary_vars column (e.g. 'era' is often a single value within a
        # short WINDOW_SEASONS slice) even though it's never constant over
        # full history; R2D2M2CP's internal decomposition divides by
        # input_sigma, so a literal 0 here produces inf/nan at every
        # initialization point (nutpie's "All initialization points
        # failed" / ErrorCode(3)). A column that's constant within this
        # fold carries zero within-fold signal either way -- R2D2M2CP has
        # nothing to identify that coefficient from regardless of what
        # input_sigma says, so flooring it to 1.0 just avoids the
        # division-by-zero without changing what the model can actually
        # learn from that column in this fold.
        input_sigma = np.where(input_sigma > 0, input_sigma, 1.0)
        linear_scale = pm.Gamma('linear_scale', alpha=2.0, sigma=1.0)
        _, linear_beta = pmx.R2D2M2CP(
            'linear_effects', output_sigma=linear_scale, input_sigma=input_sigma,
            dims='linear_vars', r2=0.35, r2_std=0.08,
        )
        f_linear = pt.dot(design, linear_beta)
 
        # -- career: player_skill random walk, independent Gamma scale
        career_scale = pm.Gamma('career_scale', alpha =2.0, beta=1.0)
        player_skill = add_player_skill(coords, position_of_player, position_group_counts,
                                         n_positions, total_scale=career_scale)
        f_career = player_skill[dat['player_data'], dat['tenure_data']]
 
        # -- opp: opponent-strength AR(1), independent Gamma scale
        opp_scale = pm.Gamma('opp_scale', mu=1.8, sigma=1.0)
        f_opp = add_strength_ar('opp', 'opp_team', dat['opp_team_data'], dat['global_time_data'],
                                 T, boundary_cols, total_scale=opp_scale)
 
        # -- team_form_effects: R2D2M2CP over [team_form, opp_team_form]
        team_form_data = pm.Data('team_form_data', D['team_form_z'])
        opp_team_form_data = pm.Data('opp_team_form_data', D['opp_team_form_z'])
        tf_design = pt.stack([team_form_data, opp_team_form_data], axis=-1)
        team_form_scale = pm.Gamma('team_form_scale', mu=1.8, sigma=1.0)
        _, team_form_beta = pmx.R2D2M2CP(
            'team_form_effects', output_sigma=team_form_scale, input_sigma=np.array([1.0, 1.0]),
            dims='team_form_vars', r2=0.35, r2_std=0.08,
        )
        f_team_form = pt.dot(tf_design, team_form_beta)
 
        mu = intercepts + f_linear + f_career + f_opp + f_team_form
 
        sigma_obs = pm.HalfNormal('sigma_obs', 2.5)
        add_role_mixture(mu, sigma_obs, dat['position_data'],
                          dat['games_missed_data'], dat['snap_pct_data'],dat['target_share_data'], dat['rush_share_data'] , dat['obs_data'])
    return model
 
 
def build_bart_mod(D, coords, position_of_player, position_group_counts, m=50):
    """bart_mod3 -- ported from src/bart_mod.py's third (most-refined) BART
    sandbox variant. Structurally much simpler than ar_mod/hs_version/
    team_mod/r2d2_mod: no opp_ar weekly AR timeline, no
    add_recent_form_ar/hsgp, no add_role_mixture, no LKJ position
    intercept -- just this fold's RAW (unstandardized) continuous +
    binary covariates fed straight into a pymc_bart.BART regression-tree
    ensemble, plus the SAME add_player_skill random-walk-over-tenure term
    every other architecture uses for the player's own career trajectory.
    Raw rather than z-scored on purpose: BART splits on a feature's own
    value and doesn't need or benefit from scaling the way every other
    model's cont_priors dot-product does. The covariate list
    (backtest_harness.BART_CONT_VARS) also includes team_form/
    opp_team_form directly as raw features -- unlike ar_mod/hs_version's
    STD_COLS, which deliberately excludes them (see backtest_harness.py's
    module docstring) so ar_mod/hs_version stay a clean ablation against
    team_mod's separate beta_team_form/beta_opp_team_form coefficients.
    BART has no such ablation concern: it can use or ignore a raw feature
    however the trees find useful, so there's no reason to withhold it.
    StudentT(nu=6) likelihood (fat tails instead of Normal -- fantasy
    scores have real outlier games; matches bart_mod3's original choice).
 
    D must carry 'bart_cont' (raw (n, n_bart_cont) array -- see
    backtest_harness.Fold.arrays), 'player', 'tenure', 'y'.
 
    Prediction on new (held-out) rows goes through the ordinary
    pm.set_data + pm.sample_posterior_predictive path every other build_*
    function's caller uses (backtest_harness.fit_predict_bart) --
    verified empirically against this project's pinned pymc-bart that
    swapping 'bart_cont_data' really does re-evaluate the fitted trees at
    the new X, not just at whatever rows it was trained on (pymc-bart's
    BART op lazily re-.eval()s its captured X reference, which tracks
    pm.set_data's in-place update of the underlying shared variable).
    ADVI is NOT valid for this model -- see fit_predict_bart's docstring."""
    n_positions = len(coords['position'])
    with pm.Model(coords=coords) as model:
        cont_data = pm.Data('bart_cont_data', np.ascontiguousarray(D['bart_cont']))
        player_data = pm.Data('player_data', D['player'])
        tenure_data = pm.Data('tenure_data', D['tenure'])
        obs_data = pm.Data('obs_data', D['y'])
 
        bart_mu = pmb.BART('bart_mu', X=cont_data, Y=D['y'], m=m)
 
        player_skill = add_player_skill(
            coords, position_of_player, position_group_counts,
            n_positions, total_scale=None,
        )
        f_career = pm.Deterministic('f_career', player_skill[player_data, tenure_data])
 
        mu = f_career + bart_mu
        sigma_obs = pm.HalfNormal('sigma_obs', sigma=2.5)
        pm.StudentT('y_obs', mu=mu, sigma=sigma_obs, nu=6, observed=obs_data)
    return model
 

# ============================================================================
# het_gp / het_gp_qb -- heteroskedastic latent-GP models, one position group
# per fit. Ported from src/ff_scratchpad_skill.py and src/ff_scratchpad_qb.py
# (the versions whose prior-sensitivity checks came back clean).
#
#   mu = alpha[pos] + f_career[age_or_tenure, pos]          aging bump(s)
#      + skill[player] + f_season[player, age_or_tenure]    career level + season deviation
#      + cont @ b + bi @ b + team form + opp form           context
#   skill : log_sigma = log_sigma0[pos] + g_noise[tenure]   (noise GP)
#   qb    : sigma = sigma_obs                               (single scale)
#   y_z ~ Normal(mu, sigma)          y standardized with the fit frame's mean/sd
#
# Everything FIT on data (scalers, binary means, team-form scaler, y
# standardization, centering weights, coords) lives in HetPrep, fit on ONE
# training frame (a fold's windowed train in backtest_harness.py) and then
# applied unchanged to any other frame (the test week).
# ============================================================================
from dataclasses import dataclass


@dataclass(frozen=True)
class HetConfig:
    positions: tuple
    std_cols: tuple              # globally standardized covariates
    time_col: str                # 'tenure' (skill) or 'age' (QB): input to the bump and season GP
    asymmetric_bump: bool        # separate rise / decline widths (QB)
    noise_gp: bool               # tenure GP on log sigma (skill) vs single sigma (QB)
    t_peak: tuple                # Normal(mean, sd) on the peak
    season_ell: tuple            # maxent InverseGamma range for ell_season
    noise_ell: tuple = (4.0, 8.0)


# same-week rush_share leaks the week being predicted -> lag_rush_share (as in STD_COLS)
HET_SKILL_CONFIG = HetConfig(
    positions=('RB', 'TE', 'WR'),
    std_cols=('spread_line', 'roll_avg_points_allowed', 'lag_yards_per_rush_attempt',
              'lag_yards_per_target', 'lag_avg_depth_of_target', 'lag_target_share',
              'lag_rush_share', 'temp', 'wind', 'games_missed_ytd', 'lag_offense_pct', 'lag_st_pct'),
    time_col='tenure', asymmetric_bump=False, noise_gp=True,
    t_peak=(4.0, 1.25), season_ell=(1.5, 3.0), noise_ell=(4.0, 8.0),
)
HET_QB_CONFIG = HetConfig(
    positions=('QB',),
    std_cols=('spread_line', 'roll_avg_points_allowed', 'lag_yards_per_rush_attempt',
              'lag_yards_per_pass_attempt', 'lag_avg_depth_of_target', 'lag_rush_share',
              'temp', 'wind', 'games_missed_ytd', 'lag_offense_pct'),
    time_col='age', asymmetric_bump=True, noise_gp=False,
    t_peak=(28.5, 3.0), season_ell=(1.0, 4.0),
)
HET_CONFIGS = {'skill': HET_SKILL_CONFIG, 'qb': HET_QB_CONFIG}


def primary_positions(frame):
    """player_id -> the position he's listed at most often (ties alphabetical)."""
    return (
        frame.group_by('player_id', 'position').agg(pl.len().alias('_n'))
        .sort(['player_id', '_n', 'position'], descending=[False, True, False])
        .group_by('player_id', maintain_order=True).first()
        .select('player_id', pl.col('position').alias('_primary_position'))
    )


def with_primary_position(frame):
    """Overwrite each row's position with the player's primary position."""
    return (frame.join(primary_positions(frame), on='player_id', how='left')
            .with_columns(pl.col('_primary_position').alias('position'))
            .drop('_primary_position'))


def _inverse_gamma_maxent(lower, upper):
    import preliz as pz
    d = pz.InverseGamma()
    pz.maxent(d, lower=lower, upper=upper, plot=False)
    return float(d.alpha), float(d.beta)


class HetPrep:
    """Everything het_gp fits on data, fit once on `train` for one HetConfig.
    arrays(frame) turns any frame into build_het_gp's D using train-fit
    quantities only. Rows whose player isn't in train must be dropped first."""

    def __init__(self, train, cfg):
        self.cfg = cfg
        train = train.filter(pl.col('position').is_in(list(cfg.positions)))
        tcol = cfg.time_col
        self.players = train['player_id'].unique().sort().to_list()
        self.positions = list(cfg.positions)
        t_min, t_max = int(train[tcol].min()), int(train[tcol].max())
        self.times = list(range(t_min, t_max + 2))              # contiguous + next-season slot
        self._player_map = {p: i for i, p in enumerate(self.players)}
        self._pos_map = {p: i for i, p in enumerate(self.positions)}
        self._time_map = {t: i for i, t in enumerate(self.times)}

        self.scalers = fit_scalers(train, list(cfg.std_cols))
        self.bi_means = {c: float(train[c].mean()) for c in BINARY_VARS}
        self.tf_mu = float(train['team_form'].mean())
        self.tf_sd = float(train['team_form'].std() or 1.0)
        y = train['total_fantasy_points'].to_numpy()
        self.y_mu, self.y_sd = float(y.mean()), float(y.std())

        D = self.arrays(train)
        n_pl, n_pos, n_t = len(self.players), len(self.positions), len(self.times)
        self.position_of_player = np.zeros(n_pl, dtype=int)
        self.position_of_player[D['player']] = D['position']
        self.pos_onehot = np.eye(n_pos)[self.position_of_player]
        self.pos_counts = np.maximum(self.pos_onehot.sum(0), 1.0)

        counts = np.zeros((n_t, n_pos))
        np.add.at(counts, (D['time'], D['position']), 1)
        self.t_weights_pos = counts / np.maximum(counts.sum(0, keepdims=True), 1.0)
        self.t_weights = counts.sum(1) / counts.sum()

        self.active = np.zeros((n_pl, n_t))
        self.active[D['player'], D['time']] = 1.0
        self.n_active = np.maximum(self.active.sum(0), 1.0)
        self.n_seasons_player = np.maximum(self.active.sum(1), 1.0)

        self.season_prior = _inverse_gamma_maxent(*cfg.season_ell)
        self.noise_prior = _inverse_gamma_maxent(*cfg.noise_ell)
        self.coords = {'position': self.positions, 'player_id': self.players, 'time': self.times,
                       'cont_vars': [f'{c}_z' for c in cfg.std_cols], 'binary_vars': BINARY_VARS}
        self.grid = np.asarray(self.times, dtype='float64')

    def arrays(self, frame):
        n = len(frame)
        cont = (np.column_stack([((frame[c].cast(pl.Float64).fill_null(mu) - mu) / sd).to_numpy()
                                 for c, (mu, sd) in self.scalers.items()])
                if self.scalers else np.zeros((n, 0)))
        return {
            'player': frame['player_id'].replace_strict(self._player_map, return_dtype=pl.Int64).to_numpy(),
            'position': frame['position'].replace_strict(self._pos_map, return_dtype=pl.Int64).to_numpy(),
            'time': frame[self.cfg.time_col].cast(pl.Int64)
                    .replace_strict(self._time_map, return_dtype=pl.Int64).to_numpy(),
            'cont': np.ascontiguousarray(cont.astype('float64')),
            'bi': np.ascontiguousarray(np.column_stack(
                [frame[c].cast(pl.Float64).to_numpy() - self.bi_means[c] for c in BINARY_VARS])),
            'team_form_z': ((frame['team_form'] - self.tf_mu) / self.tf_sd).to_numpy(),
            'opp_team_form_z': ((frame['opp_team_form'] - self.tf_mu) / self.tf_sd).to_numpy(),
            'y': frame['total_fantasy_points'].to_numpy().astype('float64'),
            'y_z': ((frame['total_fantasy_points'] - self.y_mu) / self.y_sd).to_numpy(),
        }


def het_payload(D):
    """{pm_data_name: array} for pm.set_data() on build_het_gp's model."""
    return {
        'player_data': D['player'], 'position_data': D['position'], 'time_data': D['time'],
        'cont_data': D['cont'], 'bi_data': D['bi'],
        'team_form_data': D['team_form_z'], 'opp_team_form_data': D['opp_team_form_z'],
        'obs_data': np.zeros(len(D['player'])),
    }


def build_het_gp(D, prep, jitter=1e-6):
    """het_gp on STANDARDIZED points. Predictions * prep.y_sd + prep.y_mu = points."""
    cfg = prep.cfg
    grid = prep.grid
    X = grid[:, None]
    multi = len(prep.positions) > 1
    with pm.Model(coords=prep.coords) as model:
        player = pm.Data('player_data', D['player'])
        pos = pm.Data('position_data', D['position'])
        t = pm.Data('time_data', D['time'])
        cont = pm.Data('cont_data', D['cont'])
        bi = pm.Data('bi_data', D['bi'])
        tf = pm.Data('team_form_data', D['team_form_z'])
        otf = pm.Data('opp_team_form_data', D['opp_team_form_z'])
        obs = pm.Data('obs_data', D['y_z'])

        alpha = pm.Normal('alpha_pos', 0.0, 0.5, dims='position')

        # 1. aging bump(s)
        if multi:
            t_peak_mu = pm.Normal('t_peak_mu', *cfg.t_peak)
            t_peak_off = pm.ZeroSumNormal('t_peak_off', sigma=1.0, dims='position')
            t_peak = pm.Deterministic('t_peak', t_peak_mu + t_peak_off, dims='position')
            log_height_mu = pm.Normal('log_height_mu', np.log(0.5), 0.4)
            log_height_off = pm.ZeroSumNormal('log_height_off', sigma=0.25, dims='position')
            peak_height = pm.Deterministic('peak_height', pt.exp(log_height_mu + log_height_off), dims='position')
        else:
            t_peak = pm.Normal('t_peak', *cfg.t_peak)[None]
            log_height = pm.Normal('log_height', np.log(0.5), 0.4)
            peak_height = pm.Deterministic('peak_height', pt.exp(log_height))[None]
        if cfg.asymmetric_bump:
            width_rise = pm.Deterministic('width_rise', 1.0 + pm.Gamma('width_rise_raw', alpha=4, beta=1.5))
            width_decline = pm.Deterministic('width_decline', 1.0 + pm.Gamma('width_decline_raw', alpha=8, beta=1.2))
            width = pt.switch(grid[:, None] < t_peak[None, :], width_rise, width_decline)
        else:
            width = pm.Deterministic('peak_width', 1.0 + pm.Gamma('peak_width_raw', alpha=6, beta=1.5))
        curve = peak_height[None, :] * pt.exp(-0.5 * ((grid[:, None] - t_peak[None, :]) / width) ** 2)
        f_career = pm.Deterministic('f_career', curve - (prep.t_weights_pos * curve).sum(0),
                                    dims=('time', 'position'))

        # 2. skill: career level, mean-zero within position
        sigma_skill = pm.HalfNormal('sigma_skill', 1.0)
        if multi:
            z_skill = pm.Normal('z_skill', 0.0, 1.0, dims='player_id')
            raw = sigma_skill * z_skill
            skill = pm.Deterministic(
                'skill', raw - ((prep.pos_onehot.T @ raw) / prep.pos_counts)[prep.position_of_player],
                dims='player_id')
        else:
            z_skill = pm.ZeroSumNormal('z_skill', sigma=1.0, dims='player_id')
            skill = pm.Deterministic('skill', sigma_skill * z_skill, dims='player_id')

        # 3. season level: per-player GP, centered within player and per time cell
        ell_season = pm.InverseGamma('ell_season', *prep.season_prior)
        eta_season = pm.HalfNormal('eta_season', 0.5)
        L_season = pt.linalg.cholesky(pm.gp.util.stabilize(
            (eta_season**2 * pm.gp.cov.Matern32(1, ls=ell_season))(X), jitter))
        z_season = pm.Normal('z_season', 0.0, 1.0, dims=('player_id', 'time'))
        f_raw = z_season @ L_season.T
        f_p = f_raw - ((f_raw * prep.active).sum(1) / prep.n_seasons_player)[:, None]
        f_season = f_p - (f_p * prep.active).sum(0) / prep.n_active

        # 4. covariates
        cont_prior = pm.Normal('cont_prior', 0.0, 0.5, dims='cont_vars')
        bi_prior = pm.Normal('bi_prior', 0.0, 0.5, dims='binary_vars')
        team_form_prior = pm.Normal('team_form_prior', 0.0, 0.5)
        opp_form_prior = pm.Normal('opp_form_prior', 0.0, 0.5)
        mu = (alpha[pos] + f_career[t, pos] + skill[player] + f_season[player, t]
              + pt.dot(cont, cont_prior) + pt.dot(bi, bi_prior)
              + team_form_prior * tf + opp_form_prior * otf)

        # 5. observation scale
        if cfg.noise_gp:
            log_sigma0 = pm.Normal('log_sigma0', 0.0, 0.5, dims='position')
            ell_noise = pm.InverseGamma('ell_noise', *prep.noise_prior)
            eta_noise = pm.HalfNormal('eta_noise', 0.15)
            gp_noise = pm.gp.Latent(cov_func=eta_noise**2 * pm.gp.cov.Matern52(1, ls=ell_noise))
            g_raw = gp_noise.prior('g_noise_raw', X=X, jitter=jitter, dims='time')
            g_noise = pm.Deterministic('g_noise', g_raw - pt.dot(prep.t_weights, g_raw), dims='time')
            sigma = pt.exp(log_sigma0[pos] + g_noise[t])
        else:
            sigma = pm.HalfNormal('sigma_obs', 1.0)
        pm.Normal('y_obs', mu=mu, sigma=sigma, observed=obs)
    return model
