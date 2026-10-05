"""
Shared data loading / feature engineering for the fantasy-predictor Bayesian models.

This is the same polars pipeline used at the top of hierarchical-model.py
(load -> add era/player_season -> add lags -> train/test split on the last
observed week -> fit scalers on train -> encode categoricals -> build coords),
factored out so the alternative model scripts (dynamic-skill-model.py,
hsgp-model.py, bart-model.py) can all build directly comparable train/test
frames and coords without re-deriving the pipeline three times.

hierarchical-model.py itself is left untouched -- this module doesn't change
its behavior, it just gives the new scripts something to import.
"""

import polars as pl
import polars.selectors as cs

SEED = 41029041

# The raw feed uses inconsistent franchise codes across the 2006-2025 span:
# three real relocations (Chargers, Rams, Raiders) plus four teams where the
# source just changed its abbreviation convention in 2016 with no relocation
# involved. Left alone, these count as 39 distinct "teams" for encode()/coords
# instead of the real 32 -- each aliased franchise's history gets split into
# two categories that never share partial pooling, and the ZeroSumNormal team
# effect ends up spread across 7 too many buckets.
TEAM_ALIASES = {
    "SD": "LAC",   # San Diego -> LA Chargers (2017 relocation)
    "SL": "LA",    # St. Louis -> LA Rams (2016 relocation)
    "OAK": "LV",   # Oakland -> Las Vegas Raiders (2020 relocation)
    "ARZ": "ARI",  # abbreviation convention change only, no relocation
    "BLT": "BAL",
    "CLV": "CLE",
    "HST": "HOU",
}

# Continuous covariates, standardized to z-scores on train stats.
STD_COLS = [
    'spread_line', 'roll_avg_points_allowed', 'lag_yards_per_rush_attempt',
    'lag_target_share', 'lag_yards_per_target', 'lag_yards_per_pass_attempt',
    'temp', 'wind',
]

# Binary/indicator covariates, used as-is (0/1).
BINARY_COLS = ['is_grass', 'is_indoors', 'is_home', 'div_game', 'era']

# Categorical columns that get an integer index + a pymc coord.
IDX_COLS = ['player', 'position', 'team', 'week', 'tenure', 'player_season']


def load_raw(path='processed-data/ff-processed.parquet'):
    return (
        pl.read_parquet(path)
        .fill_nan(0)
        .with_columns(
            pl.col('team').replace(TEAM_ALIASES),
            pl.when(pl.col('season') >= 2018).then(1).otherwise(0).alias('era'),
            pl.concat_str(
                [pl.col('player_id'), pl.lit('_'), pl.col('season')]
            ).alias('player_season'),
        )
        .rename({'full_name': 'player'})
    )


def add_lags(df):
    return (
        df.with_columns(
            pl.col(
                'yards_per_rush_attempt', 'target_share',
                'yards_per_target', 'yards_per_pass_attempt',
            ).shift(1).name.prefix('lag_')
        )
        .fill_null(0)
    )


def fit_scalers(df, cols):
    stats = df.select(
        [pl.col(c).mean().alias(f"{c}__mu") for c in cols] +
        [pl.col(c).std().alias(f"{c}__sd") for c in cols]
    ).row(0, named=True)
    return {c: (stats[f"{c}__mu"], stats[f"{c}__sd"]) for c in cols}


def apply_scalers(frame, scalers):
    return frame.with_columns([
        ((pl.col(c) - mu) / sd).alias(f"{c}_z") for c, (mu, sd) in scalers.items()
    ])


def make_indices(df, cols):
    """Return {col: Enum dtype} fit over the FULL frame (train+test) so
    test-only categories never blow up encode() with nulls."""
    return {
        col: pl.Enum(df[col].unique().sort().cast(pl.String).to_list())
        for col in cols
    }


def encode(df, cols, enums):
    return df.with_columns([
        pl.col(col).cast(pl.String).cast(enums[col]).to_physical().cast(pl.Int32).alias(f"{col}_idx")
        for col in cols
    ])


def prepare(path='processed-data/ff-processed.parquet', last_season=2025,
            std_cols=STD_COLS, binary_cols=BINARY_COLS, idx_cols=IDX_COLS):
    """Load, engineer, split, scale, and encode -- mirrors the exact split
    used in hierarchical-model.py (train = everything before the last
    observed week of `last_season`; test = that last week) so any alternative
    model here is scored on the same held-out slice.

    Returns
    -------
    train, test : pl.DataFrame
    coords      : dict for pm.Model(coords=...), includes 'cont_vars' /
                  'binary_vars' for the covariate design matrices.
    scalers     : dict[col] -> (mu, sd), fit on train only.
    idxs        : dict[col] -> pl.Enum, fit on the full frame.
    """
    raw = add_lags(load_raw(path))

    lw = raw.filter(pl.col('season') == last_season)['week'].max()
    train = raw.filter(
        (pl.col('season') < last_season) |
        ((pl.col('season') == last_season) & (pl.col('week') < lw))
    )
    test = raw.filter((pl.col('season') == last_season) & (pl.col('week') == lw))

    scalers = fit_scalers(train, std_cols)
    idxs = make_indices(raw, idx_cols)

    train = encode(apply_scalers(train, scalers), idx_cols, idxs)
    test = encode(apply_scalers(test, scalers), idx_cols, idxs)

    coords = {col: enum.categories.to_list() for col, enum in idxs.items()}

    cont_train = train.select(cs.ends_with('_z'))
    coords['cont_vars'] = cont_train.columns
    coords['binary_vars'] = list(binary_cols)

    return train, test, coords, scalers, idxs


def cont_binary_arrays(frame, coords, binary_cols=BINARY_COLS):
    """Design matrices matching coords['cont_vars'] / coords['binary_vars']."""
    cont = frame.select(coords['cont_vars']).to_numpy()
    bi = frame.select(binary_cols).to_numpy().astype(float)
    return cont, bi
