"""Loading, validating and joining the processed player-week data.

Everything here is pure polars on in-memory frames, except the two
``load_*`` functions, which read parquet.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

# pyrefly: ignore [missing-import]
from .config import PATHS, TARGET_COL, VEGAS_COLS, WINDOW_SEASONS, HetConfig

log = logging.getLogger(__name__)

KEY = ('player_id', 'season', 'week')

# Columns every processed frame must carry for the production models.
REQUIRED_PROCESSED_COLS = (
    'player_id', 'player', 'position', 'team', 'opp_team', 'season', 'week', 'game_id',
    TARGET_COL, 'tenure', 'age', 'spread_line', 'total_line', 'is_home', 'is_grass',
    'is_indoors', 'div_game', 'era', 'temp', 'wind', 'roll_avg_points_allowed', 'points_allowed',
    'games_missed_ytd', 'lag_yards_per_rush_attempt', 'lag_yards_per_target',
    'lag_yards_per_pass_attempt', 'lag_avg_depth_of_target', 'lag_target_share',
    'lag_rush_share', 'lag_offense_pct', 'lag_st_pct',
)


class DataError(ValueError):
    """Raised when an input frame doesn't satisfy what the models need."""


def require_columns(df: pl.DataFrame, cols, what: str = 'frame') -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise DataError(f'{what} is missing required columns: {missing}')


# ------------------------------------------------------------------ loading

def load_processed(path: Path | str = PATHS.processed, dedupe: bool = True) -> pl.DataFrame:
    """Read the processed player-week parquet, check its schema, and drop
    exact duplicate (player_id, season, week) rows.

    The current file has 6 such duplicates (2013 and 2015, a snap-count join
    fan-out in the data build); a duplicated row would be fit twice."""
    df = pl.read_parquet(path)
    require_columns(df, REQUIRED_PROCESSED_COLS, f'processed data ({path})')
    df = df.with_columns(pl.col('week').cast(pl.Float64), pl.col('season').cast(pl.Int32))
    if dedupe:
        n0 = df.height
        df = df.unique(subset=list(KEY), keep='first', maintain_order=True)
        if df.height != n0:
            log.info('dropped %d duplicate (player_id, season, week) rows', n0 - df.height)
    return add_vegas_features(df)


def load_team_form(path: Path | str = PATHS.team_form) -> pl.DataFrame:
    tf = pl.read_parquet(path)
    require_columns(tf, ('season', 'week', 'team', 'team_form'), f'team form ({path})')
    return tf


# ----------------------------------------------------------------- features

def add_vegas_features(df: pl.DataFrame) -> pl.DataFrame:
    """Signed Vegas lines from the player's team's side.

    ``spread_line`` / ``total_line`` are game-level and home-perspective
    (positive = home favoured), so as one linear feature their effect cancels
    between home and away rows. team_spread > 0 means the player's team is
    favoured; vegas_team_total is the implied points for the player's team."""
    return df.with_columns(
        (pl.col('spread_line') * (2 * pl.col('is_home') - 1)).alias('team_spread'),
    ).with_columns(
        (pl.col('total_line') / 2 + pl.col('team_spread') / 2).alias('vegas_team_total'),
    )


def join_team_form(df: pl.DataFrame, team_form: pl.DataFrame) -> pl.DataFrame:
    """Add ``team_form`` / ``opp_team_form``: each team's form going INTO the
    week, i.e. its latest team_form value from strictly before (season, week).

    team_form.parquet holds the Kalman *filtered* state, which includes that
    week's own result, so an exact join would leak the game being predicted.
    The as-of join also handles byes, carries last season's final form into
    week 1, and works for future weeks not yet in team_form. Teams with no
    earlier form get 0.0 (the prior mean)."""
    dupes = team_form.height - team_form.select('season', 'week', 'team').unique().height
    if dupes:
        raise DataError(f'team_form has {dupes} duplicate (season, week, team) keys')
    t = (pl.col('season').cast(pl.Float64) * 100 + pl.col('week').cast(pl.Float64)).alias('_t')
    tf = team_form.select('team', t, pl.col('team_form').cast(pl.Float64)).sort('_t')
    n0 = df.height
    out = (df.drop([c for c in ('team_form', 'opp_team_form') if c in df.columns])
             .with_row_index('_rid').with_columns(t).sort('_t')
             .join_asof(tf, on='_t', by='team', strategy='backward', allow_exact_matches=False,
                        check_sortedness=False)
             .join_asof(tf.rename({'team': 'opp_team', 'team_form': 'opp_team_form'}),
                        on='_t', by='opp_team', strategy='backward', allow_exact_matches=False,
                        check_sortedness=False)
             .sort('_rid').drop('_rid', '_t'))
    if out.height != n0:
        raise DataError(f'team_form join changed row count: {n0} -> {out.height}')
    return out.with_columns(pl.col('team_form', 'opp_team_form').fill_null(0.0))


def primary_positions(frame: pl.DataFrame) -> pl.DataFrame:
    """player_id -> the position he's listed at most often (ties alphabetical)."""
    return (
        frame.group_by('player_id', 'position').agg(pl.len().alias('_n'))
        .sort(['player_id', '_n', 'position'], descending=[False, True, False])
        .group_by('player_id', maintain_order=True).first()
        .select('player_id', pl.col('position').alias('_primary_position'))
    )


def with_primary_position(frame: pl.DataFrame, reference: pl.DataFrame | None = None) -> pl.DataFrame:
    """Overwrite each row's position with the player's primary position,
    computed from ``reference`` (default: ``frame`` itself). Players absent
    from ``reference`` keep their listed position."""
    ref = primary_positions(frame if reference is None else reference)
    return (frame.join(ref, on='player_id', how='left')
            .with_columns(pl.coalesce('_primary_position', 'position').alias('position'))
            .drop('_primary_position'))


# ------------------------------------------------------------------ windows

def before(season: int, week: float) -> pl.Expr:
    """Rows strictly before (season, week)."""
    return (pl.col('season') < season) | ((pl.col('season') == season) & (pl.col('week') < week))


def training_window(df: pl.DataFrame, season: int, week: float,
                    window_seasons: int = WINDOW_SEASONS) -> pl.DataFrame:
    """Rows from the last ``window_seasons`` seasons (target included),
    strictly before the target week."""
    return df.filter((pl.col('season') >= season - (window_seasons - 1)) & before(season, week))


def week_rows(df: pl.DataFrame, season: int, week: float) -> pl.DataFrame:
    return df.filter((pl.col('season') == season) & (pl.col('week') == week))


def model_frame(df: pl.DataFrame, cfg: HetConfig) -> pl.DataFrame:
    """Rows a model covers (by position), with the columns it needs checked."""
    require_columns(df, cfg.required_cols, f'{cfg.name} input')
    return df.filter(pl.col('position').is_in(list(cfg.positions)))


__all__ = [
    'KEY',
    'REQUIRED_PROCESSED_COLS',
    'VEGAS_COLS',
    'DataError',
    'add_vegas_features',
    'before',
    'join_team_form',
    'load_processed',
    'load_team_form',
    'model_frame',
    'primary_positions',
    'require_columns',
    'training_window',
    'week_rows',
    'with_primary_position',
]
