"""Build processed-data/ff-processed.parquet from nflverse sources.

A function-based port of ``src/data_cleaning.py`` (up to the
``ff-processed.parquet`` write; the ESPN QBR / Next Gen Stats joins are not
ported because no production model reads them -- see the advanced-stats
ablation doc).

``build_processed(legacy=True)`` reproduces the src/ output row-for-row
(checked against the current parquet). ``legacy=False`` (the default)
applies three fixes:

1. Roster nulls are forward-filled *within* a player. The original sorted
   all rosters by name and forward-filled across the whole frame before
   dropping rows with a null gsis_id, so a null field (or a null id) could
   inherit the previous player's value.
2. The snap-count id crosswalk is de-duplicated, so one game can't attach
   twice to a player (it produced 6 duplicated player-weeks).
3. The (player, season, week) de-dup after the roster join keeps a
   deterministic row (sorted first) instead of whatever order the join
   produced.

Network: needs nflverse (GitHub releases) and the DynastyProcess id
crosswalk (raw.githubusercontent.com).
"""

from __future__ import annotations

import datetime as _dt
import logging
from pathlib import Path

import nflreadpy as nfl
import polars as pl
import polars.selectors as cs

# pyrefly: ignore [missing-import]
from .config import PATHS

log = logging.getLogger(__name__)

FIRST_SEASON = 2006
POSITIONS = ('TE', 'RB', 'FB', 'QB', 'WR')
TEAM_MAPPING = {'SL': 'LA', 'ARZ': 'ARI', 'OAK': 'LV', 'SD': 'LAC', 'BLT': 'BAL',
                'CLV': 'CLE', 'HST': 'HOU', 'STL': 'LA'}
SCHEDULE_COLS = ['game_id', 'season', 'away_team', 'away_score', 'home_team', 'home_score',
                 'away_rest', 'home_rest', 'total_line', 'spread_line', 'is_grass', 'is_indoors',
                 'div_game', 'wind', 'temp']
PLAYER_IDS_URL = ('https://raw.githubusercontent.com/dynastyprocess/data/refs/heads/master/'
                  'files/db_playerids.csv')
# two games with a recorded 70 mph wind, hand-corrected from PFR
WIND_FIXES = {'2016_13_NYG_PIT': 7, '2008_02_TEN_CIN': 13}


def current_season(today: _dt.date | None = None) -> int:
    """NFL season in progress (or most recently finished) on ``today``."""
    today = today or _dt.date.today()
    return today.year if today.month >= 9 else today.year - 1


# ----------------------------------------------------------------- schedules

def clean_schedules(raw: pl.DataFrame, keep_week: bool = False) -> pl.DataFrame:
    """Regular-season games with the game-level covariates the models use.
    Indoor games get temp 68 / wind 0; missing outdoor temp/wind get 60 / 8."""
    out = (
        raw.filter(pl.col('game_type') == 'REG')
        .with_columns(
            pl.when(pl.col('surface').str.contains('grass')).then(1).otherwise(0).alias('is_grass'),
            pl.when(pl.col('roof').is_in(['closed', 'dome'])).then(1).otherwise(0).alias('is_indoors'))
        .with_columns(
            # conditional values taken from https://github.com/ffverse/ffopportunity/blob/main/R/ep_preprocess.R
            pl.when(pl.col('is_indoors') == 1).then(pl.lit(68))
            .when(pl.col('temp').is_null()).then(pl.lit(60))
            .otherwise(pl.col('temp')).alias('temp_clean'),
            pl.when(pl.col('is_indoors') == 1).then(pl.lit(0))
            .when(pl.col('wind').is_null()).then(pl.lit(8))
            .when(pl.col('game_id') == '2016_13_NYG_PIT').then(WIND_FIXES['2016_13_NYG_PIT'])
            .when(pl.col('game_id') == '2008_02_TEN_CIN').then(WIND_FIXES['2008_02_TEN_CIN'])
            .otherwise(pl.col('wind')).alias('wind_clean'))
        .drop('temp', 'wind')
        .rename({'temp_clean': 'temp', 'wind_clean': 'wind'})
    )
    cols = SCHEDULE_COLS + (['week'] if keep_week else [])
    return out.select(cols)


def fetch_schedule(season: int, week: float | None = None) -> pl.DataFrame:
    """One season's (or week's) regular-season schedule, cleaned, with team
    codes mapped the same way as the processed data. Used to build the
    frame for an upcoming week."""
    sched = clean_schedules(nfl.load_schedules(seasons=[season]), keep_week=True)
    sched = sched.with_columns(pl.col('week').cast(pl.Float64),
                               pl.col('home_team', 'away_team').replace(TEAM_MAPPING))
    if week is not None:
        sched = sched.filter(pl.col('week') == float(week))
    return sched


# ------------------------------------------------------------------- build

def _load_sources(seasons: list[int], raw_dir: Path | None):
    import nflreadpy as nfl
    schedules = nfl.load_schedules(seasons=True)
    rosters = nfl.load_rosters(seasons=seasons)
    opportunity = nfl.load_ff_opportunity(seasons=True, stat_type='weekly')
    snaps = nfl.load_snap_counts(True)
    ids = pl.read_csv(PLAYER_IDS_URL, null_values='NA', infer_schema_length=10_000)
    if raw_dir is not None:
        raw_dir.mkdir(parents=True, exist_ok=True)
        ids.write_csv(raw_dir / 'nfl-playerids.csv')
    return schedules, rosters, opportunity, snaps, ids


def build_processed(seasons: list[int] | None = None, legacy: bool = False,
                    raw_dir: Path | None = PATHS.raw_dir, sources=None) -> pl.DataFrame:
    """Return the processed player-week frame (one row per player-game).

    ``sources`` lets tests pass pre-loaded (schedules, rosters, opportunity,
    snaps, ids) instead of downloading."""
    seasons = seasons or list(range(FIRST_SEASON, current_season() + 1))
    first, last = min(seasons), max(seasons)
    schedules, rosters_raw, opportunity, snaps, ids = sources or _load_sources(seasons, raw_dir)

    vegas = clean_schedules(schedules.filter(pl.col('season').is_between(first, last)))

    rosters = (rosters_raw.filter(pl.col('position').is_in(POSITIONS))
               .with_columns(pl.col('birth_date').dt.year().alias('birth_year')))
    if legacy:
        rosters = rosters.sort(['full_name', 'season']).fill_null(strategy='forward')
    else:
        rosters = (rosters.filter(pl.col('gsis_id').is_not_null())
                   .sort(['gsis_id', 'season'])
                   .with_columns(pl.all().exclude('gsis_id').fill_null(strategy='forward').over('gsis_id')))
    rosters = (
        rosters
        .with_columns((pl.col('season') - pl.col('birth_year')).alias('age'),
                      (pl.col('season') - pl.col('entry_year')).alias('tenure'))
        .select('full_name', 'team', 'season', 'birth_date', 'gsis_id', 'tenure', 'age')
        .filter(pl.col('gsis_id').is_not_null())
        .rename({'gsis_id': 'player_id'})
        .filter(pl.col('full_name') != 'George Farmer')   # bad entry_year upstream
        .drop('full_name')
    )

    fantasy = (
        opportunity
        .select(pl.col('player_id', 'full_name', 'position', 'total_fantasy_points',
                       'total_fantasy_points_diff_team', 'game_id', 'receptions', 'pass_completions',
                       'season', 'week'),
                cs.ends_with('gained'), cs.ends_with('points_diff'), cs.ends_with('attempt'),
                pl.col('rec_air_yards'))
        .filter(pl.col('player_id').is_not_null() & pl.col('position').is_in(['TE', 'RB', 'WR', 'QB', 'FB']))
        .with_columns(pl.col('season').cast(pl.Int32))
        .filter(pl.col('season').is_between(first, last))
    )

    # A roster can list a player more than once per season (trades, roster
    # snapshot noise). game_id embeds both teams, so keep only the roster
    # team that actually played in this game; then one row per player-week.
    joined = (
        fantasy.join(rosters, on=['player_id', 'season'])
        .with_columns(pl.col('game_id').str.split('_').list.get(2).alias('_away_team'),
                      pl.col('game_id').str.split('_').list.get(3).alias('_home_team'))
        .filter((pl.col('team') == pl.col('_away_team')) | (pl.col('team') == pl.col('_home_team')))
        .drop('_away_team', '_home_team')
    )
    if not legacy:
        joined = joined.sort(['player_id', 'season', 'week', 'team'])
    joined = (joined.unique(subset=['player_id', 'season', 'week'], keep='first')
              .sort(['full_name', 'season', 'week'])
              .join(vegas, on='game_id'))

    cov = (
        joined
        .with_columns(
            pl.when(pl.col('team') == pl.col('home_team')).then(1).otherwise(0).alias('is_home'),
            pl.when(pl.col('team') != pl.col('home_team')).then(pl.col('home_team'))
            .otherwise(pl.col('away_team')).alias('opp_team'))
        .with_columns(
            pl.when(pl.col('is_home') == 1).then(pl.col('home_rest')).otherwise(pl.col('away_rest')).alias('player_rest'),
            pl.when(pl.col('team') != pl.col('home_team')).then(pl.col('home_rest'))
            .otherwise(pl.col('away_rest')).alias('opponent_rest'),
            pl.when(pl.col('is_home') == 1).then(pl.col('home_score')).otherwise(pl.col('away_score')).alias('team_score'),
            pl.when(pl.col('team') != pl.col('home_team')).then(pl.col('home_score'))
            .otherwise(pl.col('away_score')).alias('opponent_score'),
            pl.when(pl.col('is_home') == 1).then(pl.col('spread_line')).otherwise(-pl.col('spread_line')).alias('team_spread'))
        .with_columns(
            (pl.col('total_line') / 2 + pl.col('team_spread') / 2).alias('vegas_team_total'),
            (pl.col('player_rest') - pl.col('opponent_rest')).alias('rest_diff'),
            cs.ends_with('attempt').sum().over(['team', 'game_id']).name.suffix('_total'),
            pl.col('pass_completions').sum().over(['team', 'game_id']).alias('total_completions'))
        .with_columns(
            pl.when(pl.col('rush_attempt') > 0).then(pl.col('rush_yards_gained') / pl.col('rush_attempt'))
            .otherwise(None).alias('yards_per_rush_attempt'),
            pl.when(pl.col('rec_attempt_total') > 0).then(pl.col('rec_attempt') / pl.col('rec_attempt_total'))
            .otherwise(None).alias('target_share'),
            pl.when(pl.col('rush_attempt_total') > 0).then(pl.col('rush_attempt') / pl.col('rush_attempt_total'))
            .otherwise(None).alias('rush_share'),
            pl.when(pl.col('rec_attempt') > 0).then(pl.col('rec_yards_gained') / pl.col('rec_attempt'))
            .otherwise(None).alias('yards_per_target'),
            pl.when(pl.col('pass_attempt_total') > 0).then(pl.col('pass_yards_gained') / pl.col('pass_attempt_total'))
            .otherwise(None).alias('yards_per_pass_attempt'),
            pl.when(pl.col('rec_attempt') > 0).then(pl.col('rec_air_yards') / pl.col('rec_attempt'))
            .otherwise(None).alias('avg_depth_of_target'))
    )

    # opponent's fantasy points allowed to each position, rolling average of 
    # last three games 
    allowed = (
        cov.group_by(['opp_team', 'season', 'week', 'position'])
        .agg((-pl.col('total_fantasy_points_diff')).sum().alias('points_allowed'))
        .sort(['opp_team', 'season', 'position', 'week'])
        .with_columns(pl.col('points_allowed').shift(1).rolling_mean(window_size=3, min_samples=1)
                      .over(['opp_team', 'season', 'position']).alias('roll_avg_points_allowed'))
    )

    lag_src = ['yards_per_rush_attempt', 'target_share', 'yards_per_target', 'yards_per_pass_attempt',
               'avg_depth_of_target', 'rush_share']
    base = (
        cov.join(allowed, on=['opp_team', 'season', 'week', 'position'], how='left')
        .drop('season_right', strict=False)
        .fill_nan(0)
        .with_columns(
            pl.when(pl.col('season') >= 2018).then(1).otherwise(0).alias('era'),
            pl.concat_str([pl.col('player_id'), pl.lit('_'), pl.col('season')]).alias('player_season'),
            pl.concat_str([pl.col('team'), pl.lit('_'), pl.col('season')]).alias('team_season'))
        .rename({'full_name': 'player'})
        .with_columns(pl.col('team', 'opp_team', 'home_team', 'away_team').replace(TEAM_MAPPING))
    )
    if not legacy:
        # shift(1).over() takes the previous ROW, so the frame must be in game
        # order; the joins above don't guarantee row order.
        base = base.sort(['player_id', 'season', 'week'])
    base = base.with_columns(pl.col(*lag_src).shift(1).over('player_id').name.prefix('lag_')).fill_null(0)

    team_games = (
        base.select('team', 'season', 'week').unique().sort(['team', 'season', 'week'])
        .with_columns((pl.col('week').cum_count().over(['team', 'season']) - 1).alias('team_games_played_ytd'))
    )

    snap_ids = ids
    if not legacy:
        # One row per (pfr_id, gsis_id) pair, and drop pfr_ids that map to more
        # than one gsis_id: the crosswalk has a few of those (e.g. HarrAl00 is
        # listed for both Alonzo Harris and Aaron Ripkowski), which attached one
        # player's snaps to another, or the same game twice.
        snap_ids = (ids.select('pfr_id', 'gsis_id').drop_nulls().unique()
                    .filter(pl.len().over('pfr_id') == 1))
    add_ids = (
        snaps.join(snap_ids, left_on=['pfr_player_id'], right_on=['pfr_id'])
        .filter(pl.col('position').is_in(['TE', 'RB', 'WR', 'QB', 'FB']))
        .select('game_id', 'gsis_id', 'offense_pct', 'st_pct')
    )
    if not legacy:
        add_ids = add_ids.filter(pl.col('gsis_id').is_not_null()).unique(subset=['game_id', 'gsis_id'], keep='first')

    out = (
        base.sort(['player_id', 'season', 'week'])
        .with_columns((pl.col('week').cum_count().over(['player_id', 'season']) - 1).alias('player_games_played_ytd'))
        .join(team_games, on=['team', 'season', 'week'], how='left')
        .with_columns((pl.col('team_games_played_ytd').cast(pl.Int32)
                       - pl.col('player_games_played_ytd').cast(pl.Int32))
                      .clip(lower_bound=0, upper_bound=7).alias('games_missed_ytd'))
        .join(add_ids, left_on=['game_id', 'player_id'], right_on=['game_id', 'gsis_id'])
    )
    if not legacy:
        out = out.sort(['player_id', 'season', 'week'])
    out = out.with_columns(pl.col('offense_pct', 'st_pct').shift(1).over('player_id').name.prefix('lag_')).fill_null(0)
    log.info('built processed data: %d player-weeks, seasons %d-%d', out.height,
             out['season'].min(), out['season'].max())
    return out


def write_processed(df: pl.DataFrame, path: Path = PATHS.processed) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.parquet.tmp')
    df.write_parquet(tmp)
    tmp.replace(path)
    return path
