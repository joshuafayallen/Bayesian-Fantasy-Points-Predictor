"""Build the model input frame for a week that hasn't been played yet.

The backtest predicts on a week's *played* rows, which is fine for scoring
but can't be used for a real decision: those rows only exist for players
who actually played, and their features come from the processed data. For a
real week we need one row per player who *could* play, with every feature
computed from history strictly before the week plus the published schedule
(opponent, home/away, Vegas lines, weather, roof, surface).

``build_upcoming_frame`` produces exactly the columns of the processed data
the models read. ``tests/test_upcoming.py`` rebuilds historical weeks this
way and checks the features against the processed rows of players who
played.

Features and where they come from (all as of the target week):
    lag_*                   the player's most recent game in history
    games_missed_ytd        team games this season so far - player games this season, clipped to [0, 7]
    roll_avg_points_allowed mean of the opponent's last <=3 games' points allowed to the position this season
    tenure / age            latest value + seasons elapsed
    team                    latest team, or the roster feed's team when rosters are given
    game covariates         the schedule row for the player's team
"""

from __future__ import annotations

import logging

import polars as pl

from .config import TARGET_COL
from .data import before, require_columns
from .data_build import TEAM_MAPPING

log = logging.getLogger(__name__)

# raw column in history -> lag feature it becomes for the next game
LAG_SOURCES = {
    'yards_per_rush_attempt': 'lag_yards_per_rush_attempt',
    'target_share': 'lag_target_share',
    'yards_per_target': 'lag_yards_per_target',
    'yards_per_pass_attempt': 'lag_yards_per_pass_attempt',
    'avg_depth_of_target': 'lag_avg_depth_of_target',
    'rush_share': 'lag_rush_share',
    'offense_pct': 'lag_offense_pct',
    'st_pct': 'lag_st_pct',
}
GAME_COLS = ('game_id', 'home_team', 'away_team', 'spread_line', 'total_line', 'is_grass',
             'is_indoors', 'div_game', 'temp', 'wind', 'home_rest', 'away_rest')
ACTIVE_STATUSES = ('ACT',)


def schedule_from_processed(df: pl.DataFrame, season: int, week: float) -> pl.DataFrame:
    """The game rows of a played week, recovered from the processed data
    (for replaying history through the same path as an upcoming week)."""
    return (df.filter((pl.col('season') == season) & (pl.col('week') == week))
            .select('season', 'week', *GAME_COLS).unique(subset=['game_id']).sort('game_id'))


def _team_games(schedule: pl.DataFrame) -> pl.DataFrame:
    """One row per (game, team) from the team's side."""
    g = schedule.with_columns(pl.col('home_team', 'away_team').replace(TEAM_MAPPING))
    home = g.with_columns(pl.col('home_team').alias('team'), pl.col('away_team').alias('opp_team'),
                          pl.lit(1, pl.Int32).alias('is_home'))
    away = g.with_columns(pl.col('away_team').alias('team'), pl.col('home_team').alias('opp_team'),
                          pl.lit(0, pl.Int32).alias('is_home'))
    return pl.concat([home, away])


def build_upcoming_frame(history: pl.DataFrame, schedule: pl.DataFrame, season: int, week: float,
                         rosters: pl.DataFrame | None = None, active_only: bool = True,
                         lookback_seasons: int = 1) -> pl.DataFrame:
    """One row per player expected to play in (season, week).

    history:   processed player-weeks; only rows strictly before the week are used.
    schedule:  that week's games (``data_build.fetch_schedule`` or
               ``schedule_from_processed``).
    rosters:   optional (player_id, team[, status]) for the target season/week.
               When given it decides each player's team (trades, signings) and,
               with ``active_only``, drops players whose status isn't ACT.
               Players missing from it are dropped.
    lookback_seasons: candidates are players with a game in the target season
               or the ``lookback_seasons`` before it.

    Players on bye have no game and are absent. Rookies (no history) are
    absent too: the models can't score a player they haven't seen."""
    week = float(week)
    require_columns(history, ('player_id', 'player', 'position', 'team', 'season', 'week', 'tenure',
                              'age', 'opp_team', 'points_allowed', *LAG_SOURCES), 'history')
    require_columns(schedule, GAME_COLS, 'schedule')
    hist = history.filter(before(season, week))
    if hist.height == 0:
        raise ValueError(f'no history before {season} week {week:g}')

    latest = (hist.filter(pl.col('season') >= season - lookback_seasons)
              .sort('player_id', 'season', 'week')
              .group_by('player_id', maintain_order=True).last())

    cand = latest.select(
        'player_id', 'player', 'position', 'team',
        (pl.col('tenure') + (season - pl.col('season'))).alias('tenure'),
        (pl.col('age') + (season - pl.col('season'))).alias('age'),
        *[pl.col(src).fill_null(0).fill_nan(0).alias(dst) for src, dst in LAG_SOURCES.items()],
    )

    if rosters is not None:
        require_columns(rosters, ('player_id', 'team'), 'rosters')
        r = rosters.with_columns(pl.col('team').replace(TEAM_MAPPING))
        if active_only and 'status' in r.columns:
            r = r.filter(pl.col('status').is_in(ACTIVE_STATUSES))
        r = r.select('player_id', pl.col('team').alias('_roster_team')).unique(subset=['player_id'])
        n0 = cand.height
        cand = (cand.join(r, on='player_id', how='inner')
                .with_columns(pl.col('_roster_team').alias('team')).drop('_roster_team'))
        log.info('rosters: kept %d of %d candidate players (active on a %d roster)', cand.height, n0, season)

    games = _team_games(schedule).select('team', 'opp_team', 'is_home', *GAME_COLS)
    frame = cand.join(games, on='team', how='inner')

    # availability: team games played vs the player's games, this season
    this_season = hist.filter(pl.col('season') == season)
    team_gp = (this_season.select('team', 'week').unique().group_by('team')
               .agg(pl.len().cast(pl.Int32).alias('_team_gp')))
    player_gp = this_season.group_by('player_id').agg(pl.len().cast(pl.Int32).alias('_player_gp'))
    frame = (frame.join(team_gp, on='team', how='left').join(player_gp, on='player_id', how='left')
             .with_columns((pl.col('_team_gp').fill_null(0) - pl.col('_player_gp').fill_null(0))
                           .clip(0, 7).alias('games_missed_ytd'))
             .drop('_team_gp', '_player_gp'))

    # opponent's fantasy points allowed to this position, last <= 3 games this season
    allowed = (this_season.select('opp_team', 'week', 'position', 'points_allowed')
               .unique(subset=['opp_team', 'week', 'position'])
               .sort('opp_team', 'position', 'week')
               .group_by('opp_team', 'position', maintain_order=True)
               .agg(pl.col('points_allowed').tail(3).mean().alias('roll_avg_points_allowed')))
    frame = (frame.join(allowed, on=['opp_team', 'position'], how='left')
             .with_columns(pl.col('roll_avg_points_allowed').fill_null(0.0)))

    frame = frame.with_columns(
        pl.lit(season, pl.Int32).alias('season'),
        pl.lit(week, pl.Float64).alias('week'),
        pl.lit(1 if season >= 2018 else 0, pl.Int32).alias('era'),
        pl.lit(None, pl.Float64).alias(TARGET_COL),
        pl.concat_str([pl.col('player_id'), pl.lit('_'), pl.lit(str(season))]).alias('player_season'),
        pl.col('spread_line', 'total_line').fill_null(0.0),
    ).sort('team', 'position', 'player_id')

    n_missing_lines = frame.filter(pl.col('spread_line') == 0).select('game_id').n_unique()
    if n_missing_lines:
        log.warning('%d games have no Vegas line yet (filled with 0, as in the processed data)', n_missing_lines)
    log.info('upcoming frame %d week %g: %d players in %d games', season, week, frame.height,
             frame['game_id'].n_unique())
    return frame


def fetch_rosters(season: int, week: float | None = None) -> pl.DataFrame:
    """(player_id, team, position, status) from nflverse: the weekly roster
    for ``week`` (latest available week <= it) when published, otherwise
    the season roster."""
    import nflreadpy as nfl
    try:
        r = nfl.load_rosters_weekly(seasons=[season])
        if week is not None:
            r = r.filter(pl.col('week') <= week)
            r = r.filter(pl.col('week') == r['week'].max())
    except Exception as e:   # weekly feed not published yet
        log.info('weekly rosters unavailable (%s); using season rosters', e)
        r = nfl.load_rosters(seasons=[season])
    return (r.filter(pl.col('gsis_id').is_not_null())
            .select(pl.col('gsis_id').alias('player_id'), 'team', 'position', 'status')
            .unique(subset=['player_id'], keep='last'))
