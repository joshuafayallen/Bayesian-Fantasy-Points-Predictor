import polars as pl
import pytest

from ffpred import data as D
from ffpred.upcoming import LAG_SOURCES, build_upcoming_frame, schedule_from_processed

from conftest import REAL_DATA, needs_real_data

FEATURES = ['team', 'opp_team', 'is_home', 'tenure', 'age', 'games_missed_ytd', 'roll_avg_points_allowed',
            'spread_line', 'total_line', 'temp', 'wind', 'is_grass', 'is_indoors', 'div_game', 'era',
            *LAG_SOURCES.values()]


def _mismatches(actual, built):
    j = actual.join(built, on='player_id', how='inner', suffix='_u')
    out = {}
    for c in FEATURES:
        if c in ('team', 'opp_team'):
            bad = j.filter(pl.col(c) != pl.col(f'{c}_u'))
        else:
            bad = j.filter((pl.col(c).cast(pl.Float64) - pl.col(f'{c}_u').cast(pl.Float64)).abs() > 1e-9)
        out[c] = bad.height
    return j.height, out


def test_replay_matches_processed_rows_synthetic(league):
    s, w = 2023, 4.0
    built = build_upcoming_frame(league, schedule_from_processed(league, s, w), s, w)
    actual = D.week_rows(league, s, w)
    n, bad = _mismatches(actual, built)
    assert n == actual.height                     # everyone who played was a candidate
    assert sum(bad.values()) == 0, bad
    assert built[ 'total_fantasy_points'].null_count() == built.height   # no outcome leaks in


def test_bye_teams_and_rosters(league):
    s, w = 2023, 4.0
    sched = schedule_from_processed(league, s, w).head(1)          # only one game this week
    built = build_upcoming_frame(league, sched, s, w)
    assert set(built['team'].unique()) == {sched['home_team'][0], sched['away_team'][0]}
    # a roster feed moves a player and marks another inactive
    pid_move = league.filter(pl.col('team') == 'CCC')['player_id'][0]
    pid_out = built['player_id'][0]
    rosters = pl.DataFrame({'player_id': [pid_move, pid_out] + [p for p in built['player_id'].to_list() if p != pid_out],
                            'team': [sched['home_team'][0], 'XXX'] + [None] * (built.height - 1),
                            'status': ['ACT', 'RES'] + ['ACT'] * (built.height - 1)})
    rosters = rosters.with_columns(pl.col('team').fill_null(pl.lit(sched['away_team'][0])))
    b2 = build_upcoming_frame(league, sched, s, w, rosters=rosters)
    assert pid_move in b2['player_id'].to_list()
    assert pid_out not in b2['player_id'].to_list()


def test_games_missed(league):
    s, w = 2023, 5.0
    pid = league.filter(pl.col('position') == 'WR')['player_id'][0]
    hist = league.filter(~((pl.col('player_id') == pid) & (pl.col('season') == s) & pl.col('week').is_in([2.0, 3.0])))
    built = build_upcoming_frame(hist, schedule_from_processed(league, s, w), s, w)
    assert built.filter(pl.col('player_id') == pid)['games_missed_ytd'][0] == 2


@needs_real_data
@pytest.mark.parametrize('season,week', [(2024, 6.0), (2025, 12.0), (2023, 15.0)])
def test_replay_matches_processed_rows_real(season, week):
    df = D.load_processed(REAL_DATA)
    built = build_upcoming_frame(df, schedule_from_processed(df, season, week), season, week)
    actual = D.week_rows(df, season, week)
    n, bad = _mismatches(actual, built)
    assert n >= 0.95 * actual.height                 # misses: debuts / rookies
    # mid-season, only trades / snap-join gaps should differ
    assert max(bad.values()) <= 0.03 * n, bad
