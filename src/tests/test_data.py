import polars as pl
import pytest

from ffpred import data as D
from ffpred.data_build import clean_schedules


def test_vegas_features_are_team_perspective():
    df = pl.DataFrame({'spread_line': [3.0, 3.0], 'total_line': [44.0, 44.0], 'is_home': [1, 0]})
    out = D.add_vegas_features(df)
    assert out['team_spread'].to_list() == [3.0, -3.0]
    assert out['vegas_team_total'].to_list() == [23.5, 20.5]
    assert out['vegas_team_total'].sum() == 44.0       # the two sides add up to the game total


def test_join_team_form_uses_only_earlier_weeks():
    tf = pl.DataFrame({'season': [2024, 2024, 2024, 2024], 'week': [1.0, 2.0, 1.0, 2.0],
                       'team': ['A', 'A', 'B', 'B'], 'team_form': [1.0, 2.0, -1.0, -2.0]})
    rows = pl.DataFrame({'season': [2024, 2024, 2024, 2025], 'week': [1.0, 2.0, 3.0, 1.0],
                         'team': ['A', 'A', 'A', 'A'], 'opp_team': ['B', 'B', 'B', 'B']})
    out = D.join_team_form(rows, tf)
    # week 1: nothing earlier -> 0; week 2 sees week 1; week 3 (bye week 2? no) sees week 2;
    # next season's week 1 carries the last value forward
    assert out['team_form'].to_list() == [0.0, 1.0, 2.0, 2.0]
    assert out['opp_team_form'].to_list() == [0.0, -1.0, -2.0, -2.0]


def test_join_team_form_rejects_duplicate_keys():
    tf = pl.DataFrame({'season': [2024, 2024], 'week': [1.0, 1.0], 'team': ['A', 'A'], 'team_form': [1.0, 2.0]})
    rows = pl.DataFrame({'season': [2024], 'week': [2.0], 'team': ['A'], 'opp_team': ['A']})
    with pytest.raises(D.DataError):
        D.join_team_form(rows, tf)


def test_primary_position_most_frequent_and_reference():
    df = pl.DataFrame({'player_id': ['x'] * 3 + ['y'], 'position': ['TE', 'WR', 'WR', 'RB']})
    out = D.with_primary_position(df)
    assert out['position'].to_list() == ['WR', 'WR', 'WR', 'RB']
    ref = pl.DataFrame({'player_id': ['x'], 'position': ['TE']})
    out2 = D.with_primary_position(df, reference=ref)
    assert out2['position'].to_list() == ['TE', 'TE', 'TE', 'RB']   # y not in reference keeps his listing


def test_training_window_is_strictly_before(league):
    tr = D.training_window(league, 2023, 4.0, window_seasons=2)
    assert tr['season'].min() == 2022
    assert tr.filter(pl.col('season') == 2023)['week'].max() == 3.0


def test_load_processed_dedupes(tmp_path, league):
    dup = pl.concat([league, league.head(3)])
    p = tmp_path / 'x.parquet'
    dup.write_parquet(p)
    assert D.load_processed(p).height == league.height


def test_load_processed_schema_check(tmp_path, league):
    p = tmp_path / 'x.parquet'
    league.drop('lag_st_pct').write_parquet(p)
    with pytest.raises(D.DataError, match='lag_st_pct'):
        D.load_processed(p)


def test_clean_schedules_weather_rules():
    raw = pl.DataFrame({
        'game_id': ['g1', 'g2', 'g3', '2016_13_NYG_PIT'], 'season': [2024] * 4, 'week': [1] * 4,
        'game_type': ['REG', 'REG', 'REG', 'REG'], 'away_team': ['A'] * 4, 'home_team': ['B'] * 4,
        'away_score': [1] * 4, 'home_score': [2] * 4, 'away_rest': [7] * 4, 'home_rest': [7] * 4,
        'total_line': [44.0] * 4, 'spread_line': [1.0] * 4, 'div_game': [0] * 4,
        'surface': ['grass', 'fieldturf', 'grass', 'grass'], 'roof': ['outdoors', 'dome', 'outdoors', 'outdoors'],
        'temp': [None, 50, 40, 30], 'wind': [None, 10, 12, 70]})
    out = clean_schedules(raw, keep_week=True)
    assert out['temp'].to_list() == [60, 68, 40, 30]
    assert out['wind'].to_list() == [8, 0, 12, 7]
    assert out['is_indoors'].to_list() == [0, 1, 0, 0]
    assert out['is_grass'].to_list() == [1, 0, 1, 1]
