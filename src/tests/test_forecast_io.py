"""WeekForecast as a DataTree: save -> load keeps the frame exactly (nulls, dtypes,
repeated ids), the draws to 0.05 pts, and tidy() gives proper nulls."""
import numpy as np
import polars as pl

from ffpred.forecast import WeekForecast


def _forecast(ids):
    n = len(ids)
    frame = pl.DataFrame({
        'player_id': ids,
        'player': [f'P{i}' for i in range(n)],
        'position': ['QB', 'RB', 'WR', 'TE', 'WR'][:n],
        'team': ['DET', 'DET', None, 'KC', 'KC'][:n],                 # null label
        'opp_team': ['GB'] * n,
        'model': [None, 'hurdle_qblink', 'hurdle_qblink', None, 'x'][:n],   # null text input
        'snaps': pl.Series([None, 30, 45, 12, 60][:n], dtype=pl.Int32),      # integers with a gap
        'is_home': [True, None, False, True, True][:n],                      # booleans with a gap
        'spread': [1.5, None, -3.0, 7.0, 0.0][:n],
    })
    draws = np.random.default_rng(0).gamma(2.0, 6.0, size=(8, n))           # 2 chains x 4 draws
    meta = {'settings': {'method': 'nuts', 'chains': 2}}
    skipped = pl.DataFrame({'player_id': ['z'], 'player': ['Z'], 'model': [None], 'reason': ['no history']})
    return WeekForecast(2026, 10, frame=frame, draws=draws, skipped=skipped, meta=meta)


def test_roundtrip_keeps_frame_and_draws(tmp_path):
    fc = _forecast(['a', 'b', 'c', 'd', 'e'])
    fc2 = WeekForecast.load(fc.save(tmp_path / 'f'))
    assert (tmp_path / 'f' / 'forecast.nc').exists() and (tmp_path / 'f' / 'meta.json').exists()
    assert fc2.frame.equals(fc.frame) and fc2.frame.schema == fc.frame.schema
    assert fc2.skipped.equals(fc.skipped)
    assert np.abs(fc2.draws - fc.draws).max() <= 0.05 + 1e-9                # stored in tenths of a point
    assert dict(fc2.points.sizes) == {'chain': 2, 'draw': 4, 'player_id': 5}
    assert (fc2.season, fc2.week, fc2.meta) == (fc.season, fc.week, fc.meta)


def test_repeated_player_id_survives(tmp_path):
    fc = _forecast(['a', 'b', 'a', 'd'])
    fc2 = WeekForecast.load(fc.save(tmp_path / 'f'))
    assert fc2.frame['player_id'].to_list() == ['a', 'b', 'a', 'd']
    assert fc2.frame.equals(fc.frame)


def test_tidy_has_proper_nulls(tmp_path):
    t = WeekForecast.load(_forecast(['a', 'b', 'c', 'd', 'e']).save(tmp_path / 'f')).tidy()
    assert t.height == 8 * 5
    first = t.filter(pl.col('player_id') == 'a')
    assert first['model'].null_count() == 8 and first['snaps'].null_count() == 8
    assert t['snaps'].dtype == pl.Int32 and not (t['model'] == '__null__').any()
    assert t.filter(pl.col('player_id') == 'c')['team'].null_count() == 8   # null label stays null
