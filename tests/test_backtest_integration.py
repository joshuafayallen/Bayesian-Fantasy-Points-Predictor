"""
Slow end-to-end check: actually fits each model architecture via ADVI on a
REAL fold of processed-data/ff-processed.parquet, through
src/backtest_harness.py's real CLI path (Fold -> fit_predict -> score).
This is the test that would have caught e.g. a silent shape mismatch
between src/models.py and how backtest_harness.py builds `D`, or a
regression in the team_form join -- the synthetic-data tests in
test_models.py can't see those, since they hand-construct an
already-consistent D/coords pair.

Not run by default (each model takes ~15-30s to fit via ADVI, so the full
file is ~1-2 minutes) -- opt in with:

    RUN_SLOW_TESTS=1 uv run pytest tests/test_backtest_integration.py -v

or just run the equivalent CLI command directly, which is exactly what
this test automates:

    python src/backtest_harness.py hs advi 2024 6 --advi-n 200 --advi-draws 20 --out /tmp/bt_smoke.csv
"""
import os
import sys

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import backtest_harness as bh  # noqa: E402

pytestmark = pytest.mark.skipif(
    os.environ.get('RUN_SLOW_TESTS') != '1',
    reason="set RUN_SLOW_TESTS=1 to run -- each model fit takes ~15-30s",
)

SEASON = 2024
WEEK = 6
ADVI_KWARGS = dict(advi_n=200, advi_draws=20)


def _fold(model_name):
    df = pl.read_parquet(bh.DATA_PATH)
    if model_name == 'team':
        team_form = pl.read_parquet(bh.TEAM_FORM_PATH)
        opp_team_form = team_form.rename({'team': 'opp_team', 'team_form': 'opp_team_form'})
        df = df.join(team_form, on=['season', 'week', 'team'], how='left')
        df = df.join(opp_team_form, on=['season', 'week', 'opp_team'], how='left')
        df = df.with_columns(pl.col('team_form', 'opp_team_form').fill_null(0.0))

    train = df.filter(
        (pl.col('season') >= SEASON - (bh.WINDOW_SEASONS - 1)) &
        ((pl.col('season') < SEASON) | ((pl.col('season') == SEASON) & (pl.col('week') < WEEK)))
    )
    test = df.filter((pl.col('season') == SEASON) & (pl.col('week') == WEEK))
    assert len(train) >= 200, "not enough training data for this smoke fold"
    assert len(test) > 0, "no test rows for this smoke fold"
    return train, test


@pytest.mark.parametrize('model_name', ['ar', 'hs', 'team'])
def test_fit_predict_real_fold(model_name):
    train, test = _fold(model_name)
    result = bh.fit_predict(model_name, 'advi', train, test, seed=0, **ADVI_KWARGS)
    assert result is not None, "every test row was a cold start -- check the fold"
    S = result['S']
    assert np.isfinite(S).all(), "posterior predictive draws contain NaN/inf"
    assert S.shape[1] == len(result['keep'])

    y = result['keep']['total_fantasy_points'].to_numpy()
    m = bh.score(y, S)
    assert np.isfinite(m['rmse'])
    assert np.isfinite(m['crps'])


def test_fit_predict_baseline():
    """The no-PyMC baseline path -- cheap, but still worth checking it
    returns the shape fit_predict's real callers (run_one_fold, score())
    expect."""
    train, test = _fold('ar')
    result = bh.fit_predict_baseline(train, test)
    assert result['S'].shape == (1, len(test))
