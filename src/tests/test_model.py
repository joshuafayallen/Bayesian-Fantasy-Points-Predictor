import numpy as np
import polars as pl
import pytest

from ffpred import data as D
from ffpred.config import HET_GP_QB_VEGAS, HET_GP_VEGAS, SamplerSettings
from ffpred.model import HetPrep, build_het_gp, fit


@pytest.fixture(scope='module')
def train(league, league_team_form):
    df = D.join_team_form(league, league_team_form)
    return D.training_window(df, 2023, 5.0)


@pytest.mark.parametrize('cfg', [HET_GP_VEGAS, HET_GP_QB_VEGAS])
def test_model_builds_with_finite_logp(train, cfg):
    prep = HetPrep(train, cfg)
    m = build_het_gp(prep.train_arrays, prep)
    assert np.isfinite(m.compile_logp()(m.initial_point()))
    assert prep.coords['cont_vars'] == [f'{c}_z' for c in cfg.std_cols]


def test_eligibility(train):
    prep = HetPrep(train, HET_GP_VEGAS)
    row = train.filter(pl.col('position') == 'WR').head(1)
    rookie = row.with_columns(pl.lit('new-guy').alias('player_id'))
    far_future = row.with_columns((pl.col('tenure') + 5).alias('tenure'))
    next_season = row.with_columns(pl.lit(max(prep.times)).alias('tenure'))
    qb = train.filter(pl.col('position') == 'QB').head(1)
    frame = pl.concat([row, rookie, far_future, next_season, qb])
    assert prep.eligible(frame).tolist() == [True, False, False, True, False]


def test_scalers_and_arrays_use_training_quantities(train):
    prep = HetPrep(train, HET_GP_VEGAS)
    tr = train.filter(pl.col('position').is_in(list(HET_GP_VEGAS.positions)))
    D_ = prep.arrays(tr)
    assert np.allclose(D_['cont'].mean(0), 0, atol=1e-9)
    # a frame with no outcome (upcoming week) still converts
    up = tr.head(5).with_columns(pl.lit(None, pl.Float64).alias('total_fantasy_points'))
    assert prep.arrays(up)['cont'].shape == (5, len(HET_GP_VEGAS.std_cols))


def test_fit_predict_advi_smoke(train, league, league_team_form):
    fm = fit(train, HET_GP_QB_VEGAS, SamplerSettings(method='advi', advi_n=300, advi_draws=50))
    test = D.join_team_form(D.week_rows(league, 2023, 5.0), league_team_form).filter(pl.col('position') == 'QB')
    S = fm.predict(test)
    assert S.shape == (50, test.height)
    assert np.isfinite(S).all()
    # draws are on the points scale, not standardized
    assert 5 < S.mean() < 40
    assert set(fm.coefs()['var'].to_list()) >= set(f'{c}_z' for c in HET_GP_QB_VEGAS.std_cols)
    with pytest.raises(ValueError, match='not scoreable'):
        fm.predict(test.with_columns(pl.lit('nobody').alias('player_id')))


def test_fit_is_independent_of_row_order(train):
    s = SamplerSettings(method='advi', advi_n=200, advi_draws=20, seed=3)
    a = fit(train, HET_GP_QB_VEGAS, s)
    b = fit(train.sample(fraction=1.0, shuffle=True, seed=1), HET_GP_QB_VEGAS, s)
    assert np.allclose(a.idata.posterior['cont_prior'].values, b.idata.posterior['cont_prior'].values)
