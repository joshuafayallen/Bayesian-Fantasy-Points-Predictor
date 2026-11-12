import numpy as np
import polars as pl
import pytest

from ffpred import backtest
from ffpred import data as D
from ffpred.config import HET_GP_QB_VEGAS, HURDLE_SKILL, SamplerSettings
from ffpred.forecast import WeekForecast, forecast_week
from ffpred.model import HetPrep
from ffpred.qblink import NO_QB, apply_qb_link, build_hurdle_skill, starter_rows

ADVI = SamplerSettings(method='advi', advi_n=300, advi_draws=40)


def test_trail_pts_is_previous_games_only(league):
    df = D.add_trail_pts(league, window=3)
    p = df.filter(pl.col('player_id') == 'AAA-WR0').sort('season', 'week')
    y = p['total_fantasy_points'].to_numpy()
    t = p['trail_pts'].to_numpy()
    assert p['trail_pts'][0] is None                                        # first game: no history
    assert t[1] == pytest.approx(y[0])
    assert t[5] == pytest.approx(y[2:5].mean())                             # last 3, excluding its own game
    assert df['player_id'].to_list() == league['player_id'].to_list()       # row order kept
    # the forecast-week version gives the same value as the played row
    wk = D.week_rows(league, 2023, 4.0)
    fw = D.add_trail_pts_for_week(wk, league, 2023, 4.0, window=3)
    played = D.week_rows(df, 2023, 4.0)
    assert np.allclose(fw.sort('player_id')['trail_pts'].to_numpy(), played.sort('player_id')['trail_pts'].to_numpy())


def test_starter_rows_picks_top_qb_and_flags_missing():
    q = pl.DataFrame({'season': [2024] * 3, 'week': [3.0] * 3, 'team': ['A', 'A', 'B'],
                      'lag_offense_pct': [0.05, 0.98, 0.9], 'trail_pts': [20.0, 15.0, 18.0]})
    s = pl.DataFrame({'season': [2024] * 3, 'week': [3.0] * 3, 'team': ['A', 'B', 'C']})
    assert starter_rows(q, s, ('lag_offense_pct', 'trail_pts')).tolist() == [1, 2, NO_QB]
    assert starter_rows(q, s, ('trail_pts',)).tolist() == [0, 2, NO_QB]       # first column decides
    assert starter_rows(q.head(0), s, ('trail_pts',)).tolist() == [NO_QB] * 3


def test_apply_qb_link_preserves_means_and_correlates():
    rng = np.random.default_rng(0)
    n = 200_000
    S0 = rng.gamma(2.0, 5.0, (n, 3))                          # three receivers, own draws
    lam = np.full((n, 3), 0.3)
    e_q = rng.normal(0, np.sqrt(1.07), (n, 1))                # one QB, surprise variance 1.07
    out = apply_qb_link(S0, lam, e_q, e_q.var(0), np.array([0, 0, NO_QB]))
    assert np.allclose(out.mean(0), S0.mean(0), rtol=0.01)   # projections unchanged
    assert np.corrcoef(e_q[:, 0], out[:, 0])[0, 1] > 0.1     # linked receivers co-move with the QB
    assert np.array_equal(out[:, 2], S0[:, 2])               # no QB -> untouched


def test_hurdle_skill_builds_with_finite_logp(league, league_team_form):
    df = D.add_trail_pts(D.join_team_form(league, league_team_form))
    tr = D.training_window(df, 2023, 5.0).filter(pl.col('position').is_in(list(HURDLE_SKILL.positions)))
    prep = HetPrep(tr, HURDLE_SKILL)
    n = tr.height
    m = build_hurdle_skill(prep.train_arrays, prep, tr['trail_pts'].fill_null(8.0).to_numpy(),
                           np.zeros(n), np.zeros(n), np.array([9.0, 6.0, 9.0]))
    assert np.isfinite(m.compile_logp()(m.initial_point()))


def test_qb_linked_forecast_and_backtest_advi(tmp_path, league, league_team_form):
    fc = forecast_week(league, league_team_form, 2023, 5.0, settings=ADVI, min_train_rows=20)
    assert fc.meta['forecast_model'] == 'qb_linked'
    assert fc.draws.shape == (40, fc.frame.height) and np.isfinite(fc.draws).all()
    assert set(fc.frame['position'].unique()) == {'QB', 'RB', 'WR', 'TE'}
    skill = fc.frame['position'].is_in(list(HURDLE_SKILL.positions)).to_numpy()
    assert (fc.draws[:, skill] >= 0).all()                    # hurdle-gamma: no negative skill weeks
    fits = fc.meta['fits']
    assert set(fits) == {HURDLE_SKILL.name, HET_GP_QB_VEGAS.name}
    assert fits[HURDLE_SKILL.name]['share_linked_to_qb'] == 1.0   # every team has its QB in the synthetic league
    out = fc.save(tmp_path / 'fc')
    assert WeekForecast.load(out).draws.shape == fc.draws.shape

    recs = backtest.run_qb_linked_fold(2023, 5.0, league, league_team_form, tmp_path / 'bt', ADVI, min_train_rows=20)
    assert recs['qb']['n'] == 4 and recs['skill']['n'] > 0
    assert backtest.run_qb_linked_fold(2023, 5.0, league, league_team_form, tmp_path / 'bt', ADVI,
                                       min_train_rows=20) is None   # resumes
    folds = backtest.load_run(tmp_path / 'bt')[0]
    assert set(folds['model']) == {'qb_linked_skill', 'qb_linked_qb'}
    backtest.run_fold('het_gp_qb_vegas', 2023, 5.0, league, league_team_form, tmp_path / 'bt', ADVI, min_train_rows=20)
    d = backtest.paired_diff(backtest.load_run(tmp_path / 'bt')[0], 'qb_linked_qb', 'het_gp_qb_vegas')
    assert d['n_folds'] == 1
