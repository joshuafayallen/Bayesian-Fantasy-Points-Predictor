import json

import numpy as np
import polars as pl
import pytest

from ffpred import backtest, league
from ffpred import league as league_mod
from ffpred.config import SamplerSettings
from ffpred.forecast import WeekForecast, forecast_week
from ffpred.scoring import crps, past_only_baseline, score


def test_round_robin_each_team_once_per_round():
    sched = league.round_robin_schedule([f'T{i}' for i in range(10)])
    assert len(sched) == 9
    pairs = set()
    for rnd in sched:
        teams = [t for m in rnd for t in m]
        assert sorted(teams) == sorted(set(teams)) and len(teams) == 10
        pairs |= {frozenset(m) for m in rnd}
    assert len(pairs) == 45                       # everyone meets everyone once


def test_draft_league_returns_player_ids(league):
    hist = league.filter(pl.col('season') < 2023)
    rosters = league_mod.draft_league(hist, n_teams=2, roster_depth={'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1})
    ids = [p for r in rosters.values() for p in r]
    assert len(ids) == len(set(ids)) == 12
    assert set(ids) <= set(hist['player_id'].to_list())


def test_mock_draft_by_id_and_by_name(tmp_path, league):
    ids = league['player_id'].unique().sort().to_list()[:4]
    names = league.filter(pl.col('player_id').is_in(ids)).unique('player_id').sort('player_id')['player'].to_list()
    pos = league.filter(pl.col('player_id').is_in(ids)).unique('player_id').sort('player_id')['position'].to_list()
    p1 = tmp_path / 'by_id.csv'
    # pick 1 also matched a same-named player with no stats (the Yahoo scrape's name-join fan-out)
    pl.DataFrame({'pick': [1, 1, 2, 3, 4], 'team': ['M1', 'M1', 'M1', 'M2', 'M2'],
                  'player': [names[0], names[0]] + names[1:], 'position': [pos[0], pos[0]] + pos[1:],
                  'player_id': ['ghost-id', ids[0]] + ids[1:]}).write_csv(p1)
    assert league_rosters(p1, league) == {'M1': ids[:2], 'M2': ids[2:]}
    p2 = tmp_path / 'by_name.csv'
    pl.DataFrame({'team': ['M1', 'M1', 'M2', 'M2'], 'player': [n + ' Jr.' for n in names]}).write_csv(p2)
    assert league_rosters(p2, league) == {'M1': ids[:2], 'M2': ids[2:]}


def league_rosters(path, df):
    from ffpred.league import rosters_from_mock_draft
    return rosters_from_mock_draft(path, df)


def test_score_matchups_with_fake_forecast():
    rng = np.random.default_rng(0)
    pos = ['QB', 'RB', 'RB', 'RB', 'WR', 'WR', 'WR', 'TE'] * 2
    frame = pl.DataFrame({'player_id': [f'p{i}' for i in range(16)], 'player': [f'P{i}' for i in range(16)],
                          'position': pos, 'team': ['X'] * 16, 'opp_team': ['Y'] * 16,
                          'total_fantasy_points': rng.normal(10, 5, 16)})
    draws = rng.normal(10, 5, (500, 16))
    fc = WeekForecast(season=2024, week=3.0, frame=frame, draws=draws, skipped=frame.head(0))
    rosters = {'A': [f'p{i}' for i in range(8)], 'B': [f'p{i}' for i in range(8, 16)]}
    rows = league.score_matchups(fc, rosters, [('A', 'B')])
    assert {r['rule'] for r in rows} == set(league.RISK_RULES)
    r = rows[0]
    assert r['oracle_a'] >= max(r['baseline_a_realized'], r['smart_a_realized']) - 1e-9
    assert r['baseline_b_realized'] == pytest.approx(
        league.realized_total(frame, de_lineup(draws, frame, rosters['B'])))


def de_lineup(draws, frame, pool):
    from ffpred import decision_engine as de
    return de.maximize_expected(draws, frame, pool=pool).lineup['player_id'].to_list()


def test_legacy_eligibility_drops_season_debuts(league):
    train = league.filter((pl.col('season') < 2023) | (pl.col('week') < 1))
    test = league.filter((pl.col('season') == 2023) & (pl.col('week') == 1))
    assert not backtest.legacy_eligible(train, test).any()     # week 1: every player_season is new


def test_crps_and_score_basics():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 400)
    good = rng.normal(0, 1, (2000, 400))
    bad = rng.normal(3, 1, (2000, 400))
    assert crps(good, y) < crps(bad, y)
    assert crps(good, y) == pytest.approx(0.5641896 * 1, rel=0.1)   # CRPS of N(0,1) vs its own draws ~ 1/sqrt(pi)
    s = score(y, good)
    assert 0.75 < s['cov80'] < 0.85


def test_past_only_baseline():
    train = pl.DataFrame({'player_id': ['a', 'a', 'b'], 'season': [2024, 2024, 2023],
                          'total_fantasy_points': [10.0, 20.0, 5.0]})
    rows = pl.DataFrame({'player_id': ['a', 'b', 'c'], 'season': [2024, 2024, 2024]})
    assert past_only_baseline(rows, train).tolist() == [15.0, pytest.approx(35 / 3), pytest.approx(35 / 3)]


def test_backtest_and_forecast_end_to_end_advi(tmp_path, league, league_team_form):
    s = SamplerSettings(method='advi', advi_n=300, advi_draws=40)
    rec = backtest.run_fold('het_gp_qb_vegas', 2023, 5.0, league, league_team_form, tmp_path, s, min_train_rows=20)
    assert rec['n'] == 4 and rec['n_dropped'] == 0
    assert backtest.run_fold('het_gp_qb_vegas', 2023, 5.0, league, league_team_form, tmp_path, s, min_train_rows=20) is None  # resumes
    backtest.run_fold('baseline:het_gp_qb_vegas', 2023, 5.0, league, league_team_form, tmp_path, s, min_train_rows=20)
    folds, bypos, coefs = backtest.load_run(tmp_path)
    assert folds.height == 2 and coefs.height > 0
    d = backtest.paired_diff(folds, 'het_gp_qb_vegas', 'baseline:het_gp_qb_vegas')
    assert d['n_folds'] == 1

    fc = forecast_week(league, league_team_form, 2023, 5.0, settings=s, min_train_rows=20)
    assert fc.draws.shape == (40, fc.frame.height)
    assert set(fc.frame['position'].unique()) == {'QB', 'RB', 'WR', 'TE'}
    out = fc.save(tmp_path / 'fc')
    fc2 = WeekForecast.load(out)
    assert fc2.frame.equals(fc.frame) and np.allclose(fc2.draws, fc.draws, atol=1e-3)
    assert json.loads((out / 'meta.json').read_text())['fits']['het_gp_vegas']['n_scored'] > 0


def test_tail_metrics_and_calibration_table():
    from ffpred.scoring import calibration_table, tail_metrics, tier_of
    rng = np.random.default_rng(0)
    mu = np.repeat([3.0, 9.0, 15.0, 21.0], 500)
    y = rng.normal(mu, 5)
    S = rng.normal(mu, 5, (4000, mu.size))
    tm = tail_metrics(y, S)
    assert abs(tm['below_p10'] - 0.1) < 0.02 and abs(tm['above_p90'] - 0.1) < 0.02
    assert abs(tm['pred_sd'] - tm['resid_sd']) < 0.3 and sum(tm['pit_hist']) == mu.size
    assert list(tier_of(np.array([1.0, 7.0, 13.0, 25.0]))) == ['<6', '6-12', '12-18', '18+']
    frame = pl.DataFrame({'position': ['WR'] * mu.size, 'total_fantasy_points': y})
    t = calibration_table(frame, S)
    assert t['tier'].to_list() == ['<6', '6-12', '12-18', '18+'] and t['n'].sum() == mu.size
    # a forecast that is too narrow shows up in both tails
    narrow = calibration_table(frame, rng.normal(mu, 2.5, (4000, mu.size)))
    assert (narrow['below_p10'] > 0.2).all() and (narrow['above_p90'] > 0.2).all()


def test_backtest_records_tier_calibration(tmp_path, league, league_team_form):
    s = SamplerSettings(method='advi', advi_n=200, advi_draws=40)
    backtest.run_fold('het_gp_qb_vegas', 2023, 5.0, league, league_team_form, tmp_path, s, min_train_rows=20)
    tiers = backtest.load_tiers(tmp_path)
    assert tiers.height > 0 and {'below_p10', 'above_p90', 'pred_sd', 'resid_sd', 'tier'} <= set(tiers.columns)
    assert backtest.tier_summary(tiers)['n'].sum() == 4
    backtest.summarize(tmp_path)          # prints without error


def test_scratch_hook_matches_production_fold(tmp_path, league, league_team_form):
    """A scratch fit_predict that is the production model scores the same rows identically."""
    from ffpred.config import MODELS
    from ffpred.model import fit
    from ffpred.scratch import run_scratch_fold
    s = SamplerSettings(method='advi', advi_n=200, advi_draws=30)
    prod = backtest.run_fold('het_gp_vegas', 2023, 5.0, league, league_team_form, tmp_path, s, min_train_rows=20)
    rec = run_scratch_fold('scratch', lambda tr, te: fit(tr, MODELS['het_gp_vegas'], s).predict(te),
                           2023, 5.0, league, league_team_form, tmp_path)
    assert rec['n'] == prod['n'] and rec['crps'] == pytest.approx(prod['crps'])
    assert backtest.paired_diff(backtest.load_run(tmp_path)[0], 'scratch', 'het_gp_vegas')['n_folds'] == 1
