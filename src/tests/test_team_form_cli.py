import numpy as np
import polars as pl
import pytest

from ffpred.cli import _int_list, build_parser
from ffpred.team_form import build_week_arrays, fit_team_form, games_from_processed

from conftest import slow


def test_week_arrays(league):
    games = games_from_processed(league)
    teams = sorted(games['home_team'].unique().to_list())
    seasons = sorted(games['season'].unique().to_list())
    y, gw, A = build_week_arrays(games, teams, seasons)
    T, N = y.shape
    assert (T, N) == (gw.height, len(teams))
    played = A['sign'] != 0
    assert (np.isnan(y) == ~played).all()
    # every played row loads +1 on own offense and -1 on exactly one opposing defense
    assert (A['Z'][played][:, :N].sum(1) == 1).all() and (A['Z'][played][:, N:].sum(1) == -1).all()
    assert (A['sign'].sum(1) == 0).all()                   # one home and one away per game
    assert A['boundary'].sum() == len(seasons) - 1


@slow
def test_fit_team_form_smoke(league):
    tf = fit_team_form(league, draws=50, tune=50, chains=1, filter_draws=20)
    assert {'season', 'week', 'team', 'team_form', 'team_off_form', 'team_def_form'} <= set(tf.columns)
    assert tf.filter(pl.col('week') == 0).height == 4 * league['season'].n_unique()   # week-0 rows
    # centred across teams each week
    assert tf.group_by('season', 'week').agg(pl.col('team_off_form').sum().abs()).max().item(0, 2) < 1e-6


def test_int_list():
    assert _int_list('2019 2023-2025') == [2019, 2023, 2024, 2025]
    assert _int_list('6,9,12') == [6, 9, 12]


@pytest.mark.parametrize('argv', [
    ['build-data', '--seasons', '2006-2026'],
    ['build-team-form', '--draws', '100'],
    ['forecast', '2026', '5', '--method', 'advi'],
    ['decide', 'forecasts/2026_wk05', '--roster', 'a,b', '--rule', 'cvar_averse', '--lam', '2'],
    ['backtest', '--seasons', '2024', '--weeks', '6', '--eligibility', 'legacy'],
    ['league', '--seasons', '2025', '--weeks', '1-3'],
    ['summarize-backtest', 'x'], ['summarize-league', 'x'],
])
def test_cli_parses(argv):
    a = build_parser().parse_args(argv)
    assert callable(a.fn)


def test_cli_check_forecast_parses():
    a = build_parser().parse_args(['check-forecast', 'forecasts/x'])
    assert a.fn.__name__ == 'cmd_check_forecast'
