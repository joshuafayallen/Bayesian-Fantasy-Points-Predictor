"""
Regression/smoke tests for src/models.py -- the single source of truth for
ar_mod / hs_version / team_mod, imported by both src/ff_ar.py and
src/backtest_harness.py. The point of these tests is to catch a WIRING
regression (wrong shape, wrong dim name, a dropped Deterministic, an arg
silently reordered) in seconds, on every change to models.py -- not to
validate the models' real-world fit, which is what src/backtest_harness.py
against the real parquet data is for (see test_backtest_integration.py).

Everything here runs against tiny SYNTHETIC data (a handful of
players/positions/teams/weeks), and only builds the pm.Model graph and
evaluates logp once at the initial point -- no NUTS/ADVI sampling. That's
deliberately enough to catch the pytensor graph-rewrite bugs documented
throughout models.py (add_player_skill/add_strength_ar/add_recent_form_hsgp
-- see their "FGRAPH.REPLACE" comments): those crash at graph-BUILD or
first-logp-eval time, before any sampling step would even start.

Run with (from the project root):
    uv run pytest tests/test_models.py -v
"""
import os
import sys

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import models  # noqa: E402


SEED = 0

N_PLAYERS = 4
N_POSITIONS = 2          # QB, RB
N_TEAMS = 3
WEEKS_PER_SEASON = [3, 2]     # season 0 has 3 weeks, season 1 has 2
N_SEASONS = len(WEEKS_PER_SEASON)
N_ROWS = 24


def make_toy_scenario():
    """A tiny, internally-consistent D/coords pair satisfying the contract
    every build_* function in models.py expects (see base_data()'s
    docstring). Returns a dict with everything a build_* call needs, so
    each test can just unpack what it uses."""
    rng = np.random.default_rng(SEED)

    coords = {
        'player': [f'player_{i}' for i in range(N_PLAYERS)],
        'position': ['QB', 'RB'],
        'team': [f'team_{i}' for i in range(N_TEAMS)],
        'opp_team': [f'team_{i}' for i in range(N_TEAMS)],
        'tenure': [0, 1, 2, 3],
        'season': [2023, 2024],
        'week': [1, 2, 3],  # week-of-season grid -- the max across seasons
        'player_season': [f'p{p}_s{s}' for p in range(N_PLAYERS) for s in range(N_SEASONS)],
        'cont_vars': ['feat_a', 'feat_b'],
        'binary_vars': ['bin_a', 'bin_b'],
    }

    position_of_player = rng.integers(0, N_POSITIONS, size=N_PLAYERS)
    position_group_counts = np.bincount(position_of_player, minlength=N_POSITIONS).astype('float64')

    season_offsets, T, boundary_cols = models.build_ar_calendar(WEEKS_PER_SEASON)

    row_player = rng.integers(0, N_PLAYERS, size=N_ROWS)
    row_position = position_of_player[row_player]
    row_team = rng.integers(0, N_TEAMS, size=N_ROWS)
    row_opp_team = rng.integers(0, N_TEAMS, size=N_ROWS)
    row_tenure = rng.integers(0, len(coords['tenure']), size=N_ROWS)
    row_season = rng.integers(0, N_SEASONS, size=N_ROWS)
    # week index must stay valid for that row's own season (season 1 only
    # has 2 real weeks) -- otherwise global_time() would index past T.
    row_week = np.array([rng.integers(0, WEEKS_PER_SEASON[s]) for s in row_season])
    row_player_season = row_player * N_SEASONS + row_season

    tenure_grid = np.array(coords['tenure'], dtype='float64')
    tenure_z = (row_tenure.astype('float64') - tenure_grid.mean()) / tenure_grid.std()

    D = dict(
        player=row_player,
        position=row_position,
        team=row_team,
        tenure=row_tenure,
        opp_team=row_opp_team,
        season=row_season,
        week=row_week,
        player_season=row_player_season,
        cont=rng.normal(size=(N_ROWS, len(coords['cont_vars']))),
        bi=rng.integers(0, 2, size=(N_ROWS, len(coords['binary_vars']))).astype(float),
        y=rng.normal(10, 5, size=N_ROWS),
        tenure_z=tenure_z,
        games_missed_z=rng.normal(size=N_ROWS),
        snap_pct_z=rng.normal(size=N_ROWS),
        team_form_z=rng.normal(size=N_ROWS),
        opp_team_form_z=rng.normal(size=N_ROWS),
    )

    week_grid_vals = np.array(coords['week'], dtype='float64')

    return dict(
        D=D, coords=coords,
        season_offsets=season_offsets, T=T, boundary_cols=boundary_cols,
        position_of_player=position_of_player, position_group_counts=position_group_counts,
        week_grid_vals=week_grid_vals,
    )


@pytest.fixture(scope='module')
def scenario():
    return make_toy_scenario()


def _assert_finite_logp(model):
    ip = model.initial_point()
    logp_fn = model.compile_logp()
    logp = logp_fn(ip)
    assert np.isfinite(logp), f"logp at initial point is not finite: {logp}"


# --------------------------------------------------------------- build_*

def test_build_ar_mod_smoke(scenario):
    model = models.build_ar_mod(
        scenario['D'], scenario['coords'], scenario['season_offsets'], scenario['T'],
        scenario['boundary_cols'], scenario['position_of_player'], scenario['position_group_counts'],
    )
    _assert_finite_logp(model)
    free_names = {rv.name for rv in model.free_RVs}
    assert {'positions_mu', 'cont_priors', 'bi_priors', 'opp_rho', 'form_sigma',
            'init_sigma', 'rw_sigma'} <= free_names
    det_names = {d.name for d in model.deterministics}
    assert {'intercepts', 'player_skill', 'opp_ar', 'recent_form'} <= det_names


def test_build_hs_version_smoke(scenario):
    model = models.build_hs_version(
        scenario['D'], scenario['coords'], scenario['season_offsets'], scenario['T'],
        scenario['boundary_cols'], scenario['position_of_player'], scenario['position_group_counts'],
        scenario['week_grid_vals'],
    )
    _assert_finite_logp(model)
    # regression test for the specific bug this refactor fixed:
    # backtest_harness.py's old hand-ported add_lkj_position_intercept was
    # missing the positions_mu/age_slope Deterministics that ff_ar.py's
    # always had -- both must be present now that both call the same
    # models.add_lkj_position_intercept.
    det_names = {d.name for d in model.deterministics}
    assert {'positions_mu', 'age_slope', 'intercepts', 'recent_form'} <= det_names
    free_names = {rv.name for rv in model.free_RVs}
    assert {'ab_position', 'pos_chol_cov', 'form_ell', 'form_eta'} <= free_names


def test_build_team_mod_smoke(scenario):
    model = models.build_team_mod(
        scenario['D'], scenario['coords'], scenario['season_offsets'], scenario['T'],
        scenario['boundary_cols'], scenario['position_of_player'], scenario['position_group_counts'],
    )
    _assert_finite_logp(model)
    free_names = {rv.name for rv in model.free_RVs}
    assert {'beta_team_form', 'beta_opp_team_form', 'ab_position'} <= free_names
    det_names = {d.name for d in model.deterministics}
    assert {'positions_mu', 'age_slope'} <= det_names
    # team_mod has no recent-form term (unlike ar_mod/hs_version)
    assert 'recent_form' not in det_names


def test_build_team_season_mod_smoke(scenario):
    """Retired/stale (see the module note in models.py above
    build_team_season_mod), but should still build without error -- it's
    still reachable via backtest_harness.py's CLI for anyone who deliberately
    wants it."""
    model = models.build_team_season_mod(
        scenario['D'], scenario['coords'], scenario['season_offsets'], scenario['T'],
        scenario['boundary_cols'], scenario['position_of_player'], scenario['position_group_counts'],
    )
    _assert_finite_logp(model)
    free_names = {rv.name for rv in model.free_RVs}
    assert {'team_rho', 'team_sigma'} <= free_names


# --------------------------------------------------------- calendar/lookup

def test_build_ar_calendar():
    season_offsets, T, boundary_cols = models.build_ar_calendar([3, 2, 4])
    assert season_offsets.tolist() == [0, 3, 5, 9]
    assert T == 9
    # first real week of every season but the first, minus 1
    assert boundary_cols.tolist() == [2, 4]


def test_global_time():
    season_offsets, T, _ = models.build_ar_calendar([3, 2, 4])
    season_idx = np.array([0, 1, 2])
    week_idx = np.array([0, 1, 3])
    got = models.global_time(season_idx, week_idx, season_offsets)
    # season 0 starts at global step 0, season 1 at 3, season 2 at 5
    assert got.tolist() == [0, 4, 8]


def test_player_position_map():
    frame = pl.DataFrame({
        'player': ['a', 'a', 'b', 'c'],
        'position': ['QB', 'QB', 'RB', 'QB'],
    })
    enums = models.make_enums(frame, ['player', 'position'])
    n_positions = len(enums['position'].categories)
    pos_of_player, counts = models.player_position_map(frame, 'player', 'position', enums, n_positions)
    assert len(pos_of_player) == len(enums['player'].categories)
    assert counts.sum() == len(enums['player'].categories)


# ------------------------------------------------------------- data prep

def test_cont_array_matches_apply_scalers():
    """models.py keeps two parallel representations of the same
    standardization -- apply_scalers (adds `_z` columns to a polars frame,
    ff_ar.py's style) and cont_array (a raw numpy array,
    backtest_harness.py's style). They must compute identical numbers."""
    df = pl.DataFrame({'a': [1.0, 2.0, 3.0, 4.0], 'b': [10.0, 20.0, 30.0, 40.0]})
    scalers = models.fit_scalers(df, ['a', 'b'])
    via_apply = models.apply_scalers(df, scalers).select(['a_z', 'b_z']).to_numpy()
    via_cont = models.cont_array(df, scalers)
    np.testing.assert_allclose(via_apply, via_cont)


def test_fit_scalers_floors_zero_variance():
    """A constant column (possible in a small windowed backtest fold) must
    not produce a zero sd -- that would divide by zero downstream in
    apply_scalers/cont_array."""
    df = pl.DataFrame({'const': [5.0, 5.0, 5.0]})
    scalers = models.fit_scalers(df, ['const'])
    assert scalers['const'][1] == 1.0


def test_idx_array_matches_encode():
    """encode() (polars, adds `_idx` columns) and idx_array() (numpy, one
    column at a time) must agree -- backtest_harness.py's Fold uses
    idx_array to score a TEST frame against encoders fit on train."""
    df = pl.DataFrame({'team': ['b', 'a', 'c', 'a']})
    enums = models.make_enums(df, ['team'])
    via_encode = models.encode(df, ['team'], enums)['team_idx'].to_numpy()
    via_idx_array = models.idx_array(df, 'team', enums)
    np.testing.assert_array_equal(via_encode, via_idx_array)
