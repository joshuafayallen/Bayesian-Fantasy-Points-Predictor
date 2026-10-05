import numpy as np
import polars as pl
import pytest

from ffpred import decision_engine as de

N = 6000
POOL = [  # player, position, team, mean, sd, loading on a shared team factor
    ('QB Own', 'QB', 'BUF', 19.0, 6.5, 1.0), ('QB Bench', 'QB', 'DAL', 16.5, 6.0, 0.6),
    ('RB Bell-cow', 'RB', 'BUF', 17.0, 6.0, 1.0), ('RB Committee', 'RB', 'BUF', 10.5, 5.5, 0.8),
    ('RB Boom-bust', 'RB', 'DAL', 12.0, 8.5, 0.5), ('RB Safe', 'RB', 'KC', 11.5, 3.5, 0.3),
    ('WR1', 'WR', 'BUF', 15.5, 6.0, 0.9), ('WR2', 'WR', 'DAL', 13.0, 5.5, 0.6),
    ('WR3 Boom', 'WR', 'KC', 11.0, 8.0, 0.4), ('WR4 Floor', 'WR', 'SEA', 10.0, 3.5, 0.3),
    ('TE Elite', 'TE', 'KC', 12.0, 5.0, 0.6), ('TE Streamer', 'TE', 'SEA', 6.5, 3.5, 0.2),
]


@pytest.fixture(scope='module')
def scen():
    rng = np.random.default_rng(0)
    frame = pl.DataFrame(POOL, schema=['player', 'position', 'team', 'mean', 'sd', 'beta'], orient='row')
    frame = frame.with_columns(pl.Series('player_id', [f'id{i}' for i in range(frame.height)]),
                               pl.lit('OPP').alias('opp_team'))
    tf = {t: rng.normal(0, 1, N) for t in frame['team'].unique()}
    draws = np.column_stack([r['mean'] + r['beta'] * tf[r['team']] + rng.normal(0, r['sd'], N)
                             for r in frame.iter_rows(named=True)])
    return draws, frame


def _greedy(draws, frame):
    m = draws.mean(0)
    order = np.argsort(-m)
    pos = frame['position'].to_list()
    need = dict(de.ROSTER)
    picked = []
    for i in order:
        if need.get(pos[i], 0) > 0:
            need[pos[i]] -= 1
            picked.append(i)
    flex = next(i for i in order if i not in picked and pos[i] in de.FLEX_ELIGIBLE)
    return set(frame['player_id'][picked + [flex]].to_list())


def test_maximize_expected_equals_greedy(scen):
    draws, frame = scen
    ch = de.maximize_expected(draws, frame)
    assert set(ch.lineup['player_id'].to_list()) == _greedy(draws, frame)
    assert len(ch.lineup) == sum(de.ROSTER.values()) + de.N_FLEX


def test_cvar_lambda_zero_is_maximize_expected(scen):
    draws, frame = scen
    a = de.maximize_expected(draws, frame)
    b = de.cvar_averse(draws, frame, target=90, lam=0.0)
    assert set(a.lineup['player_id']) == set(b.lineup['player_id'])


def test_chance_constrained_feasible_and_infeasible(scen):
    draws, frame = scen
    ok = de.chance_constrained(draws, frame, floor=60, beta=0.9)
    assert ok.stats['feasible'] and ok.stats['p_clear'] >= 0.9
    bad = de.chance_constrained(draws, frame, floor=200, beta=0.9)
    assert not bad.stats['feasible'] and bad.stats['floor_at_beta'] < 200


def test_head_to_head_beats_or_matches_expected_lineup(scen):
    draws, frame = scen
    opp = de.lineup_distribution(draws, frame, ['id1', 'id4', 'id5', 'id8', 'id9', 'id11', 'id3'])['totals']
    h = de.head_to_head_optimize(draws, frame, opp)
    e = de.maximize_expected(draws, frame)
    assert h.stats['p_win'] >= de.p_beats(e.totals, opp) - 1e-12


def test_pool_and_name_resolution(scen):
    draws, frame = scen
    pool = ['QB Own', 'id2', 'id3', 'id6', 'id7', 'id10', 'id5']   # names and ids mixed
    ch = de.maximize_expected(draws, frame, pool=pool)
    assert set(ch.lineup['player_id']) == {'id0', 'id2', 'id3', 'id6', 'id7', 'id10', 'id5'}
    dup = frame.with_columns(pl.when(pl.col('player_id') == 'id1').then(pl.lit('QB Own'))
                             .otherwise(pl.col('player')).alias('player'))
    with pytest.raises(ValueError, match='player_ids'):
        de.resolve_player(dup, 'QB Own')


def test_thin_roster_raises(scen):
    draws, frame = scen
    with pytest.raises(ValueError, match='not enough QB'):
        de.maximize_expected(draws, frame, pool=['id2', 'id3', 'id6', 'id7', 'id10'])


def test_decide_dispatch(scen):
    draws, frame = scen
    pool = frame['player_id'].to_list()
    for rule in de.RULES:
        ch = de.decide(rule, draws, frame, pool, opp_pool=pool if rule == 'head_to_head' else None)
        assert len(ch.lineup) == 7
    with pytest.raises(ValueError):
        de.decide('nope', draws, frame, pool)


def test_pareto_points_share_one_scenario_split(scen, monkeypatch):
    draws, frame = scen
    seen = []
    real = de._prepare_scenarios

    def spy(d, n_search=None, val_frac=0.0, rng=None):
        out = real(d, n_search, val_frac, rng)
        seen.append(out[1][:5, 0].copy())       # first rows of the report set identify the split
        return out
    monkeypatch.setattr(de, '_prepare_scenarios', spy)
    de.pareto_frontier(draws, frame, targets=[85.0, 95.0], lambdas=[0.0, 1.0, 4.0], val_frac=0.3, rng=7)
    assert len(seen) == 6 and all(np.array_equal(seen[0], x) for x in seen)


def test_stats_reported_on_full_draws_when_search_is_subsampled(scen):
    draws, frame = scen
    ch = de.cvar_averse(draws, frame, target=90, lam=1.0, n_search_scenarios=500, rng=0)
    assert len(ch.totals) == draws.shape[0]
    assert ch.stats['expected'] == pytest.approx(de.lineup_distribution(draws, frame, ch.lineup['player_id'])['expected'])


def _flex_tie_scenario(gap):
    """A roster where the FLEX is a choice between two WRs `gap` points apart."""
    rng = np.random.default_rng(3)
    rows = [('Q', 'QB', 20), ('R1', 'RB', 15), ('R2', 'RB', 14), ('W1', 'WR', 15), ('W2', 'WR', 14),
            ('T', 'TE', 9), ('WA', 'WR', 10.0), ('WB', 'WR', 10.0 - gap)]
    frame = pl.DataFrame({'player_id': [r[0] for r in rows], 'player': [r[0] for r in rows],
                          'position': [r[1] for r in rows], 'team': ['X'] * 8, 'opp_team': ['Y'] * 8})
    draws = np.column_stack([rng.normal(m, 6, 4000) for *_, m in rows])
    return draws, frame


def test_near_tie_flagged_for_a_coin_flip():
    draws, frame = _flex_tie_scenario(gap=0.0)
    ch = de.maximize_expected(draws, frame)
    s = ch.stats
    assert s['runner_up'] is not None and s['near_tie']
    assert {s['runner_up_swap']} <= {'WA -> WB', 'WB -> WA'}
    assert s['objective_gap'] >= 0 and s['objective_gap_se'] > 0


def test_clear_choice_is_not_a_near_tie():
    draws, frame = _flex_tie_scenario(gap=4.0)
    s = de.maximize_expected(draws, frame).stats
    assert not s['near_tie'] and s['runner_up_swap'] == 'WA -> WB'
    assert s['objective_gap'] == pytest.approx(4.0, abs=0.6)


@pytest.mark.parametrize('rule', ['cvar_averse', 'chance_constrained', 'head_to_head'])
def test_every_rule_reports_a_runner_up(rule):
    draws, frame = _flex_tie_scenario(gap=0.0)
    pool = frame['player_id'].to_list()
    s = de.decide(rule, draws, frame, pool, opp_pool=pool if rule == 'head_to_head' else None).stats
    assert s['runner_up'] is not None and isinstance(s['near_tie'], bool)


def test_runner_up_differs_in_players_not_just_slots(scen):
    draws, frame = scen
    s = de.maximize_expected(draws, frame).stats
    best = set(de.maximize_expected(draws, frame).lineup['player_id'])
    assert set(s['runner_up']) != best
