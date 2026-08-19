"""
Bayesian decision engine for fantasy lineup decisions.

This is the "PyMC -> decision under uncertainty" half of the pipeline that
forecast_roster.py's `optimize_lineup` deliberately punts on. That function
is a GREEDY fill: take the top-k players per slot by a per-player marginal
statistic (mean, p10, p90) and combine. Greedy fill is provably optimal for
one specific objective -- maximizing the lineup's EXPECTED total -- because
expectation is linear: E[sum of players] = sum of E[player], regardless of
how correlated the players are. So `optimize_lineup(objective='expected')`
is not an approximation, it's exact, and `maximize_expected` below exists
mainly to prove that (see the __main__ demo).

It stops being exact the moment the objective is a nonlinear functional of
the JOINT distribution of the lineup total -- which is every objective that
actually matters for a real decision:

  - "what floor do I clear beta% of the time"        (chance-constrained)
  - "how bad are my worst-case weeks"                (CVaR)
  - "what's my probability of beating this opponent"  (win probability)

None of these decompose across slots. Greedily stacking the five highest
p10 floors does NOT give you the lineup with the highest p10 floor on the
SUM, because correlation changes the shape of a sum's tail: a QB and his
own WR share game-environment variance and move together, which widens
(not narrows) the joint tail relative to treating them as independent.
Two bench options with the same marginal floor are not interchangeable if
one of them is correlated with the rest of your lineup and the other isn't.

So this module scores whole CANDIDATE LINEUPS on their joint scenario
draws, following the chance-constrained vs. CVaR framing from PyMC Labs'
"Probabilistic Forecasting for Optimization" post (energy generation
planning with FICO Xpress):
    https://www.pymc-labs.com/blog-posts/probabilistic-forecasting-optimization-under-uncertainty

The mapping from that post to fantasy lineups:

    energy planning                              fantasy lineup decision
    --------------------------------------------  --------------------------------------------
    demand scenarios (PyMC posterior draws)        player-week scenarios (this repo's PyMC draws)
    generation plan (decision variables)           which players start (decision variables)
    meet demand >= beta% of days (chance constr.)  clear a score floor >= beta% of scenarios
    CVaR of shortage cost (Rockafellar-Uryasev)    CVaR of points shortfall vs. a target/opponent
    Pareto frontier over lambda, clean-energy %    Pareto frontier over lambda (points vs. safety)

One real difference: energy planning is a big LP/MIP over a knapsack
(capacity x cost), so the blog needs FICO Xpress. Season-long fantasy
lineups have no salary cap -- the "decision variables" are just which
~9 of your ~15 rostered players start, subject to slot constraints. That
search space is small enough (typically hundreds to low thousands of
candidate lineups) for EXACT enumeration in plain Python. No ILP solver
needed here. (If you extend this to DFS with a salary cap, the enumeration
in `all_candidate_lineups` is exactly the thing you'd replace with an ILP
over the same scenario-scored objective -- everything downstream of it,
the CVaR/chance-constrained/head-to-head scoring, is unchanged.)

Expected inputs, matching forecast_roster.forecast()'s contract:
    draws : np.ndarray, shape (n_draws, n_players)
            posterior predictive scenarios; column i is player i, and row j
            across all columns is one internally-consistent scenario (same
            posterior draw) -- this is what lets sums/comparisons across
            players propagate correlation correctly.
    frame : polars.DataFrame, one row per player, aligned to draws' columns
            (row i <-> draws[:, i]). Needs at least 'player' and 'position';
            'team'/'opp_team' are used for display if present.
"""

import itertools
import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import polars as pl

from forecast_roster import ROSTER, FLEX_ELIGIBLE, N_FLEX

MAX_CANDIDATES = 200_000  # safety valve -- see all_candidate_lineups


# ------------------------------------------------------------- 0. plumbing

@dataclass
class LineupChoice:
    """Result of any decision rule below."""
    lineup: pl.DataFrame       # slot / player / position / team / opp
    totals: np.ndarray         # scenario draws of this lineup's TOTAL
    objective_value: float     # whatever the rule maximized
    stats: dict                # everything else worth printing/logging


def _position_pools(frame, roster, flex_eligible, pool=None):
    positions = frame['position'].to_list()
    names = frame['player'].to_list()
    if pool is not None:
        pool_set = set(pool)
        eligible_idx = [i for i, n in enumerate(names) if n in pool_set]
    else:
        eligible_idx = list(range(len(names)))
    wanted_positions = set(roster) | set(flex_eligible)
    return {
        pos: [i for i in eligible_idx if positions[i] == pos]
        for pos in wanted_positions
    }


def count_candidates(frame, roster=ROSTER, flex_eligible=FLEX_ELIGIBLE,
                     n_flex=N_FLEX, pool=None):
    """Upper bound on the number of candidate lineups, WITHOUT enumerating
    them -- cheap enough to call before committing to a full search. It's an
    upper bound (not exact) when flex-eligible positions overlap dedicated
    roster slots, because the true flex pool size depends on which specific
    players the fixed slots used, not just how many. Good enough for the
    safety-valve check in `all_candidate_lineups`.
    """
    by_pos = _position_pools(frame, roster, flex_eligible, pool)
    fixed = 1
    for pos, n in roster.items():
        k = len(by_pos.get(pos, []))
        if k < n:
            raise ValueError(f"not enough {pos} in pool: need {n}, have {k}")
        fixed *= math.comb(k, n)

    flex_universe = set()
    for p in flex_eligible:
        flex_universe.update(by_pos.get(p, []))
    used_in_fixed = sum(roster.get(p, 0) for p in flex_eligible)
    remaining_bound = max(len(flex_universe) - used_in_fixed, n_flex)
    flex_bound = math.comb(remaining_bound, n_flex)
    return fixed * flex_bound


def all_candidate_lineups(frame, roster=ROSTER, flex_eligible=FLEX_ELIGIBLE,
                          n_flex=N_FLEX, pool=None, max_candidates=MAX_CANDIDATES):
    """Generate every legal lineup as a list of (slot_label, column_index)
    pairs. Exact enumeration -- see module docstring for why that's the
    right call here (no salary cap) and how big this pool should be
    (`pool` = your actual roster, not the league's full player pool).
    """
    by_pos = _position_pools(frame, roster, flex_eligible, pool)
    for pos, n in roster.items():
        if len(by_pos.get(pos, [])) < n:
            raise ValueError(f"not enough {pos} in pool: need {n}, have {len(by_pos.get(pos, []))}")

    bound = count_candidates(frame, roster, flex_eligible, n_flex, pool)
    if bound > max_candidates:
        raise ValueError(
            f"~{bound:,} candidate lineups (upper bound), over max_candidates="
            f"{max_candidates:,}. Narrow `pool` to your actual roster (not the "
            f"full player pool) -- this engine does exact enumeration for "
            f"season-long start/sit among the ~15 players you own, not a "
            f"full-slate DFS search. If you genuinely need a bigger search "
            f"(e.g. DFS with a salary cap), swap this generator for an ILP "
            f"over the same scenario-scored objective."
        )

    # Exact (not bounded) FLEX feasibility check. However combo_tuple picks
    # its fixed slots, it always consumes exactly roster[p] players from
    # position p's pool -- so the number of flex-eligible players left over
    # is the SAME for every combo, and can be checked once, up front,
    # instead of discovering (possibly after iterating every fixed-slot
    # combination) that the generator has nothing left to yield. Without
    # this, a fully-consumed flex pool would make this generator silently
    # produce ZERO candidates, and every caller below would crash trying to
    # unpack an empty `best` instead of getting a clear, catchable error --
    # exactly what happens with a single-QB/thin-bench roster on a week
    # where a couple of bench-eligible players are on bye.
    if n_flex > 0:
        flex_capacity = (sum(len(by_pos.get(p, [])) for p in flex_eligible)
                        - sum(roster.get(p, 0) for p in flex_eligible))
        if flex_capacity < n_flex:
            raise ValueError(
                f"not enough FLEX-eligible players left after filling fixed "
                f"slots: need {n_flex} from {flex_eligible}, only "
                f"{flex_capacity} would remain after the fixed roster slots "
                f"are filled. (Typically a thin bench running out of "
                f"RB/WR/TE depth this week -- byes/injuries ate the pool.)"
            )

    positions_order = list(roster.keys())
    per_pos_combos = {
        pos: list(itertools.combinations(by_pos[pos], n))
        for pos, n in roster.items()
    }

    for combo_tuple in itertools.product(*(per_pos_combos[p] for p in positions_order)):
        used = set()
        slots = []
        for pos, combo in zip(positions_order, combo_tuple):
            label_multi = roster[pos] > 1
            for i, idx in enumerate(combo):
                label = f"{pos}{i + 1}" if label_multi else pos
                slots.append((label, idx))
                used.add(idx)

        flex_pool = sorted(
            (set().union(*(by_pos.get(p, []) for p in flex_eligible)) if flex_eligible else set())
            - used
        )
        if n_flex == 0:
            yield slots
            continue
        if len(flex_pool) < n_flex:
            continue
        for flex_combo in itertools.combinations(flex_pool, n_flex):
            flex_label_multi = n_flex > 1
            flex_slots = [
                (f"FLEX{i + 1}" if flex_label_multi else "FLEX", idx)
                for i, idx in enumerate(flex_combo)
            ]
            yield slots + flex_slots


def candidate_totals(draws, candidate):
    """Scenario totals for one candidate lineup -- paired sum across draws,
    so correlation between the chosen players is preserved."""
    cols = [idx for _, idx in candidate]
    return draws[:, cols].sum(axis=1)


def _package(frame, candidate, totals, objective_value, rule, **extra):
    have_team = 'team' in frame.columns
    have_opp = 'opp_team' in frame.columns or 'opp' in frame.columns
    opp_col = 'opp_team' if 'opp_team' in frame.columns else 'opp'
    rows = []
    for slot, idx in candidate:
        row = frame.row(idx, named=True)
        rows.append({
            'slot': slot,
            'player': row['player'],
            'position': row['position'],
            'team': row['team'] if have_team else None,
            'opp': row[opp_col] if have_opp else None,
        })
    lineup = pl.DataFrame(rows)
    stats = dict(
        rule=rule,
        objective_value=float(objective_value),
        expected=float(totals.mean()),
        median=float(np.median(totals)),
        sd=float(totals.std()),
        floor_p10=float(np.percentile(totals, 10)),
        ceiling_p90=float(np.percentile(totals, 90)),
        **extra,
    )
    return LineupChoice(lineup=lineup, totals=totals,
                        objective_value=float(objective_value), stats=stats)


# --------------------------------------------------------- 1. risk metrics

def shortfall(totals, target):
    """Points-below-target loss. Points ABOVE target cost nothing -- this is
    deliberately asymmetric, same as the energy-shortage cost in the blog:
    generating MORE than demand isn't penalized, falling short is."""
    return np.maximum(target - np.asarray(totals), 0.0)


def cvar(losses, alpha=0.05):
    """Conditional Value at Risk (Rockafellar-Uryasev): the average loss in
    the worst alpha-fraction of scenarios. alpha=0.05 -> average shortfall
    across your worst 5% of weeks. Higher `losses` values are worse."""
    losses = np.asarray(losses)
    var = np.percentile(losses, 100 * (1 - alpha))
    tail = losses[losses >= var]
    return float(tail.mean()) if len(tail) else float(var)


# ------------------------------------------------------------ 2. decision rules

def maximize_expected(draws, frame, roster=ROSTER, flex_eligible=FLEX_ELIGIBLE,
                      n_flex=N_FLEX, pool=None, max_candidates=MAX_CANDIDATES):
    """Maximize E[lineup total]. The point-forecast-equivalent baseline --
    what you'd get from ANY optimizer, Bayesian or not, if all you kept was
    a mean projection. Included as the reference point for the risk-aware
    rules below, and to demonstrate that it exactly reproduces
    forecast_roster.optimize_lineup(objective='expected')'s greedy fill
    (see __main__): linearity of expectation means correlation is
    irrelevant here, so greedy is exact, not approximate, for this one rule.
    """
    best = None
    for cand in all_candidate_lineups(frame, roster, flex_eligible, n_flex, pool, max_candidates):
        totals = candidate_totals(draws, cand)
        val = totals.mean()
        if best is None or val > best[1]:
            best = (cand, val, totals)
    cand, val, totals = best
    return _package(frame, cand, totals, objective_value=val, rule='maximize_expected')


def chance_constrained(draws, frame, floor, beta=0.95, roster=ROSTER,
                       flex_eligible=FLEX_ELIGIBLE, n_flex=N_FLEX, pool=None,
                       max_candidates=MAX_CANDIDATES):
    """Mirrors the blog's chance-constrained model. Among lineups that clear
    `floor` points at least `beta` fraction of scenarios, return the one
    with the highest expected total. If NO lineup clears the bar this week,
    fall back to the highest-P(clear) lineup and set `feasible=False` on
    the result -- so an unreachable target is surfaced, not silently
    swallowed into "well, this was the best we had."

    floor : the points total you need (start/sit context: a bye-week
            replacement level, "how many points to keep my season alive").
    beta  : how often you need to clear it (0.95 = clear it in 95% of
            scenarios, same convention as the blog's beta=0.95 demand
            constraint).
    """
    feasible, best_infeasible = [], None
    for cand in all_candidate_lineups(frame, roster, flex_eligible, n_flex, pool, max_candidates):
        totals = candidate_totals(draws, cand)
        p_clear = float((totals >= floor).mean())
        record = (cand, totals, p_clear)
        if p_clear >= beta:
            feasible.append(record)
        if best_infeasible is None or p_clear > best_infeasible[2]:
            best_infeasible = record

    if not feasible:
        cand, totals, p_clear = best_infeasible
        return _package(frame, cand, totals, objective_value=totals.mean(),
                        rule='chance_constrained', feasible=False,
                        p_clear=p_clear, floor=floor, beta=beta)

    cand, totals, p_clear = max(feasible, key=lambda r: r[1].mean())
    return _package(frame, cand, totals, objective_value=totals.mean(),
                    rule='chance_constrained', feasible=True,
                    p_clear=p_clear, floor=floor, beta=beta)


def cvar_averse(draws, frame, target, alpha=0.05, lam=1.0, roster=ROSTER,
                flex_eligible=FLEX_ELIGIBLE, n_flex=N_FLEX, pool=None,
                max_candidates=MAX_CANDIDATES):
    """Mirrors the blog's Rockafellar-Uryasev CVaR model:

        maximize   E[total]  -  lambda * CVaR_alpha( shortfall(total, target) )

    `target` is whatever "meeting demand" means for your decision this
    week: a must-clear threshold, a waiver-wire replacement level, or (more
    interestingly) your opponent's projected total. `lam` trades average
    points for tail safety -- lam=0 recovers `maximize_expected` exactly;
    larger lam buys a higher floor at the cost of expected points, the same
    cost/reliability tradeoff as the blog's chance-constrained-vs-CVaR
    comparison (8.5% more generation for a 22% cost increase, in exchange
    for cutting shortage risk from 5% to 0.35%). Use `pareto_frontier`
    below to see that tradeoff curve for your own lineup before picking a
    lambda.
    """
    best = None
    for cand in all_candidate_lineups(frame, roster, flex_eligible, n_flex, pool, max_candidates):
        totals = candidate_totals(draws, cand)
        loss = shortfall(totals, target)
        risk = cvar(loss, alpha=alpha)
        val = totals.mean() - lam * risk
        if best is None or val > best[1]:
            best = (cand, val, totals, risk)
    cand, val, totals, risk = best
    return _package(frame, cand, totals, objective_value=val, rule='cvar_averse',
                    cvar=risk, target=target, alpha=alpha, lam=lam)


def head_to_head_optimize(draws, frame, opp_totals, roster=ROSTER,
                          flex_eligible=FLEX_ELIGIBLE, n_flex=N_FLEX, pool=None,
                          max_candidates=MAX_CANDIDATES):
    """Maximize P(your total > opponent's total) directly. This is the
    fantasy-specific version of a chance constraint where the "floor" is
    itself a random variable (your opponent's score) instead of a fixed
    number -- there's no blog analogue because energy demand doesn't have
    an adversary, but it's the single most useful rule for head-to-head
    leagues.

    `opp_totals` should come from scoring the opponent's actual (or
    presumed) lineup with forecast_roster.lineup_distribution(...)['totals'].
    If it was drawn from the SAME posterior samples as `draws` (same model,
    same week), the comparison is correctly paired -- shared uncertainty
    (weather, game script, the officiating crew having a weird night)
    cancels out where it should. If it's a different length/source, this
    truncates to the shorter length and treats the two as independent,
    same convention as forecast_roster.p_beats.
    """
    opp_totals = np.asarray(opp_totals)
    best = None
    for cand in all_candidate_lineups(frame, roster, flex_eligible, n_flex, pool, max_candidates):
        totals = candidate_totals(draws, cand)
        n = min(len(totals), len(opp_totals))
        p_win = float((totals[:n] > opp_totals[:n]).mean())
        if best is None or p_win > best[1]:
            best = (cand, p_win, totals)
    cand, p_win, totals = best
    return _package(frame, cand, totals, objective_value=p_win,
                    rule='head_to_head', p_win=p_win)


# ------------------------------------------------------- 3. Pareto frontier

def pareto_frontier(draws, frame, target, alpha=0.05, lambdas=None, **kw):
    """Sweep lambda through `cvar_averse` and report expected points vs. tail
    risk at each step -- the fantasy analogue of the blog's Pareto analysis
    over CVaR weight and clean-energy policy. Read this before committing to
    a lambda: it tells you exactly how many expected points a given amount
    of downside protection costs THIS week, for THIS lineup pool, rather
    than picking lambda=1 out of the air.
    """
    if lambdas is None:
        lambdas = np.concatenate([[0.0], np.geomspace(0.1, 20.0, 12)])
    rows = []
    for lam in lambdas:
        choice = cvar_averse(draws, frame, target=target, alpha=alpha,
                             lam=float(lam), **kw)
        rows.append({
            'lambda': float(lam),
            'expected': choice.stats['expected'],
            'cvar_shortfall': choice.stats['cvar'],
            'p_below_target': float((choice.totals < target).mean()),
            'lineup': '+'.join(choice.lineup['player'].to_list()),
        })
    return pl.DataFrame(rows)


# ------------------------------------------------------------------ example

if __name__ == '__main__':
    # Synthetic-but-structured scenario data: a shared weekly "game
    # environment" factor plus player-specific noise, so that teammates and
    # opponents-of-the-same-defense are genuinely correlated -- the exact
    # structure that makes greedy-per-slot wrong for the risk-aware rules
    # and right for maximize_expected. In real use, `draws`/`frame` come
    # from forecast_roster.forecast(model, idata, frame, ...) using this
    # repo's fitted ff-ar or ff-mlm posterior.
    rng = np.random.default_rng(41029041)
    N_DRAWS = 20_000

    roster_pool = [
        # player,        position, team,  opp,   mean, sd,  team_beta (game-env loading)
        ('QB Own',       'QB', 'BUF', 'MIA', 19.0, 6.5, 1.0),
        ('QB Bench',     'QB', 'DAL', 'NYG', 16.5, 6.0, 0.6),

        ('RB Bell-cow',  'RB', 'BUF', 'MIA', 17.0, 6.0, 1.0),
        ('RB Committee', 'RB', 'BUF', 'MIA', 10.5, 5.5, 0.8),
        ('RB Boom-bust', 'RB', 'DAL', 'NYG', 12.0, 8.5, 0.5),
        ('RB Safe Floor','RB', 'KC',  'DEN', 11.5, 3.5, 0.3),
        ('RB Bench',     'RB', 'SEA', 'ARI',  9.0, 4.0, 0.3),

        ('WR1 Alpha',    'WR', 'BUF', 'MIA', 15.5, 6.0, 0.9),
        ('WR2 Solid',    'WR', 'DAL', 'NYG', 13.0, 5.5, 0.6),
        ('WR3 Boom',     'WR', 'KC',  'DEN', 11.0, 8.0, 0.4),
        ('WR4 Floor',    'WR', 'SEA', 'ARI', 10.0, 3.5, 0.3),
        ('WR5 Bench',    'WR', 'CIN', 'PIT',  8.5, 4.5, 0.3),

        ('TE Elite',     'TE', 'KC',  'DEN', 12.0, 5.0, 0.6),
        ('TE Streamer',  'TE', 'SEA', 'ARI',  6.5, 3.5, 0.2),
    ]
    frame = pl.DataFrame(roster_pool, schema=['player', 'position', 'team', 'opp',
                                              'mean', 'sd', 'beta'], orient='row')

    team_factor = {t: rng.normal(0, 1.0, N_DRAWS) for t in frame['team'].unique()}
    draws = np.column_stack([
        row['mean'] + row['beta'] * team_factor[row['team']] +
        rng.normal(0, row['sd'], N_DRAWS)
        for row in frame.iter_rows(named=True)
    ])
    draws = np.clip(draws, -2, None)  # fantasy points floor near zero

    MY_ROSTER = [p for p, *_ in roster_pool]  # in real use: your actual roster

    print(f"{len(MY_ROSTER)} rostered players -> "
          f"~{count_candidates(frame, pool=MY_ROSTER):,} candidate lineups\n")

    # 1. sanity check: maximize_expected should equal the greedy fill,
    #    because linearity of expectation makes greedy exact for this rule.
    from forecast_roster import optimize_lineup, lineup_distribution, p_beats
    summary = pl.DataFrame({
        'player': frame['player'], 'position': frame['position'],
        'team': frame['team'], 'opp': frame['opp'],
        'expected': draws.mean(axis=0),
        'floor_p10': np.percentile(draws, 10, axis=0),
        'ceiling_p90': np.percentile(draws, 90, axis=0),
        'sd': draws.std(axis=0),
    })
    greedy = optimize_lineup(summary, pool=MY_ROSTER, objective='expected')
    exact = maximize_expected(draws, frame, pool=MY_ROSTER)
    agree = set(greedy['player']) == set(exact.lineup['player'])
    print(f"greedy vs. exact maximize_expected agree: {agree}")
    print(f"  greedy E[total] (naive sum of means): {greedy['expected'].sum():.1f}")
    print(f"  exact  E[total] (from joint draws)  : {exact.stats['expected']:.1f}\n")

    # 2. chance-constrained: "clear 85 points in 90% of scenarios"
    cc = chance_constrained(draws, frame, floor=85.0, beta=0.90, pool=MY_ROSTER)
    print(f"chance-constrained (floor=85, beta=0.90): feasible={cc.stats['feasible']}, "
          f"p_clear={cc.stats['p_clear']:.3f}, E[total]={cc.stats['expected']:.1f}")
    print(f"  lineup: {cc.lineup['player'].to_list()}\n")

    # 3. CVaR-averse against a target (e.g. last season's median winning score)
    TARGET = 90.0
    for lam in (0.0, 1.0, 4.0):
        c = cvar_averse(draws, frame, target=TARGET, alpha=0.10, lam=lam, pool=MY_ROSTER)
        print(f"cvar_averse lam={lam:<4}: E[total]={c.stats['expected']:.1f}  "
              f"CVaR(shortfall)={c.stats['cvar']:.1f}  "
              f"P(total<{TARGET:.0f})={float((c.totals < TARGET).mean()):.3f}  "
              f"lineup={c.lineup['player'].to_list()}")
    print()

    # 4. head-to-head: build an opponent lineup from the bench-ish leftovers
    #    and score it on the SAME draws (paired uncertainty)
    opp_players = ['QB Bench', 'RB Boom-bust', 'RB Safe Floor', 'RB Bench',
                   'WR3 Boom', 'WR4 Floor', 'TE Streamer']
    opp_dist = lineup_distribution(draws, frame, pl.DataFrame({'player': opp_players}))
    h2h = head_to_head_optimize(draws, frame, opp_dist['totals'], pool=MY_ROSTER)
    print(f"head-to-head best P(win)={h2h.stats['p_win']:.3f}  "
          f"lineup={h2h.lineup['player'].to_list()}")
    print(f"  (vs. p_beats on the maximize_expected lineup: "
          f"{p_beats(exact.totals, opp_dist['totals']):.3f})\n")

    # 5. Pareto frontier: what does tail safety cost, in expected points?
    frontier = pareto_frontier(draws, frame, target=TARGET, alpha=0.10, pool=MY_ROSTER)
    print("pareto frontier (lambda -> expected pts vs. CVaR shortfall):")
    print(frontier.select(['lambda', 'expected', 'cvar_shortfall', 'p_below_target']))
