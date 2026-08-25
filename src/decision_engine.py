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
and its companion script:
    https://github.com/fico-xpress/xpress-community/blob/main/StochasticEnergyPlanning/stochastic_energy_planning.py

The mapping from that post to fantasy lineups:

    energy planning                              fantasy lineup decision
    --------------------------------------------  --------------------------------------------
    demand scenarios (PyMC posterior draws)        player-week scenarios (this repo's PyMC draws)
    generation plan (decision variables)           which players start (decision variables)
    meet demand >= beta% of days (chance constr.)  clear a score floor >= beta% of scenarios
    CVaR of shortage cost (Rockafellar-Uryasev)    CVaR of points shortfall vs. a target/opponent
    Pareto frontier over lambda, clean-energy %    Pareto frontier over lambda, target level
    scenario subset for CVaR tractability           scenario subset for search speed
    Monte Carlo validation of shortage rate         held-out validation of p_clear / CVaR / expected
    FICO Xpress LP over generation mix              FICO Xpress MILP over which players start

One real difference from the energy script: season-long fantasy lineups
have no salary cap -- the "decision variables" are just which ~9 of your
~15 rostered players start, subject to slot constraints. That search space
is small enough (typically hundreds to low thousands of candidate lineups)
for EXACT enumeration in plain Python, so the primary decision rules below
(`maximize_expected`, `chance_constrained`, `cvar_averse`,
`head_to_head_optimize`) need no solver at all.

For pools too large to enumerate (dynasty benches, best-ball, DFS with a
salary cap) there is now a second backend, `*_milp`, that reformulates
lineup selection as a FICO Xpress MILP -- binary player-select variables
plus the same Rockafellar-Uryasev CVaR (or big-M chance-constraint)
machinery the energy script uses, just swapping "generation MW" for
"player selected". See section 5 below. `xpress_available()` reports
whether the optional dependency is installed; the enumeration path never
needs it.

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
DEFAULT_SEED = 41029041   # matches forecast_roster.forecast's default seed


# ------------------------------------------------------------- 0. plumbing

@dataclass
class LineupChoice:
    """Result of any decision rule below."""
    lineup: pl.DataFrame       # slot / player / position / team / opp
    totals: np.ndarray         # scenario draws of this lineup's TOTAL
                                # (on the REPORT scenario set -- see stats
                                # for validated / n_search / n_report)
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

    If your pool is too big for this (dynasty bench, best ball, DFS with a
    salary cap), don't raise max_candidates -- switch to the `*_milp`
    functions in section 5, which solve the same objectives with an ILP
    instead of enumerating.
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
            f"full player pool) -- this path does exact enumeration for "
            f"season-long start/sit among the ~15 players you own, not a "
            f"full-slate DFS search. If you genuinely need a bigger search "
            f"(e.g. DFS with a salary cap, or dynasty bench depth), use the "
            f"`*_milp` functions in section 5 instead -- same objectives, "
            f"solved as a FICO Xpress ILP over the same scenario-scored "
            f"data instead of enumerated."
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


def _label_selected(frame, chosen_idx, roster, flex_eligible):
    """Turn a bare set of chosen column indices (e.g. from the MILP backend,
    which only knows WHO is selected, not which slot they fill) into the
    same (slot_label, column_index) format all_candidate_lineups produces.
    Any valid assignment scores identically -- this is cosmetic labeling."""
    positions = frame['position'].to_list()
    by_pos = {}
    for i in chosen_idx:
        by_pos.setdefault(positions[i], []).append(i)

    slots, used = [], set()
    for pos, n in roster.items():
        take = by_pos.get(pos, [])[:n]
        label_multi = n > 1
        for k, idx in enumerate(take):
            slots.append((f"{pos}{k + 1}" if label_multi else pos, idx))
            used.add(idx)

    leftover = [i for i in chosen_idx if i not in used and positions[i] in flex_eligible]
    label_multi = len(leftover) > 1
    for k, idx in enumerate(leftover):
        slots.append((f"FLEX{k + 1}" if label_multi else "FLEX", idx))
        used.add(idx)

    # anything still unassigned (shouldn't happen if roster/flex accounting
    # is right, but surfaced rather than silently dropped)
    stragglers = [i for i in chosen_idx if i not in used]
    for k, idx in enumerate(stragglers):
        slots.append((f"EXTRA{k + 1}" if len(stragglers) > 1 else "EXTRA", idx))
    return slots


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


# --------------------------------------------------- 2. scenario management
#
# The energy script solves its CVaR LP on a random subset of demand
# scenarios ("N_SCENARIOS_CVAR = 1000 -- due to computational tractability"
# and the Xpress Community Edition's row/column cap), then validates the
# chosen plan's shortage_rate against the FULL scenario set with a separate
# Monte Carlo pass. Two distinct things are happening there that this
# module's original version conflated into one: (1) SEARCH on a scenario
# subset for speed, and (2) REPORT stats on scenarios the search never saw,
# so the reported p_clear / CVaR / expected total isn't just how well the
# winning lineup happened to fit the noise in the exact draws it was picked
# on. `_prepare_scenarios` makes both knobs explicit and independent.

def _prepare_scenarios(draws, n_search=None, val_frac=0.0, rng=None):
    """Split `draws` into a SEARCH set (used to pick the lineup) and a
    REPORT set (used to compute the stats attached to the winner).

    val_frac=0 (default): report set = the full `draws` (or the search
        subsample itself, if val_frac is 0 and n_search < len(draws)) --
        fast, matches the old behavior, but stats are in-sample.
    val_frac>0: draws are split into disjoint train/validation scenario
        blocks first, so report stats are honestly out-of-sample -- the
        fantasy analogue of the blog's Monte Carlo validation on the full
        demand-scenario set after solving on a subset.
    n_search: further subsample the train block to this many scenarios,
        for search speed on large draws matrices. Applied AFTER the val
        split, so validation scenarios are never touched during search.

    Returns (search_draws, report_draws, rng) -- the rng is returned so
    callers needing a second random draw (e.g. head_to_head's opponent
    pairing) can keep using the same stream.
    """
    rng = np.random.default_rng(rng)
    n = draws.shape[0]

    if val_frac and val_frac > 0:
        if not (0 < val_frac < 1):
            raise ValueError(f"val_frac must be in (0, 1), got {val_frac}")
        idx = rng.permutation(n)
        n_val = max(1, int(round(n * val_frac)))
        val_idx, train_idx = idx[:n_val], idx[n_val:]
        search, report = draws[train_idx], draws[val_idx]
    else:
        search = report = draws

    if n_search is not None and n_search < search.shape[0]:
        sel = rng.choice(search.shape[0], n_search, replace=False)
        search = search[sel]

    return search, report, rng


def _validated_stats(totals_search, totals_report, validated):
    """Shared in-sample vs. out-of-sample bookkeeping, attached to every
    rule's stats dict when val_frac > 0. `overfit_gap` is expressed as a
    fraction of the in-sample expected total -- a quick smell test for
    whether the winning lineup was chosen on noise. There's no fixed
    threshold at which this becomes "bad"; it's a number to look at, not a
    pass/fail, the same way the blog reports shortage_rate as information
    rather than a hard gate.
    """
    if not validated:
        return {}
    exp_is, exp_oos = float(totals_search.mean()), float(totals_report.mean())
    gap = (exp_is - exp_oos) / abs(exp_is) if exp_is else 0.0
    return {
        'validated': True,
        'n_search_scenarios': int(len(totals_search)),
        'n_report_scenarios': int(len(totals_report)),
        'expected_in_sample': exp_is,
        'expected_out_of_sample': exp_oos,
        'overfit_gap': float(gap),
    }


# ------------------------------------------------------------ 3. decision rules
#
# All four rules share the same shape: search on `search_draws` to pick a
# candidate, then report stats computed on `report_draws` -- pass
# val_frac=0 (default) to keep the old fast/in-sample behavior, or
# val_frac>0 for an honest held-out estimate before you trust a target-week
# decision to it. n_search_scenarios subsamples the search set for speed on
# very large draws matrices (mirrors N_SCENARIOS_CVAR in the blog).

def maximize_expected(draws, frame, roster=ROSTER, flex_eligible=FLEX_ELIGIBLE,
                      n_flex=N_FLEX, pool=None, max_candidates=MAX_CANDIDATES,
                      n_search_scenarios=None, val_frac=0.0, rng=None):
    """Maximize E[lineup total]. The point-forecast-equivalent baseline --
    what you'd get from ANY optimizer, Bayesian or not, if all you kept was
    a mean projection. Included as the reference point for the risk-aware
    rules below, and to demonstrate that it exactly reproduces
    forecast_roster.optimize_lineup(objective='expected')'s greedy fill
    (see __main__): linearity of expectation means correlation is
    irrelevant here, so greedy is exact, not approximate, for this one rule.
    Validation still matters for the REPORTED expected value, though --
    scenario noise in a small draws matrix can make one lineup look better
    than another by less than the sampling error.
    """
    search, report, _ = _prepare_scenarios(draws, n_search_scenarios, val_frac, rng)
    best = None
    for cand in all_candidate_lineups(frame, roster, flex_eligible, n_flex, pool, max_candidates):
        totals = candidate_totals(search, cand)
        val = totals.mean()
        if best is None or val > best[1]:
            best = (cand, val, totals)
    cand, val, totals_search = best
    totals_report = candidate_totals(report, cand) if val_frac else totals_search
    extra = _validated_stats(totals_search, totals_report, bool(val_frac))
    return _package(frame, cand, totals_report, objective_value=val,
                    rule='maximize_expected', **extra)


def chance_constrained(draws, frame, floor, beta=0.95, roster=ROSTER,
                       flex_eligible=FLEX_ELIGIBLE, n_flex=N_FLEX, pool=None,
                       max_candidates=MAX_CANDIDATES, n_search_scenarios=None,
                       val_frac=0.0, rng=None):
    """Mirrors the blog's chance-constrained model. Among lineups that clear
    `floor` points at least `beta` fraction of search scenarios, return the
    one with the highest expected total. If NO lineup clears the bar, fall
    back to the highest-P(clear) lineup and set `feasible=False` -- so an
    unreachable target is surfaced, not silently swallowed into "well, this
    was the best we had." That fallback now also reports `floor_at_beta`:
    the highest floor the best-available lineup actually clears at your
    requested beta, i.e. "you asked for floor=X at beta=Y; the best lineup
    found clears floor_at_beta instead" -- the fantasy analogue of the
    blog's infeasibility error reporting max clean capacity vs. the demand
    that couldn't be met.

    floor : the points total you need (start/sit context: a bye-week
            replacement level, "how many points to keep my season alive").
    beta  : how often you need to clear it (0.95 = clear it in 95% of
            scenarios, same convention as the blog's beta=0.95 demand
            constraint).

    With val_frac>0, p_clear is reported both in-sample (on the scenarios
    the search used) and out-of-sample (on held-out scenarios it never
    saw) -- a lineup whose p_clear collapses out-of-sample was fit to noise
    in the search set, not a real edge.
    """
    search, report, _ = _prepare_scenarios(draws, n_search_scenarios, val_frac, rng)
    feasible, best_infeasible = [], None
    for cand in all_candidate_lineups(frame, roster, flex_eligible, n_flex, pool, max_candidates):
        totals = candidate_totals(search, cand)
        p_clear = float((totals >= floor).mean())
        record = (cand, totals, p_clear)
        if p_clear >= beta:
            feasible.append(record)
        if best_infeasible is None or p_clear > best_infeasible[2]:
            best_infeasible = record

    def _finish(cand, totals_search, p_clear_search, is_feasible):
        totals_report = candidate_totals(report, cand) if val_frac else totals_search
        p_clear_report = float((totals_report >= floor).mean())
        extra = _validated_stats(totals_search, totals_report, bool(val_frac))
        if val_frac:
            extra['p_clear_in_sample'] = p_clear_search
            extra['p_clear_out_of_sample'] = p_clear_report
        return _package(
            frame, cand, totals_report, objective_value=totals_report.mean(),
            rule='chance_constrained', feasible=is_feasible,
            p_clear=p_clear_report if val_frac else p_clear_search,
            floor=floor, beta=beta, **extra,
        )

    if not feasible:
        cand, totals_search, p_clear = best_infeasible
        result = _finish(cand, totals_search, p_clear, is_feasible=False)
        # actionable diagnostic: what floor DOES this lineup clear at beta?
        result.stats['floor_at_beta'] = float(
            np.percentile(totals_search, 100 * (1 - beta))
        )
        return result

    cand, totals_search, p_clear = max(feasible, key=lambda r: r[1].mean())
    return _finish(cand, totals_search, p_clear, is_feasible=True)


def cvar_averse(draws, frame, target, alpha=0.05, lam=1.0, roster=ROSTER,
                flex_eligible=FLEX_ELIGIBLE, n_flex=N_FLEX, pool=None,
                max_candidates=MAX_CANDIDATES, n_search_scenarios=None,
                val_frac=0.0, rng=None):
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

    n_search_scenarios subsamples the search set the way the blog's
    N_SCENARIOS_CVAR does for its LP -- CVaR here costs O(candidates x
    scenarios), so this matters more for cvar_averse than for the other
    rules once draws gets into the tens of thousands. val_frac>0 then
    reports CVaR/expected on scenarios the search never touched.
    """
    search, report, _ = _prepare_scenarios(draws, n_search_scenarios, val_frac, rng)
    best = None
    for cand in all_candidate_lineups(frame, roster, flex_eligible, n_flex, pool, max_candidates):
        totals = candidate_totals(search, cand)
        loss = shortfall(totals, target)
        risk = cvar(loss, alpha=alpha)
        val = totals.mean() - lam * risk
        if best is None or val > best[1]:
            best = (cand, val, totals, risk)
    cand, val, totals_search, risk_search = best

    totals_report = candidate_totals(report, cand) if val_frac else totals_search
    risk_report = cvar(shortfall(totals_report, target), alpha=alpha)
    extra = _validated_stats(totals_search, totals_report, bool(val_frac))
    if val_frac:
        extra['cvar_in_sample'] = risk_search
        extra['cvar_out_of_sample'] = risk_report
    return _package(frame, cand, totals_report, objective_value=val, rule='cvar_averse',
                    cvar=risk_report if val_frac else risk_search,
                    target=target, alpha=alpha, lam=lam, **extra)


def head_to_head_optimize(draws, frame, opp_totals, roster=ROSTER,
                          flex_eligible=FLEX_ELIGIBLE, n_flex=N_FLEX, pool=None,
                          max_candidates=MAX_CANDIDATES, n_search_scenarios=None,
                          val_frac=0.0, rng=None):
    """Maximize P(your total > opponent's total) directly. This is the
    fantasy-specific version of a chance constraint where the "floor" is
    itself a random variable (your opponent's score) instead of a fixed
    number -- there's no blog analogue because energy demand doesn't have
    an adversary, but it's the single most useful rule for head-to-head
    leagues.

    `opp_totals` should come from scoring the opponent's actual (or
    presumed) lineup with forecast_roster.lineup_distribution(...)['totals'],
    drawn from the SAME posterior samples as `draws` (same model, same
    week) so the comparison is correctly paired -- shared uncertainty
    (weather, game script, the officiating crew having a weird night)
    cancels out where it should. If `val_frac>0`, the train/validation
    scenario split is applied to `opp_totals` with the SAME row indices as
    `draws`, so pairing survives the split; opp_totals must therefore be
    the same length as draws (pass the full array, not a pre-truncated
    one). If it's a different length/source, this falls back to truncating
    to the shorter length and treating the two as independent, same
    convention as forecast_roster.p_beats -- but then val_frac is not
    supported (raises) since there's no shared index to split on.
    """
    opp_totals = np.asarray(opp_totals)
    paired = len(opp_totals) == draws.shape[0]

    if val_frac and not paired:
        raise ValueError(
            "val_frac>0 requires opp_totals to be paired with draws "
            f"(same length, {draws.shape[0]}); got len(opp_totals)="
            f"{len(opp_totals)}. Pass the full opponent totals array, or "
            "use val_frac=0."
        )

    if paired:
        rng = np.random.default_rng(rng)
        n = draws.shape[0]
        if val_frac and val_frac > 0:
            idx = rng.permutation(n)
            n_val = max(1, int(round(n * val_frac)))
            val_idx, train_idx = idx[:n_val], idx[n_val:]
            search_draws, report_draws = draws[train_idx], draws[val_idx]
            search_opp, report_opp = opp_totals[train_idx], opp_totals[val_idx]
        else:
            search_draws = report_draws = draws
            search_opp = report_opp = opp_totals
        if n_search_scenarios is not None and n_search_scenarios < search_draws.shape[0]:
            sel = rng.choice(search_draws.shape[0], n_search_scenarios, replace=False)
            search_draws, search_opp = search_draws[sel], search_opp[sel]
    else:
        search_draws = report_draws = draws
        search_opp = report_opp = opp_totals

    best = None
    for cand in all_candidate_lineups(frame, roster, flex_eligible, n_flex, pool, max_candidates):
        totals = candidate_totals(search_draws, cand)
        n = min(len(totals), len(search_opp))
        p_win = float((totals[:n] > search_opp[:n]).mean())
        if best is None or p_win > best[1]:
            best = (cand, p_win, totals)
    cand, p_win_search, totals_search = best

    totals_report = candidate_totals(report_draws, cand) if val_frac else totals_search
    n = min(len(totals_report), len(report_opp))
    p_win_report = float((totals_report[:n] > report_opp[:n]).mean())
    extra = _validated_stats(totals_search, totals_report, bool(val_frac))
    if val_frac:
        extra['p_win_in_sample'] = p_win_search
        extra['p_win_out_of_sample'] = p_win_report
    return _package(frame, cand, totals_report, objective_value=p_win_report if val_frac else p_win_search,
                    rule='head_to_head', p_win=p_win_report if val_frac else p_win_search, **extra)


# ------------------------------------------------------- 4. Pareto frontier

def pareto_frontier(draws, frame, targets, alpha=0.05, lambdas=None,
                    val_frac=0.0, n_search_scenarios=None, rng=None, **kw):
    """Sweep lambda (and, unlike the old 1-D version, `targets`) through
    `cvar_averse` and report expected points vs. tail risk at each step --
    the fantasy analogue of the blog's Pareto analysis, which sweeps
    CVAR_WEIGHT across multiple clean-energy policy levels
    (clean_levels x cvar_weights) rather than a single fixed policy. Here
    `targets` plays the role of "policy level" (a floor/opponent-score
    target you might tighten or relax) and `lambdas` plays the role of
    CVAR_WEIGHT. Read this before committing to a lambda: it tells you
    exactly how many expected points a given amount of downside protection
    costs THIS week, for THIS lineup pool and THIS target, rather than
    picking lambda=1 out of the air.

    `targets` may be a single number (old call signature, single-target
    sweep) or a sequence of numbers (new: full grid over targets x
    lambdas). Every row uses the SAME random scenario split when
    val_frac>0 (one `rng` seeded once, reused across the whole sweep) so
    frontier points are directly comparable to each other, the same way
    the blog samples its CVaR scenario subset once per clean-energy level
    and reuses it across the whole weight sweep.
    """
    if lambdas is None:
        lambdas = np.concatenate([[0.0], np.geomspace(0.1, 20.0, 12)])
    if np.isscalar(targets):
        targets = [targets]
    rng = np.random.default_rng(rng)  # fixed stream, reused across the whole sweep

    rows = []
    for target in targets:
        for lam in lambdas:
            choice = cvar_averse(draws, frame, target=target, alpha=alpha,
                                 lam=float(lam), val_frac=val_frac,
                                 n_search_scenarios=n_search_scenarios,
                                 rng=rng, **kw)
            rows.append({
                'target': float(target),
                'lambda': float(lam),
                'expected': choice.stats['expected'],
                'cvar_shortfall': choice.stats['cvar'],
                'p_below_target': float((choice.totals < target).mean()),
                'validated': choice.stats.get('validated', False),
                'overfit_gap': choice.stats.get('overfit_gap'),
                'lineup': '+'.join(choice.lineup['player'].to_list()),
            })
    return pl.DataFrame(rows)


# ---------------------------------------------- 5. MILP backend (optional)
#
# Everything above does exact enumeration, which is the right call for a
# season-long roster (no salary cap, ~15 players -> hundreds to low
# thousands of candidates -- see the module docstring). It stops being the
# right call for a pool too large to enumerate: a deep dynasty bench, a
# best-ball roster, or DFS with a salary cap. This section reformulates the
# same two risk-aware objectives (chance-constrained, CVaR) as a FICO
# Xpress MILP -- binary player-select variables standing in for the
# energy script's continuous generation-by-source variables, plus the SAME
# Rockafellar-Uryasev CVaR constraints and (for chance-constrained) a
# big-M per-scenario indicator. `xpress_available()` reports whether the
# optional `xpress` dependency is installed; nothing above this section
# needs it.
#
# FICO Xpress's free Community Edition needs no licence file for problems
# this size (a lineup-selection MILP plus a few hundred CVaR scenarios
# stays well under its 5,000 row/column cap) -- `pip install xpress` is
# enough. The blog's Approach 2 (CVaR) only needed a full licence because
# it optimizes a full year of daily generation across 5 sources
# simultaneously; a single week's lineup selection is much smaller.

class XpressUnavailable(RuntimeError):
    """Raised when a *_milp function is called without xpress installed,
    or when the problem exceeds the Community Edition's row/column limit."""


def xpress_available() -> bool:
    try:
        import xpress  # noqa: F401
        return True
    except ImportError:
        return False


def _require_xpress():
    try:
        import xpress as xp
    except ImportError as e:
        raise XpressUnavailable(
            "xpress is not installed. `pip install xpress` -- the free "
            "Community Edition (no licence file needed) covers problems "
            "this small; only needed for the *_milp functions. The default "
            "enumeration-based rules (chance_constrained, cvar_averse, "
            "maximize_expected, head_to_head_optimize) need no solver."
        ) from e
    return xp


def _milp_controls(prob) -> None:
    """Standard solver controls -- analogue of the blog's _solver_controls,
    minus the barrier-method LP tuning (this is a MILP; branch-and-bound
    dominates runtime, not the LP relaxation)."""
    prob.controls.outputlog = 0
    prob.controls.miprelstop = 1e-4


def _milp_lineup_vars(xp, prob, frame, roster, flex_eligible, pool, n_flex):
    """Binary select[i] per player plus the roster/flex/pool constraints,
    shared by both *_milp functions. Encodes exactly the same feasible
    region as all_candidate_lineups: dedicated positions that are NOT
    flex-eligible must hit their slot count exactly; flex-eligible
    positions must hit AT LEAST their dedicated minimum (the surplus fills
    FLEX); total roster size is fixed."""
    positions = frame['position'].to_list()
    names = frame['player'].to_list()
    n_players = len(names)
    wanted = set(roster) | set(flex_eligible)
    pool_set = set(pool) if pool is not None else None

    x = prob.addVariables(n_players, vartype=xp.binary, name='select')

    eligible = set()
    for i in range(n_players):
        ok = positions[i] in wanted and (pool_set is None or names[i] in pool_set)
        if ok:
            eligible.add(i)
        else:
            prob.addConstraint(x[i] == 0)

    non_flex = [p for p in roster if p not in flex_eligible]
    flex_positions = [p for p in roster if p in flex_eligible]

    for p in non_flex:
        idxs = [i for i in eligible if positions[i] == p]
        if len(idxs) < roster[p]:
            raise ValueError(f"not enough {p} in pool: need {roster[p]}, have {len(idxs)}")
        prob.addConstraint(xp.Sum(x[i] for i in idxs) == roster[p])

    for p in flex_positions:
        idxs = [i for i in eligible if positions[i] == p]
        if len(idxs) < roster[p]:
            raise ValueError(f"not enough {p} in pool: need {roster[p]}, have {len(idxs)}")
        prob.addConstraint(xp.Sum(x[i] for i in idxs) >= roster[p])

    flex_capacity = sum(1 for i in eligible if positions[i] in flex_eligible)
    flex_dedicated = sum(roster.get(p, 0) for p in flex_eligible)
    if flex_capacity - flex_dedicated < n_flex:
        raise ValueError(
            f"not enough FLEX-eligible players left after filling fixed "
            f"slots: need {n_flex} from {flex_eligible}, only "
            f"{flex_capacity - flex_dedicated} would remain."
        )

    total_size = sum(roster.values()) + n_flex
    prob.addConstraint(xp.Sum(x) == total_size)
    return x


def cvar_averse_milp(draws, frame, target, alpha=0.05, lam=1.0, roster=ROSTER,
                     flex_eligible=FLEX_ELIGIBLE, n_flex=N_FLEX, pool=None,
                     n_scenarios=300, val_frac=0.5, rng=None):
    """MILP version of `cvar_averse`, for pools too large to enumerate.
    Same Rockafellar-Uryasev formulation as the blog's solve_cvar:

        z + aux[s] >= shortfall[s] - z  (z free = VaR, aux[s] >= 0)
        CVaR = z + (1 / (n_scenarios * alpha)) * sum(aux)
        maximize  E[total] - lambda * CVaR

    `n_scenarios` defaults to 300 (vs. the blog's 1000) to comfortably fit
    the Xpress Community Edition's 5,000 row/column limit alongside the
    roster constraints -- raise it if you have a full licence.
    `val_frac=0.5` by default (unlike the enumeration path's val_frac=0)
    because this backend exists specifically for larger, noisier searches
    where an honest out-of-sample estimate matters more.
    """
    xp = _require_xpress()
    search, report, _ = _prepare_scenarios(draws, n_search=n_scenarios,
                                           val_frac=val_frac, rng=rng)
    n_s, n_players = search.shape

    prob = xp.problem(name='CVaR_Lineup_MILP')
    _milp_controls(prob)
    x = _milp_lineup_vars(xp, prob, frame, roster, flex_eligible, pool, n_flex)

    z = prob.addVariable(name='var', lb=-xp.infinity)
    aux = prob.addVariables(n_s, lb=0, name='aux')
    shortfall_var = prob.addVariables(n_s, lb=0, name='shortfall')
    total = [xp.Sum(float(search[s, i]) * x[i] for i in range(n_players)) for s in range(n_s)]

    prob.addConstraint(shortfall_var[s] >= target - total[s] for s in range(n_s))
    prob.addConstraint(aux[s] >= shortfall_var[s] - z for s in range(n_s))

    col_means = search.mean(axis=0)
    mean_pts = xp.Sum(float(col_means[i]) * x[i] for i in range(n_players))
    cvar_term = z + (1.0 / (n_s * alpha)) * xp.Sum(aux)
    prob.setObjective(mean_pts - lam * cvar_term, sense=xp.maximize)

    try:
        prob.optimize()
    except xp.SolverError as e:
        if 'too many rows and columns' in str(e):
            raise XpressUnavailable(
                f"CVaR MILP with n_scenarios={n_s} exceeds the Xpress "
                f"Community Edition's 5,000 row/column limit. Lower "
                f"n_scenarios (try 100-300), or get a full licence: "
                f"https://www.fico.com/en/fico-xpress-trial-and-licensing-options"
            ) from e
        raise

    if prob.attributes.solstatus not in (xp.SolStatus.OPTIMAL, xp.SolStatus.FEASIBLE):
        raise RuntimeError(
            f"CVaR MILP infeasible or no solution found "
            f"(solstatus={prob.attributes.solstatus}) -- check that `pool` "
            f"has enough players at each position."
        )

    sol = prob.getSolution(x)
    chosen = [i for i, v in enumerate(sol) if v > 0.5]
    cand = _label_selected(frame, chosen, roster, flex_eligible)

    totals_search = candidate_totals(search, cand)
    totals_report = candidate_totals(report, cand) if val_frac else totals_search
    risk_report = cvar(shortfall(totals_report, target), alpha=alpha)
    risk_search = cvar(shortfall(totals_search, target), alpha=alpha)
    extra = _validated_stats(totals_search, totals_report, bool(val_frac))
    if val_frac:
        extra['cvar_in_sample'] = risk_search
        extra['cvar_out_of_sample'] = risk_report
    obj_val = float(totals_search.mean() - lam * risk_search)
    return _package(frame, cand, totals_report, objective_value=obj_val,
                    rule='cvar_averse_milp',
                    cvar=risk_report if val_frac else risk_search,
                    target=target, alpha=alpha, lam=lam, **extra)


def chance_constrained_milp(draws, frame, floor, beta=0.95, roster=ROSTER,
                            flex_eligible=FLEX_ELIGIBLE, n_flex=N_FLEX, pool=None,
                            n_scenarios=200, val_frac=0.5, rng=None):
    """MILP version of `chance_constrained`, for pools too large to
    enumerate. Uses a big-M per-scenario binary indicator:

        clear[s] == 1  =>  total[s] >= floor        (via big-M)
        sum(clear) >= ceil(beta * n_scenarios)
        maximize E[total]  subject to the above

    Unlike the enumeration-based `chance_constrained`, this does NOT
    gracefully degrade to "best achievable p_clear" when the target is
    infeasible -- the beta constraint is hard, so an infeasible target
    raises. That's a real limitation of the big-M scenario-indicator MILP
    (relaxing it would mean re-solving with a smaller beta, not a free
    fallback). If you need the graceful-degrade behavior and your pool is
    small enough, use `chance_constrained` instead.

    `n_scenarios` defaults to 200 (smaller than CVaR's 300) because each
    scenario adds a full binary variable here, not just a continuous one --
    this formulation is more expensive per scenario than the CVaR MILP.
    """
    xp = _require_xpress()
    search, report, _ = _prepare_scenarios(draws, n_search=n_scenarios,
                                           val_frac=val_frac, rng=rng)
    n_s, n_players = search.shape

    prob = xp.problem(name='ChanceConstrained_Lineup_MILP')
    _milp_controls(prob)
    x = _milp_lineup_vars(xp, prob, frame, roster, flex_eligible, pool, n_flex)

    total = [xp.Sum(float(search[s, i]) * x[i] for i in range(n_players)) for s in range(n_s)]
    clear = prob.addVariables(n_s, vartype=xp.binary, name='clear')
    big_m = float(search.sum(axis=1).max()) + 1.0
    prob.addConstraint(total[s] >= floor - big_m * (1 - clear[s]) for s in range(n_s))
    min_clear = math.ceil(beta * n_s)
    prob.addConstraint(xp.Sum(clear) >= min_clear)

    col_means = search.mean(axis=0)
    mean_pts = xp.Sum(float(col_means[i]) * x[i] for i in range(n_players))
    prob.setObjective(mean_pts, sense=xp.maximize)

    try:
        prob.optimize()
    except xp.SolverError as e:
        if 'too many rows and columns' in str(e):
            raise XpressUnavailable(
                f"chance-constrained MILP with n_scenarios={n_s} exceeds "
                f"the Xpress Community Edition's 5,000 row/column limit. "
                f"Lower n_scenarios (try 50-200), or get a full licence."
            ) from e
        raise

    if prob.attributes.solstatus not in (xp.SolStatus.OPTIMAL, xp.SolStatus.FEASIBLE):
        raise RuntimeError(
            f"chance-constrained MILP infeasible for floor={floor}, "
            f"beta={beta} at n_scenarios={n_s} (solstatus="
            f"{prob.attributes.solstatus}). Lower beta/floor, raise "
            f"n_scenarios for a tighter estimate, or fall back to "
            f"chance_constrained() (exact enumeration reports the "
            f"best-achievable p_clear and a floor_at_beta diagnostic even "
            f"when the exact target is infeasible, instead of raising)."
        )

    sol = prob.getSolution(x)
    chosen = [i for i, v in enumerate(sol) if v > 0.5]
    cand = _label_selected(frame, chosen, roster, flex_eligible)

    totals_search = candidate_totals(search, cand)
    totals_report = candidate_totals(report, cand) if val_frac else totals_search
    p_clear_search = float((totals_search >= floor).mean())
    p_clear_report = float((totals_report >= floor).mean())
    extra = _validated_stats(totals_search, totals_report, bool(val_frac))
    if val_frac:
        extra['p_clear_in_sample'] = p_clear_search
        extra['p_clear_out_of_sample'] = p_clear_report
    return _package(frame, cand, totals_report, objective_value=totals_report.mean(),
                    rule='chance_constrained_milp', feasible=True,
                    p_clear=p_clear_report if val_frac else p_clear_search,
                    floor=floor, beta=beta, **extra)


# ------------------------------------------------------------------ example

if __name__ == '__main__':
    # Synthetic-but-structured scenario data: a shared weekly "game
    # environment" factor plus player-specific noise, so that teammates and
    # opponents-of-the-same-defense are genuinely correlated -- the exact
    # structure that makes greedy-per-slot wrong for the risk-aware rules
    # and right for maximize_expected. In real use, `draws`/`frame` come
    # from forecast_roster.forecast(model, idata, frame, ...) using this
    # repo's fitted ff-ar or ff-mlm posterior.
    rng = np.random.default_rng(DEFAULT_SEED)
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

    # 2. chance-constrained: "clear 85 points in 90% of scenarios", with an
    #    honest held-out validation split.
    cc = chance_constrained(draws, frame, floor=85.0, beta=0.90, pool=MY_ROSTER,
                            val_frac=0.4, rng=DEFAULT_SEED)
    print(f"chance-constrained (floor=85, beta=0.90): feasible={cc.stats['feasible']}, "
          f"p_clear(in-sample)={cc.stats.get('p_clear_in_sample'):.3f}, "
          f"p_clear(out-of-sample)={cc.stats.get('p_clear_out_of_sample'):.3f}, "
          f"E[total]={cc.stats['expected']:.1f}")
    print(f"  lineup: {cc.lineup['player'].to_list()}\n")

    # 2b. an infeasible target, to show the floor_at_beta diagnostic.
    cc_bad = chance_constrained(draws, frame, floor=140.0, beta=0.90, pool=MY_ROSTER)
    print(f"chance-constrained (floor=140, beta=0.90) -- deliberately too high: "
          f"feasible={cc_bad.stats['feasible']}, p_clear={cc_bad.stats['p_clear']:.3f}, "
          f"floor_at_beta={cc_bad.stats['floor_at_beta']:.1f} "
          f"(this is the floor the best lineup actually clears 90% of the time)\n")

    # 3. CVaR-averse against a target (e.g. last season's median winning score)
    TARGET = 90.0
    for lam in (0.0, 1.0, 4.0):
        c = cvar_averse(draws, frame, target=TARGET, alpha=0.10, lam=lam, pool=MY_ROSTER,
                        val_frac=0.4, rng=DEFAULT_SEED)
        print(f"cvar_averse lam={lam:<4}: E[total]={c.stats['expected']:.1f}  "
              f"CVaR(shortfall, oos)={c.stats['cvar']:.1f}  "
              f"P(total<{TARGET:.0f})={float((c.totals < TARGET).mean()):.3f}  "
              f"overfit_gap={c.stats['overfit_gap']:+.3f}  "
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

    # 5. Pareto frontier: what does tail safety cost, in expected points,
    #    across a couple of plausible weekly targets? (2-D sweep: the
    #    fantasy analogue of the blog's clean_levels x cvar_weights grid.)
    frontier = pareto_frontier(draws, frame, targets=[85.0, TARGET, 95.0],
                               alpha=0.10, pool=MY_ROSTER)
    print("pareto frontier (target, lambda -> expected pts vs. CVaR shortfall):")
    print(frontier.select(['target', 'lambda', 'expected', 'cvar_shortfall', 'p_below_target']))

    # 6. MILP backend, if xpress is installed -- solves the SAME cvar_averse
    #    objective as an ILP instead of enumeration, as a correctness check
    #    (small pool here so both should land on the same/an equally-good
    #    lineup) and a demo for when you outgrow enumeration.
    if xpress_available():
        print("\nxpress available -- cross-checking cvar_averse vs. cvar_averse_milp:")
        milp_choice = cvar_averse_milp(draws, frame, target=TARGET, alpha=0.10, lam=1.0,
                                       pool=MY_ROSTER, n_scenarios=250, val_frac=0.4,
                                       rng=DEFAULT_SEED)
        enum_choice = cvar_averse(draws, frame, target=TARGET, alpha=0.10, lam=1.0,
                                  pool=MY_ROSTER)
        print(f"  MILP lineup total (E, in-sample)  : {milp_choice.stats['expected_in_sample']:.1f}")
        print(f"  enum lineup total (E, full draws)  : {enum_choice.stats['expected']:.1f}")
        print(f"  MILP lineup : {sorted(milp_choice.lineup['player'].to_list())}")
        print(f"  enum lineup : {sorted(enum_choice.lineup['player'].to_list())}")
    else:
        print("\n(xpress not installed -- skipping MILP cross-check. "
              "`pip install xpress` to enable the *_milp functions.)")
