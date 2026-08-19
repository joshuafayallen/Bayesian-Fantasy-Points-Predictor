"""
Calibrate decision_engine.py against REAL, realized outcomes, walk-forward.

decision_engine.py's rules make three kinds of promises at decision time:
    chance_constrained  -- "this lineup clears `floor` in >= beta% of scenarios"
    cvar_averse         -- "average shortfall in the worst alpha% of scenarios is X"
    head_to_head        -- "this lineup beats the opponent with probability p"

Every one of those is a claim about a posterior predictive distribution
computed BEFORE the games are played. Calibration asks: when you make that
claim over and over across many real weeks, does reality agree? If you
pick lineups promised beta=0.90 clear-rate, do they actually clear the
floor ~90% of the time? If you pick the lineup with predicted P(win)=0.70,
do you actually win about 70% of the time? This is the same idea as the
cov50/cov80/cov90 columns already in results/backtest_walkforward.csv, just
applied to the DECISION (a whole chosen lineup, under a specific rule)
instead of to a single player's marginal prediction.

This only works walk-forward and out-of-sample: for each historical
(season, week), fit on everything strictly BEFORE that week (exactly the
windowed fold in sandbox/backtest_harness.py), get real posterior
predictive draws for that week's real players, run each decision_engine
rule on those draws, and then compare against what ACTUALLY happened
(total_fantasy_points, already in the data) once the week is over. Doing
this in-sample (fitting on data that includes the target week) would let
the model "know" the answer and make every rule look artificially
well-calibrated.

NO REAL FANTASY ROSTER EXISTS IN THIS DATASET (it's play-by-play/weekly
stats, not a specific league's rosters), so `simulate_roster` stands in
for "your team": the top-K players at each position by TRAILING mean
fantasy points, computed only from data strictly before the target week
(no leakage). A second, lower tier of the same ranking stands in for a
head-to-head opponent. This is obviously a simplification -- swap
`simulate_roster` for a real roster export if/when you have one from an
actual league.

Three things get scored, per fold, per rule:
  1. calibration   -- promised probability/risk vs. what actually happened
                       (aggregated across folds into reliability tables --
                       see `summarize`)
  2. economic value -- mean realized points and REGRET (points left on the
                       table vs. the best lineup obtainable from the same
                       pool with perfect hindsight -- `oracle_lineup`)
  3. resumability   -- one row per (season, week, rule), checkpointed to
                       CSV exactly like sandbox/backtest_harness.py, so a
                       sweep can be run fold-by-fold from a shell loop and
                       resumed if interrupted.

Usage (run from the project root):
    python src/calibrate_decisions.py 2025 12
    python src/calibrate_decisions.py 2025 14 --beta 0.85 --lam 2.0
    ...
    python src/calibrate_decisions.py --summarize   # after several folds

Expected cost per fold: same as backtest_harness's ADVI path, ~25-30s on a
normal workstation (see that module's docstring for the sandbox-specific
NUTS cost caveat -- use ADVI here for the same reason).
"""

import os
import sys
import argparse

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                 # src/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'sandbox'))

import decision_engine as de
from decision_engine import ROSTER, FLEX_ELIGIBLE, N_FLEX

DATA_PATH = "processed-data/ff-processed.parquet"
OUT_CSV = "results/decision_calibration.csv"
WINDOW_SEASONS = 3   # same windowing convention as sandbox/backtest_harness.py

DEFAULT_ROSTER_DEPTH = {'QB': 2, 'RB': 5, 'WR': 5, 'TE': 2}   # ~14 "rostered" players


# --------------------------------------------------------- simulated roster

def simulate_roster(history, per_position=None, tier=0, min_games=3):
    """Top-K players per position by TRAILING mean fantasy points in
    `history` (must already be filtered to strictly-before-target-week data
    -- this function does not do that filtering itself, to keep it honest
    about where leakage could sneak in).

    tier=0 takes the top `per_position[pos]` at each position (your
    simulated roster). tier=1 takes the NEXT `per_position[pos]` players at
    each position -- i.e. the offset is applied PER POSITION, not as one
    global cutoff -- to stand in for a league opponent of the same roster
    shape but one tier down in talent.
    """
    per_position = per_position or DEFAULT_ROSTER_DEPTH
    agg = (
        history.group_by(['player', 'position'])
        .agg(pl.col('total_fantasy_points').mean().alias('trail_mean'),
             pl.len().alias('n_games'))
        .filter(pl.col('n_games') >= min_games)
    )
    roster = []
    for pos, k in per_position.items():
        offset = tier * k
        top = (
            agg.filter(pl.col('position') == pos)
            .sort('trail_mean', descending=True)
            .slice(offset, k)
        )
        roster += top['player'].to_list()
    return roster


# ------------------------------------------------------------- realized scoring

def realized_total(keep, players):
    """Sum of ACTUAL total_fantasy_points for a chosen set of players, read
    straight from the (already-played) test week -- ground truth, no draws
    involved."""
    if not players:
        return 0.0
    return float(keep.filter(pl.col('player').is_in(players))['total_fantasy_points'].sum())


def oracle_lineup(keep, pool, roster=ROSTER, flex_eligible=FLEX_ELIGIBLE, n_flex=N_FLEX,
                  max_candidates=de.MAX_CANDIDATES):
    """Best lineup obtainable from `pool`, scored on REALIZED (not
    predicted) points -- a perfect-hindsight upper bound. The gap between
    this and what a decision rule actually picked (using only information
    available BEFORE the week) is that rule's regret for this fold.

    Reuses decision_engine's own combinatorial enumeration so the oracle
    respects exactly the same slot/FLEX constraints as every other rule --
    it would be a meaningless comparison against an oracle allowed to break
    the roster shape.
    """
    y = keep['total_fantasy_points'].to_numpy()
    best = None
    for cand in de.all_candidate_lineups(keep, roster, flex_eligible, n_flex, pool, max_candidates):
        idx = [i for _, i in cand]
        val = float(y[idx].sum())
        if best is None or val > best[1]:
            best = (cand, val)
    cand, val = best
    players = [keep.row(i, named=True)['player'] for _, i in cand]
    return players, val


# ------------------------------------------------------------------ one fold

def already_done(out_csv, season, week):
    if not os.path.exists(out_csv):
        return False
    existing = pl.read_csv(out_csv)
    return existing.filter(
        (pl.col('season') == season) & (pl.col('week') == week)
    ).height > 0


def run_one_fold(season, week, out_csv=OUT_CSV, floor=None, beta=0.90, target=None,
                 alpha=0.10, lam=1.0, roster_depth=None, seed=0, method='advi',
                 advi_n=6000, advi_draws=400, nuts_draws=300, nuts_tune=300, nuts_chains=2,
                 data_path=DATA_PATH):
    # NOTE: already_done() only keys on (season, week) -- NOT method. Running
    # method='nuts' into the SAME out_csv you already used for method='advi'
    # folds will look "already done" and silently skip every fold. Point
    # --out at a fresh CSV (e.g. results/decision_calibration_nuts.csv) when
    # comparing ADVI vs NUTS for the same (season, week), the same way
    # run_param_sweep.sh gives every swept value its own file.
    if already_done(out_csv, season, week):
        print(f"SKIP: {season} wk{week} already in {out_csv}")
        return

    # local import: only needed here, and only ever runs on a real pymc/pytensor
    # install (see module docstring) -- kept out of the top-level imports so the
    # rest of this file (simulate_roster, oracle_lineup, summarize) stays testable
    # without that dependency.
    from backtest_harness import fit_predict, past_only_baseline  # noqa: F401

    df = pl.read_parquet(data_path)
    train = df.filter(
        (pl.col('season') >= season - (WINDOW_SEASONS - 1)) &
        ((pl.col('season') < season) | ((pl.col('season') == season) & (pl.col('week') < week)))
    )
    test = df.filter((pl.col('season') == season) & (pl.col('week') == week))
    if len(test) == 0 or len(train) < 200:
        print(f"SKIP: no data for {season} wk{week}")
        return

    r = fit_predict('ar', method, train, test, seed=seed, advi_n=advi_n, advi_draws=advi_draws,
                    nuts_draws=nuts_draws, nuts_tune=nuts_tune, nuts_chains=nuts_chains)
    if r is None:
        print(f"SKIP: all test rows were cold starts for {season} wk{week}")
        return
    keep, S = r['keep'], r['S']

    rows = _score_fold(keep, S, train, season, week, floor=floor, beta=beta,
                       target=target, alpha=alpha, lam=lam, roster_depth=roster_depth)
    if rows is None:
        print(f"SKIP: not enough players to fill a roster for {season} wk{week}")
        return
    for row in rows:
        row['method'] = method   # stamp so ADVI/NUTS rows stay distinguishable if ever combined

    new = pl.DataFrame(rows)
    if os.path.exists(out_csv):
        existing = pl.read_csv(out_csv)
        pl.concat([existing, new], how='diagonal_relaxed').write_csv(out_csv)
    else:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        new.write_csv(out_csv)
    print(f"wrote {len(rows)} rows for {season} wk{week} -> {out_csv}")


def _score_fold(keep, S, train, season, week, floor=None, beta=0.90, target=None,
                alpha=0.10, lam=1.0, roster_depth=None):
    """The model-agnostic half of one fold: given real out-of-sample draws
    `S` + realized frame `keep` for one week, build simulated rosters, run
    every decision rule, and score each against realized outcomes. Split
    out from run_one_fold so it can be exercised with a fake `S`/`keep` in
    tests without touching pymc at all.
    """
    roster_depth = roster_depth or DEFAULT_ROSTER_DEPTH
    my_pool_full = simulate_roster(train, roster_depth, tier=0)
    opp_pool_full = simulate_roster(train, roster_depth, tier=1)

    available = set(keep['player'].to_list())
    my_pool = [p for p in my_pool_full if p in available]
    opp_pool = [p for p in opp_pool_full if p in available]

    try:
        exact = de.maximize_expected(S, keep, pool=my_pool)
        default_floor = floor if floor is not None else 0.85 * exact.stats['expected']
        default_target = target if target is not None else exact.stats['expected']

        cc = de.chance_constrained(S, keep, floor=default_floor, beta=beta, pool=my_pool)
        cv = de.cvar_averse(S, keep, target=default_target, alpha=alpha, lam=lam, pool=my_pool)
        opp_choice = de.maximize_expected(S, keep, pool=opp_pool)
        h2h = de.head_to_head_optimize(S, keep, opp_choice.totals, pool=my_pool)
    except ValueError:
        return None

    oracle_players, oracle_val = oracle_lineup(keep, my_pool)
    opp_realized = realized_total(keep, opp_choice.lineup['player'].to_list())

    rows = []
    for rule_name, choice in [('maximize_expected', exact), ('chance_constrained', cc),
                              ('cvar_averse', cv), ('head_to_head', h2h)]:
        players = choice.lineup['player'].to_list()
        realized = realized_total(keep, players)
        row = dict(season=season, week=week, rule=rule_name,
                   predicted_expected=choice.stats['expected'],
                   realized_total=realized, oracle_total=oracle_val,
                   regret=oracle_val - realized)
        if rule_name == 'chance_constrained':
            row.update(floor=default_floor, beta=beta,
                      predicted_p_clear=choice.stats['p_clear'],
                      feasible=choice.stats['feasible'],
                      cleared=int(realized >= default_floor))
        elif rule_name == 'cvar_averse':
            row.update(target=default_target, alpha=alpha, lam=lam,
                      predicted_cvar=choice.stats['cvar'],
                      realized_shortfall=max(default_target - realized, 0.0))
        elif rule_name == 'head_to_head':
            row.update(predicted_p_win=choice.stats['p_win'],
                      opp_realized_total=opp_realized,
                      won=int(realized > opp_realized))
        rows.append(row)
    return rows


# --------------------------------------------------------------- summarize

def summarize(out_csv=OUT_CSV):
    """Reliability tables + economic comparison across every fold written
    to `out_csv` so far. Safe to call after any number of folds -- more
    folds just make the calibration buckets less noisy.
    """
    df = pl.read_csv(out_csv)
    n_folds = df.select(['season', 'week']).unique().height
    print(f"=== {n_folds} folds, {df.height} rule-rows in {out_csv} ===\n")

    print("--- economic comparison: mean realized points, mean regret vs. oracle ---")
    print(
        df.group_by('rule')
        .agg(pl.col('realized_total').mean().alias('mean_realized'),
             pl.col('regret').mean().alias('mean_regret'),
             pl.len().alias('n'))
        .sort('rule')
    )

    cc = df.filter(pl.col('rule') == 'chance_constrained').drop_nulls('predicted_p_clear')
    if cc.height >= 5:
        print(f"\n--- chance-constrained calibration ---")
        print(f"promised beta (avg): {cc['beta'].mean():.3f}   "
              f"empirical clear-rate: {cc['cleared'].mean():.3f}   "
              f"feasible folds: {cc['feasible'].mean():.1%}")
        print(_reliability_table(cc, 'predicted_p_clear', 'cleared'))

    cv = df.filter(pl.col('rule') == 'cvar_averse').drop_nulls('predicted_cvar')
    if cv.height >= 5:
        alpha = float(cv['alpha'][0])
        print(f"\n--- CVaR calibration (alpha={alpha:.2f}) ---")
        print(f"mean predicted CVaR(shortfall) at decision time: {cv['predicted_cvar'].mean():.2f}")
        var_thresh = cv['realized_shortfall'].quantile(1 - alpha)
        tail = cv.filter(pl.col('realized_shortfall') >= var_thresh)
        print(f"realized CVaR (avg shortfall in worst {alpha:.0%} of folds, n={tail.height}): "
              f"{tail['realized_shortfall'].mean():.2f}")

    h2h = df.filter(pl.col('rule') == 'head_to_head').drop_nulls('predicted_p_win')
    if h2h.height >= 5:
        print(f"\n--- head-to-head calibration ---")
        print(f"promised P(win) (avg): {h2h['predicted_p_win'].mean():.3f}   "
              f"empirical win-rate: {h2h['won'].mean():.3f}")
        print(_reliability_table(h2h, 'predicted_p_win', 'won'))


def _reliability_table(df, pred_col, outcome_col, n_buckets=5):
    """Classic reliability/calibration curve: bucket by predicted
    probability, compare mean-predicted to empirical outcome rate per
    bucket. Well-calibrated => the two columns track each other."""
    n_buckets = min(n_buckets, df.height)
    bucketed = df.with_columns(
        pl.col(pred_col).rank(method='ordinal').alias('_rank')
    ).with_columns(
        (((pl.col('_rank') - 1) * n_buckets) // df.height).alias('bucket')
    )
    return (
        bucketed.group_by('bucket')
        .agg(pl.col(pred_col).mean().alias('mean_predicted'),
             pl.col(outcome_col).mean().alias('empirical_rate'),
             pl.len().alias('n'))
        .sort('bucket')
    )


# ------------------------------------------------------------------ main

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('season', type=int, nargs='?')
    p.add_argument('week', type=float, nargs='?')
    p.add_argument('--floor', type=float, default=None,
                  help='points floor for chance_constrained (default: 0.85x expected)')
    p.add_argument('--beta', type=float, default=0.90)
    p.add_argument('--target', type=float, default=None,
                  help='target for cvar_averse (default: expected total)')
    p.add_argument('--alpha', type=float, default=0.10)
    p.add_argument('--lam', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--method', choices=['advi', 'nuts'], default='advi',
                  help='fitting method for the ar model (default advi, same as every prior '
                       'calibrate_decisions.py run). Use nuts to check whether ADVI\'s '
                       'mean-field approximation is itself contributing to a bias seen in '
                       'the top-K "my roster" subgroup -- see backtest_ar_nuts.csv for the '
                       'equivalent NUTS-vs-ADVI check on the general player population.')
    p.add_argument('--advi-n', type=int, default=6000)
    p.add_argument('--advi-draws', type=int, default=400)
    p.add_argument('--nuts-draws', type=int, default=300)
    p.add_argument('--nuts-tune', type=int, default=300)
    p.add_argument('--chains', type=int, default=2)
    p.add_argument('--out', type=str, default=OUT_CSV)
    p.add_argument('--summarize', action='store_true',
                  help='skip fitting; just print calibration tables from --out')
    args = p.parse_args()

    if args.summarize:
        summarize(args.out)
    else:
        if args.season is None or args.week is None:
            p.error('season and week are required unless --summarize is given')
        run_one_fold(args.season, args.week, out_csv=args.out, floor=args.floor,
                    beta=args.beta, target=args.target, alpha=args.alpha, lam=args.lam,
                    seed=args.seed, method=args.method, advi_n=args.advi_n, advi_draws=args.advi_draws,
                    nuts_draws=args.nuts_draws, nuts_tune=args.nuts_tune, nuts_chains=args.chains)
