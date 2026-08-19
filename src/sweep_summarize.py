"""
Summarize league_validate.py sweep output broken down by the swept
parameter (beta for chance_constrained, lam for cvar_averse) -- unlike
league_validate.summarize_league, which only groups by `rule` and so
hides exactly the comparison a sweep needs: does lowering beta actually
fix feasibility? does raising lam actually move win rate versus the
default? Each row already carries the parameter value it was run with
(see league_validate._score_matchups), so this just groups on it.

Usage:
    python src/sweep_summarize.py results/league_validation_chance_constrained_beta_*.csv --by beta
    python src/sweep_summarize.py results/league_validation_cvar_averse_lam_*.csv --by lam
"""

import argparse

import polars as pl
from scipy.stats import binomtest


def summarize_by(paths, by):
    df = pl.concat([pl.read_csv(p) for p in paths], how='diagonal_relaxed')
    for rule in df['rule'].unique().sort().to_list():
        d0 = df.filter(pl.col('rule') == rule)
        if by not in d0.columns or d0[by].null_count() == d0.height:
            continue
        for val in sorted(d0[by].drop_nulls().unique().to_list()):
            d = d0.filter(pl.col(by) == val)
            n = d.height
            if n == 0:
                continue
            wr_base = d['baseline_a_won'].mean()
            wr_smart = d['smart_a_won'].mean()
            rescued = d.filter((pl.col('smart_a_won') == 1) & (pl.col('baseline_a_won') == 0)).height
            cost = d.filter((pl.col('smart_a_won') == 0) & (pl.col('baseline_a_won') == 1)).height
            same = d['same_lineup_as_baseline'].mean() if 'same_lineup_as_baseline' in d.columns else None

            print(f"=== rule={rule}  {by}={val}  (n={n}) ===")
            print(f"  baseline win rate: {wr_base:.3f}   {rule} win rate: {wr_smart:.3f}  (lift {wr_smart - wr_base:+.3f})")
            print(f"  rescued: {rescued}   cost: {cost}")
            if same is not None:
                print(f"  same lineup as baseline: {same:.1%}")
            if 'feasible' in d.columns and d['feasible'].null_count() < n:
                print(f"  feasible: {d['feasible'].mean():.1%}")
            if rescued + cost > 0:
                p = binomtest(min(rescued, cost), rescued + cost, 0.5).pvalue
                print(f"  McNemar exact p={p:.3f}  (discordant pairs={rescued + cost})")
            else:
                print("  no discordant pairs -- no power yet")
            print(f"  mean regret  baseline: {d['regret_baseline_a'].mean():.2f}   {rule}: {d['regret_smart_a'].mean():.2f}")
            print()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out_csvs', nargs='+', help='one or more league_validate.py --out CSVs (glob-expanded by your shell)')
    ap.add_argument('--by', required=True, choices=['beta', 'lam'])
    args = ap.parse_args()
    summarize_by(args.out_csvs, args.by)
