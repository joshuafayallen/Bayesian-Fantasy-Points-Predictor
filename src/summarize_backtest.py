"""
Summarize a backtest_harness.py results CSV: per-season weighted metrics,
the overall pooled metric, a fold-by-fold gain-over-baseline check, and --
when the CSV has more than one `model` -- a head-to-head comparison that
declares a winner (hs_version vs ar_mod, held-out CRPS as the primary
criterion, RMSE as a tiebreak).

Usage:
    python sandbox/summarize_backtest.py results/backtest_walkforward.csv
"""
import sys
import numpy as np
import polars as pl


def wavg(df, col, weight_col="n"):
    w = df[weight_col].to_numpy()
    x = df[col].to_numpy()
    return float(np.average(x, weights=w))


def compare_models(df, primary_metric="crps"):
    """Pool folds by model (across methods/seasons, weighted by n) and
    declare a winner. CRPS is the primary criterion because it's a proper
    scoring rule over the full predictive distribution, not just the mean
    -- RMSE alone can reward an overconfident model that happens to nail
    the point estimate. Only paired folds (same season/week run for both
    models) are used for the head-to-head, so both models are judged on
    identical held-out weeks."""
    models = sorted(df["model"].unique().to_list())
    print("\n" + "=" * 78)
    print(f"MODEL COMPARISON  (primary metric: {primary_metric})")
    print("=" * 78)

    if len(models) < 2:
        print(f"  only one model present ({models}) -- nothing to compare yet.")
        return None

    keys = [c for c in ("season", "week") if c in df.columns]

    # Fold coverage check -- run BEFORE restricting to the common set below,
    # so a model that's only partially run (or run on a disjoint grid, e.g.
    # a quick single-fold smoke test vs. the real sweep's season/week grid)
    # is visible here instead of just silently vanishing from the table
    # further down. This is what would have caught `stack` looking "missing"
    # from the comparison the CSV it was actually run into never had it.
    per_model_folds = {
        m: set(map(tuple, df.filter(pl.col("model") == m).select(keys).unique().rows()))
        for m in models
    }
    n_expected = len(set().union(*per_model_folds.values()))
    incomplete = {m: fk for m, fk in per_model_folds.items() if len(fk) != n_expected}
    if incomplete:
        print(f"  FOLD COVERAGE WARNING -- not every model has run on the same folds "
              f"({n_expected} distinct (season, week) folds across all models):")
        for m in models:
            n = len(per_model_folds[m])
            flag = "" if n == n_expected else "  <-- incomplete, will only be compared on its overlap below"
            print(f"    {m:12s} {n:4d} / {n_expected} folds{flag}")
        print()

    # restrict to folds run for every model, on the same (season, week), so
    # the comparison isn't skewed by one model having an easier subset of folds
    paired = df
    for m in models:
        have = df.filter(pl.col("model") == m).select(keys).unique()
        paired = paired.join(have, on=keys, how="inner")
    counts = paired.group_by("model").len()
    n_common = paired.select(keys).unique().height
    print(f"  common folds across all models: {n_common}")

    summary = []
    for m in models:
        sub = paired.filter(pl.col("model") == m)
        if sub.height == 0:
            print(f"  {m}: no folds overlap with the other model(s) -- skipping")
            continue
        row = {"model": m, "n_folds": sub.height, "n_obs": int(sub["n"].sum())}
        for metric in ("crps", "rmse", "mae", "spearman", "cov80"):
            if metric in sub.columns:
                row[metric] = wavg(sub, metric)
        summary.append(row)

    if len(summary) < 2:
        print("  not enough overlapping folds yet to compare -- run more folds for both models.")
        return None

    header = f"{'model':12s}" + "".join(f"{k:>10s}" for k in summary[0] if k not in ("model",))
    print("\n" + header)
    for row in summary:
        line = f"{row['model']:12s}"
        for k, v in row.items():
            if k == "model":
                continue
            line += f"{v:>10.3f}" if isinstance(v, float) else f"{v:>10d}"
        print(line)

    winner = min(summary, key=lambda r: r[primary_metric])
    loser = [r for r in summary if r is not winner][0]
    margin = loser[primary_metric] - winner[primary_metric]
    print(f"\n  WINNER: {winner['model']}  "
          f"({primary_metric}={winner[primary_metric]:.4f} vs "
          f"{loser['model']}'s {loser[primary_metric]:.4f}, "
          f"margin={margin:.4f})")
    if "rmse" in winner and "rmse" in loser:
        rmse_agrees = winner["rmse"] <= loser["rmse"]
        print(f"  rmse agrees with crps ranking: {rmse_agrees}")
    return winner["model"]


def summarize(path):
    df = pl.read_csv(path)
    df = df.with_columns(
        (100 * (1 - pl.col("rmse") / pl.col("rmse_base"))).alias("gain_pct")
    )

    metrics = ["rmse", "mae", "crps", "spearman", "cov50", "cov80", "cov90", "gain_pct"]

    print("=" * 78)
    print(f"PER-SEASON  (weighted by n)   [{path}]")
    print("=" * 78)
    header = f"{'season':>8s}{'n':>7s}" + "".join(f"{m:>10s}" for m in metrics)
    print(header)
    for season in sorted(df["season"].unique().to_list()):
        sub = df.filter(pl.col("season") == season)
        n_tot = int(sub["n"].sum())
        row = f"{int(season):>8d}{n_tot:>7d}"
        for m in metrics:
            row += f"{wavg(sub, m):>10.3f}"
        print(row)

    print("\n" + "=" * 78)
    print("OVERALL (all seasons/folds pooled, weighted by n)")
    print("=" * 78)
    n_tot = int(df["n"].sum())
    print(f"n = {n_tot}  across {df.height} folds, {df['season'].n_unique()} seasons")
    for m in metrics:
        print(f"  {m:10s} {wavg(df, m):.3f}")

    print("\n" + "=" * 78)
    print("FOLD-BY-FOLD GAIN OVER BASELINE (unweighted -- one row per fold)")
    print("=" * 78)
    gains = df["gain_pct"].to_numpy()
    n_beat = int((gains > 0).sum())
    print(f"  folds beating baseline : {n_beat} / {len(gains)}  ({100*n_beat/len(gains):.0f}%)")
    print(f"  mean gain              : {gains.mean():+.2f}%   (se: {gains.std(ddof=1)/np.sqrt(len(gains)):.2f}%)")
    print(f"  min / max gain         : {gains.min():+.2f}% / {gains.max():+.2f}%")

    worst = df.sort("gain_pct").head(5)
    print("\n  worst 5 folds by gain:")
    for row in worst.iter_rows(named=True):
        print(f"    {row['model']:5s} {row['method']:5s} {int(row['season'])} "
              f"wk{row['week']:<4.0f} gain={row['gain_pct']:+6.2f}%  "
              f"rmse={row['rmse']:.2f}  base={row['rmse_base']:.2f}  n={row['n']}")

    compare_models(df)
    return df


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python summarize_backtest.py <results.csv>")
        sys.exit(1)
    summarize(sys.argv[1])
