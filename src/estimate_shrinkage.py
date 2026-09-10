"""
Auxiliary, side-channel script -- computes empirical-Bayes shrinkage for
the three attempt-based rate features (yards_per_rush_attempt,
yards_per_target, yards_per_pass_attempt) and writes both the raw and
shrunk LAGGED versions to a CSV, WITHOUT touching data_cleaning.py or
models.py.

Why: at weekly grain a player's per-game rate is dominated by
small-sample noise -- 1 rush for 15 yards reads as "15.0 yards/attempt,"
swamping the prior-predictive range (see lag_yards_per_rush_attempt_z's
17-21 max discussed in the prior-checking sessions). The fix is
James-Stein / empirical-Bayes shrinkage toward the league mean, weighted
by attempts:

    shrunk = w * raw + (1 - w) * league_avg,   w = n / (n + k)

Rather than hand-picking k, this script derives it from the data via a
one-way random-effects (variance components) decomposition -- the same
idea behind the classic Efron-Morris/James-Stein baseball-average
estimator, and behind the DerSimonian-Laird tau^2 estimator used in
random-effects meta-analysis:

    r_ij = mu + b_i + e_ij
    b_i  ~ N(0, tau^2)        -- true between-player variance
    e_ij ~ N(0, sigma^2/n_ij)  -- sampling noise, shrinks with attempts

sigma^2 (the per-attempt noise variance) is estimated directly and
robustly from PURELY WITHIN-PLAYER variation: for each player with 2+
games, take the attempt-weighted sum of squared deviations from that
player's own weighted-mean rate, then pool across all such players
(a standard one-way-ANOVA "mean square within" estimator, weighted by
attempts since each row is itself an average of n_ij per-attempt
outcomes). This isolates noise without ever needing to separate skill
from noise -- it only uses each player's own repeated observations.

tau^2 (the true between-player variance) then falls out of the
variance identity

    Var(r_ij) = tau^2 + sigma^2 * E[1/n_ij]

using the already-estimated sigma^2 and the population moments of the
observed rates -- one equation, one unknown, no iteration.

k = sigma^2 / tau^2, separately per feature. No hand-tuned pseudocounts.

An earlier version of this estimator tried to get sigma^2/tau^2 from a
single regression of attempt-count-BINNED rate variance against 1/n.
That's an errors-in-variables problem (each bin's variance is itself a
noisy statistic), and it systematically biased tau^2 toward zero for
rush/pass, producing nonsensical k values in the tens of millions.
Isolating sigma^2 from within-player variation alone (this version)
avoids that bias. tau^2 can still come out at/near zero for a feature
that's genuinely mostly noise even after this fix -- see the printed
diagnostics; that is treated as a real finding (heavy shrinkage is then
the data-driven answer), not silently patched over, but k is capped
(K_CAP below) so a near-zero tau^2 doesn't produce a k so large that
downstream code overflows or the shrunk column is float-precision
garbage.

Output: processed-data/shrinkage_features.csv, one row per
(player_id, season, week), with both raw and shrunk lagged versions of
the three features (shift(1).over(player_id), matching
data_cleaning.py's bring_back_in step) plus this-week's own attempt
counts. Meant to be joined into prior_checks.py's prep_data() in place
of ff-processed.parquet's lag_* columns, so the shrunk features can run
through the exact same scaling / prior-predictive pipeline as a quick
A/B -- before deciding whether to ever fold this into data_cleaning.py
for real.

Note: yards_per_pass_attempt here is computed against the player's OWN
pass_attempt, not the team-total pass_attempt_total that
data_cleaning.py currently uses -- so its shrunk column isn't a
drop-in replacement for the pipeline's current (still-team-denominator)
version. Flagging this rather than silently matching the known-stale
denominator; the shrinkage math is identical either way if that
denominator gets fixed later.

Usage:
    python src/estimate_shrinkage.py
"""
import numpy as np
import polars as pl
import nflreadpy as nfl

OUT_PATH = 'processed-data/shrinkage_features.csv'
K_CAP = 200.0   # ceiling on k -- guards against a near-zero tau^2 blowing up
                # the shrinkage weight formula; still means "shrink hard"


def estimate_k(df: pl.DataFrame, rate_col: str, n_col: str, player_col: str = 'player_id'):
    """Method-of-moments estimate of k = sigma^2/tau^2 for one feature,
    via a one-way random-effects (player) decomposition. `df` must
    already be filtered to n_col > 0 and rate_col non-null. See module
    docstring for the derivation."""
    d = df.select(player_col, n_col, rate_col)

    # each player's own attempt-weighted mean rate (the MLE of b_i + mu
    # under the model, given that player's own games)
    player_means = (
        d.group_by(player_col)
        .agg(
            (pl.col(n_col) * pl.col(rate_col)).sum().alias('_num'),
            pl.col(n_col).sum().alias('_den'),
            pl.len().alias('_m'),
        )
        .with_columns((pl.col('_num') / pl.col('_den')).alias('_player_mean'))
    )
    d = d.join(player_means.select(player_col, '_player_mean', '_m'), on=player_col)

    # sigma^2: pooled within-player weighted sum of squares / pooled dof.
    # Only players with 2+ games carry within-player information.
    multi = d.filter(pl.col('_m') >= 2).with_columns(
        (pl.col(n_col) * (pl.col(rate_col) - pl.col('_player_mean')) ** 2).alias('_wss')
    )
    dof = multi.group_by(player_col).agg(pl.len().alias('_c')).select((pl.col('_c') - 1).sum()).item()
    sigma2 = float(multi['_wss'].sum() / dof)

    # tau^2: from the population variance identity
    # Var(r) = tau^2 + sigma^2 * E[1/n], using attempt-weighted mu (a
    # more efficient population-mean estimate than an unweighted one)
    # and unweighted moments for the variance/1-over-n expectations,
    # matching the per-row generative model.
    n = d[n_col].to_numpy().astype(float)
    r = d[rate_col].to_numpy().astype(float)
    mu = float((n * r).sum() / n.sum())
    v_total = float(((r - mu) ** 2).mean())
    mean_inv_n = float((1.0 / n).mean())

    tau2_raw = v_total - sigma2 * mean_inv_n
    tau2 = max(tau2_raw, 1e-6)
    k = min(sigma2 / tau2, K_CAP)
    return k, sigma2, tau2_raw, mu


def shrink(df: pl.DataFrame, rate_col: str, n_col: str, k: float,
           league_avg: float, out_col: str) -> pl.DataFrame:
    w = pl.col(n_col) / (pl.col(n_col) + k)
    return df.with_columns(
        pl.when(pl.col(n_col) > 0)
        .then(w * pl.col(rate_col) + (1 - w) * league_avg)
        .otherwise(league_avg)
        .alias(out_col)
    )


def main():
    raw = (
        nfl.load_ff_opportunity(seasons=True, stat_type='weekly')
        .select('player_id', 'position', 'season', 'week',
                'rush_attempt', 'rush_yards_gained',
                'rec_attempt', 'rec_yards_gained',
                'pass_attempt', 'pass_yards_gained')
        .filter((pl.col('player_id').is_not_null()) &
                (pl.col('position').is_in(['TE', 'RB', 'WR', 'QB', 'FB'])))
        .with_columns(pl.col('season').cast(pl.Int32))
        .with_columns(
            pl.when(pl.col('rush_attempt') > 0)
            .then(pl.col('rush_yards_gained') / pl.col('rush_attempt'))
            .otherwise(None).alias('yards_per_rush_attempt'),
            pl.when(pl.col('rec_attempt') > 0)
            .then(pl.col('rec_yards_gained') / pl.col('rec_attempt'))
            .otherwise(None).alias('yards_per_target'),
            pl.when(pl.col('pass_attempt') > 0)
            .then(pl.col('pass_yards_gained') / pl.col('pass_attempt'))
            .otherwise(None).alias('yards_per_pass_attempt'),
        )
    )

    specs = [
        ('yards_per_rush_attempt', 'rush_attempt'),
        ('yards_per_target', 'rec_attempt'),
        ('yards_per_pass_attempt', 'pass_attempt'),
    ]

    print(f"{'feature':<24} {'k':>8} {'sigma2':>9} {'tau2_raw':>10} {'league_avg':>11}  note")
    for rate_col, n_col in specs:
        pos = raw.filter((pl.col(n_col) > 0) & pl.col(rate_col).is_not_null())
        k, sigma2, tau2_raw, league_avg = estimate_k(pos, rate_col, n_col)
        note = ''
        if tau2_raw <= 1e-6:
            note = '<- tau2 ~0: little/no true skill signal beyond attempt-count noise; k capped'
        elif k >= K_CAP:
            note = '<- k capped'
        raw = shrink(raw, rate_col, n_col, k, league_avg, out_col=f'{rate_col}_shrunk')
        print(f"{rate_col:<24} {k:>8.2f} {sigma2:>9.3f} {tau2_raw:>10.4f} {league_avg:>11.3f}  {note}")

    lag_cols = ['yards_per_rush_attempt', 'yards_per_target', 'yards_per_pass_attempt',
                'yards_per_rush_attempt_shrunk', 'yards_per_target_shrunk',
                'yards_per_pass_attempt_shrunk']

    out = (
        raw.sort(['player_id', 'season', 'week'])
        .with_columns(
            pl.col(lag_cols).shift(1).over('player_id').name.prefix('lag_')
        )
        .fill_null(0)
        .select('player_id', 'season', 'week',
                'rush_attempt', 'rec_attempt', 'pass_attempt',
                *[f'lag_{c}' for c in lag_cols])
    )
    out.write_csv(OUT_PATH)
    print(f"\nwrote {out.height} rows to {OUT_PATH}")


if __name__ == '__main__':
    main()
