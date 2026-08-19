#!/bin/bash
# Walk-forward calibration sweep for decision_engine.py's rules.
# Mirrors run_backtest.sh's convention: one process per fold, checkpointed
# to results/decision_calibration.csv, safe to interrupt/resume.
#
# FULL season coverage (weeks 1-18, all three seasons) rather than the
# earlier 6-9 sampled weeks -- the sampled sweep is what surfaced the
# systematic under-prediction bias on the top-K "my roster" subgroup
# (mean +20pts, 16/20 folds over-realizing), and a handful of sampled
# weeks isn't enough to tell a real seasonal pattern (e.g. bias concentrated
# in early-season cold-start weeks, or in specific weeks with more byes)
# apart from noise. Every season/week here already ran under earlier,
# partial versions of this loop, so already_done() will SKIP anything
# already in results/decision_calibration.csv and only fill in the gaps --
# safe to just rerun this in full.
for s in 2023 2024 2025; do
    for wk in $(seq 1 18); do
        python src/calibrate_decisions.py "$s" "$wk"
    done
done

python src/calibrate_decisions.py --summarize

# --- ADVI vs NUTS isolation check ------------------------------------------
# The ADVI sweep above showed a systematic under-prediction bias specific to
# the top-K "my roster" subgroup (mean +20pts, 16/20 folds over-realizing --
# see decision_calibration.csv), while backtest_ar_nuts.csv already showed
# the GENERAL player population is well-calibrated under full NUTS. That
# points to either (a) shrinkage in the player-skill/position priors
# specifically hurting top-tier players, or (b) ADVI's mean-field
# approximation compounding that shrinkage for this high-correlation
# subgroup. Rerun just the worst-bias folds (by realized - predicted, from
# the run above) under full NUTS, into a SEPARATE csv -- already_done()
# only keys on (season, week), not method, so NUTS rows would look
# "already done" and get silently skipped if written into
# decision_calibration.csv itself. NUTS costs minutes/fold, not seconds
# (see calibrate_decisions.py / backtest_harness.py docstrings), which is
# why this reruns only 4 folds, not the full season.
NUTS_OUT=results/decision_calibration_nuts.csv
NUTS_FOLDS=(
    "2024 14"   # +78.4 bias under ADVI, the single worst fold
    "2025 15"   # +75.6
    "2023 3"    # +66.3
    "2025 10"   # +45.7
)
for pair in "${NUTS_FOLDS[@]}"; do
    read -r s wk <<< "$pair"
    python src/calibrate_decisions.py "$s" "$wk" --method nuts --out "$NUTS_OUT"
done

echo "=== NUTS refit of the worst-bias folds -- compare predicted_expected against decision_calibration.csv for the same season/week ==="
python src/calibrate_decisions.py --summarize --out "$NUTS_OUT"
