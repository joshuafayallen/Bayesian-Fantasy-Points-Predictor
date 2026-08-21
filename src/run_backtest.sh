#!/bin/bash
# Walk-forward backtest sweep comparing model architectures from
# src/ff-ar.py (see sandbox/backtest_harness.py's module docstring for
# what each one is: hs, ar, team, plus `stack`, a blend of team+hs via
# az.weight_predictions -- team_ar and team_season are retired/not in
# src/ff-ar.py's current model set, see the harness's own "retired
# models" note), via sandbox/backtest_harness.py. Every (model, season,
# week) fold is
# checkpointed to $OUT as it completes -- safe to Ctrl-C and rerun,
# already-done folds are skipped (see already_done() in the harness).
# Ends by printing the per-model comparison + winner via
# summarize_backtest.py.
#
# Run from anywhere -- this cd's to the repo root itself.
#
# Usage:
#   src/run_backtest.sh                              # full NUTS sweep (slow, high-fidelity)
#   METHOD=advi src/run_backtest.sh                   # fast ADVI sweep (good for a quick local check)
#   SEASONS="2023 2024" WEEKS="6 12" src/run_backtest.sh   # smaller custom sweep
#   MODELS="ar team" src/run_backtest.sh              # skip hs, e.g. once it's ruled out
#   MODELS="team hs stack" src/run_backtest.sh        # check whether stacking beats either alone
#   MODELS=stack STACK_WEIGHTS=0.34,0.33,0.33 src/run_backtest.sh   # try even weights instead of LOO's
#
# NUTS is the high-fidelity path but is slow: chains=4, draws=500, tune=500
# per fold is a genuine full fit, budget real time for the default sweep
# size (seasons x weeks x models -- `stack` counts as its own model but
# internally fits BOTH team and hs per fold, so it roughly doubles the
# cost of whichever folds include it -- if you're already sweeping team
# and hs separately, consider running `stack` as its own narrower sweep
# rather than lumping it into MODELS with everything else). ADVI is the
# fast/cheap path for a first pass -- ~1-2min/fold (stack ~2x that, since
# it fits two models).
#
# STACK_WEIGHTS is "team,hs,ar" and is only used when MODELS includes
# `stack` -- default (0.58, 0.29, 0.13) is az.compare()'s LOO stacking
# weight for team_mod/hs_version/ar_mod from the full-history
# src/ff-ar.py run. Passed straight through to backtest_harness.py's
# --stack-weights.

set -euo pipefail
cd "$(dirname "$0")/.."   # repo root, so processed-data/ resolves

METHOD="${METHOD:-nuts}"
SEASONS="${SEASONS:-2013 2019 2023 2024}"
WEEKS="${WEEKS:-6 7 8 9 10 11 12 13 14 15}"
MODELS="${MODELS:-hs ar team stack baseline}"
OUT="${OUT:-results/backtest_walkforward.csv}"
STACK_WEIGHTS="${STACK_WEIGHTS:-0.58,0.29,0.13}"

NUTS_ARGS=(--chains 4 --nuts-draws 500 --nuts-tune 500)
ADVI_ARGS=(--advi-n 6000 --advi-draws 400)

for s in $SEASONS; do
    for wk in $WEEKS; do
        for m in $MODELS; do
            EXTRA_ARGS=()
            if [ "$m" = "stack" ]; then
                EXTRA_ARGS+=(--stack-weights "$STACK_WEIGHTS")
            fi
            # ${arr[@]+"${arr[@]}"} instead of plain "${arr[@]}" -- under
            # `set -u`, an empty (but declared) bash array expanded the
            # normal way trips "unbound variable" on older bash (notably
            # macOS's default /bin/bash, still 3.2). This form checks
            # "is EXTRA_ARGS set at all" via `+` before expanding it,
            # which is safe even when it's empty.
            if [ "$METHOD" = "nuts" ]; then
                python src/backtest_harness.py "$m" nuts "$s" "$wk" "${NUTS_ARGS[@]}" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} --out "$OUT"
            else
                python src/backtest_harness.py "$m" advi "$s" "$wk" "${ADVI_ARGS[@]}" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} --out "$OUT"
            fi
        done
    done
done

echo
echo "== summary =="
python src/summarize_backtest.py "$OUT"
