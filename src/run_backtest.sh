#!/bin/bash
# Walk-forward backtest sweep comparing model architectures from
# src/ff_ar.py (see src/backtest_harness.py's module docstring for
# what each one is: hs, ar, team, plus `stack`, a blend of team+hs via
# az.weight_predictions -- team_ar and team_season are retired/not in
# src/ff_ar.py's current model set, see the harness's own "retired
# models" note), via src/backtest_harness.py. Every (model, season,
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
#   MODELS="r2d2:plain r2d2:shrunk team:shrunk" src/run_backtest.sh # mixed shrunk settings, one sweep
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
# src/ff_ar.py run. Passed straight through to backtest_harness.py's
# --stack-weights.
#
# SHRUNK=true fits every fold in this sweep with the shrinkage_features.csv
# -derived shrunk lag columns instead of the plain lags (models.py's
# apply_shrunk_features/SHRUNK_SWAP_COLS -- same swap src/fit_model.py's
# --shrunk uses). Each fold's `shrunk` flag is also written to $OUT and is
# part of backtest_harness.py's already_done() dedup key, so a plain sweep
# and a shrunk sweep into the SAME $OUT file coexist as separate rows
# rather than colliding or silently skipping each other.
#
#   SHRUNK=true src/run_backtest.sh                  # shrunk-features sweep
#   SHRUNK=true MODELS="team r2d2" src/run_backtest.sh   # narrow to whichever
#                                                          combo src/compare_fits.py flagged
#
# Individual MODELS entries can also override SHRUNK per-model by
# appending `:plain` or `:shrunk` -- e.g. after src/compare_fits.py flags
# a specific (model, shrunk) combo as promising, or when you want several
# different combos confirmed in ONE sweep instead of running SHRUNK=true
# and SHRUNK=false as two separate invocations:
#
#   MODELS="r2d2:shrunk r2d2:plain team:shrunk" src/run_backtest.sh
#
# An entry with no `:plain`/`:shrunk` suffix (e.g. plain "ar") falls back
# to the global SHRUNK setting, so existing MODELS values without a colon
# keep working exactly as before.

set -euo pipefail
cd "$(dirname "$0")/.."   # repo root, so processed-data/ resolves

METHOD="${METHOD:-nuts}"
SEASONS="${SEASONS:-2013 2019 2023 2024}"
WEEKS="${WEEKS:-6 7 8 9 10 11 12 13 14 15}"
MODELS="${MODELS:-hs ar team stack baseline}"
OUT="${OUT:-results/backtest_walkforward_09_10.csv}"
STACK_WEIGHTS="${STACK_WEIGHTS:-0.58,0.29,0.13}"
SHRUNK="${SHRUNK:-false}"

NUTS_ARGS=(--chains 4 --nuts-draws 500 --nuts-tune 500)
ADVI_ARGS=(--advi-n 6000 --advi-draws 400)

# Parse MODELS once into two parallel arrays (model name, shrunk true/
# false) instead of a single token list -- lets each entry independently
# override the global SHRUNK setting via a `:plain`/`:shrunk` suffix (see
# the usage note above). Parallel arrays rather than an associative array
# so this still runs under bash 3.2 (macOS's default /bin/bash, which
# predates associative arrays).
MODEL_NAMES=()
MODEL_SHRUNKS=()
for tok in $MODELS; do
    case "$tok" in
        *:plain)  MODEL_NAMES+=("${tok%:plain}");   MODEL_SHRUNKS+=(false) ;;
        *:shrunk) MODEL_NAMES+=("${tok%:shrunk}");  MODEL_SHRUNKS+=(true)  ;;
        *:*)
            echo "error: unrecognized MODELS entry '$tok' -- suffix must be ':plain' or ':shrunk'" >&2
            exit 1
            ;;
        *)        MODEL_NAMES+=("$tok");            MODEL_SHRUNKS+=("$SHRUNK") ;;
    esac
done

for s in $SEASONS; do
    for wk in $WEEKS; do
        for i in "${!MODEL_NAMES[@]}"; do
            m="${MODEL_NAMES[$i]}"
            m_shrunk="${MODEL_SHRUNKS[$i]}"

            SHRUNK_ARGS=()
            if [ "$m_shrunk" = "true" ]; then
                SHRUNK_ARGS+=(--shrunk)
            fi
            EXTRA_ARGS=("${SHRUNK_ARGS[@]+"${SHRUNK_ARGS[@]}"}")
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
