#!/bin/bash
# Full-history sweep: fits every (model x shrunk) combination via
# src/fit_model.py, then prints a ranked az.compare() table across all the
# resulting model-nc/*.nc files via src/compare_fits.py.
#
# This is the "which architecture / feature set looks best on everything
# we have" pass -- az.compare()'s LOO-CV ranking here is what should decide
# which model(s) are worth pointing src/run_backtest.sh at for a real,
# genuinely-held-out comparison (LOO on the full-history fit still trains
# and evaluates on overlapping data; the backtest harness's walk-forward
# folds are the actual out-of-sample check). Treat this sweep as a cheap
# first filter, not a substitute for that.
#
# Run from anywhere -- this cd's to the repo root itself.
#
# Usage:
#   src/run_fit_all.sh                                  # all 4 models x {plain, shrunk} = 8 fits
#   MODELS="team r2d2" src/run_fit_all.sh                # just team/r2d2
#   SHRUNK="false" src/run_fit_all.sh                    # plain lags only, skip the shrunk pass
#   DRAWS=300 TUNE=300 CHAINS=2 src/run_fit_all.sh        # faster/cheaper sampling for a first look
#   FORCE=1 src/run_fit_all.sh                            # refit even if the .nc file already exists
#
# Each (model, shrunk) fit is checkpointed as its own model-nc/ff_<model>_
# <plain|shrunk>.nc -- safe to Ctrl-C and rerun, already-done combos are
# skipped unless FORCE=1 (see already_done() below; mirrors
# src/run_backtest.sh's resume behavior).

set -euo pipefail
cd "$(dirname "$0")/.."   # repo root, so processed-data/ resolves

MODELS="${MODELS:-team r2d2 bart}"
SHRUNK="${SHRUNK:-false true}"
DRAWS="${DRAWS:-1000}"
TUNE="${TUNE:-1000}"
CHAINS="${CHAINS:-4}"
TARGET_ACCEPT="${TARGET_ACCEPT:-0.99}"
SEED="${SEED:-41029041}"
OUT_DIR="${OUT_DIR:-model-nc}"
FORCE="${FORCE:-0}"

mkdir -p "$OUT_DIR"

for m in $MODELS; do
    for s in $SHRUNK; do
        if [ "$s" = "true" ]; then
            suffix="shrunk"
            SHRUNK_FLAG="--shrunk"
        else
            suffix="plain"
            SHRUNK_FLAG=""
        fi
        out="$OUT_DIR/ff_${m}_${suffix}.nc"

        if [ "$FORCE" != "1" ] && [ -f "$out" ]; then
            echo "SKIP: $out already exists (set FORCE=1 to refit)"
            continue
        fi

        echo
        echo "== fitting model=$m shrunk=$s -> $out =="
        python src/fit_model.py "$m" $SHRUNK_FLAG \
            --draws "$DRAWS" --tune "$TUNE" --chains "$CHAINS" \
            --target-accept "$TARGET_ACCEPT" --seed "$SEED" --out "$out"
    done
done

echo

