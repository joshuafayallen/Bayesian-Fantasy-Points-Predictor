#!/usr/bin/env bash
# League simulation of the decision rules on the current pipeline (src/ffpred,
# het_gp_vegas + het_gp_qb_vegas, NUTS 4x1000/1000). Same design as the old
# results/league_validation.csv: seasons 2023-2025, weeks 2-9, 10 teams.
#   1. synthetic trailing-mean draft, 2023-2025  -> results/league/synthetic
#   2. Yahoo expert mock draft, 2025              -> results/league/yahoo2025
#      (reuses step 1's cached 2025 forecasts, so no extra model fits)
# Resumable: finished weeks are skipped, so just rerun it if it stops.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results/league/synthetic results/league/yahoo2025 logs

uv run python src/ff.py league --seasons 2023-2025 --weeks 2-9 \
    --out-dir results/league/synthetic 2>&1 | grep -v -e NumbaPerformanceWarning -e 'return inner' \
    | tee -a logs/league_synthetic.log

[ -e results/league/yahoo2025/forecasts ] || ln -s ../synthetic/forecasts results/league/yahoo2025/forecasts
uv run python src/ff.py league --seasons 2025 --weeks 2-9 \
    --mock-draft mock-rosters/yahoo_expert_2025.csv \
    --out-dir results/league/yahoo2025 2>&1 | grep -v -e NumbaPerformanceWarning -e 'return inner' \
    | tee -a logs/league_yahoo2025.log

echo; echo "=== synthetic draft, 2023-2025 ==="; uv run python src/ff.py summarize-league results/league/synthetic
echo; echo "=== Yahoo mock, 2025 ===";           uv run python src/ff.py summarize-league results/league/yahoo2025
