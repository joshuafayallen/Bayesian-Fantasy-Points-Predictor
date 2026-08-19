#!/bin/bash
# Parameter sweep for league_validate.py: reruns the same 3-season,
# 9-week-per-season league validation at several settings of one
# rule-specific knob (beta for chance_constrained, lam for cvar_averse),
# each value written to its OWN --out file. This matters because
# league_validate.already_done() only keys on (season, week, rule) -- if
# two different parameter values wrote to the SAME csv, the second value
# would see every fold already logged and silently skip all of it.
#
# Motivation (from the 220-row results/league_validation.csv run):
#   - chance_constrained: feasible=false on every single row. beta=0.90
#     but the best-achievable clear-probability across all matchups only
#     ever reached 0.6675-0.77 (mean 0.73). beta has to come down into
#     that range for the constraint to ever bind for real instead of
#     silently falling back to the highest-p_clear lineup.
#   - cvar_averse: picked the SAME lineup as baseline in ~71% of matchups
#     at lam=1.0 -- the CVaR penalty mostly isn't beating expected-value
#     ranking. Needs a materially larger lam to actually bite, given
#     expected-value gaps between alternative lineups run 10-40+ points.
#
# Usage (run from the project root, on a machine with nflreadpy network
# access -- this is blocked in the sandbox that authored these scripts):
#   ./src/run_param_sweep.sh chance_constrained beta 0.65,0.70,0.75
#   ./src/run_param_sweep.sh cvar_averse lam 2.0,5.0,10.0
#
# Then compare against the existing default-settings run:
#   python src/sweep_summarize.py results/league_validation_chance_constrained_beta_*.csv --by beta
#   python src/sweep_summarize.py results/league_validation_cvar_averse_lam_*.csv --by lam
set -e

rule=$1
param=$2
values=$3

if [[ -z "$rule" || -z "$param" || -z "$values" ]]; then
    echo "usage: $0 <rule> <beta|lam> <comma,separated,values>"
    exit 1
fi

IFS=',' read -ra VALS <<< "$values"
for v in "${VALS[@]}"; do
    out="results/league_validation_${rule}_${param}_${v}.csv"
    rm -f "$out"
    for s in 2023 2024 2025; do
        for wk in 1 2 3 4 5 6 7 8 9; do
            python src/league_validate.py "$s" "$wk" --rule "$rule" --"$param" "$v" \
                --adp-draft --out "$out"
        done
    done
    echo "done: $rule $param=$v -> $out"
done
