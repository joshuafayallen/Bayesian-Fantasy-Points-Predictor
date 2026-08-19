#!/bin/bash
# Ten-team league validation sweep: for each rule, drafts a 10-team league
# (once per season, from prior-season data only) and scores every scheduled
# matchup across a season's worth of weeks, checkpointed to
# results/league_validation.csv. Safe to interrupt/resume.
for rule in chance_constrained cvar_averse head_to_head; do
    for s in 2023 2024 2025; do
        for wk in 1 2 3 4 5 6 7 8 9; do
            python src/league_validate.py "$s" "$wk" --rule "$rule"
        done
    done
done

python src/league_validate.py --summarize
