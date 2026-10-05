# src — production pipeline for `het_gp_vegas` / `het_gp_qb_vegas`

A cleaned-up, tested rewrite of the earlier research code (the old `src/`, now in git history) that does one job: forecast a week's
fantasy points with the two het_gp Vegas models and feed those draws to the
start/sit decision engine. Everything that only served experiments (ar / hs /
team / r2d2 / BART / stack, the ablation arms, the scratchpads, shrinkage,
the team-pregame features) stayed in the research code.

| model | positions | time axis | config |
|---|---|---|---|
| `het_gp_vegas` | RB, WR, TE | tenure | `config.HET_GP_VEGAS` |
| `het_gp_qb_vegas` | QB | age | `config.HET_GP_QB_VEGAS` |

Both use `team_spread` and `vegas_team_total` (the player's side of the line)
in place of the home-perspective `spread_line`, plus last week's team form.
Why: `claude/team-context-ablation.md` in the project.

## Layout

```
src/
  ff.py                 CLI entry point (python src/ff.py --help)
  ffpred/
    config.py           paths, roster shape, the two model configs, sampler settings
    data_build.py       nflverse -> processed-data/ff-processed.parquet (port of data_cleaning.py)
    team_form.py        off/def team-strength state space -> processed-data/team_form.parquet
                        (port of the old src/team_ability_ssm.py, the model that wrote the current file)
    data.py             load + validate processed data, Vegas features, as-of team-form join, windows
    upcoming.py         model rows for a week that hasn't been played (schedule + history)
    model.py            HetPrep / build_het_gp (model unchanged), fit, predict, diagnostics
    forecast.py         both models -> WeekForecast (frame + joint draws), save/load
    decision_engine.py  lineup rules (unchanged logic) + decide() dispatcher
    backtest.py         walk-forward folds, one JSON per fold, summaries
    league.py           league simulation of the decision rules (player_id based)
    scoring.py          CRPS / RMSE / coverage, naive baseline
  tests/                pytest; synthetic data + checks against the real parquet
```

## Weekly use

Run from the project root (`uv run python ...` if you use the uv env).

```bash
# 1. refresh data through last week (about 30 s, needs network)
python src/ff.py build-data
# 2. refresh team form (the as-of join otherwise carries the last fitted week forward)
python src/ff.py build-team-form
# 3. fit both models and forecast the coming week -> forecasts/2026_wk05/
python src/ff.py forecast 2026 5
# 4. choose a lineup
python src/ff.py decide forecasts/2026_wk05 --roster my_roster.txt --rule maximize_expected
python src/ff.py decide forecasts/2026_wk05 --roster my_roster.txt --rule head_to_head --opponent their_roster.csv
python src/ff.py decide forecasts/2026_wk05 --roster my_roster.txt --rule cvar_averse --lam 2
```

`decide` also reports the runner-up lineup and whether the choice is a
**near tie**: for the next few lineups with different players it computes the
objective gap with a bootstrap standard error over the draws, and flags any
gap under 2 SE. It then prints the start/sit calls that are coin flips at this
many draws (`contested_players`, `near_ties`), so "Pickens or Henry, either"
doesn't look like a confident pick.

`--roster` takes a comma list or a file of `player_id`s (gsis ids) or unique
display names; a CSV needs a `player_id` or `player` column. Players not in
the forecast (bye, inactive, rookie) are listed and left out.

`forecast` writes `frame.parquet`, `draws.npy` (n_draws x players, float32),
`summary.csv`, `skipped.parquet` (who couldn't be scored and why) and
`meta.json` (settings, rows, fit time, divergences, max r_hat, min bulk ESS per
model). A fit with divergences or r_hat > 1.05 logs a warning.

For an upcoming week the rows are built by `upcoming.build_upcoming_frame`:
the nflverse schedule supplies opponent, home/away, lines, roof, surface and
weather; each player's lag features come from his last game; the nflverse
weekly roster sets his current team and drops anyone whose status isn't `ACT`
(`--rosters none`, `--include-inactive` to change that). `--source played`
replays a past week on its actual rows instead.

## Validation

```bash
# walk-forward backtest (resumable; one JSON per fold under results/backtest/<run>/)
python src/ff.py backtest --seasons "2019 2023-2025" --weeks "6 9 12 15" \
    --models het_gp_vegas,baseline:het_gp_vegas --run-dir results/backtest/vegas
python src/ff.py summarize-backtest results/backtest/vegas

# tail calibration of a replayed week (forecast made with --source played)
python src/ff.py check-forecast forecasts/nuts_2025_wk09

# league simulation of the decision rules (one forecast per week, cached)
python src/ff.py league --seasons 2025 --weeks 1-9 --mock-draft mock-rosters/yahoo_expert_2025.csv \
    --out-dir results/league/yahoo2025
python src/ff.py summarize-league results/league/yahoo2025

# tests (the real-data tests read processed-data/; slow ones need RUN_SLOW_TESTS=1)
python -m pytest src/tests -q
```

Every backtest fold also records tail calibration by projection tier
(<6, 6-12, 12-18, 18+ expected points) and position: below_p10 and
above_p90 (0.10 each when calibrated), predictive sd vs actual residual sd,
and a PIT histogram. `summarize-backtest` pools them over folds. These are
the parts of the forecast the risk rules (floors, CVaR, P(clear)) read, and
central-interval coverage can look fine while the tails are off in opposite
directions (the case in 2025 week 9: floors too low for low projections, too
generous for high ones).

Defaults: NUTS via nutpie, 4 chains x 1000 draws, 1000 tune (4000 draws per
forecast), 3-season window. The team-context ablation used 2 chains x 500 /
500; pass `--chains 2 --draws 500 --tune 500` to reproduce it. `--method advi`
is for smoke runs only.

## What changed relative to the research code (old `src/`)

Behaviour changes (each one is deliberate; the old behaviour is reachable
where it matters for reproducing old numbers):

1. **Backtest eligibility.** The old code scored a test row only if every one of
   `player_id, position, team, tenure, opp_team, season, week, player_season`
   appeared in training. het_gp uses none of the last six, and the rule
   silently drops every week-1 row and every player's first game of a season
   (e.g. back from injury). this package scores every row the fitted model can
   score (player seen in the window, tenure/age inside the fitted grid).
   `--eligibility legacy` reproduces the old rows.
2. **Upcoming-week forecasts.** The old code only ever predicted a week's *played*
   rows, so who-played leaked in from the future and there was no way to
   forecast a real week. See `upcoming.py`. Replaying every 2023-2025 week
   through it (history + the week's games only): for weeks 2-18 it builds a
   row for 98.0% of players who played, and 97.9% of those have every
   feature identical to the processed row (the rest: mid-season trades,
   previous game dropped by the snap-count join). Week 1 is weaker (84% built,
   80% identical) because of offseason team changes and rookies; the
   nflverse roster feed fixes the team changes in real use.
3. **player_id everywhere.** League rosters, pools and realized points were
   matched on display name. The Yahoo mock CSV carries `player_id`, but it
   was scraped by joining on name, so 12 of its 145 rows are same-name
   duplicates (three Josh Allens, three Kyle Williamses, two Aaron Joneses);
   this package keeps one id per pick (position match, most recent season).
   Name matching is the fallback for CSVs without ids and refuses ambiguous
   names.
4. **Deterministic fits.** Training rows are sorted before fitting. In the old code
   row order came from a polars join, and with the same data and seed a
   different row order gives a different MCMC run: about 1.5% of fold CRPS on
   a QB fold (3.541 vs 3.595 on 2024 wk 9). That's larger than the per-arm
   effects in the ablations, so keep it in mind when reading them.
5. **Primary position** is computed from data up to the target week instead
   of all seasons (it looked ahead, though only at position labels).
6. **Data build fixes** (`build-data`; `--legacy` reproduces the old parquet
   byte-for-byte, checked):
   - roster nulls were forward-filled across *different* players (sorted by
     name) before null ids were dropped -> 7 rookie rows had tenure 8-9
     instead of 0 (Paul Lasike, Don Jackson, 2016);
   - the snap-count id crosswalk attached other players' snaps (`HarrAl00` is
     both Alonzo Harris and Aaron Ripkowski) and duplicated 6 player-weeks;
   - three ambiguous same-game team picks are now deterministic;
   - lag features are shifted after an explicit sort (they relied on join
     order, which the fixes above changed).
   Net: 62,260 -> 62,251 rows; 9 rows dropped, 7 tenure fixes, a few dozen
   share / points-allowed values change.
7. **Team form** is ported from old `src/team_ability_ssm.py` (`TeamOffDef`),
   which is what produced the `team_form.parquet` the models were validated
   on (it has `team_off_form` / `team_def_form` and week-0 rows). The
   docstrings in old `src/backtest_harness.py` and old `src/league_validate.py` still
   point at `estimate_latent_ability.py`, which is stale.
8. **Decision engine vs the FICO energy-planning script it's adapted from**
   (mapping in `decision_engine.py`'s docstring). Two fixes so it does what
   that script does: `pareto_frontier` now uses one scenario split for every
   frontier point (the old code advanced a shared random generator, so each point
   was searched and scored on a different split); and when the search runs
   on a scenario subsample, reported stats now use all draws, like the
   script's Monte Carlo `shortage_rate` (the old code reported on the subsample).
   `chance_constrained` is intentionally not the script's percentile
   approximation; it constrains the joint lineup total.
9. Fold results are one JSON file each (resumable, safe to run folds in
   parallel); the old shared-CSV append wasn't.

Parity checks done while building this:
- `HetPrep` arrays and the compiled model log-density match old `src/models.py`
  exactly for both configs (5 random parameter points, difference 0.0).
- With the same rows in the same order, this package reproduces the old
  harness's fold to the last digit (QB 2024 wk 9: CRPS 3.5406160025).
- `build-data --legacy` output equals the current `ff-processed.parquet`.
- `build-team-form` reproduces the current `team_form.parquet`: same schema
  and 7,648 rows, correlation 1.0000 for team_form / off / def (mean absolute
  difference 0.03 on a scale with sd 3.6, from fewer draws).
- End to end on live data: `build-data` through 2026 week 3, then
  `forecast 2026 4` from the nflverse schedule and weekly rosters (445 active
  players in 16 games, no divergences, max r_hat 1.015), then `decide`.

## Known limitations

- Rookies and anyone not in the 3-season window can't be forecast (no
  career level). They're listed in `skipped.parquet`.
- QB and skill draws come from separate fits, so the joint draws carry no
  QB-receiver correlation; head-to-head and CVaR rules see sums of
  independent position groups.
- The model is conditional on a player appearing in the box score. The
  roster status filter removes IR / practice squad, but not game-day
  inactives or "questionable" tags; `decide` will start a player who ends up
  not playing.
- Team form's variance parameters are estimated on all seasons fit, so in a
  backtest they've seen later seasons (`build-team-form --through-season`
  for a strict walk-forward run).
- Weather for future games is unknown in the schedule and falls back to the
  same 60°F / 8 mph defaults the training data uses for missing values.
