# Fantasy Football Bayesian Predictor

A fully Bayesian pipeline for forecasting weekly fantasy football points and turning those forecasts into lineup decisions. Every model here outputs a full posterior predictive *distribution* per player-week (not a point estimate), which is what makes downstream questions like "what's my floor 80% of the time" or "what's P(player A > player B)" answerable at all.

The pipeline has three stages:

1. **Data prep** — pull and clean play-by-play / roster data (`nflreadpy`), engineer lagged rate stats, and fit a standalone team-strength state-space model used as a feature.
2. **Modeling** — several alternative Bayesian model families for `total_fantasy_points`, fit in PyMC and compared via LOO-CV.
3. **Decision layer** — turn posterior predictive draws into actual start/sit and lineup choices under uncertainty (chance constraints, CVaR, head-to-head win probability).

## Project structure

```
src/
  data-cleaning.py            data pull + feature engineering -> processed-data/ff-processed.parquet
  estimate-latent-ability.py  local-level state-space team-strength filter (feeds team_form feature; see note in Models section)
  ff-ar.py                    primary hierarchical model family: ar_mod, hs_version, team_mod (see below)
  ff-mlm.py                   structural time-series / multilevel alternative (pymc-extras statespace)
  bart-mod.py                 BART (pymc-bart) alternative -- nonparametric, no explicit AR/hierarchy structure
  forecast_roster.py          posterior predictive draws -> per-player summaries -> greedy roster fill
  decision_engine.py          scenario-based lineup optimization under uncertainty (chance-constrained / CVaR)
  calibrate_decisions.py      calibration checks for the decision layer
  league_validate.py          backtests model output against real league outcomes
  adp_league.py, mock_league.py   ADP / mock-draft league simulation utilities
  sweep_summarize.py          summarize parameter-sweep results
  run_*.sh                    shell entry points for backtests / calibration / param sweeps

processed-data/   cleaned, model-ready parquet files (small, tracked)
raw-data/         raw pulled data (large, gitignored -- regenerate via data-cleaning.py)
model-nc/         fitted InferenceData netCDF files per model (huge, gitignored)
results/, figs/   diagnostic plots and backtest/calibration output
mock-rosters/     sample roster inputs for the decision engine
eval_scripts/     shared evaluation helpers
sandbox/backtest_harness.py    walk-forward backtest harness (see Backtesting below; rest of sandbox/ is gitignored scratch work)
sandbox/summarize_backtest.py  summarizes a backtest_harness.py results CSV into pooled metrics + a model winner
.agents/skills/   Claude Code skill references used during development (pymc-modeling, prior-elicitation, model-evaluation, pymc-extras)
```

## Models in `src/ff-ar.py`

`ff-ar.py` builds three related hierarchical models for `total_fantasy_points`, sharing the same data-prep, priors on continuous/binary covariates, and a two-component observation mixture. That mixture isn't modeling usage/games-missed directly — both components are distributions over *points output* (a "limited role" Student-t vs. the model's main "full role" Student-t); recent games-missed and trailing snap share instead drive the *mixing weight* between them, on the reasoning that a barely-used player's score isn't well explained by matchup/skill covariates the way a full-role player's is. They're compared against each other with `az.compare` / LOO at the bottom of the script, and against genuinely held-out folds via the walk-forward harness (see Backtesting below).

**`ar_mod`** — the baseline. A flat per-position intercept, a random-walk-over-tenure player-skill curve (centered to zero mean within each position so it doesn't fight the position intercept), a non-centered AR(1)-with-season-jumps opponent-defense strength curve (`opp_ar`) shared across the whole weekly timeline, and a per-player-season "recent form" random intercept — despite the name (`add_recent_form_ar`), this is *not* an AR process: it's a single non-centered `Normal(0, form_sigma)` draw per player-season, no time dimension at all. It replaced an earlier genuine within-season AR(1)-over-week version whose clean fit found `rho ≈ 0.92` but `sigma`'s posterior piled at the zero boundary — evidence the data supported essentially no real week-to-week evolution beyond one persistent per-season shift, so the AR machinery was dropped and only that persistent shift was kept.

**`hs_version`** — same skeleton as `ar_mod`, with two structural swaps. The flat position intercept is replaced by an LKJ-correlated varying-effects intercept: each position gets a jointly-drawn (baseline level, age/tenure slope) pair, partially pooled across positions via an LKJ prior on their correlation. The per-season random-intercept form term is replaced by a Hilbert-space GP (HSGP): a smooth nonparametric within-season curve sharing one eigenbasis across all player-seasons, with per-player-season coefficients on that shared basis, so it can bend into things like a gradual return-from-injury ramp that a static per-season shift can't. Statistically tied with `ar_mod` on held-out CRPS/RMSE, but calibrates slightly better.

**`team_mod`** — `ar_mod`'s core (opponent AR, player skill, LKJ position intercept), but with the recent-form term dropped entirely and replaced by a direct test of whether a player's *own* team's offensive quality matters. Neither `ar_mod` nor `hs_version` ever use the player's own team, only the opponent's defense — `team_mod` adds two separate coefficients, one on the player's own team-strength signal and one on the opponent's, rather than a single differenced term (since a good offense inflating a player's volume and a good opposing defense suppressing it aren't assumed to be equal-and-opposite effects). A separate, unused-in-any-active-model helper (`add_team_season_ar`, for a season-level AR(1) team-quality effect) also lives in this file — a fourth architecture (`team_season_mod`) that hasn't been wired up yet; see To-do.

All three use `pytensor.scan`-based recursions (rather than closed-form cumulative-sum/matrix constructions) for the AR and random-walk components, worked around a PyTensor graph-rewrite bug that a couple of structurally-similar closed-form formulations kept triggering.

### Other model alternatives

- **`ff-mlm.py`** is a simpler multilevel model on the same player-week data: a `ZeroSumNormal` per-team intercept, a per-position intercept, a player-skill random walk over tenure (the original closed-form `cumsum` construction, not the `scan`-based fix used in `ff-ar.py`), and one HSGP curve shared across *all* players over `week` (a league-wide seasonal shape, not per-player-season), with a single plain Student-t likelihood — no opponent-strength AR and no role/participation mixture.
- **`bart-mod.py`** swaps only the linear covariate terms for BART (Bayesian Additive Regression Trees, via `pymc-bart`) — it keeps a flat per-player intercept and the same opponent-strength AR(1) structure (built via the closed-form `rho_powers` matrix rather than `scan`), with a single plain Student-t likelihood and no recent-form term.
- **`estimate-latent-ability.py`** fits a local-level-trend + measurement-error state-space model (via `pymc-extras`'s statespace module) on home/rest-adjusted score margins, extracting each team's filtered per-week strength as the `team_form` feature `team_mod` uses. **Note:** `ff-ar.py`'s own comment attributes `team_form.parquet` to a different, more polished script — `sandbox/team_strength_ssm.py` (gitignored, not in this tracked repo) — which does the same thing but pulls schedules fresh via `nflreadpy` rather than reading the already-processed fantasy data. Both exist on disk; which one is actually canonical hasn't been reconciled yet.

## Backtesting

`src/ff-ar.py`'s own `fit_and_diagnose` only ever fits on one "everything we have" slice, so its in-sample comparison isn't real held-out accuracy. `sandbox/backtest_harness.py` is the genuine walk-forward evaluation: it ports `ar_mod`, `hs_version`, and `team_mod` faithfully out of `ff-ar.py` (same priors, same player-skill and opponent-AR construction, same two-component role-participation likelihood) and, for a given `(model, method, season, week)` fold, fits on a rolling window of prior seasons and scores the held-out week, appending one row to `results/backtest_walkforward.csv`. Each invocation runs exactly one fold and is checkpointed, so folds can be run one at a time, looped over a season, or run in parallel — `src/run_backtest.sh` does the looping; `sandbox/summarize_backtest.py` then pools the resulting CSV by model (weighted by fold size) and declares a winner, using held-out CRPS as the primary criterion (a proper scoring rule over the full predictive distribution, not just the mean) with RMSE as a tiebreak. Supports both a fast ADVI sweep and a slower NUTS confirmation pass on a subset of folds.

One noted limitation: `team_mod`'s team-strength feature is precomputed once by `estimate-latent-ability.py` across the full history, and while its per-week filtered estimates are genuinely causal (no future scores leak into a given week's value), the state-space model's own hyperparameters were fit on the full-history posterior — a form of leakage the harness does not yet correct for (would need refitting that state-space model per fold).

## Decision layer

`decision_engine.py` scores whole *candidate lineups* on their joint posterior predictive scenario draws (not per-player marginals), because greedy per-slot optimization is only exact for maximizing expected points — chance-constrained floors, CVaR of shortfall, and head-to-head win probability all depend on the joint distribution of the lineup total, which correlated players (e.g. a QB and his own WR) distort in ways per-player summaries miss.

## To-do

- [ ] Extend the decision engine into a fuller probabilistic decision-under-uncertainty framework — chance-constrained score floors, CVaR of shortfall vs. a target/opponent, and a Pareto frontier over a risk-aversion parameter — following the framing in PyMC Labs' [Probabilistic Forecasting for Optimization Under Uncertainty](https://www.pymc-labs.com/blog-posts/probabilistic-forecasting-optimization-under-uncertainty) post (energy-generation planning under demand uncertainty, mapped here onto lineup decisions under score uncertainty).
- [ ] Decide whether `team_season_mod` (a season-level AR(1) team-quality variant; the `add_team_season_ar` helper already exists in `ff-ar.py` but isn't wired into an active model) is worth adding as a fourth architecture, and port it into `backtest_harness.py` if so.
- [ ] Reconcile `src/estimate-latent-ability.py` and `sandbox/team_strength_ssm.py` — both fit essentially the same team-strength state-space model and write `team_form.parquet`, but only one should be canonical (the sandbox version is more complete: it pulls schedules fresh via `nflreadpy` and is properly modularized). Update `ff-ar.py`'s comment to match whichever wins.

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/) (`pyproject.toml` / `uv.lock`, Python >= 3.14):

```
uv sync
```

Model fits write large (multi-GB) `InferenceData` netCDF files to `model-nc/`; these are gitignored and regenerated by running the model scripts in `src/`.
