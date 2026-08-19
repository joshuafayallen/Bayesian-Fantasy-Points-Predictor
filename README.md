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
  estimate-latent-ability.py  Bradley-Terry-style state-space team-strength filter (feeds team_form feature)
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
.agents/skills/   Claude Code skill references used during development (pymc-modeling, prior-elicitation, model-evaluation, pymc-extras)
```

## Models in `src/ff-ar.py`

`ff-ar.py` builds three related hierarchical models for `total_fantasy_points`, sharing the same data-prep, priors on continuous/binary covariates, and a two-component observation mixture (a "limited role" component vs. a "full" component, gated on recent games-missed and trailing snap share, so barely-used players don't drag down the main regression's residual scale). They're compared against each other with `az.compare` / LOO at the bottom of the script.

**`ar_mod`** — the baseline. A flat per-position intercept, a random-walk-over-tenure player-skill curve (centered to zero mean within each position so it doesn't fight the position intercept), a non-centered AR(1)-with-season-jumps opponent-defense strength curve (`opp_ar`) shared across the whole weekly timeline, and a simple per-player-season "recent form" term (a single persistent random shift per season rather than a within-season AR — an earlier within-season AR(1)-over-week version came back with `rho ≈ 0.92` and `sigma` piled at zero, i.e. no real week-to-week evolution beyond one seasonal shift).

**`hs_version`** — same skeleton as `ar_mod`, with two structural swaps. The flat position intercept is replaced by an LKJ-correlated varying-effects intercept: each position gets a jointly-drawn (baseline level, age/tenure slope) pair, partially pooled across positions via an LKJ prior on their correlation. The per-season "recent form" AR term is replaced by a Hilbert-space GP (HSGP): a smooth nonparametric within-season curve sharing one eigenbasis across all player-seasons, with per-player-season coefficients on that shared basis, so it can bend into things like a gradual return-from-injury ramp that AR(1)'s pure exponential decay can't. Statistically tied with `ar_mod` on held-out CRPS/RMSE, but calibrates slightly better.

**`team_mod`** — `ar_mod`'s core (opponent AR, player skill, LKJ position intercept) plus a direct test of whether a player's *own* team's offensive quality matters. Neither `ar_mod` nor `hs_version` ever use the player's own team, only the opponent's defense — `team_mod` adds two separate coefficients, one on the player's own team-strength signal and one on the opponent's, rather than a single differenced term (since a good offense inflating a player's volume and a good opposing defense suppressing it aren't assumed to be equal-and-opposite effects).

All three use `pytensor.scan`-based recursions (rather than closed-form cumulative-sum/matrix constructions) for the AR and random-walk components, worked around a PyTensor graph-rewrite bug that a couple of structurally-similar closed-form formulations kept triggering.

### Other model alternatives

- **`ff-mlm.py`** explores a structural time-series / multilevel formulation via `pymc-extras`'s statespace module, fit on team-level score margins.
- **`bart-mod.py`** swaps the parametric hierarchical structure for BART (Bayesian Additive Regression Trees, via `pymc-bart`) — a nonparametric alternative to hand-specifying the AR/hierarchy structure above.
- **`estimate-latent-ability.py`** is a standalone Bradley-Terry-style local-level Kalman filter over score margins, fit separately and joined back in as the `team_form` feature used by `team_mod`.

## Decision layer

`decision_engine.py` scores whole *candidate lineups* on their joint posterior predictive scenario draws (not per-player marginals), because greedy per-slot optimization is only exact for maximizing expected points — chance-constrained floors, CVaR of shortfall, and head-to-head win probability all depend on the joint distribution of the lineup total, which correlated players (e.g. a QB and his own WR) distort in ways per-player summaries miss.

## To-do

- [ ] Extend the decision engine into a fuller probabilistic decision-under-uncertainty framework — chance-constrained score floors, CVaR of shortfall vs. a target/opponent, and a Pareto frontier over a risk-aversion parameter — following the framing in PyMC Labs' [Probabilistic Forecasting for Optimization Under Uncertainty](https://www.pymc-labs.com/blog-posts/probabilistic-forecasting-optimization-under-uncertainty) post (energy-generation planning under demand uncertainty, mapped here onto lineup decisions under score uncertainty).

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/) (`pyproject.toml` / `uv.lock`, Python >= 3.14):

```
uv sync
```

Model fits write large (multi-GB) `InferenceData` netCDF files to `model-nc/`; these are gitignored and regenerated by running the model scripts in `src/`.
