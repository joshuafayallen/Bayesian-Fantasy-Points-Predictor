"""
Alternative #2 to the strict hierarchical model: smooth (HSGP) age, in-season
form, and secular-trend curves in place of discrete hierarchical buckets.

WHAT'S DIFFERENT FROM hierarchical-model.py
--------------------------------------------
The strict hierarchical model has no explicit notion of "career stage" or
"time of season" at all -- tenure and week only show up as encoded indices
used for train/test bookkeeping, plus a `lag_*` feature or two. Any aging
curve (rookie ramp-up, prime, decline) or in-season trend (early rust, bye
weeks, playoff-push usage bumps) has to be absorbed into the iid
player-season intercept, which can't distinguish "this player is declining"
from "this player just had a noisy season".

This script models all three as smooth latent *functions* via Hilbert Space
GPs (HSGP, the scalable approximate-GP the project already had the
hyperparameter-selection code for -- see the originally-unused
`games_m/games_c` and `age_m/age_c` calls near the top of
hierarchical-model.py, which this script finishes):

    f_age(tenure)         -- smooth curve over career season number
    f_form(week)          -- smooth curve over week within season
    f_season(season)      -- smooth secular league-wide trend, 2006-2025
                             (replaces the binary 'era' flag -- see below)
    f_avail(games_missed) -- smooth curve over how many of the player's own
                             team's games they've missed so far this season,
                             before the current week. 0 = fully available
                             (played every team game so far, including a
                             week-1 debut -- team's played 0 games too at
                             that point, so no false injury signal); rises
                             for however many games they've sat out.
                             Anchored to the team's own game count rather
                             than calendar week specifically so a bye week
                             or a normal season-opening debut doesn't read
                             as "missed time" the way a raw games-played
                             count would. Stands in for injury-report data,
                             which can't help here anyway: a player who's
                             fully Out has no stat line to attach a flag to,
                             so there's nothing to join. This catches the
                             "played through it" and "just back, still
                             rusty" cases via games actually missed instead.

GP evaluated on the unique-value grid, not per-row
----------------------------------------------------
tenure/week/season each take only a handful of distinct values (~13-23,
18, and 20 respectively) but are repeated across ~98k rows. Evaluating the
HSGP basis at every row wastes compute building a huge, massively redundant
basis matrix. Instead each curve is evaluated once per distinct value
(`dims='tenure'` / `'week'` / `'season'`) and the per-row effect is a plain
gather, `f_age_grid[tenure_data]` -- exactly the same "evaluate once per
level, index per row" pattern already used for player/team effects
everywhere else in this project.

Lengthscale priors use `preliz.maxent` to pick InverseGamma(alpha, beta) that
concentrate mass over a plausible range, same convention as the (unused)
`games_form`/`age_curve` calls in hierarchical-model.py -- rather than
guessing alpha/beta by hand.

THIRD CURVE: replacing the binary 'era' flag
---------------------------------------------
The training data spans 2006-2025. Over that span the league's offense has
drifted steadily (rising pass volume/scoring), but the only adjustment for
that in the original pipeline is a single binary flag (season >= 2018).
That's a hard step where the real drift is gradual, and the cutoff year is
arbitrary. This script drops that binary flag (it's excluded from
`binary_vars` via the `binary_cols=` argument to `prepare()` below) and
replaces it with `f_season` above. 'season' isn't one of data_prep.py's
shared IDX_COLS, so it gets a small local index/grid built right here.

The other two alternative-model scripts (dynamic-skill-model.py,
bart-model.py) still use the binary 'era' flag from data_prep.py -- this
smooth-trend treatment is specific to this script, since the GP machinery
for it already lives here.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import polars as pl
import pymc as pm
import preliz as pz
import arviz as az

from data_prep import prepare, cont_binary_arrays, BINARY_COLS
from eval_utils import convergence_report, get_pp_array, rmse_report

SEED = 41029041
np.random.seed(SEED)

RESULTS = Path(__file__).parent.parent / "results" / "hsgp-model"
RESULTS.mkdir(parents=True, exist_ok=True)

# Drop 'era' from the binary covariates -- it's replaced by f_season below,
# so it shouldn't also enter as a blunt step dummy.
BI_COLS_NO_ERA = [c for c in BINARY_COLS if c != "era"]

train, test, coords, scalers, idxs = prepare(binary_cols=BI_COLS_NO_ERA)
cont_train, bi_train = cont_binary_arrays(train, coords, binary_cols=BI_COLS_NO_ERA)

# 'season' isn't in data_prep.IDX_COLS (it's a plain int column, not one of
# the shared categorical dims) -- build a small local index + coord for it
# here, same recipe data_prep.make_indices/encode use for everything else.
season_categories = sorted(train["season"].unique().to_list())
season_code = {s: i for i, s in enumerate(season_categories)}
train = train.with_columns(pl.col("season").replace(season_code).alias("season_idx"))
coords["season"] = [str(s) for s in season_categories]

# Games MISSED so far this season, before this week: how many of the
# player's own team's games has this player NOT appeared in. 0 = fully
# available (played every team game so far, including a week-1 debut --
# team's played 0 games too at that point, so no false injury signal).
# >0 = sat out that many of the team's games, the injury-return proxy.
# Built locally rather than trusting ff-processed.parquet's own
# games_missed_ytd column, which reflects the pre-fix data-cleaning.py:
# not yet regenerated as of this run, and cum_count() is an unsigned type,
# so the naive team-minus-player subtraction can underflow to ~4.3 billion
# instead of going negative (see the data-cleaning.py fix for the full
# story). Recomputed here with the same dedup guard as before, cast to a
# signed type, and clipped at 0.
train = (
    train
    .unique(subset=["player_id", "season", "week"], keep="first")
    .sort(["player_id", "season", "week"])
    .with_columns(
        (pl.col("week").cum_count().over(["player_id", "season"]) - 1)
        .alias("player_games_played_ytd")
    )
)
team_games = (
    train.select(["team", "season", "week"]).unique()
    .sort(["team", "season", "week"])
    .with_columns(
        (pl.col("week").cum_count().over(["team", "season"]) - 1)
        .alias("team_games_played_ytd")
    )
)
train = (
    train.join(team_games, on=["team", "season", "week"], how="left")
    .with_columns(
        (
            pl.col("team_games_played_ytd").cast(pl.Int32)
            - pl.col("player_games_played_ytd").cast(pl.Int32)
        ).clip(lower_bound=0).alias("games_missed_ytd")
    )
)
games_missed_categories = sorted(train["games_missed_ytd"].unique().to_list())
games_missed_code = {g: i for i, g in enumerate(games_missed_categories)}
train = train.with_columns(
    pl.col("games_missed_ytd").replace(games_missed_code).alias("games_missed_idx")
)
coords["games_missed"] = [str(g) for g in games_missed_categories]

# Grids of the actual (float) covariate values, one per distinct level, in
# the same order as their coord/dim -- this is the GP input. The gather
# index for row i is just the *_idx column already sitting on train.
tenure_grid_vals = np.array(coords["tenure"], dtype="float64")
week_grid_vals = np.array(coords["week"], dtype="float64")
season_grid_vals = np.array(season_categories, dtype="float64")
games_missed_grid_vals = np.array(games_missed_categories, dtype="float64")

# Lengthscale priors via preliz.maxent (same convention as the unused
# games_form/age_curve calls in hierarchical-model.py): concentrate mass of
# an InverseGamma over a plausible lengthscale range instead of guessing
# alpha/beta by hand. Season gets a longer range -- it's meant to capture a
# slow secular drift, not year-to-year noise.
age_curve, _ = pz.maxent(pz.InverseGamma(), lower=2, upper=6, plot=False)
games_form, _ = pz.maxent(pz.InverseGamma(), lower=2, upper=5, plot=False)
season_curve, _ = pz.maxent(pz.InverseGamma(), lower=3, upper=10, plot=False)
# Shorter range than the others: a "just back from injury / early rust"
# effect plausibly resolves within a handful of games, not the 2-10 game
# spans used for age/form/season -- a starting guess, check against the
# prior-predictive curve shape before trusting it.
avail_curve, _ = pz.maxent(pz.InverseGamma(), lower=1, upper=4, plot=False)

# m/c heuristic keyed off the grid's own range (identical to the per-row
# range since the grid *is* the set of distinct values).
age_m, age_c = pm.gp.hsgp_approx.approx_hsgp_hyperparams(
    x_range=[tenure_grid_vals.min(), tenure_grid_vals.max()], lengthscale_range=[2, 6], cov_func="matern52"
)
form_m, form_c = pm.gp.hsgp_approx.approx_hsgp_hyperparams(
    x_range=[week_grid_vals.min(), week_grid_vals.max()], lengthscale_range=[2, 5], cov_func="matern52"
)
season_m, season_c = pm.gp.hsgp_approx.approx_hsgp_hyperparams(
    x_range=[season_grid_vals.min(), season_grid_vals.max()], lengthscale_range=[3, 10], cov_func="matern52"
)
avail_m, avail_c = pm.gp.hsgp_approx.approx_hsgp_hyperparams(
    x_range=[games_missed_grid_vals.min(), games_missed_grid_vals.max()],
    lengthscale_range=[1, 4], cov_func="matern52"
)
print(f"age curve HSGP:    m={age_m} c={age_c:.2f}   ({len(tenure_grid_vals)} tenure levels)")
print(f"form curve HSGP:   m={form_m} c={form_c:.2f}  ({len(week_grid_vals)} week levels)")
print(f"season curve HSGP: m={season_m} c={season_c:.2f}  ({len(season_grid_vals)} season levels)")
print(f"avail curve HSGP:  m={avail_m} c={avail_c:.2f}  ({len(games_missed_grid_vals)} games-missed levels)")

with pm.Model(coords=coords) as hsgp_model:
    player_data = pm.Data("player_data", train["player_idx"].to_numpy())
    player_season_data = pm.Data("player_season_data", train["player_season_idx"].to_numpy())
    team_data = pm.Data("team_data", train["team_idx"].to_numpy())
    position_data = pm.Data("position_data", train["position_idx"].to_numpy())

    # Gather indices (one per row) -- these pick a row's value out of the grid.
    tenure_data = pm.Data("tenure_data", train["tenure_idx"].to_numpy())
    week_data = pm.Data("week_data", train["week_idx"].to_numpy())
    season_data = pm.Data("season_data", train["season_idx"].to_numpy())
    games_missed_data = pm.Data("games_missed_data", train["games_missed_idx"].to_numpy())

    # Grids (one row per distinct level) -- these are what the GP is evaluated on.
    tenure_grid = pm.Data("tenure_grid", tenure_grid_vals, dims="tenure")
    week_grid = pm.Data("week_grid", week_grid_vals, dims="week")
    season_grid = pm.Data("season_grid", season_grid_vals, dims="season")
    games_missed_grid = pm.Data("games_missed_grid", games_missed_grid_vals, dims="games_missed")

    cont_data = pm.Data("cont_data", cont_train)
    bi_data = pm.Data("bi_data", bi_train)
    obs_data = pm.Data("obs_data", train["total_fantasy_points"].to_numpy())

    # ---- position/team/player-season structure: unchanged, non-centered --
    positions_sigma = pm.HalfStudentT("positions_sigma", nu=4, sigma=1)
    positions_mu = pm.Normal("positions_mu", 0, 1)
    positions_intercepts = pm.Normal(
        "positions_intercepts", positions_mu, positions_sigma, dims="position"
    )

    team_sigma = pm.HalfStudentT("team_sigma", nu=4, sigma=2.5)
    team_intercepts = pm.ZeroSumNormal("team_intercepts", team_sigma, dims="team")

    player_sigma = pm.HalfStudentT("player_sigma", nu=4, sigma=2.5)
    player_mu = pm.Normal("player_mu", 0, 1.0, dims="player")
    player_intercepts = pm.Deterministic("player_intercepts", player_sigma * player_mu, dims="player")

    player_season_sigma = pm.HalfStudentT("player_season_sigma", nu=4, sigma=2.5)
    player_season_mu = pm.Normal("player_season_mu", 0, 1, dims="player_season")
    player_season_intercept = pm.Deterministic(
        "player_season_intercept", player_season_sigma * player_season_mu, dims="player_season"
    )

    # ---- THE ALTERNATIVE: three curves, each evaluated once per level on
    # its grid (dims='tenure'/'week'/'season'), then gathered per row. This
    # is the same "evaluate once, index per row" pattern as the hierarchical
    # effects above -- just with a GP prior over the levels instead of an
    # iid/partial-pooling one, since tenure/week/season have a meaningful
    # *order* that a plain hierarchical prior would ignore.
    #
    # parametrization="noncentered" is PyMC's own HSGP default and the
    # generally-safer choice for MCMC geometry; flip to "centered" if you'd
    # rather match the amount-of-data-per-level tradeoff differently.
    ell_age = pm.InverseGamma("ell_age", alpha=age_curve.alpha, beta=age_curve.beta)
    eta_age = pm.Exponential("eta_age", 1)
    cov_age = eta_age**2 * pm.gp.cov.Matern52(1, ls=ell_age)
    gp_age = pm.gp.HSGP(m=[age_m], c=age_c, cov_func=cov_age, drop_first=True, parametrization="noncentered")
    f_age_grid = gp_age.prior("f_age_grid", X=tenure_grid[:, None], dims="tenure")
    f_age = f_age_grid[tenure_data]

    ell_form = pm.InverseGamma("ell_form", alpha=games_form.alpha, beta=games_form.beta)
    eta_form = pm.Exponential("eta_form", 1)
    cov_form = eta_form**2 * pm.gp.cov.Matern52(1, ls=ell_form)
    gp_form = pm.gp.HSGP(m=[form_m], c=form_c, cov_func=cov_form, drop_first=True, parametrization="noncentered")
    f_form_grid = gp_form.prior("f_form_grid", X=week_grid[:, None], dims="week")
    f_form = f_form_grid[week_data]

    ell_season = pm.InverseGamma("ell_season", alpha=season_curve.alpha, beta=season_curve.beta)
    eta_season = pm.Exponential("eta_season", 1)
    cov_season = eta_season**2 * pm.gp.cov.Matern52(1, ls=ell_season)
    gp_season = pm.gp.HSGP(
        m=[season_m], c=season_c, cov_func=cov_season, drop_first=True, parametrization="noncentered"
    )
    f_season_grid = gp_season.prior("f_season_grid", X=season_grid[:, None], dims="season")
    f_season = f_season_grid[season_data]

    ell_avail = pm.InverseGamma("ell_avail", alpha=avail_curve.alpha, beta=avail_curve.beta)
    eta_avail = pm.Exponential("eta_avail", 1)
    cov_avail = eta_avail**2 * pm.gp.cov.Matern52(1, ls=ell_avail)
    gp_avail = pm.gp.HSGP(m=[avail_m], c=avail_c, cov_func=cov_avail, drop_first=True, parametrization="noncentered")
    f_avail_grid = gp_avail.prior("f_avail_grid", X=games_missed_grid[:, None], dims="games_missed")
    f_avail = f_avail_grid[games_missed_data]

    intercepts = pm.Deterministic(
        "intercepts",
        positions_intercepts[position_data]
        + team_intercepts[team_data]
        + player_intercepts[player_data]
        + player_season_intercept[player_season_data]
        + f_age
        + f_form
        + f_season
        + f_avail,
    )

    cont_priors = pm.Normal("cont_priors", 0, 1, dims="cont_vars")
    bi_priors = pm.Normal("bi_priors", 0, 1.5, dims="binary_vars")

    mu = pm.Deterministic(
        "mu",
        intercepts + pm.math.dot(cont_data, cont_priors) + pm.math.dot(bi_data, bi_priors),
    )

    # ---- skew-Student-T likelihood, position-indexed skew (same as base) --
    FLOOR = 1.05
    s = pm.Deterministic(
        "skew_sum", 2 * FLOOR + pm.Gamma("skew_sum_raw", mu=12, sigma=3, dims="position")
    )
    f = pm.Beta("skew_frac", 2, 2, dims="position")
    half_range = s / 2 - FLOOR
    skew_a = pm.Deterministic("skew_a", s / 2 + (2 * f - 1) * half_range, dims="position")
    skew_b = pm.Deterministic("skew_b", s / 2 - (2 * f - 1) * half_range, dims="position")
    sigma_obs = pm.HalfNormal("sigma_obs", 5.0)

    pm.SkewStudentT(
        "y_obs",
        a=skew_a[position_data],
        b=skew_b[position_data],
        nu=4,
        mu=mu,
        sigma=sigma_obs,
        observed=obs_data,
    )


if __name__ == "__main__":
    with hsgp_model:
        idata = pm.sample_prior_predictive(random_seed=SEED)
    y_prior = idata.prior_predictive["y_obs"].values
    print(f"prior predictive y_obs: min={y_prior.min():.1f} max={y_prior.max():.1f} "
          f"mean={y_prior.mean():.1f} std={y_prior.std():.1f}")

    with hsgp_model:
        try:
            idata.extend(pm.sample(nuts_sampler="nutpie", random_seed=SEED))
        except Exception as e:
            print(f"nutpie unavailable/failed ({e}); falling back to default NUTS")
            idata.extend(pm.sample(random_seed=SEED))
    idata.to_netcdf(RESULTS / "idata.nc")

    convergence_report(idata)

    with hsgp_model:
        idata.extend(pm.sample_posterior_predictive(idata, var_names=["y_obs"]))
    idata.to_netcdf(RESULTS / "idata.nc")

    pp_arr = get_pp_array(idata)
    rmse_report(train, pp_arr)

    # Plot the three learned curves -- this is the whole point of the model.
    # Each is already one value per grid level, so no dedup/sort-by-x dance
    # is needed the way it would be for a per-row GP.
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(19, 4))
    for ax, x, var, title in [
        (axes[0], tenure_grid_vals, "f_age_grid", "age / tenure curve"),
        (axes[1], week_grid_vals, "f_form_grid", "in-season form curve"),
        (axes[2], season_grid_vals, "f_season_grid", "season / secular trend curve"),
        (axes[3], games_missed_grid_vals, "f_avail_grid", "games-missed / availability curve"),
    ]:
        f_post = idata.posterior[var].stack(s=("chain", "draw")).values
        mean_curve = f_post.mean(axis=1)
        lo, hi = np.percentile(f_post, [5.5, 94.5], axis=1)
        ax.plot(x, mean_curve, "C0")
        ax.fill_between(x, lo, hi, alpha=0.2)
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(RESULTS / "curves.png", dpi=110)
