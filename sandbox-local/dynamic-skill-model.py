"""
Alternative #1 to the strict hierarchical model: a dynamic latent-skill
(state-space) model.

WHAT'S DIFFERENT FROM hierarchical-model.py
--------------------------------------------
The strict hierarchical model treats each player-season as an *iid* draw
from a per-player distribution:

    player_season_intercept[p, s] ~ Normal(player_intercept[p], sigma)

That's exchangeable pooling -- a great season and a bad season next door
carry no information about each other beyond both being "this player's
seasons". There's no persistence, no momentum, no aging curve.

Here, each player's scoring ability is a latent state that evolves smoothly
across tenure (career season number) via a non-centered Gaussian random walk:

    skill[p, 0]   ~ Normal(0, init_sigma)
    skill[p, t]    = skill[p, t-1] + Normal(0, rw_sigma)

So adjacent seasons are correlated -- a breakout year nudges the trajectory
up and it decays gradually rather than snapping back to an independent
redraw next season. This is a genuinely different generative story for
"how does a player's true ability change over a career", not just a
reparameterization of the same prior.

Everything else (position/team structure, covariates, skew-Student-T
likelihood with position-indexed skew) is kept identical to
hierarchical-model.py so the two are comparable apples-to-apples.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import arviz as az

from data_prep import prepare, cont_binary_arrays
from eval_utils import convergence_report, get_pp_array, rmse_report

SEED = 41029041
np.random.seed(SEED)

RESULTS = Path(__file__).parent.parent / "results" / "dynamic-skill-model"
RESULTS.mkdir(parents=True, exist_ok=True)

train, test, coords, scalers, idxs = prepare()
cont_train, bi_train = cont_binary_arrays(train, coords)

print(f"train rows: {len(train):,}  players: {len(coords['player']):,}  "
      f"tenure levels: {len(coords['tenure'])}")

with pm.Model(coords=coords) as dynamic_skill_model:
    player_data = pm.Data("player_data", train["player_idx"].to_numpy())
    tenure_data = pm.Data("tenure_data", train["tenure_idx"].to_numpy())
    team_data = pm.Data("team_data", train["team_idx"].to_numpy())
    position_data = pm.Data("position_data", train["position_idx"].to_numpy())
    cont_data = pm.Data("cont_data", cont_train)
    bi_data = pm.Data("bi_data", bi_train)
    obs_data = pm.Data("obs_data", train["total_fantasy_points"].to_numpy())

    # ---- position + team: unchanged from the strict hierarchical model ----
    positions_sigma = pm.HalfStudentT("positions_sigma", nu=4, sigma=1)
    positions_mu = pm.Normal("positions_mu", 0, 1)
    positions_intercepts = pm.Normal(
        "positions_intercepts", positions_mu, positions_sigma, dims="position"
    )

    team_sigma = pm.HalfStudentT("team_sigma", nu=4, sigma=2.5)
    team_intercepts = pm.ZeroSumNormal("team_intercepts", team_sigma, dims="team")

    # ---- THE ALTERNATIVE: player skill as a career-long random walk -------
    init_sigma = pm.HalfNormal("init_sigma", 1.0)   # spread of rookie-year skill across players
    rw_sigma = pm.HalfNormal("rw_sigma", 0.35)      # season-to-season innovation size

    z_innov = pm.Normal("z_innov", 0, 1, dims=("player", "tenure"))  # non-centered

    level0 = init_sigma * z_innov[:, 0]
    steps = rw_sigma * z_innov[:, 1:]
    player_skill = pm.Deterministic(
        "player_skill",
        pt.concatenate(
            [level0[:, None], level0[:, None] + pt.cumsum(steps, axis=1)], axis=1
        ),
        dims=("player", "tenure"),
    )

    intercepts = pm.Deterministic(
        "intercepts",
        positions_intercepts[position_data]
        + team_intercepts[team_data]
        + player_skill[player_data, tenure_data],
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
    with dynamic_skill_model:
        idata = pm.sample_prior_predictive(random_seed=SEED)
    y_prior = idata.prior_predictive["y_obs"].values
    print(f"prior predictive y_obs: min={y_prior.min():.1f} max={y_prior.max():.1f} "
          f"mean={y_prior.mean():.1f} std={y_prior.std():.1f}")

    with dynamic_skill_model:
        try:
            idata.extend(pm.sample(nuts_sampler="nutpie", random_seed=SEED))
        except Exception as e:
            print(f"nutpie unavailable/failed ({e}); falling back to default NUTS")
            idata.extend(pm.sample(random_seed=SEED))
    idata.to_netcdf(RESULTS / "idata.nc")   # save immediately, before any post-processing

    convergence_report(
        idata,
    )

    with dynamic_skill_model:
        idata.extend(pm.sample_posterior_predictive(idata, var_names=["y_obs"]))
    idata.to_netcdf(RESULTS / "idata.nc")

    pp_arr = get_pp_array(idata)
    rmse_report(train, pp_arr)

    az.plot_trace(
        idata, var_names=["init_sigma", "rw_sigma", "team_sigma", "positions_sigma", "sigma_obs"],
        compact=True,
    )
    import matplotlib.pyplot as plt
    plt.savefig(RESULTS / "trace.png", dpi=110)
