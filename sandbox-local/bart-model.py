"""
Alternative #3 to the strict hierarchical model: BART (Bayesian Additive
Regression Trees) in place of the hand-specified linear/additive covariate
structure.

WHAT'S DIFFERENT FROM hierarchical-model.py
--------------------------------------------
The strict hierarchical model bakes in a specific functional form: position,
team, and every continuous/binary covariate enter as a flat linear
combination (`intercepts + cont_data @ cont_priors + bi_data @ bi_priors`).
Any nonlinearity or interaction (e.g. "the wind effect only matters for
outdoor stadiums", "spread_line matters differently for QBs than for TEs")
has to be hand-engineered as a new feature.

Here, position, team, and the continuous/binary covariates all go through a
single BART ensemble (`pymc_bart.BART`) instead -- a sum of shallow
regression trees that discovers nonlinearities and interactions among those
features automatically, with no interaction terms to specify by hand. This
follows the standard "BART + random effects" pattern: player and
player-season ability stay as ordinary (non-centered) hierarchical
intercepts -- persistent player-level skill is exactly the kind of smooth,
partially-pooled effect trees are bad at -- while BART absorbs the
situational/contextual part of the prediction.

Requires `pymc-bart` (added to pyproject.toml). BART uses a particle-Gibbs
step under the hood, so this uses plain `pm.sample()` rather than nutpie
(nutpie can't sample discrete tree structures).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pymc as pm
import pymc_bart as pmb
import arviz as az

from data_prep import prepare, cont_binary_arrays
from eval_utils import convergence_report, get_pp_array, rmse_report

SEED = 41029041
np.random.seed(SEED)

RESULTS = Path(__file__).parent.parent / "results" / "bart-model"
RESULTS.mkdir(parents=True, exist_ok=True)

train, test, coords, scalers, idxs = prepare()
cont_train, bi_train = cont_binary_arrays(train, coords)

n = len(train)


def one_hot(idx, n_levels):
    out = np.zeros((len(idx), n_levels), dtype=float)
    out[np.arange(len(idx)), idx] = 1.0
    return out


position_oh = one_hot(train["position_idx"].to_numpy(), len(coords["position"]))
team_oh = one_hot(train["team_idx"].to_numpy(), len(coords["team"]))

X_bart = np.hstack([cont_train, bi_train, position_oh, team_oh])
X_names = (
    list(coords["cont_vars"])
    + list(coords["binary_vars"])
    + [f"pos_{p}" for p in coords["position"]]
    + [f"team_{t}" for t in coords["team"]]
)
Y_bart = train["total_fantasy_points"].to_numpy()

print(f"X_bart shape: {X_bart.shape}  (n_features={len(X_names)})")

with pm.Model(coords=coords) as bart_model:
    player_data = pm.Data("player_data", train["player_idx"].to_numpy())
    player_season_data = pm.Data("player_season_data", train["player_season_idx"].to_numpy())
    position_data = pm.Data("position_data", train["position_idx"].to_numpy())

    # ---- player / player-season: kept as ordinary hierarchical, non-centered
    player_sigma = pm.HalfStudentT("player_sigma", nu=4, sigma=2.5)
    player_mu = pm.Normal("player_mu", 0, 1.0, dims="player")
    player_intercepts = pm.Deterministic("player_intercepts", player_sigma * player_mu, dims="player")

    player_season_sigma = pm.HalfStudentT("player_season_sigma", nu=4, sigma=2.5)
    player_season_mu = pm.Normal("player_season_mu", 0, 1, dims="player_season")
    player_season_intercept = pm.Deterministic(
        "player_season_intercept", player_season_sigma * player_season_mu, dims="player_season"
    )

    # ---- THE ALTERNATIVE: position/team/covariates all go through BART ----
    mu_bart = pmb.BART("mu_bart", X=X_bart, Y=Y_bart, m=50)

    mu = pm.Deterministic(
        "mu",
        mu_bart + player_intercepts[player_data] + player_season_intercept[player_season_data],
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
        observed=Y_bart,
    )


if __name__ == "__main__":
    try:
        with bart_model:
            idata = pm.sample_prior_predictive(random_seed=SEED)
        y_prior = idata.prior_predictive["y_obs"].values
        print(f"prior predictive y_obs: min={y_prior.min():.1f} max={y_prior.max():.1f} "
              f"mean={y_prior.mean():.1f} std={y_prior.std():.1f}")
    except Exception as e:
        print(f"prior predictive check skipped (BART priors don't sample cleanly "
              f"outside of `pm.sample`): {e}")
        idata = None

    with bart_model:
        # No nuts_sampler="nutpie" here -- BART needs the particle-Gibbs step,
        # nutpie can't handle it. pm.sample() auto-selects a compound sampler
        # (PGBART for mu_bart, NUTS for everything else).
        posterior_idata = pm.sample(random_seed=SEED)
    if idata is not None:
        idata.extend(posterior_idata)
    else:
        idata = posterior_idata
    idata.to_netcdf(RESULTS / "idata.nc")

    convergence_report(idata, exclude_dims=("obs", "mu_bart_dim_0"))

    with bart_model:
        idata.extend(pm.sample_posterior_predictive(idata, var_names=["y_obs"]))
    idata.to_netcdf(RESULTS / "idata.nc")

    pp_arr = get_pp_array(idata)
    rmse_report(train, pp_arr)

    # ---- calibration: log_likelihood + loo_pit + pareto-k ----------------
    # RMSE and PPC-by-position only check the point prediction (mean) and
    # low-order moments -- they don't test whether the predictive *spread*
    # is right. loo_pit does, so run it every time, alongside plot_khat:
    # PSIS-LOO's importance-sampling approximation can itself be unreliable
    # for BART (particle-Gibbs draws over discrete tree topologies, not a
    # smooth HMC trajectory), so a bad-looking loo_pit needs pareto_k ruled
    # out as the cause before it's read as a real miscalibration.
    with bart_model:
        pm.compute_log_likelihood(idata)
    idata.to_netcdf(RESULTS / "idata.nc")

    import matplotlib.pyplot as plt
    import arviz as az

    loo = az.loo(idata, pointwise=True, var_name="y_obs")
    print(loo)
    n_bad = int((loo.pareto_k.values > 0.7).sum())
    print(f"observations with pareto_k > 0.7: {n_bad} / {loo.pareto_k.size}")
    if n_bad:
        bad_idx = np.where(loo.pareto_k.values > 0.7)[0]
        print(f"row indices: {bad_idx[:20]}{'...' if len(bad_idx) > 20 else ''}")
        print("PSIS-LOO unreliable at these points -- treat loo_pit shape "
              "with caution until refit via az.reloo or k-fold CV on them.")

    az.plot_khat(loo, show_bins=True)
    plt.savefig(RESULTS / "khat.png", dpi=110)
    plt.close()

    az.plot_loo_pit(idata, y="y_obs")
    plt.savefig(RESULTS / "loo_pit.png", dpi=110)
    plt.close()

    vi = pmb.compute_variable_importance(idata, bart_model["mu_bart"], X_bart)
    pmb.plot_variable_importance(vi, labels=X_names)
    plt.savefig(RESULTS / "variable_importance.png", dpi=110)
