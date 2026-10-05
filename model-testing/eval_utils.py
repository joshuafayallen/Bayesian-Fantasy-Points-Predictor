"""
Small, model-agnostic diagnostics/scoring helpers shared by the model scripts.
 
Everything here takes a single InferenceData object and pulls what it needs out
of the standard groups:
 
    idata.posterior_predictive['y_obs']   -> predictions
    idata.observed_data['y_obs']          -> actuals
    idata.constant_data['position_data']  -> position index per row
    idata.posterior.coords['position']    -> index -> label mapping
 
so you never have to keep a separate `train` frame lined up with the fit. That
alignment was the main source of silent errors -- if the frame and the fit ever
drifted out of row order, the RMSE was quietly wrong with no warning.
 
This is NOT the ModelSpec/backtest/CRPS harness in hierarchical-model.py --
these are the generic "is this fit healthy / how good are the predictions"
utilities for a single fit.
"""
 
import numpy as np
import arviz as az
 
 
def rmse(pred, actual):
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(actual)) ** 2)))
 
 
def mae(pred, actual):
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(actual))))
 
 
# --------------------------------------------------------------- extractors

def _obs_dim(da, n_obs=None):
    """Name of the observation dim on a posterior-predictive DataArray.

    Don't assume the sample dims are exactly ('chain', 'draw'). Depending on how
    the idata was produced -- predictions=True, a reloaded netcdf, an already
    stacked group -- they can differ, and a hardcoded
    .mean(dim=('chain','draw')) then leaves a stray sample dim attached. That's
    what produces "Boolean array size N is used to index array with shape
    (4, N)" downstream: the (4, N) is (chain, obs).

    Identify the obs dim by matching its length to the observed data, falling
    back to "whatever isn't a known sample dim".
    """
    if n_obs is not None:
        matches = [d for d in da.dims if da.sizes[d] == n_obs]
        if len(matches) == 1:
            return matches[0]
    candidates = [d for d in da.dims if d not in ("chain", "draw", "sample")]
    if not candidates:
        raise ValueError(f"can't identify an observation dim among {da.dims}")
    return candidates[-1]
 
def pp_mean(idata, y_var="y_obs"):
    """Posterior predictive mean per observation, averaged over chain AND draw.
 
    Note the dim-name reduction rather than an axis number -- averaging over
    `axis=1` only collapses `draw` and silently leaves a (chain, obs) array,
    which then broadcasts against y in confusing ways.
    """
    if "posterior_predictive" not in idata:
        raise KeyError(
            "no posterior_predictive group -- run "
            "pm.sample_posterior_predictive(idata, extend_inferencedata=True) first"
        )
    da = idata.posterior_predictive[y_var]
    n_obs = int(idata.observed_data[y_var].size) if "observed_data" in idata else None
    obs_dim = _obs_dim(da, n_obs)
    out = da.mean(dim=[d for d in da.dims if d != obs_dim]).values
    assert out.ndim == 1, f"expected 1-D predictions, got {out.shape} (dims={da.dims})"
    return out
 
 
def pp_draws(idata, y_var="y_obs"):
    """Posterior predictive as (n_obs, n_samples), aligned to data row order.
 
    Keep this when you need the full distribution -- quantiles, coverage,
    P(A > B) style comparisons -- rather than just the mean.
    """
    da = idata.posterior_predictive[y_var]
    n_obs = int(idata.observed_data[y_var].size) if "observed_data" in idata else None
    obs_dim = _obs_dim(da, n_obs)
    sample_dims = [d for d in da.dims if d != obs_dim]
    return da.stack(s=sample_dims).transpose(obs_dim, "s").values
 
 
def observed(idata, y_var="y_obs"):
    return idata.observed_data[y_var].values
 
 
def positions(idata, idx_var="position_data", coord="position"):
    """Per-row position labels, reconstructed from the model's own index data
    and coords. Returns None if the model didn't register them, so callers can
    degrade gracefully to overall-only scoring."""
    if "constant_data" not in idata or idx_var not in idata.constant_data:
        return None
    if coord not in idata.posterior.coords:
        return None
    idx = idata.constant_data[idx_var].values.astype(int)
    labels = np.asarray(idata.posterior.coords[coord].values)
    return labels[idx]
 
 
# -------------------------------------------------------------- diagnostics
 
def convergence_report(idata, ess_target=400, rhat_target=1.01, exclude_dims=("obs",)):
    """Divergences + worst r_hat/ESS across every free parameter. Returns the
    headline numbers so callers can assert on them if they want."""
    post = idata.posterior
    ss = idata.sample_stats if "sample_stats" in idata else None
 
    print("=" * 62)
    print("SAMPLER DIAGNOSTICS")
    print("=" * 62)
    n_div = None
    if ss is not None and "diverging" in ss:
        n_div, n = int(ss["diverging"].sum()), int(ss["diverging"].size)
        flag = "   <-- INVESTIGATE" if n_div else ""
        print(f"  divergences      : {n_div} / {n}  ({100 * n_div / n:.2f}%){flag}")
 
    keep = [v for v in post.data_vars if not any(d in exclude_dims for d in post[v].dims)]
    rh = az.rhat(idata, var_names=keep)
    eb = az.ess(idata, var_names=keep, method="bulk")
    et = az.ess(idata, var_names=keep, method="tail")
 
    def flat(ds):
        return {v: np.atleast_1d(ds[v].values.reshape(-1).astype(float)) for v in ds.data_vars}
 
    RH, EB, ET = flat(rh), flat(eb), flat(et)
    a_rh = np.concatenate(list(RH.values()))
    a_eb = np.concatenate(list(EB.values()))
    a_et = np.concatenate(list(ET.values()))
 
    print("\n" + "=" * 62)
    print("CONVERGENCE ACROSS ALL PARAMETERS")
    print("=" * 62)
    print(f"  parameters checked : {a_rh.size}")
    print(f"  max r_hat          : {np.nanmax(a_rh):.3f}   (# > {rhat_target}: {int(np.nansum(a_rh > rhat_target))})")
    print(f"  min ess_bulk       : {np.nanmin(a_eb):.0f}   (# < {ess_target}: {int(np.nansum(a_eb < ess_target))})")
    print(f"  min ess_tail       : {np.nanmin(a_et):.0f}   (# < {ess_target}: {int(np.nansum(a_et < ess_target))})")
 
    print("\n  worst variables (by min ess_bulk):")
    for v in sorted(EB, key=lambda v: np.nanmin(EB[v]))[:6]:
        print(f"    {v:26s} ess_bulk_min={np.nanmin(EB[v]):>7.0f}  r_hat={np.nanmax(RH[v]):.3f}")
 
    return dict(divergences=n_div, max_rhat=float(np.nanmax(a_rh)),
                min_ess_bulk=float(np.nanmin(a_eb)), min_ess_tail=float(np.nanmin(a_et)))
 
 
def rmse_report(idata, y_var="y_obs", pos_labels=("QB", "RB", "WR", "TE")):
    """Model predictive mean vs. a global-mean baseline, overall and by position."""
    y = observed(idata, y_var)
    pred = pp_mean(idata, y_var)
    pos = positions(idata)
 
    b_global = np.full(len(y), y.mean())
 
    print(f"{'predictor':22s}{'RMSE':>8s}{'MAE':>8s}")
    print(f"{'global mean':22s}{rmse(b_global, y):>8.2f}{mae(b_global, y):>8.2f}")
    print(f"{'MODEL':22s}{rmse(pred, y):>8.2f}{mae(pred, y):>8.2f}")
 
    if pos is not None:
        print(f"\n{'pos':5s}{'n':>7s}{'obs_sd':>8s}{'model_RMSE':>12s}")
        for p in pos_labels:
            m = pos == p
            if m.sum() == 0:
                continue
            print(f"{p:5s}{int(m.sum()):>7d}{y[m].std():>8.2f}{rmse(pred[m], y[m]):>12.2f}")
 
    return dict(model_rmse=rmse(pred, y), model_mae=mae(pred, y))
 
 
def ppc_by_position(idata, y_var="y_obs"):
    """Observed vs posterior-predictive mean and skew, per position.
 
    Skew is the interesting column: a symmetric likelihood can still produce
    positive aggregate skew purely from mu varying across heterogeneous rows,
    so a gap here isn't automatically an argument for a skewed likelihood.
    """
    from scipy.stats import skew
 
    y = observed(idata, y_var)
    draws = pp_draws(idata, y_var)
    pos = positions(idata)
    if pos is None:
        raise KeyError("model didn't register position_data / position coord")
 
    print(f"{'pos':5s}{'obs_mean':>10s}{'pp_mean':>9s}{'obs_skew':>10s}{'pp_skew':>9s}")
    out = {}
    for p in np.unique(pos):
        m = pos == p
        obs_mean, pp_m = y[m].mean(), draws[m].mean()
        obs_sk = float(skew(y[m]))
        pp_sk = float(np.mean([skew(draws[m, j]) for j in range(0, draws.shape[1], 25)]))
        print(f"{p:5s}{obs_mean:>10.1f}{pp_m:>9.1f}{obs_sk:>10.2f}{pp_sk:>9.2f}")
        out[p] = dict(obs_mean=obs_mean, pp_mean=pp_m, obs_skew=obs_sk, pp_skew=pp_sk)
    return out
 
 
# ------------------------------------------------------------------ driver
 
def report(models, compare=True, ic="loo"):
    """Run every diagnostic over a {name: idata} dict, then compare them.
 
        report({'mlm': idata_mlm, 'ar': idata_ar})
 
    az.compare() takes the same dict, so one variable drives everything.
    """
    out = {}
    for name, idata in models.items():
        print(f"\n{'#' * 62}\n# {name}\n{'#' * 62}")
        conv = convergence_report(idata)
        print()
        sc = rmse_report(idata)
        out[name] = {**conv, **sc}
 
    if compare and len(models) > 1:
        print(f"\n{'#' * 62}\n# MODEL COMPARISON ({ic.upper()})\n{'#' * 62}")
        # nutpie doesn't store log_likelihood -- call
        # pm.compute_log_likelihood(idata, model=model) after sampling or this
        # will raise.
        cmp = az.compare(models, ic=ic)
        print(cmp[["rank", "elpd_loo", "elpd_diff", "dse", "weight"]])
        out["comparison"] = cmp
 
    return out