"""
Extract sampling diagnostics (ESS bulk/tail, R-hat, divergences, tree depth,
mean_tree_accept) from a ground-truth idata .nc file, without loading the
full multi-GB posterior into memory.

- "Small" params (no big indexed dim) are pulled in full.
- "Large" indexed params (player_skill, opp_ar, recent_form, intercepts,
  z_innov, opp_z_theta, opp_z_eta) are spot-checked at a handful of random
  coordinates along their big dimension(s) -- exactly the
  sample_check_coords spot-check pattern ff_ar.py itself already uses.
- sample_stats (divergences, tree depth, accept rate) are summarized in full
  (they're small: chain x draw).

Prints a JSON blob of {var_name: {rhat, ess_bulk, ess_tail}} plus a
sample_stats summary, so it's easy to diff against a fresh fit's numbers.
"""
import sys
import json
import numpy as np
import netCDF4
import arviz as az
import xarray as xr

SEED = 41029041
rng = np.random.default_rng(SEED)

SMALL_VARS = [
    'positions_mu', 'cont_priors', 'bi_priors', 'opp_theta_init',
    'role_intercept', 'role_beta_missed', 'role_beta_snap',
    'init_sigma', 'rw_sigma', 'opp_rho', 'opp_sigma_theta',
    'opp_sigma_season', 'form_sigma', 'sigma_obs', 'mu_limited', 'sigma_limited',
]
# name -> (big_dim_axis_indices_into_var.shape_after_chain_draw, n_spotchecks)
LARGE_VARS = {
    'z_innov': 2,          # (chain, draw, player, tenure) -> spot check player idx
    'opp_z_theta': 2,      # (chain, draw, opp_team, global_step) -> spot check opp_team
    'opp_z_eta': 2,        # (chain, draw, opp_team, season_gap)
    'form_z': 2,           # (chain, draw, player_season)
    'intercepts': 2,       # (chain, draw, intercepts_dim_0)
    'player_skill': 2,     # (chain, draw, player, tenure)
    'opp_ar': 2,           # (chain, draw, opp_team, global_step_full)
    'recent_form': 2,      # (chain, draw, player_season)
}
N_SPOT = 8


def load_small(post, name):
    var = post.variables[name]
    data = var[:]  # (chain, draw, ...)
    return xr.DataArray(data, dims=var.dimensions)


def load_large_spotcheck(post, name):
    var = post.variables[name]
    shape = var.shape  # (chain, draw, dim0, [dim1])
    ndim = len(shape)
    idx0_size = shape[2]
    picks0 = rng.choice(idx0_size, size=min(N_SPOT, idx0_size), replace=False)
    picks0.sort()
    results = {}
    if ndim == 3:
        for i in picks0:
            data = var[:, :, i]  # (chain, draw)
            results[f"{name}[{i}]"] = xr.DataArray(data, dims=('chain', 'draw'))
    elif ndim == 4:
        dim1_size = shape[3]
        picks1 = rng.choice(dim1_size, size=min(3, dim1_size), replace=False)
        for i in picks0:
            for j in picks1:
                data = var[:, :, i, j]  # (chain, draw)
                results[f"{name}[{i},{j}]"] = xr.DataArray(data, dims=('chain', 'draw'))
    return results


def main(path):
    out = {}
    with netCDF4.Dataset(path, 'r') as ds:
        post = ds.groups['posterior']

        for name in SMALL_VARS:
            if name not in post.variables:
                continue
            da = load_small(post, name)
            ds_ = xr.Dataset({name: da})
            rhat = az.rhat(ds_)[name].values
            ess_bulk = az.ess(ds_, method='bulk')[name].values
            ess_tail = az.ess(ds_, method='tail')[name].values
            out[name] = dict(
                rhat_max=float(np.nanmax(rhat)),
                rhat_min=float(np.nanmin(rhat)),
                ess_bulk_min=float(np.nanmin(ess_bulk)),
                ess_tail_min=float(np.nanmin(ess_tail)),
            )
            print(f"done small: {name}", file=sys.stderr)

        for name in LARGE_VARS:
            if name not in post.variables:
                continue
            spot = load_large_spotcheck(post, name)
            rhats, ess_bulks, ess_tails = [], [], []
            for key, da in spot.items():
                ds_ = xr.Dataset({key: da})
                rhats.append(float(az.rhat(ds_)[key].values))
                ess_bulks.append(float(az.ess(ds_, method='bulk')[key].values))
                ess_tails.append(float(az.ess(ds_, method='tail')[key].values))
            out[name] = dict(
                n_spotchecked=len(spot),
                rhat_max=float(np.nanmax(rhats)),
                rhat_min=float(np.nanmin(rhats)),
                ess_bulk_min=float(np.nanmin(ess_bulks)),
                ess_bulk_median=float(np.nanmedian(ess_bulks)),
                ess_tail_min=float(np.nanmin(ess_tails)),
            )
            print(f"done large: {name}", file=sys.stderr)

        ss = ds.groups['sample_stats']
        diverging = ss.variables['diverging'][:]
        depth = ss.variables['depth'][:] if 'depth' in ss.variables else None
        maxdepth = ss.variables['maxdepth_reached'][:] if 'maxdepth_reached' in ss.variables else None
        accept = ss.variables['mean_tree_accept'][:] if 'mean_tree_accept' in ss.variables else None
        n_chains, n_draws = diverging.shape
        out['__sample_stats__'] = dict(
            n_chains=int(n_chains),
            n_draws=int(n_draws),
            n_divergences=int(np.sum(diverging)),
            pct_divergences=float(100 * np.mean(diverging)),
            mean_tree_accept=float(np.mean(accept)) if accept is not None else None,
            max_depth_seen=int(np.max(depth)) if depth is not None else None,
            pct_hit_maxdepth=float(100 * np.mean(maxdepth)) if maxdepth is not None else None,
        )

    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main(sys.argv[1])
