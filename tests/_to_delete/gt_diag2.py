"""
Generalized version of gt_diag.py: auto-discovers which posterior variables
are "small" (<= ~50 elements along non chain/draw dims -> pull in full) vs
"large" (indexed by player/opp_team/player_season/intercepts_dim_0/etc ->
spot-check a handful of random coordinates), instead of a hardcoded var
list. Works across ff_ar.nc / ff_hs.nc / ff_team.nc without editing.

Usage: python gt_diag2.py model-nc/ff_hs.nc
"""
import sys
import json
import numpy as np
import netCDF4
import arviz as az
import xarray as xr

SEED = 41029041
rng = np.random.default_rng(SEED)
N_SPOT = 8
SMALL_THRESHOLD = 64  # product of non-(chain,draw) dims <= this => pull in full


def main(path):
    out = {}
    with netCDF4.Dataset(path, 'r') as ds:
        post = ds.groups['posterior']

        for name, var in post.variables.items():
            if name in ('chain', 'draw') or name not in post.variables:
                continue
            shape = var.shape
            if len(shape) < 2 or var.dimensions[0] != 'chain' or var.dimensions[1] != 'draw':
                continue  # coordinate variable, not a posterior sample var
            extra_shape = shape[2:]
            extra_size = int(np.prod(extra_shape)) if extra_shape else 1

            try:
                if extra_size <= SMALL_THRESHOLD:
                    data = var[:]
                    da = xr.DataArray(data, dims=var.dimensions)
                    ds_ = xr.Dataset({name: da})
                    rhat = az.rhat(ds_)[name].values
                    ess_bulk = az.ess(ds_, method='bulk')[name].values
                    ess_tail = az.ess(ds_, method='tail')[name].values
                    out[name] = dict(
                        kind='full', shape=list(shape),
                        rhat_max=float(np.nanmax(rhat)), rhat_min=float(np.nanmin(rhat)),
                        ess_bulk_min=float(np.nanmin(ess_bulk)),
                        ess_tail_min=float(np.nanmin(ess_tail)),
                    )
                else:
                    ndim = len(shape)
                    idx0_size = shape[2]
                    picks0 = np.sort(rng.choice(idx0_size, size=min(N_SPOT, idx0_size), replace=False))
                    rhats, eb, et = [], [], []
                    if ndim == 3:
                        for i in picks0:
                            data = var[:, :, i]
                            da = xr.DataArray(data, dims=('chain', 'draw'))
                            ds_ = xr.Dataset({'v': da})
                            rhats.append(float(az.rhat(ds_)['v'].values))
                            eb.append(float(az.ess(ds_, method='bulk')['v'].values))
                            et.append(float(az.ess(ds_, method='tail')['v'].values))
                        n_checked = len(picks0)
                    else:
                        dim1_size = shape[3]
                        picks1 = np.sort(rng.choice(dim1_size, size=min(3, dim1_size), replace=False))
                        for i in picks0:
                            for j in picks1:
                                data = var[:, :, i, j]
                                da = xr.DataArray(data, dims=('chain', 'draw'))
                                ds_ = xr.Dataset({'v': da})
                                rhats.append(float(az.rhat(ds_)['v'].values))
                                eb.append(float(az.ess(ds_, method='bulk')['v'].values))
                                et.append(float(az.ess(ds_, method='tail')['v'].values))
                        n_checked = len(picks0) * len(picks1)
                    out[name] = dict(
                        kind='spotcheck', shape=list(shape), n_spotchecked=n_checked,
                        rhat_max=float(np.nanmax(rhats)), rhat_min=float(np.nanmin(rhats)),
                        ess_bulk_min=float(np.nanmin(eb)), ess_bulk_median=float(np.nanmedian(eb)),
                        ess_tail_min=float(np.nanmin(et)),
                    )
                print(f"done: {name}", file=sys.stderr)
            except Exception as e:
                out[name] = dict(error=str(e))
                print(f"ERROR on {name}: {e}", file=sys.stderr)

        ss = ds.groups['sample_stats']
        diverging = ss.variables['diverging'][:]
        depth = ss.variables['depth'][:] if 'depth' in ss.variables else None
        maxdepth = ss.variables['maxdepth_reached'][:] if 'maxdepth_reached' in ss.variables else None
        accept = ss.variables['mean_tree_accept'][:] if 'mean_tree_accept' in ss.variables else None
        n_chains, n_draws = diverging.shape
        out['__sample_stats__'] = dict(
            n_chains=int(n_chains), n_draws=int(n_draws),
            n_divergences=int(np.sum(diverging)),
            pct_divergences=float(100 * np.mean(diverging)),
            mean_tree_accept=float(np.mean(accept)) if accept is not None else None,
            max_depth_seen=int(np.max(depth)) if depth is not None else None,
            pct_hit_maxdepth=float(100 * np.mean(maxdepth)) if maxdepth is not None else None,
        )

    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main(sys.argv[1])
