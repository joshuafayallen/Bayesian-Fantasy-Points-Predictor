"""
Sampling-health comparison between a ground-truth idata .nc file and a
freshly-fit one -- NOT a value-level or exact-match comparison (independent
MCMC chains never match draw-for-draw), but a check that the sampler
geometry is still healthy: R-hat, ESS efficiency (ESS / total post-warmup
draws, since two fits can use different draws/chains), divergences, tree
depth, and acceptance rate.

Typical use: after touching src/models.py, refit a model with a modest
draws/tune/chains budget (a full-scale refit isn't needed -- these
diagnostics are about geometry, not final precision) and compare against
the archived ground-truth fit in model-nc/.

    python tests/diagnostics_report.py model-nc/ff_ar.nc my_new_fit.nc

Requires: netCDF4 (already a project dependency), arviz, xarray.
"""
import sys
import numpy as np
import netCDF4
import arviz as az
import xarray as xr

SEED = 41029041
N_SPOT = 8
SMALL_THRESHOLD = 64


def extract_diagnostics(path, seed=SEED):
    rng = np.random.default_rng(seed)
    out = {}
    with netCDF4.Dataset(path, 'r') as ds:
        post = ds.groups['posterior']
        for name, var in post.variables.items():
            if name in ('chain', 'draw'):
                continue
            shape = var.shape
            if len(shape) < 2 or var.dimensions[0] != 'chain' or var.dimensions[1] != 'draw':
                continue
            extra_shape = shape[2:]
            extra_size = int(np.prod(extra_shape)) if extra_shape else 1
            try:
                if extra_size <= SMALL_THRESHOLD:
                    data = var[:]
                    ds_ = xr.Dataset({name: xr.DataArray(data, dims=var.dimensions)})
                    rhat = az.rhat(ds_)[name].values
                    ess_bulk = az.ess(ds_, method='bulk')[name].values
                    out[name] = dict(
                        rhat_max=float(np.nanmax(rhat)),
                        ess_bulk_min=float(np.nanmin(ess_bulk)),
                    )
                else:
                    ndim = len(shape)
                    idx0_size = shape[2]
                    picks0 = np.sort(rng.choice(idx0_size, size=min(N_SPOT, idx0_size), replace=False))
                    rhats, eb = [], []
                    if ndim == 3:
                        for i in picks0:
                            ds_ = xr.Dataset({'v': xr.DataArray(var[:, :, i], dims=('chain', 'draw'))})
                            rhats.append(float(az.rhat(ds_)['v'].values))
                            eb.append(float(az.ess(ds_, method='bulk')['v'].values))
                    else:
                        dim1_size = shape[3]
                        picks1 = np.sort(rng.choice(dim1_size, size=min(3, dim1_size), replace=False))
                        for i in picks0:
                            for j in picks1:
                                ds_ = xr.Dataset({'v': xr.DataArray(var[:, :, i, j], dims=('chain', 'draw'))})
                                rhats.append(float(az.rhat(ds_)['v'].values))
                                eb.append(float(az.ess(ds_, method='bulk')['v'].values))
                    out[name] = dict(
                        rhat_max=float(np.nanmax(rhats)),
                        ess_bulk_min=float(np.nanmin(eb)),
                    )
            except Exception as e:
                out[name] = dict(error=str(e))

        ss = ds.groups['sample_stats']
        diverging = ss.variables['diverging'][:]
        accept = ss.variables['mean_tree_accept'][:] if 'mean_tree_accept' in ss.variables else None
        depth = ss.variables['depth'][:] if 'depth' in ss.variables else None
        n_chains, n_draws = diverging.shape
        out['__sample_stats__'] = dict(
            n_chains=int(n_chains), n_draws=int(n_draws),
            n_divergences=int(np.sum(diverging)),
            mean_tree_accept=float(np.mean(accept)) if accept is not None else None,
            max_depth_seen=int(np.max(depth)) if depth is not None else None,
        )
    return out


def compare(old_path, new_path):
    old = extract_diagnostics(old_path)
    new = extract_diagnostics(new_path)
    old_ss = old.pop('__sample_stats__')
    new_ss = new.pop('__sample_stats__')
    old_total = old_ss['n_chains'] * old_ss['n_draws']
    new_total = new_ss['n_chains'] * new_ss['n_draws']

    print(f"ground truth: {old_path}  ({old_ss['n_chains']} chains x {old_ss['n_draws']} draws)")
    print(f"new fit:      {new_path}  ({new_ss['n_chains']} chains x {new_ss['n_draws']} draws)")
    print()
    print(f"{'variable':<20} {'gt rhat':>8} {'new rhat':>9} {'gt ess%':>9} {'new ess%':>9}")
    worst_new_rhat, min_new_eff = 0.0, 100.0
    for name in old:
        if name not in new or 'error' in old[name] or 'error' in new[name]:
            continue
        g, n = old[name], new[name]
        gt_eff = g['ess_bulk_min'] / old_total * 100
        new_eff = n['ess_bulk_min'] / new_total * 100
        worst_new_rhat = max(worst_new_rhat, n['rhat_max'])
        min_new_eff = min(min_new_eff, new_eff)
        print(f"{name:<20} {g['rhat_max']:>8.4f} {n['rhat_max']:>9.4f} {gt_eff:>8.1f}% {new_eff:>8.1f}%")

    print()
    print(f"ground truth: divergences={old_ss['n_divergences']}  "
          f"mean_tree_accept={old_ss['mean_tree_accept']:.4f}  max_depth={old_ss['max_depth_seen']}")
    print(f"new fit:      divergences={new_ss['n_divergences']}  "
          f"mean_tree_accept={new_ss['mean_tree_accept']:.4f}  max_depth={new_ss['max_depth_seen']}")
    print()

    # Divergences are judged relative to the ground truth's own rate, not
    # against an absolute zero -- a model with LKJ/HSGP funnel geometry can
    # legitimately show a handful of divergences even in a known-good fit
    # (this repo's own ff_hs.nc/ff_team.nc ground truth has 1/4000 each).
    # Flag only a rate that's both meaningfully worse in absolute terms
    # (>0.5% of draws) AND several times the ground truth's own rate.
    old_pct = old_ss['n_divergences'] / old_total * 100
    new_pct = new_ss['n_divergences'] / new_total * 100
    divergence_regression = new_pct > 0.5 and new_pct > 5 * max(old_pct, 0.025)

    healthy = worst_new_rhat < 1.1 and not divergence_regression and min_new_eff > 1.0
    verdict = "HEALTHY" if healthy else "CHECK NEEDED"
    print(f"verdict: {verdict}  (worst new rhat={worst_new_rhat:.4f}, "
          f"min ess eff={min_new_eff:.1f}%, "
          f"divergence rate: gt={old_pct:.3f}% new={new_pct:.3f}%)")
    return healthy


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    ok = compare(sys.argv[1], sys.argv[2])
    sys.exit(0 if ok else 1)
