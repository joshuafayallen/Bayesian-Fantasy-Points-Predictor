"""Same as gt_diag2.py but only processes the variable names given on argv[2:] (comma-separated in one arg, or multiple args)."""
import sys
import json
import numpy as np
import netCDF4
import arviz as az
import xarray as xr

SEED = 41029041
rng = np.random.default_rng(SEED)
N_SPOT = 8
SMALL_THRESHOLD = 64


def process_one(var, name):
    shape = var.shape
    extra_shape = shape[2:]
    extra_size = int(np.prod(extra_shape)) if extra_shape else 1
    if extra_size <= SMALL_THRESHOLD:
        data = var[:]
        da = xr.DataArray(data, dims=var.dimensions)
        ds_ = xr.Dataset({name: da})
        rhat = az.rhat(ds_)[name].values
        ess_bulk = az.ess(ds_, method='bulk')[name].values
        ess_tail = az.ess(ds_, method='tail')[name].values
        return dict(kind='full', shape=list(shape),
                    rhat_max=float(np.nanmax(rhat)), rhat_min=float(np.nanmin(rhat)),
                    ess_bulk_min=float(np.nanmin(ess_bulk)), ess_tail_min=float(np.nanmin(ess_tail)))
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
        return dict(kind='spotcheck', shape=list(shape), n_spotchecked=n_checked,
                    rhat_max=float(np.nanmax(rhats)), rhat_min=float(np.nanmin(rhats)),
                    ess_bulk_min=float(np.nanmin(eb)), ess_bulk_median=float(np.nanmedian(eb)),
                    ess_tail_min=float(np.nanmin(et)))


def main(path, names):
    out = {}
    with netCDF4.Dataset(path, 'r') as ds:
        post = ds.groups['posterior']
        for name in names:
            if name not in post.variables:
                print(f"skip missing: {name}", file=sys.stderr)
                continue
            try:
                out[name] = process_one(post.variables[name], name)
                print(f"done: {name}", file=sys.stderr)
            except Exception as e:
                out[name] = dict(error=str(e))
                print(f"ERROR {name}: {e}", file=sys.stderr)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    path = sys.argv[1]
    names = sys.argv[2].split(',')
    main(path, names)
