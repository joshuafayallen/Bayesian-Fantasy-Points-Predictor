"""
Structural diff between a freshly-fit idata .nc file and one of the
"ground truth" fits in model-nc/ (ff_ar.nc / ff_hs.nc / ff_team.nc --
produced by the pre-refactor src/ff-ar.py, before src/models.py existed).

This does NOT compare posterior values (MCMC posteriors are stochastic
even with a fixed seed across different hardware/BLAS backends -- a
value-level diff would need a full, deterministic refit to mean anything,
and these files are 10-16GB, expensive to refit just to check). Instead it
compares STRUCTURE: does the new file's `posterior` group contain exactly
the same set of variables, with the same shapes, as the ground truth? That
catches exactly the class of regression a refactor like the models.py
extraction risks -- a renamed variable, a dropped Deterministic (this is
literally how the positions_mu/age_slope gap in the old hand-ported
backtest_harness.py would have shown up), a reordered/wrong-shape dim --
without needing to move or fully load either file.

Uses the netCDF4 package directly (NOT arviz.from_netcdf, which loads
full arrays into memory) so this stays fast and low-memory even on a
10+GB file -- opening a Dataset and listing a group's variable
names/shapes never touches the array data itself. netCDF4 is already a
project dependency (see pyproject.toml), so this needs nothing extra.

Usage:
    python tests/compare_ground_truth_netcdf.py model-nc/ff_ar.nc model-nc/ff_ar_new.nc
    python tests/compare_ground_truth_netcdf.py model-nc/ff_hs.nc model-nc/ff_hs_new.nc
    python tests/compare_ground_truth_netcdf.py model-nc/ff_team.nc model-nc/ff_team_new.nc

Exits non-zero (and prints what differs) if the two files' `posterior`
groups don't match structurally.
"""
import sys

import netCDF4


def posterior_signature(path):
    """{var_name: shape} for every variable in `path`'s posterior group."""
    with netCDF4.Dataset(path, 'r') as ds:
        if 'posterior' not in ds.groups:
            raise ValueError(f"{path} has no 'posterior' group -- is this a real idata .nc file?")
        post = ds.groups['posterior']
        return {name: tuple(var.shape) for name, var in post.variables.items()}


def compare(old_path, new_path):
    old_sig = posterior_signature(old_path)
    new_sig = posterior_signature(new_path)

    old_names, new_names = set(old_sig), set(new_sig)
    missing = sorted(old_names - new_names)      # in ground truth, not in new fit
    extra = sorted(new_names - old_names)         # in new fit, not in ground truth
    shape_mismatch = sorted(
        n for n in (old_names & new_names) if old_sig[n] != new_sig[n]
    )

    ok = not (missing or extra or shape_mismatch)

    print(f"ground truth: {old_path} ({len(old_names)} posterior variables)")
    print(f"new fit:      {new_path} ({len(new_names)} posterior variables)")
    print()
    if missing:
        print(f"MISSING from new fit ({len(missing)}):")
        for n in missing:
            print(f"  - {n}  shape={old_sig[n]}")
    if extra:
        print(f"EXTRA in new fit, not in ground truth ({len(extra)}):")
        for n in extra:
            print(f"  + {n}  shape={new_sig[n]}")
    if shape_mismatch:
        print(f"SHAPE MISMATCH ({len(shape_mismatch)}):")
        for n in shape_mismatch:
            print(f"  ~ {n}  ground_truth={old_sig[n]}  new={new_sig[n]}")
    if ok:
        print(f"MATCH -- all {len(old_names)} posterior variables present with identical shapes.")

    return ok


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    ok = compare(sys.argv[1], sys.argv[2])
    sys.exit(0 if ok else 1)
