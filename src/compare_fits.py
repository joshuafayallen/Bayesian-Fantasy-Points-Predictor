
import sys

import arviz as az
import numpy as np

check = az.from_netcdf(PATHS[0])

check.groups

PATH = 'model-nc/'
MODS = ['ff_ar_plain.nc', 'ff_ar_shrunk.nc', 'ff_hs_plain.nc', 'ff_hs_shrunk.nc', 'ff_team_plain.nc', 'ff_team_shrunk.nc', 'ff_r2d2_plain.nc', 'ff_r2d2_shrunk.nc']

PATHS = [f"{PATH}{mods}" for mods in MODS]


def label_for(idata, path):
    model = idata.attrs.get('model')
    shrunk = idata.attrs.get('shrunk')
    if model is not None and shrunk is not None:
        return f"{model}_{'shrunk' if shrunk else 'plain'}"
    return path.rsplit('/', 1)[-1].removesuffix('.nc')


def load_all(paths):
    idatas = {}
    for path in paths:
        idata = az.from_netcdf(path)
        if '/log_likelihood' not in idata.groups:
            print(f"skipping {path}: no log_likelihood group -- was this fit with "
                  f"pm.compute_log_likelihood()? (src/fit_model.py's fit_and_save does this "
                  f"automatically, so this most likely means an older/hand-saved .nc)")
            continue
        label = label_for(idata, path)
        if label in idatas:
            print(f"warning: {path} and a previous file both map to label '{label}' -- "
                  f"the later one wins. Pass explicit --out paths to src/fit_model.py to avoid this.")
        idatas[label] = idata
    return idatas


def check_log_likelihood_health(idatas):
    """Same nan/inf sanity check ff_ar.py ran by hand before az.compare() --
    a single bad observation in one model's log_likelihood silently breaks
    (or badly skews) the whole comparison."""
    any_bad = False
    for name, idata in idatas.items():
        ll = idata.log_likelihood['y_obs'].values
        n_nan = np.isnan(ll).sum()
        n_inf = np.isinf(ll).sum()
        if n_nan or n_inf:
            any_bad = True
            bad_obs = np.where(np.isnan(ll).any(axis=(0, 1)) | np.isinf(ll).any(axis=(0, 1)))[0]
            print(f"  WARNING {name}: shape={ll.shape} nan={n_nan} inf={n_inf} "
                  f"-- {len(bad_obs)} distinct observations affected, e.g. indices: {bad_obs[:10]}")
    return not any_bad


def compare_models(paths):
    if len(paths) < 2:
        print("need at least 2 .nc files to compare -- got:", paths)
        sys.exit(1)

    idatas = load_all(paths)
    if len(idatas) < 2:
        print(f"only {len(idatas)} usable idata(s) after filtering -- nothing to compare.")
        sys.exit(1)

    print("=" * 78)
    print(f"loaded {len(idatas)} fits: {list(idatas)}")
    print("=" * 78)

    print("\nlog_likelihood health check:")
    ok = check_log_likelihood_health(idatas)
    if not ok:
        print("  -- nan/inf present in at least one model's log_likelihood; "
              "az.compare() below may be unreliable for that model.")

    print("\n" + "=" * 78)
    print("az.compare()  (rank 0 = best; weight = LOO-stacking weight)")
    print("=" * 78)
    result = az.compare(idatas)
    print(result)

    best = result.index[0]
    print(f"\nBEST (by elpd_loo): {best}")
    print("Next step: point src/run_backtest.sh at this model (and its shrunk/plain "
          "feature choice, if src/backtest_harness.py has been extended to support --shrunk) "
          "for a genuinely held-out walk-forward confirmation before trusting this ranking.")
    return result

compare_models(PATHS)
