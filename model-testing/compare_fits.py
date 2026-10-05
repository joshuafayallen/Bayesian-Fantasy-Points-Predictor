
import sys
import polars as pl
import arviz as az
import numpy as np
import pymc as pm
import fit_model


PATH = 'model-nc/'
MODS = ['ff_bart_plain.nc', 'ff_bart_shrunk.nc', 'ff_team_plain.nc', 'ff_team_shrunk.nc', 'ff_r2d2_plain.nc', 'ff_r2d2_shrunk.nc']

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
    return result, idatas
 
 
# ============================================================================
# log_prior -- rebuild each idata's matching model and attach log_prior, then
# use it alongside the log_likelihood az.compare() already loaded to check
# whether the data actually informed the fit, not just how the fits rank
# against each other.
# ============================================================================
 
def build_matching_model(model_name, shrunk, prep_cache):
    """Reconstruct the pm.Model that produced one saved idata, using the
    SAME src/fit_model.py plumbing that trained it -- fit_model.prep_data()
    is deterministic given the same raw data/last_season/shrunk, so this
    reproduces the identical model graph (same coords, same dims) the
    posterior draws in the .nc file were sampled from. Same
    reconstruct-the-training-model pattern src/forecast_roster.py's own
    __main__ already uses to load a fitted model back up for prediction.
 
    prep_cache: {shrunk: prep_data(...)result} -- prep_data() re-reads the
    full parquet and refits scalers/enums, which is the expensive part;
    shrunk is the only thing that changes its output, so this avoids
    calling it 6 times (once per .nc file) when 2 calls (shrunk=True/False)
    cover every model here.
 
    ASSUMES every .nc file here was fit with fit_model.py's DEFAULT
    --last-season (fit_model.LAST_SEASON). If any of these were fit with
    an explicit --last-season override, this reconstructs a
    differently-shaped model (different training window -> different
    coords/dims), and pm.compute_log_prior below will fail loudly on a
    dimension mismatch rather than silently produce something wrong --
    that failure is the signal to pass the right last_season here."""
    if shrunk not in prep_cache:
        prep_cache[shrunk] = fit_model.prep_data(shrunk=shrunk)
    prep = prep_cache[shrunk]
    build = fit_model.BUILDERS[model_name]
    return build(prep['D'], prep['coords'], prep['season_offsets'], prep['T'], prep['boundary_cols'],
                prep['position_of_player'], prep['position_group_counts'], prep['week_grid_vals'])
 
 
def attach_log_prior(idatas):
    """Mutates each idata in place, adding a log_prior group (matching
    pm.compute_log_prior's default extend_inferencedata=True) -- skips (with
    a warning) any idata missing the model/shrunk attrs fit_and_save always
    writes, since there's no way to know which architecture to rebuild
    without them."""
    prep_cache = {}
    for name, idata in idatas.items():
        model_name = idata.attrs.get('model')
        shrunk = idata.attrs.get('shrunk')
        if model_name is None or shrunk is None:
            print(f"  skipping log_prior for {name}: missing model/shrunk attrs, "
                  f"can't tell which architecture to rebuild")
            continue
        model = build_matching_model(model_name, bool(shrunk), prep_cache)
        pm.compute_log_prior(idata, model=model)
 
 
def power_scaling_report(idatas, threshold=0.05):
    """Prior power-scaling sensitivity per model."""
    print("\n" + "=" * 78)
    print(f"prior power-scaling sensitivity (az.psense, threshold={threshold})")
    print("=" * 78)

    results = []

    for name, idata in idatas.items():
        if "/log_prior" not in idata.groups:
            continue

        print(f"\n  {name}:")

        res = az.psense_summary(idata).reset_index()

        # Keep track of which model produced these results
        res["model"] = name

        results.append(res)

    if not results:
        return pl.DataFrame()

    return pl.concat(
        [pl.from_pandas(df) for df in results],
        how="vertical_relaxed",
    )
res = compare_models(PATHS)


idatas = res[1]

keep_keys = {'team_plain', 'team_shrunk', 'r2d2_plain', 'r2d2_shrunk'}

idatas_report = {key: idatas[key] for key in keep_keys}

scales = power_scaling_report(idatas_report)

clean_scales = (
    pl.col('index')
)



