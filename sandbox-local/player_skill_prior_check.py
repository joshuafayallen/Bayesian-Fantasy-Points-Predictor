"""
Prior predictive check for the `player_skill` random-walk term shared by
sandbox/dynamic-skill-model.py and backtest_harness.build_ar/build_mlm:

    init_sigma = pm.HalfNormal('init_sigma', 1.0)
    rw_sigma   = pm.HalfNormal('rw_sigma', 0.5)     # backtest_harness (0.35 in
                                                       # dynamic-skill-model.py)
    z_innov    = pm.Normal('z_innov', 0, 1, dims=('player', 'tenure'))
    level0     = init_sigma * z_innov[:, 0]
    steps      = rw_sigma * z_innov[:, 1:]
    player_skill = concat([level0, level0 + cumsum(steps)])

This does NOT touch the likelihood, covariates, or any observed data -- it
only samples from these priors (pm.sample_prior_predictive) to ask: before
seeing a single row of real data, what spread of player_skill values does
this prior structurally allow across players, at each tenure (career-season)
level? That's compared against a real, position-adjusted benchmark computed
directly from processed-data/ff-processed.parquet: the SD and top-vs-
replacement gap of career-mean fantasy points, net of each position's own
mean -- i.e. the actual size of the "skill" gap player_skill is supposed to
be able to represent.

NOTEBOOK-STYLE FILE: cells are marked with `# %%`, recognized by Jupyter
(open directly via jupytext, or `jupyter nbconvert --to notebook` it), VS
Code's Python/Jupyter extension ("Run Cell" / "Run Below"), and Spyder.
Open this file in any of those and run cell-by-cell instead of top-to-
bottom -- e.g. rerun just the "sample priors" cell after editing PRIOR_SETS,
without re-touching the real-data benchmark cell.

Can also still be run as a plain script end-to-end:
    python sandbox/player_skill_prior_check.py
"""

# %% imports + setup
from pathlib import Path

import numpy as np
import polars as pl
import pymc as pm
import matplotlib.pyplot as plt

# __file__ isn't defined in a real .ipynb kernel -- fall back to cwd (assumes
# the notebook/kernel was launched from the project root) so this cell works
# either way.
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_ROOT = Path.cwd()

DATA_PATH = PROJECT_ROOT / "processed-data" / "ff-processed.parquet"
RESULTS = PROJECT_ROOT / "results"

SEED = 41029041
N_PLAYERS = 200          # just needs to be large enough for stable spread stats
TENURE_LEVELS = list(range(16))   # 0..15, covers the bulk of real tenure

# priors actually used in the two production model files, kept side by side
PRIOR_SETS = {
    'current (backtest_harness.build_ar)':       dict(init_sigma_sigma=1.0, rw_sigma_sigma=0.5),
    'current (dynamic-skill-model.py)':          dict(init_sigma_sigma=1.0, rw_sigma_sigma=0.35),
    'wider (proposed)':                          dict(init_sigma_sigma=4.0, rw_sigma_sigma=1.5),
}


# %% function defs
def sample_player_skill(init_sigma_sigma, rw_sigma_sigma, n_players=N_PLAYERS,
                        tenure_levels=TENURE_LEVELS, draws=2000, seed=SEED,
                        return_idata=False):
    """Prior-only sample of the player_skill term. Returns a plain
    (draws, players, tenure) numpy array by default; pass return_idata=True
    to get the full InferenceData back instead (e.g. for az.plot_ppc,
    az.plot_posterior, or any other ArviZ-based interactive inspection)."""
    coords = {'player': np.arange(n_players), 'tenure': tenure_levels}
    with pm.Model(coords=coords) as model:
        init_sigma = pm.HalfNormal('init_sigma', init_sigma_sigma)
        rw_sigma = pm.HalfNormal('rw_sigma', rw_sigma_sigma)
        z_innov = pm.Normal('z_innov', 0, 1, dims=('player', 'tenure'))
        level0 = init_sigma * z_innov[:, 0]
        steps = rw_sigma * z_innov[:, 1:]
        pm.Deterministic(
            'player_skill',
            pm.math.concatenate([level0[:, None], level0[:, None] + pm.math.cumsum(steps, axis=1)], axis=1),
            dims=('player', 'tenure'),
        )
        idata = pm.sample_prior_predictive(draws=draws, random_seed=seed)
    if return_idata:
        return idata
    # shape: (chain=1, draw, player, tenure) -> (draw, player, tenure)
    return idata.prior['player_skill'].values[0]


def real_skill_benchmark(data_path=DATA_PATH, min_games=15):
    """Real, position-adjusted career-mean skill residual from the actual
    data -- what player_skill needs to be ABLE to represent. Returns
    (bench, real_resid): bench is a per-position summary table (sd/max/n),
    real_resid is the raw pooled residuals (for the histogram overlay)."""
    df = pl.read_parquet(data_path)
    per_player = (
        df.group_by(['player_id', 'player', 'position'])
        .agg(pl.col('total_fantasy_points').mean().alias('career_mean'),
             pl.len().alias('n_games'))
        .filter(pl.col('n_games') >= min_games)
    )
    pos_mean = per_player.group_by('position').agg(pl.col('career_mean').mean().alias('pos_mean'))
    resid = per_player.join(pos_mean, on='position').with_columns(
        (pl.col('career_mean') - pl.col('pos_mean')).alias('skill_resid')
    )
    bench = resid.group_by('position').agg(
        pl.col('skill_resid').std().alias('sd'),
        pl.col('skill_resid').max().alias('max'),
        pl.len().alias('n'),
    ).sort('position')
    return bench, resid['skill_resid'].to_numpy()


def print_prior_summary(label, kw, arr, tenures=(0, 2, 5, 10, 15)):
    print(f"--- {label}  (init_sigma~HalfNormal({kw['init_sigma_sigma']}), "
          f"rw_sigma~HalfNormal({kw['rw_sigma_sigma']})) ---")
    for t in tenures:
        slice_t = arr[:, :, t]  # (draws, players)
        sd = slice_t.std()
        p2_5, p97_5 = np.percentile(slice_t, [2.5, 97.5])
        print(f"  tenure={t:>2}: SD across players={sd:5.2f}   "
              f"95% range=[{p2_5:6.2f}, {p97_5:6.2f}]   "
              f"implied top-vs-replacement (~+/-2SD) gap={4*sd:5.1f} pts")


def plot(prior_arrays, real_sd_range, real_resid, tenure_levels=TENURE_LEVELS,
        tenure_dist_check=10, out_path=None, show=True):
    """Two panels: (left) SD of player_skill across players vs. tenure, one
    line per prior, with the real position-adjusted SD range shaded; (right)
    the actual player_skill distribution at tenure=`tenure_dist_check` for
    each prior, overlaid on the real pooled skill-residual histogram.

    In a notebook, just call plot(...) and let inline display show the
    figure -- out_path/show only matter for saving when running headless."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for label, arr in prior_arrays.items():
        sds = [arr[:, :, t].std() for t in tenure_levels]
        ax1.plot(tenure_levels, sds, marker='o', markersize=3, label=label)
    ax1.axhspan(real_sd_range[0], real_sd_range[1], color='green', alpha=0.15,
               label='real position-adjusted SD range')
    ax1.set_xlabel('tenure (career season)')
    ax1.set_ylabel('SD of player_skill across players')
    ax1.set_title('Prior-implied skill spread vs. tenure')
    ax1.legend(fontsize=8)

    bins = np.linspace(-20, 20, 60)
    ax2.hist(real_resid, bins=bins, density=True, alpha=0.35, color='green',
            label='real skill residual (data)')
    for label, arr in prior_arrays.items():
        ax2.hist(arr[:, :, tenure_dist_check].ravel(), bins=bins, density=True,
                histtype='step', linewidth=1.8, label=f'{label} (prior, tenure={tenure_dist_check})')
    ax2.set_xlabel('points above/below position mean')
    ax2.set_title(f'player_skill @ tenure={tenure_dist_check} vs. real skill residuals')
    ax2.legend(fontsize=8)

    fig.tight_layout()
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=130)
        print(f"saved plot -> {out_path}")
    if show:
        plt.show()
    return fig


# %% real-data benchmark -- run this cell once, reuse bench/real_resid below
bench, real_resid = real_skill_benchmark()
real_sd_range = (bench['sd'].min(), bench['sd'].max())
print(bench)
print(f"\nacross positions: SD ranges {real_sd_range[0]:.1f}-{real_sd_range[1]:.1f}, "
      f"single best-player residual up to +{bench['max'].max():.1f}")

# %% prior predictive sampling -- edit PRIOR_SETS above and rerun this cell to try new values
prior_arrays = {}
for label, kw in PRIOR_SETS.items():
    prior_arrays[label] = sample_player_skill(**kw)
    print_prior_summary(label, kw, prior_arrays[label])
    print()

# %% plot -- rerun after either cell above changes
plot(prior_arrays, real_sd_range, real_resid, out_path=RESULTS / 'player_skill_prior_check.png')

# %% [markdown]
# Read: compare each prior's "implied top-vs-replacement gap" against the
# REAL gap (top-5 vs bottom-quintile career-mean) of 12-17.5 points across
# positions. If the prior's implied gap at a realistic tenure (5-10 years)
# falls well short of that, player_skill is structurally too tight to ever
# learn the real spread, regardless of how much data it sees -- shrinkage
# isn't a fitting artifact here, it's baked into the prior.
