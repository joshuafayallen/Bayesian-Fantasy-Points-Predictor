"""
Prior-sensitivity sandbox for ar_mod's observation-noise prior
(`sigma_obs`) -- WITHOUT touching src/models.py.

Why this is possible without a refactor: build_ar_mod hardcodes
    sigma_obs = pm.HalfNormal('sigma_obs', 5.0, dims='position')
as a local variable, then passes it straight into
add_role_mixture(mu, sigma_obs, ...) -- sigma_obs is just a PyMC
random variable by the time add_role_mixture sees it, not a prior spec
baked into that function. So a different prior for sigma_obs can be
built here and passed to the SAME add_role_mixture, inside a model that
reuses every other shared component from models.py unchanged
(base_data, add_flat_position_intercept, add_player_skill,
add_strength_ar, add_recent_form_ar -- all imported, not
re-implemented).

This intentionally re-assembles the ~10-line mu = intercepts + ...
block from build_ar_mod's body here (that part isn't its own function
in models.py, so there's no cleaner hook to swap sigma_obs's prior
without duplicating those lines). Once a variant looks like a clear
win, that's the signal to parameterize build_ar_mod/build_hs_version/
build_team_mod's sigma_obs argument for real -- this script is
deliberately the fast/dirty way to find out first, kept separate from
models.py so exploratory priors never risk the ground-truth builders.

Built for hands-on REPL use, not as a run-and-done script: run it with
`ipython -i src/prior_checks.py` (or `python -i ...`). It does the data
prep ONCE (the only slow, boring part) and then drops you into the REPL
with `dat` (a PriorCheckData bundle) and `y_real` already sitting there
-- nothing else runs automatically, no argparse, no auto-looping over
variants. From the prompt:

    >>> sigma_obs_tight = lambda: pm.HalfNormal('sigma_obs', 2.0, dims='position')
    >>> m = build_ar_variant(dat, sigma_obs_builder=sigma_obs_tight)
    >>> with m:
    ...     idata = pm.sample_prior_predictive(samples=200, random_seed=SEED)
    >>> prior_pred_summary(idata, y_real)          # quick percentile check
    >>> plot_prior_pred(idata)                      # az.plot_ppc_dist, same as ff_ar.py

    >>> with m:
    ...     idata = pm.sample(draws=300, tune=300, target_accept=0.99, random_seed=SEED)
    >>> sample_summary(idata)                       # sigma_obs mean / divergences / step size / ESS%
    >>> az.plot_trace(idata, var_names=['sigma_obs'])   # or anything else from arviz directly

You can also build `dat` yourself (`dat = prep_data()`) if you're
iterating on this script line by line instead of running it as a whole
-- prep_data() is safe to call repeatedly, just slow (rereads the
parquet files each time).

SIGMA_OBS_VARIANTS below is just a few starting points to grab from or
extend -- nothing iterates over it automatically; loop over it yourself
if/when you want a batch comparison (or use tests/... scripts for that
kind of end-to-end run). As of the latest round of checks, HalfNormal
in the 0.75-0.9 range gave noticeably saner prior-predictive ranges
than the original HalfNormal(5.0) or the earlier "tighter" 1.0 guess --
'sane range: HalfNormal(0.9)'/'HalfNormal(0.75)' below capture that.

SHRINKAGE FEATURES: src/estimate_shrinkage.py computes empirical-Bayes
shrunk versions of lag_yards_per_rush_attempt / lag_yards_per_target /
lag_yards_per_pass_attempt (small-sample-noise fix for the extreme
z-scores found during prior checking -- see that script's docstring
for the method) and writes them to
processed-data/shrinkage_features.csv. Run it once:

    python src/estimate_shrinkage.py

then pull the shrunk features straight into this sandbox instead of
ff-processed.parquet's originals, with zero changes to
data_cleaning.py:

    >>> dat = prep_data(shrinkage_csv='processed-data/shrinkage_features.csv')

Everything downstream (scaling, cont_vars, mu_component_ranges, prior
predictive) is unchanged -- the CSV's shrunk lag_* columns just replace
the originals in `train` before scalers are fit, so they flow through
the exact same z-scoring path as before.

Usage:
    ipython -i src/prior_checks.py
"""
import sys
from dataclasses import dataclass

sys.path.insert(0, 'src')

import numpy as np
import polars as pl
import polars.selectors as cs
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import models


@dataclass
class PriorCheckData:
    """Everything build_ar_variant() needs, bundled so you can pass one
    object around in the REPL instead of unpacking a 7-tuple every time."""
    D: dict
    coords: dict
    season_offsets: object
    T: int
    boundary_cols: object
    position_of_player: object
    position_group_counts: object

SEED = 41029041
DATA_PATH = 'processed-data/ff-processed.parquet'
LAST_SEASON = 2025

# Each entry: (label, zero-arg callable that builds sigma_obs inside an
# open pm.Model context). All dims='position' to match build_ar_mod.
SIGMA_OBS_VARIANTS = {
    'current: HalfNormal(5.0)': lambda: pm.HalfNormal('sigma_obs', 5.0, dims='position'),
    'tight: HalfNormal(2.0)': lambda: pm.HalfNormal('sigma_obs', 2.0, dims='position'),
    'tighter: HalfNormal(1.0)': lambda: pm.HalfNormal('sigma_obs', 1.0, dims='position'),
    'sane range: HalfNormal(0.9)': lambda: pm.HalfNormal('sigma_obs', 0.9, dims='position'),
    'sane range: HalfNormal(0.75)': lambda: pm.HalfNormal('sigma_obs', 0.75, dims='position'),
    'sane range, heavy tail: HalfStudentT(nu=4, sigma=0.75)': lambda: pm.HalfStudentT(
        'sigma_obs', nu=4, sigma=0.75, dims='position'),
}


def _apply_shrinkage_features(train: pl.DataFrame, shrinkage_csv: str) -> pl.DataFrame:
    """Swaps lag_yards_per_rush_attempt / lag_yards_per_target /
    lag_yards_per_pass_attempt in `train` for the empirical-Bayes-shrunk
    versions from src/estimate_shrinkage.py's output CSV, joined on
    (player_id, season, week). Rows the CSV doesn't cover (shouldn't
    normally happen -- it's built from the same source data) fall back
    to the original column via coalesce rather than going null.

    Runs BEFORE scalers are fit, so the shrunk values get z-scored by
    the same fit_scalers/apply_scalers call as everything else --
    nothing downstream needs to know these came from a different
    source."""
    shrunk = pl.read_csv(shrinkage_csv).with_columns(
        pl.col('season').cast(pl.Int32),
        pl.col('week').cast(pl.Float64),
    )
    train = (
        train.join(
            shrunk.select(
                'player_id', 'season', 'week',
                'lag_yards_per_rush_attempt_shrunk',
                'lag_yards_per_target_shrunk',
                'lag_yards_per_pass_attempt_shrunk',
            ),
            on=['player_id', 'season', 'week'], how='left',
        )
        .with_columns(
            pl.coalesce('lag_yards_per_rush_attempt_shrunk', 'lag_yards_per_rush_attempt')
              .alias('lag_yards_per_rush_attempt'),
            pl.coalesce('lag_yards_per_target_shrunk', 'lag_yards_per_target')
              .alias('lag_yards_per_target'),
            pl.coalesce('lag_yards_per_pass_attempt_shrunk', 'lag_yards_per_pass_attempt')
              .alias('lag_yards_per_pass_attempt'),
        )
        .drop('lag_yards_per_rush_attempt_shrunk', 'lag_yards_per_target_shrunk',
              'lag_yards_per_pass_attempt_shrunk')
    )
    return train


def prep_data(shrinkage_csv=None):
    """Same data prep as ff_ar.py / tests/refit_all_and_compare.py --
    duplicated here rather than imported since it isn't shared plumbing
    (see models.py's module docstring on why data-prep stays local to
    each entry point). Returns a PriorCheckData bundle -- pass it whole
    to build_ar_variant(dat, sigma_obs_builder=...).

    shrinkage_csv: optional path to src/estimate_shrinkage.py's output
    (processed-data/shrinkage_features.csv). When given, the three
    attempt-based lag rate features are swapped for their
    empirical-Bayes-shrunk versions before scaling -- see module
    docstring's SHRINKAGE FEATURES section."""
    raw_data = pl.read_parquet(DATA_PATH)
    team_form = pl.read_parquet('processed-data/team_form.parquet')
    add_team_form = raw_data.select(pl.exclude("^.*_right$")).join(
        team_form, on=['season', 'week', 'team'], how='left')
    add_team_form = add_team_form.with_columns(pl.col('team_form').fill_null(0.0))
    opp_team_form = team_form.rename({'team': 'opp_team', 'team_form': 'opp_team_form'})
    add_opp_form = add_team_form.join(opp_team_form, on=['season', 'week', 'opp_team'], how='left')
    add_opp_form = add_opp_form.with_columns(pl.col('opp_team_form').fill_null(0.0))

    lw = add_opp_form.filter(pl.col('season') == LAST_SEASON)['week'].max()
    train = add_opp_form.filter(
        (pl.col('season') < LAST_SEASON) | ((pl.col('season') == LAST_SEASON) & (pl.col('week') < lw))
    )

    if shrinkage_csv is not None:
        train = _apply_shrinkage_features(train, shrinkage_csv)

    std_these = ['spread_line', 'roll_avg_points_allowed', 'lag_yards_per_rush_attempt',
                 'lag_target_share', 'lag_yards_per_target', 'lag_yards_per_pass_attempt',
                 'temp', 'wind', 'games_missed_ytd', 'lag_offense_pct', 'lag_st_pct']
    binary_vars = ['is_grass', 'is_indoors', 'is_home', 'div_game', 'era']
    idx_cols = ['player', 'position', 'team', 'week', 'tenure', 'player_season',
                'team_season', 'opp_team', 'season', 'games_missed_ytd']

    scalers = models.fit_scalers(df=train, cols=std_these)
    idxs = models.make_indices(raw_data, idx_cols)
    train = models.apply_scalers(train, scalers)
    train = models.encode(train, cols=idx_cols, enums=idxs)

    coords = {col: enum.categories.to_list() for col, enum in idxs.items()}
    cont_dat_train = train.select(cs.ends_with('_z'))
    coords['cont_vars'] = cont_dat_train.columns
    binary_train = train.select(binary_vars)
    coords['binary_vars'] = binary_train.columns

    weeks_per_season = (
        raw_data.group_by('season').agg(pl.col('week').max().alias('max_week'))
        .sort('season')['max_week'].cast(pl.Int64).to_numpy()
    )
    season_offsets, T, boundary_cols = models.build_ar_calendar(weeks_per_season)

    n_positions = len(coords['position'])
    position_of_player, position_group_counts = models.player_position_map(
        raw_data, 'player', 'position', idxs, n_positions)

    tenure_grid_vals = np.array(coords['tenure'], dtype='float64')
    tenure_z_row_vals = (
        train['tenure_idx'].to_numpy().astype('float64') - tenure_grid_vals.mean()
    ) / tenure_grid_vals.std()

    D = dict(
        player=train['player_idx'].to_numpy().squeeze(),
        position=train['position_idx'].to_numpy().squeeze(),
        team=train['team_idx'].to_numpy().squeeze(),
        tenure=train['tenure_idx'].to_numpy().squeeze(),
        opp_team=train['opp_team_idx'].to_numpy().squeeze(),
        season=train['season_idx'].to_numpy().squeeze(),
        week=train['week_idx'].to_numpy().squeeze(),
        player_season=train['player_season_idx'].to_numpy().squeeze(),
        cont=cont_dat_train.to_numpy(),
        bi=binary_train.to_numpy(),
        y=train['total_fantasy_points'].to_numpy(),
        tenure_z=tenure_z_row_vals,
        games_missed_z=train['games_missed_ytd_z'].to_numpy(),
        snap_pct_z=train['lag_offense_pct_z'].to_numpy(),
        # team_form_z/opp_team_form_z intentionally omitted -- base_data()
        # doesn't read them and ar_mod's mu never uses them.
    )
    return PriorCheckData(D, coords, season_offsets, T, boundary_cols,
                           position_of_player, position_group_counts)


def build_ar_variant(dat, sigma_obs_builder):
    """Reassembles build_ar_mod's mu from models.py's shared components
    exactly as build_ar_mod does, but with a swappable sigma_obs prior
    (`sigma_obs_builder`). `dat` is a PriorCheckData bundle, normally
    from prep_data(). See module docstring for why this doesn't require
    editing models.py."""
    D, coords, season_offsets, T, boundary_cols, position_of_player, position_group_counts = (
        dat.D, dat.coords, dat.season_offsets, dat.T, dat.boundary_cols,
        dat.position_of_player, dat.position_group_counts,
    )
    n_positions = len(coords['position'])
    n_seasons = len(coords['season'])
    coords = models._extend_coords(coords, T, n_seasons)

    with pm.Model(coords=coords) as model:
        row = models.base_data(D, season_offsets)
        intercepts = models.add_flat_position_intercept(row['position_data'])
        cont_priors = pm.Normal('cont_priors', 0, 1, dims='cont_vars')
        bi_priors = pm.Normal('bi_priors', 0, 1.5, dims='binary_vars')

        player_skill = models.add_player_skill(coords, position_of_player, position_group_counts, n_positions)
        f_career = player_skill[row['player_data'], row['tenure_data']]
        f_opp = models.add_strength_ar('opp', 'opp_team', row['opp_team_data'], row['global_time_data'], T, boundary_cols)
        f_form = models.add_recent_form_ar(row['player_season_data'])

        mu = intercepts + pm.math.dot(row['cont_data'], cont_priors) + pm.math.dot(row['bi_data'], bi_priors) \
            + f_career + f_opp + f_form

        sigma_obs = sigma_obs_builder()  # <-- the only line that differs from build_ar_mod
        models.add_role_mixture(mu, sigma_obs, row['position_data'],
                                 row['games_missed_data'], row['snap_pct_data'], row['obs_data'])
    return model


# Same idea as SIGMA_OBS_VARIANTS but for the "limited" mixture
# component's own priors -- mu_limited/sigma_limited are hardcoded
# INSIDE add_role_mixture (unlike sigma_obs, which build_ar_mod passes
# in as an already-built variable), so testing alternates means
# duplicating add_role_mixture's body locally (_role_mixture_variant
# below) rather than editing models.py, same spirit as sigma_obs.
MU_LIMITED_VARIANTS = {
    'current: HalfNormal(1.5)': lambda: pm.HalfNormal('mu_limited', 1.5, dims='position'),
    'tight: HalfNormal(0.75)': lambda: pm.HalfNormal('mu_limited', 0.75, dims='position'),
}
SIGMA_LIMITED_VARIANTS = {
    'current: HalfNormal(1.5)': lambda: pm.HalfNormal('sigma_limited', 1.5, dims='position'),
    'tight: HalfNormal(0.75)': lambda: pm.HalfNormal('sigma_limited', 0.75, dims='position'),
}


def _role_mixture_variant(mu_full, sigma_full, position_data, games_missed_data, snap_pct_data,
                           obs_data, mu_limited_builder, sigma_limited_builder):
    """Duplicates add_role_mixture's body (models.py) exactly, except
    mu_limited/sigma_limited come from swappable builders instead of
    being hardcoded HalfNormal(1.5). See MU_LIMITED_VARIANTS/
    SIGMA_LIMITED_VARIANTS above -- default to the SAME priors
    add_role_mixture uses, so passing the 'current' entries reproduces
    add_role_mixture exactly."""
    mu_limited = mu_limited_builder()
    sigma_limited = sigma_limited_builder()

    role_intercept = pm.Normal('role_intercept', 0, 1.5, dims='position')
    role_beta_missed = pm.Normal('role_beta_missed', 0, 1, dims='position')
    role_beta_snap = pm.Normal('role_beta_snap', 0, 1, dims='position')

    logit_p_full = (
        role_intercept[position_data]
        + role_beta_missed[position_data] * games_missed_data
        + role_beta_snap[position_data] * snap_pct_data
    )
    p_full = pm.math.sigmoid(logit_p_full)
    w = pt.stack([1 - p_full, p_full], axis=-1)

    comp_limited = pm.StudentT.dist(nu=4, mu=mu_limited[position_data], sigma=sigma_limited[position_data])
    comp_full = pm.StudentT.dist(nu=4, mu=mu_full, sigma=sigma_full[position_data])

    return pm.Mixture('y_obs', w=w, comp_dists=[comp_limited, comp_full], observed=obs_data)


def build_ar_variant_debug(dat, sigma_obs_builder,
                            mu_limited_builder=None, sigma_limited_builder=None):
    """Same model as build_ar_variant, but every additive term in mu is
    also wrapped in its own pm.Deterministic -- 'mu_intercepts',
    'mu_cont', 'mu_bi', 'mu_f_career' (player_skill), 'mu_f_opp'
    (opponent-strength AR), 'mu_f_form' (recent-form), plus 'mu' itself
    -- and mu_limited/sigma_limited (the OTHER mixture component's
    priors, see _role_mixture_variant) are swappable too, not just
    sigma_obs. Defaults to add_role_mixture's own priors
    (MU_LIMITED_VARIANTS['current: HalfNormal(1.5)'] etc.) if you only
    want to vary sigma_obs, matching build_ar_variant's behavior.

    Use this when a prior-predictive check on y_obs looks implausibly
    wide (e.g. ranging into the hundreds) and you need to find WHICH
    term is actually responsible, rather than guessing -- including
    whether it's the 'full' component (sigma_obs) or the 'limited'
    component (mu_limited/sigma_limited) driving an extreme min/max.
    Costs a bit more memory in the idata (one extra array per row per
    term) -- fine for prior-predictive-only use, not meant for real
    posterior sampling."""
    if mu_limited_builder is None:
        mu_limited_builder = MU_LIMITED_VARIANTS['current: HalfNormal(1.5)']
    if sigma_limited_builder is None:
        sigma_limited_builder = SIGMA_LIMITED_VARIANTS['current: HalfNormal(1.5)']

    D, coords, season_offsets, T, boundary_cols, position_of_player, position_group_counts = (
        dat.D, dat.coords, dat.season_offsets, dat.T, dat.boundary_cols,
        dat.position_of_player, dat.position_group_counts,
    )
    n_positions = len(coords['position'])
    n_seasons = len(coords['season'])
    coords = models._extend_coords(coords, T, n_seasons)

    with pm.Model(coords=coords) as model:
        row = models.base_data(D, season_offsets)
        intercepts = models.add_flat_position_intercept(row['position_data'])
        cont_priors = pm.Normal('cont_priors', 0, 1, dims='cont_vars')
        bi_priors = pm.Normal('bi_priors', 0, 1.5, dims='binary_vars')

        player_skill = models.add_player_skill(coords, position_of_player, position_group_counts, n_positions)
        f_career = player_skill[row['player_data'], row['tenure_data']]
        f_opp = models.add_strength_ar('opp', 'opp_team', row['opp_team_data'], row['global_time_data'], T, boundary_cols)
        f_form = models.add_recent_form_ar(row['player_season_data'])

        mu_cont = pm.math.dot(row['cont_data'], cont_priors)
        mu_bi = pm.math.dot(row['bi_data'], bi_priors)
        pm.Deterministic('mu_intercepts', intercepts)
        pm.Deterministic('mu_cont', mu_cont)
        pm.Deterministic('mu_bi', mu_bi)
        pm.Deterministic('mu_f_career', f_career)
        pm.Deterministic('mu_f_opp', f_opp)
        pm.Deterministic('mu_f_form', f_form)

        mu = intercepts + mu_cont + mu_bi + f_career + f_opp + f_form
        pm.Deterministic('mu_total', mu)

        sigma_obs = sigma_obs_builder()
        _role_mixture_variant(mu, sigma_obs, row['position_data'],
                               row['games_missed_data'], row['snap_pct_data'], row['obs_data'],
                               mu_limited_builder, sigma_limited_builder)
    return model


def mu_component_ranges(idata):
    """Print min/1%/50%/99%/max for each mu_* Deterministic in an idata
    from build_ar_variant_debug() (mu_intercepts, mu_cont, mu_bi,
    mu_f_career, mu_f_opp, mu_f_form, mu_total), PLUS the observation
    layer's own likelihood parameters -- sigma_obs, mu_limited,
    sigma_limited -- and y_obs itself. The min/max are what matter here
    (a component with implausible min/max, not just a wide but sane
    1-99% band, is the one blowing up the prior predictive range).

    IMPORTANT: mu_total staying sane while y_obs's min/max stay wide
    does NOT mean sigma_obs is the culprit -- add_role_mixture's
    observation layer is a TWO-component mixture. The 'full' component
    uses sigma_obs (mu-driven, what you've been tightening); the
    'limited' component is entirely separate and uses its own
    mu_limited/sigma_limited (both HalfNormal(1.5) in models.py,
    untouched by any sigma_obs experiment). Check sigma_limited/
    mu_limited's own ranges below before concluding sigma_obs is (or
    isn't) responsible for an extreme min/max.

    Deterministics/free RVs live in idata.prior, NOT
    idata.prior_predictive -- only the observed node (y_obs) ends up in
    prior_predictive. This pulls each from the right group
    automatically. Call after pm.sample_prior_predictive()."""
    mu_names = [n for n in idata.prior.data_vars if n.startswith('mu_')]
    likelihood_names = [n for n in ('sigma_obs', 'mu_limited', 'sigma_limited') if n in idata.prior.data_vars]
    print(f"{'component':<16} {'min':>10} {'1%':>10} {'50%':>10} {'99%':>10} {'max':>10}")
    for name in mu_names + likelihood_names:
        v = idata.prior[name].values.reshape(-1)
        print(f"{name:<16} {v.min():>10.1f} {np.percentile(v,1):>10.1f} {np.percentile(v,50):>10.1f} "
              f"{np.percentile(v,99):>10.1f} {v.max():>10.1f}")
    if 'y_obs' in idata.prior_predictive.data_vars:
        v = idata.prior_predictive['y_obs'].values.reshape(-1)
        print(f"{'y_obs':<16} {v.min():>10.1f} {np.percentile(v,1):>10.1f} {np.percentile(v,50):>10.1f} "
              f"{np.percentile(v,99):>10.1f} {v.max():>10.1f}")


def plot_prior_pred(idata, **kwargs):
    """plot_ppc_dist(idata, group='prior_predictive', ...) -- same call
    ff_ar.py's own prior-predictive check uses, so the visual read is
    consistent with what you're used to there. Requires observed_data in
    idata, which pm.sample_prior_predictive() fills in automatically from
    add_role_mixture's `observed=obs_data`. Pass kind='ecdf' or any other
    plot_ppc_dist kwarg through.

    plot_ppc_dist lives on arviz in some installs and only on the newer
    arviz_plots package in others (this repo depends on both -- see
    pyproject.toml) -- try az.plot_ppc_dist first (matches ff_ar.py
    exactly) and fall back to arviz_plots.plot_ppc_dist so this doesn't
    break either way."""
    kwargs.setdefault('visuals', {'observed_dist': True})
    if hasattr(az, 'plot_ppc_dist'):
        return az.plot_ppc_dist(idata, group='prior_predictive', **kwargs)
    import arviz_plots
    return arviz_plots.plot_ppc_dist(idata, group='prior_predictive', **kwargs)


def prior_pred_summary(idata, y_real=None, var_name='y_obs'):
    """One-liner sanity check on a prior-predictive idata: does the
    implied y_obs spread look plausible against the real
    total_fantasy_points distribution? Call after your own
    pm.sample_prior_predictive() -- doesn't build or sample anything
    itself, just summarizes what you already have."""
    y_pp = idata.prior_predictive[var_name].values.reshape(-1)
    print(f"{'':<20} {'5%':>10} {'50%':>10} {'95%':>10}")
    print(f"{'prior predictive':<20} {np.percentile(y_pp, 5):>10.2f} "
          f"{np.percentile(y_pp, 50):>10.2f} {np.percentile(y_pp, 95):>10.2f}")
    if y_real is not None:
        print(f"{'real data':<20} {np.percentile(y_real, 5):>10.2f} "
              f"{np.percentile(y_real, 50):>10.2f} {np.percentile(y_real, 95):>10.2f}")


def sample_summary(idata, var_name='sigma_obs'):
    """One-liner sanity check on a real NUTS fit: var_name's posterior
    mean, divergences, mean step size, and var_name's own ESS%. Call
    after your own pm.sample() -- doesn't sample anything itself."""
    n_chains, n_draws = idata.sample_stats['diverging'].shape
    n_total = n_chains * n_draws
    mean = float(idata.posterior[var_name].mean())
    n_div = int(idata.sample_stats['diverging'].values.sum())
    step = float(idata.sample_stats['step_size'].values.mean())
    ess = float(az.ess(idata, var_names=[var_name], method='bulk')[var_name].values.min())
    print(f"{var_name} mean={mean:.3f}  divergences={n_div}/{n_total} ({100*n_div/n_total:.2f}%)  "
          f"step_size={step:.4f}  ess%={100*ess/n_total:.1f}%")


if __name__ == '__main__':
    print("prepping data...")
    dat = prep_data()
    y_real = dat.D['y']
    print("dat ready (a PriorCheckData bundle). y_real ready. SIGMA_OBS_VARIANTS has a few starting priors.")
    print("build a model with:")
    print("  m = build_ar_variant(dat, sigma_obs_builder=SIGMA_OBS_VARIANTS['sane range: HalfNormal(0.9)'])")
    print("or write your own zero-arg builder and pass it directly. See module docstring for the full flow.")
    print()
    print("to check the empirical-Bayes shrunk rate features instead of the raw ones:")
    print("  (first: python src/estimate_shrinkage.py)")
    print("  dat_shrunk = prep_data(shrinkage_csv='processed-data/shrinkage_features.csv')")
