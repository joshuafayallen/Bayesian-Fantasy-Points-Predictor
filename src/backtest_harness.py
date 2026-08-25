"""
Walk-forward backtest harness comparing the model architectures defined in
src/models.py (ported from src/ff_ar.py, which remains the ground truth
for what each architecture is) on held-out weekly accuracy:

    hs    hs_version -- LKJ-correlated position intercept + age slope +
                         opp_ar + an HSGP within-season recent-form curve
    ar    ar_mod     -- flat per-position intercept + opp_ar + a flat
                         per-player_season recent-form offset
    team  team_mod   -- hs_version's LKJ intercept + opp_ar (no
                         recent-form term) + two regression coefficients
                         (beta_team_form, beta_opp_team_form) on a
                         precomputed team-strength feature from
                         src/estimate_latent_ability.py's standalone
                         Bradley-Terry-style state-space fit on real game
                         scores.

`team_season_mod` (a season-level AR(1) team-quality version) is wired up
here (build_team_season_mod, via src/models.py) but NOT in the CLI
`choices` list: it is not one of src/ff_ar.py's three currently active
models. See src/models.py's module note above build_team_season_mod
before reviving it.

Every model architecture (the shared blocks AND the three build_* model
functions) now lives in src/models.py, imported here rather than
duplicated -- this file used to carry its own hand-ported copy of all of
it, which had an explicit "no way to detect drift from src/ff_ar.py
automatically" warning in this docstring. That drift risk is gone: this
file and src/ff_ar.py both call the exact same src/models.py functions,
so a change to how a model is built can't silently apply to only one of
them.

What's left HERE, and genuinely fold-specific rather than shared
plumbing:
  - Fold: fits scalers/categorical encoders on one fold's WINDOWED train
    slice (src/ff_ar.py fits these once across all of history, since it
    only ever has one "fold").
  - ar_calendar: this fold's own weeks-per-season footprint (a windowed
    fold's season/week coverage differs fold to fold; src/ff_ar.py derives
    this once from the full raw_data).
  - the walk-forward CLI, fit/predict/score loop, and the stacked-
    ensemble blend (fit_predict_stack) -- none of which src/ff_ar.py needs
    at all.

KNOWN LIMITATION -- team_form leakage via hyperparameters, not
observations: team's team_form/opp_team_form feature is loaded from
processed-data/team_form.parquet, precomputed ONCE by
src/estimate_latent_ability.py across the full 2006-2025 history. Its
filtered_states are genuinely "through week i" at the OBSERVATION level
(no future scores enter a given week's filtered estimate), but the
state-space model's own hyperparameters (sigma_level_trend,
sigma_MeasurementError) were fit on the full-history posterior, which
does use information from every fold's future. This harness does NOT
refit that state-space model per fold -- doing so would be a real
follow-up if team_mod's backtest results look surprisingly good, to rule
out this specific leak before trusting them.

Each fold fits on a WINDOWED slice of history (last WINDOW_SEASONS
seasons, not the full 2006-2025 history) so that player/team/tenure
cardinality -- and therefore per-fold runtime/memory -- stays tractable.
This means absolute error numbers here are not directly comparable to
the full-history models' in-sample numbers; the point is the RANKING
across architectures on genuinely held-out weeks.

Usage (run from the project root, so processed-data/ resolves):
    python src/backtest_harness.py hs advi 2025 12
    python src/backtest_harness.py team advi 2025 12

Each invocation runs exactly ONE fold and appends one row to
results/backtest_walkforward.csv (checkpointed -- safe to run fold-by-fold
from a shell loop, or in parallel processes since each writes its own row).

ADVI is the fast/cheap path for sweeping many folds; NUTS is also wired
up (`method=nuts`) for a smaller, higher-fidelity confirmation set:

    python src/backtest_harness.py team nuts 2025 12 \\
        --nuts-draws 500 --nuts-tune 500 --chains 4

To sweep every fold for every model (e.g. a full season, walking week by
week), just loop the single-fold call -- or use src/run_backtest.sh,
which already does this.

Once a sweep is done, `python src/summarize_backtest.py
results/backtest_walkforward.csv` prints per-model pooled metrics and
picks the winner.
"""
import sys
import json
import time
import argparse
import numpy as np
import polars as pl
import pymc as pm
import arviz as az
from scipy.stats import spearmanr

import models

DATA_PATH = "processed-data/ff-processed.parquet"   # run from the project root
TEAM_FORM_PATH = "processed-data/team_form.parquet"  # from src/estimate_latent_ability.py
RESULTS_CSV = "results/backtest_walkforward.csv"
WINDOW_SEASONS = 3   # how many seasons of history to feed each fold

# games_missed_ytd/lag_offense_pct/lag_st_pct match src/ff_ar.py's
# std_these -- these feed add_role_mixture's p_full logit, used by every
# model.
STD_COLS = ['spread_line', 'roll_avg_points_allowed', 'lag_yards_per_rush_attempt',
            'lag_target_share', 'lag_yards_per_target', 'lag_yards_per_pass_attempt',
            'temp', 'wind', 'games_missed_ytd', 'lag_offense_pct', 'lag_st_pct']
BINARY_VARS = ['is_grass', 'is_indoors', 'is_home', 'div_game', 'era']

# 'player_season' (add_recent_form_ar/add_recent_form_hsgp) and 'week'
# (add_recent_form_hsgp's within-season time axis) match src/ff_ar.py's
# idx_cols -- every current architecture needs these.
IDX_COLS = ['player', 'position', 'team', 'tenure', 'opp_team', 'season', 'week',
            'player_season']


# --------------------------------------------------------------- data prep

class Fold:
    """Encapsulates encoders + coords fit on ONE fold's windowed train
    slice. Produces arrays in exactly the shape src/models.py's build_*
    functions expect as `D` (player/position/team/tenure/opp_team/season/
    week/player_season/cont/bi/y/tenure_z/games_missed_z/snap_pct_z, plus
    team_form_z/opp_team_form_z when available) -- see base_data() and the
    build_* docstrings in src/models.py."""

    def __init__(self, train, idx_cols):
        self.idx_cols = idx_cols
        self.enums = models.make_enums(train, idx_cols)
        self.scalers = models.fit_scalers(train, STD_COLS)
        self.coords = {c: e.categories.to_list() for c, e in self.enums.items()}
        self.coords['cont_vars'] = STD_COLS
        self.coords['binary_vars'] = BINARY_VARS
        # add_recent_form_hsgp's shared basis dim -- m must match the m=
        # passed there (default 6, same as src/models.py/src/ff_ar.py).
        self.coords['form_basis'] = list(range(6))
        if 'tenure' in idx_cols:
            # Same construction as src/ff_ar.py's hs_version: z-score the
            # *tenure index* against the mean/sd of this fold's own tenure
            # category grid.
            tvals = np.array(self.coords['tenure'], dtype='float64')
            self._tenure_mean = tvals.mean()
            self._tenure_std = tvals.std()
        if 'week' in idx_cols:
            # within-season week grid for add_recent_form_hsgp's GP input --
            # same role as week_grid_vals in src/ff_ar.py.
            self.week_grid_vals = np.array(self.coords['week'], dtype='float64')
        # team_form/opp_team_form scaler -- fit fold-locally like every
        # other continuous covariate here (unlike src/ff_ar.py, which fits
        # one scaler across the full 2006-2025 history since it only ever
        # has one "fold"). Only meaningful if the caller already joined
        # team_form/opp_team_form onto `train` (see run_one_fold) --
        # models.build_hs_version/build_ar_mod never read these, only
        # models.build_team_mod does.
        if 'team_form' in train.columns:
            self._team_form_mu = train['team_form'].mean()
            self._team_form_sd = train['team_form'].std() or 1.0

    def arrays(self, frame):
        out = {c: models.idx_array(frame, c, self.enums) for c in self.idx_cols}
        out['cont'] = models.cont_array(frame, self.scalers)
        out['bi'] = models.bi_array(frame, BINARY_VARS)
        out['y'] = frame['total_fantasy_points'].to_numpy()
        if 'tenure' in self.idx_cols:
            out['tenure_z'] = (out['tenure'].astype('float64') - self._tenure_mean) / self._tenure_std
        if hasattr(self, '_team_form_mu') and 'team_form' in frame.columns:
            # same scaler for both -- keeps beta_team_form/beta_opp_team_form
            # directly comparable in magnitude, matching src/ff_ar.py.
            out['team_form_z'] = ((frame['team_form'] - self._team_form_mu) / self._team_form_sd).to_numpy()
            out['opp_team_form_z'] = ((frame['opp_team_form'] - self._team_form_mu) / self._team_form_sd).to_numpy()
        # add_role_mixture's p_full logit inputs -- z-scored the same way
        # as their entries in cont_array (they're ALSO in STD_COLS/cont_vars,
        # matching src/ff_ar.py: games_missed_ytd_z/lag_offense_pct_z feed
        # BOTH cont_priors' general effect on the 'full' component AND
        # role_beta_missed/role_beta_snap's effect on P(full role) -- not a
        # bug, the same two lagged signals are used for two different
        # purposes.
        gm_mu, gm_sd = self.scalers['games_missed_ytd']
        sp_mu, sp_sd = self.scalers['lag_offense_pct']
        out['games_missed_z'] = ((frame['games_missed_ytd'] - gm_mu) / gm_sd).to_numpy()
        out['snap_pct_z'] = ((frame['lag_offense_pct'] - sp_mu) / sp_sd).to_numpy()
        return out

    def drop_cold_starts(self, frame):
        checks = frame.with_columns([
            pl.col(c).cast(pl.String).cast(self.enums[c], strict=False).alias(f"_{c}_chk")
            for c in self.idx_cols
        ])
        keep = checks.drop_nulls([f"_{c}_chk" for c in self.idx_cols]) \
                     .drop([f"_{c}_chk" for c in self.idx_cols])
        return keep


def ar_calendar(df_train, df_test, season_enum):
    """Per-fold global weekly timeline for opp_ar. One AR step per week
    actually present in df_train (in season order, using this fold's own
    season encoding), PLUS exactly one extra step appended at the very end
    for the test week being forecast. obs_data never reaches that final
    step, so its posterior is a genuine one-step-ahead AR forecast, sampled
    jointly with everything else via NUTS/ADVI -- no separate extrapolation
    step needed at predict time. Fold-specific (each fold's windowed
    history has a different season/week footprint) -- delegates the
    shared season_offsets/T/boundary_cols arithmetic to
    models.build_ar_calendar, the same helper src/ff_ar.py's top-level
    calendar derivation uses.

    Assumes df_test is a single (season, week) -- true for this harness's
    one-fold-per-week design."""
    seasons = [int(s) for s in season_enum.categories.to_list()]
    train_max = {
        int(s): float(mw) for s, mw in
        df_train.group_by('season').agg(pl.col('week').max().alias('mw')).iter_rows()
    }
    test_season = int(df_test['season'][0])
    test_week = float(df_test['week'][0])
    train_max[test_season] = max(train_max.get(test_season, 0.0), test_week)

    weeks_per_season = np.array([train_max.get(s, 0.0) for s in seasons])
    assert (weeks_per_season > 0).all(), "a season has zero weeks in this fold's window"
    return models.build_ar_calendar(weeks_per_season)


# ------------------------------------------------------------ fit/predict

def fit_predict(model_name, method, df_train, df_test, seed=0,
                 advi_n=20000, advi_draws=500,
                 nuts_draws=300, nuts_tune=300, nuts_chains=2):
    fold = Fold(df_train, IDX_COLS)
    Dtr = fold.arrays(df_train)
    n_positions = len(fold.coords['position'])
    pos_of_player, pos_counts = models.player_position_map(
        df_train, 'player', 'position', fold.enums, n_positions
    )
    season_offsets, T, boundary_cols = ar_calendar(df_train, df_test, fold.enums['season'])

    # Straight calls into src/models.py's build_* functions -- this fold's
    # Fold/D/season_offsets/T/boundary_cols are the only fold-specific
    # inputs; the model structure itself lives entirely in src/models.py.
    # 'team' requires D['team_form_z']/D['opp_team_form_z'] -- i.e.
    # run_one_fold must have joined processed-data/team_form.parquet onto
    # df_train/df_test before calling fit_predict('team', ...).
    # 'team_season' (build_team_season_mod) is STALE and not in
    # src/ff_ar.py's current active model set -- see the module note above
    # it in src/models.py before using it; kept here only for anyone
    # deliberately reaching for it, not in the CLI `choices` list.
    builders = {
        'hs': lambda: models.build_hs_version(Dtr, fold.coords, season_offsets, T, boundary_cols,
                                               pos_of_player, pos_counts, fold.week_grid_vals),
        'ar': lambda: models.build_ar_mod(Dtr, fold.coords, season_offsets, T, boundary_cols,
                                           pos_of_player, pos_counts),
        'team': lambda: models.build_team_mod(Dtr, fold.coords, season_offsets, T, boundary_cols,
                                               pos_of_player, pos_counts),
        'team_season': lambda: models.build_team_season_mod(Dtr, fold.coords, season_offsets, T, boundary_cols,
                                                              pos_of_player, pos_counts),
    }
    model = builders[model_name]()

    t0 = time.time()
    with model:
        if method == 'advi':
            approx = pm.fit(n=advi_n, method='advi', random_seed=seed, progressbar=False)
            idata = approx.sample(advi_draws)
        else:
            idata = pm.sample(draws=nuts_draws, tune=nuts_tune, chains=nuts_chains,
                               cores=nuts_chains, random_seed=seed, progressbar=False,
                               target_accept=0.9)
    fit_time = time.time() - t0

    keep = fold.drop_cold_starts(df_test)
    n_dropped = len(df_test) - len(keep)
    if len(keep) == 0:
        return None

    Dte = fold.arrays(keep)
    with model:
        set_data = {
            'player_data': Dte['player'],
            'position_data': Dte['position'],
            'team_data': Dte['team'],
            'tenure_data': Dte['tenure'],
            'cont_data': Dte['cont'],
            'bi_data': Dte['bi'],
            'opp_team_data': Dte['opp_team'],
            'season_data': Dte['season'],
            'week_data': Dte['week'],
            'player_season_data': Dte['player_season'],
            'games_missed_data': Dte['games_missed_z'],
            'snap_pct_data': Dte['snap_pct_z'],
            'global_time_data': models.global_time(Dte['season'], Dte['week'], season_offsets),
            'obs_data': np.zeros(len(keep)),  # placeholder, not used for y_obs draws
        }
        if model_name in ('hs', 'team'):  # both use the LKJ position intercept, which needs tenure_z_data
            set_data['tenure_z_data'] = Dte['tenure_z']
        if model_name == 'team':  # team_form/opp_team_form only exist for team_mod
            set_data['team_form_data'] = Dte['team_form_z']
            set_data['opp_team_form_data'] = Dte['opp_team_form_z']
        pm.set_data(set_data)
        ppc = pm.sample_posterior_predictive(idata, var_names=['y_obs'], progressbar=False,
                                              random_seed=seed + 1)
    S = ppc.posterior_predictive['y_obs'].values.reshape(-1, len(keep))
    return dict(keep=keep, S=S, ppc=ppc, n_dropped=n_dropped, fit_time=fit_time, n_train=len(df_train))


def fit_predict_stack(method, df_train, df_test, seed=0, weights=(0.58, 0.29, 0.13), **kwargs):
    """Blends team_mod, hs_version, and ar_mod's posterior-predictive draws
    via az.weight_predictions -- ArviZ's own implementation of exactly this
    (weighted resampling of each model's posterior_predictive draws in
    proportion to `weights`), rather than reimplementing it by hand.
    Default weights (0.58, 0.29, 0.13) are az.compare()'s LOO stacking
    weight for team_mod/hs_version/ar_mod (in that order) from the
    full-history src/ff_ar.py run. NOT re-optimized per fold; the point of
    running this through the harness at all is to check whether that
    LOO-derived blend actually helps on weeks none of the three models was
    fit on, since LOO weights are still computed from data all three saw
    during training."""
    r_team = fit_predict('team', method, df_train, df_test, seed=seed, **kwargs)
    r_hs = fit_predict('hs', method, df_train, df_test, seed=seed + 1000, **kwargs)
    r_ar = fit_predict('ar', method, df_train, df_test, seed=seed + 2000, **kwargs)
    if r_team is None or r_hs is None or r_ar is None:
        return None

    # all three models use IDX_COLS identically, so drop_cold_starts should
    # keep the same rows -- verify rather than assume, since a silent
    # misalignment here would score garbage without ever raising.
    if not (r_team['keep']['player'].to_list() == r_hs['keep']['player'].to_list()
            == r_ar['keep']['player'].to_list()):
        raise ValueError(
            "team/hs/ar kept different rows for this fold -- cold-start filtering diverged, "
            "can't blend their posterior-predictive draws row-for-row."
        )

    blended = az.weight_predictions([r_team['ppc'], r_hs['ppc'], r_ar['ppc']], weights=list(weights))
    # az.weight_predictions goes through az.extract internally, which
    # returns dims as (obs_dim, sample) -- NOT (sample, obs_dim) like the
    # raw pm.sample_posterior_predictive output fit_predict's S uses.
    # Reshaping blindly (as if it were the latter) silently scrambles
    # every row instead of erroring -- transpose by name instead of
    # trusting positional order.
    da = blended.posterior_predictive['y_obs']
    obs_dim = [d for d in da.dims if d != 'sample'][0]
    S = da.transpose('sample', obs_dim).values

    return dict(keep=r_team['keep'], S=S, n_dropped=r_team['n_dropped'],
                fit_time=r_team['fit_time'] + r_hs['fit_time'] + r_ar['fit_time'],
                n_train=r_team['n_train'])


# --------------------------------------------------------------- scoring

def crps(S, y, max_pairs=200, seed=0):
    t1 = np.abs(S - y[None, :]).mean(0)
    i = np.random.default_rng(seed).choice(S.shape[0], size=(2, min(max_pairs, S.shape[0])))
    t2 = np.abs(S[i[0]] - S[i[1]]).mean(0)
    return float((t1 - 0.5 * t2).mean())


def score(y, S, baseline=None, levels=(0.5, 0.8, 0.9)):
    pred = S.mean(0)
    out = dict(n=len(y), rmse=float(np.sqrt(((pred - y) ** 2).mean())),
               mae=float(np.abs(pred - y).mean()), crps=crps(S, y),
               spearman=float(spearmanr(pred, y).statistic))
    for L in levels:
        lo, hi = np.percentile(S, [100 * (1 - L) / 2, 100 * (1 + L) / 2], axis=0)
        out[f'cov{int(L * 100)}'] = float(((y >= lo) & (y <= hi)).mean())
    if baseline is not None:
        out['rmse_base'] = float(np.sqrt(((baseline - y) ** 2).mean()))
    return out


def past_only_baseline(keep, train, key='player_season'):
    b = train.group_by(key).agg(pl.col('total_fantasy_points').mean().alias('_pm'))
    return (keep.join(b, on=key, how='left')['_pm']
            .fill_null(float(train['total_fantasy_points'].mean())).to_numpy())


def fit_predict_baseline(df_train, df_test):
    """The naive past_only_baseline point prediction, run through the exact
    same score()/CSV pipeline as every real model so it shows up as its own
    'model' row instead of only ever being the rmse_base side-column other
    rows compute gain_pct against. No PyMC fit -- this is just
    past_only_baseline wrapped so it satisfies fit_predict's return
    contract (keep/S/n_dropped/fit_time/n_train).

    S is a single pseudo-draw per row (a point prediction has no posterior
    uncertainty). This is not a hack that needs special-casing in score():
    crps()'s pairwise term needs >=2 samples and is exactly 0 with only 1,
    so crps degrades gracefully to plain MAE; cov50/80/90 will come out
    ~0% since a single point can't form an interval. Both are the CORRECT
    answer for a point-estimate 'model', not something to fix.

    No cold-start filtering either -- past_only_baseline doesn't depend on
    any model's categorical encoders, so every df_test row is scoreable."""
    pred = past_only_baseline(df_test, df_train)
    S = pred[None, :]  # (1, n_obs)
    return dict(keep=df_test, S=S, n_dropped=0, fit_time=0.0, n_train=len(df_train))


# ------------------------------------------------------------------ main

def already_done(out_csv, model_name, method, season, week):
    """True if this exact (model, method, season, week) fold is already a row
    in out_csv -- lets a sweep be re-run/resumed without silently duplicating
    folds (same seed + same data => bit-identical metrics, just a wasted
    refit and a double-counted row in any later aggregation)."""
    import os
    if not os.path.exists(out_csv):
        return False
    existing = pl.read_csv(out_csv)
    match = existing.filter(
        (pl.col('model') == model_name) & (pl.col('method') == method) &
        (pl.col('season') == season) & (pl.col('week') == week)
    )
    return match.height > 0


def run_one_fold(model_name, method, season, week, out_csv=RESULTS_CSV, stack_weights=(0.58, 0.29, 0.13), **kwargs):
    if already_done(out_csv, model_name, method, season, week):
        print(f"SKIP: {model_name} {method} {season} wk{week} already in {out_csv}")
        return

    df = pl.read_parquet(DATA_PATH)
    if model_name in ('team', 'stack'):
        # same construction as src/ff_ar.py's top-of-file join: own-team
        # form keyed on (season, week, team), opponent form re-keyed on
        # (season, week, opp_team). Row-count assertions guard against the
        # exact fan-out bug src/estimate_latent_ability.py had (a join
        # missing 'team' as a key matched every team playing that week
        # against every other team's team_form row, 216704 rows instead of
        # 7232) -- if team_form.parquet ever regresses, this fails loudly
        # here instead of silently ballooning memory three models later.
        team_form = pl.read_parquet(TEAM_FORM_PATH)
        assert team_form.height == team_form.select('season', 'week', 'team').unique().height, (
            "team_form.parquet has duplicate (season, week, team) keys -- "
            "re-check the join in src/estimate_latent_ability.py before using it here"
        )
        opp_team_form = team_form.rename({'team': 'opp_team', 'team_form': 'opp_team_form'})

        n0 = df.height
        df = df.join(team_form, on=['season', 'week', 'team'], how='left')
        assert df.height == n0, f"team_form join fanned out rows: {n0} -> {df.height}"
        df = df.join(opp_team_form, on=['season', 'week', 'opp_team'], how='left')
        assert df.height == n0, f"opp_team_form join fanned out rows: {n0} -> {df.height}"

        n_missing = df.select((pl.col('team_form').is_null() | pl.col('opp_team_form').is_null()).sum()).item()
        if n_missing:
            print(f"warning: {n_missing} rows have no team_form/opp_team_form match -- filling with 0.0")
        df = df.with_columns(pl.col('team_form', 'opp_team_form').fill_null(0.0))

    train = df.filter(
        (pl.col('season') >= season - (WINDOW_SEASONS - 1)) &
        ((pl.col('season') < season) | ((pl.col('season') == season) & (pl.col('week') < week)))
    )
    test = df.filter((pl.col('season') == season) & (pl.col('week') == week))
    if len(test) == 0 or len(train) < 200:
        print(f"SKIP: no data for {season} wk{week}")
        return

    if model_name == 'stack':
        r = fit_predict_stack(method, train, test, weights=stack_weights, **kwargs)
    elif model_name == 'baseline':
        # no PyMC fit, no method/seed/draws kwargs apply -- method is still
        # a required CLI arg (accepted, just ignored) so already_done()'s
        # (model, method, season, week) key still works for skip/resume.
        r = fit_predict_baseline(train, test)
    else:
        r = fit_predict(model_name, method, train, test, **kwargs)
    if r is None:
        print(f"SKIP: all test rows were cold starts for {season} wk{week}")
        return

    y = r['keep']['total_fantasy_points'].to_numpy()
    base = past_only_baseline(r['keep'], train)
    m = score(y, r['S'], baseline=base)
    m.update(model=model_name, method=method, season=season, week=week,
             fit_time=r['fit_time'], n_train=r['n_train'], n_dropped=r['n_dropped'])
    print(json.dumps(m, indent=2))

    row = pl.DataFrame([m])
    import os
    if os.path.exists(out_csv):
        existing = pl.read_csv(out_csv)
        pl.concat([existing, row], how='diagonal_relaxed').write_csv(out_csv)
    else:
        row.write_csv(out_csv)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    # team_season left out of the default choices since it's not in
    # src/ff_ar.py's current active model set -- pass model_name=
    # 'team_season' to fit_predict() directly if you deliberately want it.
    p.add_argument('model', choices=['hs', 'ar', 'team', 'stack', 'baseline'])
    p.add_argument('method', choices=['advi', 'nuts'])
    p.add_argument('season', type=int)
    p.add_argument('week', type=float)
    p.add_argument('--nuts-draws', type=int, default=300)
    p.add_argument('--nuts-tune', type=int, default=300)
    p.add_argument('--chains', type=int, default=2)
    p.add_argument('--advi-n', type=int, default=20000)
    p.add_argument('--advi-draws', type=int, default=500)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', type=str, default=RESULTS_CSV)
    p.add_argument('--stack-weights', type=str, default='0.58,0.29,0.13',
                    help="comma-separated 'team,hs,ar' weights for model=stack, e.g. from az.compare()'s weight column")
    args = p.parse_args()

    stack_weights = tuple(float(x) for x in args.stack_weights.split(','))

    run_one_fold(args.model, args.method, args.season, args.week,
                 out_csv=args.out,
                 stack_weights=stack_weights,
                 seed=args.seed,
                 advi_n=args.advi_n, advi_draws=args.advi_draws,
                 nuts_draws=args.nuts_draws, nuts_tune=args.nuts_tune,
                 nuts_chains=args.chains)
