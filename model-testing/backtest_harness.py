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
import team_context as tc   # signed Vegas lines + walk-forward team-model context

DATA_PATH = "processed-data/ff-processed-adv-added.parquet"   # run from the project root
TEAM_FORM_PATH = "processed-data/team_form.parquet"  # from src/estimate_latent_ability.py
# walk-forward team context (src/team_pregame_features.py --model division): season s from a fit through s-1
TEAM_PREGAME_PATH = "processed-data/team_pregame_division.parquet"
RESULTS_CSV = "results/backtest_walkforward.csv"
WINDOW_SEASONS = 3   # how many seasons of history to feed each fold
MAX_WEEK = 18   # NFL schedule's full week range (17 through 2020, 18 since
                # the 2021 expansion) -- see Fold.__init__: `week`'s enum is
                # pinned to this full range rather than fit from whatever
                # weeks happen to appear in a fold's windowed train, so a
                # fold can't treat its own test week as an out-of-vocabulary
                # "cold start" just because no prior season is in the window
                # yet (e.g. every fold in the dataset's first season, 2013,
                # under WINDOW_SEASONS=3).

# STD_COLS/BINARY_VARS/BART_CONT_VARS/BART_IDX_COLS now live in
# src/models.py as the single shared copy -- this file used to carry its
# own hand-copied STD_COLS, and it had already drifted from
# src/fit_model.py's (missing 'lag_avg_depth_of_target') despite
# fit_model.py's docstring claiming the two matched exactly. See
# models.py's comment above STD_COLS for the full story. Local aliases
# kept so the rest of this file (and its own docstrings/comments) don't
# need touching beyond this import.
STD_COLS = models.STD_COLS
BINARY_VARS = models.BINARY_VARS
BART_CONT_VARS = models.BART_CONT_VARS
BART_IDX_COLS = models.BART_IDX_COLS

# 'player_season' (add_recent_form_ar/add_recent_form_hsgp) and 'week'
# (add_recent_form_hsgp's within-season time axis) match src/ff_ar.py's
# idx_cols -- every current architecture needs these.
IDX_COLS = ['player_id', 'position', 'team', 'tenure', 'opp_team', 'season', 'week',
            'player_season']

# het_gp models -> the --positions group each one requires
HET_MODEL_POSITIONS = {'het_gp': 'skill', 'het_gp_t': 'skill', 'het_gp_qb': 'qb', 'het_gp_qb_t': 'qb',
                       # advanced-stats ablation: noadv drops the NGS/QBR lags, asof swaps in
                       # src/build_asof_advanced.py's as-of versions (needs --data pointing at
                       # processed-data/ff-processed-adv-asof.parquet)
                       'het_gp_noadv': 'skill', 'het_gp_asof': 'skill',
                       'het_gp_qb_noadv': 'qb', 'het_gp_qb_asof': 'qb',
                       # team-context ablation (src/run_team_ctx_ablation.sh): signed Vegas lines,
                       # and the off/def team model's exp_pts_for / exp_margin (needs TEAM_PREGAME_PATH)
                       'het_gp_vegas': 'skill', 'het_gp_ctx': 'skill', 'het_gp_ctx_novegas': 'skill',
                       'het_gp_qb_vegas': 'qb', 'het_gp_qb_ctx': 'qb', 'het_gp_qb_ctx_novegas': 'qb'}
HET_MODEL_CONFIG    = {'het_gp': 'skill', 'het_gp_t': 'skill_t', 'het_gp_qb': 'qb', 'het_gp_qb_t': 'qb_t',
                       'het_gp_noadv': 'skill_noadv', 'het_gp_asof': 'skill_asof',
                       'het_gp_qb_noadv': 'qb_noadv', 'het_gp_qb_asof': 'qb_asof',
                       'het_gp_vegas': 'skill_vegas', 'het_gp_ctx': 'skill_ctx',
                       'het_gp_ctx_novegas': 'skill_ctx_novegas',
                       'het_gp_qb_vegas': 'qb_vegas', 'het_gp_qb_ctx': 'qb_ctx',
                       'het_gp_qb_ctx_novegas': 'qb_ctx_novegas'}


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
        if 'week' in idx_cols:
            # `week` is a bounded, known-in-advance index (1..MAX_WEEK), not
            # an open-ended category -- make_enums' default (whatever values
            # happen to appear in this fold's train slice) breaks down
            # whenever train can't yet see the full week range (no prior
            # season in the window, e.g. every fold in the dataset's first
            # season under WINDOW_SEASONS=3): the test week is then
            # guaranteed absent from train's vocabulary -- train only has
            # week < test_week by construction -- so drop_cold_starts used
            # to drop 100% of the test set even though nothing was actually
            # a genuinely new player/team. Pinning the vocabulary to the
            # full range fixes that, and also keeps week_idx exactly equal
            # to week - 1 (which global_time's
            # season_offsets[:-1][season_idx] + week_idx arithmetic depends
            # on) instead of depending on the incidental sort order of
            # whichever weeks happen to be present.
            self.enums['week'] = pl.Enum([str(float(w)) for w in range(1, MAX_WEEK + 1)])
        # models.SHRUNK_ADD_COLS (lag_avg_depth_of_target) is kept
        # ALONGSIDE its raw column when --shrunk is passed -- run_one_fold
        # calls models.apply_shrunk_features(df) before Fold is
        # constructed, which adds a 'lag_avg_depth_of_target_shrunk'
        # column onto `train` (raw column itself untouched). Detected the
        # same way team_form/opp_team_form presence is detected below --
        # a plain (non-shrunk) fold simply won't have this column, so
        # cont_vars/scalers stay exactly STD_COLS in that case.
        self.shrunk_add_cols = [f'{c}_shrunk' for c in models.SHRUNK_ADD_COLS if f'{c}_shrunk' in train.columns]
        cont_vars = STD_COLS + self.shrunk_add_cols
        self.scalers = models.fit_scalers(train, cont_vars)
        self.coords = {c: e.categories.to_list() for c, e in self.enums.items()}
        self.coords['cont_vars'] = cont_vars
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
        # models.py's build_* functions (via base_data/build_bart_mod) all
        # read D['player'] -- kept as that key regardless of which raw
        # column backs it, so it doesn't matter to them that the raw
        # column is 'player_id' (unique) rather than a display name.
        # 'player_id' is used for encoding/enums specifically because
        # player NAMES aren't unique (e.g. more than one real "Josh
        # Allen" can appear in the same data -- a QB and, at other times,
        # a different position player -- so deduplicating/joining on name
        # would silently merge distinct careers into one estimate).
        if 'player_id' in out:
            out['player'] = out['player_id']
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
        # models.base_data also registers target_share_data/rush_share_data (the
        # role mixture's inputs) -- same standardization as their cont_array columns.
        for key, col in (('target_share_z', 'lag_target_share'), ('rush_share_z', 'lag_rush_share')):
            mu_c, sd_c = self.scalers[col]
            out[key] = ((frame[col] - mu_c) / sd_c).to_numpy()
        # bart_mod3's raw (unstandardized) covariate matrix -- see
        # BART_CONT_VARS above. Only buildable once team_form/
        # opp_team_form have been joined onto `frame` (run_one_fold's
        # bart branch does this, same join team_mod/r2d2_mod use).
        if 'team_form' in frame.columns and 'opp_team_form' in frame.columns:
            out['bart_cont'] = frame.select(
                BART_CONT_VARS + self.shrunk_add_cols + BINARY_VARS
            ).to_numpy().astype('float64')
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

    # Cold-start check moved ahead of model construction/fitting -- it used
    # to run only after a full NUTS/ADVI fit completed, wasting the entire
    # fit whenever a fold turned out to be all-cold (see the week-vocabulary
    # fix above, written for exactly this: every fold in season 2013 was
    # silently all-cold and burning a full fit before being discarded).
    # Checking here means a doomed fold is caught in milliseconds instead.
    keep = fold.drop_cold_starts(df_test)
    n_dropped = len(df_test) - len(keep)
    if len(keep) == 0:
        return None

    Dtr = fold.arrays(df_train)
    n_positions = len(fold.coords['position'])
    pos_of_player, pos_counts = models.player_position_map(
        df_train, 'player_id', 'position', fold.enums, n_positions
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
        'r2d2': lambda: models.build_r2d2_mod(Dtr, fold.coords, season_offsets, T, boundary_cols,
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
                               target_accept=0.99)
    fit_time = time.time() - t0

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
            'target_share_data': Dte['target_share_z'],
            'rush_share_data': Dte['rush_share_z'],
            'global_time_data': models.global_time(Dte['season'], Dte['week'], season_offsets),
            'obs_data': np.zeros(len(keep)),  # placeholder, not used for y_obs draws
        }
        if model_name in ('hs', 'team', 'r2d2'):  # all three use the LKJ position intercept, which needs tenure_z_data
            set_data['tenure_z_data'] = Dte['tenure_z']
        if model_name in ('team', 'r2d2'):  # team_form/opp_team_form only exist for team_mod/r2d2_mod
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
    if not (r_team['keep']['player_id'].to_list() == r_hs['keep']['player_id'].to_list()
            == r_ar['keep']['player_id'].to_list()):
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


def fit_predict_bart(method, df_train, df_test, seed=0, m=50,
                      nuts_draws=300, nuts_tune=300, nuts_chains=2, **_ignored):
    """bart_mod3 (models.build_bart_mod3, ported from src/bart_mod.py) gets
    its own fit+predict path rather than slotting into fit_predict's
    generic builders dict: it needs none of the AR-calendar plumbing
    (season_offsets/T/boundary_cols) every other architecture shares, and
    its covariates are fold-local RAW (unscaled) values (BART_CONT_VARS)
    instead of the z-scored STD_COLS every other model's cont_priors
    dot-product uses. **_ignored swallows advi_n/advi_draws when this is
    called through run_one_fold's generic kwargs -- irrelevant here since
    ADVI isn't supported (see below).

    NUTS only -- ADVI is not a valid fit method for a model containing a
    BART term. pymc_bart.BART.logp is a constant pt.zeros_like(x) (BART's
    "prior" isn't a differentiable density over tree structure), so
    pm.fit(method='advi') runs without erroring but never moves bart_mu
    away from its initval (Y.mean()) -- confirmed empirically against
    this project's pinned pymc-bart: ADVI silently freezes the BART
    component instead of failing loudly, which is worse than an error,
    hence the explicit guard below.

    Predicting the fitted trees at new (held-out) rows DOES go through the
    ordinary pm.set_data + pm.sample_posterior_predictive path used by
    every other model here -- verified empirically against this project's
    pinned pymc-bart that swapping 'bart_cont_data' really does
    re-evaluate the fitted trees at the new X, not just at whatever rows
    it was trained on. (A cold read of pymc_bart's BARTRV.rng_fn looks
    like it always uses a frozen `cls.X` captured at construction, which
    would make this broken -- but that captured reference is itself a
    lazy pytensor expression over the pm.Data shared variable, and
    pymc_bart's own prediction path calls .eval() on it, which re-reads
    the shared variable's CURRENT (post-set_data) value. Don't rewire this
    to the all_trees/_sample_posterior private-API workaround without
    re-confirming that reasoning against whatever pymc-bart version is
    pinned at the time.)"""
    if method != 'nuts':
        raise ValueError(
            "model='bart' only supports method='nuts' -- pymc-bart's BART term has no "
            "ADVI-compatible logp, see fit_predict_bart's docstring"
        )

    fold = Fold(df_train, BART_IDX_COLS)
    keep = fold.drop_cold_starts(df_test)
    n_dropped = len(df_test) - len(keep)
    if len(keep) == 0:
        return None

    Dtr = fold.arrays(df_train)
    n_positions = len(fold.coords['position'])
    pos_of_player, pos_counts = models.player_position_map(
        df_train, 'player_id', 'position', fold.enums, n_positions
    )
    model = models.build_bart_mod(Dtr, fold.coords, pos_of_player, pos_counts, m=m)

    t0 = time.time()
    with model:
        idata = pm.sample(draws=nuts_draws, tune=nuts_tune, chains=nuts_chains,
                           cores=nuts_chains, random_seed=seed, progressbar=False,
                           target_accept=0.9)
    fit_time = time.time() - t0

    Dte = fold.arrays(keep)
    with model:
        pm.set_data({
            'bart_cont_data': Dte['bart_cont'],
            'player_data': Dte['player'],
            'tenure_data': Dte['tenure'],
            'obs_data': np.zeros(len(keep)),  # placeholder, not used for y_obs draws
        })
        ppc = pm.sample_posterior_predictive(idata, var_names=['y_obs'], progressbar=False,
                                              random_seed=seed + 1)
    S = ppc.posterior_predictive['y_obs'].values.reshape(-1, len(keep))
    return dict(keep=keep, S=S, ppc=ppc, n_dropped=n_dropped, fit_time=fit_time, n_train=len(df_train))


def fit_predict_het_gp(method, df_train, df_test, group='skill', seed=0, advi_n=20000, advi_draws=500,
                        nuts_draws=300, nuts_tune=300, nuts_chains=2, nuts_sampler='pymc'):
    """het_gp (group='skill', RB/WR/TE) / het_gp_qb (group='qb') -- the
    heteroskedastic latent-GP models from src/ff_scratchpad_skill.py /
    src/ff_scratchpad_qb.py, built by models.build_het_gp.

    Scored rows are chosen by the SAME Fold(df_train, IDX_COLS).drop_cold_starts
    every other model uses, so metrics are row-for-row comparable. Everything
    het_gp fits on data lives in models.HetPrep and is fit on df_train only."""
    keep = Fold(df_train, IDX_COLS).drop_cold_starts(df_test)
    n_dropped = len(df_test) - len(keep)
    if len(keep) == 0:
        return None

    cfg = models.HET_CONFIGS[group]
    train_g = df_train.filter(pl.col('position').is_in(list(cfg.positions)))
    prep = models.HetPrep(train_g, cfg)
    Dtr = prep.arrays(train_g)
    model = models.build_het_gp(Dtr, prep)

    t0 = time.time()
    with model:
        if method == 'advi':
            approx = pm.fit(n=advi_n, method='advi', random_seed=seed, progressbar=False)
            idata = approx.sample(advi_draws)
        else:
            idata = pm.sample(draws=nuts_draws, tune=nuts_tune, chains=nuts_chains,
                              cores=nuts_chains, random_seed=seed, progressbar=False,
                              target_accept=0.95, nuts_sampler=nuts_sampler)
    fit_time = time.time() - t0
    n_div = int(idata.sample_stats['diverging'].sum()) if method == 'nuts' else 0

    Dte = prep.arrays(keep)
    with model:
        pm.set_data(models.het_payload(Dte))
        ppc = pm.sample_posterior_predictive(idata, var_names=['y_obs'], progressbar=False,
                                             random_seed=seed + 1)
    S = ppc.posterior_predictive['y_obs'].values.reshape(-1, len(keep)) * prep.y_sd + prep.y_mu
    cp = idata.posterior['cont_prior']
    coefs = pl.DataFrame({'cont_var': [str(v) for v in cp['cont_vars'].values],
                          'coef_mean': cp.mean(('chain', 'draw')).values,
                          'coef_sd': cp.std(('chain', 'draw')).values})
    if cfg.team_ctx:
        # team-context slopes, one per (position, variable) -- named ctx:<var>[<pos>]
        name = 'ctx_beta' if 'ctx_beta' in idata.posterior else 'ctx_beta_bar'
        cb = idata.posterior[name]
        m, sd = cb.mean(('chain', 'draw')).values, cb.std(('chain', 'draw')).values
        if name == 'ctx_beta':
            labels = [f'ctx:{v}[{p_}]' for p_ in cb['position'].values for v in cb['ctx_vars'].values]
        else:
            labels = [f'ctx:{v}[{prep.positions[0]}]' for v in cb['ctx_vars'].values]
        coefs = pl.concat([coefs, pl.DataFrame({'cont_var': labels, 'coef_mean': m.ravel(),
                                                'coef_sd': sd.ravel()})])
    return dict(keep=keep, S=S, ppc=ppc, n_dropped=n_dropped, fit_time=fit_time,
                n_train=len(Dtr['y']), divergences=n_div, coefs=coefs)


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

def already_done(out_csv, model_name, method, season, week, shrunk=False, positions='all'):
    """True if this exact (model, method, season, week, shrunk) fold is
    already a row in out_csv -- lets a sweep be re-run/resumed without
    silently duplicating folds (same seed + same data => bit-identical
    metrics, just a wasted refit and a double-counted row in any later
    aggregation). shrunk is part of the dedup key so a plain and a shrunk
    run of the same (model, season, week) don't collide -- they're
    genuinely different fits, not the same fold rerun."""
    import os
    if not os.path.exists(out_csv):
        return False
    existing = pl.read_csv(out_csv)
    if 'shrunk' not in existing.columns:
        # a CSV written before --shrunk existed -- every row in it is
        # implicitly a plain-features run.
        existing = existing.with_columns(pl.lit(False).alias('shrunk'))
    if 'positions' not in existing.columns:
        # a CSV written before --positions existed -- every row scored all positions.
        existing = existing.with_columns(pl.lit('all').alias('positions'))
    match = existing.filter(
        (pl.col('model') == model_name) & (pl.col('method') == method) &
        (pl.col('season') == season) & (pl.col('week') == week) &
        (pl.col('shrunk') == shrunk) & (pl.col('positions').fill_null('all') == positions)
    )
    return match.height > 0


def run_one_fold(model_name, method, season, week, out_csv=RESULTS_CSV, stack_weights=(0.58, 0.29, 0.13),
                  shrunk=False, bart_m=50, positions='all', nuts_sampler='pymc', data_path=DATA_PATH,
                  team_pregame_path=TEAM_PREGAME_PATH, **kwargs):
    required = HET_MODEL_POSITIONS.get(model_name)
    if required is not None and positions != required:
        raise ValueError(f"model={model_name!r} covers positions={required!r} only -- run it (and every model "
                         f"you compare it against) with --positions {required} so all rows scored are the same")
    if already_done(out_csv, model_name, method, season, week, shrunk=shrunk, positions=positions):
        print(f"SKIP: {model_name} {method} {season} wk{week} shrunk={shrunk} positions={positions} "
              f"already in {out_csv}")
        return

    df = pl.read_parquet(data_path)
    if positions != 'all':
        # one primary position per player, then keep only this group -- applied
        # before the window/test split so train, test, the baseline and every
        # model's cold-start check all see the same rows.
        df = models.with_primary_position(df)
        df = df.filter(pl.col('position').is_in(list(models.HET_CONFIGS[positions].positions)))
    if shrunk:
        # same swap src/fit_model.py's --shrunk uses (models.py's shared
        # helper) -- left-joins processed-data/shrinkage_features.csv and
        # overwrites SHRUNK_SWAP_COLS's VALUES in place, so STD_COLS's
        # column NAMES (and therefore Fold's cont_vars coord) are
        # unchanged either way; only the row's actual lag values differ.
        df = models.apply_shrunk_features(df)
    if model_name in ('team', 'stack', 'r2d2', 'bart') or model_name in HET_MODEL_POSITIONS:
        # same construction as src/ff_ar.py's top-of-file join: own-team
        # form keyed on (season, week, team), opponent form re-keyed on
        # (season, week, opp_team). Row-count assertions guard against the
        # exact fan-out bug src/estimate_latent_ability.py had (a join
        # missing 'team' as a key matched every team playing that week
        # against every other team's team_form row, 216704 rows instead of
        # 7232) -- if team_form.parquet ever regresses, this fails loudly
        # here instead of silently ballooning memory three models later.
        df = models.join_team_form(df, path=TEAM_FORM_PATH)   # previous week's form, all models alike
    if model_name in HET_MODEL_POSITIONS:
        # signed Vegas lines (team_spread, vegas_team_total) -- extra columns,
        # only read by the *_vegas / *_ctx configs' std_cols
        df = tc.add_vegas_features(df)
        cfg = models.HET_CONFIGS[HET_MODEL_CONFIG[model_name]]
        if cfg.team_ctx:
            # pre-game team context, joined EXACTLY on (season, week, team): each
            # season's values come from a team-model fit through the season before
            df = tc.join_team_pregame(df, path=team_pregame_path)
            n_null = df.select(pl.any_horizontal(pl.col(list(cfg.team_ctx)).is_null()).sum()).item()
            assert n_null == 0, f'{n_null} rows have no team context in {team_pregame_path}'

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
    elif model_name == 'bart':
        r = fit_predict_bart(method, train, test, m=bart_m, **kwargs)
    elif model_name in HET_MODEL_POSITIONS:
        r = fit_predict_het_gp(method, train, test, group=HET_MODEL_CONFIG[model_name],
                               nuts_sampler=nuts_sampler, **kwargs)
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
    m.update(model=model_name, method=method, season=season, week=week, shrunk=shrunk, positions=positions,
             divergences=r.get('divergences'),
             fit_time=r['fit_time'], n_train=r['n_train'], n_dropped=r['n_dropped'])
    print(json.dumps(m, indent=2))

    row = pl.DataFrame([m])
    import os
    if r.get('coefs') is not None:
        # per-fold posterior summary of the linear covariate effects, next to out_csv
        coef_csv = out_csv.removesuffix('.csv') + '_coefs.csv'
        c = r['coefs'].with_columns(pl.lit(model_name).alias('model'), pl.lit(season).alias('season'),
                                    pl.lit(week).alias('week'))
        if os.path.exists(coef_csv):
            c = pl.concat([pl.read_csv(coef_csv), c], how='diagonal_relaxed')
        c.write_csv(coef_csv)
    # per-position scores for multi-position folds (e.g. --positions skill), from the
    # same draws -- written next to out_csv so already_done()'s key is unaffected
    pos = r['keep']['position'].to_numpy()
    if len(np.unique(pos)) > 1:
        by_pos = []
        for p_ in np.unique(pos):
            mask = pos == p_
            if mask.sum() < 5:          # too few rows to score meaningfully
                continue
            mp = score(y[mask], r['S'][:, mask], baseline=base[mask])
            mp.update(model=model_name, method=method, season=season, week=week,
                      shrunk=shrunk, positions=positions, pos=str(p_))
            by_pos.append(mp)
        if by_pos:
            pos_csv = out_csv.removesuffix('.csv') + '_bypos.csv'
            new = pl.DataFrame(by_pos)
            if os.path.exists(pos_csv):
                new = pl.concat([pl.read_csv(pos_csv), new], how='diagonal_relaxed')
            new.write_csv(pos_csv)
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
    p.add_argument('model', choices=['hs', 'ar', 'team', 'r2d2', 'stack', 'baseline', 'bart', *HET_MODEL_POSITIONS])
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
    p.add_argument('--bart-m', type=int, default=50,
                    help="number of trees for model=bart's pymc_bart.BART ensemble (default matches "
                         "src/bart_mod.py's bart_mod3)")
    p.add_argument('--shrunk', action='store_true',
                    help="fit with the shrinkage_features.csv-derived shrunk lag columns instead of the plain lags "
                         "(models.SHRUNK_SWAP_COLS) -- same feature swap as src/fit_model.py's --shrunk")
    p.add_argument('--positions', choices=['all', 'skill', 'qb'], default='all',
                   help="'skill' (RB/WR/TE) or 'qb' restricts train AND test to that group -- required for "
                        "het_gp / het_gp_qb, and for any model you compare against them")
    p.add_argument('--data', type=str, default=DATA_PATH,
                   help="processed parquet to read (the het_gp*_asof models need "
                        "processed-data/ff-processed-adv-asof.parquet from src/build_asof_advanced.py)")
    p.add_argument('--sampler', choices=['pymc', 'nutpie'], default='pymc',
                   help="NUTS backend for het_gp / het_gp_qb")
    p.add_argument('--team-pregame', type=str, default=TEAM_PREGAME_PATH,
                   help="walk-forward team context for the het_gp*_ctx models "
                        "(python src/team_pregame_features.py --model division --out <this path>)")
    args = p.parse_args()

    stack_weights = tuple(float(x) for x in args.stack_weights.split(','))

    run_one_fold(args.model, args.method, args.season, args.week,
                 out_csv=args.out,
                 stack_weights=stack_weights,
                 shrunk=args.shrunk,
                 bart_m=args.bart_m,
                 positions=args.positions,
                 nuts_sampler=args.sampler,
                 data_path=args.data,
                 team_pregame_path=args.team_pregame,
                 seed=args.seed,
                 advi_n=args.advi_n, advi_draws=args.advi_draws,
                 nuts_draws=args.nuts_draws, nuts_tune=args.nuts_tune,
                 nuts_chains=args.chains)
