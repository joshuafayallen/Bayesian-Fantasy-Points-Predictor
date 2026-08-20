"""
Walk-forward backtest harness comparing the THREE model architectures
currently defined in src/ff-ar.py on held-out weekly accuracy:

    hs    hs_version -- LKJ-correlated position intercept + age slope +
                         opp_ar + an HSGP within-season recent-form curve
    ar    ar_mod     -- flat per-position intercept + opp_ar + a flat
                         per-player_season recent-form offset
    team  team_mod   -- hs_version's LKJ intercept + opp_ar (no
                         recent-form term) + two regression coefficients
                         (beta_team_form, beta_opp_team_form) on a
                         precomputed team-strength feature from
                         src/estimate-latent-ability.py's standalone
                         Bradley-Terry-style state-space fit on real game
                         scores. This REPLACED an earlier ZeroSumNormal
                         flat team intercept -- see src/ff-ar.py's
                         team_mod comment for why.

`team_season_mod` (a season-level AR(1) team-quality version) and
`team_ar_mod` (a week-level AR(1) team-quality version, dropped early)
are NOT ported here: neither is in src/ff-ar.py's current active model
set. team_ar_mod is gone for good (confirmed dead in src/ff-ar.py's own
history); team_season_mod could come back if src/ff-ar.py re-adds it,
but re-check this file against src/ff-ar.py's actual content before
assuming it still matches -- see the IMPORTANT note below.

All three architectures are ported faithfully from src/ff-ar.py: same
priors, same player_skill random-walk + position-centering fix (now a
one-hot matmul instead of inc_subtensor -- see add_player_skill), same
opponent AR construction (scatter-matmul boundary-jump injection PLUS a
pytensor.scan-based AR(1) recursion instead of the closed-form
rho-power-matrix trick -- all three were required to actually clear a
pytensor graph-rewrite bug, "BUG IN FGRAPH.REPLACE OR A LISTENER" /
local_add_of_sparse_write, that kept recurring in src/ff-ar.py no matter
how the surrounding code was restructured; see add_strength_ar), same
recent-form
components (add_recent_form_ar for ar_mod, add_recent_form_hsgp for
hs_version), and the same two-component role/participation mixture
likelihood (add_role_mixture) instead of a plain StudentT -- every
model's observation layer changed when src/ff-ar.py added this, so a
harness still scoring against the old single-StudentT likelihood would
be comparing genuinely different generative models, not just refit
copies.

KNOWN LIMITATION -- team_form leakage via hyperparameters, not
observations: team's team_form/opp_team_form feature is loaded from
processed-data/team_form.parquet, precomputed ONCE by
src/estimate-latent-ability.py across the full 2006-2025 history. Its
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
    python sandbox/backtest_harness.py hs advi 2025 12
    python sandbox/backtest_harness.py team advi 2025 12

Each invocation runs exactly ONE fold and appends one row to
results/backtest_walkforward.csv (checkpointed -- safe to run fold-by-fold
from a shell loop, or in parallel processes since each writes its own row).

ADVI is the fast/cheap path for sweeping many folds; NUTS is also wired
up (`method=nuts`) for a smaller, higher-fidelity confirmation set:

    python sandbox/backtest_harness.py team nuts 2025 12 \\
        --nuts-draws 500 --nuts-tune 500 --chains 4

To sweep every fold for every model (e.g. a full season, walking week by
week), just loop the single-fold call -- or use src/run_backtest.sh,
which already does this.

Once a sweep is done, `python sandbox/summarize_backtest.py
results/backtest_walkforward.csv` prints per-model pooled metrics and
picks the winner.

IMPORTANT: this file has no way to detect drift from src/ff-ar.py
automatically. If ff-ar.py's architectures change again (new covariates,
a different likelihood, a different team_mod feature), re-check IDX_COLS,
STD_COLS, and every build_* function here against it by hand.
"""
import sys
import json
import time
import argparse
import numpy as np
import polars as pl
import pymc as pm
import pytensor
import pytensor.tensor as pt
import arviz as az
from scipy.stats import spearmanr

DATA_PATH = "processed-data/ff-processed.parquet"   # run from the project root
TEAM_FORM_PATH = "processed-data/team_form.parquet"  # from src/estimate-latent-ability.py
RESULTS_CSV = "results/backtest_walkforward.csv"
WINDOW_SEASONS = 3   # how many seasons of history to feed each fold

# games_missed_ytd/lag_offense_pct/lag_st_pct added to match src/ff-ar.py's
# std_these -- these feed add_role_mixture's p_full logit, used by every
# model now that all three share the role/participation mixture likelihood.
STD_COLS = ['spread_line', 'roll_avg_points_allowed', 'lag_yards_per_rush_attempt',
            'lag_target_share', 'lag_yards_per_target', 'lag_yards_per_pass_attempt',
            'temp', 'wind', 'games_missed_ytd', 'lag_offense_pct', 'lag_st_pct']
BINARY_VARS = ['is_grass', 'is_indoors', 'is_home', 'div_game', 'era']

# 'player_season' (add_recent_form_ar/add_recent_form_hsgp) and 'week'
# (add_recent_form_hsgp's within-season time axis) added to match
# src/ff-ar.py's idx_cols -- all three current architectures need these
# now, not just team-quality variants.
IDX_COLS = ['player', 'position', 'team', 'tenure', 'opp_team', 'season', 'week',
            'player_season']


# --------------------------------------------------------------- data prep

def fit_scalers(df, cols):
    stats = df.select(
        [pl.col(c).mean().alias(f"{c}__mu") for c in cols] +
        [pl.col(c).std().alias(f"{c}__sd") for c in cols]
    ).row(0, named=True)
    return {c: (stats[f"{c}__mu"], stats[f"{c}__sd"] or 1.0) for c in cols}


def make_enums(df, cols):
    return {c: pl.Enum(df[c].unique().sort().cast(pl.String).to_list()) for c in cols}


def idx_array(frame, col, enums):
    return (frame[col].cast(pl.String).cast(enums[col]).to_physical()
            .cast(pl.Int32).to_numpy())


def cont_array(frame, scalers):
    cols = list(scalers.keys())
    return np.column_stack([((frame[c] - mu) / sd).to_numpy() for c, (mu, sd) in scalers.items()]) \
        if cols else np.zeros((len(frame), 0))


def bi_array(frame):
    return frame.select(BINARY_VARS).to_numpy().astype(float)


class Fold:
    """Encapsulates encoders + coords fit on ONE fold's windowed train slice."""

    def __init__(self, train, idx_cols):
        self.idx_cols = idx_cols
        self.enums = make_enums(train, idx_cols)
        self.scalers = fit_scalers(train, STD_COLS)
        self.coords = {c: e.categories.to_list() for c, e in self.enums.items()}
        self.coords['cont_vars'] = STD_COLS
        self.coords['binary_vars'] = BINARY_VARS
        # add_recent_form_hsgp's shared basis dim -- m must match the m=
        # passed there (default 6, same as src/ff-ar.py).
        self.coords['form_basis'] = list(range(6))
        if 'tenure' in idx_cols:
            # Same construction as coords2['tenure']-derived tenure_grid_vals
            # in src/ff-ar.py's hs_version: z-score the *tenure index* against
            # the mean/sd of this fold's own tenure category grid.
            tvals = np.array(self.coords['tenure'], dtype='float64')
            self._tenure_mean = tvals.mean()
            self._tenure_std = tvals.std()
        if 'week' in idx_cols:
            # within-season week grid for add_recent_form_hsgp's GP input --
            # same role as week_grid_vals in src/ff-ar.py.
            self.week_grid_vals = np.array(self.coords['week'], dtype='float64')
        # team_form/opp_team_form scaler -- fit fold-locally like every
        # other continuous covariate here (unlike src/ff-ar.py, which fits
        # one scaler across the full 2006-2025 history since it only ever
        # has one "fold"). Only meaningful if the caller already joined
        # team_form/opp_team_form onto `train` (see run_one_fold) -- build_hs
        # /build_ar never read these, only build_team does.
        if 'team_form' in train.columns:
            self._team_form_mu = train['team_form'].mean()
            self._team_form_sd = train['team_form'].std() or 1.0

    def arrays(self, frame):
        out = {c: idx_array(frame, c, self.enums) for c in self.idx_cols}
        out['cont'] = cont_array(frame, self.scalers)
        out['bi'] = bi_array(frame)
        out['y'] = frame['total_fantasy_points'].to_numpy()
        if 'tenure' in self.idx_cols:
            out['tenure_z'] = (out['tenure'].astype('float64') - self._tenure_mean) / self._tenure_std
        if hasattr(self, '_team_form_mu') and 'team_form' in frame.columns:
            # same scaler for both -- keeps beta_team_form/beta_opp_team_form
            # directly comparable in magnitude, matching src/ff-ar.py.
            out['team_form_z'] = ((frame['team_form'] - self._team_form_mu) / self._team_form_sd).to_numpy()
            out['opp_team_form_z'] = ((frame['opp_team_form'] - self._team_form_mu) / self._team_form_sd).to_numpy()
        # add_role_mixture's p_full logit inputs -- z-scored the same way
        # as their entries in cont_array (they're ALSO in STD_COLS/cont_vars,
        # matching src/ff-ar.py's common_data(): games_missed_ytd_z/
        # lag_offense_pct_z feed BOTH cont_priors' general effect on the
        # 'full' component AND role_beta_missed/role_beta_snap's effect on
        # P(full role) -- not a bug, the same two lagged signals are used
        # for two different purposes.
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


def player_position_map(train, fold):
    """player_idx -> position_idx lookup + per-position player counts, built
    from this fold's own train slice/enums. Mirrors player_position_map /
    position_group_counts in src/ff-ar.py, just re-derived per fold."""
    lut = train.select(['player', 'position']).unique(subset=['player'], keep='first')
    parr = idx_array(lut, 'player', fold.enums)
    posarr = idx_array(lut, 'position', fold.enums)
    n_players = len(fold.coords['player'])
    n_positions = len(fold.coords['position'])
    out = np.zeros(n_players, dtype=int)
    out[parr] = posarr
    counts = np.bincount(out, minlength=n_positions).astype('float64')
    return out, counts


def ar_calendar(df_train, df_test, season_enum):
    """Per-fold global weekly timeline for opp_ar. One AR step per week
    actually present in df_train (in season order, using this fold's own
    season encoding), PLUS exactly one extra step appended at the very end
    for the test week being forecast. obs_data never reaches that final
    step, so its posterior is a genuine one-step-ahead AR forecast, sampled
    jointly with everything else via NUTS/ADVI -- no separate extrapolation
    step needed at predict time. Mirrors the season_offsets/boundary_cols
    construction in src/ff-ar.py, just re-derived per fold since each
    fold's windowed history has a different season/week footprint.

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
    season_offsets = np.concatenate([[0], np.cumsum(weeks_per_season)]).astype(int)
    T = int(season_offsets[-1])
    boundary_step_idx = season_offsets[1:-1]     # first real week of every season but the first
    boundary_cols = boundary_step_idx - 1
    return season_offsets, T, boundary_cols


def global_time(season_idx, week_idx, season_offsets):
    return (season_offsets[:-1][season_idx] + week_idx).astype('int32')


# ---------------------------------------------------- shared model pieces

def add_player_skill(coords, position_of_player, position_group_counts, n_positions):
    """The random-walk-over-tenure player_skill term, hard-centered to zero
    mean within each position at every tenure level, shared identically by
    both models. See the "Hard-center player_skill" comment in
    src/ff-ar.py -- without this, positions_mu/ab_position and the average
    starting level of player_skill within a position are redundant, which
    tanks the position-level parameters' ESS."""
    n_tenure = len(coords['tenure'])
    init_sigma = pm.HalfNormal('init_sigma', 4.0)
    rw_sigma = pm.HalfNormal('rw_sigma', 0.5)
    z_innov = pm.Normal('z_innov', 0, 1, dims=('player', 'tenure'))
    level0 = init_sigma * z_innov[:, 0]
    steps = rw_sigma * z_innov[:, 1:]  # (n_player, n_tenure - 1)

    # player_skill_raw via pytensor.scan instead of
    # concatenate([level0[:, None], level0[:, None] + cumsum(steps, axis=1)])
    # -- ported from src/ff-ar.py's add_player_skill, where this exact
    # concatenate (combining level0 with a cumsum-of-a-Mul-derived block)
    # turned out to be the actual FGRAPH.REPLACE/local_add_of_sparse_write
    # trigger on hs_version's ADVI compile (confirmed via a (6724, 18) =
    # (player, tenure)-shaped AdvancedIncSubtensor/Mul crash). Same fix
    # family as add_strength_ar's theta_full -- scan avoids the
    # concatenate-of-a-computed-block pattern pytensor's canonicalizer was
    # rewriting into a sparse write.
    def _tenure_step(step_t, prev):
        return prev + step_t

    steps_seq = steps.T  # (n_tenure - 1, n_player)
    path, _ = pytensor.scan(
        fn=_tenure_step,
        sequences=[steps_seq],
        outputs_info=[level0],
        strict=True,
    )  # (n_tenure - 1, n_player) -- tenure steps 1..n_tenure-1

    player_skill_raw = pt.concatenate([level0[None, :], path], axis=0).T  # (n_player, n_tenure)

    position_of_player_data = pm.Data('position_of_player', position_of_player, dims='player')

    # one-hot (position x player) matmul instead of
    # inc_subtensor(group_sum[position_of_player_data, :], player_skill_raw)
    # -- ported from src/ff-ar.py's add_player_skill, which hit the
    # FGRAPH.REPLACE / local_add_of_sparse_write pytensor rewrite bug on
    # this exact AdvancedIncSubtensor-then-Add pattern. position_of_player
    # is a plain numpy array (not pm.Data), known at graph-build time, so
    # the one-hot matrix is a static constant -- one dense matmul instead.
    one_hot_position = np.zeros((n_positions, len(position_of_player)))
    one_hot_position[position_of_player, np.arange(len(position_of_player))] = 1.0
    group_sum = pt.dot(one_hot_position, player_skill_raw)
    group_mean = group_sum / position_group_counts[:, None]

    return pm.Deterministic(
        'player_skill',
        player_skill_raw - group_mean[position_of_player_data, :],
        dims=('player', 'tenure'),
    )


def add_strength_ar(prefix, group_dim, group_idx_data, global_time_data, T, boundary_cols):
    """Continuous non-centered AR(1)-with-season-jumps strength curve over
    `group_dim` (e.g. 'opp_team' or 'team') across this fold's full weekly
    timeline. Ported faithfully from src/ff-ar.py's opponent-strength
    derivation (used identically by ar_mod/hs_version for `opp_team`) and
    from ar_component() there (used by team_ar_mod for `team`) -- one
    helper here so both call sites in this harness share it instead of
    duplicating the tensor algebra per model."""
    rho          = pm.Beta(f'{prefix}_rho', 4, 2)
    sigma_theta  = pm.HalfNormal(f'{prefix}_sigma_theta', 1.0)
    sigma_season = pm.HalfNormal(f'{prefix}_sigma_season', 1.0)
    theta_init   = pm.Normal(f'{prefix}_theta_init', 0, 1.0, dims=group_dim)
    z_theta = pm.Normal(f'{prefix}_z_theta', 0, 1, dims=(group_dim, 'global_step'))
    z_eta   = pm.Normal(f'{prefix}_z_eta', 0, 1, dims=(group_dim, 'season_gap'))

    # boundary-column injection as a dense scatter matmul instead of
    # innov[:, boundary_cols] += ... -- pt.set_subtensor/AdvancedIncSubtensor
    # immediately followed by an Add downstream is a known trigger for a
    # pytensor graph-rewrite bug ("BUG IN FGRAPH.REPLACE OR A LISTENER",
    # local_add_of_sparse_write). scatter_mat is a small fixed 0/1 constant,
    # so this is one extra small matmul, not a meaningfully different
    # computation.
    if len(boundary_cols):
        n_gaps = len(boundary_cols)
        scatter_mat = np.zeros((n_gaps, T - 1))
        scatter_mat[np.arange(n_gaps), boundary_cols] = 1.0
        season_jump = sigma_season * pt.dot(z_eta, scatter_mat)  # (group_dim, T-1)
        innov = sigma_theta * z_theta + season_jump
    else:
        innov = sigma_theta * z_theta

    # theta_full via pytensor.scan instead of the closed-form rho-power-
    # matrix trick -- ported from src/ff-ar.py's add_strength_ar, where the
    # closed-form version's rho_powers (a large static triangular,
    # mostly-zero constant) kept triggering the same FGRAPH.REPLACE crash
    # no matter how the surrounding code was written, since pytensor's own
    # optimizer was rewriting the resulting tensordot internally. scan
    # avoids the triangular constant entirely.
    # recursion: theta_full[:, 0] = theta_init
    #            theta_full[:, t] = rho * theta_full[:, t-1] + innov[:, t-1]
    def _ar_step(innov_t, theta_prev, rho):
        return rho * theta_prev + innov_t

    innov_seq = innov.T  # (T-1, group_dim) -- scan iterates over axis 0

    theta_path, _ = pytensor.scan(
        fn=_ar_step,
        sequences=[innov_seq],
        outputs_info=[theta_init],
        non_sequences=[rho],
        strict=True,
    )  # (T-1, group_dim) -- steps t=1..T-1

    theta_full = pt.concatenate([theta_init[None, :], theta_path], axis=0).T  # (group_dim, T)

    strength = pm.Deterministic(
        f'{prefix}_ar',
        theta_full - pt.mean(theta_full, axis=0, keepdims=True),
        dims=(group_dim, 'global_step_full'),
    )
    return strength[group_idx_data, global_time_data]


def add_recent_form_ar(player_season_data):
    """Per-player-season 'recent form' offset. Ported from src/ff-ar.py's
    add_recent_form_ar -- simplified there from a within-season AR(1) over
    `week` after that version's form_sigma posterior piled at the zero
    boundary (essentially no week-to-week evolution beyond a persistent
    per-season shift). Used by build_ar."""
    form_sigma = pm.HalfNormal('form_sigma', 2.0)
    form_z = pm.Normal('form_z', 0, 1, dims='player_season')
    form = pm.Deterministic('recent_form', form_sigma * form_z, dims='player_season')
    return form[player_season_data]


def add_recent_form_hsgp(week_grid_vals, player_season_data, week_data, m=6, c=1.5):
    """HSGP analogue of add_recent_form_ar -- a smooth nonparametric curve
    over week-within-season instead of a first-order Markov process.
    Ported from src/ff-ar.py's add_recent_form_hsgp: one shared eigenbasis
    across every player_season, with per-player_season coefficients on
    that shared basis (the standard trick for making HSGP tractable across
    many groups). Used by build_hs. drop_first=False since the per-
    player_season mean-centering below already removes the constant term."""
    ell = pm.InverseGamma('form_ell', alpha=3, beta=6)
    eta = pm.HalfNormal('form_eta', 1.0)
    cov = eta ** 2 * pm.gp.cov.Matern52(1, ls=ell)
    gp = pm.gp.HSGP(m=[m], c=c, cov_func=cov, drop_first=False)
    phi, sqrt_psd = gp.prior_linearized(X=week_grid_vals[:, None])  # phi: (week, m)

    beta_raw = pm.Normal('form_beta_raw', 0, 1, dims=('player_season', 'form_basis'))
    form_curve = pt.dot(beta_raw * sqrt_psd[None, :], phi.T)  # (player_season, week)

    # row mean via a matmul against a static ones-vector instead of
    # pt.mean(form_curve, axis=1, keepdims=True) -- ported from
    # src/ff-ar.py's add_recent_form_hsgp, where this was the actual
    # FGRAPH.REPLACE/local_add_of_sparse_write trigger on hs_version
    # (pt.mean(keepdims=True)'s internal Sum -> Mul(1/n) -> ExpandDims
    # graph matched the crash's node names exactly).
    n_week_form = len(week_grid_vals)
    row_mean = pt.dot(form_curve, pt.ones((n_week_form, 1))) / n_week_form  # (player_season, 1)
    form = pm.Deterministic(
        'recent_form',
        form_curve - row_mean,
        dims=('player_season', 'week'),
    )
    return form[player_season_data, week_data]


def add_role_mixture(mu_full, sigma_full, position_data, games_missed_data, snap_pct_data, obs_data):
    """Two-component mixture for the observation layer. Ported from
    src/ff-ar.py's add_role_mixture -- replaces this harness's original
    plain pm.StudentT('y_obs', ...) likelihood, which no longer matches
    what any of the three current src/ff-ar.py architectures actually fit.
    'limited' deliberately doesn't see opp_ar/recent_form/cont_priors --
    when a player barely plays, matchup quality is close to irrelevant to
    how little they score. p_full's logit is driven by two lagged,
    pre-game role signals (games_missed_data, snap_pct_data) rather than a
    flat per-position rate. Used by build_ar/build_hs/build_team, replacing
    the sigma_obs + pm.StudentT('y_obs', ...) call each used to end with."""
    mu_limited = pm.HalfNormal('mu_limited', 1.5, dims='position')
    sigma_limited = pm.HalfNormal('sigma_limited', 1.5, dims='position')

    role_intercept = pm.Normal('role_intercept', 0, 1.5, dims='position')
    role_beta_missed = pm.Normal('role_beta_missed', 0, 1, dims='position')
    role_beta_snap = pm.Normal('role_beta_snap', 0, 1, dims='position')

    logit_p_full = (
        role_intercept[position_data]
        + role_beta_missed[position_data] * games_missed_data
        + role_beta_snap[position_data] * snap_pct_data
    )
    p_full = pm.math.sigmoid(logit_p_full)
    w = pt.stack([1 - p_full, p_full], axis=-1)  # (n_obs, 2)

    comp_limited = pm.StudentT.dist(nu=4, mu=mu_limited[position_data], sigma=sigma_limited[position_data])
    comp_full = pm.StudentT.dist(nu=4, mu=mu_full, sigma=sigma_full[position_data])

    return pm.Mixture('y_obs', w=w, comp_dists=[comp_limited, comp_full], observed=obs_data)


def add_team_season_ar(team_data, season_data):
    """Team quality as a season-level AR(1) -- persistent/geometrically-
    decaying like opp_ar's rho, but held constant across every week within
    a season rather than reshuffling weekly. Ported from team_season_mod
    in src/ff-ar.py: a contender's strength should carry over
    season-to-season, but there's no reason to expect team identity
    (roster/scheme/coaching) to move week to week the way opp_ar's
    game-to-game defensive variation does -- and running team quality on
    a coarser clock than opp_ar also avoids the opp_sigma_theta collision
    team_ar_mod hit (two week-level AR processes competing for the same
    variance budget). Uses pm.AR directly since there's no separate
    weekly-jump term needed here, unlike add_strength_ar."""
    team_rho = pm.Beta('team_rho', 4, 2)
    team_sigma = pm.HalfNormal('team_sigma', 1.0)
    team_season_raw = pm.AR(
        'team_season_ar_raw', rho=team_rho, sigma=team_sigma,
        init_dist=pm.Normal.dist(0, 1.0), constant=False, ar_order=1,
        dims=('team', 'season'),
    )
    team_season_ar = pm.Deterministic(
        'team_season_ar',
        team_season_raw - pt.mean(team_season_raw, axis=0, keepdims=True),
        dims=('team', 'season'),
    )
    return team_season_ar[team_data, season_data]


def add_lkj_position_intercept(position_data, tenure_z_data, eta=4):
    """Correlated position intercept + age slope via LKJ, shared by
    build_hs and build_team -- matches add_lkj_position_intercept in
    src/ff-ar.py. Factored out here now that both hs and team use it, so
    they can't drift the way build_team did before (it had been left on
    an old flat-intercept + plain-Normal team-effect version after
    src/ff-ar.py's team_mod moved to LKJ + ZeroSumNormal)."""
    pos_chol, pos_corr, pos_sigmas = pm.LKJCholeskyCov(
        'pos_chol_cov', n=2, eta=eta,
        sd_dist=pm.Exponential.dist(1.0),
        compute_corr=True,
    )
    intercept = pm.Normal('intercept', 6, 3)
    beta = pm.Normal('beta', 0, 1)
    ab_position = pm.MvNormal(
        'ab_position',
        mu=pt.stack([intercept, beta]),
        chol=pos_chol,
        dims=('position', 'pos_param'),
    )
    return pm.Deterministic(
        'intercepts',
        ab_position[position_data, 0] + ab_position[position_data, 1] * tenure_z_data,
    )


def _base_data(D, season_offsets):
    """pm.Data nodes shared by all models -- everything except the
    position-intercept structure and the team-quality term, which are
    model-specific. games_missed_data/snap_pct_data/player_season_data/
    week_data added to match src/ff-ar.py's common_data(): every current
    architecture uses add_role_mixture (needs the first two), and
    build_ar/build_hs use recent_form (needs the last two -- week_data
    doubles as both add_recent_form_hsgp's within-season time axis and
    global_time's step-within-season input, same dual role week_idx plays
    in src/ff-ar.py)."""
    return dict(
        player_data=pm.Data('player_data', D['player']),
        position_data=pm.Data('position_data', D['position']),
        team_data=pm.Data('team_data', D['team']),
        tenure_data=pm.Data('tenure_data', D['tenure']),
        cont_data=pm.Data('cont_data', D['cont']),
        bi_data=pm.Data('bi_data', D['bi']),
        obs_data=pm.Data('obs_data', D['y']),
        opp_team_data=pm.Data('opp_team_data', D['opp_team']),
        season_data=pm.Data('season_data', D['season']),
        week_data=pm.Data('week_data', D['week']),
        player_season_data=pm.Data('player_season_data', D['player_season']),
        games_missed_data=pm.Data('games_missed_data', D['games_missed_z']),
        snap_pct_data=pm.Data('snap_pct_data', D['snap_pct_z']),
        global_time_data=pm.Data(
            'global_time_data', global_time(D['season'], D['week'], season_offsets)
        ),
    )


# ------------------------------------------------------------- hs_version

def build_hs(fold: Fold, D, position_of_player, position_group_counts,
             season_offsets, T, boundary_cols):
    coords = dict(fold.coords)
    coords['pos_param'] = ['positions_mu', 'age_slope']
    n_positions = len(coords['position'])
    n_seasons = len(coords['season'])
    coords['global_step'] = list(range(T - 1))
    coords['global_step_full'] = list(range(T))
    coords['season_gap'] = list(range(max(n_seasons - 1, 0)))

    with pm.Model(coords=coords) as model:
        dat = _base_data(D, season_offsets)
        tenure_z_data = pm.Data('tenure_z_data', D['tenure_z'])

        # correlated varying effects across position: baseline scoring level
        # and linear tenure/age slope, drawn jointly per position via an
        # LKJ-correlated covariance (eta=4 -- shrinks the correlation toward
        # 0, matching the current src/ff-ar.py; an earlier eta=2 run and a
        # non-centered attempt both gave worse ESS -- see the comments on
        # hs_version there for why centered + tighter eta won).
        intercepts = add_lkj_position_intercept(dat['position_data'], tenure_z_data)

        cont_priors = pm.Normal('cont_priors', 0, 1, dims='cont_vars')
        bi_priors = pm.Normal('bi_priors', 0, 1.5, dims='binary_vars')

        player_skill = add_player_skill(coords, position_of_player, position_group_counts, n_positions)
        f_career = player_skill[dat['player_data'], dat['tenure_data']]

        f_opp = add_strength_ar('opp', 'opp_team', dat['opp_team_data'], dat['global_time_data'], T, boundary_cols)
        f_form = add_recent_form_hsgp(fold.week_grid_vals, dat['player_season_data'], dat['week_data'])

        mu = (
            intercepts
            + pm.math.dot(dat['cont_data'], cont_priors)
            + pm.math.dot(dat['bi_data'], bi_priors)
            + f_career
            + f_opp
            + f_form
        )
        sigma_obs = pm.HalfNormal('sigma_obs', 5.0, dims='position')
        add_role_mixture(mu, sigma_obs, dat['position_data'],
                          dat['games_missed_data'], dat['snap_pct_data'], dat['obs_data'])
    return model


# ----------------------------------------------------------------- ar_mod

def build_ar(fold: Fold, D, season_offsets, T, boundary_cols, position_of_player, position_group_counts):
    coords = dict(fold.coords)
    n_seasons = len(coords['season'])
    n_positions = len(coords['position'])
    coords['global_step'] = list(range(T - 1))
    coords['global_step_full'] = list(range(T))
    coords['season_gap'] = list(range(max(n_seasons - 1, 0)))

    with pm.Model(coords=coords) as model:
        dat = _base_data(D, season_offsets)

        # flat per-position intercept -- no LKJ correlation with age slope.
        positions_mu = pm.Normal('positions_mu', 6, 3, dims='position')
        intercepts = pm.Deterministic('intercepts', positions_mu[dat['position_data']])

        cont_priors = pm.Normal('cont_priors', 0, 1, dims='cont_vars')
        bi_priors = pm.Normal('bi_priors', 0, 1.5, dims='binary_vars')

        player_skill = add_player_skill(coords, position_of_player, position_group_counts, n_positions)
        f_career = player_skill[dat['player_data'], dat['tenure_data']]

        f_opp = add_strength_ar('opp', 'opp_team', dat['opp_team_data'], dat['global_time_data'], T, boundary_cols)
        f_form = add_recent_form_ar(dat['player_season_data'])

        mu = intercepts + pm.math.dot(dat['cont_data'], cont_priors) + pm.math.dot(dat['bi_data'], bi_priors) \
            + f_career + f_opp + f_form

        sigma_obs = pm.HalfNormal('sigma_obs', 5.0, dims='position')
        add_role_mixture(mu, sigma_obs, dat['position_data'],
                          dat['games_missed_data'], dat['snap_pct_data'], dat['obs_data'])
    return model


# -------------------------------------------------------------- team_mod

def build_team(fold: Fold, D, season_offsets, T, boundary_cols, position_of_player, position_group_counts):
    """Matches src/ff-ar.py's CURRENT team_mod: the same LKJ-correlated
    position/age-slope intercept as build_hs + opp_ar (no recent-form
    term -- team_mod doesn't have one in src/ff-ar.py) + two regression
    coefficients (beta_team_form, beta_opp_team_form) on a precomputed
    team-strength feature. This REPLACED an earlier ZeroSumNormal flat
    team_effect -- see src/ff-ar.py's team_mod comment for why (that
    version kept colliding with other team-level components and
    collapsing ESS to zero; the current version is a fixed, precomputed
    input with no joint estimation risk). Requires D['team_form_z']/
    D['opp_team_form_z'] to be present -- i.e. run_one_fold must have
    joined processed-data/team_form.parquet onto df_train/df_test before
    calling fit_predict('team', ...)."""
    coords = dict(fold.coords)
    coords['pos_param'] = ['positions_mu', 'age_slope']
    n_seasons = len(coords['season'])
    n_positions = len(coords['position'])
    coords['global_step'] = list(range(T - 1))
    coords['global_step_full'] = list(range(T))
    coords['season_gap'] = list(range(max(n_seasons - 1, 0)))

    with pm.Model(coords=coords) as model:
        dat = _base_data(D, season_offsets)
        tenure_z_data = pm.Data('tenure_z_data', D['tenure_z'])
        intercepts = add_lkj_position_intercept(dat['position_data'], tenure_z_data)

        cont_priors = pm.Normal('cont_priors', 0, 1, dims='cont_vars')
        bi_priors = pm.Normal('bi_priors', 0, 1.5, dims='binary_vars')

        player_skill = add_player_skill(coords, position_of_player, position_group_counts, n_positions)
        f_career = player_skill[dat['player_data'], dat['tenure_data']]

        f_opp = add_strength_ar('opp', 'opp_team', dat['opp_team_data'], dat['global_time_data'], T, boundary_cols)

        # own-team and opponent-team strength as two separate
        # coefficients, not a single differenced term -- matches
        # src/ff-ar.py's team_mod: no reason to assume a player's own
        # offense helping them and the opponent's defense hurting them
        # are equal-and-opposite effects on THEIR fantasy points
        # specifically.
        team_form_data = pm.Data('team_form_data', D['team_form_z'])
        opp_team_form_data = pm.Data('opp_team_form_data', D['opp_team_form_z'])
        beta_team_form = pm.Normal('beta_team_form', 0, 1)
        beta_opp_team_form = pm.Normal('beta_opp_team_form', 0, 1)
        f_team_form = beta_team_form * team_form_data + beta_opp_team_form * opp_team_form_data

        mu = intercepts + pm.math.dot(dat['cont_data'], cont_priors) + pm.math.dot(dat['bi_data'], bi_priors) \
            + f_career + f_opp + f_team_form

        sigma_obs = pm.HalfNormal('sigma_obs', 5.0, dims='position')
        add_role_mixture(mu, sigma_obs, dat['position_data'],
                          dat['games_missed_data'], dat['snap_pct_data'], dat['obs_data'])
    return model


# --------------------------------------------------------- retired models
#
# build_team_ar (a week-level AR(1) team-quality version) is gone --
# confirmed dead in src/ff-ar.py's own history/module docstring, not
# something to keep testing against.
#
# build_team_season is kept below for reference/potential revival, but is
# NOT wired into `builders`/the CLI `choices` list by default: src/ff-ar.py
# currently has no team_season_mod at all (it was dropped from the active
# comparison there). This version is also stale relative to how team_mod
# itself changed -- it still uses ar_mod's flat position intercept and the
# plain StudentT likelihood, not add_role_mixture. If team_season_mod comes
# back in src/ff-ar.py, port it properly (LKJ intercept? recent_form?
# role_mixture? -- check ff-ar.py, don't assume this old version is right)
# rather than just re-enabling this function as-is.

def build_team_season(fold: Fold, D, season_offsets, T, boundary_cols, position_of_player, position_group_counts):
    """STALE -- see the module-level note above before using this. Kept as
    ar_mod's structure + opp_ar + team quality as a season-level AR(1)
    instead of a flat intercept (build_team, in its pre-team_form form) or
    a week-level AR (the retired build_team_ar). See add_team_season_ar
    for the rationale. Note WINDOW_SEASONS=3 means most folds only give
    this pm.AR 2-3 time steps to work with."""
    coords = dict(fold.coords)
    n_seasons = len(coords['season'])
    n_positions = len(coords['position'])
    coords['global_step'] = list(range(T - 1))
    coords['global_step_full'] = list(range(T))
    coords['season_gap'] = list(range(max(n_seasons - 1, 0)))

    with pm.Model(coords=coords) as model:
        dat = _base_data(D, season_offsets)

        positions_mu = pm.Normal('positions_mu', 6, 3, dims='position')
        intercepts = pm.Deterministic('intercepts', positions_mu[dat['position_data']])

        cont_priors = pm.Normal('cont_priors', 0, 1, dims='cont_vars')
        bi_priors = pm.Normal('bi_priors', 0, 1.5, dims='binary_vars')

        player_skill = add_player_skill(coords, position_of_player, position_group_counts, n_positions)
        f_career = player_skill[dat['player_data'], dat['tenure_data']]

        f_opp = add_strength_ar('opp', 'opp_team', dat['opp_team_data'], dat['global_time_data'], T, boundary_cols)
        f_team_season = add_team_season_ar(dat['team_data'], dat['season_data'])

        mu = intercepts + pm.math.dot(dat['cont_data'], cont_priors) + pm.math.dot(dat['bi_data'], bi_priors) \
            + f_career + f_opp + f_team_season

        sigma_obs = pm.HalfNormal('sigma_obs', 0.75)
        pm.StudentT('y_obs', nu=4, mu=mu, sigma=sigma_obs, observed=dat['obs_data'])
    return model


# ------------------------------------------------------------ fit/predict

def fit_predict(model_name, method, df_train, df_test, seed=0,
                 advi_n=20000, advi_draws=500,
                 nuts_draws=300, nuts_tune=300, nuts_chains=2):
    fold = Fold(df_train, IDX_COLS)
    Dtr = fold.arrays(df_train)
    pos_of_player, pos_counts = player_position_map(df_train, fold)
    season_offsets, T, boundary_cols = ar_calendar(df_train, df_test, fold.enums['season'])

    builders = {
        'hs': lambda: build_hs(fold, Dtr, pos_of_player, pos_counts, season_offsets, T, boundary_cols),
        'ar': lambda: build_ar(fold, Dtr, season_offsets, T, boundary_cols, pos_of_player, pos_counts),
        'team': lambda: build_team(fold, Dtr, season_offsets, T, boundary_cols, pos_of_player, pos_counts),
        # not in src/ff-ar.py's current active model set -- see the
        # "retired models" note above build_team_season. Callable here for
        # anyone deliberately reaching for it, but not in the CLI choices.
        'team_season': lambda: build_team_season(fold, Dtr, season_offsets, T, boundary_cols, pos_of_player, pos_counts),
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
            'global_time_data': global_time(Dte['season'], Dte['week'], season_offsets),
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
    full-history src/ff-ar.py run. NOT re-optimized per fold; the point of
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
        # same construction as src/ff-ar.py's top-of-file join: own-team
        # form keyed on (season, week, team), opponent form re-keyed on
        # (season, week, opp_team). Row-count assertions guard against the
        # exact fan-out bug src/estimate-latent-ability.py had (a join
        # missing 'team' as a key matched every team playing that week
        # against every other team's team_form row, 216704 rows instead of
        # 7232) -- if team_form.parquet ever regresses, this fails loudly
        # here instead of silently ballooning memory three models later.
        team_form = pl.read_parquet(TEAM_FORM_PATH)
        assert team_form.height == team_form.select('season', 'week', 'team').unique().height, (
            "team_form.parquet has duplicate (season, week, team) keys -- "
            "re-check the join in src/estimate-latent-ability.py before using it here"
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
    # team_ar removed (dead in src/ff-ar.py); team_season left out of the
    # default choices since it's not in src/ff-ar.py's current model set --
    # call build_team_season directly if you deliberately want it.
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
