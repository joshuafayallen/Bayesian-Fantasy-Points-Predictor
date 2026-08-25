"""
Shared PyMC model-building plumbing for the fantasy-points predictor.

WHY THIS FILE EXISTS
---------------------
Before this module existed, src/ff_ar.py (the full-history "fit and
diagnose" script) and src/backtest_harness.py (the windowed walk-forward
backtest) each defined their own copies of every model-building function
(add_player_skill, add_strength_ar, add_recent_form_ar/hsgp,
add_role_mixture, add_lkj_position_intercept, ...) and the three model
architectures built from them (ar_mod, hs_version, team_mod). The two
copies were ported "by hand" and had already drifted at least once (see
git history / backtest_harness.py's old module docstring, which had an
explicit "IMPORTANT: this file has no way to detect drift from
src/ff_ar.py automatically" warning).

This module is now the single source of truth for that model-building
code. src/ff_ar.py and src/backtest_harness.py both import from here.
Nothing in this module reads data from disk, does any Polars-frame-level
data prep tied to a *specific* run, or has module-level state -- every
function takes exactly the coords/arrays it needs as arguments and
returns a value (a pm.Model, a pm.Data node, a dict of arrays). That is
what makes it safe to call once from ff_ar.py's single full-history
"fold" and once per fold from backtest_harness.py's walk-forward loop.

SOURCE OF TRUTH
----------------
Every function here is ported from src/ff_ar.py's inline model-building
code, which remains the canonical description of "what the model is."
When ff_ar.py's models change, this file should change to match, and
callers (ff_ar.py and backtest_harness.py) should not need to change at
all beyond picking up the new coords/data plumbing requires.
"""

import numpy as np
import polars as pl
import polars.selectors as cs
import pymc as pm
import pytensor
import pytensor.tensor as pt


# ============================================================================
# Data prep -- scalers, categorical encoders, arrays
# ============================================================================
# Two parallel representations are kept on purpose:
#   * the polars/frame-level helpers (fit_scalers, apply_scalers, encode)
#     operate on a whole training frame at once and add `_z`/`_idx` columns
#     -- this is ff_ar.py's original style, convenient for a single
#     full-history fit where "the frame" and "the model's training data"
#     are the same object throughout the file.
#   * the numpy-array helpers (cont_array, bi_array, idx_array) build the
#     flat arrays a pm.Data node actually wants -- this is
#     backtest_harness.py's original style, convenient when a fold's
#     train/test frames need to be turned into arrays independently using
#     encoders fit once on `train`.
# Both compute the exact same standardization ((x - mu) / sd) and the same
# categorical -> contiguous-int encoding; pick whichever fits the calling
# code.

def fit_scalers(df, cols):
    """{col: (mean, sd)} fit on `df`. sd is floored to 1.0 when a column is
    constant in this slice (e.g. a windowed fold with no variance in some
    covariate) so apply_scalers/cont_array never divides by zero."""
    stats = df.select(
        [pl.col(c).mean().alias(f"{c}__mu") for c in cols] +
        [pl.col(c).std().alias(f"{c}__sd") for c in cols]
    ).row(0, named=True)
    return {c: (stats[f"{c}__mu"], stats[f"{c}__sd"] or 1.0) for c in cols}


def apply_scalers(frame, scalers):
    """Add a `{col}_z` column for every entry in `scalers`."""
    return frame.with_columns([
        ((pl.col(c) - mu) / sd).alias(f"{c}_z") for c, (mu, sd) in scalers.items()
    ])


def cont_array(frame, scalers):
    """Same standardization as apply_scalers, as a single (n_rows, n_cols)
    numpy array in `scalers`' key order -- for building a cont_data pm.Data
    node directly instead of going through apply_scalers + cs.ends_with('_z')."""
    cols = list(scalers.keys())
    if not cols:
        return np.zeros((len(frame), 0))
    return np.column_stack([((frame[c] - mu) / sd).to_numpy() for c, (mu, sd) in scalers.items()])


def bi_array(frame, binary_vars):
    return frame.select(binary_vars).to_numpy().astype(float)


def make_enums(df, cols):
    """{col: pl.Enum(sorted unique categories)}, fit on `df`."""
    return {
        col: pl.Enum(df[col].unique().sort().cast(pl.String).to_list())
        for col in cols
    }


# kept as an alias -- ff_ar.py originally called this make_indices.
make_indices = make_enums


def encode(df, cols, enums):
    """Add a `{col}_idx` Int32 column for every entry in `enums`, using each
    column's already-fit pl.Enum categorical encoding."""
    return df.with_columns([
        pl.col(col).cast(pl.String).cast(enums[col]).to_physical().cast(pl.Int32).alias(f"{col}_idx")
        for col in cols
    ])


def idx_array(frame, col, enums):
    """Same encoding as encode(), as a single numpy int array for one column
    -- for arrays that were encoded with a *different* fold's enums than the
    frame they're being applied to (e.g. scoring a test frame against
    train-fit enums)."""
    return (frame[col].cast(pl.String).cast(enums[col]).to_physical()
            .cast(pl.Int32).to_numpy())


# ============================================================================
# Calendar helpers -- the global weekly AR timeline shared by every model's
# opp_ar (and, for team_season_mod, the coarser team/season timeline)
# ============================================================================

def build_ar_calendar(weeks_per_season):
    """Given the number of weeks present for each season (in season order),
    return (season_offsets, T, boundary_cols):
      season_offsets -- cumulative week count, one extra leading 0. Season i
                         occupies global steps season_offsets[i]:season_offsets[i+1].
      T              -- total number of global weekly steps.
      boundary_cols  -- innovation-array column indices (into a (group, T-1)
                         innovation matrix) corresponding to the season-jump
                         steps: the first real week of every season but the
                         first. Used by add_strength_ar's season_jump term.

    Shared math behind ff_ar.py's top-level season_offsets/T/boundary_cols
    derivation and backtest_harness.py's per-fold ar_calendar()."""
    weeks_per_season = np.asarray(weeks_per_season, dtype='int64')
    season_offsets = np.concatenate([[0], np.cumsum(weeks_per_season)]).astype(int)
    T = int(season_offsets[-1])
    boundary_step_idx = season_offsets[1:-1]  # first real week of every season but the first
    boundary_cols = boundary_step_idx - 1
    return season_offsets, T, boundary_cols


def global_time(season_idx, week_idx, season_offsets):
    """Global weekly step for each row, given per-row season/week indices
    and the season_offsets from build_ar_calendar."""
    return (season_offsets[:-1][season_idx] + week_idx).astype('int32')


# ============================================================================
# Player/position lookup -- for add_player_skill's within-position centering
# ============================================================================

def player_position_map(frame, col_player, col_position, enums, n_positions):
    """player_idx -> position_idx lookup (`position_of_player`, one entry per
    player) plus per-position player counts (`position_group_counts`),
    built from `frame`'s own (player, position) pairs and `enums`.

    Every player must map to exactly one position in `frame` (a player who
    switched position mid-history would violate add_player_skill's
    within-position centering assumption) -- this is not defended against
    here; `.unique(subset=[col_player], keep='first')` silently takes
    whichever row comes first, matching the historical behavior of both
    ff_ar.py and backtest_harness.py."""
    n_players = len(enums[col_player].categories)
    lut = frame.select([col_player, col_position]).unique(subset=[col_player], keep='first')
    parr = idx_array(lut, col_player, enums)
    posarr = idx_array(lut, col_position, enums)
    out = np.zeros(n_players, dtype=int)
    out[parr] = posarr
    counts = np.bincount(out, minlength=n_positions).astype('float64')
    return out, counts


# ============================================================================
# pm.Data plumbing -- must be called inside an active `with pm.Model(...):`
# ============================================================================

def base_data(D, season_offsets):
    """The pm.Data nodes needed by every model below. Not every model uses
    every key as a *used* input (e.g. team_data/season_data are unused by
    ar_mod/hs_version/team_mod's mu, only by the retired team_season_mod)
    -- unused pm.Data nodes are harmless, and one definition here is much
    less likely to drift than several near-identical copies.

    `D` must be a dict with the arrays produced by fold_arrays() (or an
    equivalent construction): player, position, team, tenure, cont, bi, y,
    opp_team, season, week, player_season, games_missed_z, snap_pct_z."""
    return dict(
        player_data=pm.Data('player_data', D['player']),
        position_data=pm.Data('position_data', D['position']),
        team_data=pm.Data('team_data', D['team']),
        tenure_data=pm.Data('tenure_data', D['tenure']),
        cont_data=pm.Data('cont_data', np.ascontiguousarray(D['cont'])),
        bi_data=pm.Data('bi_data', np.ascontiguousarray(D['bi'])),
        obs_data=pm.Data('obs_data', D['y']),
        opp_team_data=pm.Data('opp_team_data', D['opp_team']),
        season_data=pm.Data('season_data', D['season']),
        week_data=pm.Data('week_data', D['week']),
        player_season_data=pm.Data('player_season_data', D['player_season']),
        games_missed_data=pm.Data('games_missed_data', D['games_missed_z']),
        snap_pct_data=pm.Data('snap_pct_data', D['snap_pct_z']),
        global_time_data=pm.Data('global_time_data', global_time(D['season'], D['week'], season_offsets)),
    )


# ============================================================================
# Shared model-building blocks
# ============================================================================
# Every function below must be called inside an active `with pm.Model(...):`
# context.

def add_player_skill(coords, position_of_player, position_group_counts, n_positions):
    """The random-walk-over-tenure player_skill term, hard-centered to zero
    mean within each position at every tenure level. Without this
    centering, positions_mu/ab_position and the average starting level of
    player_skill within a position are redundant, which tanks the
    position-level parameters' ESS."""
    n_tenure = len(coords['tenure'])
    init_sigma = pm.HalfNormal('init_sigma', 4.0)
    rw_sigma = pm.HalfNormal('rw_sigma', 0.5)
    z_innov = pm.Normal('z_innov', 0, 1, dims=('player', 'tenure'))
    level0 = init_sigma * z_innov[:, 0]
    steps = rw_sigma * z_innov[:, 1:]  # (player, n_tenure - 1)

    # player_skill_raw via pytensor.scan instead of
    # concatenate([level0[:, None], level0[:, None] + cumsum(steps, axis=1)])
    # -- that concatenate (combining level0 with a cumsum-of-a-Mul-derived
    # block) is a known trigger for a pytensor graph-rewrite bug ("BUG IN
    # FGRAPH.REPLACE OR A LISTENER", local_add_of_sparse_write). scan builds
    # the recursion step-by-step, so there's no concatenate-of-a-computed-
    # block for pytensor's canonicalizer to rewrite into a sparse write.
    # Verified numerically identical to the cumsum-based construction in a
    # standalone numpy script.
    def _tenure_step(step_t, prev):
        return prev + step_t

    steps_seq = steps.T  # (n_tenure - 1, player)
    path, _ = pytensor.scan(
        fn=_tenure_step,
        sequences=[steps_seq],
        outputs_info=[level0],
        strict=True,
    )  # (n_tenure - 1, player) -- tenure steps 1..n_tenure-1

    player_skill_raw = pt.concatenate([level0[None, :], path], axis=0).T  # (player, n_tenure)

    position_of_player_data = pm.Data('position_of_player', position_of_player, dims='player')

    # one-hot (position x player) matmul instead of
    # inc_subtensor(group_sum[position_of_player_data, :], player_skill_raw)
    # -- same local_add_of_sparse_write trigger family as above.
    # position_of_player is a plain numpy array (not pm.Data), known at
    # graph-build time, so the one-hot matrix is a static constant -- one
    # dense matmul, not a meaningfully different computation.
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
    `group_dim` (e.g. 'opp_team') across the full weekly timeline. Used for
    opp_ar in every model."""
    rho          = pm.Beta(f'{prefix}_rho', 4, 2)
    sigma_theta  = pm.HalfNormal(f'{prefix}_sigma_theta', 1.0)
    sigma_season = pm.HalfNormal(f'{prefix}_sigma_season', 1.0)
    theta_init   = pm.Normal(f'{prefix}_theta_init', 0, 1, dims=group_dim)
    z_theta = pm.Normal(f'{prefix}_z_theta', 0, 1, dims=(group_dim, 'global_step'))
    z_eta   = pm.Normal(f'{prefix}_z_eta', 0, 1, dims=(group_dim, 'season_gap'))

    # boundary-column injection as a dense scatter matmul instead of
    # innov[:, boundary_cols] += ... -- pt.set_subtensor/AdvancedIncSubtensor
    # immediately followed by an Add downstream is a known trigger for the
    # same FGRAPH.REPLACE / local_add_of_sparse_write bug. scatter_mat is a
    # small fixed 0/1 constant, so this is one extra small matmul.
    # `if len(boundary_cols)` guards a single-season slice (a short
    # backtest fold's train window can legitimately have no season
    # boundary at all) -- harmless no-op for the usual multi-season case.
    if len(boundary_cols):
        n_gaps = len(boundary_cols)
        scatter_mat = np.zeros((n_gaps, T - 1))
        scatter_mat[np.arange(n_gaps), boundary_cols] = 1.0
        season_jump = sigma_season * pt.dot(z_eta, scatter_mat)  # (group_dim, T-1)
        innov = sigma_theta * z_theta + season_jump
    else:
        innov = sigma_theta * z_theta

    # theta_full via pytensor.scan instead of the closed-form rho-power-
    # matrix trick -- rho_powers (a large static triangular, mostly-zero
    # constant) kept triggering the same FGRAPH.REPLACE crash regardless of
    # how the surrounding code was written, since pytensor's own optimizer
    # was rewriting the resulting tensordot internally. scan avoids the
    # triangular constant entirely.
    #
    # recursion: theta_full[:, 0] = theta_init
    #            theta_full[:, t] = rho * theta_full[:, t-1] + innov[:, t-1]
    # (verified against the closed-form construction in a standalone numpy
    # script -- identical to float precision.)
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


def add_team_season_ar(team_data, season_data):
    """Team quality as a season-level AR(1) -- persistent/geometrically-
    decaying like opp_ar's rho, but held constant across every week within
    a season rather than reshuffling weekly. Used only by the retired
    team_season_mod (see build_team_season_mod) -- neither ar_mod,
    hs_version nor team_mod use this."""
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


def add_recent_form_ar(player_season_data):
    """Per-player-season 'recent form' offset -- simplified from an
    original within-season AR(1) over `week` after that version's
    form_sigma posterior piled at the zero boundary (essentially no
    week-to-week evolution beyond a persistent per-season shift). Used by
    ar_mod."""
    form_sigma = pm.HalfNormal('form_sigma', 2.0)
    form_z = pm.Normal('form_z', 0, 1, dims='player_season')
    form = pm.Deterministic('recent_form', form_sigma * form_z, dims='player_season')
    return form[player_season_data]


def add_recent_form_hsgp(week_grid_vals, player_season_data, week_data, m=6, c=1.5):
    """HSGP analogue of add_recent_form_ar: same role (the within-season
    wobble around player_skill's season-level average), different
    generative story -- a smooth nonparametric curve instead of a
    first-order Markov process. Used by hs_version.

    Hierarchical/shared-basis HSGP: the eigenbasis (and its lengthscale/
    amplitude hyperparameters) is shared across every player_season -- one
    GP, not thousands of independent ones -- but each player_season gets
    its own partially-pooled coefficients on that shared basis via
    `beta_raw`."""
    ell = pm.InverseGamma('form_ell', alpha=3, beta=6)  # weeks -- a few-week correlation range
    eta = pm.HalfNormal('form_eta', 1.0)
    cov = eta ** 2 * pm.gp.cov.Matern52(1, ls=ell)
    gp = pm.gp.HSGP(m=[m], c=c, cov_func=cov, drop_first=False)
    phi, sqrt_psd = gp.prior_linearized(X=week_grid_vals[:, None])  # phi: (week, m)

    beta_raw = pm.Normal('form_beta_raw', 0, 1, dims=('player_season', 'form_basis'))
    form_curve = pt.dot(beta_raw * sqrt_psd[None, :], phi.T)  # (player_season, week)

    # row mean via a matmul against a static ones-vector instead of
    # pt.mean(form_curve, axis=1, keepdims=True) -- pt.mean(keepdims=True)'s
    # internal Sum -> Mul(1/n) -> ExpandDims graph is the same
    # FGRAPH.REPLACE/local_add_of_sparse_write trigger family as
    # add_strength_ar/add_player_skill above. n_week_form is a small fixed
    # constant, so this is one cheap matmul.
    n_week_form = len(week_grid_vals)
    row_mean = pt.dot(form_curve, pt.ones((n_week_form, 1))) / n_week_form  # (player_season, 1)
    form = pm.Deterministic(
        'recent_form',
        form_curve - row_mean,
        dims=('player_season', 'week'),
    )
    return form[player_season_data, week_data]


def add_flat_position_intercept(position_data):
    """Flat per-position baseline scoring level -- no age/tenure slope, no
    correlation across positions. Used by ar_mod."""
    positions_mu = pm.Normal('positions_mu', 6, 3, dims='position')
    return pm.Deterministic('intercepts', positions_mu[position_data])


def add_lkj_position_intercept(position_data, tenure_z_data, eta=4):
    """Correlated varying effects across position: baseline scoring level
    and a linear tenure/age slope, drawn jointly per position via an
    LKJ-correlated covariance -- McElreath's "varying slopes" cafes
    pattern. Partially pools both quantities across positions and lets the
    model pick up genuine correlation between them (e.g. if
    higher-baseline positions also tend to decline faster/slower). Used by
    hs_version and team_mod.

    Centered (not non-centered): non-centering through pos_chol_cov made
    ESS worse across the board. Each position has thousands of
    player-weeks pinning down its (intercept, slope) pair, so this isn't a
    data-starved-group funnel where non-centering helps -- it's stuck the
    other way.

    Caller is responsible for creating tenure_z_data (a pm.Data node with
    dims aligned to the model's rows) and passing it in -- this function
    only builds the LKJ structure and the resulting intercepts, matching
    the every-other-block-takes-its-data-as-an-argument convention used
    throughout this module."""
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
    pm.Deterministic('positions_mu', ab_position[:, 0], dims='position')
    pm.Deterministic('age_slope', ab_position[:, 1], dims='position')
    return pm.Deterministic(
        'intercepts',
        ab_position[position_data, 0] + ab_position[position_data, 1] * tenure_z_data,
    )


def add_role_mixture(mu_full, sigma_full, position_data, games_missed_data, snap_pct_data, obs_data):
    """Two-component mixture for the observation layer: when a player
    barely plays, matchup quality is close to irrelevant to how little
    they score. 'full' is the existing linear predictor, unchanged.
    'limited' deliberately doesn't see opp_ar/recent_form/cont_priors.

    p_full's logit is driven by two lagged, pre-game role signals instead
    of a flat per-position rate:
      - games_missed_data: has this player recently been absent (injury-
        return risk)
      - snap_pct_data: trailing snap share when they DID play -- catches
        healthy-but-limited-role cases (committee backs, WR4s) that
        games_missed_ytd can't see.
    Both must already be lagged in data prep -- never this week's actual
    snap count, which would leak the outcome into its own predictor. Used
    by every current model (ar_mod, hs_version, team_mod)."""
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
    # NOT wrapped in pm.Deterministic -- this is one value per row (tens of
    # thousands+), and idata files are already large; tracking a full
    # per-row vector every draw would make that meaningfully worse for a
    # quantity that's cheap to recompute post-hoc from role_intercept/
    # role_beta_* if needed.
    p_full = pm.math.sigmoid(logit_p_full)
    w = pt.stack([1 - p_full, p_full], axis=-1)  # (n_obs, 2)

    comp_limited = pm.StudentT.dist(nu=4, mu=mu_limited[position_data], sigma=sigma_limited[position_data])
    comp_full = pm.StudentT.dist(nu=4, mu=mu_full, sigma=sigma_full[position_data])

    return pm.Mixture('y_obs', w=w, comp_dists=[comp_limited, comp_full], observed=obs_data)


# ============================================================================
# Full model builders
# ============================================================================
# Each of these takes exactly the data/coords a fold (or the full-history
# "single fold") has prepared, and returns a ready-to-sample pm.Model.
# `coords` must already include every dim these models' shared blocks need:
# player, position, team, tenure, opp_team, season, week, player_season,
# cont_vars, binary_vars -- plus 'pos_param', 'global_step',
# 'global_step_full', 'season_gap', and (hs_version/team_mod only)
# 'form_basis', built here from `T`/`boundary_cols`/`n_seasons` so callers
# don't have to duplicate that bookkeeping.

def _extend_coords(coords, T, n_seasons, pos_param=False):
    coords = dict(coords)
    coords['global_step'] = list(range(T - 1))
    coords['global_step_full'] = list(range(T))
    coords['season_gap'] = list(range(max(n_seasons - 1, 0)))
    if pos_param:
        coords['pos_param'] = ['positions_mu', 'age_slope']
    return coords


def build_ar_mod(D, coords, season_offsets, T, boundary_cols,
                  position_of_player, position_group_counts):
    """ar_mod -- flat per-position intercept + opponent-strength AR(1) +
    a flat per-player_season recent-form offset."""
    n_positions = len(coords['position'])
    n_seasons = len(coords['season'])
    coords = _extend_coords(coords, T, n_seasons)

    with pm.Model(coords=coords) as model:
        dat = base_data(D, season_offsets)
        intercepts = add_flat_position_intercept(dat['position_data'])
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


def build_hs_version(D, coords, season_offsets, T, boundary_cols,
                      position_of_player, position_group_counts, week_grid_vals):
    """hs_version -- LKJ-correlated position intercept + age slope +
    opponent-strength AR(1) + an HSGP within-season recent-form curve.
    Statistically tied with ar_mod on held-out CRPS/RMSE; kept since it's
    not clearly worse and calibrates slightly better."""
    n_positions = len(coords['position'])
    n_seasons = len(coords['season'])
    coords = _extend_coords(coords, T, n_seasons, pos_param=True)
    coords.setdefault('form_basis', list(range(6)))

    with pm.Model(coords=coords) as model:
        dat = base_data(D, season_offsets)
        tenure_z_data = pm.Data('tenure_z_data', D['tenure_z'])
        intercepts = add_lkj_position_intercept(dat['position_data'], tenure_z_data)

        cont_priors = pm.Normal('cont_priors', 0, 1, dims='cont_vars')
        bi_priors = pm.Normal('bi_priors', 0, 1.5, dims='binary_vars')

        player_skill = add_player_skill(coords, position_of_player, position_group_counts, n_positions)
        f_career = player_skill[dat['player_data'], dat['tenure_data']]
        f_opp = add_strength_ar('opp', 'opp_team', dat['opp_team_data'], dat['global_time_data'], T, boundary_cols)
        f_form = add_recent_form_hsgp(week_grid_vals, dat['player_season_data'], dat['week_data'])

        mu = intercepts + pm.math.dot(dat['cont_data'], cont_priors) + pm.math.dot(dat['bi_data'], bi_priors) \
            + f_career + f_opp + f_form

        sigma_obs = pm.HalfNormal('sigma_obs', 5.0, dims='position')
        add_role_mixture(mu, sigma_obs, dat['position_data'],
                          dat['games_missed_data'], dat['snap_pct_data'], dat['obs_data'])
    return model


def build_team_mod(D, coords, season_offsets, T, boundary_cols,
                    position_of_player, position_group_counts):
    """team_mod -- hs_version's LKJ intercept + opponent-strength AR(1)
    (no recent-form term) + two regression coefficients (beta_team_form,
    beta_opp_team_form) on a precomputed team-strength feature (see
    src/estimate-latent-ability.py). Own-team and opponent-team strength
    are two separate coefficients, not a single differenced term -- there
    's no reason to assume a player's own offense helping them and the
    opponent's defense hurting them are equal-and-opposite effects on
    THEIR fantasy points specifically.

    Requires D['team_form_z'] / D['opp_team_form_z']."""
    n_positions = len(coords['position'])
    n_seasons = len(coords['season'])
    coords = _extend_coords(coords, T, n_seasons, pos_param=True)

    with pm.Model(coords=coords) as model:
        dat = base_data(D, season_offsets)
        tenure_z_data = pm.Data('tenure_z_data', D['tenure_z'])
        intercepts = add_lkj_position_intercept(dat['position_data'], tenure_z_data)

        cont_priors = pm.Normal('cont_priors', 0, 1, dims='cont_vars')
        bi_priors = pm.Normal('bi_priors', 0, 1.5, dims='binary_vars')

        player_skill = add_player_skill(coords, position_of_player, position_group_counts, n_positions)
        f_career = player_skill[dat['player_data'], dat['tenure_data']]
        f_opp = add_strength_ar('opp', 'opp_team', dat['opp_team_data'], dat['global_time_data'], T, boundary_cols)

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
# team_season_mod is NOT one of ff_ar.py's three currently active models
# (ar_mod/hs_version/team_mod) -- it's kept here only because
# add_team_season_ar (a leftover, unused-by-the-active-set helper still
# present in ff_ar.py) needs somewhere to live, and backtest_harness.py's
# build_team_season used it too. If ff_ar.py ever re-adds a
# team_season_mod, re-check this against its actual construction there
# rather than assuming this stale version is still right -- in particular
# it still uses the flat position intercept and a plain StudentT
# likelihood, not add_role_mixture.

def build_team_season_mod(D, coords, season_offsets, T, boundary_cols,
                           position_of_player, position_group_counts):
    """STALE -- see the module note above before using this."""
    n_positions = len(coords['position'])
    n_seasons = len(coords['season'])
    coords = _extend_coords(coords, T, n_seasons)

    with pm.Model(coords=coords) as model:
        dat = base_data(D, season_offsets)
        intercepts = add_flat_position_intercept(dat['position_data'])
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
