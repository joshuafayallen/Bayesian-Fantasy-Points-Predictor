
import arviz as az
import numpy as np
import polars as pl
import polars.selectors as cs
import pymc as pm
import matplotlib.pyplot as plt
import pytensor.tensor as pt
import pytensor

SEED = 41029041
np.random.seed(SEED)

DATA_PATH = 'processed-data/ff-processed.parquet'
LAST_SEASON = 2025


# ============================================================================
# Data prep -- shared by every model below
# ============================================================================

def fit_scalers(df, cols):
    stats = df.select(
        [pl.col(c).mean().alias(f"{c}__mu") for c in cols] +
        [pl.col(c).std().alias(f"{c}__sd") for c in cols]
    ).row(0, named=True)
    return {c: (stats[f"{c}__mu"], stats[f"{c}__sd"]) for c in cols}


def apply_scalers(frame, scalers):
    return frame.with_columns([
        ((pl.col(c) - mu) / sd).alias(f"{c}_z") for c, (mu, sd) in scalers.items()
    ])


def make_indices(df, cols):
    return {
        col: pl.Enum(df[col].unique().sort().cast(pl.String).to_list())
        for col in cols
    }


def encode(df, cols, enums):
    return df.with_columns([
        pl.col(col).cast(pl.String).cast(enums[col]).to_physical().cast(pl.Int32).alias(f"{col}_idx")
        for col in cols
    ])


raw_data = pl.read_parquet(DATA_PATH)

# team_form: filtered ("through week i") team-strength signal from the
# standalone Bradley-Terry-style state-space model in
# sandbox/team_strength_ssm.py -- a batched local-level Kalman filter over
# real score margins, fit separately and joined in here as a precomputed
# feature. Deliberately kept OUT of std_these/cont_vars below: that list
# feeds every model's shared cont_priors design matrix, but team_mod's
# whole point (see its module comment) is testing whether the player's own
# team's quality matters at all -- silently adding it to ar_mod/hs_version
# too would defeat that ablation. Left-joined + fill_null(0.0) rather than
# inner-joined so a schedule mismatch shows up as a printed warning instead
# of silently dropping rows.
team_form = pl.read_parquet('processed-data/team_form.parquet')
add_team_form = raw_data.select(pl.exclude("^.*_right$")).join(team_form, on=['season', 'week', 'team'], how='left')
n_missing_form = add_team_form['team_form'].null_count()
if n_missing_form:
    print(f"warning: {n_missing_form} rows have no team_form match -- filling with 0.0")
add_team_form = add_team_form.with_columns(pl.col('team_form').fill_null(0.0))
# same signal, keyed on the player's OPPONENT this week -- lets team_mod
# estimate separate coefficients for "my team's strength" vs "their
# defense's strength" instead of assuming the two effects are equal and
# opposite (which is what feeding in team_form - opp_team_form as a single
# differenced term would silently assume). Rename before the join since
# both sides of the join otherwise collide on 'team'/'team_form'.
opp_team_form = team_form.rename({'team': 'opp_team', 'team_form': 'opp_team_form'})
add_opp_form = add_team_form.join(opp_team_form, on=['season', 'week', 'opp_team'], how='left')
n_missing_opp_form = add_opp_form['opp_team_form'].null_count()
if n_missing_opp_form:
    print(f"warning: {n_missing_opp_form} rows have no opp_team_form match -- filling with 0.0")
add_opp_form = add_opp_form.with_columns(pl.col('opp_team_form').fill_null(0.0))


# fit on everything up to (not including) the last, still-in-progress week
# of LAST_SEASON. See sandbox/backtest_harness.py for genuine held-out
# accuracy across many such cutoffs -- this file's `train` is just the
# single "fit on everything we have" slice used for diagnostics.
lw = add_opp_form.filter(pl.col('season') == LAST_SEASON)['week'].max()
train = add_opp_form.filter(
    (pl.col('season') < LAST_SEASON) | ((pl.col('season') == LAST_SEASON) & (pl.col('week') < lw))
)

std_these = ['spread_line',
            'roll_avg_points_allowed',
            'lag_yards_per_rush_attempt',
            'lag_target_share',
            'lag_yards_per_target',
            'lag_yards_per_pass_attempt',
            'temp', 'wind', 'games_missed_ytd', 
            'lag_offense_pct', 'lag_st_pct']
binary_vars = ['is_grass', 'is_indoors', 'is_home', 'div_game', 'era']
idx_cols = ['player', 'position', 'team', 'week', 'tenure', 'player_season',
            'team_season', 'opp_team', 'season', 'games_missed_ytd']

scalers = fit_scalers(df=train, cols=std_these)
idxs = make_indices(raw_data, idx_cols)
train = apply_scalers(train, scalers)
train = encode(train, cols=idx_cols, enums=idxs)

# standardized separately from std_these/apply_scalers on purpose -- that
# helper always suffixes '_z', and cont_dat_train below sweeps up every
# '_z' column automatically. Naming this '_std' instead keeps it out of
# that sweep, so it only appears where a model explicitly asks for it
# (team_mod, via team_form_data below) rather than silently entering
# ar_mod/hs_version's shared design matrix too.
_team_form_mu = train['team_form'].mean()
_team_form_sd = train['team_form'].std()
train = train.with_columns(
    ((pl.col('team_form') - _team_form_mu) / _team_form_sd).alias('team_form_std'),
    # same scaler as team_form (not its own mean/sd) -- it's the same
    # underlying quantity just re-keyed on opp_team, so sharing the scaler
    # keeps beta_team_form and beta_opp_team_form directly comparable in
    # magnitude rather than each being in a slightly different unit.
    ((pl.col('opp_team_form') - _team_form_mu) / _team_form_sd).alias('opp_team_form_std'),
)

coords = {col: enum.categories.to_list() for col, enum in idxs.items()}

cont_dat_train = train.select(cs.ends_with('_z'))
coords['cont_vars'] = cont_dat_train.columns

binary_train = train.select(binary_vars)
coords['binary_vars'] = binary_train.columns

# only hs_version/team_mod's LKJ-correlated position intercept uses this
# dim, but it's harmless to have present in `coords` for models that don't
weeks_per_season = (
    raw_data.group_by('season').agg(pl.col('week').max().alias('max_week'))
    .sort('season')['max_week'].cast(pl.Int64).to_numpy()
)
n_seasons = len(coords['season'])
assert len(weeks_per_season) == n_seasons
season_offsets = np.concatenate([[0], np.cumsum(weeks_per_season)])
T = season_offsets[-1]
boundary_step_idx = season_offsets[1:-1]     # first real week of every season but the first
boundary_cols = boundary_step_idx - 1
season_offset_lookup = season_offsets[:-1]

coords['global_step'] = list(range(T - 1))
coords['global_step_full'] = list(range(T))
coords['season_gap'] = list(range(n_seasons - 1))
coords['pos_param'] = ['positions_mu', 'age_slope']
# add_recent_form_hsgp's shared basis dim -- m must match the `m=` passed
# there (default 6); kept as a plain range rather than derived from the GP
# object since coords has to exist before the model (and the GP) is built.
coords['form_basis'] = list(range(6))

global_time_vals = (
    season_offset_lookup[train['season_idx'].to_numpy().squeeze()]
    + train['week_idx'].to_numpy().squeeze()
).astype('int32')

# player -> position lookup, for the within-position centering of
# player_skill in add_player_skill() below: positions_mu[position] (or
# ab_position's intercept for the LKJ models) and the average starting
# level of the player_skill random walk within that position are
# otherwise redundant, which tanks the position-level parameters' ESS.
player_position_lookup = encode(raw_data, cols=['player', 'position'], enums=idxs)
player_position_map = (
    player_position_lookup.select(['player_idx', 'position_idx'])
    .group_by('player_idx')
    .agg(pl.col('position_idx').first())
    .sort('player_idx')['position_idx']
    .to_numpy()
)
n_positions = len(coords['position'])
assert len(player_position_map) == len(coords['player']), (
    f"player_position_map covers {len(player_position_map)} players, "
    f"expected {len(coords['player'])} -- check idx_cols/coords alignment"
)
position_group_counts = np.bincount(player_position_map, minlength=n_positions).astype('float64')

# standardized tenure, one value per row -- the LKJ models' linear
# age/tenure effect (varying by position) acts on this directly instead
# of a per-position HSGP curve.
tenure_grid_vals = np.array(coords['tenure'], dtype='float64')
tenure_z_row_vals = (
    train['tenure_idx'].to_numpy().astype('float64') - tenure_grid_vals.mean()
) / tenure_grid_vals.std()

# week-within-season grid for the recent_form components (add_recent_form_ar/
# add_recent_form_hsgp/add_team_var below) -- coords['week'] is already just
# the 18 distinct week-of-season values (see the week_data comment in
# common_data()), so this doubles as both the AR/VAR time axis and the HSGP
# GP input with no extra derivation.
week_grid_vals = np.array(coords['week'], dtype='float64')
n_week = len(week_grid_vals)


# ============================================================================
# Shared model-building blocks
# ============================================================================
# Every function below must be called inside an active `with pm.Model(...):`
# context. Factoring these out is what makes the four models below each ~30
# lines instead of ~150 -- before this cleanup, the pm.Data boilerplate,
# player_skill construction, and opponent-AR derivation were each
# copy-pasted 4-5 times and had already started drifting (team_mod was
# silently reusing hs_version's tenure_z_data pm.Data node across model
# contexts instead of creating its own).

def common_data():
    """pm.Data nodes needed by most models below. Not every model uses
    every key (team_season_mod doesn't need global_time_data, ar_mod/
    team_season_mod don't need teams_data as a *used* input even though
    it's created here) -- unused pm.Data nodes are harmless, and one
    definition here is much less likely to drift than nine copies of the
    same ten lines."""
    return {
        'teams_data': pm.Data('teams_data', train['team_idx'].to_numpy().squeeze()),
        'player_data': pm.Data('player_data', train['player_idx'].to_numpy().squeeze()),
        'position_data': pm.Data('position_data', train['position_idx'].to_numpy().squeeze()),
        'obs_data': pm.Data('obs_data', train['total_fantasy_points'].to_numpy()),
        'cont_data': pm.Data('cont_data', np.ascontiguousarray(cont_dat_train.to_numpy())),
        'bi_data': pm.Data('bi_data', np.ascontiguousarray(binary_train.to_numpy())),
        'tenure_data': pm.Data('tenure_data', train['tenure_idx'].to_numpy().squeeze()),
        'opp_team_data': pm.Data('opp_team_data', train['opp_team_idx'].to_numpy().squeeze()),
        'season_data': pm.Data('season_data', train['season_idx'].to_numpy().squeeze()),
        'global_time_data': pm.Data('global_time_data', global_time_vals),

        'week_data': pm.Data('week_data', train['week_idx'].to_numpy().squeeze()),
        'player_season_data': pm.Data('player_season_data', train['player_season_idx'].to_numpy().squeeze()),
        'games_missed_data': pm.Data('games_missed_data', train['games_missed_ytd_z'].to_numpy()),
        'snap_pct_data': pm.Data('snap_pct_data', train['lag_offense_pct_z'].to_numpy()),
        'team_form_data': pm.Data('team_form_data', train['team_form_std'].to_numpy()),
        'opp_team_form_data': pm.Data('opp_team_form_data', train['opp_team_form_std'].to_numpy()),
    }


def add_player_skill():
    """The random-walk-over-tenure player_skill term, hard-centered to zero
    mean within each position at every tenure level. See the comment on
    player_position_map above for why the centering matters."""
    n_tenure = len(coords['tenure'])
    init_sigma = pm.HalfNormal('init_sigma', 4.0)
    rw_sigma = pm.HalfNormal('rw_sigma', 0.5)
    z_innov = pm.Normal('z_innov', 0, 1, dims=('player', 'tenure'))
    level0 = init_sigma * z_innov[:, 0]
    steps = rw_sigma * z_innov[:, 1:]  # (player, n_tenure - 1)

    # player_skill_raw via pytensor.scan instead of
    # concatenate([level0[:, None], level0[:, None] + cumsum(steps, axis=1)])
    # -- this concatenate (combining level0 with a cumsum-of-a-Mul-derived
    # block) is structurally the same FGRAPH.REPLACE/local_add_of_sparse_write
    # trigger pattern chased elsewhere in this file (add_strength_ar,
    # add_recent_form_hsgp). Same fix: scan builds the recursion step-by-
    # step, so there's no concatenate-of-a-computed-block for pytensor's
    # canonicalizer to rewrite into a sparse write. Verified numerically
    # identical to the original cumsum-based construction in a standalone
    # numpy script.
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

    position_of_player = pm.Data('position_of_player', player_position_map, dims='player')

    # one-hot (position x player) matmul instead of
    # inc_subtensor(group_sum[position_of_player, :], player_skill_raw) --
    # that AdvancedIncSubtensor scatter-add runs in every model (add_player_
    # skill is shared) and is structurally the same pattern as the
    # local_add_of_sparse_write trigger chased in add_strength_ar: an
    # IncSubtensor immediately feeding an Add/divide a few lines later.
    # Switching add_strength_ar to scan did NOT clear the FGRAPH.REPLACE
    # crash on hs_version, which points here instead -- this was the one
    # remaining AdvancedIncSubtensor construction in the shared model code.
    # one_hot_position is a static 0/1 constant, known at graph-build time
    # (player_position_map never depends on pm.Data), so this is one dense
    # matmul, not a meaningfully different computation.
    one_hot_position = np.zeros((n_positions, len(player_position_map)))
    one_hot_position[player_position_map, np.arange(len(player_position_map))] = 1.0
    group_sum = pt.dot(one_hot_position, player_skill_raw)
    group_mean = group_sum / position_group_counts[:, None]

    return pm.Deterministic(
        'player_skill',
        player_skill_raw - group_mean[position_of_player, :],
        dims=('player', 'tenure'),
    )


def add_strength_ar(prefix, group_dim, group_idx_data, global_time_data):
    """Continuous non-centered AR(1)-with-season-jumps strength curve over
    `group_dim` (e.g. 'opp_team' or 'team') across the full weekly
    timeline. Used for opp_ar in every model, and was tried (and dropped)
    for a week-level team_ar -- see the module docstring."""
    rho          = pm.Beta(f'{prefix}_rho', 4, 2)
    sigma_theta  = pm.HalfNormal(f'{prefix}_sigma_theta', 1.0)
    sigma_season = pm.HalfNormal(f'{prefix}_sigma_season', 1.0)
    theta_init   = pm.Normal(f'{prefix}_theta_init', 0, 1, dims=group_dim)
    z_theta = pm.Normal(f'{prefix}_z_theta', 0, 1, dims=(group_dim, 'global_step'))
    z_eta   = pm.Normal(f'{prefix}_z_eta', 0, 1, dims=(group_dim, 'season_gap'))

    # boundary-column injection done as a dense scatter matmul instead of
    # innov[:, boundary_cols] += ... (pt.set_subtensor/AdvancedIncSubtensor
    # immediately followed by the Add a few lines below in theta_full's
    # concatenate) -- that op sequence is a known trigger for a pytensor
    # graph-rewrite bug ("BUG IN FGRAPH.REPLACE OR A LISTENER",
    # local_add_of_sparse_write) that surfaced once team_mod's larger graph
    # changed the set of rewrites that fire. scatter_mat is a fixed 0/1
    # constant (n_gaps x T-1), so this is one extra small matmul, not a
    # meaningfully different computation.
    n_gaps = len(boundary_cols)
    scatter_mat = np.zeros((n_gaps, T - 1))
    scatter_mat[np.arange(n_gaps), boundary_cols] = 1.0
    season_jump = sigma_season * pt.dot(z_eta, scatter_mat)  # (group_dim, T-1)
    innov = sigma_theta * z_theta + season_jump

    # theta_full via pytensor.scan instead of the closed-form rho-power-
    # matrix trick. The closed-form version's rho_powers is a large static
    # triangular (mostly-zero) constant, and pytensor's optimizer appears
    # to recognize that structure and rewrite the resulting tensordot into
    # an internal AdvancedIncSubtensor/sparse-write representation
    # (local_add_of_sparse_write) regardless of how the surrounding Python
    # code is written -- confirmed because three independently-verified,
    # subtensor-free rewrites of the surrounding code all hit the identical
    # (32, 226)-shaped FGRAPH.REPLACE crash. scan builds the recursion
    # step-by-step instead, so there's no static triangular constant in the
    # graph for that rewrite to misfire on. Slower to compile/sample than
    # the closed form, but correctness beats speed here.
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
    a season rather than reshuffling weekly. A contender's strength should
    carry over season-to-season; there's no reason to expect team identity
    (roster/scheme/coaching) to move week to week the way opp_ar's
    game-to-game defensive variation does. Uses pm.AR directly since there
    's no separate weekly-innovation/season-jump split needed here."""
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
    original within-season AR(1) over `week` (see git history / the
    module discussion this replaced). That version's clean, non-ablated
    fit found form_rho ~= 0.92 with form_sigma's posterior piled at the
    zero boundary (mean 0.074, hdi_3% == 0.0, ess ~150 and flat under
    nutpie despite 0 divergences) -- i.e. the data supports essentially no
    week-to-week evolution beyond a single persistent per-season shift.
    An ablation dropping the lag_* rate features made nutpie's geometry
    globally stiff (constant tree-depth-7 every draw, step_size collapsed
    to ~0.03, ESS near-zero for unrelated parameters too) rather than
    freeing up form_sigma -- evidence the AR-over-week structure was
    entangled with player_skill/opp_ar/the lag features over the same
    slice of variance, not that it was cleanly separable from them.

    Dropping the `week` dimension removes that non-identified AR degree of
    freedom (and the geometry cost it was imposing) while keeping the one
    piece of signal that was actually pinned down: a per-player_season
    constant wobble around player_skill's season-level average. If a
    future fit shows real leftover within-season structure once this
    settles, add `week` back in deliberately rather than by default."""
    form_sigma = pm.HalfNormal('form_sigma', 2.0)
    form_z = pm.Normal('form_z', 0, 1, dims='player_season')
    form = pm.Deterministic('recent_form', form_sigma * form_z, dims='player_season')
    return form[player_season_data]


def add_recent_form_hsgp(player_season_data, week_data, m=6, c=1.5):
    """HSGP analogue of add_recent_form_ar: same role (the within-season
    wobble around player_skill's season-level average), different
    generative story -- a smooth nonparametric curve instead of a
    first-order Markov process, so it can pick up e.g. a gradual ramp back
    from injury or a mid-season role change that AR(1)'s
    exponential-decay-from-shocks shape can't bend into as naturally.

    Hierarchical/shared-basis HSGP: the eigenbasis (and its lengthscale/
    amplitude hyperparameters) is shared across every player_season -- one
    GP, not 9500 independent ones -- but each player_season gets its own
    partially-pooled coefficients on that shared basis via `beta_raw`, so
    the *shape vocabulary* of plausible within-season curves is shared
    while amplitude/actual wiggle is player-specific. This is the standard
    trick for making HSGP tractable across many groups: with the basis
    fixed, a per-group curve is just a per-group linear combination, so
    9500 groups is 9500 more regression coefficients, not 9500 more GPs.

    """
    ell = pm.InverseGamma('form_ell', alpha=3, beta=6)  # weeks -- a few-week correlation range
    eta = pm.HalfNormal('form_eta', 1.0)
    cov = eta ** 2 * pm.gp.cov.Matern52(1, ls=ell)
    gp = pm.gp.HSGP(m=[m], c=c, cov_func=cov, drop_first=False)
    phi, sqrt_psd = gp.prior_linearized(X=week_grid_vals[:, None])  # phi: (week, m)

    beta_raw = pm.Normal('form_beta_raw', 0, 1, dims=('player_season', 'form_basis'))
    form_curve = pt.dot(beta_raw * sqrt_psd[None, :], phi.T)  # (player_season, week)

    # row mean via a matmul against a static ones-vector instead of
    # pt.mean(form_curve, axis=1, keepdims=True) -- this was the actual
    # FGRAPH.REPLACE/local_add_of_sparse_write trigger on hs_version's
    # ADVI/NUTS compile (confirmed via a (6724, 18) = (player_season, week)
    # -shaped crash whose node names, Add(AdvancedIncSubtensor.0,
    # ExpandDims{axis=1}.0)/Add(AdvancedIncSubtensor.0, Mul.0), match
    # pt.mean(keepdims=True)'s internal Sum -> Mul(1/n) -> ExpandDims graph
    # exactly. This is also why only hs_version crashed: it's the only one
    # of the three models with a per-group mean-centering step at all --
    # team_mod has no recent-form term and add_recent_form_ar doesn't
    # center. n_week is a small fixed constant (18), so this is one cheap
    # matmul, not a meaningfully different computation.
    n_week_form = len(week_grid_vals)
    row_mean = pt.dot(form_curve, pt.ones((n_week_form, 1))) / n_week_form  # (player_season, 1)
    form = pm.Deterministic(
        'recent_form',
        form_curve - row_mean,
        dims=('player_season', 'week'),
    )
    return form[player_season_data, week_data]




def add_flat_position_intercept(position_data):
    positions_mu = pm.Normal('positions_mu', 6, 3, dims='position')
    return pm.Deterministic('intercepts', positions_mu[position_data])


def add_lkj_position_intercept(position_data, eta=4):
    """Correlated varying effects across position: baseline scoring level
    and a linear tenure/age slope, drawn jointly per position via an
    LKJ-correlated covariance -- McElreath's "varying slopes" cafes
    pattern. Partially pools both quantities across positions and lets the
    model pick up genuine correlation between them (e.g. if
    higher-baseline positions also tend to decline faster/slower).

    Centered (not non-centered): tried non-centering through pos_chol_cov
    and it made ESS worse across the board. Each position has thousands
    of player-weeks pinning down its (intercept, slope) pair, so this
    isn't a data-starved-group funnel where non-centering helps -- it's
    stuck the other way."""
    tenure_z_data = pm.Data('tenure_z_data', tenure_z_row_vals)
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
    """Two-component mixture for the observation layer when a
    player barely plays, matchup quality is close to irrelevant to how
    little they score. 'full' is the existing linear predictor, unchanged.

    p_full's logit is now driven by two lagged, pre-game role signals
    instead of a flat per-position rate:
      - games_missed_data: has this player recently been absent (injury-
        return risk)
      - snap_pct_data: trailing snap share when they DID play -- catches
        healthy-but-limited-role cases (committee backs, WR4s) that
        games_missed_ytd can't see, since that player hasn't missed
        anything
    Both are already lagged in data prep -- never this week's actual snap
    count, which would leak the outcome into its own predictor."""
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
    # NOT wrapped in pm.Deterministic -- this is one value per row (88k+),
    # and the .nc files are already ~15GB; tracking a full per-row vector
    # every draw would make that meaningfully worse for a quantity you can
    # recompute cheaply post-hoc from role_intercept/role_beta_* if needed.
    p_full = pm.math.sigmoid(logit_p_full)
    w = pt.stack([1 - p_full, p_full], axis=-1)  # (n_obs, 2)

    comp_limited = pm.StudentT.dist(nu=4, mu=mu_limited[position_data], sigma=sigma_limited[position_data])
    comp_full = pm.StudentT.dist(nu=4, mu=mu_full, sigma=sigma_full[position_data])

    return pm.Mixture('y_obs', w=w, comp_dists=[comp_limited, comp_full], observed=obs_data)

def fit_and_diagnose(model, name, target_accept=0.95):
    """Standard workflow shared by all four models: prior predictive check
    -> NUTS -> posterior predictive + log-likelihood -> save to
    model-nc/ff_{name}.nc. Returns the idata. Deliberately does NOT call
    az.plot_ess_evolution here -- run those separately, after the fact,
    against the saved idata (model-nc/ff_{name}.nc) rather than inline with
    fitting, so a too-many-subplots error doesn't cost a refit."""
    with model:
        idata = pm.sample_prior_predictive(random_seed=SEED)
    az.plot_ppc_dist(idata, group='prior_predictive', visuals={'observed_dist': True})

    with model:
        idata.update(pm.sample(random_seed=SEED, target_accept=target_accept))

    with model:
        pm.compute_log_likelihood(idata)
        idata.update(pm.sample_posterior_predictive(idata))

    idata.to_netcdf(f'model-nc/ff_{name}.nc')
    return idata


def sample_check_coords(frame, col, n=5, seed=None):
    """n random category labels from `frame[col]`, for spot-checking ESS at
    specific coords (a handful of players/teams/etc.) rather than every
    one -- az.plot_ess_evolution over the full dim is unreadable and slow."""
    return frame[col].unique().sample(n, seed=seed).to_list()

# ============================================================================
# ar_mod -- flat per-position intercept + opponent-strength AR(1)
# ============================================================================

with pm.Model(coords=coords) as ar_mod:
    dat = common_data()
    intercepts = add_flat_position_intercept(dat['position_data'])
    cont_priors = pm.Normal('cont_priors', 0, 1, dims='cont_vars')
    bi_priors = pm.Normal('bi_priors', 0, 1.5, dims='binary_vars')

    player_skill = add_player_skill()
    f_career = player_skill[dat['player_data'], dat['tenure_data']]
    f_opp = add_strength_ar('opp', 'opp_team', dat['opp_team_data'], dat['global_time_data'])
    f_form = add_recent_form_ar(dat['player_season_data'])

    mu = intercepts + pm.math.dot(dat['cont_data'], cont_priors) + pm.math.dot(dat['bi_data'], bi_priors) \
        + f_career + f_opp + f_form

    pos = dat['position_data']
    sigma_obs = pm.HalfNormal('sigma_obs', 5.0, dims='position')
    add_role_mixture(mu, sigma_obs, dat['position_data'],
                dat['games_missed_data'], dat['snap_pct_data'], dat['obs_data'])

with ar_mod: 
    id_smoke = pm.sample(draws = 25, tune = 25)


idata_ar = fit_and_diagnose(ar_mod, 'ar')

az.plot_ess_evolution(
        idata_ar, var_names=[rv.name for rv in ar_mod.free_RVs if rv.size.eval() <= 10]
    )

check_players = sample_check_coords(train, 'player')
check_tenure = pl.from_records(coords['tenure']).sample(3)['column_0']
az.plot_ess_evolution(idata_ar, var_names=['player_skill'],
                       coords={'player': check_players, 'tenure': check_tenure})

check_opp = sample_check_coords(train, 'opp_team')
check_global_step = pl.from_records(coords['global_step_full']).sample(5)['column_0']
az.plot_ess_evolution(idata_ar, var_names=['opp_ar'],
                       coords={'opp_team': check_opp, 'global_step_full': check_global_step})
az.plot_ess_evolution(idata_ar, var_names=['opp_theta_init', 'opp_rho', 'opp_sigma_theta'])

check_player_seasons = sample_check_coords(train, 'player_season')
check_week = pl.from_records(coords['week']).sample(4)['column_0']  # still used by hs_version/team_mod below
az.plot_ess_evolution(idata_ar, var_names=['recent_form'],
                       coords={'player_season': check_player_seasons})
az.plot_ess_evolution(idata_ar, var_names=['form_sigma'])

az.plot_ppc_dist(idata_ar, kind='ecdf')
az.plot_ppc_dist(idata_ar, num_samples=20)
az.plot_energy(idata_ar)



# ============================================================================
# hs_version -- LKJ-correlated position intercept + age slope + opp_ar
# ============================================================================
# Statistically tied with ar_mod on held-out CRPS/RMSE (see module
# docstring). Kept since it's not clearly worse and calibrates slightly
# better; not the model to invest further complexity in until team_mod /
# team_season_mod settle whether own-team quality is worth adding.

with pm.Model(coords=coords) as hs_version:
    dat = common_data()
    intercepts = add_lkj_position_intercept(dat['position_data'])
    cont_priors = pm.Normal('cont_priors', 0, 1, dims='cont_vars')
    bi_priors = pm.Normal('bi_priors', 0, 1.5, dims='binary_vars')

    player_skill = add_player_skill()
    f_career = player_skill[dat['player_data'], dat['tenure_data']]
    f_opp = add_strength_ar('opp', 'opp_team', dat['opp_team_data'], dat['global_time_data'])
    f_form = add_recent_form_hsgp(dat['player_season_data'], dat['week_data'])

    mu = intercepts + pm.math.dot(dat['cont_data'], cont_priors) + pm.math.dot(dat['bi_data'], bi_priors) \
        + f_career + f_opp + f_form
    pos = dat['position_data']
    sigma_obs = pm.HalfNormal('sigma_obs', 5.0, dims='position')
    
    add_role_mixture(mu, sigma_obs, dat['position_data'],
                dat['games_missed_data'], dat['snap_pct_data'], dat['obs_data'])

with hs_version: 
    id_smoke = pm.sample(tune = 25, draws=25)


id_hs = fit_and_diagnose(hs_version, 'hs')

az.rcParams['plot.max_subplots'] = 50


az.plot_ess_evolution(
        id_hs, var_names=[rv.name for rv in hs_version.free_RVs if rv.size.eval() <= 10]
    )

check_players = sample_check_coords(train, 'player')
check_tenure = pl.from_records(coords['tenure']).sample(3)['column_0']

az.plot_ess_evolution(
        id_hs, var_names=[rv.name for rv in hs_version.free_RVs if rv.size.eval() <= 10]
    )
az.plot_ess_evolution(id_hs, var_names=['player_skill'],
                       coords={'player': check_players, 'tenure': check_tenure})
az.plot_ess_evolution(id_hs, var_names=['ab_position'])

az.plot_ess_evolution(id_hs, var_names=['recent_form'],
                       coords={'player_season': check_player_seasons, 'week': check_week})
az.plot_ess_evolution(id_hs, var_names=['form_ell', 'form_eta'])


# ============================================================================
# team_mod -- ar_mod + a partially-pooled, time-FLAT per-team offensive-
# environment effect. See module docstring for why: neither ar_mod nor
# hs_version ever use `team_idx` (the player's own team) in mu -- only the
# OPPONENT's defensive quality is modeled (roll_avg_points_allowed_z,
# opp_ar). This is the simplest possible test of whether the player's own
# team's quality matters at all.
#
# Uses the LKJ position-intercept structure (same as hs_version) purely
# because that's what was already built when this model was added; the
# team-quality question and the position-intercept-structure question are
# independent, so this isn't an endorsement of LKJ over flat.
# ============================================================================

with pm.Model(coords=coords) as team_mod:
    dat = common_data()
    intercepts = add_lkj_position_intercept(dat['position_data'])
    cont_priors = pm.Normal('cont_priors', 0, 1, dims='cont_vars')
    bi_priors = pm.Normal('bi_priors', 0, 1.5, dims='binary_vars')

    player_skill = add_player_skill()
    f_career = player_skill[dat['player_data'], dat['tenure_data']]
    f_opp = add_strength_ar('opp', 'opp_team', dat['opp_team_data'], dat['global_time_data'])


    # own-team and opponent-team strength as two separate coefficients,
    # not a single differenced term -- there's no reason to assume a
    # player's own offense helping them and the opponent's defense
    # hurting them are equal-and-opposite effects on THEIR fantasy points
    # specifically (e.g. a good offense inflating volume/game-script
    # value might matter more than a good opposing defense suppressing
    # it). Both are fixed pm.Data inputs, so this costs one extra cheap
    # parameter, not any extra sampling risk.
    beta_team_form = pm.Normal('beta_team_form', 0, 1)
    beta_opp_team_form = pm.Normal('beta_opp_team_form', 0, 1)
    f_team_form = beta_team_form * dat['team_form_data'] + beta_opp_team_form * dat['opp_team_form_data']

    mu = intercepts + pm.math.dot(dat['cont_data'], cont_priors) + pm.math.dot(dat['bi_data'], bi_priors) \
        + f_career + f_opp + f_team_form

    pos = dat['position_data']
    sigma_obs = pm.HalfNormal('sigma_obs', 5.0, dims='position')
    add_role_mixture(mu, sigma_obs, dat['position_data'],
                dat['games_missed_data'], dat['snap_pct_data'], dat['obs_data'])


with team_mod: 
    id_smoke_test = pm.sample(tune = 25, draws = 25)

id_team = fit_and_diagnose(team_mod, 'team')

id_team.sample_stats['diverging'].sum().item()

az.rcParams['plot.max_subplots'] = 60

az.plot_ess_evolution(
    id_team,
    var_names=[rv.name for rv in team_mod.free_RVs if rv.size.eval() <= 10]
)
az.plot_ess_evolution(id_team, var_names=['beta_team_form', 'beta_opp_team_form'])



# ============================================================================
# Model comparison
# ============================================================================
idata_ar = az.from_netcdf('model-nc/ff_ar.nc')
id_hs = az.from_netcdf('model-nc/ff_hs.nc')
id_team = az.from_netcdf('model-nc/ff_team.nc')



## one run of the comparisions had observations in the log likelihood that prevented comparision 
for name, idata in {'ar_mod': idata_ar, 'hs_version': id_hs, 'team_mod': id_team}.items():
    ll = idata.log_likelihood['y_obs'].values
    n_nan = np.isnan(ll).sum()
    n_inf = np.isinf(ll).sum()
    print(f"{name}: shape={ll.shape} nan={n_nan} inf={n_inf}")
    if n_nan or n_inf:
        bad_obs = np.where(np.isnan(ll).any(axis=(0, 1)) | np.isinf(ll).any(axis=(0, 1)))[0]
        print(f"  {len(bad_obs)} distinct observations affected, e.g. indices: {bad_obs[:10]}")

az.compare({
    'ar_mod': idata_ar,
    'hs_version': id_hs,
    'team_mod': id_team,

})

for name, idata in {'ar_mod': idata_ar, 'hs_version': id_hs,
                     'team_mod': id_team, }.items():
    print(f"\n{name}")
    print(az.loo(idata, pointwise=True))
