from models import base_data
import arviz as az
import numpy as np
import polars as pl
import polars.selectors as cs
import pymc as pm
import pymc_extras as pmx
import preliz as pz
import pytensor.tensor as pt
import pytensor

import models

SEED = 41029041
np.random.seed(SEED)

DATA_PATH = 'processed-data/ff-processed.parquet'
LAST_SEASON = 2025


# ============================================================================
# Data prep -- full-history-specific (one "fold" covering everything up to
# the current in-progress week of LAST_SEASON). Uses src/models.py's
# generic scaler/encoder helpers so this prep is provably the same
# arithmetic backtest_harness.py's per-fold prep uses, just run once over
# all of history instead of once per fold.
# ============================================================================

raw_data = pl.read_parquet(DATA_PATH)

# team_form: filtered ("through week i") team-strength signal from the
# standalone Bradley-Terry-style state-space model in
# sandbox/team_strength_ssm.py -- a batched local-level Kalman filter over
# real score margins, fit separately and joined in here as a precomputed
# feature. Deliberately kept OUT of std_these/cont_vars below: that list
# feeds every model's shared cont_priors design matrix, but team_mod's
# whole point (see src/models.py's build_team_mod docstring) is testing
# whether the player's own team's quality matters at all -- silently
# adding it to ar_mod/hs_version too would defeat that ablation.
# Left-joined + fill_null(0.0) rather than inner-joined so a schedule
# mismatch shows up as a printed warning instead of silently dropping rows.
team_form = pl.read_parquet('processed-data/team_form.parquet')
add_team_form = raw_data.select(pl.exclude("^.*_right$")).join(team_form, on=['season', 'week', 'team'], how='left')
n_missing_form = add_team_form['team_form'].null_count()
if n_missing_form:
    print(f"warning: {n_missing_form} rows have no team_form match -- filling with 0.0")
add_team_form = add_team_form.with_columns(pl.col('team_form').fill_null(0.0))
# same signal, keyed on the player's OPPONENT this week -- lets team_mod
# estimate separate coefficients for "my team's strength" vs "their
# defense's strength" instead of assuming the two effects are equal and
# opposite. Rename before the join since both sides otherwise collide on
# 'team'/'team_form'.
opp_team_form = team_form.rename({'team': 'opp_team', 'team_form': 'opp_team_form'})
add_opp_form = add_team_form.join(opp_team_form, on=['season', 'week', 'opp_team'], how='left')
n_missing_opp_form = add_opp_form['opp_team_form'].null_count()
if n_missing_opp_form:
    print(f"warning: {n_missing_opp_form} rows have no opp_team_form match -- filling with 0.0")
add_opp_form = add_opp_form.with_columns(pl.col('opp_team_form').fill_null(0.0))

shrunk_estimates = pl.read_csv('processed-data/shrinkage_features.csv').select('player_id', 'season','week', cs.ends_with('shrunk'))
join_shrink = (
    add_opp_form
    .join(shrunk_estimates, on = ['player_id', 'season', 'week'])
)

# fit on everything up to (not including) the last, still-in-progress week
# of LAST_SEASON. See src/backtest_harness.py for genuine held-out
# accuracy across many such cutoffs -- this file's `train` is just the
# single "fit on everything we have" slice used for diagnostics.
lw = join_shrink.filter(pl.col('season') == LAST_SEASON)['week'].max()
train = join_shrink.filter(
    (pl.col('season') < LAST_SEASON) | ((pl.col('season') == LAST_SEASON) & (pl.col('week') < lw))
)


std_these = ['spread_line',
            'roll_avg_points_allowed',
            'lag_yards_per_rush_attempt_shrunk',
            'lag_yards_per_target_shrunk',
            'lag_yards_per_pass_attempt_shrunk',
            'lag_target_share',
            'temp', 'wind', 'games_missed_ytd',
            'lag_offense_pct', 'lag_st_pct']
binary_vars = ['is_grass', 'is_indoors', 'is_home', 'div_game', 'era']
idx_cols = ['player', 'position', 'team', 'week', 'tenure', 'player_season',
            'team_season', 'opp_team', 'season', 'games_missed_ytd']

scalers = models.fit_scalers(df=train, cols=std_these)
idxs = models.make_indices(raw_data, idx_cols)
train = models.apply_scalers(train, scalers)
train = models.encode(train, cols=idx_cols, enums=idxs)

# standardized separately from std_these/apply_scalers on purpose -- that
# helper always suffixes '_z', and cont_dat_train below sweeps up every
# '_z' column automatically. Naming this '_std' instead keeps it out of
# that sweep, so it only appears where a model explicitly asks for it
# (team_mod, via D['team_form_z']/D['opp_team_form_z'] below) rather than
# silently entering ar_mod/hs_version's shared design matrix too.
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

weeks_per_season = (
    raw_data.group_by('season').agg(pl.col('week').max().alias('max_week'))
    .sort('season')['max_week'].cast(pl.Int64).to_numpy()
)
n_seasons = len(coords['season'])
assert len(weeks_per_season) == n_seasons
season_offsets, T, boundary_cols = models.build_ar_calendar(weeks_per_season)

# player -> position lookup, for the within-position centering of
# player_skill in models.add_player_skill(): positions_mu[position] (or
# ab_position's intercept for the LKJ models) and the average starting
# level of the player_skill random walk within that position are
# otherwise redundant, which tanks the position-level parameters' ESS.
n_positions = len(coords['position'])
position_of_player, position_group_counts = models.player_position_map(
    raw_data, 'player', 'position', idxs, n_positions
)
assert len(position_of_player) == len(coords['player']), (
    f"position_of_player covers {len(position_of_player)} players, "
    f"expected {len(coords['player'])} -- check idx_cols/coords alignment"
)

# standardized tenure, one value per row -- the LKJ models' linear
# age/tenure effect (varying by position) acts on this directly instead
# of a per-position HSGP curve.
tenure_grid_vals = np.array(coords['tenure'], dtype='float64')
tenure_z_row_vals = (
    train['tenure_idx'].to_numpy().astype('float64') - tenure_grid_vals.mean()
) / tenure_grid_vals.std()

# week-within-season grid for the recent_form components -- coords['week']
# is already just the 18 distinct week-of-season values, so this doubles
# as both the AR/VAR time axis and the HSGP GP input with no extra
# derivation.
week_grid_vals = np.array(coords['week'], dtype='float64')

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
    team_form_z=train['team_form_std'].to_numpy(),
    opp_team_form_z=train['opp_team_form_std'].to_numpy(),
)


def sample_check_coords(frame, col, n=5, seed=None):
    """n random category labels from `frame[col]`, for spot-checking ESS at
    specific coords (a handful of players/teams/etc.) rather than every
    one -- az.plot_ess_evolution over the full dim is unreadable and slow."""
    return frame[col].unique().sample(n, seed=seed).to_list()


def fit_and_diagnose(model, name, target_accept=0.99, prior_check_only=True, samps = 50, **sample_kwargs):
    """Standard workflow shared by all three models: prior predictive check
    -> NUTS -> posterior predictive + log-likelihood -> save to
    model-nc/ff_{name}.nc. Returns the idata. Deliberately does NOT call
    az.plot_ess_evolution here -- run those separately, after the fact,
    against the saved idata (model-nc/ff_{name}.nc) rather than inline with
    fitting, so a too-many-subplots error doesn't cost a refit.

    prior_check_only=True stops right after the prior predictive check and
    returns (idata, ppc_fig) WITHOUT running NUTS -- use this to look at
    the prior predictive plot (and hang onto the figure object -- save it,
    close it, whatever) before committing to a real, possibly slow,
    sample() call. Same spirit as the ESS-evolution note above, just at
    the other end of the function. Default (False) is unchanged from
    before: single idata returned, NUTS + posterior predictive always run,
    existing call sites (id_ar = fit_and_diagnose(...)) don't need to
    change."""
    with model:
        idata = pm.sample_prior_predictive(random_seed=SEED)
    ppc_fig = az.plot_ppc_dist(idata, group='prior_predictive', visuals={'observed_dist': True}, num_samples=samps)
    if prior_check_only:
        return idata, ppc_fig

    with model:
        idata.update(pm.sample(random_seed=SEED, target_accept=target_accept, **sample_kwargs))

    with model:
        pm.compute_log_likelihood(idata)
        idata.update(pm.sample_posterior_predictive(idata))

    idata.to_netcdf(f'model-nc/ff_{name}.nc')
    return idata



pz.Beta(mu = 0.37, nu = 35).plot_pdf()

def add_player_skill(position_of_player, total_scale=None):
    n_tenure = len(coords['tenure'])
    if total_scale is None:
        init_sigma = pm.HalfNormal('init_sigma', 4.0)
        rw_sigma = pm.HalfNormal('rw_sigma', 0.5)
    else:
        career_shape = pm.Beta('career_shape', 2, 2)
        init_sigma = total_scale * pt.sqrt(career_shape)
        rw_sigma = total_scale * pt.sqrt(1 - career_shape)
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
def add_strength_ar(prefix, group_dim, group_idx_data, global_time_data, total_scale):
    rho = pm.Beta(f'{prefix}_rho', 4, 2)
    ar_shape = pm.Beta(f'{prefix}_shape', 2, 2)     # within-season innovation vs. season-jump
    sigma_theta  = total_scale * pt.sqrt(ar_shape)
    sigma_season = total_scale * pt.sqrt(1 - ar_shape)
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

#coords['variance_block'] = ['linear', 'context', 'team_form']
coords['linear_vars'] = list(cont_dat_train.columns) + list(binary_train.columns)
coords['team_form_vars'] = ['team_form', 'opp_team_form']
coords['global_step'] = list(range(T - 1))
coords['global_step_full'] = list(range(T))
coords['season_gap'] = list(range(n_seasons - 1))
coords['pos_param'] = ['positions_mu', 'age_slope']

with pm.Model(coords=coords) as r2d2:
    dat = base_data(D, season_offsets)
    tenure_z_data = pm.Data('tenure_z_data', D['tenure_z'])
    intercepts = models.add_lkj_position_intercept(dat['position_data'], tenure_z_data, eta=4)

    # Independent scale priors per block, centered on the estimates that
    # already converged cleanly (r_hat=1.00) every time each block's own
    # scale was estimated in isolation, before being forced to compete for
    # a shared, hard-constrained R2 budget -- see this file's history for
    # why the shared-budget version was abandoned (a real, persistent
    # ridge that just relocated between whichever two blocks were left
    # sharing a Dirichlet, not a fixable parameterization bug).
    linear_scale    = pm.Gamma('linear_scale', mu=0.55, sigma=0.15)
    career_scale    = pm.Gamma('career_scale', mu=2.6, sigma=0.5)
    opp_scale       = pm.Gamma('opp_scale', mu=1.8, sigma=0.4)
    team_form_scale = pm.Gamma('team_form_scale', mu=1.8, sigma=0.4)

    # linear block: R2D2M2CP still does its own internal decomposition
    # across these columns -- the one part of this machinery that's
    # actually earned its complexity, clean r_hat/ESS on every run.
    design = pt.concatenate([dat['cont_data'], dat['bi_data']], axis=1)
    input_sigma = np.concatenate([
        cont_dat_train.std().to_numpy().flatten(),
        binary_train.std().to_numpy().flatten(),
    ])
    _, linear_beta = pmx.R2D2M2CP(
        'linear_effects', output_sigma=linear_scale, input_sigma=input_sigma,
        dims='linear_vars', r2=0.5, r2_std=0.2,
    )
    f_linear = pt.dot(design, linear_beta)

    player_skill = add_player_skill(position_of_player, total_scale=career_scale)
    f_career = player_skill[dat['player_data'], dat['tenure_data']]

    f_opp = add_strength_ar('opp', 'opp_team', dat['opp_team_data'],
                            dat['global_time_data'], total_scale=opp_scale)

    team_form_data = pm.Data('team_form_data', D['team_form_z'])
    opp_form_data = pm.Data('opp_form_data', D['opp_team_form_z'])
    tf_design = pt.stack([team_form_data, opp_form_data], axis=-1)
    _, team_form_beta = pmx.R2D2M2CP(
        'team_form_effects', output_sigma=team_form_scale,
        input_sigma=np.array([1.0, 1.0]), dims='team_form_vars',
        r2=0.5, r2_std=0.2,
    )
    f_team_form = pt.dot(tf_design, team_form_beta)

    mu = intercepts + f_linear + f_career + f_opp + f_team_form

    sigma_obs = pm.HalfNormal('sigma_obs', 2.5, dims='position')
    models.add_role_mixture(mu, sigma_obs, dat['position_data'],
                             dat['games_missed_data'], dat['snap_pct_data'], dat['obs_data'])


id_rd = fit_and_diagnose(r2d2, 'id_rd', samps = 50)

id_rd = fit_and_diagnose(r2d2, 'id_rd', prior_check_only=False, target_accept=0.95)

az.rcParams['plot.max_subplots'] =60

az.plot_ess_evolution(
    id_rd ,
    var_names=[rv.name for rv in r2d2.free_RVs if rv.size.eval() <= 10]
)


az.plot_ess_evolution(
    id_rd, 
    var_names=['linear_scale', 'career_scale', 'opp_scale', 'team_form_scale']
)

az.summary(id_rd, var_names=['linear_scale', 'career_scale', 'opp_scale', 'team_form_scale'])[['mean','sd','ess_bulk','ess_tail','r_hat']]

az.plot_ppc_dist(
    id_rd, num_samples=100
)


