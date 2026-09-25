import pandas as pd
from dataclasses import dataclass, field, replace
 
import arviz as az
import numpy as np
import polars as pl
import polars.selectors as cs
import pymc as pm
import pymc_extras as pmx
import pytensor
import pytensor.tensor as pt
 
import models
 
SEED = 41029041
np.random.seed(SEED)
 
DATA_PATH = 'processed-data/ff-processed.parquet'
TEAM_FORM_PATH = 'processed-data/team_form.parquet'
SHRINKAGE_PATH = 'processed-data/shrinkage_features.csv'
LAST_SEASON = 2025
 
 
# ============================================================================
# Data prep -- same shape as ff_ar.py's, since this is diagnosing the SAME
# team_mod/r2d2_mod full-history fit, not a separate fold. Kept flat and
# top-level rather than wrapped in a loader function, to match.
# ============================================================================
 
raw_data = pl.read_parquet(DATA_PATH)
 
team_form = pl.read_parquet(TEAM_FORM_PATH)
add_team_form = raw_data.select(pl.exclude("^.*_right$")).join(team_form, on=['season', 'week', 'team'], how='left')
n_missing_form = add_team_form['team_form'].null_count()
if n_missing_form:
    print(f"warning: {n_missing_form} rows have no team_form match -- filling with 0.0")
add_team_form = add_team_form.with_columns(pl.col('team_form').fill_null(0.0))
 
opp_team_form = team_form.rename({'team': 'opp_team', 'team_form': 'opp_team_form'})
add_opp_form = add_team_form.join(opp_team_form, on=['season', 'week', 'opp_team'], how='left')
n_missing_opp_form = add_opp_form['opp_team_form'].null_count()
if n_missing_opp_form:
    print(f"warning: {n_missing_opp_form} rows have no opp_team_form match -- filling with 0.0")
add_opp_form = add_opp_form.with_columns(pl.col('opp_team_form').fill_null(0.0))
 
shrunk_estimates = pl.read_csv(SHRINKAGE_PATH).select('player_id', 'season', 'week', cs.ends_with('shrunk'))
join_shrink = add_opp_form.join(shrunk_estimates, on=['player_id', 'season', 'week'])
 
# fit on everything up to (not including) the last, still-in-progress week
# of LAST_SEASON -- same cutoff convention as ff_ar.py's `train`, so
# whatever az.psense flags here is about the SAME data ff_ar.py fits on,
# not an artifact of a different slice.
lw = join_shrink.filter(pl.col('season') == LAST_SEASON)['week'].max()
train = join_shrink.filter(
    (pl.col('season') < LAST_SEASON) | ((pl.col('season') == LAST_SEASON) & (pl.col('week') < lw))
)
 
std_these = ['spread_line',
             'roll_avg_points_allowed',
             'lag_yards_per_rush_attempt_shrunk',
             'lag_yards_per_target_shrunk',
             'lag_yards_per_pass_attempt_shrunk',
             'lag_avg_depth_of_target_shrunk',
             'lag_target_share',
             'temp', 'wind', 'games_missed_ytd',
             'lag_offense_pct', 'lag_st_pct']
binary_vars = ['is_grass', 'is_indoors', 'is_home', 'div_game', 'era']
idx_cols = ['player_id', 'position', 'team', 'week', 'tenure', 'player_season',
            'team_season', 'opp_team', 'season', 'games_missed_ytd']
 
scalers = models.fit_scalers(df=train, cols=std_these)
idxs = models.make_indices(raw_data, idx_cols)
train = models.apply_scalers(train, scalers)
train = models.encode(train, cols=idx_cols, enums=idxs)
 
_team_form_mu = train['team_form'].mean()
_team_form_sd = train['team_form'].std()
train = train.with_columns(
    ((pl.col('team_form') - _team_form_mu) / _team_form_sd).alias('team_form_std'),
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
 
n_positions = len(coords['position'])
position_of_player, position_group_counts = models.player_position_map(
    raw_data, 'player_id', 'position', idxs, n_positions
)
assert len(position_of_player) == len(coords['player_id']), (
    f"position_of_player covers {len(position_of_player)} players, "
    f"expected {len(coords['player_id'])} -- check idx_cols/coords alignment"
)
 
tenure_grid_vals = np.array(coords['tenure'], dtype='float64')
tenure_z_row_vals = (
    train['tenure_idx'].to_numpy().astype('float64') - tenure_grid_vals.mean()
) / tenure_grid_vals.std()
 
D = dict(
    player=train['player_id_idx'].to_numpy().squeeze(),
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
 
 
# ============================================================================
# Config -- every prior block az.psense flagged (player skill, team_form
# effects, opp_ar, varying position intercept/slope) is a DebugConfig
# field, so a run changes exactly one knob instead of hand-editing
# models.build_team_mod/build_r2d2_mod per test.
# ============================================================================
 
@dataclass
class DebugConfig:
    name: str
 
    use_mixture: bool = True
    sigma_obs_scale: float = 2.5
    sigma_obs_mode: str = "per_position"      # "per_position" | "pooled"
 
    intercept_mode: str = "lkj"               # "lkj" | "flat"
    lkj_eta: float = 2.0
    lkj_sd: tuple = (1.0, 3.0)
    lkj_intercept_mu: float = 6.0
    lkj_intercept_sigma: float = 3.0
    lkj_slope_mu: float = 0.0
    lkj_slope_sigma: float = 1.0
    flat_intercept_sigma: float = 3.0
    flat_slope_sigma: float = 1.0
 
    linear_mode: str = "r2d2"                 # "independent" | "r2d2"
    linear_normal_sigma_cont: float = 1.0
    linear_normal_sigma_bi: float = 1.5
    linear_r2: float = 0.35
    linear_r2_std: float = 0.08
 
    team_form_mode: str = "r2d2"              # "independent" | "r2d2"
    team_form_normal_sigma: float = 1.0
    team_form_r2: float = 0.05
    team_form_r2_std: float = 0.02
 
    player_skill_mode: str = "shared_budget"  # "independent" | "shared_budget"
    career_scale_kwargs: dict = field(default_factory=lambda: dict(alpha=2, beta=1))
    player_skill_init_sigma: float = 4.0
    player_skill_rw_sigma: float = 0.5
    player_skill_innovation: str = "studentt"  # "normal" | "studentt"
    player_skill_nu: float = 5.0
 
    opp_ar_mode: str = "shared_budget"        # "independent" | "shared_budget"
    opp_scale_kwargs: dict = field(default_factory=lambda: dict(mu=1.8, sigma=1.0))
    opp_sigma_theta: float = 1.0
    opp_sigma_season: float = 1.0
    opp_rho_ab: tuple = (2.0, 2.0)
 
 
# ============================================================================
# Parameterized blocks -- math copied verbatim from models.py's
# add_player_skill/add_strength_ar/add_lkj_position_intercept (the scan
# recursions and one-hot-matmul tricks are load-bearing, see those
# docstrings for why), only the prior specifications are exposed as
# DebugConfig fields.
# ============================================================================
 
def _player_skill(coords, position_of_player, position_group_counts, n_positions, cfg):
    if cfg.player_skill_mode == "independent":
        init_sigma = pm.HalfNormal('init_sigma', cfg.player_skill_init_sigma)
        rw_sigma = pm.HalfNormal('rw_sigma', cfg.player_skill_rw_sigma)
    elif cfg.player_skill_mode == "shared_budget":
        career_scale = pm.Gamma('career_scale', **cfg.career_scale_kwargs)
        career_shape = pm.Beta('career_shape', 2, 2)
        init_sigma = career_scale * pt.sqrt(career_shape)
        rw_sigma = career_scale * pt.sqrt(1 - career_shape)
    else:
        raise ValueError(f"unknown player_skill_mode: {cfg.player_skill_mode!r}")
 
    if cfg.player_skill_innovation == "normal":
        z_innov = pm.Normal('z_innov', 0, 1, dims=('player_id', 'tenure'))
    elif cfg.player_skill_innovation == "studentt":
        z_innov = pm.StudentT('z_innov', nu=cfg.player_skill_nu, mu=0, sigma=1,
                               dims=('player_id', 'tenure'))
    else:
        raise ValueError(f"unknown player_skill_innovation: {cfg.player_skill_innovation!r}")
 
    level0 = init_sigma * z_innov[:, 0]
    steps = rw_sigma * z_innov[:, 1:]
 
    def _tenure_step(step_t, prev):
        return prev + step_t
 
    steps_seq = steps.T
    path, _ = pytensor.scan(fn=_tenure_step, sequences=[steps_seq],
                             outputs_info=[level0], strict=True)
    player_skill_raw = pt.concatenate([level0[None, :], path], axis=0).T
 
    position_of_player_data = pm.Data('position_of_player', position_of_player, dims='player_id')
 
    one_hot_position = np.zeros((n_positions, len(position_of_player)))
    one_hot_position[position_of_player, np.arange(len(position_of_player))] = 1.0
    group_sum = pt.dot(one_hot_position, player_skill_raw)
    group_mean = group_sum / position_group_counts[:, None]
 
    return pm.Deterministic(
        'player_skill',
        player_skill_raw - group_mean[position_of_player_data, :],
        dims=('player_id', 'tenure'),
    )
 
 
def _opp_ar(prefix, group_dim, group_idx_data, global_time_data, T, boundary_cols, cfg):
    rho = pm.Beta(f'{prefix}_rho', *cfg.opp_rho_ab)
    if cfg.opp_ar_mode == "independent":
        sigma_theta = pm.HalfNormal(f'{prefix}_sigma_theta', cfg.opp_sigma_theta)
        sigma_season = pm.HalfNormal(f'{prefix}_sigma_season', cfg.opp_sigma_season)
    elif cfg.opp_ar_mode == "shared_budget":
        opp_total = pm.Gamma(f'{prefix}_total_scale', **cfg.opp_scale_kwargs)
        ar_shape = pm.Beta(f'{prefix}_shape', 2, 2)
        sigma_theta = opp_total * pt.sqrt(ar_shape)
        sigma_season = opp_total * pt.sqrt(1 - ar_shape)
    else:
        raise ValueError(f"unknown opp_ar_mode: {cfg.opp_ar_mode!r}")
 
    theta_init = pm.Normal(f'{prefix}_theta_init', 0, 1, dims=group_dim)
    z_theta = pm.Normal(f'{prefix}_z_theta', 0, 1, dims=(group_dim, 'global_step'))
    z_eta = pm.Normal(f'{prefix}_z_eta', 0, 1, dims=(group_dim, 'season_gap'))
 
    if len(boundary_cols):
        n_gaps = len(boundary_cols)
        scatter_mat = np.zeros((n_gaps, T - 1))
        scatter_mat[np.arange(n_gaps), boundary_cols] = 1.0
        season_jump = sigma_season * pt.dot(z_eta, scatter_mat)
        innov = sigma_theta * z_theta + season_jump
    else:
        innov = sigma_theta * z_theta
 
    def _ar_step(innov_t, theta_prev, rho):
        return rho * theta_prev + innov_t
 
    innov_seq = innov.T
    theta_path, _ = pytensor.scan(fn=_ar_step, sequences=[innov_seq],
                                   outputs_info=[theta_init], non_sequences=[rho], strict=True)
    theta_full = pt.concatenate([theta_init[None, :], theta_path], axis=0).T
 
    strength = pm.Deterministic(
        f'{prefix}_ar', theta_full - pt.mean(theta_full, axis=0, keepdims=True),
        dims=(group_dim, 'global_step_full'),
    )
    return strength[group_idx_data, global_time_data]
 
 
def _lkj_intercept(position_data, tenure_z_data, cfg):
    """Pooled, non-centered. Keeping the pooling deliberately -- position-
    level covariance between intercept and age slope is a real modeling
    target here (RB/WR are expected to show an age-cliff, a steep negative
    slope, while TE shows a longer/flatter learning curve; letting those
    covary across positions rather than fitting them independently is the
    point, not an artifact). Non-centered because the CENTERED version
    (ab_position ~ MvNormal(mu, chol=pos_chol) as a stochastic node) makes
    ab_position's sampled density a direct function of pos_chol/pos_sigmas
    -- confirmed via free_RVs inspection that ab_position is itself a free
    RV there, which is what defines centered, regardless of chol appearing
    in the call signature (chol is required either way just to construct
    the covariance; it doesn't by itself make something non-centered).
    That coupling produced a funnel: divergences persisted even at
    target_accept=0.99, which pointed at genuine geometry rather than a
    step-size problem a tuning knob could fix. Here, z_ab ~ Normal(0, 1)
    is the actual free RV NUTS samples -- its distribution does NOT depend
    on pos_chol at all, so its geometry stays flat regardless of what
    pos_sigmas is doing. ab_position is reconstructed as a Deterministic
    AFTER sampling, pushing the funnel into a transform where it can't
    cause sampling difficulty."""
    pos_chol, pos_corr, pos_sigmas = pm.LKJCholeskyCov(
        'pos_chol_cov', n=2, eta=cfg.lkj_eta,
        sd_dist=pm.Exponential.dist(list(cfg.lkj_sd)),
        compute_corr=True,
    )
    intercept = pm.Normal('intercept', cfg.lkj_intercept_mu, cfg.lkj_intercept_sigma)
    beta = pm.Normal('beta', cfg.lkj_slope_mu, cfg.lkj_slope_sigma)
    z_ab = pm.Normal('z_ab', 0, 1, dims=('position', 'pos_param'))
    ab_position = pm.Deterministic(
        'ab_position', pt.stack([intercept, beta]) + pt.dot(z_ab, pos_chol.T),
        dims=('position', 'pos_param'),
    )
    pm.Deterministic('positions_mu', ab_position[:, 0], dims='position')
    pm.Deterministic('age_slope', ab_position[:, 1], dims='position')
    return pm.Deterministic(
        'intercepts',
        ab_position[position_data, 0] + ab_position[position_data, 1] * tenure_z_data,
    )
 
 
def _flat_intercept(position_data, tenure_z_data, cfg):
    """No-pooling alternative to _lkj_intercept: independent Normal
    intercept/slope per position, no shared hypermean or LKJ covariance.
    Thousands of player-weeks per position means little strength-borrowing
    benefit to pooling here -- tests whether the pooling STRUCTURE itself,
    not just eta/sd_dist scale, is what az.psense is flagging."""
    positions_mu = pm.Normal('positions_mu', cfg.lkj_intercept_mu,
                              cfg.flat_intercept_sigma, dims='position')
    age_slope = pm.Normal('age_slope', 0, cfg.flat_slope_sigma, dims='position')
    return pm.Deterministic(
        'intercepts',
        positions_mu[position_data] + age_slope[position_data] * tenure_z_data,
    )
 
 
def _linear_block(cont_data, bi_data, linear_input_sigma, cfg):
    if cfg.linear_mode == "independent":
        cont_priors = pm.Normal('cont_priors', 0, cfg.linear_normal_sigma_cont, dims='cont_vars')
        bi_priors = pm.Normal('bi_priors', 0, cfg.linear_normal_sigma_bi, dims='binary_vars')
        return pt.dot(cont_data, cont_priors) + pt.dot(bi_data, bi_priors)
    elif cfg.linear_mode == "r2d2":
        design = pt.concatenate([cont_data, bi_data], axis=1)
        linear_scale = pm.Gamma('linear_scale', mu=2.0, sigma=1.0)
        _, linear_beta = pmx.R2D2M2CP(
            'linear_effects', output_sigma=linear_scale, input_sigma=linear_input_sigma,
            dims='linear_vars', r2=cfg.linear_r2, r2_std=cfg.linear_r2_std,
        )
        return pt.dot(design, linear_beta)
    else:
        raise ValueError(f"unknown linear_mode: {cfg.linear_mode!r}")
 
 
def _team_form_block(team_form_data, opp_team_form_data, cfg):
    if cfg.team_form_mode == "independent":
        beta_team_form = pm.Normal('beta_team_form', 0, cfg.team_form_normal_sigma)
        beta_opp_team_form = pm.Normal('beta_opp_team_form', 0, cfg.team_form_normal_sigma)
        return beta_team_form * team_form_data + beta_opp_team_form * opp_team_form_data
    elif cfg.team_form_mode == "r2d2":
        tf_design = pt.stack([team_form_data, opp_team_form_data], axis=-1)
        team_form_scale = pm.Gamma('team_form_scale', mu=1.8, sigma=1.0)
        _, team_form_beta = pmx.R2D2M2CP(
            'team_form_effects', output_sigma=team_form_scale, input_sigma=np.array([1.0, 1.0]),
            dims='team_form_vars', r2=cfg.team_form_r2, r2_std=cfg.team_form_r2_std,
        )
        return pt.dot(tf_design, team_form_beta)
    else:
        raise ValueError(f"unknown team_form_mode: {cfg.team_form_mode!r}")
 
 
def _sigma_obs(cfg, n_positions):
    """per_position (current model) lets each position have its own
    residual scale -- real boom/bust-vs-steady scoring profiles differ by
    position, distinct from the MEAN varying via ab_position/player_skill.
    pooled collapses to one shared scalar, as an ablation: if
    per-position sigma_obs isn't earning its keep out-of-sample (see
    run_loo_comparison), the 'signal' in its posterior may just be
    soaking up mean-structure misspecification rather than true
    heteroskedasticity."""
    if cfg.sigma_obs_mode == "per_position":
        return pm.HalfNormal('sigma_obs', cfg.sigma_obs_scale, dims='position')
    elif cfg.sigma_obs_mode == "pooled":
        sigma_obs_scalar = pm.HalfNormal('sigma_obs_scalar', cfg.sigma_obs_scale)
        return pm.Deterministic('sigma_obs', sigma_obs_scalar * pt.ones(n_positions), dims='position')
    else:
        raise ValueError(f"unknown sigma_obs_mode: {cfg.sigma_obs_mode!r}")
 
 
def _observation_layer(mu, sigma_obs, position_data, games_missed_data, snap_pct_data, obs_data, cfg):
    if cfg.use_mixture:
        return models.add_role_mixture(mu, sigma_obs, position_data, games_missed_data,
                                        snap_pct_data, obs_data)
    return pm.StudentT('y_obs', mu=mu, sigma=sigma_obs[position_data], nu=4, observed=obs_data)
 
 
def build_debug_model(D, coords, season_offsets, T, boundary_cols,
                       position_of_player, position_group_counts, cfg):
    n_positions = len(coords['position'])
    n_seasons = len(coords['season'])
    coords = models._extend_coords(coords, T, n_seasons, pos_param=True)
    coords.setdefault('linear_vars', coords['cont_vars'] + coords['binary_vars'])
    coords.setdefault('team_form_vars', ['team_form', 'opp_team_form'])
 
    linear_input_sigma = np.concatenate([
        np.asarray(D['cont']).std(axis=0), np.asarray(D['bi']).std(axis=0),
    ])
    linear_input_sigma = np.where(linear_input_sigma > 0, linear_input_sigma, 1.0)
 
    with pm.Model(coords=coords) as model:
        dat = models.base_data(D, season_offsets)
        tenure_z_data = pm.Data('tenure_z_data', D['tenure_z'])
 
        if cfg.intercept_mode == "lkj":
            intercepts = _lkj_intercept(dat['position_data'], tenure_z_data, cfg)
        elif cfg.intercept_mode == "flat":
            intercepts = _flat_intercept(dat['position_data'], tenure_z_data, cfg)
        else:
            raise ValueError(f"unknown intercept_mode: {cfg.intercept_mode!r}")
 
        f_linear = _linear_block(dat['cont_data'], dat['bi_data'], linear_input_sigma, cfg)
 
        player_skill = _player_skill(coords, position_of_player, position_group_counts,
                                      n_positions, cfg)
        f_career = player_skill[dat['player_data'], dat['tenure_data']]
 
        f_opp = _opp_ar('opp', 'opp_team', dat['opp_team_data'], dat['global_time_data'],
                         T, boundary_cols, cfg)
 
        team_form_data = pm.Data('team_form_data', D['team_form_z'])
        opp_team_form_data = pm.Data('opp_team_form_data', D['opp_team_form_z'])
        f_team_form = _team_form_block(team_form_data, opp_team_form_data, cfg)
 
        mu = intercepts + f_linear + f_career + f_opp + f_team_form
 
        sigma_obs = _sigma_obs(cfg, n_positions)
        _observation_layer(mu, sigma_obs, dat['position_data'], dat['games_missed_data'],
                            dat['snap_pct_data'], dat['obs_data'], cfg)
    return model
 
 
# ============================================================================
# Named configs -- team_mod and r2d2_mod's current prior structure, plus
# the ablations from this round of debugging.
# ============================================================================
 
CONFIGS = {
    # true original team_mod prior structure -- independent priors
    # throughout, eta=4 (matches r2d2_mod_original's eta, both predate the
    # eta=2/split-sd_dist fix applied below).
    "team_mod_original": DebugConfig(
        name="team_mod_original",
        linear_mode="independent", team_form_mode="independent",
        player_skill_mode="independent", player_skill_innovation="normal",
        opp_ar_mode="independent",
        lkj_eta=4.0, lkj_sd=(1.0, 1.0),
        opp_rho_ab=(4.0, 2.0),
    ),
    "r2d2_mod_original": DebugConfig(
        name="r2d2_mod_original",
        linear_mode="r2d2", team_form_mode="r2d2",
        player_skill_mode="shared_budget",
        career_scale_kwargs=dict(mu=2.6, sigma=1.0),
        player_skill_innovation="normal",
        opp_ar_mode="shared_budget", opp_scale_kwargs=dict(mu=1.8, sigma=1.0),
        lkj_eta=4.0, lkj_sd=(1.0, 1.0),
        opp_rho_ab=(4.0, 2.0),
        linear_r2=0.35, linear_r2_std=0.08, team_form_r2=0.35, team_form_r2_std=0.08,
    ),
    "r2d2_mod_fixed": DebugConfig(
        name="r2d2_mod_fixed",
        linear_mode="r2d2", team_form_mode="r2d2",
        player_skill_mode="shared_budget",
        career_scale_kwargs=dict(alpha=2, beta=1),
        player_skill_innovation="studentt", player_skill_nu=5.0,
        opp_ar_mode="shared_budget", opp_scale_kwargs=dict(mu=1.8, sigma=1.0),
        lkj_eta=2.0, lkj_sd=(1.0, 3.0),
        opp_rho_ab=(2.0, 2.0),
        linear_r2=0.35, linear_r2_std=0.08, team_form_r2=0.05, team_form_r2_std=0.02,
    ),
}
CONFIGS["r2d2_mod_fixed_no_mixture"] = replace(
    CONFIGS["r2d2_mod_fixed"], name="r2d2_mod_fixed_no_mixture", use_mixture=False,
)
CONFIGS["r2d2_mod_original_no_mixture"] = replace(
    CONFIGS["r2d2_mod_original"], name="r2d2_mod_original_no_mixture", use_mixture=False,
)
CONFIGS["r2d2_mod_flat_intercept"] = replace(
    CONFIGS["r2d2_mod_fixed"], name="r2d2_mod_flat_intercept", intercept_mode="flat",
)
CONFIGS["r2d2_mod_flat_intercept_pooled_sigma"] = replace(
    CONFIGS["r2d2_mod_flat_intercept"], name="r2d2_mod_flat_intercept_pooled_sigma",
    sigma_obs_mode="pooled",
)
# team_mod's independent-priors architecture never had r2d2_mod's circular
# career_scale (that was specific to the shared_budget split), but the
# OTHER fixes generalize regardless of mode: StudentT innovations, a less
# peaked opp_rho prior, eta=2 + split sd_dist on the LKJ intercept. Same
# two-step ablation as r2d2 -- fixed, then flat_intercept on top of fixed.
CONFIGS["team_mod_fixed"] = replace(
    CONFIGS["team_mod_original"], name="team_mod_fixed",
    player_skill_innovation="studentt", player_skill_nu=5.0,
    opp_rho_ab=(2.0, 2.0),
    lkj_eta=2.0, lkj_sd=(1.0, 3.0),
)
CONFIGS["team_mod_flat_intercept"] = replace(
    CONFIGS["team_mod_fixed"], name="team_mod_flat_intercept", intercept_mode="flat",
)
 
 
def ab_position_rows(*summaries):
    """Concatenate one or more fit_and_check-produced summaries and filter
    down to just the ab_position entries, for the side-by-side comparison
    across configs that actually answers 'did this fix it'."""
    combined = pd.concat(summaries)
    mask = combined.index.astype(str).str.startswith("ab_position")
    return combined.loc[mask].sort_values(["config", "diagnosis"])
 
 
# ============================================================================
# fit_and_check -- az.psense analogue of ff_ar.py's fit_and_diagnose. Same
# two-phase shape (prior_check_only stops after the prior predictive check
# and returns the figure to look at before committing to a real sample()
# call), but ends in az.psense_summary instead of ESS-evolution plots,
# since sensitivity -- not convergence -- is what this file is for.
# nutpie doesn't store log_likelihood/log_prior automatically -- both are
# computed explicitly below so psense_summary has both groups to work with.
# ============================================================================
 
def fit_and_check(model, name, target_accept=0.95, prior_check_only=True,
                   draws=500, tune=1000, **sample_kwargs):
    with model:
        idata = pm.sample_prior_predictive(random_seed=SEED)
    ppc_fig = az.plot_ppc_dist(idata, group='prior_predictive', visuals={'observed_dist': True})
    if prior_check_only:
        return idata, ppc_fig
 
    with model:
        idata.update(pm.sample(draws=draws, tune=tune, nuts_sampler="nutpie",
                                random_seed=SEED, target_accept=target_accept, **sample_kwargs))
        pm.compute_log_likelihood(idata, model=model)
        pm.compute_log_prior(idata, model=model)
 
    idata.to_netcdf(f'model-nc/psense_{name}.nc')
 
    summary = az.psense_summary(idata).copy()
    summary["config"] = name
    summary.to_csv(f'model-nc/psense_{name}.csv')
    return idata, summary
 
 
# ============================================================================
# r2d2_mod_fixed -- previous round's fixes (StudentT innovations +
# Gamma(2, 1) career_scale to break the circular prior, eta=2 + split
# sd_dist on the LKJ intercept). Run this one first -- see models.py's
# build_r2d2_mod docstring for what "fixed" means relative to the original.
# ============================================================================
 
r2d2_fixed_model = build_debug_model(D, coords, season_offsets, T, boundary_cols,
                                      position_of_player, position_group_counts,
                                      CONFIGS['r2d2_mod_fixed'])
 
idata_r2d2_fixed, ppc_fig = fit_and_check(r2d2_fixed_model, 'r2d2_fixed')
idata_r2d2_fixed, summary_r2d2_fixed = fit_and_check(r2d2_fixed_model, 'r2d2_fixed', prior_check_only=False)
 
print(summary_r2d2_fixed[summary_r2d2_fixed['diagnosis'] != '-'])
 
 
# ============================================================================
# r2d2_mod_flat_intercept -- drops the pooled LKJ intercept/slope for
# independent per-position priors, to test whether the ab_position
# conflict is the pooling STRUCTURE (too few groups to hierarchically pool)
# rather than eta/sd_dist scale.
# ============================================================================
 
r2d2_flat_model = build_debug_model(D, coords, season_offsets, T, boundary_cols,
                                     position_of_player, position_group_counts,
                                     CONFIGS['r2d2_mod_flat_intercept'])
 
idata_r2d2_flat, ppc_fig = fit_and_check(r2d2_flat_model, 'r2d2_flat')
idata_r2d2_flat, summary_r2d2_flat = fit_and_check(r2d2_flat_model, 'r2d2_flat', prior_check_only=False)
 
print(summary_r2d2_flat[summary_r2d2_flat['diagnosis'] != '-'])
 
 
# ============================================================================
# team_mod_fixed -- the other architecture actually in play (see models.py's
# build_team_mod docstring: independent Normal priors throughout, no
# R2D2M2CP, no shared-budget player_skill/opp_ar split), with the
# generalizable half of r2d2_mod_fixed's fixes applied (StudentT
# innovations, less-peaked opp_rho, eta=2 + split sd_dist on the LKJ
# intercept -- team_mod never had r2d2_mod's circular career_scale, since
# that was specific to the shared_budget split team_mod doesn't use).
# Worth checking az.psense here too, not just on the r2d2 variants --
# team_mod's simpler, fully-independent priors could show a different
# conflict pattern (or none) on the same four blocks, which would itself
# be informative about whether R2D2M2CP's shared-budget structure is
# driving the conflict rather than the raw block choices themselves.
# ============================================================================
 
team_fixed_model = build_debug_model(D, coords, season_offsets, T, boundary_cols,
                                      position_of_player, position_group_counts,
                                      CONFIGS['team_mod_fixed'])
 
idata_team_fixed, ppc_fig = fit_and_check(team_fixed_model, 'team_fixed')
idata_team_fixed, summary_team_fixed = fit_and_check(team_fixed_model, 'team_fixed', prior_check_only=False)
 
print(summary_team_fixed[summary_team_fixed['diagnosis'] != '-'])
 
 
# ============================================================================
# team_mod_flat_intercept -- same no-pooling ablation as
# r2d2_mod_flat_intercept, on top of team_mod_fixed.
# ============================================================================
 
team_flat_model = build_debug_model(D, coords, season_offsets, T, boundary_cols,
                                     position_of_player, position_group_counts,
                                     CONFIGS['team_mod_flat_intercept'])
 
idata_team_flat, ppc_fig = fit_and_check(team_flat_model, 'team_flat')
idata_team_flat, summary_team_flat = fit_and_check(team_flat_model, 'team_flat', prior_check_only=False)
 
print(summary_team_flat[summary_team_flat['diagnosis'] != '-'])
 
 
# ============================================================================
# Sensitivity comparison across configs -- same spirit as ff_ar.py's
# az.compare model-comparison section, just comparing psense diagnoses
# instead of LOO.
# ============================================================================
 
sensitivity_comparison = pd.concat([
    summary_r2d2_fixed, summary_r2d2_flat, summary_team_fixed, summary_team_flat,
])
print(sensitivity_comparison[sensitivity_comparison['diagnosis'] != '-'].sort_values(['diagnosis', 'config']))
 
ab_position_rows(summary_r2d2_fixed, summary_r2d2_flat, summary_team_fixed, summary_team_flat)
 
 
# ============================================================================
# Posterior predictive check by position -- is the skew in
# total_fantasy_points already explained by the mixture's two components
# marginalizing over role, or is there real within-role skew left over
# that a skew-t component would actually fix? Cheaper and more decisive
# than guessing: simulate y_obs from a fitted model and overlay against
# the real by-position density. y_obs is the mixture's/plain-StudentT's
# observed variable name either way (see add_role_mixture in models.py),
# so this works unchanged regardless of which CONFIGS entry produced the
# idata being checked.
# ============================================================================
 
import matplotlib.pyplot as plt
import seaborn as sns
 
POSITION_COLORS = {'QB': '#F8766D', 'RB': '#7CAE00', 'TE': '#00BFC4', 'WR': '#C77CFF'}
 
 
def ppc_by_position(idata, model, D, coords, title='', n_draws=200):
    """Generic: takes an ALREADY-BUILT model and its matching idata
    directly, not tied to this script's build_debug_model/CONFIGS -- works
    for any pymc model whose observed node is named 'y_obs', which is
    every architecture in models.py (ar_mod, hs_version, team_mod,
    r2d2_mod, bart_mod all name it 'y_obs' -- see each build_*'s final
    add_role_mixture/pm.StudentT call). D only needs 'position' and 'y' to
    do the by-position grouping and observed comparison; the model can
    have been built from any of them, plain or feature-shrunk data
    included, as long as D matches what that idata was actually fit on."""
    n_total_draws = idata.posterior.sizes['chain'] * idata.posterior.sizes['draw']
    thinned = idata.sel(draw=slice(0, None, max(1, n_total_draws // n_draws)))
 
    with model:
        ppc = pm.sample_posterior_predictive(thinned, var_names=['y_obs'], random_seed=SEED,
                                              extend_inferencedata=False, progressbar=False)
 
    sim_y = ppc.posterior_predictive['y_obs'].values.reshape(-1, len(D['y']))
    position_labels = np.array(coords['position'])[D['position']]
 
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True)
    for ax, pos in zip(axes.flat, coords['position']):
        mask = position_labels == pos
        color = POSITION_COLORS.get(pos, 'gray')
        sns.kdeplot(D['y'][mask], ax=ax, color=color, fill=True, alpha=0.3, label='observed')
        # a handful of posterior predictive draws overlaid as thin lines --
        # more honest than one averaged density, since it also shows
        # draw-to-draw variability in the simulated shape
        n_show = min(15, sim_y.shape[0])
        for draw_idx in np.random.default_rng(SEED).choice(sim_y.shape[0], size=n_show, replace=False):
            sns.kdeplot(sim_y[draw_idx][mask], ax=ax, color=color, alpha=0.15, linestyle='--',
                        linewidth=1, label=None)
        ax.set_title(pos)
        ax.set_xlim(-10, 60)
    axes.flat[0].legend()
    fig.suptitle(f'{title}: observed vs. posterior predictive by position')
    fig.tight_layout()
    return fig, sim_y, position_labels
 
 
def posterior_predictive_by_position(idata_path, cfg_name, D, coords, n_draws=200):
    """Convenience wrapper for THIS script's own debug CONFIGS/
    build_debug_model -- loads the idata and builds the matching debug
    model, then delegates to the generic ppc_by_position above."""
    idata = az.from_netcdf(idata_path)
    model = build_debug_model(D, coords, season_offsets, T, boundary_cols,
                               position_of_player, position_group_counts, CONFIGS[cfg_name])
    return ppc_by_position(idata, model, D, coords, title=cfg_name, n_draws=n_draws)
 
 
ppc_fig_r2d2, sim_y_r2d2, position_labels = posterior_predictive_by_position(
    'model-nc/psense_r2d2_fixed.nc', 'r2d2_mod_fixed', D, coords,
)
plt.show()
 
ppc_fig_r2d2_flat, sim_y_r2d2_flat, _ = posterior_predictive_by_position(
    'model-nc/psense_r2d2_flat.nc', 'r2d2_mod_flat_intercept', D, coords,
)
plt.show()
 
ppc_fig_team_fixed, sim_y_team_fixed, _ = posterior_predictive_by_position(
    'model-nc/psense_team_fixed.nc', 'team_mod_fixed', D, coords,
)
plt.show()
 
ppc_fig_team_flat, sim_y_team_flat, _ = posterior_predictive_by_position(
    'model-nc/psense_team_flat.nc', 'team_mod_flat_intercept', D, coords,
)
plt.show()
 
 
# ============================================================================
# Sensitivity decision log -- per the arviz priorsense docs: "avoid the
# repeated adjustment of priors solely to resolve discrepancies or
# diagnostic warnings... If a prior is modified to address a warning, the
# change must be justified based on domain expertise, data properties, or
# model assumptions... choosing not to alter the model despite warnings
# can also be valid... it is essential to be transparent."
#
# This is a running record of every psense-flagged parameter across this
# round of debugging, whether it was changed or kept, and WHY -- the
# justification has to stand on its own, independent of "the flag went
# away," or it isn't a justification. Filling this in (rather than just
# rerunning psense and eyeballing green/red) is what actually satisfies
# the transparency requirement above.
# ============================================================================
 
SENSITIVITY_DECISIONS = {
    'career_scale': dict(
        flagged=True, action='changed',
        justification=(
            "Gamma(mu=2.6, sigma=1.0) was fit from the SAME backtest now "
            "validating the model (circular) -- changed to Gamma(alpha=2, "
            "beta=1) to remove the circularity itself. The prior change is "
            "justified by the circularity being a genuine methodological "
            "error, independent of what psense reported; psense flagging "
            "it was a symptom, not the reason for the fix."
        ),
    ),
    'z_innov': dict(
        flagged=True, action='changed',
        justification=(
            "Normal -> StudentT(nu=5) innovations. Justified by domain "
            "knowledge (breakout/bust player-weeks are fat-tailed events, "
            "not evidence the Normal's SCALE was wrong) rather than by the "
            "flag disappearing -- a Normal random walk structurally cannot "
            "represent that heavy-tailed behavior at any scale."
        ),
    ),
    'team_form_scale': dict(
        flagged=True, action='changed',
        justification=(
            "r2=0.35 was applied to a 2-covariate block AND the "
            "16-covariate linear block simultaneously, double-counting the "
            "same R^2 budget -- reduced to r2=0.05 for team_form "
            "specifically. Justified by the double-counting being a real "
            "mis-specification (the model cannot deliver 0.35 R^2 from two "
            "blocks at once), not by chasing the diagnostic."
        ),
    ),
    'opp_rho': dict(
        flagged=True, action='changed',
        justification=(
            "Beta(4,2) (mean 0.667) asserted more week-to-week defensive-"
            "form persistence than is plausible -- moved to Beta(2,2). "
            "Justified by a domain judgment about how volatile defensive "
            "form actually is week to week, not solely by the flag."
        ),
    ),
    'pos_chol_cov / ab_position': dict(
        flagged=True, action='reparameterized (kept pooling)',
        justification=(
            "Divergences persisted even at target_accept=0.99 under the "
            "centered MvNormal(mu, chol) parameterization -- an "
            "independent sampling diagnostic (divergences/r_hat), not "
            "psense, that pointed at real funnel geometry. Rewrote as "
            "non-centered (z_ab ~ Normal(0,1), ab_position as a "
            "Deterministic). Pooling was KEPT deliberately despite "
            "ongoing weak identification from n_positions=4 -- domain "
            "reasoning (RB/WR age-cliff vs TE's flatter curve genuinely "
            "covary with position-level baseline) justifies wanting the "
            "covariance structure even though 4 groups can't estimate it "
            "with much precision. If psense still flags this after the "
            "non-centered rerun, that's an honest reflection of genuine "
            "small-group weak identification, not a defect to keep "
            "engineering around -- see the arviz guidance above."
        ),
    ),
    'sigma_obs (per-position)': dict(
        flagged=None, action='pending -- see run_loo_comparison',
        justification=(
            "NOT decided by psense at all -- whether per-position "
            "residual variance is real heteroskedasticity or absorbing "
            "mean-structure misspecification is an out-of-sample question "
            "(LOO), deliberately kept separate from the sensitivity "
            "analysis."
        ),
    ),
}
 
 
def print_sensitivity_decisions():
    for param, entry in SENSITIVITY_DECISIONS.items():
        print(f"{param} [{entry['action']}]")
        print(f"  {entry['justification']}\n")
 
 
print_sensitivity_decisions()
 
 
# ============================================================================
# Real fitted models -- ff_bart_plain.nc, ff_team_shrunk.nc, etc., NOT this
# script's debug CONFIGS/build_debug_model. 'plain'/'shrunk' here means
# models.apply_shrunk_features applied or not -- it coalesces
# SHRUNK_SWAP_COLS IN PLACE (same STD_COLS/BART_CONT_VARS column NAMES
# either way, only the values under the 4 swapped columns differ), which
# is models.py's own convention (BART_CONT_VARS is built from STD_COLS
# directly) and is DIFFERENT from the separate _shrunk-suffixed-column
# join this script's own data prep at the top does for the debug CONFIGS.
# ASSUMPTION WORTH CONFIRMING: if whatever script actually produced these
# six .nc files built plain/shrunk some other way (different columns
# included, different scaler/index fit windows), the D/coords below won't
# exactly match what those idata were fit on, and the PPC will silently
# compare against the wrong covariates rather than erroring -- if the
# plots below look implausibly bad for a model you otherwise trust, this
# mismatch is the first thing to suspect, not the model itself.
# ============================================================================
 
PATH = 'model-nc/'
MODS = ['ff_bart_plain.nc', 'ff_bart_shrunk.nc', 'ff_team_plain.nc', 'ff_team_shrunk.nc',
        'ff_r2d2_plain.nc', 'ff_r2d2_shrunk.nc']
 
 
def build_variant(shrunk):
    """Same shape as this file's own top-of-file data prep (idxs fit on
    the FULL, uncut history; scalers/team_form_scaler fit on the
    LAST_SEASON-cutoff `frame`), just using models.py's own
    apply_shrunk_features/frame_to_D instead of a manual _shrunk-suffix
    join, and branching from add_opp_form (team_form already joined,
    before any shrinkage decision) rather than the top-of-file `train`
    (which already has the OTHER shrinkage convention baked in)."""
    base = models.apply_shrunk_features(add_opp_form, path=SHRINKAGE_PATH) if shrunk else add_opp_form
    lw_v = base.filter(pl.col('season') == LAST_SEASON)['week'].max()
    frame = base.filter(
        (pl.col('season') < LAST_SEASON) | ((pl.col('season') == LAST_SEASON) & (pl.col('week') < lw_v))
    )
 
    scalers = models.fit_scalers(df=frame, cols=models.STD_COLS)
    idxs = models.make_enums(base, models.IDX_COLS)
    team_form_scaler = models.fit_team_form_scaler(frame)
    D_v = models.frame_to_D(frame, scalers, idxs, team_form_scaler=team_form_scaler)
 
    coords_v = {c: e.categories.to_list() for c, e in idxs.items()}
    coords_v['cont_vars'] = list(scalers.keys())
    coords_v['binary_vars'] = models.BINARY_VARS
 
    weeks_per_season_v = (
        base.group_by('season').agg(pl.col('week').max().alias('m'))
        .sort('season')['m'].cast(pl.Int64).to_numpy()
    )
    season_offsets_v, T_v, boundary_cols_v = models.build_ar_calendar(weeks_per_season_v)
 
    n_positions_v = len(coords_v['position'])
    pop_v, pgc_v = models.player_position_map(base, 'player_id', 'position', idxs, n_positions_v)
 
    return D_v, coords_v, season_offsets_v, T_v, boundary_cols_v, pop_v, pgc_v
 
 
VARIANTS = {'plain': build_variant(shrunk=False), 'shrunk': build_variant(shrunk=True)}
 
MODEL_BUILDERS = {
    'bart': lambda D_v, coords_v, so_v, T_v, bc_v, pop_v, pgc_v:
        models.build_bart_mod(D_v, coords_v, pop_v, pgc_v),
    'team': lambda D_v, coords_v, so_v, T_v, bc_v, pop_v, pgc_v:
        models.build_team_mod(D_v, coords_v, so_v, T_v, bc_v, pop_v, pgc_v),
    'r2d2': lambda D_v, coords_v, so_v, T_v, bc_v, pop_v, pgc_v:
        models.build_r2d2_mod(D_v, coords_v, so_v, T_v, bc_v, pop_v, pgc_v),
}
 
for fname in MODS:
    arch, variant = fname.removeprefix('ff_').removesuffix('.nc').rsplit('_', 1)
    D_v, coords_v, *rest = VARIANTS[variant]
    model_v = MODEL_BUILDERS[arch](D_v, coords_v, *rest)
    idata_v = az.from_netcdf(PATH + fname)
    fig, sim_y, labels = ppc_by_position(idata_v, model_v, D_v, coords_v, title=fname)
    plt.show()
 
 
# ============================================================================
# fit_and_check -- az.psense analogue of ff_ar.py's fit_and_diagnose. Same
# two-phase shape (prior_check_only stops after the prior predictive check
# and returns the figure to look at before committing to a real sample()
# call), but ends in az.psense_summary instead of ESS-evolution plots,
# since sensitivity -- not convergence -- is what this file is for.
# nutpie doesn't store log_likelihood/log_prior automatically -- both are
# computed explicitly below so psense_summary has both groups to work with.
# ============================================================================
 
def fit_and_check(model, name, target_accept=0.95, prior_check_only=True,
                   draws=500, tune=1000, **sample_kwargs):
    with model:
        idata = pm.sample_prior_predictive(random_seed=SEED)
    ppc_fig = az.plot_ppc_dist(idata, group='prior_predictive', visuals={'observed_dist': True})
    if prior_check_only:
        return idata, ppc_fig
 
    with model:
        idata.update(pm.sample(draws=draws, tune=tune, nuts_sampler="nutpie",
                                random_seed=SEED, target_accept=target_accept, **sample_kwargs))
        pm.compute_log_likelihood(idata, model=model)
        pm.compute_log_prior(idata, model=model)
 
    idata.to_netcdf(f'model-nc/psense_{name}.nc')
 
    summary = az.psense_summary(idata).copy()
    summary["config"] = name
    summary.to_csv(f'model-nc/psense_{name}.csv')
    return idata, summary
 
for v in ['mu_limited', 'sigma_limited', 'role_intercept', 'role_beta_missed', 'role_beta_snap']:
    print(v)
    print(idata_v.posterior[v].mean(dim=['chain', 'draw']).to_dataframe())
# ============================================================================
# r2d2_mod_fixed -- previous round's fixes (StudentT innovations +
# Gamma(2, 1) career_scale to break the circular prior, eta=2 + split
# sd_dist on the LKJ intercept). Run this one first -- see models.py's
# build_r2d2_mod docstring for what "fixed" means relative to the original.
# ============================================================================
 
r2d2_fixed_model = build_debug_model(D, coords, season_offsets, T, boundary_cols,
                                      position_of_player, position_group_counts,
                                      CONFIGS['r2d2_mod_fixed'])
 
idata_r2d2_fixed, ppc_fig = fit_and_check(r2d2_fixed_model, 'r2d2_fixed')
idata_r2d2_fixed, summary_r2d2_fixed = fit_and_check(r2d2_fixed_model, 'r2d2_fixed', prior_check_only=False)
 
print(summary_r2d2_fixed[summary_r2d2_fixed['diagnosis'] != '-'])
 
 
# ============================================================================
# r2d2_mod_flat_intercept -- drops the pooled LKJ intercept/slope for
# independent per-position priors, to test whether the ab_position
# conflict is the pooling STRUCTURE (too few groups to hierarchically pool)
# rather than eta/sd_dist scale.
# ============================================================================
 
r2d2_flat_model = build_debug_model(D, coords, season_offsets, T, boundary_cols,
                                     position_of_player, position_group_counts,
                                     CONFIGS['r2d2_mod_flat_intercept'])
 
idata_r2d2_flat, ppc_fig = fit_and_check(r2d2_flat_model, 'r2d2_flat')
idata_r2d2_flat, summary_r2d2_flat = fit_and_check(r2d2_flat_model, 'r2d2_flat', prior_check_only=False)
 
print(summary_r2d2_flat[summary_r2d2_flat['diagnosis'] != '-'])
 
 
# ============================================================================
# team_mod_fixed -- the other architecture actually in play (see models.py's
# build_team_mod docstring: independent Normal priors throughout, no
# R2D2M2CP, no shared-budget player_skill/opp_ar split), with the
# generalizable half of r2d2_mod_fixed's fixes applied (StudentT
# innovations, less-peaked opp_rho, eta=2 + split sd_dist on the LKJ
# intercept -- team_mod never had r2d2_mod's circular career_scale, since
# that was specific to the shared_budget split team_mod doesn't use).
# Worth checking az.psense here too, not just on the r2d2 variants --
# team_mod's simpler, fully-independent priors could show a different
# conflict pattern (or none) on the same four blocks, which would itself
# be informative about whether R2D2M2CP's shared-budget structure is
# driving the conflict rather than the raw block choices themselves.
# ============================================================================
 
team_fixed_model = build_debug_model(D, coords, season_offsets, T, boundary_cols,
                                      position_of_player, position_group_counts,
                                      CONFIGS['team_mod_fixed'])
 
idata_team_fixed, ppc_fig = fit_and_check(team_fixed_model, 'team_fixed')
idata_team_fixed, summary_team_fixed = fit_and_check(team_fixed_model, 'team_fixed', prior_check_only=False)
 
print(summary_team_fixed[summary_team_fixed['diagnosis'] != '-'])
 

 
# ============================================================================
# team_mod_flat_intercept -- same no-pooling ablation as
# r2d2_mod_flat_intercept, on top of team_mod_fixed.
# ============================================================================
 
team_flat_model = build_debug_model(D, coords, season_offsets, T, boundary_cols,
                                     position_of_player, position_group_counts,
                                     CONFIGS['team_mod_flat_intercept'])
 
idata_team_flat, ppc_fig = fit_and_check(team_flat_model, 'team_flat')
idata_team_flat, summary_team_flat = fit_and_check(team_flat_model, 'team_flat', prior_check_only=False)
 
print(summary_team_flat[summary_team_flat['diagnosis'] != '-'])
 
 
# ============================================================================
# Sensitivity comparison across configs -- same spirit as ff_ar.py's
# az.compare model-comparison section, just comparing psense diagnoses
# instead of LOO.
# ============================================================================
 
sensitivity_comparison = pd.concat([
    summary_r2d2_fixed, summary_r2d2_flat, summary_team_fixed, summary_team_flat,
])
print(sensitivity_comparison[sensitivity_comparison['diagnosis'] != '-'].sort_values(['diagnosis', 'config']))
 
ab_position_rows(summary_r2d2_fixed, summary_r2d2_flat, summary_team_fixed, summary_team_flat)


check_these = pl.from_pandas(pd.concat(
    [summary_r2d2_fixed, summary_r2d2_flat, summary_team_fixed, summary_team_flat]
).reset_index())

find_good = check_these.filter(pl.col('diagnosis') == '✓')
az.summary(idata_r2d2_fixed, var_names=["~z_innov", "~ab_position"])["r_hat"].sort_values(ascending=False).head(10)
