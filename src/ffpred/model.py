"""The heteroskedastic latent-GP player model (het_gp), fit and predict.

Ported from ``src/models.py`` (``HetPrep`` / ``build_het_gp``) with the
model math unchanged. What changed around it:

* only the two production configs exist (``config.MODELS``);
* the team-context and Student-t branches are gone (unused by these configs);
* ``HetPrep.eligible`` says which rows the model can score, using the
  model's own requirements (player seen in training, time index inside the
  fitted grid) instead of the old cross-model cold-start filter;
* ``fit`` / ``FittedModel.predict`` wrap sampling and posterior prediction
  and return draws on the fantasy-points scale.

Model (on standardized points y_z):
    mu = alpha[pos] + f_career[time, pos] + skill[player] + f_season[player, time]
         + cont @ beta_cont + bi @ beta_bi + b_tf * team_form + b_otf * opp_team_form
    y_z ~ Normal(mu, sigma)  with sigma = exp(log_sigma0[pos] + g_noise[time])  (skill)
                             or a single sigma                                (QB)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import polars as pl
import preliz as pz
import pymc as pm
import pytensor.tensor as pt

# pyrefly: ignore [missing-import]
from .config import BINARY_VARS, TARGET_COL, HetConfig, SamplerSettings

log = logging.getLogger(__name__)


def _inverse_gamma_maxent(lower: float, upper: float) -> tuple[float, float]:
    d = pz.InverseGamma()
    pz.maxent(d, lower=lower, upper=upper, plot=False)   # plot defaults on: one figure per call
    return float(d.alpha), float(d.beta)


class HetPrep:
    """Everything het_gp estimates from data before sampling, fit once on a
    training frame. ``arrays(frame)`` turns any frame into model inputs using
    training-time quantities only (scalers, encoders, centring weights)."""

    def __init__(self, train: pl.DataFrame, cfg: HetConfig):
        self.cfg = cfg
        train = train.filter(pl.col('position').is_in(list(cfg.positions)))
        if train.height == 0:
            raise ValueError(f'{cfg.name}: no training rows for positions {cfg.positions}')
        tcol = cfg.time_col
        self.players = train['player_id'].unique().sort().to_list()
        self.positions = list(cfg.positions)
        t_min, t_max = int(train[tcol].min()), int(train[tcol].max())
        self.times = list(range(t_min, t_max + 2))          # contiguous + one next-season slot
        self._player_map = {p: i for i, p in enumerate(self.players)}
        self._pos_map = {p: i for i, p in enumerate(self.positions)}
        self._time_map = {t: i for i, t in enumerate(self.times)}

        self.scalers = _fit_scalers(train, cfg.std_cols)
        self.bi_means = {c: float(train[c].mean()) for c in BINARY_VARS}
        self.tf_mu = float(train['team_form'].mean())
        self.tf_sd = float(train['team_form'].std() or 1.0)
        y = train[TARGET_COL].to_numpy()
        self.y_mu, self.y_sd = float(y.mean()), float(y.std())

        D = self.arrays(train)
        n_pl, n_pos, n_t = len(self.players), len(self.positions), len(self.times)
        self.position_of_player = np.zeros(n_pl, dtype=int)
        self.position_of_player[D['player']] = D['position']
        self.pos_onehot = np.eye(n_pos)[self.position_of_player]
        self.pos_counts = np.maximum(self.pos_onehot.sum(0), 1.0)

        counts = np.zeros((n_t, n_pos))
        np.add.at(counts, (D['time'], D['position']), 1)
        self.t_weights_pos = counts / np.maximum(counts.sum(0, keepdims=True), 1.0)
        self.t_weights = counts.sum(1) / counts.sum()

        self.active = np.zeros((n_pl, n_t))
        self.active[D['player'], D['time']] = 1.0
        self.n_active = np.maximum(self.active.sum(0), 1.0)
        self.n_seasons_player = np.maximum(self.active.sum(1), 1.0)

        self.season_prior = _inverse_gamma_maxent(*cfg.season_ell)
        self.noise_prior = _inverse_gamma_maxent(*cfg.noise_ell)
        self.coords = {'position': self.positions, 'player_id': self.players, 'time': self.times,
                       'cont_vars': [f'{c}_z' for c in cfg.std_cols], 'binary_vars': list(BINARY_VARS)}
        self.grid = np.asarray(self.times, dtype='float64')
        self.train_arrays = D

    # ------------------------------------------------------------ eligibility
    def eligible(self, frame: pl.DataFrame) -> np.ndarray:
        """Boolean mask of rows this fitted model can score: the player was
        in training (his career level is estimated), the position is covered,
        and his time index (tenure / age) falls inside the fitted grid."""
        tcol = self.cfg.time_col
        return (frame.select(
            pl.col('player_id').is_in(self.players)
            & pl.col('position').is_in(self.positions)
            & pl.col(tcol).is_not_null()
            & pl.col(tcol).cast(pl.Int64, strict=False).is_in(self.times)
        ).to_series().fill_null(False).to_numpy())

    # ----------------------------------------------------------------- arrays
    def arrays(self, frame: pl.DataFrame) -> dict[str, np.ndarray]:
        n = len(frame)
        cont = (np.column_stack([((frame[c].cast(pl.Float64).fill_null(mu) - mu) / sd).to_numpy()
                                 for c, (mu, sd) in self.scalers.items()])
                if self.scalers else np.zeros((n, 0)))
        y = frame[TARGET_COL].cast(pl.Float64).fill_null(0.0) if TARGET_COL in frame.columns \
            else pl.Series(np.zeros(n))
        return {
            'player': frame['player_id'].replace_strict(self._player_map, return_dtype=pl.Int64).to_numpy(),
            'position': frame['position'].replace_strict(self._pos_map, return_dtype=pl.Int64).to_numpy(),
            'time': frame[self.cfg.time_col].cast(pl.Int64)
                    .replace_strict(self._time_map, return_dtype=pl.Int64).to_numpy(),
            'cont': np.ascontiguousarray(cont.astype('float64')),
            'bi': np.ascontiguousarray(np.column_stack(
                [frame[c].cast(pl.Float64).to_numpy() - self.bi_means[c] for c in BINARY_VARS])),
            'team_form_z': ((frame['team_form'] - self.tf_mu) / self.tf_sd).to_numpy(),
            'opp_team_form_z': ((frame['opp_team_form'] - self.tf_mu) / self.tf_sd).to_numpy(),
            'y': y.to_numpy().astype('float64'),
            'y_z': ((y - self.y_mu) / self.y_sd).to_numpy(),
        }


def _fit_scalers(df: pl.DataFrame, cols) -> dict[str, tuple[float, float]]:
    """{col: (mean, sd)}; sd floored to 1.0 for a column constant in this slice."""
    stats = df.select([pl.col(c).mean().alias(f'{c}__mu') for c in cols]
                      + [pl.col(c).std().alias(f'{c}__sd') for c in cols]).row(0, named=True)
    return {c: (stats[f'{c}__mu'], stats[f'{c}__sd'] or 1.0) for c in cols}


def payload(D: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """{pm.Data name: array} for pm.set_data() on build_het_gp's model."""
    return {
        'player_data': D['player'], 'position_data': D['position'], 'time_data': D['time'],
        'cont_data': D['cont'], 'bi_data': D['bi'],
        'team_form_data': D['team_form_z'], 'opp_team_form_data': D['opp_team_form_z'],
        'obs_data': np.zeros(len(D['player'])),
    }


def build_het_gp(D: dict[str, np.ndarray], prep: HetPrep, jitter: float = 1e-6) -> pm.Model:
    """het_gp on STANDARDIZED points. Predictions * prep.y_sd + prep.y_mu = points."""
    cfg = prep.cfg
    grid = prep.grid
    X = grid[:, None]
    multi = len(prep.positions) > 1
    with pm.Model(coords=prep.coords) as model:
        player = pm.Data('player_data', D['player'])
        pos = pm.Data('position_data', D['position'])
        t = pm.Data('time_data', D['time'])
        cont = pm.Data('cont_data', D['cont'])
        bi = pm.Data('bi_data', D['bi'])
        tf = pm.Data('team_form_data', D['team_form_z'])
        otf = pm.Data('opp_team_form_data', D['opp_team_form_z'])
        obs = pm.Data('obs_data', D['y_z'])

        alpha = pm.Normal('alpha_pos', 0.0, 0.5, dims='position')

        # 1. aging bump(s), centred on the training time distribution
        if multi:
            t_peak_mu = pm.Normal('t_peak_mu', *cfg.t_peak)
            t_peak_off = pm.ZeroSumNormal('t_peak_off', sigma=1.0, dims='position')
            t_peak = pm.Deterministic('t_peak', t_peak_mu + t_peak_off, dims='position')
            log_height_mu = pm.Normal('log_height_mu', np.log(0.5), 0.4)
            log_height_off = pm.ZeroSumNormal('log_height_off', sigma=0.25, dims='position')
            peak_height = pm.Deterministic('peak_height', pt.exp(log_height_mu + log_height_off), dims='position')
        else:
            t_peak = pm.Normal('t_peak', *cfg.t_peak)[None]
            log_height = pm.Normal('log_height', np.log(0.5), 0.4)
            peak_height = pm.Deterministic('peak_height', pt.exp(log_height))[None]
        if cfg.asymmetric_bump:
            width_rise = pm.Deterministic('width_rise', 1.0 + pm.Gamma('width_rise_raw', alpha=4, beta=1.5))
            width_decline = pm.Deterministic('width_decline', 1.0 + pm.Gamma('width_decline_raw', alpha=8, beta=1.2))
            width = pt.switch(grid[:, None] < t_peak[None, :], width_rise, width_decline)
        else:
            width = pm.Deterministic('peak_width', 1.0 + pm.Gamma('peak_width_raw', alpha=6, beta=1.5))
        curve = peak_height[None, :] * pt.exp(-0.5 * ((grid[:, None] - t_peak[None, :]) / width) ** 2)
        f_career = pm.Deterministic('f_career', curve - (prep.t_weights_pos * curve).sum(0),
                                    dims=('time', 'position'))

        # 2. career skill level, mean-zero within position
        sigma_skill = pm.HalfNormal('sigma_skill', 1.0)
        if multi:
            z_skill = pm.Normal('z_skill', 0.0, 1.0, dims='player_id')
            raw = sigma_skill * z_skill
            skill = pm.Deterministic(
                'skill', raw - ((prep.pos_onehot.T @ raw) / prep.pos_counts)[prep.position_of_player],
                dims='player_id')
        else:
            z_skill = pm.ZeroSumNormal('z_skill', sigma=1.0, dims='player_id')
            skill = pm.Deterministic('skill', sigma_skill * z_skill, dims='player_id')

        # 3. season-to-season level: per-player GP, centred within player and per time cell
        ell_season = pm.InverseGamma('ell_season', *prep.season_prior)
        eta_season = pm.HalfNormal('eta_season', 0.5)
        L_season = pt.linalg.cholesky(pm.gp.util.stabilize(
            (eta_season**2 * pm.gp.cov.Matern32(1, ls=ell_season))(X), jitter))
        z_season = pm.Normal('z_season', 0.0, 1.0, dims=('player_id', 'time'))
        f_raw = z_season @ L_season.T
        f_p = f_raw - ((f_raw * prep.active).sum(1) / prep.n_seasons_player)[:, None]
        f_season = f_p - (f_p * prep.active).sum(0) / prep.n_active

        # 4. covariates
        cont_prior = pm.Normal('cont_prior', 0.0, 0.5, dims='cont_vars')
        bi_prior = pm.Normal('bi_prior', 0.0, 0.5, dims='binary_vars')
        team_form_prior = pm.Normal('team_form_prior', 0.0, 0.5)
        opp_form_prior = pm.Normal('opp_form_prior', 0.0, 0.5)
        mu = (alpha[pos] + f_career[t, pos] + skill[player] + f_season[player, t]
              + pt.dot(cont, cont_prior) + pt.dot(bi, bi_prior)
              + team_form_prior * tf + opp_form_prior * otf)

        # 5. observation scale
        if cfg.noise_gp:
            log_sigma0 = pm.Normal('log_sigma0', 0.0, 0.5, dims='position')
            ell_noise = pm.InverseGamma('ell_noise', *prep.noise_prior)
            eta_noise = pm.HalfNormal('eta_noise', 0.15)
            gp_noise = pm.gp.Latent(cov_func=eta_noise**2 * pm.gp.cov.Matern52(1, ls=ell_noise))
            g_raw = gp_noise.prior('g_noise_raw', X=X, jitter=jitter, dims='time')
            g_noise = pm.Deterministic('g_noise', g_raw - pt.dot(prep.t_weights, g_raw), dims='time')
            sigma = pt.exp(log_sigma0[pos] + g_noise[t])
        else:
            sigma = pm.HalfNormal('sigma_obs', 1.0)
        pm.Normal('y_obs', mu=mu, sigma=sigma, observed=obs)
    return model


# ---------------------------------------------------------------- fit / predict

SCALAR_DIAG_VARS = ('sigma_skill', 'ell_season', 'eta_season', 'alpha_pos', 'cont_prior',
                    'bi_prior', 'team_form_prior', 'opp_form_prior')


@dataclass
class FittedModel:
    cfg: HetConfig
    prep: HetPrep
    model: pm.Model
    idata: object
    settings: SamplerSettings
    fit_time: float
    divergences: int
    max_rhat: float | None
    min_ess_bulk: float | None

    @property
    def n_train(self) -> int:
        return len(self.prep.train_arrays['y'])

    def predict(self, frame: pl.DataFrame, seed: int | None = None) -> np.ndarray:
        """Posterior-predictive draws on the fantasy-points scale, shape
        (n_draws, len(frame)), columns in frame's row order. Every row must be
        eligible (see HetPrep.eligible)."""
        if len(frame) == 0:
            return np.zeros((self.settings.n_draws, 0))
        ok = self.prep.eligible(frame)
        if not ok.all():
            bad = frame.filter(~pl.Series(ok)).select('player_id', 'player', 'position').head(5)
            raise ValueError(f'{self.cfg.name}: {(~ok).sum()} rows are not scoreable by this fit, e.g. {bad.rows()}')
        seed = self.settings.seed + 1 if seed is None else seed
        D = self.prep.arrays(frame)
        with self.model:
            pm.set_data(payload(D))
            ppc = pm.sample_posterior_predictive(self.idata, var_names=['y_obs'], progressbar=False,
                                                 random_seed=seed)
        draws = ppc.posterior_predictive['y_obs'].values.reshape(-1, len(frame))
        return draws * self.prep.y_sd + self.prep.y_mu

    def coefs(self) -> pl.DataFrame:
        """Posterior mean / sd of the linear covariate effects (standardized
        y per sd of x)."""
        post = self.idata.posterior
        rows = []
        for var, dim in (('cont_prior', 'cont_vars'), ('bi_prior', 'binary_vars')):
            da = post[var]
            for name, m, s in zip(da[dim].values, da.mean(('chain', 'draw')).values,
                                  da.std(('chain', 'draw')).values):
                rows.append({'var': str(name), 'coef_mean': float(m), 'coef_sd': float(s)})
        for var in ('team_form_prior', 'opp_form_prior'):
            da = post[var]
            rows.append({'var': var, 'coef_mean': float(da.mean()), 'coef_sd': float(da.std())})
        return pl.DataFrame(rows)


def fit(train: pl.DataFrame, cfg: HetConfig, settings: SamplerSettings = SamplerSettings()) -> FittedModel:
    """Fit one het_gp config on ``train`` (rows outside cfg.positions are ignored)."""
    # Sorted so a fit doesn't depend on upstream join order: with identical
    # data and seed, a different row order gives a different (equally valid)
    # MCMC run -- about 1.5% fold CRPS for a QB fold, larger than most of the
    # ablation effects being compared.
    train = (train.filter(pl.col('position').is_in(list(cfg.positions)))
             .sort('player_id', 'season', 'week'))
    prep = HetPrep(train, cfg)
    model = build_het_gp(prep.train_arrays, prep)
    s = settings
    t0 = time.time()
    with model:
        if s.method == 'advi':
            approx = pm.fit(n=s.advi_n, method='advi', random_seed=s.seed, progressbar=False)
            idata = approx.sample(s.advi_draws, random_seed=s.seed)
        else:
            idata = pm.sample(draws=s.draws, tune=s.tune, chains=s.chains, cores=s.chains,
                              random_seed=s.seed, progressbar=False, target_accept=s.target_accept,
                              nuts_sampler=s.nuts_sampler, **s.extra)
    fit_time = time.time() - t0

    divergences, max_rhat, min_ess = 0, None, None
    if s.method == 'nuts':
        divergences = int(idata.sample_stats['diverging'].sum())
        max_rhat, min_ess = _diagnostics(idata, s.chains)
    log.info('%s: fit %d rows / %d players in %.0fs (divergences=%d, max r_hat=%s, min ess_bulk=%s)',
             cfg.name, len(prep.train_arrays['y']), len(prep.players), fit_time, divergences,
             None if max_rhat is None else f'{max_rhat:.3f}', None if min_ess is None else f'{min_ess:.0f}')
    if divergences:
        log.warning('%s: %d divergent transitions', cfg.name, divergences)
    if max_rhat is not None and max_rhat > 1.05:
        log.warning('%s: max r_hat %.3f > 1.05 on the population parameters', cfg.name, max_rhat)
    return FittedModel(cfg=cfg, prep=prep, model=model, idata=idata, settings=s, fit_time=fit_time,
                       divergences=divergences, max_rhat=max_rhat, min_ess_bulk=min_ess)


def _diagnostics(idata, chains: int) -> tuple[float | None, float | None]:
    """Max r_hat / min bulk ESS over the population-level parameters."""
    if chains < 2:
        return None, None
    try:
        import arviz as az
        names = [v for v in SCALAR_DIAG_VARS if v in idata.posterior]
        summ = az.summary(idata, var_names=names, kind='diagnostics')
        cols = {c.lower(): c for c in summ.columns}
        rhat = summ[cols['r_hat']].max() if 'r_hat' in cols else None
        ess = summ[cols['ess_bulk']].min() if 'ess_bulk' in cols else None
        return (None if rhat is None else float(rhat)), (None if ess is None else float(ess))
    except Exception as e:  # diagnostics must never break a fit
        log.warning('could not compute r_hat/ess: %s', e)
        return None, None
