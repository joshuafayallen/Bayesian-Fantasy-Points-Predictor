"""The QB-linked model: production QB model + hurdle-gamma skill model linked
through the starting QB's game residual. Fit in two stages (a "cut").

Stage 1  het_gp_qb_vegas (``model.fit``, unchanged), fit alone. For every
         training QB game, r = (points - E[mean]) / E[sigma]: how far the QB
         beat or missed his projection, in standard deviations.
Stage 2  RB / WR / TE hurdle-gamma on the points scale (log link):
             P(dud)       = 1 - sigmoid(a_psi[pos] + cont @ w_psi + bi @ wb_psi)
             E[y | no dud] = m = exp(log m_ref[pos] + alpha[pos] + f_career + skill + f_season
                                     + covariates + lam_rec[pos] * r - lam_rec[pos]**2 / 2)
             positive part Gamma(mean m, sd cv * m), cv falling with the trailing level
         r enters as data, so the receivers learn lam_rec from the QB's results
         but cannot bend the QB fit.
Forecast QB draw S_q from stage 1; the QB's surprise e = (S_q - E[S_q]) / E[sigma]
         is shared by every receiver on his team:  S_s = S0 * exp(lam e - lam**2 var(e) / 2),
         S0 = the receiver's own draw with the link off. Scaling m scales the whole
         gamma (its sd is cv * m) and leaves P(dud) alone, so this is exact, and the
         -lam**2 var(e)/2 term keeps every projection unchanged. Row i of the QB
         and skill draws carries the same e: stacks co-move.

Why two stages: fitting both blocks in one model let the receivers pull on the
QB's parameters (QB CRPS +0.076 vs production over 12 folds); the cut version
matches production on QBs (+0.011 +- 0.014) and keeps the skill gain (-0.150 +- 0.016).
See the project doc claude/joint-qb-skill-correlation.md.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import polars as pl
import pymc as pm
import pytensor.tensor as pt

from .config import (DUD_LINE, HET_GP_QB_VEGAS, HURDLE_SKILL, STARTER_COLS_FORECAST, STARTER_COLS_TRAIN,
                     TARGET_COL, SamplerSettings)
from .model import FittedModel, HetPrep, _diagnostics, fit as fit_het, payload, sample_posterior

log = logging.getLogger(__name__)

SKILL_DIAG_VARS = ('sigma_skill', 'ell_season', 'eta_season', 'alpha_pos', 'lam_rec', 'a_psi',
                   'log_cv0', 'delta_cv')
NO_QB = -1


# ------------------------------------------------------------------ starters

def _team_game() -> pl.Expr:
    return pl.concat_str([pl.col('season').cast(pl.String), pl.col('week').cast(pl.String),
                          pl.col('team')], separator='|').alias('_tg')


def starter_rows(frame_q: pl.DataFrame, frame_s: pl.DataFrame, cols) -> np.ndarray:
    """For each row of ``frame_s``: the row index in ``frame_q`` of that
    team-game's starting QB, or NO_QB (-1). The starter is the QB row ranking
    highest on ``cols`` (first column decides, the rest break ties)."""
    if frame_q.height == 0 or frame_s.height == 0:
        return np.full(frame_s.height, NO_QB, dtype=int)
    cols = [c for c in cols if c in frame_q.columns] or ['trail_pts']
    best = (frame_q.with_row_index('_qrow').with_columns(_team_game())
            .sort(cols, descending=True, nulls_last=True)
            .unique('_tg', keep='first').select('_tg', '_qrow'))
    rows = (frame_s.with_columns(_team_game()).select('_tg')
            .join(best, on='_tg', how='left', maintain_order='left')['_qrow'])
    return rows.fill_null(NO_QB).cast(pl.Int64).to_numpy()


def apply_qb_link(S0: np.ndarray, lam: np.ndarray, e_q: np.ndarray, v_q: np.ndarray,
                  qrow: np.ndarray) -> np.ndarray:
    """Receivers' draws with the QB link. S0, lam: (draws, n_skill); e_q: (draws, n_qb)
    QB surprises; v_q: (n_qb,) their variances; qrow: (n_skill,) QB column or NO_QB.
    Mean-preserving: E[exp(lam e - lam^2 var(e) / 2)] = 1 for Normal e."""
    on = qrow != NO_QB
    e = np.zeros_like(S0)
    v = np.zeros(S0.shape[1])
    e[:, on] = e_q[:, qrow[on]]
    v[on] = v_q[qrow[on]]
    return S0 * np.exp(lam * e - 0.5 * lam ** 2 * v[None, :])


# ---------------------------------------------------------------- skill model

def build_hurdle_skill(D: dict[str, np.ndarray], prep: HetPrep, trail: np.ndarray, r: np.ndarray,
                       has_qb: np.ndarray, m_ref: np.ndarray, jitter: float = 1e-6,
                       dud_line: float = DUD_LINE) -> pm.Model:
    """Stage-2 skill model. Draws of y_obs are in POINTS (no rescaling)."""
    grid = prep.grid
    X = grid[:, None]
    log_mref = np.log(m_ref)
    with pm.Model(coords=prep.coords) as model:
        player = pm.Data('player_data', D['player'])
        pos = pm.Data('position_data', D['position'])
        t = pm.Data('time_data', D['time'])
        cont = pm.Data('cont_data', D['cont'])
        bi = pm.Data('bi_data', D['bi'])
        tf = pm.Data('team_form_data', D['team_form_z'])
        otf = pm.Data('opp_team_form_data', D['opp_team_form_z'])
        obs = pm.Data('obs_data', D['y_z'])
        trail_d = pm.Data('trail_data', trail)
        r_d = pm.Data('r_qb_data', r)
        has_d = pm.Data('has_qb_data', has_qb)
        lmref = pt.as_tensor(log_mref)

        # ---- QB link (r is data: the cut) ----
        lam_rec = pm.Normal('lam_rec', 0.0, 0.3, dims='position')

        # ---- mean given no dud, log scale ----
        alpha = pm.Normal('alpha_pos', 0.15, 0.3, dims='position')
        t_peak_mu = pm.Normal('t_peak_mu', *prep.cfg.t_peak)
        t_peak_off = pm.ZeroSumNormal('t_peak_off', sigma=1.0, dims='position')
        t_peak = t_peak_mu + t_peak_off
        log_height_mu = pm.Normal('log_height_mu', np.log(0.2), 0.4)
        log_height_off = pm.ZeroSumNormal('log_height_off', sigma=0.25, dims='position')
        peak_height = pt.exp(log_height_mu + log_height_off)
        peak_width = 1.0 + pm.Gamma('peak_width_raw', alpha=6, beta=1.5)
        curve = peak_height[None, :] * pt.exp(-0.5 * ((grid[:, None] - t_peak[None, :]) / peak_width) ** 2)
        f_career = curve - (prep.t_weights_pos * curve).sum(0)

        sigma_skill = pm.HalfNormal('sigma_skill', 0.35)
        z_skill = pm.Normal('z_skill', 0.0, 1.0, dims='player_id')
        raw = sigma_skill * z_skill
        skill = raw - ((prep.pos_onehot.T @ raw) / prep.pos_counts)[prep.position_of_player]

        ell_season = pm.InverseGamma('ell_season', *prep.season_prior)
        eta_season = pm.HalfNormal('eta_season', 0.15)
        L = pt.linalg.cholesky(pm.gp.util.stabilize(
            (eta_season ** 2 * pm.gp.cov.Matern32(1, ls=ell_season))(X), jitter))
        z_season = pm.Normal('z_season', 0.0, 1.0, dims=('player_id', 'time'))
        f_raw = z_season @ L.T
        f_p = f_raw - ((f_raw * prep.active).sum(1) / prep.n_seasons_player)[:, None]
        f_season = f_p - (f_p * prep.active).sum(0) / prep.n_active

        cont_prior = pm.Normal('cont_prior', 0.0, 0.08, dims='cont_vars')
        bi_prior = pm.Normal('bi_prior', 0.0, 0.08, dims='binary_vars')
        team_form_prior = pm.Normal('team_form_prior', 0.0, 0.08)
        opp_form_prior = pm.Normal('opp_form_prior', 0.0, 0.08)

        link = has_d * (lam_rec[pos] * r_d - 0.5 * lam_rec[pos] ** 2)
        eta = (lmref[pos] + alpha[pos] + f_career[t, pos] + skill[player] + f_season[player, t]
               + pt.dot(cont, cont_prior) + pt.dot(bi, bi_prior)
               + team_form_prior * tf + opp_form_prior * otf + link)
        m = pt.exp(eta)

        # ---- dud probability: its own predictors, not tied to the mean ----
        y_pts = prep.y_mu + prep.y_sd * obs
        y_h = pt.switch(y_pts > dud_line, y_pts, 0.0)
        a_psi = pm.Normal('a_psi', 2.0, 0.5, dims='position')
        w_psi = pm.Normal('w_psi', 0.0, 0.5, dims='cont_vars')
        wb_psi = pm.Normal('wb_psi', 0.0, 0.5, dims='binary_vars')
        psi = pm.math.sigmoid(a_psi[pos] + pt.dot(cont, w_psi) + pt.dot(bi, wb_psi))

        # ---- gamma CV falls with the trailing level, smoothly capped at 1.2 ----
        log_cv0 = pm.Normal('log_cv0', np.log(0.6), 0.15, dims='position')
        delta = pm.Normal('delta_cv', -0.3, 0.2, dims='position')
        lvl = pt.clip(pt.log(pt.maximum(trail_d, 1.0)) - lmref[pos], -1.5, 1.5)
        cv_raw = pt.exp(log_cv0[pos] + delta[pos] * lvl)
        cv = cv_raw / pt.sqrt(1.0 + (cv_raw / 1.2) ** 2)
        pm.HurdleGamma('y_obs', psi=psi, mu=m, sigma=cv * m, observed=y_h)
    return model


@dataclass
class FittedSkill:
    prep: HetPrep
    model: pm.Model
    idata: object
    settings: SamplerSettings
    m_ref: np.ndarray                 # training mean points per position (prep.positions order)
    fit_time: float
    divergences: int
    max_rhat: float | None
    min_ess_bulk: float | None

    @property
    def n_train(self) -> int:
        return len(self.prep.train_arrays['y'])

    def trail(self, frame: pl.DataFrame) -> np.ndarray:
        """trail_pts with no history filled by the position's training mean."""
        fill = dict(zip(self.prep.positions, self.m_ref))
        return (frame.select(pl.col('trail_pts').cast(pl.Float64)
                             .fill_null(pl.col('position').replace_strict(fill, return_dtype=pl.Float64)))
                .to_series().to_numpy())

    def predict_base(self, frame: pl.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray]:
        """(S0, lam): draws in points with the QB link off, and the lam_rec draw
        for each row's position; both (n_draws, len(frame))."""
        ok = self.prep.eligible(frame)
        if not ok.all():
            raise ValueError(f'{HURDLE_SKILL.name}: {(~ok).sum()} rows are not scoreable by this fit')
        n = frame.height
        D = self.prep.arrays(frame)
        with self.model:
            pm.set_data({**payload(D), 'trail_data': self.trail(frame),
                         'r_qb_data': np.zeros(n), 'has_qb_data': np.zeros(n)})
            ppc = pm.sample_posterior_predictive(self.idata, var_names=['y_obs'], progressbar=False,
                                                 random_seed=seed)
        S0 = ppc.posterior_predictive['y_obs'].values.reshape(-1, n)
        lam = self.idata.posterior['lam_rec'].values.reshape(-1, len(self.prep.positions))[:, D['position']]
        return S0, lam

    def loadings(self) -> dict[str, float]:
        lam = self.idata.posterior['lam_rec'].mean(('chain', 'draw')).values
        return {p: float(v) for p, v in zip(self.prep.positions, lam)}

    def coefs(self) -> pl.DataFrame:
        """Posterior mean / sd of the covariate effects (log points per sd of x)
        and the QB loadings."""
        post = self.idata.posterior
        rows = []
        for var, dim in (('cont_prior', 'cont_vars'), ('bi_prior', 'binary_vars'), ('lam_rec', 'position')):
            da = post[var]
            for name, m, s in zip(da[dim].values, da.mean(('chain', 'draw')).values,
                                  da.std(('chain', 'draw')).values):
                rows.append({'var': f'lam_rec[{name}]' if var == 'lam_rec' else str(name),
                             'coef_mean': float(m), 'coef_sd': float(s)})
        for var in ('team_form_prior', 'opp_form_prior'):
            rows.append({'var': var, 'coef_mean': float(post[var].mean()), 'coef_sd': float(post[var].std())})
        return pl.DataFrame(rows)


def fit_skill(train_s: pl.DataFrame, train_q: pl.DataFrame, r_q: np.ndarray,
              settings: SamplerSettings = SamplerSettings()) -> FittedSkill:
    """Stage 2. ``train_q`` / ``r_q``: the QB training rows and their residuals
    in the same order. ``train_s`` needs a ``trail_pts`` column."""
    cfg = HURDLE_SKILL
    train_s = (train_s.filter(pl.col('position').is_in(list(cfg.positions)))
               .sort('player_id', 'season', 'week'))
    prep = HetPrep(train_s, cfg)
    m_ref = (train_s.group_by('position').agg(pl.col(TARGET_COL).mean())
             .join(pl.DataFrame({'position': prep.positions}), on='position', how='right')
             [TARGET_COL].to_numpy())
    qrow = starter_rows(train_q, train_s, STARTER_COLS_TRAIN)
    has = (qrow != NO_QB).astype(float)
    r = np.where(qrow != NO_QB, np.append(r_q, 0.0)[qrow], 0.0)
    fs = FittedSkill(prep=prep, model=None, idata=None, settings=settings, m_ref=m_ref, fit_time=0.0,
                     divergences=0, max_rhat=None, min_ess_bulk=None)
    fs.model = build_hurdle_skill(prep.train_arrays, prep, fs.trail(train_s), r, has, m_ref)
    t0 = time.time()
    fs.idata = sample_posterior(fs.model, settings)
    fs.fit_time = time.time() - t0
    if settings.method == 'nuts':
        fs.divergences = int(fs.idata.sample_stats['diverging'].sum())
        fs.max_rhat, fs.min_ess_bulk = _diagnostics(fs.idata, settings.chains, SKILL_DIAG_VARS)
    log.info('%s: fit %d rows / %d players in %.0fs (QB-matched %.0f%%, divergences=%d, max r_hat=%s); '
             'lam_rec %s', cfg.name, train_s.height, len(prep.players), fs.fit_time, 100 * has.mean(),
             fs.divergences, None if fs.max_rhat is None else f'{fs.max_rhat:.3f}',
             {k: round(v, 3) for k, v in fs.loadings().items()})
    if fs.divergences:
        log.warning('%s: %d divergent transitions', cfg.name, fs.divergences)
    if fs.max_rhat is not None and fs.max_rhat > 1.05:
        log.warning('%s: max r_hat %.3f > 1.05', cfg.name, fs.max_rhat)
    return fs


# ---------------------------------------------------------------- both stages

def qb_residuals(fm_q: FittedModel, train_q: pl.DataFrame, seed: int | None = None) -> np.ndarray:
    """Standardized game residual of each QB training row (train_q's order):
    (points - E[predictive]) / E[sigma]. The predictive mean over all draws
    stands in for E[mu]; its Monte Carlo error is sigma / sqrt(n_draws)."""
    S = fm_q.predict(train_q, seed=seed)
    y = fm_q.prep.arrays(train_q)['y']
    sigma_pts = float(fm_q.idata.posterior['sigma_obs'].mean()) * fm_q.prep.y_sd
    return (y - S.mean(0)) / sigma_pts


@dataclass
class QBLinkedFit:
    qb: FittedModel
    skill: FittedSkill
    r_train: np.ndarray

    def predict(self, frame_s: pl.DataFrame, frame_q: pl.DataFrame,
                seed: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(S_s, S_q, qrow): skill and QB draws in points with the link applied, and
        each skill row's QB column in frame_q (NO_QB when his team has no scored QB).
        Rows must be eligible for their model; frame_s needs trail_pts."""
        seed = self.qb.settings.seed + 1 if seed is None else seed
        S_q = self.qb.predict(frame_q, seed=seed)
        S0, lam = self.skill.predict_base(frame_s, seed=seed + 1)
        n = min(len(S0), len(S_q)) if frame_q.height else len(S0)
        S0, lam, S_q = S0[:n], lam[:n], S_q[:n]
        qrow = starter_rows(frame_q, frame_s, STARTER_COLS_FORECAST)
        if frame_q.height == 0:
            return S0, S_q, qrow
        sigma_pts = float(self.qb.idata.posterior['sigma_obs'].mean()) * self.qb.prep.y_sd
        e_q = (S_q - S_q.mean(0)) / sigma_pts          # QB surprise, on the scale of the training r
        return apply_qb_link(S0, lam, e_q, e_q.var(0), qrow), S_q, qrow


def fit_qb_linked(train_s: pl.DataFrame, train_q: pl.DataFrame,
                  settings: SamplerSettings = SamplerSettings()) -> QBLinkedFit:
    """Both stages. ``train_s`` needs trail_pts; ``train_q`` the starter columns
    (offense_pct; falls back to trail_pts)."""
    train_q = (train_q.filter(pl.col('position').is_in(list(HET_GP_QB_VEGAS.positions)))
               .sort('player_id', 'season', 'week'))
    fm_q = fit_het(train_q, HET_GP_QB_VEGAS, settings)
    r_q = qb_residuals(fm_q, train_q, seed=settings.seed + 7)
    log.info('QB training residuals: mean %.3f, sd %.3f (expect ~0, ~1)', r_q.mean(), r_q.std())
    fs = fit_skill(train_s, train_q, r_q, settings)
    return QBLinkedFit(qb=fm_q, skill=fs, r_train=r_q)
