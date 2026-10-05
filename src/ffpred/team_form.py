"""Team form: weekly offense / defense strength from game scores.

Port of ``src/team_ability_ssm.py`` (the non-division ``TeamOffDef`` model),
which is what wrote the ``processed-data/team_form.parquet`` the production
models were validated with. (``src/estimate_latent_ability.py`` still says it
writes that file, but its LevelTrend output was replaced.)

Model, as a pymc_extras state space so the Kalman filter integrates the
latent path out and NUTS only samples ~30 static parameters:

    y[t, i]  = points scored by team i in global week t (NaN on a bye)
    state    = [off_1..off_N, def_1..def_N]
    y[t, i]  = mu_s + sign_i*(hfa_s + beta_rest*rest)/2 + off_i - def_opp(i,t) + e
    off/def: random walk within a season; at each season boundary
             x -> rho * x + offseason shock
    cov(e_i, e_opp) = rho_obs * sigma_obs^2   (the two scores in one game)

Output contract (``data.join_team_form`` joins it as-of the previous week):
    season, week, team, team_form (= off + def, margin scale),
    team_off_form, team_def_form
team_form at (season, week) is the posterior-mean FILTERED state (through
that week's game). Each season also gets a week-0 row with the PREDICTED
state for week 1 (last season's form after the offseason regression).

Known limitation kept from src/: the static parameters are estimated on all
seasons passed in, so for a historical week the filter's noise levels have
seen later seasons. Use ``through_season`` for a strict walk-forward fit.
"""

from __future__ import annotations

import logging

import numpy as np
import polars as pl
import pytensor.tensor as pt
from pymc_extras.statespace.core import PyMCStateSpace
from pymc_extras.statespace.core.properties import Coord, Parameter, Shock, State

log = logging.getLogger(__name__)

COMPONENTS = ('off', 'def')


def games_from_processed(processed: pl.DataFrame) -> pl.DataFrame:
    return (processed.unique(subset='game_id')
            .select('game_id', 'season', 'week', 'home_team', 'away_team',
                    'home_score', 'away_score', 'home_rest', 'away_rest'))


def build_week_arrays(games: pl.DataFrame, teams, seasons):
    """The (week x team) score matrix plus the fixed numpy pieces of the
    time-varying statespace matrices."""
    global_week = games.select('season', 'week').unique().sort(['season', 'week']).with_row_index('global_week')
    g = games.join(global_week, on=['season', 'week'])
    team_pos = {t: i for i, t in enumerate(teams)}
    season_pos = {s: i for i, s in enumerate(seasons)}
    T, N = global_week.height, len(teams)

    y = np.full((T, N), np.nan)
    opp = np.full((T, N), -1)
    sign = np.zeros((T, N))          # +1 home, -1 away, 0 bye
    rest = np.zeros((T, N))          # rest advantage from this team's side, in weeks (clipped +-1)
    for r in g.iter_rows(named=True):
        t, h, a = r['global_week'], team_pos[r['home_team']], team_pos[r['away_team']]
        rd = float(np.clip(r['home_rest'] - r['away_rest'], -7, 7)) / 7
        y[t, h], y[t, a] = r['home_score'], r['away_score']
        opp[t, h], opp[t, a] = a, h
        sign[t, h], sign[t, a] = 1.0, -1.0
        rest[t, h], rest[t, a] = rd, -rd

    played = sign != 0
    Z = np.zeros((T, N, 2 * N))
    tt, ii = np.nonzero(played)
    Z[tt, ii, ii] = 1.0                       # + own offense
    Z[tt, ii, N + opp[tt, ii]] = -1.0         # - opponent's defense
    pair = np.zeros((T, N, N))
    pair[tt, ii, opp[tt, ii]] = 1.0

    season_of_week = global_week['season'].replace_strict(season_pos).to_numpy()
    # boundary[t] = 1 if the transition t -> t+1 crosses an offseason
    boundary = np.r_[season_of_week[1:] != season_of_week[:-1], False].astype(float)
    return y, global_week, dict(Z=Z, pair=pair, sign=sign, rest=rest, season_of_week=season_of_week,
                                boundary=boundary)


class TeamOffDef(PyMCStateSpace):
    def __init__(self, teams, seasons, arrays, verbose=False):
        self.teams, self.seasons, self.arrays = list(teams), list(seasons), arrays
        self.n_team = N = len(self.teams)
        super().__init__(k_endog=N, k_states=2 * N, k_posdef=2 * N, measurement_error=True, verbose=verbose)

    def set_states(self):
        hidden = [State(name=f'{c}[{t}]', observed=False) for c in COMPONENTS for t in self.teams]
        observed = [State(name=t, observed=True) for t in self.teams]
        return *hidden, *observed

    def set_shocks(self):
        return tuple(Shock(name=f'{c}[{t}]') for c in COMPONENTS for t in self.teams)

    def set_coords(self):
        return (*self.default_coords(), Coord(dimension='comp', labels=COMPONENTS),
                Coord(dimension='season', labels=tuple(self.seasons)))

    def set_parameters(self):
        S = len(self.seasons)
        return (
            Parameter(name='init_sd', shape=(2,), dims=('comp',), constraints='Positive'),
            Parameter(name='sigma_week', shape=(2,), dims=('comp',), constraints='Positive'),
            Parameter(name='sigma_season', shape=(2,), dims=('comp',), constraints='Positive'),
            Parameter(name='rho', shape=(2,), dims=('comp',), constraints='(0, 1)'),
            Parameter(name='mu', shape=(S,), dims=('season',)),
            Parameter(name='hfa', shape=(S,), dims=('season',)),
            Parameter(name='beta_rest', shape=()),
            Parameter(name='sigma_obs', shape=(), constraints='Positive'),
            Parameter(name='rho_obs', shape=(), constraints='(-1, 1)'),
        )

    def make_symbolic_graph(self):
        N, S = self.n_team, len(self.seasons)
        A = self.arrays
        v = {name: self.make_and_register_variable(name, shape=shape) for name, shape in (
            ('init_sd', (2,)), ('sigma_week', (2,)), ('sigma_season', (2,)), ('rho', (2,)),
            ('mu', (S,)), ('hfa', (S,)), ('beta_rest', ()), ('sigma_obs', ()), ('rho_obs', ()))}
        rep = lambda x: pt.repeat(x, N)  # noqa: E731
        eye = pt.eye(2 * N)
        boundary = pt.as_tensor(A['boundary'])[:, None]

        self.ssm['initial_state', :] = pt.zeros(2 * N)
        self.ssm['initial_state_cov', :, :] = pt.diag(rep(v['init_sd']) ** 2)
        self.ssm['selection', :, :] = eye
        t_diag = boundary * rep(v['rho'])[None, :] + (1 - boundary)
        self.ssm['transition'] = t_diag[:, :, None] * eye[None]
        q_diag = boundary * rep(v['sigma_season'])[None, :] ** 2 + (1 - boundary) * rep(v['sigma_week'])[None, :] ** 2
        self.ssm['state_cov'] = q_diag[:, :, None] * eye[None]
        self.ssm['design'] = pt.as_tensor(A['Z'])
        s_idx = A['season_of_week']
        sign, rest = pt.as_tensor(A['sign']), pt.as_tensor(A['rest'])
        self.ssm['obs_intercept'] = (v['mu'][s_idx][:, None] * pt.abs(sign)
                                     + 0.5 * sign * v['hfa'][s_idx][:, None] + 0.5 * v['beta_rest'] * rest)
        self.ssm['obs_cov'] = v['sigma_obs'] ** 2 * (pt.eye(N)[None] + v['rho_obs'] * pt.as_tensor(A['pair']))
        self.ssm.declare_time_varying('transition', 'state_cov', 'design', 'obs_intercept', 'obs_cov')


def team_form_from_filter(kf, teams, global_week) -> pl.DataFrame:
    """sample_filter_outputs(...) -> the team_form.parquet contract."""
    pp = kf.posterior_predictive
    off_lab, def_lab = [f'off[{t}]' for t in teams], [f'def[{t}]' for t in teams]

    def comps(name):
        x = pp[name]
        o = x.sel(state=off_lab).mean(('chain', 'draw')).values
        d = x.sel(state=def_lab).mean(('chain', 'draw')).values
        return o - o.mean(-1, keepdims=True), d - d.mean(-1, keepdims=True)   # only differences identified

    def frame(o, d, t_idx):
        n = len(teams)
        return pl.DataFrame({'global_week': np.repeat(t_idx, n), 'team': np.tile(teams, len(t_idx)),
                             'team_off_form': o[t_idx].ravel(), 'team_def_form': d[t_idx].ravel()}
                            ).with_columns((pl.col('team_off_form') + pl.col('team_def_form')).alias('team_form'))

    gw = global_week.with_columns(pl.col('global_week').cast(pl.Int64))
    o, d = comps('filtered_states')
    filt = frame(o, d, np.arange(gw.height)).join(gw, on='global_week')
    season_start = gw.group_by('season').agg(pl.col('global_week').min()).sort('season')['global_week'].to_numpy()
    o, d = comps('predicted_states')
    week0 = (frame(o, d, season_start).join(gw.select('global_week', 'season'), on='global_week')
             .with_columns(pl.lit(0, dtype=gw['week'].dtype).alias('week')))
    cols = ['season', 'week', 'team', 'team_form', 'team_off_form', 'team_def_form']
    out = (pl.concat([week0.select(cols), filt.select(cols)])
           .with_columns(pl.col('week').cast(pl.Int64)).sort('season', 'week', 'team'))
    if out.height != out.select('season', 'week', 'team').unique().height:
        raise RuntimeError('team_form has duplicate (season, week, team) keys')
    return out


def fit_team_form(processed: pl.DataFrame, through_season: int | None = None, draws: int = 1000,
                  tune: int = 1000, chains: int = 4, seed: int = 0, nuts_sampler: str = 'nutpie',
                  filter_draws: int = 400) -> pl.DataFrame:
    """Fit TeamOffDef on every game in ``processed`` and return team_form.

    Kalman outputs are not stored during sampling (4 x weeks x 64 x 64
    covariances per draw); filtered / predicted states are computed afterwards
    on at most ``filter_draws`` posterior draws."""
    import pymc as pm

    games = games_from_processed(processed)
    if through_season is not None:
        games = games.filter(pl.col('season') <= through_season)
    teams = sorted(games['home_team'].unique().to_list())
    seasons = sorted(games['season'].unique().to_list())
    wide, global_week, arrays = build_week_arrays(games, teams, seasons)
    ss = TeamOffDef(teams, seasons, arrays)

    with pm.Model(coords=ss.coords):
        mu_bar, mu_sd = pm.Normal('mu_bar', 22, 5), pm.HalfNormal('mu_sd', 2)
        pm.Deterministic('mu', mu_bar + mu_sd * pm.Normal('mu_z', 0, 1, dims='season'), dims='season')
        hfa_bar, hfa_sd = pm.Normal('hfa_bar', 2, 2), pm.HalfNormal('hfa_sd', 1)
        pm.Deterministic('hfa', hfa_bar + hfa_sd * pm.Normal('hfa_z', 0, 1, dims='season'), dims='season')
        pm.Normal('beta_rest', 0, 2)
        pm.HalfNormal('init_sd', 5, dims='comp')
        pm.HalfNormal('sigma_week', 1.0, dims='comp')
        pm.HalfNormal('sigma_season', 3.0, dims='comp')
        pm.Beta('rho', 4, 3, dims='comp')
        pm.HalfNormal('sigma_obs', 12)
        pm.Uniform('rho_obs', -0.5, 0.8)
        ss.build_statespace_graph(wide, missing_fill_value=-9999.0, mvn_method='cholesky')
        idata = pm.sample(draws=draws, tune=tune, chains=chains, random_seed=seed, progressbar=False,
                          nuts_sampler=nuts_sampler)

    n_div = int(idata.sample_stats['diverging'].sum())
    if n_div:
        log.warning('team form: %d divergent transitions', n_div)
    step = max(1, -(-chains * draws // filter_draws))      # ceil division
    kf = ss.sample_filter_outputs(idata.sel(draw=slice(None, None, step)),
                                  filter_output_names=['filtered_states', 'predicted_states'],
                                  compile_kwargs={'mode': 'NUMBA'}, progressbar=False)
    out = team_form_from_filter(kf, teams, global_week)
    log.info('team form: %d team-weeks, seasons %d-%d', out.height, min(seasons), max(seasons))
    return out
