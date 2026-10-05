"""
The dynamic offense/defense team model, written as a pymc_extras state space
model so the Kalman filter does the work instead of NUTS.

Same statistical model as the latent-path version (off/def random walk within a
season, rho-reversion + offseason shock between seasons, opponent-adjusted
scores, pooled sigmas, per-season scoring level and HFA). What changes is how
it is fit:

  - latent-path version: NUTS samples ~14.5k latent states + ~30 parameters
  - this version:        the Kalman filter integrates the states out exactly,
                         NUTS only samples the ~30 static parameters

and you get filtered / predicted / smoothed states from pymc_extras the same way
as the old LevelTrend model.

Layout (same wide (week x team) matrix as the old model, but a different
number goes in it):

    y[t, i]   = points scored by team i in global week t (NaN on a bye)
    state     = [off_1..off_32, def_1..def_32]
    y[t, i]   = mu_s + sign_i*(hfa_s + beta_rest*rest)/2 + off_i - def_opp(i,t) + e
    cov(e_i, e_opp) = rho_obs * sigma_obs^2   (the two scores in a game)

Every game enters exactly once per team's score -- no double counting -- and
the opponent's defense is in the same equation, so results are
opponent-adjusted. The five statespace matrices that change week to week
(design, obs_intercept, obs_cov, transition, state_cov) are declared
time-varying.

Timing (pymc_extras / statsmodels convention): x0/P0 are the predicted state
for week 0; after updating on week t the filter predicts week t+1 with
transition[t] and state_cov[t]. So the season-boundary entries sit at the LAST
week of each season.

TeamOffDefDivision (below) adds the division model: each season start reverts
toward the team's own long-run mean (pooled by division) instead of league
average. Set DIVISION = True in the __main__ block to fit it.
"""

import numpy as np
import polars as pl
import pytensor.tensor as pt

from pymc_extras.statespace.core import PyMCStateSpace
from pymc_extras.statespace.core.properties import Coord, Parameter, Shock, State

COMPONENTS = ('off', 'def')

DIVISIONS = {
    'AFC East': ['BUF', 'MIA', 'NE', 'NYJ'],   'AFC North': ['BAL', 'CIN', 'CLE', 'PIT'],
    'AFC South': ['HOU', 'IND', 'JAX', 'TEN'], 'AFC West': ['DEN', 'KC', 'LAC', 'LV'],
    'NFC East': ['DAL', 'NYG', 'PHI', 'WAS'],  'NFC North': ['CHI', 'DET', 'GB', 'MIN'],
    'NFC South': ['ATL', 'CAR', 'NO', 'TB'],   'NFC West': ['ARI', 'LA', 'SEA', 'SF'],
}


def build_week_arrays(games, teams, seasons):
    """games: one row per game with season, week, home_team, away_team,
    home_score, away_score, home_rest, away_rest.

    Returns the (week x team) observation matrix plus the fixed numpy pieces of
    the time-varying matrices."""
    global_week = (games.select('season', 'week').unique().sort(['season', 'week'])
                   .with_row_index('global_week'))
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
    # design: y[t,i] loads +1 on off_i and -1 on def_opp
    Z = np.zeros((T, N, 2 * N))
    tt, ii = np.nonzero(played)
    Z[tt, ii, ii] = 1.0
    Z[tt, ii, N + opp[tt, ii]] = -1.0
    # pairs of opponents, for the within-game score correlation
    pair = np.zeros((T, N, N))
    pair[tt, ii, opp[tt, ii]] = 1.0

    season_of_week = global_week['season'].replace_strict(season_pos).to_numpy()
    # boundary[t] = 1 if week t is the last week of a season (transition t -> t+1 crosses the offseason)
    boundary = np.r_[season_of_week[1:] != season_of_week[:-1], False].astype(float)

    arrays = dict(Z=Z, pair=pair, sign=sign, rest=rest, season_of_week=season_of_week,
                  boundary=boundary)
    return y, global_week, arrays


class TeamOffDef(PyMCStateSpace):
    def __init__(self, teams, seasons, arrays, verbose=True):
        self.teams = list(teams)
        self.seasons = list(seasons)
        self.arrays = arrays
        N = len(self.teams)
        self.n_team = N
        super().__init__(k_endog=N, k_states=2 * N, k_posdef=2 * N,
                         measurement_error=True, verbose=verbose)

    # ---- metadata --------------------------------------------------------
    def set_states(self):
        hidden = [State(name=f'{c}[{t}]', observed=False) for c in COMPONENTS for t in self.teams]
        observed = [State(name=t, observed=True) for t in self.teams]
        return *hidden, *observed

    def set_shocks(self):
        return tuple(Shock(name=f'{c}[{t}]') for c in COMPONENTS for t in self.teams)

    def set_coords(self):
        # default state / observed_state / shock coords (needed by
        # sample_filter_outputs) plus this model's own
        return (*self.default_coords(),
                Coord(dimension='comp', labels=COMPONENTS),
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

    # ---- matrices --------------------------------------------------------
    def make_symbolic_graph(self):
        N, S = self.n_team, len(self.seasons)
        A = self.arrays
        T = A['Z'].shape[0]

        init_sd = self.make_and_register_variable('init_sd', shape=(2,))
        sigma_week = self.make_and_register_variable('sigma_week', shape=(2,))
        sigma_season = self.make_and_register_variable('sigma_season', shape=(2,))
        rho = self.make_and_register_variable('rho', shape=(2,))
        mu = self.make_and_register_variable('mu', shape=(S,))
        hfa = self.make_and_register_variable('hfa', shape=(S,))
        beta_rest = self.make_and_register_variable('beta_rest', shape=())
        sigma_obs = self.make_and_register_variable('sigma_obs', shape=())
        rho_obs = self.make_and_register_variable('rho_obs', shape=())

        # per-state (64,) versions of the component parameters
        rep = lambda v: pt.repeat(v, N)
        eye = pt.eye(2 * N)
        boundary = pt.as_tensor(A['boundary'])[:, None]                  # (T, 1)

        # initial state: league average, spread init_sd
        self.ssm['initial_state', :] = pt.zeros(2 * N)
        self.ssm['initial_state_cov', :, :] = pt.diag(rep(init_sd) ** 2)
        self.ssm['selection', :, :] = eye

        # transition: identity in season, rho at the offseason
        t_diag = boundary * rep(rho)[None, :] + (1 - boundary)            # (T, 2N)
        self.ssm['transition'] = t_diag[:, :, None] * eye[None]
        # shocks: weekly drift in season, offseason shock at the boundary
        q_diag = boundary * rep(sigma_season)[None, :] ** 2 + (1 - boundary) * rep(sigma_week)[None, :] ** 2
        self.ssm['state_cov'] = q_diag[:, :, None] * eye[None]

        # observation: off_i - def_opp, plus scoring level / home field / rest
        self.ssm['design'] = pt.as_tensor(A['Z'])
        s_idx = A['season_of_week']
        sign, rest = pt.as_tensor(A['sign']), pt.as_tensor(A['rest'])
        self.ssm['obs_intercept'] = (
            mu[s_idx][:, None] * pt.abs(sign)
            + 0.5 * sign * hfa[s_idx][:, None]
            + 0.5 * beta_rest * rest
        )
        # correlated scores within a game
        self.ssm['obs_cov'] = sigma_obs ** 2 * (pt.eye(N)[None] + rho_obs * pt.as_tensor(A['pair']))

        self.ssm.declare_time_varying('transition', 'state_cov', 'design', 'obs_intercept', 'obs_cov')


class TeamOffDefDivision(TeamOffDef):
    """TeamOffDef with team long-run means (the division model).

    One extra parameter, team_mean (comp, team) -- give it the division
    hierarchy in the PyMC model (division_priors below). Two matrices change:

      initial_state     = team_mean                       (first season starts at the long-run mean)
      state_intercept_t = boundary_t * (1 - rho) * team_mean   (time-varying)

    With transition_t = rho at the season boundary, the new season starts at
        rho * x_end + (1 - rho) * team_mean  =  team_mean + rho * (x_end - team_mean),
    i.e. the latent-path division model's reversion. Everything else is TeamOffDef.
    """

    def set_coords(self):
        return (*super().set_coords(),
                Coord(dimension='team', labels=tuple(self.teams)),
                Coord(dimension='division', labels=tuple(DIVISIONS)))

    def set_parameters(self):
        return (*super().set_parameters(),
                Parameter(name='team_mean', shape=(2, self.n_team), dims=('comp', 'team')))

    def make_symbolic_graph(self):
        super().make_symbolic_graph()
        N = self.n_team
        tm = self.make_and_register_variable('team_mean', shape=(2, N)).ravel()   # [off..., def...]
        rho = self._name_to_variable['rho']
        boundary = pt.as_tensor(self.arrays['boundary'])[:, None]                  # (T, 1)

        self.ssm['initial_state', :] = tm
        self.ssm['state_intercept'] = boundary * (1 - pt.repeat(rho, N))[None, :] * tm[None, :]
        self.ssm.declare_time_varying('state_intercept')


def division_priors(teams, init_sd_scale=3.0):
    """Inside a pm.Model built with TeamOffDefDivision.coords: team_mean with
    partial pooling by division (same priors as the latent-path division model),
    plus its init_sd -- deviation of the first season from the long-run mean."""
    import pymc as pm
    team_div = {t: d for d, ts in DIVISIONS.items() for t in ts}
    div_idx = np.array([list(DIVISIONS).index(team_div[t]) for t in teams])
    sigma_div = pm.HalfNormal('sigma_div', 1.5, dims='comp')
    sigma_team = pm.HalfNormal('sigma_team', 2.0, dims='comp')
    div_eff = pm.Deterministic(
        'div_eff', sigma_div[:, None] * pm.Normal('div_z', 0, 1, dims=('comp', 'division')),
        dims=('comp', 'division'))
    tm = div_eff[:, div_idx] + sigma_team[:, None] * pm.Normal('team_z', 0, 1, dims=('comp', 'team'))
    pm.Deterministic('team_mean', tm - tm.mean(axis=-1, keepdims=True), dims=('comp', 'team'))
    pm.HalfNormal('init_sd', init_sd_scale, dims='comp')


def team_form_from_filter(kf, teams, global_week, seasons):
    """Turn sample_filter_outputs(...) into the team_form.parquet contract used
    by models.join_team_form:

      (season, week, team, team_form, team_off_form, team_def_form)

    - team_form at (season, week) is the FILTERED state, i.e. strength through
      that week including its game (join_team_form joins it as-of the previous
      week, so there is no leak).
    - a week-0 row per season holds the PREDICTED state for week 1, i.e. last
      season's final form after the offseason regression, so the as-of join
      picks it up for week 1 instead of carrying week 18 forward unchanged.
    - off and def are centered across teams each week (only differences are
      identified); team_form = off + def is on the margin scale.
    - posterior mean over the draws passed to sample_filter_outputs.
    """
    pp = kf.posterior_predictive
    off_lab = [f'off[{t}]' for t in teams]
    def_lab = [f'def[{t}]' for t in teams]

    def comps(name):
        x = pp[name]
        o = x.sel(state=off_lab).mean(('chain', 'draw')).values      # (time, team)
        d = x.sel(state=def_lab).mean(('chain', 'draw')).values
        o = o - o.mean(-1, keepdims=True)
        d = d - d.mean(-1, keepdims=True)
        return o, d

    def frame(o, d, t_idx):
        n = len(teams)
        return pl.DataFrame({
            'global_week': np.repeat(t_idx, n),
            'team': np.tile(teams, len(t_idx)),
            'team_off_form': o[t_idx].ravel(),
            'team_def_form': d[t_idx].ravel(),
        }).with_columns((pl.col('team_off_form') + pl.col('team_def_form')).alias('team_form'))

    gw = global_week.with_columns(pl.col('global_week').cast(pl.Int64))
    T = gw.height
    o, d = comps('filtered_states')
    filt = frame(o, d, np.arange(T)).join(gw, on='global_week')

    season_start = gw.group_by('season').agg(pl.col('global_week').min()).sort('season')['global_week'].to_numpy()
    o, d = comps('predicted_states')
    week0 = (frame(o, d, season_start).join(gw.select('global_week', 'season'), on='global_week')
             .with_columns(pl.lit(0, dtype=gw['week'].dtype).alias('week')))

    cols = ['season', 'week', 'team', 'team_form', 'team_off_form', 'team_def_form']
    out = (pl.concat([week0.select(cols), filt.select(cols)])
           .with_columns(pl.col('week').cast(pl.Int64))
           .sort('season', 'week', 'team'))
    assert out.height == out.select('season', 'week', 'team').unique().height
    return out


# ---------------------------------------------------------------------------
# fit, in the same shape as the old script
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import pymc as pm
    import arviz as az

    dat = pl.read_parquet('processed-data/ff-processed.parquet').unique(subset='game_id')
    games = dat.select('game_id', 'season', 'week', 'home_team', 'away_team',
                       'home_score', 'away_score', 'home_rest', 'away_rest')
    teams = sorted(games['home_team'].unique().to_list())
    seasons = sorted(games['season'].unique().to_list())

    DIVISION = False    # True: team long-run means pooled by division (TeamOffDefDivision)

    wide, global_week, arrays = build_week_arrays(games, teams, seasons)
    ss_mod = (TeamOffDefDivision if DIVISION else TeamOffDef)(teams, seasons, arrays)

    with pm.Model(coords=ss_mod.coords) as team_ssm:
        mu_bar = pm.Normal('mu_bar', 22, 5)
        mu_sd = pm.HalfNormal('mu_sd', 2)
        pm.Deterministic('mu', mu_bar + mu_sd * pm.Normal('mu_z', 0, 1, dims='season'), dims='season')
        hfa_bar = pm.Normal('hfa_bar', 2, 2)
        hfa_sd = pm.HalfNormal('hfa_sd', 1)
        pm.Deterministic('hfa', hfa_bar + hfa_sd * pm.Normal('hfa_z', 0, 1, dims='season'), dims='season')
        pm.Normal('beta_rest', 0, 2)

        if DIVISION:
            division_priors(teams)                     # team_mean + its init_sd
        else:
            pm.HalfNormal('init_sd', 5, dims='comp')
        pm.HalfNormal('sigma_week', 1.0, dims='comp')
        pm.HalfNormal('sigma_season', 3.0, dims='comp')
        pm.Beta('rho', 4, 3, dims='comp')

        pm.HalfNormal('sigma_obs', 12)
        pm.Uniform('rho_obs', -0.5, 0.8)

        # don't save KF outputs during sampling: 4 x (226, 64, 64) covariances per draw
        ss_mod.build_statespace_graph(wide, missing_fill_value=-9999.0, mvn_method='cholesky')
        id_ssm = pm.sample(nuts_sampler='nutpie')

    az.summary(id_ssm, var_names=['hfa_bar', 'beta_rest', 'init_sd', 'sigma_week', 'sigma_season',
                                  'rho', 'sigma_obs', 'rho_obs'])

    # filtered / predicted / smoothed states afterwards, on a thinned posterior
    kf = ss_mod.sample_filter_outputs(
        id_ssm.sel(draw=slice(None, None, 10)),
        filter_output_names=['filtered_states', 'predicted_states', 'smoothed_states'],
        compile_kwargs={'mode': 'NUMBA'})
    # states are [off[team]..., def[team]...]; team strength on the margin scale = off + def.
    # Only differences between teams are identified, so center across teams each week before use.

    team_form = team_form_from_filter(kf, teams, global_week, seasons)
    suffix = '_division' if DIVISION else ''
    team_form.write_parquet(f'processed-data/team_form{suffix}.parquet')
    id_ssm.to_netcdf(f'model-nc/team_ability_ssm{suffix}.nc')   # small: static parameters only, no latent path
