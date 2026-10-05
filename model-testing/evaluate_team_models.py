"""
Out-of-sample comparison of the two latent team-ability models.

Why not LOO / compute_log_likelihood: the two models have likelihoods on
different data (old: 32-dim per-team adjusted margins by week; new: raw home/
away scores per game), so their ELPDs aren't on the same scale. Instead, both
models predict the SAME target before kickoff and are scored on it:

    target = a team's margin in a game (one row per team per game -- the old
             model's native unit; for the new model the two rows of a game are
             mirror images)

For every posterior draw of the static parameters, a Kalman filter runs forward
through the season using only games before week t, giving a Normal predictive
for each week-t game. Mixing over draws gives the full posterior predictive,
which is scored with:

    elpd      mean log predictive density (higher = better)
    crps      continuous ranked probability score, in points (lower = better)
    rmse/mae  of the predictive mean
    cov50/80/95, width50/80/95   central interval coverage and width
    PIT       histogram should be flat if calibrated

Vegas (spread_line) is included as a point-forecast reference (rmse/mae only).

Caveat shared by both models: the ~static parameters (sigmas, hfa, rho ...)
are fit on all seasons, so this is one-step-ahead for the STATES only. For a
strict test, refit on seasons < s and evaluate season s.

Usage (in the session where the models are fit):
    from evaluate_team_models import evaluate
    summary, rows = evaluate(id_test, id_bt)
    summary, rows = evaluate(id_test, id_bt, extra={'division': id_div})
    # walk-forward: every model fit on seasons <= 2021, score 2022+
    summary, rows = evaluate(id_test_2021, id_bt_2021, extra={'division': id_div_2021}, first_season=2022)
or from saved files:
    python src/evaluate_team_models.py model-nc/team_ability.nc model-nc/team_ability_bt.nc
"""

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from scipy.stats import norm

DATA_PATH = 'processed-data/ff-processed.parquet'
LEVELS = (0.5, 0.8, 0.95)


# ---------------------------------------------------------------------------
# data: exactly the prep each model used
# ---------------------------------------------------------------------------
def load_games(path=DATA_PATH):
    dat = pl.read_parquet(path).unique(subset='game_id')
    games = (
        dat.select(
            'game_id', 'season', 'week', 'home_team', 'away_team', 'spread_line',
            pl.col('home_score', 'away_score').cast(pl.Float64),
            (pl.col('home_rest').cast(pl.Float64) - pl.col('away_rest').cast(pl.Float64)).alias('rest_days'),
        )
        .with_columns((pl.col('rest_days').clip(-7, 7) / 7).alias('rest_diff'))   # new model's scaling
        .sort(['season', 'week', 'game_id'])
    )
    global_week = (games.select('season', 'week').unique().sort(['season', 'week'])
                   .with_row_index('t').with_columns(pl.col('t').cast(pl.Int64)))
    teams = sorted(games['home_team'].unique().to_list())
    seasons = sorted(games['season'].unique().to_list())
    team_pos = {t: i for i, t in enumerate(teams)}
    season_pos = {s: i for i, s in enumerate(seasons)}
    games = games.join(global_week, on=['season', 'week']).with_columns(
        pl.col('home_team').replace_strict(team_pos).alias('h'),
        pl.col('away_team').replace_strict(team_pos).alias('a'),
        pl.col('season').replace_strict(season_pos).alias('s'),
    ).sort(['t', 'game_id'])
    season_of_t = global_week['season'].replace_strict(season_pos).to_numpy()
    starts = np.flatnonzero(np.r_[True, np.diff(season_of_t) != 0])
    return games, teams, seasons, global_week.height, set(starts.tolist())


def team_rows(games):
    """One row per team per game; `sign` = +1 home row, -1 away row."""
    base = games.with_row_index('g')
    home = base.select('g', 'game_id', 'season', 'week', 't', 'spread_line', 'rest_days',
                       pl.col('home_team').alias('team'), pl.col('h').alias('i'), pl.col('a').alias('j'),
                       (pl.col('home_score') - pl.col('away_score')).alias('margin'),
                       pl.lit(1.0).alias('is_home'), pl.lit(1.0).alias('sign'))
    away = base.select('g', 'game_id', 'season', 'week', 't', 'spread_line', 'rest_days',
                       pl.col('away_team').alias('team'), pl.col('a').alias('i'), pl.col('h').alias('j'),
                       (pl.col('away_score') - pl.col('home_score')).alias('margin'),
                       pl.lit(0.0).alias('is_home'), pl.lit(-1.0).alias('sign'))
    return (pl.concat([home, away])
            .with_columns((pl.col('rest_days') * pl.col('sign')).alias('rest_days'),
                          (pl.col('spread_line') * pl.col('sign')).alias('vegas'))
            .sort(['g', 'sign'], descending=[False, True]))


def _flat(post, name):
    v = np.asarray(post[name].values)
    return v.reshape(v.shape[0] * v.shape[1], -1).squeeze(-1) if v.ndim == 2 else v.reshape(v.shape[0] * v.shape[1], *v.shape[2:])


def _pick(n_total, n_draws, seed):
    return np.random.default_rng(seed).choice(n_total, size=min(n_draws, n_total), replace=False)


# ---------------------------------------------------------------------------
# old model: independent local level per team on OLS-adjusted margins
#   level_t = level_{t-1} + N(0, sigma_level^2);  adj_margin = level + N(0, sigma_meas^2)
#   x0 = initial_level_trend, P0 = 10 (as in the fit)
# ---------------------------------------------------------------------------
def old_predictive(idata, games, rows, n_team, n_week, n_draws=400, seed=1, p0=10.0, fit_through=None):
    # the first-stage OLS exactly as in the fit script (no intercept, rest in days),
    # on the seasons the model was fit on
    X = rows.select('is_home', 'rest_days').to_numpy()
    y = rows['margin'].to_numpy()
    in_fit = np.ones(len(y), bool) if fit_through is None else (rows['season'] <= fit_through).to_numpy()
    beta, *_ = np.linalg.lstsq(X[in_fit], y[in_fit], rcond=None)
    fitted = X @ beta
    adj = y - fitted

    post = idata.posterior
    x0 = _flat(post, 'initial_level_trend').reshape(-1, n_team)
    s_lvl = _flat(post, 'sigma_level_trend').reshape(-1, n_team)
    s_obs = _flat(post, 'sigma_MeasurementError').reshape(-1, n_team)
    draws = _pick(x0.shape[0], n_draws, seed)

    wide = np.full((n_week, n_team), np.nan)
    t_r, i_r = rows['t'].to_numpy(), rows['i'].to_numpy()
    wide[t_r, i_r] = adj

    D = len(draws)
    lvl_m = np.zeros((D, n_week, n_team)); lvl_v = np.zeros((D, n_week, n_team))
    for k, d in enumerate(draws):
        m, P = x0[d].copy(), np.full(n_team, p0)
        q, r = s_lvl[d] ** 2, s_obs[d] ** 2
        for t in range(n_week):
            if t > 0:
                P = P + q                          # predict (random walk)
            lvl_m[k, t], lvl_v[k, t] = m, P        # pre-game state for week t
            obs = ~np.isnan(wide[t])
            K = P[obs] / (P[obs] + r[obs])
            m[obs] = m[obs] + K * (wide[t, obs] - m[obs])
            P[obs] = (1 - K) * P[obs]

    M = fitted[None, :] + lvl_m[:, t_r, i_r]
    V = lvl_v[:, t_r, i_r] + (s_obs[draws] ** 2)[:, i_r]
    return M, V, dict(level_mean=lvl_m, level_var=lvl_v, beta=beta)


# ---------------------------------------------------------------------------
# new model: dynamic off/def, one game = two correlated scores
# ---------------------------------------------------------------------------
NEW_PARAMS = ['mu', 'hfa', 'beta_rest', 'init_sd', 'sigma_week', 'sigma_season', 'rho', 'sigma_obs', 'rho_obs']


def new_predictive(idata, games, rows, n_team, n_week, season_starts, n_draws=400, seed=1):
    post = idata.posterior
    flat = {n: _flat(post, n) for n in NEW_PARAMS}
    n_total = flat['mu'].shape[0]
    draws = _pick(n_total, n_draws, seed)
    N, Dm = n_team, 2 * n_team

    # long-run means (division model); zeros = revert to league average
    if 'team_mean' in post:
        flat['team_mean'] = _flat(post, 'team_mean').reshape(n_total, 2 * N)   # [off..., def...]
    else:
        flat['team_mean'] = np.zeros((n_total, 2 * N))

    # walk-forward: seasons after the fit window get a fresh draw of their
    # scoring level / HFA from the fitted hierarchy (no information from them)
    S_all = int(games['s'].max()) + 1
    n_fit = flat['mu'].shape[1]
    if n_fit < S_all:
        rng = np.random.default_rng(seed + 1)
        for name in ('mu', 'hfa'):
            bar, sd = _flat(post, f'{name}_bar'), _flat(post, f'{name}_sd')
            extra = bar[:, None] + sd[:, None] * rng.standard_normal((n_total, S_all - n_fit))
            flat[name] = np.concatenate([flat[name], extra], axis=1)
    comp = np.repeat([0, 1], N)
    idx = np.arange(N)
    G = games.height

    by_t = {}
    for (t,), g in games.with_row_index('g').group_by(['t'], maintain_order=True):
        by_t[t] = dict(g=g['g'].to_numpy(), h=g['h'].to_numpy(), a=g['a'].to_numpy(), s=g['s'].to_numpy(),
                       rest=g['rest_diff'].to_numpy(),
                       y=g.select('home_score', 'away_score').to_numpy())

    D = len(draws)
    pm_ = np.zeros((D, G, 2)); pc_ = np.zeros((D, G, 2, 2))
    net_m = np.zeros((D, n_week, N)); net_v = np.zeros((D, n_week, N))
    off_m = np.zeros((D, n_week, N)); def_m = np.zeros((D, n_week, N))
    obs_var = np.zeros((D, 2, 2))
    for k, d in enumerate(draws):
        p = {n: flat[n][d] for n in NEW_PARAMS + ['team_mean']}
        tm = p['team_mean']
        R = p['sigma_obs'] ** 2 * np.array([[1.0, p['rho_obs']], [p['rho_obs'], 1.0]])
        obs_var[k] = R
        m = tm.copy(); P = np.diag(p['init_sd'][comp] ** 2)
        for t in range(n_week):
            if t > 0:
                if t in season_starts:
                    r = p['rho'][comp]
                    m = tm + r * (m - tm)
                    P = r[:, None] * P * r[None, :] + np.diag(p['sigma_season'][comp] ** 2)
                else:
                    P = P + np.diag(p['sigma_week'][comp] ** 2)
            nm = m[:N] + m[N:]
            net_m[k, t] = nm - nm.mean()
            off_m[k, t] = m[:N] - m[:N].mean()          # pre-game components, centered
            def_m[k, t] = m[N:] - m[N:].mean()
            net_v[k, t] = P[idx, idx] + P[N + idx, N + idx] + 2 * P[idx, N + idx]
            g = by_t.get(t)
            if g is None:
                continue
            ng = len(g['h']); rr = np.arange(ng)
            Z = np.zeros((2 * ng, Dm))
            Z[2 * rr, g['h']] = 1; Z[2 * rr, N + g['a']] = -1
            Z[2 * rr + 1, g['a']] = 1; Z[2 * rr + 1, N + g['h']] = -1
            shift = 0.5 * (p['hfa'][g['s']] + p['beta_rest'] * g['rest'])
            base = np.column_stack([p['mu'][g['s']] + shift, p['mu'][g['s']] - shift]).ravel()
            mean = base + Z @ m
            PZt = P @ Z.T
            F = Z @ PZt + np.kron(np.eye(ng), R)
            # pre-game predictive for each game (2x2 blocks of F)
            pm_[k, g['g']] = mean.reshape(ng, 2)
            pc_[k, g['g']] = F.reshape(ng, 2, ng, 2)[rr, :, rr, :]
            # update with this week's scores (games not played yet -- NaN
            # scores, e.g. an upcoming week -- get a prediction but no update)
            yv = g['y'].ravel()
            ok = ~np.isnan(yv)
            if ok.any():
                Fo, PZo = F[np.ix_(ok, ok)], PZt[:, ok]
                K = np.linalg.solve(Fo, PZo.T).T
                m = m + K @ (yv[ok] - mean[ok])
                P = P - K @ PZo.T
                P = 0.5 * (P + P.T)

    # team-row margin = sign * (home - away)
    gi, sg = rows['g'].to_numpy(), rows['sign'].to_numpy()
    M = sg[None, :] * (pm_[:, gi, 0] - pm_[:, gi, 1])
    V = pc_[:, gi, 0, 0] + pc_[:, gi, 1, 1] - 2 * pc_[:, gi, 0, 1]
    return M, V, dict(score_mean=pm_, score_cov=pc_, net_mean=net_m, net_var=net_v,
                      off_mean=off_m, def_mean=def_m, obs_cov=obs_var)


# ---------------------------------------------------------------------------
# scoring a Normal mixture predictive: M, V are (draws, rows)
# ---------------------------------------------------------------------------
def score_mixture(y, M, V, n_rep=4, seed=2):
    D = M.shape[0]
    sd = np.sqrt(V)
    lpd = logsumexp(norm.logpdf(y[None], M, sd), axis=0) - np.log(D)
    pit = norm.cdf((y[None] - M) / sd).mean(0)
    rng = np.random.default_rng(seed)
    x = (M[None] + sd[None] * rng.standard_normal((n_rep, *M.shape))).reshape(n_rep * D, -1)
    x.sort(axis=0)
    n = x.shape[0]
    w = (2 * np.arange(1, n + 1) - n - 1)[:, None]
    crps = np.abs(x - y[None]).mean(0) - (w * x).sum(0) / n ** 2     # E|X-y| - 0.5 E|X-X'|
    out = dict(mean=M.mean(0), lpd=lpd, crps=crps, pit=pit)
    for lv in LEVELS:
        lo, hi = np.quantile(x, [(1 - lv) / 2, (1 + lv) / 2], axis=0)
        out[f'lo{int(lv*100)}'], out[f'hi{int(lv*100)}'] = lo, hi
    return out


def summarise(df, by=None):
    aggs = [
        pl.len().alias('n'),
        pl.col('lpd').mean().alias('elpd'),
        pl.col('crps').mean().alias('crps'),
        ((pl.col('margin') - pl.col('mean')) ** 2).mean().sqrt().alias('rmse'),
        (pl.col('margin') - pl.col('mean')).abs().mean().alias('mae')
    ]
    keys = ['model'] + (by or [])
    return df.group_by(keys).agg(aggs).sort(keys)


def to_long(rows, sc, model):
    cols = {'model': model, 'mean': sc['mean'], 'lpd': sc['lpd'], 'crps': sc['crps'], 'pit': sc['pit']}
    y = rows['margin'].to_numpy()
    for lv in LEVELS:
        k = int(lv * 100)
        cols[f'in{k}'] = ((y >= sc[f'lo{k}']) & (y <= sc[f'hi{k}'])).astype(float)
        cols[f'w{k}'] = sc[f'hi{k}'] - sc[f'lo{k}']
        cols[f'lo{k}'], cols[f'hi{k}'] = sc[f'lo{k}'], sc[f'hi{k}']
    return rows.select('game_id', 'season', 'week', 'team', 'margin', 'vegas', 'sign').with_columns(
        **{k: (pl.lit(v) if isinstance(v, str) else pl.Series(v)) for k, v in cols.items()})


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------
def evaluate(id_old, id_new, extra=None, n_draws=400, burn_in_seasons=1, first_season=None,
             fit_through=None, plot=True, team='SF'):
    """id_old: LevelTrend fit. id_new: off/def fit. extra: {name: idata} for more
    off/def-style fits (e.g. {'division': id_div}); any with a `team_mean`
    variable revert toward it.

    fit_through: last season every model was fit on (walk-forward). Scoring
    then starts the season after, and the old model's home/rest regression
    uses only the fit seasons.
    first_season: score only seasons >= this. Default: fit_through + 1 if
    given, else everything after `burn_in_seasons`."""
    games, teams, seasons, n_week, season_starts = load_games()
    rows = team_rows(games)
    y = rows['margin'].to_numpy()
    if first_season is None:
        first_season = fit_through + 1 if fit_through is not None else seasons[burn_in_seasons]

    M, V, ex = {}, {}, {}
    M['old_leveltrend'], V['old_leveltrend'], ex['old_leveltrend'] = old_predictive(
        id_old, games, rows, len(teams), n_week, n_draws, fit_through=fit_through)
    new_style = {'new_offdef': id_new, **(extra or {})}
    for name, idata in new_style.items():
        M[name], V[name], ex[name] = new_predictive(idata, games, rows, len(teams), n_week, season_starts, n_draws)

    long = pl.concat([to_long(rows, score_mixture(y, M[k], V[k]), k) for k in M])
    long = long.filter(pl.col('season') >= first_season).with_columns(
        pl.when(pl.col('week') <= 4).then(pl.lit('wk 01-04'))
          .when(pl.col('week') <= 9).then(pl.lit('wk 05-09'))
          .otherwise(pl.lit('wk 10+')).alias('phase'))

    vegas = (long.filter(pl.col('model') == 'new_offdef')
             .select(pl.lit('vegas_spread (point only)').alias('model'),
                     ((pl.col('margin') - pl.col('vegas')) ** 2).mean().sqrt().alias('rmse'),
                     (pl.col('margin') - pl.col('vegas')).abs().mean().alias('mae')))

    # every model vs the old one, and every extra model vs new_offdef
    pairs = [(k, 'old_leveltrend') for k in new_style] + [(k, 'new_offdef') for k in (extra or {})]
    summary = {
        'overall': summarise(long),
        'by_phase': summarise(long, ['phase']),
        'by_season': summarise(long, ['season']),
        'vegas': vegas,
        'paired_diff': pl.concat([paired_diff(long, a, b) for a, b in pairs]),
    }
    with pl.Config(tbl_cols=-1, tbl_width_chars=200, float_precision=3, tbl_rows=60):
        for k, v in summary.items():
            print(f'\n== {k} ==\n{v}')

    if plot:
        plot_calibration(long, list(M))
        bands = {'old_leveltrend': (ex['old_leveltrend']['level_mean'], ex['old_leveltrend']['level_var'])}
        bands.update({k: (ex[k]['net_mean'], ex[k]['net_var']) for k in new_style})
        plot_team(team, teams, games, rows, bands, n_week)
        for k in new_style:
            plot_scores(games, ex[k], first_season, label=k)
    return summary, long


def paired_diff(long, a='new_offdef', b='old_leveltrend'):
    """a - b per row, with a game-clustered SE (rows of the same game are
    not independent). Negative d_crps / positive d_elpd favours a."""
    w = (long.filter(pl.col('model').is_in([a, b]))
         .select('game_id', 'team', 'model', 'lpd', 'crps')
         .pivot(on='model', index=['game_id', 'team'], values=['lpd', 'crps']))
    w = w.with_columns((pl.col(f'lpd_{a}') - pl.col(f'lpd_{b}')).alias('d_elpd'),
                       (pl.col(f'crps_{a}') - pl.col(f'crps_{b}')).alias('d_crps'))
    g = w.group_by('game_id').agg(pl.col('d_elpd', 'd_crps').mean())
    n = g.height
    return g.select(
        pl.lit(f'{a} - {b}').alias('comparison'),
        pl.col('d_elpd').mean().alias('d_elpd'), (pl.col('d_elpd').std() / np.sqrt(n)).alias('se_elpd'),
        pl.col('d_crps').mean().alias('d_crps'), (pl.col('d_crps').std() / np.sqrt(n)).alias('se_crps'),
        pl.lit(n).alias('n_games'))


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------
def plot_calibration(long, models):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    nominal = np.linspace(0.05, 0.95, 19)
    for model, c in zip(models, [f'C{i}' for i in range(len(models))]):
        pit = long.filter(pl.col('model') == model)['pit'].to_numpy()
        axes[0].hist(pit, bins=20, range=(0, 1), histtype='step', density=True, color=c, lw=2, label=model)
        cov = [np.mean(np.abs(pit - 0.5) <= lv / 2) for lv in nominal]
        axes[1].plot(nominal, cov, 'o-', ms=3, color=c, label=model)
        by = (long.filter(pl.col('model') == model).group_by('phase')
              .agg(pl.col('crps').mean()).sort('phase'))
        axes[2].plot(by['phase'], by['crps'], 'o-', color=c, label=model)
    axes[0].axhline(1, color='grey', lw=0.8); axes[0].set_title('PIT (flat = calibrated)')
    axes[1].plot([0, 1], [0, 1], color='grey', lw=0.8)
    axes[1].set_xlabel('nominal central interval'); axes[1].set_ylabel('empirical coverage')
    axes[1].set_title('interval calibration')
    axes[2].set_title('CRPS by point in season (lower = better)')
    for ax in axes:
        ax.legend(fontsize=8)
    fig.tight_layout()
    plt.show()


def plot_team(team, teams, games, rows, bands, n_week, q=(0.05, 0.95)):
    """Pre-game strength with 90% bands under each model, plus the team's
    actual margins. bands: {name: (mean, var)}, each (draws, week, team)."""
    i = teams.index(team)
    t = np.arange(n_week)
    fig, ax = plt.subplots(figsize=(13, 4))
    for k, (name, (m, v)) in enumerate(bands.items()):
        m, v = m[:, :, i], v[:, :, i]
        s = m + np.sqrt(v) * np.random.default_rng(0).standard_normal(m.shape)
        ax.fill_between(t, *np.quantile(s, q, axis=0), color=f'C{k}', alpha=0.15)
        ax.plot(t, m.mean(0), color=f'C{k}', label=f'{name} (pre-game mean, 90% band)')
    r = rows.filter(pl.col('team') == team)
    ax.scatter(r['t'], r['margin'], s=8, color='k', alpha=0.4, label='actual margin')
    for s0 in games.group_by('season').agg(pl.col('t').min())['t']:
        ax.axvline(s0, color='grey', lw=0.4)
    ax.set_ylim(-30, 30); ax.set_xlabel('global week'); ax.set_title(f'{team}: pre-game team strength')
    ax.legend(fontsize=8, loc='upper left')
    fig.tight_layout()
    plt.show()


def plot_scores(games, ex_new, first_season, label="new_offdef"):
    """New model only: pre-game coverage of home pts, away pts, total, margin."""
    Mu, C = ex_new['score_mean'], ex_new['score_cov']
    keep = (games['season'] >= first_season).to_numpy()
    y = games.select('home_score', 'away_score').to_numpy()[keep]
    Mu, C = Mu[:, keep], C[:, keep]
    targets = {
        'home pts': (Mu[..., 0], C[..., 0, 0], y[:, 0]),
        'away pts': (Mu[..., 1], C[..., 1, 1], y[:, 1]),
        'total': (Mu.sum(-1), C[..., 0, 0] + C[..., 1, 1] + 2 * C[..., 0, 1], y.sum(1)),
        'home margin': (Mu[..., 0] - Mu[..., 1], C[..., 0, 0] + C[..., 1, 1] - 2 * C[..., 0, 1], y[:, 0] - y[:, 1]),
    }
    nominal = np.linspace(0.05, 0.95, 19)
    fig, ax = plt.subplots(figsize=(5, 4))
    print(f'\n== {label}: score-level targets ==')
    for name, (m, v, yy) in targets.items():
        pit = norm.cdf((yy[None] - m) / np.sqrt(v)).mean(0)
        ax.plot(nominal, [np.mean(np.abs(pit - 0.5) <= lv / 2) for lv in nominal], 'o-', ms=3, label=name)
        print(f'{name:12s} rmse {np.sqrt(np.mean((yy - m.mean(0))**2)):5.2f}  '
              f'cov80 {np.mean(np.abs(pit - 0.5) <= 0.4):.3f}  cov95 {np.mean(np.abs(pit - 0.5) <= 0.475):.3f}')
    ax.plot([0, 1], [0, 1], color='grey', lw=0.8)
    ax.set_xlabel('nominal'); ax.set_ylabel('empirical'); ax.set_title(f'{label}: score calibration')
    ax.legend(fontsize=8)
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    import sys
    import arviz as az
    old_path = sys.argv[1] if len(sys.argv) > 1 else 'model-nc/team_ability.nc'
    new_path = sys.argv[2] if len(sys.argv) > 2 else 'model-nc/team_ability_bt.nc'
    evaluate(az.from_netcdf(old_path), az.from_netcdf(new_path))
