"""Do the risk-aware rules pick different lineups from maximize_expected, and are
the lineup forecasts calibrated?

Each run is a directory of saved draws, in either format:
  league forecasts   <dir>/forecasts/<season>_wk<NN>/{draws.npy, frame.parquet}
                     (written by `ff.py league --out-dir <dir>`)
  backtest draws     <dir>/S_<season>_<NN>.npy + te_<season>_<NN>.parquet
Weeks present in every run are compared; the rosters are the synthetic 10-team
draft the league command uses (league.load_rosters).

Every week, for both teams of each scheduled matchup (same round robin as the league):
  own-target rules   cvar_averse (alpha 0.10, lam 1 / 3) with target = your mean
                     lineup's expected total; chance_constrained (beta 0.90) with
                     floor = its p10 (the old defaults)
  opponent rules     cvar_averse with the opponent's total in each draw (lam 1 / 3);
                     chance_constrained with floor = his expected total (beta 0.5 / 0.9);
                     head_to_head
plus lineup calibration of the mean lineup and P(win) calibration.

Run from the project root:
  uv run python src/check_decisions.py                                  # the backtest draws vs old league forecasts
  uv run python src/check_decisions.py --runs qb_linked=results/league/season_qb_linked \\
                                              independent=results/league/season_independent
  uv run python src/check_decisions.py --summary                        # re-print the last results
Writes results/decision_check.parquet.
"""
import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import polars as pl

from ffpred.forecast import WeekForecast
from ffpred import data as D, decision_engine as de, league
from ffpred.config import PATHS, TARGET_COL

RES = PATHS.results_dir
OUT = RES / 'decision_check.parquet'
DEFAULT_RUNS = {'new': RES / 'scratch_joint_qb_resid_cut', 'old': RES / 'league' / 'synthetic'}
OWN_RULES = {'cvar_l1': dict(rule='cvar_averse', alpha=0.10, lam=1.0),
             'cvar_l3': dict(rule='cvar_averse', alpha=0.10, lam=3.0),
             'chance_b90': dict(rule='chance_constrained', beta=0.90)}
OPP_RULES = {'cvar_opp_l1': dict(rule='cvar_averse', alpha=0.10, lam=1.0),
             'cvar_opp_l3': dict(rule='cvar_averse', alpha=0.10, lam=3.0),
             'chance_opp_b50': dict(rule='chance_constrained', beta=0.50),
             'chance_opp_b90': dict(rule='chance_constrained', beta=0.90),
             'h2h': dict(rule='head_to_head')}
RULES = {**OWN_RULES, **OPP_RULES}


def weeks_in(run_dir: Path) -> dict:
    """{(season, week): loader} for a run directory of either format."""
    out = {}
    for d in (run_dir / 'forecasts').glob('*_wk*'):
        m = re.fullmatch(r'(\d{4})_wk(\d+)', d.name)
        if m and (d / 'meta.json').exists():           # forecast.nc, or the older draws.npy + frame.parquet
            out[(int(m[1]), int(m[2]))] = lambda d=d: (lambda fc: (fc.draws, fc.frame))(WeekForecast.load(d))
    for f in run_dir.glob('S_*.npy'):
        m = re.fullmatch(r'S_(\d{4})_(\d+)\.npy', f.name)
        te = run_dir / f'te_{m[1]}_{m[2]}.parquet' if m else None
        if m and te.exists():
            out[(int(m[1]), int(m[2]))] = lambda f=f, te=te: (np.load(f).astype('float64'), pl.read_parquet(te))
    return out


def structure(S, frame, pool):
    """Across all legal lineups: corr(E, E - CVaR_10%(shortfall below the best E)) and the
    sd spread of the top 5 lineups by mean."""
    cands = list(de.all_candidate_lineups(frame, pool=pool))
    T = np.stack([de.candidate_totals(S, c) for c in cands])
    E, SD = T.mean(1), T.std(1)
    k = max(1, int(0.10 * T.shape[1]))
    cv = np.sort(np.maximum(E.max() - T, 0), axis=1)[:, ::-1][:, :k].mean(1)
    top5 = np.argsort(-E)[:5]
    corr = float(np.corrcoef(E, E - cv)[0, 1]) if len(cands) > 2 else float('nan')
    return dict(n_cands=len(cands), corr_E_obj=corr, sd_best=float(SD[top5[0]]),
                sd_spread_top5=float(SD[top5].max() - SD[top5].min()))


def run(runs: dict):
    hist = D.load_processed()
    found = {name: weeks_in(Path(d)) for name, d in runs.items()}
    for name, wk in found.items():
        print(f'{name}: {len(wk)} weeks in {runs[name]}')
    common = sorted(set.intersection(*(set(w) for w in found.values())))
    if not common:
        sys.exit('no week is present in every run')
    print(f'comparing {len(common)} common weeks: {common[0]} .. {common[-1]}')
    rows, rosters_by_season, t0 = [], {}, time.time()
    for s, w in common:
        rosters = rosters_by_season.setdefault(s, league.load_rosters(s, hist))
        sched = league.round_robin_schedule(list(rosters))
        matchups = sched[int(w - 1) % len(sched)]
        for name in runs:
            S, frame = found[name][(s, w)]()
            y = frame[TARGET_COL].fill_null(0).to_numpy()
            col = {p: i for i, p in enumerate(frame['player_id'].to_list())}
            ids = lambda c: c.lineup['player_id'].to_list()               # noqa: E731
            real = lambda c: float(y[[col[p] for p in ids(c)]].sum())     # noqa: E731
            for a, b in matchups:
                for me, opp in ((a, b), (b, a)):
                    pool = [p for p in rosters[me] if p in col]
                    opool = [p for p in rosters[opp] if p in col]
                    try:
                        base = de.maximize_expected(S, frame, pool=pool)
                        ob = de.maximize_expected(S, frame, pool=opool)
                    except ValueError:           # can't fill a legal lineup (bye / not scoreable)
                        continue
                    rb, ro = real(base), real(ob)
                    r = dict(run=name, season=s, week=w, team=me, **structure(S, frame, pool),
                             base_expected=float(base.totals.mean()), base_realized=rb,
                             base_pit=float((base.totals < rb).mean()),
                             p_win_base=de.p_beats(base.totals, ob.totals), won_base=int(rb > ro))
                    for k, kw in RULES.items():
                        kw = dict(kw)
                        rule = kw.pop('rule')
                        opp_kw = {'opp_totals': ob.totals} if k in OPP_RULES else {}
                        c = de.decide(rule, S, frame, pool, **opp_kw, **kw)
                        rc = real(c)
                        r[f'{k}_differs'] = int(set(ids(c)) != set(ids(base)))
                        r[f'{k}_gain'] = rc - rb
                        r[f'{k}_won'] = int(rc > ro)
                        r[f'{k}_exp_cost'] = float(base.totals.mean() - c.totals.mean())
                        r[f'{k}_dp_win'] = de.p_beats(c.totals, ob.totals) - r['p_win_base']
                        if k == 'chance_b90':
                            r['chance_cleared'] = int(rc >= c.stats['floor'])
                            r['chance_p_clear'] = float(c.stats['p_clear'])
                    rows.append(r)
        print(s, w, f'{time.time() - t0:.0f}s', flush=True)
    pl.DataFrame(rows).write_parquet(OUT)
    print(f'{len(rows)} decisions -> {OUT}')


def _se(x):
    x = np.asarray(x, float)
    return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float('nan')


def summary():
    d = pl.read_parquet(OUT)
    with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=3, tbl_width_chars=250):
        print('=== lineup calibration (mean lineup) and structure')
        cal = []
        for (r,), g in d.group_by('run', maintain_order=True):
            pit = g['base_pit'].to_numpy()
            miss = (g['base_realized'] - g['base_expected']).to_numpy()
            cal.append({'run': r, 'decisions': g.height, 'weeks': g.select('season', 'week').n_unique(),
                        'pred_sd': g['sd_best'].mean(), 'actual_miss_sd': float(miss.std()),
                        'cov80': float(((pit > .1) & (pit < .9)).mean()),
                        'below_p10': float((pit < .1).mean()), 'above_p90': float((pit > .9).mean()),
                        'sd_spread_top5': g['sd_spread_top5'].mean(),
                        'corr_E_cvar': float(np.nanmean(g['corr_E_obj'].to_numpy())),
                        'chance_b90_promised': g['chance_p_clear'].mean(),
                        'chance_b90_realized': g['chance_cleared'].mean()})
        print(pl.DataFrame(cal))

        print('=== how often each rule differs from maximize_expected, and what it buys when it does')
        out = []
        for (r,), g in d.group_by('run', maintain_order=True):
            for k in RULES:
                h = g.filter(pl.col(f'{k}_differs') == 1)
                out.append({'run': r, 'rule': k, 'differs': g[f'{k}_differs'].mean(), 'n_differs': h.height,
                            'exp_cost': h[f'{k}_exp_cost'].mean() if h.height else None,
                            'pred_dp_win': h[f'{k}_dp_win'].mean() if h.height else None,
                            'realized_gain': h[f'{k}_gain'].mean() if h.height else None,
                            'gain_se': _se(h[f'{k}_gain']) if h.height > 1 else None,
                            'won_base_when_differs': h['won_base'].mean() if h.height else None,
                            'won_rule_when_differs': h[f'{k}_won'].mean() if h.height else None})
        print(pl.DataFrame(out))

        print('=== P(win) calibration of the mean lineup (predicted vs realized, by predicted bin)')
        for (r,), g in d.group_by('run', maintain_order=True):
            b = (g.with_columns((pl.col('p_win_base') * 5).floor().clip(0, 4).alias('bin'))
                 .group_by('bin').agg(pl.len().alias('n'), pl.col('p_win_base').mean().alias('pred'),
                                      pl.col('won_base').mean().alias('won')).sort('bin'))
            print(r, [(int(x['n']), round(x['pred'], 2), round(x['won'], 2)) for x in b.iter_rows(named=True)])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='*', help='name=dir pairs (default: the backtest draws vs the old league forecasts)')
    ap.add_argument('--summary', action='store_true', help='only re-print the summary of the last run')
    a = ap.parse_args()
    if not a.summary:
        runs = dict(x.split('=', 1) for x in a.runs) if a.runs else DEFAULT_RUNS
        run({k: Path(v) for k, v in runs.items()})
    summary()
