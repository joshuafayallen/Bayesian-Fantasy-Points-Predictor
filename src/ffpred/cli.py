"""Command line entry point: ``python src-dev/ff.py <command> ...`` from the
project root (or ``python -m ffpred`` with src-dev on PYTHONPATH).

Weekly production flow:
    build-data        refresh processed-data/ff-processed.parquet from nflverse
    build-team-form   refit processed-data/team_form.parquet
    forecast          fit both models, forecast a week -> forecasts/<season>_wk<NN>/
    decide            pick a lineup for your roster from a saved forecast
                      (flags near-ties: start/sit calls within Monte Carlo noise)

Validation:
    check-forecast    tail calibration of a replayed (played) week by projection tier
    backtest          walk-forward folds (resumable) -> results/backtest/<run>/
    summarize-backtest
    league            league simulation of the decision rules -> results/league/<run>/
    summarize-league
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import polars as pl

from .config import MODELS, PATHS, SamplerSettings

log = logging.getLogger('ffpred')


def _int_list(spec: str) -> list[int]:
    """'2019 2023-2025' or '6,9,12' -> ints."""
    out = []
    for tok in spec.replace(',', ' ').split():
        if '-' in tok:
            a, b = tok.split('-')
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(tok))
    return out


def _settings(a) -> SamplerSettings:
    return SamplerSettings(method=a.method, nuts_sampler=a.sampler, draws=a.draws, tune=a.tune,
                           chains=a.chains, advi_n=a.advi_n, advi_draws=a.advi_draws, seed=a.seed)


def _add_fit_args(p):
    g = p.add_argument_group('fitting')
    g.add_argument('--method', choices=['nuts', 'advi'], default='nuts')
    g.add_argument('--sampler', choices=['nutpie', 'pymc'], default='nutpie')
    g.add_argument('--draws', type=int, default=1000)
    g.add_argument('--tune', type=int, default=1000)
    g.add_argument('--chains', type=int, default=4)
    g.add_argument('--advi-n', type=int, default=20_000)
    g.add_argument('--advi-draws', type=int, default=500)
    g.add_argument('--seed', type=int, default=0)


def _add_data_args(p):
    p.add_argument('--data', type=Path, default=PATHS.processed, help='processed player-week parquet')
    p.add_argument('--team-form', type=Path, default=PATHS.team_form)


def _load(a):
    from .data import load_processed, load_team_form
    return load_processed(a.data), load_team_form(a.team_form)


def _read_ids(spec: str | None) -> list[str] | None:
    """A roster: comma-separated ids/names, or a CSV/TXT file (player_id or player column, or one per line)."""
    if spec is None:
        return None
    p = Path(spec)
    if p.exists():
        if p.suffix == '.csv':
            df = pl.read_csv(p)
            col = 'player_id' if 'player_id' in df.columns else 'player'
            return df[col].drop_nulls().to_list()
        return [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
    return [s.strip() for s in spec.split(',') if s.strip()]


# ------------------------------------------------------------------ commands

def cmd_build_data(a):
    from .data_build import build_processed, current_season, write_processed
    seasons = _int_list(a.seasons) if a.seasons else list(range(2006, current_season() + 1))
    df = build_processed(seasons, legacy=a.legacy)
    print(f'wrote {write_processed(df, a.out)} ({df.height} rows, seasons {df["season"].min()}-{df["season"].max()})')


def cmd_build_team_form(a):
    from .data import load_processed
    from .team_form import fit_team_form
    tf = fit_team_form(load_processed(a.data), through_season=a.through_season, draws=a.draws,
                       tune=a.tune, chains=a.chains, seed=a.seed, nuts_sampler=a.sampler)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tf.write_parquet(a.out)
    print(f'wrote {a.out} ({tf.height} team-weeks)')


def cmd_forecast(a):
    from .forecast import forecast_week
    from .upcoming import build_upcoming_frame, fetch_rosters, schedule_from_processed
    from .data_build import fetch_schedule
    hist, tf = _load(a)
    if a.source == 'played':
        frame = None
    else:
        sched = (schedule_from_processed(hist, a.season, float(a.week)) if a.schedule == 'processed'
                 else fetch_schedule(a.season, a.week))
        if sched.height == 0:
            sys.exit(f'no games found for {a.season} week {a.week}')
        rosters = fetch_rosters(a.season, a.week) if a.rosters == 'nflverse' else None
        frame = build_upcoming_frame(hist, sched, a.season, a.week, rosters=rosters, active_only=not a.include_inactive)
    fc = forecast_week(hist, tf, a.season, a.week, frame=frame, settings=_settings(a))
    out = fc.save(a.out or PATHS.forecasts_dir / f'{a.season}_wk{a.week:02d}')
    with pl.Config(tbl_rows=25, tbl_cols=12, tbl_width_chars=160, float_precision=1, fmt_str_lengths=24):
        print(fc.summary().drop('player_id').head(25))
    print(f'{fc.frame.height} players forecast, {fc.skipped.height} skipped -> {out}')
    print(json.dumps(fc.meta['fits'], indent=1, default=str))


def cmd_decide(a):
    from . import decision_engine as de
    from .forecast import WeekForecast
    fc = WeekForecast.load(a.forecast)
    pool = _read_ids(a.roster)
    have = set(fc.frame['player_id'].to_list()) | set(fc.frame['player'].to_list())
    missing = [p for p in pool if p not in have]
    if missing:
        print(f'not in this forecast (bye, inactive, rookie, or unknown id): {missing}')
    pool = [p for p in pool if p in have]
    opp = _read_ids(a.opponent)
    if opp is not None:
        opp = [p for p in opp if p in have]
    choice = de.decide(a.rule, fc.draws, fc.frame, pool, floor=a.floor, beta=a.beta, target=a.target,
                       alpha=a.alpha, lam=a.lam, opp_pool=opp)
    summ = fc.summary().select('player_id', 'expected', 'floor_p10', 'ceiling_p90')
    with pl.Config(tbl_rows=20, tbl_cols=10, float_precision=1):
        print(choice.lineup.join(summ, on='player_id', how='left', maintain_order='left'))
    print(json.dumps({k: v for k, v in choice.stats.items() if k not in ('runner_up', 'near_ties')},
                     indent=1, default=float))
    st = choice.stats
    if st.get('runner_up') is None:
        print('only one legal lineup from this roster')
    elif st['near_tie']:
        print(f"NEAR TIE: these start/sit calls are within Monte Carlo noise: {', '.join(st['contested_players'])}")
        for sw in st['near_ties']:
            print(f'  equally good: {sw}')
    else:
        gap = st.get('objective_gap')
        print(f"clear choice: next best is {st['runner_up_swap']}"
              + (f" ({gap:.3f} worse, se {st['objective_gap_se']:.3f})" if gap is not None else ''))
    if a.out:
        Path(a.out).write_text(json.dumps({'lineup': choice.lineup.to_dicts(), 'stats': choice.stats}, indent=1, default=float))


def cmd_check_forecast(a):
    from .forecast import WeekForecast
    from .scoring import calibration_table, score
    fc = WeekForecast.load(a.forecast)
    if not fc.meta.get('has_outcomes'):
        sys.exit('this forecast has no outcomes (an upcoming week) -- nothing to check yet')
    y = fc.frame['total_fantasy_points'].to_numpy().astype(float)
    m = score(y, fc.draws)
    print(f"n={m['n']}  crps={m['crps']:.3f}  rmse={m['rmse']:.3f}  cov80={m['cov80']:.3f}  "
          f"below_p10={m['below_p10']:.3f}  above_p90={m['above_p90']:.3f}")
    print(f"PIT deciles: {m['pit_hist']}  (calibrated: ~{m['n'] / 10:.0f} each)")
    with pl.Config(tbl_rows=40, tbl_cols=12, tbl_width_chars=160, float_precision=3):
        print(calibration_table(fc.frame, fc.draws))
        print(calibration_table(fc.frame, fc.draws, by_position=True))


def cmd_backtest(a):
    from .backtest import run_fold
    hist, tf = _load(a)
    run_dir = a.run_dir or PATHS.results_dir / 'backtest' / 'default'
    models = a.models.split(',') if a.models else list(MODELS)
    for s in _int_list(a.seasons):
        for w in _int_list(a.weeks):
            for m in models:
                try:
                    run_fold(m, s, w, hist, tf, run_dir, _settings(a), eligibility=a.eligibility,
                             overwrite=a.overwrite)
                except Exception:
                    log.exception('fold %s %d wk%d failed', m, s, w)
                    if a.fail_fast:
                        raise
    from .backtest import summarize
    summarize(run_dir)


def cmd_summarize_backtest(a):
    from .backtest import summarize
    summarize(a.run_dir)


def cmd_league(a):
    from .league import load_rosters, run_league_week, summarize_league, RISK_RULES
    hist, tf = _load(a)
    out_dir = a.out_dir or PATHS.results_dir / 'league' / 'default'
    rule_kwargs = {'chance_constrained': dict(beta=a.beta), 'cvar_averse': dict(alpha=a.alpha, lam=a.lam)}
    for s in _int_list(a.seasons):
        rosters = load_rosters(s, hist, mock_draft_csv=a.mock_draft, n_teams=a.n_teams)
        for w in _int_list(a.weeks):
            try:
                run_league_week(s, w, hist, tf, rosters, out_dir, _settings(a), rules=RISK_RULES,
                                rule_kwargs=rule_kwargs, overwrite=a.overwrite)
            except Exception:
                log.exception('league %d wk%d failed', s, w)
                if a.fail_fast:
                    raise
    summarize_league(out_dir)


def cmd_summarize_league(a):
    from .league import summarize_league
    summarize_league(a.out_dir)


# ---------------------------------------------------------------------- main

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog='ff', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('-v', '--verbose', action='store_true')
    sub = p.add_subparsers(dest='cmd', required=True)

    s = sub.add_parser('build-data', help='rebuild the processed parquet from nflverse')
    s.add_argument('--seasons', help="e.g. '2006-2026' (default: 2006 through the current season)")
    s.add_argument('--legacy', action='store_true', help='reproduce src/data_cleaning.py exactly (no fixes)')
    s.add_argument('--out', type=Path, default=PATHS.processed)
    s.set_defaults(fn=cmd_build_data)

    s = sub.add_parser('build-team-form', help='refit the team-form state space model')
    s.add_argument('--data', type=Path, default=PATHS.processed)
    s.add_argument('--out', type=Path, default=PATHS.team_form)
    s.add_argument('--through-season', type=int)
    s.add_argument('--draws', type=int, default=1000)
    s.add_argument('--tune', type=int, default=1000)
    s.add_argument('--chains', type=int, default=4)
    s.add_argument('--seed', type=int, default=0)
    s.add_argument('--sampler', choices=['nutpie', 'pymc'], default='nutpie')
    s.set_defaults(fn=cmd_build_team_form)

    s = sub.add_parser('forecast', help='forecast one week with both models')
    s.add_argument('season', type=int)
    s.add_argument('week', type=int)
    s.add_argument('--source', choices=['upcoming', 'played'], default='upcoming',
                   help="upcoming: build rows from the schedule + history (real use); "
                        "played: the week's actual rows (replay)")
    s.add_argument('--schedule', choices=['nflverse', 'processed'], default='nflverse',
                   help='where the games come from for --source upcoming')
    s.add_argument('--rosters', choices=['nflverse', 'none'], default='nflverse',
                   help='current team + active status from nflverse rosters')
    s.add_argument('--include-inactive', action='store_true', help='keep players whose roster status is not ACT')
    s.add_argument('--out', type=Path)
    _add_data_args(s)
    _add_fit_args(s)
    s.set_defaults(fn=cmd_forecast)

    s = sub.add_parser('decide', help='choose a lineup from a saved forecast')
    s.add_argument('forecast', type=Path, help='forecast directory')
    s.add_argument('--roster', required=True, help='player_ids/names (comma list) or a file')
    s.add_argument('--rule', choices=['maximize_expected', 'chance_constrained', 'cvar_averse', 'head_to_head'],
                   default='maximize_expected')
    s.add_argument('--opponent', help="opponent's roster (head_to_head)")
    s.add_argument('--floor', type=float)
    s.add_argument('--beta', type=float, default=0.90)
    s.add_argument('--target', type=float)
    s.add_argument('--alpha', type=float, default=0.10)
    s.add_argument('--lam', type=float, default=1.0)
    s.add_argument('--out', type=Path, help='write the decision as JSON')
    s.set_defaults(fn=cmd_decide)

    s = sub.add_parser('check-forecast', help='tail calibration of a played-week forecast vs outcomes')
    s.add_argument('forecast', type=Path, help='forecast directory (made with --source played)')
    s.set_defaults(fn=cmd_check_forecast)

    s = sub.add_parser('backtest', help='walk-forward folds')
    s.add_argument('--seasons', required=True)
    s.add_argument('--weeks', required=True)
    s.add_argument('--models', help=f"comma list from {list(MODELS)} and 'baseline:<model>'")
    s.add_argument('--run-dir', type=Path)
    s.add_argument('--eligibility', choices=['model', 'legacy'], default='model')
    s.add_argument('--overwrite', action='store_true')
    s.add_argument('--fail-fast', action='store_true')
    _add_data_args(s)
    _add_fit_args(s)
    s.set_defaults(fn=cmd_backtest)

    s = sub.add_parser('summarize-backtest')
    s.add_argument('run_dir', type=Path)
    s.set_defaults(fn=cmd_summarize_backtest)

    s = sub.add_parser('league', help='league simulation of the decision rules')
    s.add_argument('--seasons', required=True)
    s.add_argument('--weeks', required=True)
    s.add_argument('--mock-draft', type=Path, help='mock-draft CSV (else a synthetic trailing-mean draft)')
    s.add_argument('--n-teams', type=int, default=10)
    s.add_argument('--beta', type=float, default=0.90)
    s.add_argument('--alpha', type=float, default=0.10)
    s.add_argument('--lam', type=float, default=1.0)
    s.add_argument('--out-dir', type=Path)
    s.add_argument('--overwrite', action='store_true')
    s.add_argument('--fail-fast', action='store_true')
    _add_data_args(s)
    _add_fit_args(s)
    s.set_defaults(fn=cmd_league)

    s = sub.add_parser('summarize-league')
    s.add_argument('out_dir', type=Path)
    s.set_defaults(fn=cmd_summarize_league)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format='%(asctime)s %(levelname)s %(name)s: %(message)s', datefmt='%H:%M:%S')
    for noisy in ('pymc', 'pytensor', 'numba', 'httpx', 'nutpie', 'arviz'):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    a.fn(a)


if __name__ == '__main__':
    main()
