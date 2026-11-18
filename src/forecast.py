import argparse
import logging
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
 
sys.path.insert(0, str(Path(__file__).resolve().parent))
 
import polars as pl
 
from ffpred import data as D
from ffpred.config import MAX_WEEK, PATHS, QB_LINKED, SamplerSettings
 
SPEEDS = {'full': dict(chains=4, draws=2000, tune=2000), 'fast': dict(chains=2, draws=250, tune=250)}
 
 
def parse_weeks(spec: str) -> list[int]:
    """'1-10' / '3,5,9' / '1-4,9' -> sorted week numbers."""
    out = set()
    for part in spec.split(','):
        a, _, b = part.partition('-')
        out.update(range(int(a), int(b or a) + 1))
    return sorted(out)
 
 
def out_dir(season, week, source, speed, method='nuts') -> Path:
    tag = speed if method == 'nuts' else method
    return PATHS.forecasts_dir / f'{season}_wk{week:02d}_{source}_{tag}'
 
 
def fit_one(season, week, source, frame, settings, out):
    """One week, in a worker process: load the data, fit, save atomically."""
    logging.basicConfig(level=logging.WARNING)
    from ffpred.forecast import forecast_week
    t0 = time.time()
    hist, tf = D.load_processed(), D.load_team_form()
    fc = forecast_week(hist, tf, season, week, frame=frame, settings=settings, model=QB_LINKED)
    tmp = out.with_name(out.name + '.partial')
    shutil.rmtree(tmp, ignore_errors=True)
    fc.save(tmp)
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)
    return dict(week=week, source=source, minutes=(time.time() - t0) / 60, players=fc.frame.height,
                diag={k: v.get('rhat_max') for k, v in fc.meta.get('fits', {}).items() if isinstance(v, dict)})
 
 
def plan(hist, season, weeks, include_next):
    """(week, source) for each requested week: played if the week is in your data, else
    upcoming for the first week after it; later weeks have no games or rosters yet."""
    played = set(hist.filter(pl.col('season') == season)['week'].cast(int).unique().to_list())
    nxt = max(played) + 1 if played else 1
    jobs = []
    for w in weeks:
        if w in played:
            jobs.append((w, 'played'))
        elif w == nxt and include_next:
            jobs.append((w, 'upcoming'))
    return jobs, nxt
 
 
def main(a):
    hist = D.load_processed()
    season = a.season or int(hist['season'].max())
    weeks = parse_weeks(a.weeks) if a.weeks else list(range(1, MAX_WEEK + 1))
    jobs, nxt = plan(hist, season, weeks, include_next=not a.no_next)
    settings = SamplerSettings(seed=a.seed, method=a.method, **SPEEDS[a.speed])
    todo = [(w, src) for w, src in jobs if a.force or not (out_dir(season, w, src, a.speed, a.method) / 'meta.json').exists()]
    n_jobs = a.jobs or max(1, (os.cpu_count() or 4) // settings.chains)
    print(f'{season}: {len(jobs)} weeks requested, {len(jobs) - len(todo)} already saved, '
          f'{len(todo)} to fit ({a.speed}: {settings.chains} chains x {settings.draws}), {n_jobs} at a time')
    if not todo:
        return
 
    frames = {}
    for w, src in todo:                         # network calls stay in this process
        if src == 'upcoming':
            from ffpred.data_build import fetch_schedule
            from ffpred.upcoming import build_upcoming_frame, fetch_rosters
            frames[w] = build_upcoming_frame(hist, fetch_schedule(season, w), season, w,
                                             rosters=fetch_rosters(season, w))
 
    failed = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        futs = {pool.submit(fit_one, season, w, src, frames.get(w), settings, out_dir(season, w, src, a.speed, a.method)):
                (w, src) for w, src in todo}
        for i, f in enumerate(as_completed(futs), 1):
            w, src = futs[f]
            try:
                r = f.result()
                print(f'[{i}/{len(todo)}] wk{w:02d} {src}: {r["players"]} players, {r["minutes"]:.1f} min, '
                      f'rhat_max {r["diag"]}', flush=True)
            except Exception as e:              # one bad week shouldn't stop the season
                failed.append((w, e))
                print(f'[{i}/{len(todo)}] wk{w:02d} {src}: FAILED {type(e).__name__}: {e}', flush=True)
    print(f'done in {(time.time() - t0) / 60:.1f} min -> {PATHS.forecasts_dir}')
    if failed:
        print('failed weeks:', ', '.join(f'{w}' for w, _ in failed), '(rerun to retry just these)')
 
 
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Fit and save a forecast for every week of a season, for the app to load.')
    ap.add_argument('--season', type=int, help='default: the latest season in your data')
    ap.add_argument('--weeks', help="e.g. '1-10' or '3,5,9' (default: every week played so far, plus the next)")
    ap.add_argument('--speed', choices=list(SPEEDS), default='full',
                    help='full: 4 chains x 1000 (production); fast: 2 chains x 250')
    ap.add_argument('--jobs', type=int, help='weeks fit at once (default: cores // chains)')
    ap.add_argument('--no-next', action='store_true', help="don't forecast the upcoming week")
    ap.add_argument('--force', action='store_true', help='refit weeks that are already saved')
    ap.add_argument('--method', choices=['nuts', 'pathfinder', 'advi'], default='nuts',
                    help='advi only for smoke tests (see check_approx.py)')
    ap.add_argument('--seed', type=int, default=0)
    main(ap.parse_args())