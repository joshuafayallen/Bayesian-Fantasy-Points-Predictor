
"""Time the decision engine: enumeration path vs. CVaR MILP, on a saved forecast.
 
Usage (from the project root):
    python model-testing/time_decision_engine.py forecasts/nuts_2025_wk09
"""
import sys
import time
import statistics
 
import numpy as np
import polars as pl
 
from ffpred import decision_engine as de
from ffpred.forecast import WeekForecast
 
FC_DIR = sys.argv[1] if len(sys.argv) > 1 else 'forecasts/nuts_2025_wk09'
REPEATS = 5
POS_ORDER = ['QB', 'RB', 'WR', 'TE']
 
 
def top_pool(frame, draws, per_pos):
    """Top-projected players per position -> list of player_ids (a 'roster' / pool)."""
    exp = draws.mean(axis=0)
    f = frame.with_columns(pl.Series('exp', exp)).with_row_index('col')
    ids = []
    for p, n in per_pos.items():
        ids += f.filter(pl.col('position') == p).sort('exp', descending=True).head(n)['player_id'].to_list()
    return ids
 
 
def timeit(fn, repeats=REPEATS):
    times, out = [], None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times), min(times), max(times), out
 
 
fc = WeekForecast.load(FC_DIR)
draws, frame = fc.draws, fc.frame
print(f'forecast {FC_DIR}: {draws.shape[0]} draws x {draws.shape[1]} players')
 
# season-long roster: ~15 players -> enumeration path
roster = top_pool(frame, draws, {'QB': 2, 'RB': 5, 'WR': 6, 'TE': 2})
n_cand = de.count_candidates(frame, pool=roster)
target = 105.0   # fixed target; set to your league's typical winning score
print(f'\nroster pool: {len(roster)} players, ~{n_cand} candidate lineups, target={target}')
 
for name, fn in [
    ('maximize_expected', lambda: de.maximize_expected(draws, frame, pool=roster)),
    ('cvar_averse (enum)', lambda: de.cvar_averse(draws, frame, target=target, pool=roster)),
]:
    med, lo, hi, _ = timeit(fn)
    print(f'  {name:<22} median {med:7.3f}s  (min {lo:.3f}, max {hi:.3f})')
 
# larger pools -> MILP path
if de.xpress_available():
    for per_pos in ({'QB': 4, 'RB': 10, 'WR': 14, 'TE': 4}, {'QB': 8, 'RB': 18, 'WR': 26, 'TE': 8}):
        pool = top_pool(frame, draws, per_pos)
        for n_s in (100, 300):
            med, lo, hi, choice = timeit(
                lambda: de.cvar_averse_milp(draws, frame, target=target, pool=pool,
                                            n_scenarios=n_s, rng=0), repeats=3)
            print(f'  cvar_averse_milp  pool={len(pool):>3}  scenarios={n_s:>3}  '
                  f'median {med:7.3f}s  (min {lo:.3f}, max {hi:.3f})')
else:
    print('xpress not installed -- skipping MILP timings (pip install xpress)')