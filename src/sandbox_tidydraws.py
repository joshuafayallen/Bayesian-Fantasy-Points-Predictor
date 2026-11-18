#  [markdown]
# # Sandbox: tidy forecast draws + Altair
#
# A scratch script for exploring a saved forecast with tidydraws and trying out
# Altair charts before they go into the app. Run it cell by cell (`# `) in
# VS Code / Positron / Jupyter's interactive window, where charts show inline,
# or top to bottom from the project root, which saves each chart as HTML:
#
#     uv run python src/sandbox_tidydraws.py
#
# Nothing here changes the app or the saved forecasts.

#  setup: pick a forecast and your players
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import altair as alt
import numpy as np
import polars as pl
import tidydraws as td
import xarray as xr

from ffpred.config import PATHS
from ffpred.forecast import WeekForecast

alt.data_transformers.disable_max_rows()   # draw-level frames are bigger than Altair's 5,000-row default

FORECAST = PATHS.forecasts_dir / '2026_wk10_upcoming_full'          # any saved forecast folder
ROSTER_FILE = PATHS.root / 'rosters' / 'my_roster.txt'               # or set MY_PLAYERS by hand below
OUT = PATHS.root / 'results' / 'sandbox'                             # where charts are saved when run as a script
N_SHOW = 1500                                                        # draws per player in plots (all 4,000 is slow)

fc = WeekForecast.load(FORECAST)
draws, frame = fc.draws, fc.frame
print(f'{fc.season} week {fc.week:g}: {frame.height} players x {draws.shape[0]:,} draws')

#  tidy draws: one row per player per simulated week
# draws.npy pools the chains, so they're treated as one chain here
dt = xr.DataTree.from_dict({'predictions': xr.Dataset(
    {'points': (('chain', 'draw', 'obs_ind'), draws[np.newaxis])})})
tidy = td.prediction_draws(
    dt, frame.select('player_id', 'player', 'position', 'team', 'opp_team').with_row_index('obs_ind'),
    'points')
tidy.head()

#  your players (from the saved roster, or a hand-written list of names)
_ids = ROSTER_FILE.read_text().split() if ROSTER_FILE.exists() else []
MY_PLAYERS = frame.filter(pl.col('player_id').is_in(_ids))['player'].to_list()
# MY_PLAYERS = ['Jared Goff', 'Amon-Ra St. Brown', 'Jahmyr Gibbs']   # or pick by hand
mine = tidy.filter(pl.col('player').is_in(MY_PLAYERS))
print(MY_PLAYERS)


def show(chart, name):
    """Inline in an interactive window; saved as HTML when run as a script."""
    if 'ipykernel' in sys.modules:
        return chart
    OUT.mkdir(parents=True, exist_ok=True)
    chart.save(OUT / f'{name}.html')
    print('saved', OUT / f'{name}.html')


#  1. summaries: mean with 50% / 80% intervals, and chance of clearing a score
summary = (td.point_interval(mine, 'points', group_by=['player', 'position'], probs=(0.5, 0.8), point='mean')
           .join(mine.group_by('player').agg((pl.col('points') >= 15).mean().alias('p_15plus'),
                                             (pl.col('points') <= 0.05).mean().alias('p_dud')),
                 on='player')
           .sort('points', descending=True)
           # Altair reads '.' in a field name as nesting, so give the interval columns plain names
           .rename({'points_lower_0.50': 'lo50', 'points_upper_0.50': 'hi50',
                    'points_lower_0.80': 'lo80', 'points_upper_0.80': 'hi80'}))
summary

#  2. interval chart: thick bar = 50%, thin = 80%, dot = mean
base = alt.Chart(summary).encode(
    y=alt.Y('player:N', sort='-x', title=None),
    color=alt.Color('position:N', legend=alt.Legend(orient='bottom')))
intervals = (
    base.mark_rule(strokeWidth=1.5).encode(x=alt.X('lo80:Q', title='Fantasy points'), x2='hi80:Q')
    + base.mark_rule(strokeWidth=6, opacity=0.6).encode(x='lo50:Q', x2='hi50:Q')
    + base.mark_point(filled=True, size=60, color='black').encode(
        x='points:Q', tooltip=['player', alt.Tooltip('points:Q', format='.1f', title='mean'),
                               alt.Tooltip('p_15plus:Q', format='.0%', title='P(15+)'),
                               alt.Tooltip('p_dud:Q', format='.0%', title='P(dud)')])
).properties(width=520, height=alt.Step(22), title=f'Your roster, {fc.season} week {fc.week:g}')
show(intervals, '1_intervals')

#  3. distributions: compare a few players' full score distributions
COMPARE = MY_PLAYERS[:4]
sample = (tidy.filter(pl.col('player').is_in(COMPARE) & (pl.col('draw') < N_SHOW)))
density = (alt.Chart(sample)
           .transform_density('points', groupby=['player'], as_=['points', 'density'], extent=[0, 50])
           .mark_area(opacity=0.35, interpolate='monotone')
           .encode(x=alt.X('points:Q', title='Fantasy points'), y=alt.Y('density:Q', title=None, stack=None),
                   color='player:N')
           .properties(width=520, height=220, title='Score distributions'))
show(density, '2_distributions')

#  4. QB link: one QB's draws against his pass-catchers'
QB = (frame.filter(pl.col('position') == 'QB').join(
          frame.filter(pl.col('player').is_in(MY_PLAYERS)).select('team'), on='team')['player'].head(1).to_list()
      or ['Jared Goff'])[0]
team = frame.filter(pl.col('player') == QB)['team'][0]
catchers = (tidy.filter((pl.col('team') == team) & pl.col('position').is_in(['WR', 'TE']))
            .group_by('player').agg(pl.col('points').mean()).sort('points', descending=True)['player'].head(2).to_list())
qb = tidy.filter(pl.col('player') == QB).select('draw', pl.col('points').alias('qb_points'))
pairs = (tidy.filter(pl.col('player').is_in(catchers) & (pl.col('draw') < N_SHOW))
         .join(qb, on='draw'))
print(pairs.group_by('player').agg(pl.corr('qb_points', 'points').alias('corr_with_qb')))
scatter = (alt.Chart(pairs).mark_circle(size=12, opacity=0.25)
           .encode(x=alt.X('qb_points:Q', title=f'{QB} points'), y=alt.Y('points:Q', title='Receiver points'),
                   color='player:N')
           .properties(width=260, height=260)
           .facet(column=alt.Column('player:N', title=None))
           .properties(title=f'Same simulated week: {QB} vs his top pass-catchers'))
show(scatter, '3_qb_link')

#  5. a lineup's total, draw by draw (pick any 7+ names)
LINEUP = MY_PLAYERS[:7]
totals = (tidy.filter(pl.col('player').is_in(LINEUP)).group_by('draw').agg(pl.col('points').sum().alias('total')))
print(totals.select(pl.col('total').mean().alias('mean'), pl.col('total').quantile(0.1).alias('p10'),
                    pl.col('total').quantile(0.9).alias('p90')))
hist = (alt.Chart(totals).mark_bar(opacity=0.8)
        .encode(x=alt.X('total:Q', bin=alt.Bin(maxbins=50), title='Lineup total'), y=alt.Y('count()', title=None))
        .properties(width=520, height=220, title='Lineup total across simulated weeks'))
show(hist, '4_lineup_total')
