#  [markdown]
# # Forecast the upcoming week, in memory
#
# Fits the QB-linked model on everything before the upcoming week (the first week
# after your data) and returns the forecast as an xarray DataTree. Nothing is saved.
#
# Run cell by cell (`# `) in VS Code / Positron so `dt` stays in memory to explore,
# or from the project root:
#
#     uv run python src/forecast_week.py
#
# Rebuild the data first (src/ff.py build-data) so "upcoming" is really next week.

#  setup
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import polars as pl
import xarray as xr
import tidydraws as td
import altair as alt

from ffpred import data as D
from ffpred.config import SamplerSettings, PATHS
from ffpred.forecast import WeekForecast, forecast_week

try:                               # use the season script's speeds, so 'full' means the same everywhere
    from forecast_season import SPEEDS
except ImportError:
    try:
        from forecast import SPEEDS
    except ImportError:
        SPEEDS = {'full': dict(chains=4, draws=2000, tune=2000), 'fast': dict(chains=2, draws=250, tune=250)}

# ------------------------------------------------------------------ settings
SPEED = 'full'     # 'full' (production) or 'fast'
METHOD = 'nuts'    # 'nuts' for real use; 'advi' only for a quick smoke test
SEED = 0
OUT = PATHS.root / 'results' / 'sandbox'    

alt.data_transformers.disable_max_rows()
def forecast_upcoming(speed: str = 'fast', method: str = 'nuts', seed: int = 0) -> xr.DataTree:
    """Fit on all data before the upcoming week and forecast it from the nflverse schedule
    and rosters. Returns the forecast as a DataTree (nothing is written to disk):

      /predictions                 points (chain, draw, player_id) with player/position/team/opp_team
      /predictions_constant_data   each player's model inputs for the week
      /skipped                     players the models couldn't score
      attrs                        season, week, meta (settings, fit diagnostics) as JSON
    """
    import json
    from ffpred.data_build import fetch_schedule
    from ffpred.upcoming import build_upcoming_frame, fetch_rosters

    hist, team_form = D.load_processed(), D.load_team_form()
    last_season, last_week = hist.select('season', 'week').unique().sort('season', 'week').row(-1)
    season, week = (last_season, int(last_week) + 1) if last_week < 18 else (last_season + 1, 1)
    print(f'data ends at {last_season} week {int(last_week)}; forecasting {season} week {week} ({speed}, {method})')

    frame = build_upcoming_frame(hist, fetch_schedule(season, week), season, week,
                                 rosters=fetch_rosters(season, week))
    settings = SamplerSettings(seed=seed, method=method, **SPEEDS[speed])
    fc = forecast_week(hist, team_form, season, week, frame=frame, settings=settings)

    dt = fc.dt
    dt.attrs = {**dt.attrs, 'season': fc.season, 'week': fc.week, 'meta': json.dumps(fc.meta, default=str)}
    return dt


#  fit (takes a few minutes; nothing is saved)
dt = forecast_upcoming(SPEED, METHOD, SEED)

draws = (td.prediction_draws(dt, None, 'points', join_on='player_id')
            .select(pl.col('chain', 'draw', 'player_id', 'player', 'position', 'team', 'season', 'week', 'points'))
            .with_columns(pl.col('points').round(2)))



summary = (td.point_interval(
    draws, 'points', group_by=['player_id', 'position', 'player', 'team'], probs=(0.5,0.80), 
    interval='hdi'
).rename({'points_lower_0.50': 'lo50', 'points_upper_0.50': 'hi50',
                    'points_lower_0.85': 'lo80', 'points_upper_0.85': 'hi80'}))





def show(chart, name):
    """Inline in an interactive window; saved as HTML when run as a script."""
    if 'ipykernel' in sys.modules:
        return chart
    OUT.mkdir(parents=True, exist_ok=True)
    chart.save(OUT / f'{name}.html')
    print('saved', OUT / f'{name}.html')

check = ['Jared Goff', 'Christian McCaffrey', 'Parker Washington', 'Christian Watson', 'Brock Bowers', 'Chase Brown']



roster2 = (
    draws.filter(pl.col("player").is_in(check))
)

roster2

ridge = (
    alt.Chart(
        roster2, height = 38
    )
    .transform_density('points', groupby= ['player', 'position'], as_ = ['points', 'density'], extent = [-5,60], steps = 200)
    .mark_area(interpolate = 'monotone', fillOpacity=0.75, stroke = 'white', strokeWidth=0.8)
    .encode(x = alt.X('points:Q', title = 'Fantasy points'), 
            y = alt.Y('density:Q', axis = None, scale = alt.Scale(range=[38, -38 * 1.6])),
            color = alt.Color('player:N', legend = None))
    .properties(width=560)
    .facet(row=alt.Row('player:N', sort=alt.EncodingSortField(field='player', op='min'), title=None,
                       header=alt.Header(labelAngle=0, labelAlign='left', labelOrient='left', labelPadding=4)))
    .properties(bounds='flush', title=f'Score distributions')
    .configure_facet(spacing=0)
    .configure_view(stroke=None)
    .configure_title(anchor='end')
)

