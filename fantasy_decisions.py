from altair import StrokeDash
from sympy import interpolate
from narwhals import group_by
import marimo

__generated_with = "0.25.2"
app = marimo.App(width = 'medium', app_title = 'Fantasy lineup lab', css_file = 'lineup.css')


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # A Fantasy Football Lineup

    Want to make uncertainty aware lineup decisions? Then this forecaster is for you! This is fully Bayesian model which means you get honest uncertainty about how the uncertain your players' performance could be. To make use of distribution of predictions you can also use several decision rules to make (hopefully) better lineup decisions.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Getting started

    Dependecies for this project are managed via `uv` so if you haven't installed it you first run

    ```
    # Mac or Linux
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Windows
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```


    After installation just run

    ```
    uv sync
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup
    """)
    return


@app.cell
def _():
    import json
    import sys
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import numpy as np
    import polars as pl
    sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
    from ffpred import decision_engine as de
    from ffpred.config import PATHS, TARGET_COL
    from ffpred.forecast import WeekForecast
    from ffpred.upcoming import build_upcoming_frame, fetch_rosters
    ROSTER_DIR = PATHS.root / 'rosters'      # remembered rosters (player_ids, one per line)
    LINEUP_DIR = PATHS.root / 'lineups'      # saved lineups, one JSON per forecast week

    alt.data_transformers.disable_max_rows()
    return (
        LINEUP_DIR,
        PATHS,
        Path,
        ROSTER_DIR,
        TARGET_COL,
        WeekForecast,
        alt,
        build_upcoming_frame,
        de,
        fetch_rosters,
        json,
        mo,
        np,
        pl,
    )


@app.cell
def _(mo):
    mo.md(r"""
    Select the week you want to forecast first. Then set your lineup
    """)
    return


@app.cell(hide_code=True)
def _():
    from ffpred import data as D
    from ffpred.config import SamplerSettings

    history, team_form = D.load_processed(), D.load_team_form()
    _last = history.select('season', 'week').unique().sort('season', 'week').row(-1)
    # default: the week after the last one in the data (the week you're setting a lineup for)
    next_season, next_week = (_last[0], int(_last[1]) + 1) if _last[1] < 18 else (_last[0] + 1, 1)
    return D, SamplerSettings, history, next_season, next_week, team_form


@app.cell(hide_code=True)
def _(mo, next_season, next_week):
    _args = mo.cli_args()
    season_in = mo.ui.number(start = 2010, stop = 2026, step = 1, value = int(_args.get('season') or next_season), label = 'Season')
    week_in = mo.ui.number(start = 1, stop = 18, step = 1, value = int(_args.get('week') or next_week), label = 'Week')
    mo.hstack([season_in, week_in], justify = 'start')
    return season_in, week_in


@app.cell(hide_code=True)
def _(fetch_rosters, pl):
    from functools import lru_cache

    @lru_cache(maxsize=8)
    def roster_feed(season: int, week: int):
        """nflreadpy rosters as of `week`, fetched once per season/week: each player's latest
        listing from the last two published weeks (a week's feed leaves out teams on bye)."""
        r = fetch_rosters(season, week, all_weeks=True)
        if 'week' not in r.columns:                   # weekly feed not out yet: season roster
            return r
        return (r.filter(pl.col('week') >= r['week'].max() - 1)
                .unique(subset=['player_id'], keep='last', maintain_order=True).drop('week'))

    return (roster_feed,)


@app.cell(hide_code=True)
def _(history, mo, pl, roster_feed, season_in, week_in):
    # The player list comes from the nflverse rosters for the selected week, so rookies,
    # signings and trades show with their current team; the processed data adds recent scoring.
    _s, _w = int(season_in.value), int(week_in.value)
    _skill = ['QB', 'RB', 'WR', 'TE']
    STATUS = {'ACT': '', 'RES': 'IR', 'INA': 'inactive', 'DEV': 'practice squad'}
    _recent = (history.filter(pl.col('season') <= _s).sort('player_id', 'season', 'week')
               .group_by('player_id', maintain_order=True)
               .agg(pl.col('player').last().alias('_hist_name'),
                    pl.col('total_fantasy_points').tail(8).mean().alias('recent')))
    try:
        nfl_roster = roster_feed(_s, _w)
        players = (nfl_roster.filter(pl.col('position').is_in(_skill) & pl.col('status').is_in(list(STATUS)))
                   .join(_recent, on='player_id', how='left')
                   .with_columns(pl.coalesce('player', '_hist_name').alias('player')).drop('_hist_name'))
        roster_source = mo.md(f'Players: nflverse rosters for {_s} (latest listing up to week {_w}), '
                              f'{players.height} QB/RB/WR/TE.').style({'color': 'var(--muted-foreground, #888)'})
    except Exception as e:     # offline, or the season isn't published: fall back to the processed data
        nfl_roster = None
        players = (history.filter((pl.col('season') >= _s - 1) & (pl.col('season') <= _s)
                                  & pl.col('position').is_in(_skill))
                   .sort('player_id', 'season', 'week').group_by('player_id', maintain_order=True)
                   .agg(pl.col('player').last(), pl.col('position').last(), pl.col('team').last())
                   .with_columns(pl.lit('ACT').alias('status'))
                   .join(_recent.drop('_hist_name'), on='player_id', how='left'))
        roster_source = mo.callout(mo.md(f'Could not load nflverse rosters for {_s} (`{e}`); '
                                         'using players with games in your data this season or last.'), kind='warn')
    players = players.sort('position', 'recent', descending=[False, True], nulls_last=True)
    return STATUS, nfl_roster, players, roster_source


@app.cell(hide_code=True)
def _(STATUS, history, pl, players):
    labels = {f"{r['player']} · {r['position']} · {r['team']}" + (f" ({STATUS[r['status']]})" if STATUS.get(r['status']) else ''):
              r['player_id'] for r in players.iter_rows(named=True)}
    label_of = {v: k for k, v in labels.items()}

    # each player's last known name / position / team in your data (sort first, then take .last())
    hist_players = (history.sort('season', 'week').group_by('player_id')
                    .agg(pl.col('player').last(), pl.col('position').last(), pl.col('team').last()))
    name_of = dict(hist_players.select('player_id', 'player').iter_rows())
    return label_of, labels, name_of


@app.cell(hide_code=True)
def _(ROSTER_DIR, history, mo, pl):
    # Your picks, as player_ids. Nothing here changes with the season, week or filters,
    # so this cell never reruns and your picks survive changing them.
    def _demo_roster(offset):
        """Top recent scorers in your data, dealt round-robin into rosters (for trying the app)."""
        _last = history.filter(pl.col('season') == history['season'].max())
        picks = []
        for pos, k in (('QB', 2), ('RB', 5), ('WR', 5), ('TE', 2)):
            ranked = (_last.filter(pl.col('position') == pos).group_by('player_id')
                      .agg(pl.col('total_fantasy_points').mean()).sort('total_fantasy_points', descending=True))
            picks += ranked['player_id'].to_list()[offset::10][:k]
        return picks

    def _saved(name):
        f = ROSTER_DIR / name
        return f.read_text().split() if f.exists() else []

    _demo = mo.cli_args().get('demo')
    get_mine, set_mine = mo.state(_demo_roster(0) if _demo else _saved('my_roster.txt'))
    get_opp, set_opp = mo.state(_demo_roster(1) if _demo else _saved('opponent.txt'))
    return get_mine, get_opp, set_mine, set_opp


@app.cell(hide_code=True)
def _(ROSTER_DIR, mo, players, roster_source, set_mine, set_opp):
    roster_pos = mo.ui.multiselect(options = ['QB', 'RB', 'WR', 'TE'], label = 'Positions (all if empty)')
    roster_team = mo.ui.multiselect(options = sorted(players['team'].drop_nulls().unique().to_list()), 
                                    label = 'Teams (all if empty)')
    roster_active = mo.ui.checkbox(label = 'Active only (hide IR, inactive, practice squad)', value = False)


    try:
        _size = int((ROSTER_DIR / 'roster_size.txt').read_text().strip())
    except (OSError, ValueError):
        _size = 16

    roster_max = mo.ui.number(start = 7, stop = 40, step = 1, value = _size)
    clear_mine = mo.ui.button(label = 'Clear my roster', kind = 'warn', on_click = lambda _: set_mine([]))
    clear_opp = mo.ui.button(label = "Clear opponent's roster", kind = 'warn', on_click = lambda _: set_opp([]))


    mo.vstack([mo.md('## Rosters'), roster_source,
            mo.hstack([roster_pos, roster_team, roster_active], justify = 'start', align = 'end'),
            mo.hstack([roster_max, clear_mine, clear_opp], justify = 'start', align = 'end')])
    return roster_active, roster_pos, roster_team


@app.cell(hide_code=True)
def _(
    ROSTER_DIR,
    get_mine,
    get_opp,
    label_of,
    labels,
    mo,
    name_of,
    pl,
    players,
    roster_active,
    roster_pos,
    roster_team,
    season_in,
    set_mine,
    set_opp,
    week_in,
):
    _shown = players
    if roster_pos.value:
        _shown = _shown.filter(pl.col('position').is_in(roster_pos.value))
    if roster_team.value:
        _shown = _shown.filter(pl.col('team').is_in(roster_team.value))
    if roster_active.value:
        _shown = _shown.filter(pl.col('status') == 'ACT')
    _keep = set(_shown['player_id'])

    def _label(p):
        # a pick not on an NFL roster in this season/week keeps a placeholder instead of being dropped
        return label_of.get(p) or f"{name_of.get(p, p)} (not on a roster in {season_in.value} wk {week_in.value})"

    def _options(selected):
        # filtered players, plus whoever is already picked so a filter never drops a pick
        return {**{k: v for k, v in labels.items() if v in _keep}, **{_label(p): p for p in selected}}

    roster_pick = mo.ui.multiselect(options=_options(get_mine()), value=[_label(p) for p in get_mine()],
                                    on_change=set_mine, label='Your roster',
                                    full_width=True)
    opp_pick = mo.ui.multiselect(options=_options(get_opp()), value=[_label(p) for p in get_opp()],
                                 on_change=set_opp, full_width=True,
                                 label="Opponent's roster (optional: enables head-to-head and opponent targets)")
    pasted = mo.ui.text_area(label='…or paste your roster (names, GSIS, Sleeper, ESPN or Yahoo ids; one per line)',
                             rows=3, full_width=True)
    save_rosters = mo.ui.run_button(label='Remember these rosters',
                                    tooltip=f'Saves player_ids to {ROSTER_DIR}/my_roster.txt and opponent.txt; '
                                            'they are preselected next time')
    mo.vstack([mo.md(f'{len(_keep)} players match the filters.'), roster_pick, pasted, opp_pick, save_rosters])
    return opp_pick, pasted, roster_pick, save_rosters


@app.cell(hide_code=True)
def _(ROSTER_DIR, mo, opp_pick, pasted, players, roster_pick, save_rosters):
    import re as _re

    def _norm(name):
        """'D.J. Moore' == 'DJ Moore', 'Kenneth Walker III' == 'Kenneth Walker'."""
        n = _re.sub(r"[.'’]", '', name.lower())
        n = _re.sub(r'\b(jr|sr|ii|iii|iv|v)\b', '', n)
        return ' '.join(n.replace('-', ' ').split())

    _lookup = {}
    for _r in players.iter_rows(named=True):                  # sorted by recent scoring: best match first
        for _k in [_r['player_id'], *[_r.get(c) for c in ('sleeper_id', 'espn_id', 'yahoo_id')]]:
            if _k:
                _lookup.setdefault(str(_k), _r['player_id'])
        _lookup.setdefault(_norm(_r['player']), _r['player_id'])
    _typed = [t.strip() for t in pasted.value.splitlines() if t.strip()]
    _resolved = [_lookup.get(t) or _lookup.get(_norm(t)) for t in _typed]
    _unknown = [t for t, pid in zip(_typed, _resolved) if pid is None]
    roster_ids = list(dict.fromkeys([*roster_pick.value, *[p for p in _resolved if p]]))
    opp_ids = list(opp_pick.value)
    _note = (mo.callout(mo.md('Not recognised (typo, or not on an NFL roster this week): ' + ', '.join(_unknown)),
                        kind='warn') if _unknown else mo.md(''))
    if save_rosters.value:
        ROSTER_DIR.mkdir(parents=True, exist_ok=True)
        (ROSTER_DIR / 'my_roster.txt').write_text('\n'.join(roster_ids) + '\n')
        (ROSTER_DIR / 'opponent.txt').write_text('\n'.join(opp_ids) + '\n')
        _note = mo.vstack([_note, mo.callout(mo.md(f'Saved rosters to `{ROSTER_DIR}`.'), kind='success')])
    _note
    return opp_ids, roster_ids


@app.cell(hide_code=True)
def _(PATHS, Path, built_dir, json, season_in, week_in):
    def find_forecasts(root: Path, season: int, week: int) -> dict[str, Path]:
        """Saved WeekForecast directories for (season, week) under forecasts/ and results/, newest first."""
        out = []
        for base in (root / 'forecasts', root / 'results'):
            for m in base.rglob('meta.json') if base.exists() else []:
                d = m.parent
                if d.name.endswith('.partial') or not (d / 'meta.json').exists():   # meta.json is written last
                    continue
                meta = json.loads(m.read_text())
                if int(meta.get('season', -1)) == season and float(meta.get('week', -1)) == week:
                    out.append(d)
        out.sort(key=lambda d: (d / 'meta.json').stat().st_mtime, reverse=True)
        return {f"{d.relative_to(root)}  ·  {json.loads((d / 'meta.json').read_text()).get('forecast_model', 'independent')}": d
                for d in out}

    built_dir                               # re-scan after a build
    forecasts = find_forecasts(PATHS.root, int(season_in.value), int(week_in.value))
    return (forecasts,)


@app.cell(hide_code=True)
def _(D, history, mo, season_in, week_in):
    _played = D.week_rows(history, int(season_in.value), float(week_in.value)).height > 0
    speed = mo.ui.radio(options={'Full (NUTS 4 × 1000, several minutes)': 'full',
                                 'Fast (NUTS 2 × 250, about half the time)': 'fast'},
                        value='Fast (NUTS 2 × 250, about half the time)', label='Fit')
    source = mo.ui.radio(options={'Upcoming: schedule + nflverse rosters for the week': 'upcoming',
                                  "Played: the week's actual rows (to check a past week)": 'played'},
                         value="Played: the week's actual rows (to check a past week)" if _played
                         else 'Upcoming: schedule + nflverse rosters for the week', label='Rows')
    save_fc = mo.ui.checkbox(label='Save to forecasts/ (off: keep it in memory for this session only)', value=False)
    build = mo.ui.run_button(label=f'Build forecast for {int(season_in.value)} week {int(week_in.value)}')
    mo.accordion({'Build a forecast for this week': mo.vstack([speed, source, save_fc, build])},
                 lazy=False)
    return build, save_fc, source, speed


@app.cell(hide_code=True)
def _(
    PATHS,
    SamplerSettings,
    build,
    build_upcoming_frame,
    history,
    mo,
    nfl_roster,
    roster_feed,
    save_fc,
    season_in,
    source,
    speed,
    team_form,
    week_in,
):
    built_dir, built_fc = None, None
    _msg = mo.md('')
    if build.value:
        from ffpred.forecast import forecast_week
        _s, _w = int(season_in.value), int(week_in.value)
        _settings = SamplerSettings() if speed.value == 'full' else SamplerSettings(chains=2, draws=250, tune=250)
        _out = PATHS.forecasts_dir / f'{_s}_wk{_w:02d}_{source.value}_{speed.value}'
        try:
            with mo.status.spinner(title=f'Forecasting {_s} week {_w} ({speed.value} fit)…',
                                   subtitle='fetching schedule and rosters, then fitting both stages'):
                if source.value == 'upcoming':
                    from ffpred.data_build import fetch_schedule
                    _frame = build_upcoming_frame(history, fetch_schedule(_s, _w), _s, _w,
                                                  rosters=nfl_roster if nfl_roster is not None
                                                  else roster_feed(_s, _w))
                else:
                    _frame = None                      # the week's played rows
                built_fc = forecast_week(history, team_form, _s, _w, frame=_frame, settings=_settings)
            if save_fc.value:
                built_fc.save(_out)
                built_dir = _out
                _msg = mo.callout(mo.md(f'Saved `{_out.relative_to(PATHS.root)}` and selected it below.'), kind='success')
            else:
                _msg = mo.callout(mo.md(f'Built {_s} week {_w} in memory (not saved); it is used below until '
                                        'you rebuild or change the week.'), kind='success')
        except Exception as e:                         # no games / no rosters / not enough history
            _msg = mo.callout(mo.md(f'Could not build this forecast: `{e}`'), kind='danger')
    _msg
    return built_dir, built_fc


@app.cell(hide_code=True)
def _(built_dir, built_fc, forecasts, mo, season_in, week_in):
    mo.stop(not forecasts and built_fc is None, mo.callout(mo.md(
        f'No saved forecast for {int(season_in.value)} week {int(week_in.value)} yet: open '
        '**Build a forecast for this week** above.'), kind='warn'))
    if built_fc is not None and built_dir is None:      # just built, not saved: use it
        fc_pick = None
        _out = mo.md('**Forecast:** the one you just built (in memory, not saved).')
    else:
        _default = next((k for k, d in forecasts.items() if built_dir is not None and d == built_dir),
                        next(iter(forecasts)))
        fc_pick = mo.ui.dropdown(options=forecasts, value=_default, label='Forecast', full_width=True)
        _out = fc_pick
    _out
    return (fc_pick,)


@app.cell(hide_code=True)
def _(TARGET_COL, WeekForecast, built_fc, fc_pick, mo, np, pl):
    fc = built_fc if fc_pick is None else WeekForecast.load(fc_pick.value)
    frame, draws = fc.frame, fc.draws
    has_outcomes = bool(fc.meta.get('has_outcomes')) and TARGET_COL in frame.columns
    col_of = {pid: i for i, pid in enumerate(frame['player_id'].to_list())}

    summary = (frame.select('player_id', 'player', 'position', 'team', 'opp_team')
               .with_columns(pl.Series('proj', draws.mean(0)),
                             pl.Series('floor_p10', np.percentile(draws, 10, axis=0)),
                             pl.Series('ceiling_p90', np.percentile(draws, 90, axis=0)),
                             pl.Series('sd', draws.std(0)),
                             pl.Series('p_dud', (draws <= 0.05).mean(0)))
               .with_columns(pl.col('proj', 'floor_p10', 'ceiling_p90', 'sd').round(1), pl.col('p_dud').round(2)))
    if has_outcomes:
        summary = summary.with_columns(frame[TARGET_COL].alias('actual'))
    summary = summary.sort('proj', descending=True)

    mo.hstack([
        mo.stat(f'{fc.season} wk {fc.week:g}', label='Week'),
        mo.stat(str(frame.height), label='Players forecast'),
        mo.stat(f'{draws.shape[0]:,}', label='Draws'),
        mo.stat('replayed' if has_outcomes else 'upcoming', label='Outcomes'),
    ], justify='start', gap=2)
    return col_of, draws, fc, frame, has_outcomes, summary


@app.cell(hide_code=True)
def _(fc, pl):
    import tidydraws as td
    # the forecast as tidy draws: one row per player per simulated week (for tidydraws / Altair).
    # Named `tidy`, not `draws`: `draws` is the matrix the lineup optimizer uses.
    tidy = fc.tidy().select('chain', 'draw', 'player_id', 'player', 'position', 'team', 'points')
    return td, tidy


@app.cell(hide_code=True)
def _(mo, pool, summary):
    _names = summary['player'].to_list()
    _mine = summary.filter(summary['player_id'].is_in(pool))['player'].to_list()
    ridge_pick = mo.ui.multiselect(options=_names, value=_mine[:10], label='Players to compare', full_width=True)
    ridge_pick
    return (ridge_pick,)


@app.cell(hide_code=True)
def _(alt, fc, mo, pl, ridge_pick, tidy):
    mo.stop(not ridge_pick.value, mo.md('_Pick players above to compare their score distributions._'))
    _per_chain = max(1, 1500 // fc.points.sizes['chain'])            # ~1,500 draws per player is plenty
    _d = (tidy.filter(pl.col('player').is_in(ridge_pick.value) & (pl.col('draw') < _per_chain))
          .with_columns(pl.col('points').mean().over('player').alias('mean'))   # each player's mean, on every row
          .select('player', 'mean', 'points'))
    
    _area = (alt.Chart()
            .transform_density('points', groupby = ['player', 'mean'], as_ = ['points', 'density'], extent = [-5, 60], steps = 200)
            .mark_area(interpolate = 'monotone', fillOpacity = 0.75, stroke = 'white', strokeWidth = 0.8)
            .encode(x = alt.X('points:Q', title = 'Fantasy POints'), 
                    y = alt.Y('density:Q', axis=None, scale=alt.Scale(range=[38, -38 * 1.6])),
                    color = alt.Color('player:N', legend = None))
            )
    _mean = (alt.Chart()
            .mark_rule(strokeDash = [4,3], strokeWidth = 1.5, color = "#222", opacity = 0.8)
            .encode(x = 'mean:Q', tooltip = [alt.Tooltip('mean:Q', format = '.1f', title = 'mean')])
    )
    ridge = (
        alt.layer(_area, _mean, data = _d, height = 38, width = 560)
        .facet(row = alt.Row('player:N', sort = alt.EncodingSortField(field = 'mean', op = 'max', order = 'descending'),
        title = None, header = alt.Header(labelAngle=0, labelAlign='left', labelOrient='left', labelPadding=4)))
        .properties(bounds='flush', title=f'Score distributions, {fc.season} week {fc.week:g}')
        .configure_facet(spacing=0).configure_view(stroke=None).configure_title(anchor='end'))
    ridge
    return


@app.cell(hide_code=True)
def _(mo, summary):
    pos_filter = mo.ui.multiselect(options=['QB', 'RB', 'WR', 'TE'], value=['QB', 'RB', 'WR', 'TE'], label='Positions')
    team_filter = mo.ui.multiselect(options=sorted(summary['team'].unique().to_list()), label='Teams (all if empty)')
    mo.hstack([pos_filter, team_filter], justify='start')
    return pos_filter, team_filter


@app.cell(hide_code=True)
def _(mo, pl, pos_filter, summary, team_filter):
    shown = summary.filter(pl.col('position').is_in(pos_filter.value))
    if team_filter.value:
        shown = shown.filter(pl.col('team').is_in(team_filter.value))
    mo.vstack([
        mo.md('## Research Potential Waiver Wire Pickups'),
        mo.ui.table(shown.drop('player_id'), selection=None, page_size=15, show_download=True),
        mo.md('_proj_ = mean of the draws; _floor_ / _ceiling_ = 10th / 90th percentile; '
              '_p_dud_ = chance of ≤ 0.05 points.'),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    rule = mo.ui.radio(
        options={'Maximize expected points': 'maximize_expected',
                 'CVaR-averse (protect against bad weeks)': 'cvar_averse',
                 'Chance-constrained (clear a floor with probability β)': 'chance_constrained',
                 'Head-to-head (maximize P(win))': 'head_to_head'},
        value='Maximize expected points', label='Decision rule')
    alpha = mo.ui.slider(0.05, 0.30, step=0.05, value=0.10, label='Protect against worse x% of weeks', show_value=True)
    lam = mo.ui.slider(0.0, 5.0, step=0.5, value=1.0, label='5 = Really Conservative', show_value=True)
    beta = mo.ui.slider(0.50, 0.95, step=0.05, value=0.90, label='Probability of beating opponent', show_value=True)
    floor = mo.ui.number(start=0, stop=250, step=1, value=None, label="Points to beat (blank = opponent's projection)")
    mo.vstack([mo.md('## Decision'), mo.md("For user reference the CVAR-averse framework tends to collapse to taking the mean projections."),rule, mo.hstack([alpha, lam, beta, floor], justify='start')])
    return alpha, beta, floor, lam, rule


@app.cell(hide_code=True)
def _(col_of, frame, mo, name_of, opp_ids, players, roster_ids):
    from ffpred.config import FLEX_ELIGIBLE as _FLEX, N_FLEX as _N_FLEX, ROSTER as _ROSTER

    # match the rosters against the loaded forecast: only players in it can start
    pool = [p for p in roster_ids if p in col_of]
    opp_pool = [p for p in opp_ids if p in col_of]
    _pos_of = dict(zip(frame['player_id'].to_list(), frame['position'].to_list()))
    _names = {**name_of, **dict(zip(players['player_id'], players['player']))}

    def _count(ids):
        return {p: sum(_pos_of[i] == p for i in ids) for p in _ROSTER}

    def _gaps(ids):
        """What a roster still needs for a legal lineup, e.g. ['1 more WR']; empty = it can field one."""
        n = _count(ids)
        gaps = [f'{k - n[p]} more {p}' for p, k in _ROSTER.items() if n[p] < k]
        spare = sum(max(n[p] - _ROSTER[p], 0) for p in _FLEX)
        if spare < _N_FLEX:
            gaps.append(f'{_N_FLEX - spare} more RB/WR/TE for FLEX')
        return gaps

    def _have(ids):
        return ', '.join(f'{k} {p}' for p, k in _count(ids).items())

    my_gaps = _gaps(pool)
    # Your opponent's lineup isn't your decision: anyone of his outside this forecast (bye,
    # inactive, no data) counts as 0 points, and any slot he still can't fill is an empty
    # slot worth 0. So his lineup is always complete; the note below says what was zeroed.
    _all_pos = dict(zip(players['player_id'].to_list(), players['position'].to_list()))
    opp_zero = [(p, _names.get(p, p), _all_pos.get(p)) for p in opp_ids
                if p not in col_of and _all_pos.get(p) in _ROSTER]
    _n = {q: sum(_pos_of[i] == q for i in opp_pool) + sum(z[2] == q for z in opp_zero) for q in _ROSTER}
    _empty = [q for q, k in _ROSTER.items() for _ in range(max(k - _n[q], 0))]
    _spare = sum(max(_n[q] - _ROSTER[q], 0) for q in _FLEX)
    _empty += ['RB'] * max(_N_FLEX - _spare, 0)                       # an empty FLEX slot
    opp_zero += [(f'__empty_{q}_{i}', f'(empty {q} slot)', q) for i, q in enumerate(_empty)]
    opp_gaps = [] if opp_ids else ['an opponent roster']
    # why a rostered player can't start: his team has no game in this forecast (bye), or he
    # isn't in it (inactive, injured, or no games in your data yet)
    _team_of = dict(zip(players['player_id'].to_list(), players['team'].to_list()))
    _playing = set(frame['team'].to_list())

    def _why(p):
        t = _team_of.get(p)
        return f'{_names.get(p, p)} ({t} bye)' if t and t not in _playing else f'{_names.get(p, p)}'

    _notes = []
    for _who, _ids in (('Your roster', roster_ids), ('Opponent', opp_ids)):
        _out = [p for p in _ids if p not in col_of]
        if _out:
            _notes.append(mo.callout(mo.md(f"**{_who}: can't start this week** (bye, inactive, or no games in "
                                           f"your data yet): " + ', '.join(_why(p) for p in _out)), kind='warn'))
    _notes.append(mo.md(f'**Your roster in this forecast:** {len(pool)} players ({_have(pool)})'
                        + (f". Still needs {', '.join(my_gaps)}." if my_gaps else ' (lineup ready).')))
    if not opp_ids:
        _notes.append(mo.md("**Opponent:** none entered (optional: enables head-to-head and opponent targets)."))
    elif opp_zero:
        _zeroed = [n for p, n, _ in opp_zero if not p.startswith('__empty')]
        _slots = [q for p, _, q in opp_zero if p.startswith('__empty')]
        _notes.append(mo.callout(mo.md(
            f"**Opponent in this forecast:** {len(opp_pool)} players ({_have(opp_pool)}). "
            + (f"Counted as **0 points**: {', '.join(_zeroed)}. " if _zeroed else '')
            + (f"Empty slots counted as 0: {', '.join(_slots)}. " if _slots else '')
            + "His projection may be **too low** if any of them actually plays (e.g. a questionable "
              "player who suits up), so treat P(win) as optimistic."), kind='warn'))
    else:
        _notes.append(mo.md(f'**Opponent in this forecast:** {len(opp_pool)} players ({_have(opp_pool)}) (lineup ready).'))
    mo.vstack(_notes)
    return my_gaps, opp_gaps, opp_pool, opp_zero, pool


@app.cell(hide_code=True)
def _(alpha, beta, de, draws, floor, frame, lam, mo, my_gaps, np, opp_gaps, opp_pool, opp_zero, pl, pool, rule):
    mo.stop(bool(my_gaps), mo.callout(mo.md(
        f"Your roster can't field a lineup from this forecast yet: it needs {', '.join(my_gaps)}."), kind='info'))
    _opp_ready = not opp_gaps
    mo.stop(rule.value == 'head_to_head' and not _opp_ready, mo.callout(mo.md(
        "Head-to-head needs your opponent's roster (see the opponent line above)."), kind='warn'))
    # the opponent's lineup: his best expected lineup from his players in the forecast plus his
    # zero-point stand-ins (built separately, so his roster can't break or be blamed for yours)
    opp = None
    if _opp_ready:
        _zf = pl.DataFrame({'player_id': [z[0] for z in opp_zero], 'player': [z[1] for z in opp_zero],
                            'position': [z[2] for z in opp_zero]}, schema={'player_id': pl.String,
                                                                         'player': pl.String, 'position': pl.String})
        _odraws = np.hstack([draws, np.zeros((draws.shape[0], _zf.height))])
        _oframe = pl.concat([frame, _zf], how='diagonal_relaxed')
        opp = de.maximize_expected(_odraws, _oframe, pool=[*opp_pool, *_zf['player_id'].to_list()])
    try:
        base = de.maximize_expected(draws, frame, pool=pool)
        kw = dict(alpha=alpha.value, lam=lam.value, beta=beta.value,
                  opp_totals=None if opp is None else opp.totals)
        if floor.value is not None:
            kw['floor'] = kw['target'] = float(floor.value)
        choice = base if rule.value == 'maximize_expected' else de.decide(rule.value, draws, frame, pool, **kw)
    except ValueError as e:
        mo.stop(True, mo.callout(mo.md(f"Can't build a legal lineup from **your** roster: {e}"), kind='danger'))
    return base, choice, opp


@app.cell(hide_code=True)
def _(base, choice, de, has_outcomes, mo, opp, summary):
    p_beats = de.p_beats
    lineup = (choice.lineup.select('slot', 'player_id')
              .join(summary, on='player_id', how='left', maintain_order='left').drop('player_id'))
    st = choice.stats
    same = set(choice.lineup['player_id']) == set(base.lineup['player_id'])
    cards = [mo.stat(f"{st['expected']:.1f}", label='Expected total'),
             mo.stat(f"{st['floor_p10']:.1f} – {st['ceiling_p90']:.1f}", label='80% range'),
             mo.stat(f"{st['sd']:.1f}", label='sd')]
    if opp is not None:
        cards.append(mo.stat(f"{p_beats(choice.totals, opp.totals):.0%}", label='P(win)',
                             caption=f"opponent projects {opp.totals.mean():.1f}"))
    if has_outcomes:
        cards.append(mo.stat(f"{lineup['actual'].sum():.1f}", label='Actual (replayed week)'))

    if same:
        diff = mo.md('Same lineup as **maximize expected points**.' if st['rule'] != 'maximize_expected' else '')
    else:
        out_ = sorted(set(base.lineup['player']) - set(choice.lineup['player']))
        in_ = sorted(set(choice.lineup['player']) - set(base.lineup['player']))
        diff = mo.callout(mo.md(
            f"Differs from maximize-expected: starts **{', '.join(in_)}** instead of **{', '.join(out_)}**, "
            f"giving up {base.totals.mean() - choice.totals.mean():.2f} expected points."
            + (f" P(win) {p_beats(choice.totals, opp.totals) - p_beats(base.totals, opp.totals):+.3f}." if opp is not None else '')),
            kind='info')

    if st.get('runner_up') is None:
        tie = mo.md('_Only one legal lineup from this roster._')
    elif st.get('near_tie'):
        tie = mo.callout(mo.md(
            '**Near tie** — these calls are within Monte Carlo noise: ' + ', '.join(st['contested_players'])
            + '<br>' + '<br>'.join(f'equally good: {s}' for s in st['near_ties'])), kind='warn')
    else:
        gap = st.get('objective_gap')
        tie = mo.md(f"Clear choice: next best is {st['runner_up_swap']}"
                    + (f" ({gap:.2f} worse, se {st['objective_gap_se']:.2f})" if gap is not None else ''))

    mo.vstack([mo.md('## Lineup'), mo.hstack(cards, justify='start', gap=2), diff, tie,
               mo.ui.table(lineup, selection=None, pagination=False, show_download=False)])
    return


@app.cell(hide_code=True)
def _(alt, mo, np, pl):
    def lineup_compare(totals: dict, title: str = 'How the chosen lineup compares', n_sample: int = 1500):
        """Paired contrasts against the first lineup in ``totals`` ({name: total per draw}).

        Every lineup's totals come from the same simulated weeks (row i = the same week), so
        the useful comparison is the difference week by week, not two overlaid densities:
        lineups that share most players have strongly correlated totals, and their difference
        is far narrower than either total. Against the opponent, P(difference > 0) is P(win).

        Left: a dumbbell per comparison (10th-90th percentile of the difference, ◆ mean, line at 0).
        Right: the density of each difference, the part above 0 shaded."""
        names = list(totals)
        ref = names[0]
        if len(names) < 2:
            return mo.md(f'_Nothing to compare: the **{ref}** lineup is also the maximize-expected one, '
                         'and no opponent is entered._')
        a = np.asarray(totals[ref], dtype=float)
        rows, diffs = [], {}
        for k in names[1:]:
            d = a - np.asarray(totals[k], dtype=float)                 # same draw rows: a paired difference
            lab = f'{ref} − {k}'
            diffs[lab] = d
            p = float((d > 0).mean())
            rows.append({'comparison': lab, 'mean': float(d.mean()), 'p10': float(np.percentile(d, 10)),
                         'p90': float(np.percentile(d, 90)), 'p_ahead': p,
                         'label': f"{'P(win)' if k == 'opponent' else 'P(' + ref + ' ahead)'} {p:.0%}"})
        stats = pl.DataFrame(rows)
        order = list(diffs)
        colors = alt.Scale(domain=order, scheme='tableau10')
        tip = [alt.Tooltip('comparison:N'), alt.Tooltip('mean:Q', format='+.1f', title='mean difference'),
               alt.Tooltip('p10:Q', format='+.1f', title='10th pct'), alt.Tooltip('p90:Q', format='+.1f', title='90th pct'),
               alt.Tooltip('p_ahead:Q', format='.0%', title=f'P({ref} ahead)')]
        base = alt.Chart(stats).encode(y=alt.Y('comparison:N', sort=order, title=None),
                                       color=alt.Color('comparison:N', scale=colors, legend=None), tooltip=tip)
        x = alt.X('p10:Q', title=f'Points: {ref} minus the other lineup (10th–90th pct, ◆ mean)')
        zero = alt.Chart(pl.DataFrame({'z': [0.0]})).mark_rule(color='#666', strokeWidth=1.5).encode(x='z:Q')
        dumbbell = (zero
                    + base.mark_rule(strokeWidth=4, opacity=0.6).encode(x=x, x2='p90:Q')
                    + base.mark_circle(size=110, opacity=1).encode(x='p10:Q')
                    + base.mark_circle(size=110, opacity=1).encode(x='p90:Q')
                    + base.mark_point(shape='diamond', size=170, filled=True, opacity=1)
                          .encode(x='mean:Q', color=alt.value('#222'))
                    + base.mark_text(dy=-15, fontSize=11).encode(x='mean:Q', text='label:N', color=alt.value('#222'))
                    ).properties(width=340, height=52 * len(order), title=title)

        # the same simulated weeks for every comparison, so the pairing survives the sampling
        idx = np.random.default_rng(0).choice(len(a), min(n_sample, len(a)), replace=False)
        dist = pl.concat([pl.DataFrame({'diff': d[idx], 'comparison': k}) for k, d in diffs.items()])
        legend = alt.Legend(orient='top', title=None)
        dens = (alt.Chart().transform_density('diff', groupby=['comparison'], as_=['diff', 'density']))
        enc = dict(x=alt.X('diff:Q', title=f'{ref} minus the other lineup (points)'),
                   y=alt.Y('density:Q', title=None, axis=None, stack=None),
                   color=alt.Color('comparison:N', scale=colors, legend=legend))
        full = dens.mark_area(opacity=0.15).encode(**enc)
        ahead = dens.transform_filter('datum.diff >= 0').mark_area(opacity=0.45).encode(**enc)   # shaded: ref ahead
        means = (alt.Chart().mark_rule(strokeDash=[4, 3], strokeWidth=2)
                 .encode(x='mean(diff):Q', color=alt.Color('comparison:N', scale=colors, legend=legend)))
        zline = alt.Chart().mark_rule(color='#666', strokeWidth=1.5).encode(x=alt.datum(0))
        density = (alt.layer(full, ahead, means, zline, data=dist)
                   .properties(width=340, height=180, title=f'Difference per simulated week (shaded: {ref} ahead)'))
        return mo.hstack([dumbbell, density], justify='start', align='start', gap=2)
    return (lineup_compare,)


@app.cell(hide_code=True)
def _(base, choice, lineup_compare, opp):
    _totals = {'chosen': choice.totals}
    if set(choice.lineup['player_id']) != set(base.lineup['player_id']):
        _totals['maximize expected'] = base.totals
    if opp is not None:
        _totals['opponent'] = opp.totals
    lineup_compare(_totals, title='Chosen lineup vs the alternatives')
    return


@app.cell(hide_code=True)
def _(choice, frame, mo, pool):
    SLOTS = (('QB', ('QB',)), ('RB1', ('RB',)), ('RB2', ('RB',)), ('WR1', ('WR',)), ('WR2', ('WR',)),
             ('TE', ('TE',)), ('FLEX', ('RB', 'WR', 'TE')))
    _info = {r['player_id']: r for r in frame.select('player_id', 'player', 'position', 'team').iter_rows(named=True)}
    _lab = lambda pid: f"{_info[pid]['player']} · {_info[pid]['position']} · {_info[pid]['team']}"  # noqa: E731
    _rec = dict(zip(choice.lineup['slot'].to_list(), choice.lineup['player_id'].to_list()))
    slot_pick = mo.ui.dictionary({
        slot: mo.ui.dropdown(options={_lab(p): p for p in pool if _info[p]['position'] in allowed},
                             value=_lab(_rec[slot]) if slot in _rec else None, label=slot)
        for slot, allowed in SLOTS})
    mo.vstack([mo.md('## Set your lineup'),
               mo.md('Preset to the recommended lineup above; change any slot to evaluate your own call.'),
               mo.hstack(list(slot_pick.values()), wrap=True, justify='start')])
    return (slot_pick,)


@app.cell(hide_code=True)
def _(
    alt,
    choice,
    col_of,
    de,
    draws,
    frame,
    has_outcomes,
    lineup_compare,
    mo,
    np,
    opp,
    pl,
    slot_pick,
    summary,
):
    picked = {slot: pid for slot, pid in slot_pick.value.items()}
    _empty = [s_ for s_, p in picked.items() if p is None]
    _dupes = sorted({p for p in picked.values() if p is not None and list(picked.values()).count(p) > 1})
    mo.stop(bool(_empty), mo.callout(mo.md(f"Fill every slot (empty: {', '.join(_empty)})."), kind='info'))
    mo.stop(bool(_dupes), mo.callout(mo.md('A player is in two slots: '
                                          + ', '.join(frame.filter(pl.col('player_id').is_in(_dupes))['player'])),
                                    kind='danger'))
    mine = draws[:, [col_of[p] for p in picked.values()]].sum(1)
    _rec = choice.totals
    _is_rec = set(picked.values()) == set(choice.lineup['player_id'])

    # paired difference with the recommendation, and its Monte Carlo SE (bootstrap over draws)
    _diff = mine - _rec
    _rng = np.random.default_rng(0)
    _boot = np.array([_diff[_rng.integers(0, len(_diff), len(_diff))].mean() for _ in range(200)])
    _se = float(_boot.std())
    _cards = [mo.stat(f'{mine.mean():.1f}', label='Expected total',
                     caption=None if _is_rec else f'{_diff.mean():+.2f} vs recommended (se {_se:.2f})'),
             mo.stat(f'{np.percentile(mine, 10):.1f} – {np.percentile(mine, 90):.1f}', label='80% range')]
    if opp is not None:
        _pw = de.p_beats(mine, opp.totals)
        _cards.append(mo.stat(f'{_pw:.0%}', label='P(win)',
                             caption=None if _is_rec else f'{_pw - de.p_beats(_rec, opp.totals):+.3f} vs recommended'))
    if not _is_rec:
        _cards.append(mo.stat(f'{(mine > _rec).mean():.0%}', label='Beats recommended', caption='share of draws'))
    if has_outcomes:
        _actual = float(summary.filter(pl.col('player_id').is_in(list(picked.values())))['actual'].sum())
        _rec_actual = float(summary.filter(pl.col('player_id').is_in(choice.lineup['player_id'].to_list()))['actual'].sum())
        _cards.append(mo.stat(f'{_actual:.1f}', label='Actual (replayed week)',
                             caption=None if _is_rec else f'{_actual - _rec_actual:+.1f} vs recommended'))

    if _is_rec:
        _verdict = mo.md('This is the recommended lineup.')
    elif abs(_diff.mean()) < 2 * _se:
        _verdict = mo.callout(mo.md(f'**Within noise** of the recommendation ({_diff.mean():+.2f} points, '
                                   f'se {_se:.2f}): a judgment call, not a mistake.'), kind='info')
    else:
        _verdict = mo.callout(mo.md(f'Gives up **{-_diff.mean():.2f} expected points** vs the recommendation '
                                   f'(se {_se:.2f}).' if _diff.mean() < 0 else
                                   f'Projects **{_diff.mean():+.2f} points** better than the recommended lineup '
                                   f'(se {_se:.2f}) under this rule\'s scoring.'), kind='warn')

    _totals = {'yours': mine}
    if not _is_rec:
        _totals['recommended'] = _rec
    if opp is not None:
        _totals['opponent'] = opp.totals
    _compare = lineup_compare(_totals, title='Your lineup vs the alternatives')
    save_lineup = mo.ui.run_button(label='Save this lineup')
    mo.vstack([mo.hstack(_cards, justify='start', gap=2), _verdict, _compare, save_lineup])
    return mine, picked, save_lineup


@app.cell(hide_code=True)
def _(LINEUP_DIR, fc, frame, json, mine, mo, opp, picked, pl, save_lineup):
    mo.stop(not save_lineup.value)
    LINEUP_DIR.mkdir(parents=True, exist_ok=True)
    _out = LINEUP_DIR / f'{fc.season}_wk{int(fc.week):02d}.json'
    _rows = frame.filter(pl.col('player_id').is_in(list(picked.values())))
    _names = dict(zip(_rows['player_id'], _rows['player']))
    _out.write_text(json.dumps({
        'season': fc.season, 'week': fc.week, 'forecast_model': fc.meta.get('forecast_model'),
        'lineup': [{'slot': s_, 'player_id': p, 'player': _names[p]} for s_, p in picked.items()],
        'expected': float(mine.mean()), '10th Percentile Outcome': float(pl.Series(mine).quantile(0.1)),
        '90th Percentile Outcome': float(pl.Series(mine).quantile(0.9)),
        'p_win': None if opp is None else float((mine > opp.totals[:len(mine)]).mean()),
    }, indent=1))
    mo.callout(mo.md(f'Saved to `{_out}`.'), kind='success')
    return


@app.cell(hide_code=True)
def _(mo, summary):
    player_pick = mo.ui.dropdown(
        options={f"{r['player']} · {r['position']} · {r['team']}": r['player_id'] for r in summary.iter_rows(named=True)},
        label='Player', searchable=True)
    mo.vstack([mo.md('## Player detail'), player_pick])
    return (player_pick,)


@app.cell(hide_code=True)
def _(alt, col_of, draws, frame, mo, np, pl, player_pick, summary):
    mo.stop(player_pick.value is None, mo.md('_Pick a player to see his distribution and how he moves with his teammates._'))
    j = col_of[player_pick.value]
    me = summary.filter(pl.col('player_id') == player_pick.value).row(0, named=True)
    d = draws[:, j]
    hist = (alt.Chart(pl.DataFrame({'points': d[np.random.default_rng(1).choice(len(d), min(4000, len(d)), replace=False)]}))
            .mark_bar().encode(x=alt.X('points:Q', bin=alt.Bin(maxbins=40), title='Fantasy points'),
                               y=alt.Y('count()', title=None))
            .properties(height=200, title=f"{me['player']}: proj {me['proj']:.1f}, 80% range "
                                          f"{me['floor_p10']:.1f}–{me['ceiling_p90']:.1f}, P(dud) {me['p_dud']:.0%}"))
    mates = frame.with_row_index('_c').filter((pl.col('team') == me['team']) & (pl.col('player_id') != player_pick.value))
    corr = pl.DataFrame({
        'teammate': [f"{n} ({p})" for n, p in zip(mates['player'], mates['position'])],
        'corr': [float(np.corrcoef(d, draws[:, c])[0, 1]) for c in mates['_c'].to_list()]}).sort('corr', descending=True)
    bars = (alt.Chart(corr).mark_bar().encode(
                x=alt.X('corr:Q', title='Correlation of draws', scale=alt.Scale(domain=[-0.3, 0.5])),
                y=alt.Y('teammate:N', sort='-x', title=None),
                color=alt.condition('datum.corr > 0', alt.value('#4c78a8'), alt.value('#e45756')))
            .properties(height=max(80, 18 * corr.height), title='Moves with teammates (QB link)'))
    mo.vstack([mo.ui.altair_chart(hist), mo.ui.altair_chart(bars)])
    return


if __name__ == "__main__":
    app.run()
