"""Week-ahead forecasts. A WeekForecast holds one xarray DataTree, in the layout
ArviZ / tidydraws expect:

  /predictions                 points (chain, draw, player_id), fantasy points, with
                               player / position / team / opp_team as coordinates
  /predictions_constant_data   each player's model inputs that week, by player_id
  /skipped                     players the models couldn't score, with a reason
  attrs                        season, week, meta (settings, fit diagnostics) as JSON

``fc.dt`` is the object to work with (``fc.tidy()`` for a tidydraws frame).
``fc.draws`` (n_draws x n_players) and ``fc.frame`` (one row per player, in
the same order) are views of it for the decision engine. Saved as one NetCDF
file, forecast.nc, with the points stored in tenths of a point.

Two forecast models (``model=``):

'qb_linked' (default)   het_gp_qb_vegas for QBs + the hurdle-gamma skill model
                        linked to the starting QB (qblink.py). Row i of the
                        draws gives a QB and his receivers the same QB surprise,
                        so QB-receiver stacks are correlated.
'independent'           het_gp_vegas + het_gp_qb_vegas fit separately: a QB
                        draw is paired with a skill draw arbitrarily, with no
                        QB-receiver correlation.

Within a model, columns share posterior draws (parameter uncertainty is
correlated), which is what the paired-draw decision rules rely on.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from functools import cached_property
from pathlib import Path

import numpy as np
import polars as pl
import tidydraws as td
import xarray as xr

from . import data as D
# pyrefly: ignore [missing-import]
from .config import (FORECAST_MODELS, HET_GP_QB_VEGAS, HURDLE_SKILL, MIN_TRAIN_ROWS, MODELS, QB_LINKED,
                     TARGET_COL, WINDOW_SEASONS, SamplerSettings)
from .model import FittedModel, fit
from .qblink import NO_QB, fit_qb_linked

log = logging.getLogger(__name__)

SUMMARY_COLS = ('player_id', 'player', 'position', 'team', 'opp_team')
LABELS = ('player', 'position', 'team', 'opp_team')     # carried as coordinates on the points
FILE = 'forecast.nc'
_NULL = '__null__'                                     # stands in for a null string in the file (netCDF drops '\x00')
_DUP = '__dup'                                         # suffix that makes a repeated player_id unique
# points on disk: whole tenths of a point (scores come in 0.1 steps), at most 0.05 pts off
_POINTS_ENCODING = {'dtype': 'int16', 'scale_factor': 0.1, '_FillValue': np.iinfo(np.int16).min,
                    'zlib': True, 'complevel': 4}


class WeekForecast:
    """One week's forecast. ``dt`` (an xarray DataTree) is the forecast; ``draws``,
    ``frame`` and ``skipped`` are views of it. Build from arrays with
    ``WeekForecast(season, week, frame=..., draws=..., skipped=..., meta=...)``
    or from a tree with ``WeekForecast(season, week, dt=...)``."""

    def __init__(self, season: int, week: float, frame: pl.DataFrame | None = None,
                 draws: np.ndarray | None = None, skipped: pl.DataFrame | None = None,
                 meta: dict | None = None, dt: xr.DataTree | None = None):
        self.season, self.week = int(season), float(week)
        self.meta = dict(meta or {})
        if dt is None:
            if frame is None or draws is None:
                raise ValueError('give frame and draws, or dt')
            dt = forecast_tree(np.asarray(draws), frame, _chains(self.meta, np.shape(draws)[0]),
                               skipped=skipped)
        self.dt = dt
        if frame is not None:                         # keep the caller's frame exactly
            self.__dict__['frame'] = frame

    # ----------------------------------------------------------- views
    @property
    def points(self) -> xr.DataArray:
        """(chain, draw, player_id) fantasy points, with the player labels as coordinates."""
        return self.dt['predictions']['points']

    @cached_property
    def draws(self) -> np.ndarray:
        """(n_draws, n_players) matrix for the decision engine, chains stacked in order."""
        p = self.points
        return np.asarray(p.values, dtype='float64').reshape(-1, p.sizes['player_id'])

    @cached_property
    def frame(self) -> pl.DataFrame:
        """One row per player, in the order of ``draws``' columns."""
        return _frame_from_tree(self.dt)

    @cached_property
    def skipped(self) -> pl.DataFrame:
        return _table_from_ds(self.dt['skipped'].to_dataset()) if 'skipped' in self.dt.children else pl.DataFrame()

    def summary(self) -> pl.DataFrame:
        return player_summary(self.draws, self.frame)

    def tidy(self) -> pl.DataFrame:
        """tidydraws long frame: one row per player per draw, with his labels and model inputs.
        Player columns come from ``frame`` (proper nulls and dtypes), not the stored copies."""
        p = self.points
        bare = xr.DataTree.from_dict({'predictions': p.drop_vars([c for c in LABELS if c in p.coords]).to_dataset()})
        players = self.frame.with_columns(pl.Series('player_id', p.player_id.values.astype(str)))  # tree's ids
        return td.prediction_draws(bare, players, 'points', join_on='player_id')

    def __repr__(self) -> str:
        return (f'WeekForecast({self.season} wk {self.week:g}: {self.points.sizes["player_id"]} players, '
                f'{self.points.sizes["chain"]} chains x {self.points.sizes["draw"]} draws)')

    # ---------------------------------------------------------------- io
    def save(self, out_dir: Path | str) -> Path:
        """forecast.nc (the tree), plus summary.csv to open in a spreadsheet and meta.json,
        written last: its presence marks a complete forecast."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        tree = self.dt.copy()
        tree.attrs = {**tree.attrs, 'season': self.season, 'week': self.week,
                      'meta': json.dumps(self.meta, default=str)}
        enc = {'/predictions': {'points': _POINTS_ENCODING}}
        for path in ('/predictions_constant_data', '/skipped'):
            if path.strip('/') in tree.children:
                enc[path] = {v: {'zlib': True, 'complevel': 4} for v in tree[path].data_vars
                             if tree[path][v].dtype.kind in 'fiub'}
        tree.to_netcdf(out / FILE, encoding=enc, engine='netcdf4')
        self.summary().write_csv(out / 'summary.csv')
        (out / 'meta.json').write_text(json.dumps(
            {'season': self.season, 'week': self.week, **self.meta}, indent=2, default=str))
        return out

    @classmethod
    def load(cls, out_dir: Path | str) -> 'WeekForecast':
        out = Path(out_dir)
        if (out / FILE).exists():
            with xr.open_datatree(out / FILE, engine='netcdf4') as f:
                dt = f.load()
            meta = json.loads(dt.attrs.get('meta', '{}'))
            return cls(season=dt.attrs['season'], week=dt.attrs['week'], meta=meta, dt=dt)
        # forecasts saved before forecast.nc: frame.parquet + draws.npy (+ skipped.parquet)
        meta = json.loads((out / 'meta.json').read_text())
        frame = pl.read_parquet(out / 'frame.parquet')
        draws = np.load(out / 'draws.npy').astype('float64')
        if draws.shape[1] != frame.height:
            raise ValueError(f'{out}: draws has {draws.shape[1]} columns, frame has {frame.height} rows')
        skipped = pl.read_parquet(out / 'skipped.parquet') if (out / 'skipped.parquet').exists() else frame.head(0)
        return cls(season=int(meta.pop('season')), week=float(meta.pop('week')), frame=frame, draws=draws,
                   skipped=skipped, meta=meta)


def _chains(meta: dict, n_draws: int) -> int:
    """Draws are stacked chain by chain; NUTS fits split back into their chains."""
    s = meta.get('settings', {})
    chains = int(s.get('chains', 1)) if s.get('method', 'nuts') == 'nuts' else 1
    return chains if chains > 0 and n_draws % chains == 0 else 1


def _ds_from_table(df: pl.DataFrame, dim: str, index: str | None = None) -> xr.Dataset:
    """A polars table as a Dataset along ``dim``; column order and dtypes kept in attrs
    so _table_from_ds can rebuild it exactly (null strings and integers survive)."""
    data = {}
    for c in df.columns:
        if c == index:
            continue
        col = df[c]
        if col.dtype in (pl.String, pl.Categorical, pl.Enum):
            vals = col.cast(pl.String).fill_null(_NULL).to_numpy().astype(object)
        elif col.dtype == pl.Boolean:
            vals = col.cast(pl.Int8).fill_null(-1).to_numpy()
        elif col.dtype.is_integer() and col.null_count():
            vals = col.cast(pl.Float64).to_numpy()
        else:
            vals = col.to_numpy()
        data[c] = (dim, vals)
    coords = {dim: df[index].to_numpy().astype(object)} if index else {dim: np.arange(df.height)}
    return xr.Dataset(data, coords=coords, attrs={'columns': json.dumps(df.columns),
                                                  'dtypes': json.dumps({c: str(t) for c, t in df.schema.items()})})


def _table_from_ds(ds: xr.Dataset, extra: dict | None = None) -> pl.DataFrame:
    dim = next(iter(ds.dims), None)
    cols = json.loads(ds.attrs.get('columns', '[]'))
    if dim is None or not cols:
        return pl.DataFrame()
    dtypes = json.loads(ds.attrs.get('dtypes', '{}'))
    series = {dim: pl.Series(dim, ds[dim].values.astype(object) if ds[dim].dtype == object else ds[dim].values)}
    series.update(extra or {})
    for c in cols:
        if c in series:
            continue
        v = ds[c].values
        t = dtypes.get(c, '')
        if t.startswith(('String', 'Categorical', 'Enum')):
            s_ = pl.Series(c, v.astype(str)).replace(_NULL, None)
        elif t == 'Boolean':
            s_ = pl.Series(c, v).replace(-1, None).cast(pl.Boolean)
        else:
            s_ = pl.Series(c, v)
            if s_.dtype.is_float():
                s_ = s_.fill_nan(None)
        series[c] = s_
    out = pl.DataFrame([series[c] for c in cols if c in series])
    return out.cast({c: _dtype(t) for c, t in dtypes.items() if c in out.columns and _dtype(t) is not None},
                    strict=False)


def _dtype(name: str):
    """'Int32' -> pl.Int32 for the simple dtypes; None for anything else (left as read)."""
    return getattr(pl, name) if name.isidentifier() and hasattr(pl, name) else None


def forecast_tree(draws: np.ndarray, frame: pl.DataFrame, chains: int = 1,
                  skipped: pl.DataFrame | None = None) -> xr.DataTree:
    """The forecast as a DataTree: points (chain, draw, player_id) with the player labels as
    coordinates, the rest of ``frame`` as predictions_constant_data, and ``skipped``."""
    ids = frame['player_id'].to_list()
    dedup = len(set(ids)) != len(ids)
    if dedup:                                         # a dimension index has to be unique; undone on load
        frame = frame.with_columns(pl.Series('player_id', [f'{p}{_DUP}{i}' for i, p in enumerate(ids)]))
    n_draws, n_players = draws.shape
    labels = [c for c in LABELS if c in frame.columns]
    points = xr.DataArray(
        draws.reshape(chains, n_draws // chains, n_players), dims=('chain', 'draw', 'player_id'),
        coords={'chain': np.arange(chains), 'draw': np.arange(n_draws // chains),
                'player_id': frame['player_id'].to_numpy().astype(object),
                **{c: ('player_id', frame[c].cast(pl.String).fill_null('').to_numpy().astype(object))
                   for c in labels}},
        name='points', attrs={'units': 'fantasy points'})
    constant = _ds_from_table(frame, 'player_id', index='player_id')
    constant = constant.drop_vars([c for c in labels if c in constant.data_vars])
    constant.attrs['labels'] = json.dumps(labels)
    constant.attrs['dedup'] = int(dedup)
    groups = {'predictions': points.to_dataset(), 'predictions_constant_data': constant}
    if skipped is not None and skipped.height:
        groups['skipped'] = _ds_from_table(skipped, 'row')
    return xr.DataTree.from_dict(groups)


def _frame_from_tree(dt: xr.DataTree) -> pl.DataFrame:
    """Rebuild the player frame: labels from the points' coordinates, the rest from the
    constant-data group, in the original column order and dtypes."""
    points, const = dt['predictions']['points'], dt['predictions_constant_data'].to_dataset()
    labels = json.loads(const.attrs.get('labels', '[]'))
    dtypes = json.loads(const.attrs.get('dtypes', '{}'))

    def _label(c):
        s_ = pl.Series(c, points[c].values.astype(str))
        return s_.replace('', None) if dtypes.get(c, 'String').startswith('String') else s_

    out = _table_from_ds(const, extra={c: _label(c) for c in labels})
    if int(const.attrs.get('dedup', 0)):          # strip the suffix that made repeated ids unique
        out = out.with_columns(pl.col('player_id').str.replace(f'{_DUP}\\d+$', ''))
    return out


def player_summary(draws: np.ndarray, frame: pl.DataFrame) -> pl.DataFrame:
    """Per-player forecast table, sorted by expected points. floor / ceiling
    are the 10th / 90th percentiles of the predictive draws."""
    cols = [c for c in SUMMARY_COLS if c in frame.columns]
    return (frame.select(cols).with_columns(
        pl.Series('expected', draws.mean(0)),
        pl.Series('median', np.median(draws, 0)),
        pl.Series('sd', draws.std(0)),
        pl.Series('floor_p10', np.percentile(draws, 10, axis=0)),
        pl.Series('ceiling_p90', np.percentile(draws, 90, axis=0)),
    ).sort('expected', descending=True))


def prepare_history(processed: pl.DataFrame, team_form: pl.DataFrame) -> pl.DataFrame:
    """Processed data with the team-form columns joined (as-of, leak-free)."""
    return D.join_team_form(processed, team_form)


def forecast_week(history: pl.DataFrame, team_form: pl.DataFrame, season: int, week: float,
                  frame: pl.DataFrame | None = None, settings: SamplerSettings = SamplerSettings(),
                  window_seasons: int = WINDOW_SEASONS, models=tuple(MODELS),
                  min_train_rows: int = MIN_TRAIN_ROWS, model: str = QB_LINKED) -> WeekForecast:
    """Fit on the window before (season, week) and forecast ``frame``.

    history:   processed player-weeks (``data.load_processed``). Only rows
               strictly before the week are used for fitting.
    team_form: ``data.load_team_form``.
    frame:     rows to forecast -- ``upcoming.build_upcoming_frame`` for a
               real week; default: the week's played rows from ``history``
               (a replay, for validation).
    model:     'qb_linked' (production) or 'independent'. ``models`` picks the
               het_gp configs for 'independent' and is ignored otherwise.
    """
    if model not in FORECAST_MODELS:
        raise ValueError(f'model must be one of {FORECAST_MODELS}, got {model!r}')
    week = float(week)
    hist, frame, source = _prepare(history, team_form, season, week, frame)
    if model == QB_LINKED:
        frames, draws, skipped, fits, covered = _forecast_qb_linked(hist, frame, season, week, settings,
                                                                    window_seasons, min_train_rows)
    else:
        frames, draws, skipped, fits, covered = _forecast_independent(hist, frame, season, week, settings,
                                                                      window_seasons, models, min_train_rows)

    n = min(d.shape[0] for d in draws)
    if any(d.shape[0] != n for d in draws):
        log.warning('models returned different draw counts %s; truncating to %d', [d.shape[0] for d in draws], n)
    all_draws = np.hstack([d[:n] for d in draws])
    all_frame = pl.concat(frames, how='diagonal_relaxed')
    not_covered = frame.filter(~pl.col('position').is_in(covered))
    skipped_all = pl.concat(
        [*skipped, not_covered.with_columns(pl.lit(None, pl.String).alias('model'),
                                            pl.lit('position not modelled').alias('reason'))],
        how='diagonal_relaxed')
    meta = {'source': source, 'forecast_model': model, 'window_seasons': window_seasons,
            'settings': asdict(settings), 'fits': fits, 'n_draws': n,
            'has_outcomes': bool(TARGET_COL in all_frame.columns
                                 and all_frame[TARGET_COL].null_count() < all_frame.height)}
    return WeekForecast(season=season, week=week, frame=all_frame, draws=all_draws,
                        skipped=skipped_all.select([c for c in (*SUMMARY_COLS, 'model', 'reason')
                                                    if c in skipped_all.columns]),
                        meta=meta)


def _prepare(history, team_form, season, week, frame):
    """Training history (before the week, team form joined, one position per
    player) and the rows to forecast, prepared identically."""
    hist = D.join_team_form(history.filter(D.before(season, week)), team_form)
    if frame is None:
        frame = D.week_rows(history, season, week)
        source = 'played'
    else:
        source = 'given'
    if frame.height == 0:
        raise ValueError(f'nothing to forecast for {season} week {week:g}')
    if 'team_spread' not in frame.columns:
        frame = D.add_vegas_features(frame)
    frame = D.join_team_form(frame, team_form)
    # one position per player (his most frequent listing up to now), applied
    # identically to training rows and forecast rows
    hist = D.with_primary_position(hist)
    frame = D.with_primary_position(frame, reference=hist)
    return hist, frame, source


def _fit_info(fm, keep, ok) -> dict:
    return {'n_train': fm.n_train, 'n_players': len(fm.prep.players), 'n_scored': keep.height,
            'n_skipped': int((~ok).sum()), 'fit_time': round(fm.fit_time, 1),
            'divergences': fm.divergences, 'max_rhat': fm.max_rhat, 'min_ess_bulk': fm.min_ess_bulk}


def _skip(rows, ok, name):
    return rows.filter(~pl.Series(ok)).with_columns(
        pl.lit(name).alias('model'), pl.lit('player not in training window / time out of range').alias('reason'))


def _forecast_independent(hist, frame, season, week, settings, window_seasons, models, min_train_rows):
    frames, draws, skipped, fits = [], [], [], {}
    for name in models:
        cfg = MODELS[name]
        train = D.training_window(hist, season, week, window_seasons)
        train = D.model_frame(train, cfg)
        rows = D.model_frame(frame, cfg)
        if train.height < min_train_rows:
            raise ValueError(f'{name}: only {train.height} training rows before {season} week {week:g}')
        fm: FittedModel = fit(train, cfg, settings)
        ok = fm.prep.eligible(rows)
        keep = rows.filter(pl.Series(ok))
        skipped.append(_skip(rows, ok, name))
        draws.append(fm.predict(keep))
        frames.append(keep.with_columns(pl.lit(name).alias('model')))
        fits[name] = _fit_info(fm, keep, ok)
        log.info('%s: scored %d rows, skipped %d', name, keep.height, (~ok).sum())
    return frames, draws, skipped, fits, [p for m in models for p in MODELS[m].positions]


def _forecast_qb_linked(hist, frame, season, week, settings, window_seasons, min_train_rows):
    """Two-stage fit (QB, then skill on the QB residuals) and correlated draws."""
    hist = D.add_trail_pts(hist)                                   # leak-free: previous games only
    frame = D.add_trail_pts_for_week(frame, hist, season, week)
    window = D.training_window(hist, season, week, window_seasons)
    train_s, train_q = D.model_frame(window, HURDLE_SKILL), D.model_frame(window, HET_GP_QB_VEGAS)
    for cfg, tr in ((HURDLE_SKILL, train_s), (HET_GP_QB_VEGAS, train_q)):
        if tr.height < min_train_rows:
            raise ValueError(f'{cfg.name}: only {tr.height} training rows before {season} week {week:g}')
    fit_ = fit_qb_linked(train_s, train_q, settings)

    rows_s, rows_q = D.model_frame(frame, HURDLE_SKILL), D.model_frame(frame, HET_GP_QB_VEGAS)
    ok_s, ok_q = fit_.skill.prep.eligible(rows_s), fit_.qb.prep.eligible(rows_q)
    keep_s, keep_q = rows_s.filter(pl.Series(ok_s)), rows_q.filter(pl.Series(ok_q))
    S_s, S_q, qrow = fit_.predict(keep_s, keep_q)

    linked = qrow != NO_QB
    fits = {HURDLE_SKILL.name: {**_fit_info(fit_.skill, keep_s, ok_s), 'lam_rec': fit_.skill.loadings(),
                                'share_linked_to_qb': float(linked.mean()) if linked.size else 0.0},
            HET_GP_QB_VEGAS.name: {**_fit_info(fit_.qb, keep_q, ok_q),
                                   'train_residual_sd': float(fit_.r_train.std())}}
    if linked.size and linked.mean() < 0.8:
        log.warning('only %.0f%% of skill rows matched a starting QB', 100 * linked.mean())
    log.info('qb_linked: scored %d skill + %d QB rows, skipped %d + %d; lam_rec %s', keep_s.height, keep_q.height,
             (~ok_s).sum(), (~ok_q).sum(), {k: round(v, 3) for k, v in fit_.skill.loadings().items()})
    frames = [keep_s.with_columns(pl.lit(HURDLE_SKILL.name).alias('model')),
              keep_q.with_columns(pl.lit(HET_GP_QB_VEGAS.name).alias('model'))]
    skipped = [_skip(rows_s, ok_s, HURDLE_SKILL.name), _skip(rows_q, ok_q, HET_GP_QB_VEGAS.name)]
    return frames, [S_s, S_q], skipped, fits, [*HURDLE_SKILL.positions, *HET_GP_QB_VEGAS.positions]
