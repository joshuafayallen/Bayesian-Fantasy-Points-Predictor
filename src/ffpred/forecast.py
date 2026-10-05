"""Week-ahead forecasts from both production models, in the shape the
decision engine consumes: a frame (one row per player) and a draws matrix
(n_draws x n_players) whose columns follow the frame's rows.

The skill model (het_gp_vegas) and the QB model (het_gp_qb_vegas) are fit
separately, so the joint draws pair a QB draw with a skill draw arbitrarily:
there is no modelled correlation between a QB and his receivers. Within a
model, columns share posterior draws (parameter uncertainty is correlated),
which is what the paired-draw decision rules rely on.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from . import data as D
# pyrefly: ignore [missing-import]
from .config import MIN_TRAIN_ROWS, MODELS, TARGET_COL, WINDOW_SEASONS, SamplerSettings
from .model import FittedModel, fit

log = logging.getLogger(__name__)

SUMMARY_COLS = ('player_id', 'player', 'position', 'team', 'opp_team')


@dataclass
class WeekForecast:
    season: int
    week: float
    frame: pl.DataFrame               # scored rows, aligned with draws' columns
    draws: np.ndarray                 # (n_draws, n_rows), fantasy points
    skipped: pl.DataFrame             # rows the models couldn't score, with a reason
    meta: dict = field(default_factory=dict)

    def summary(self) -> pl.DataFrame:
        return player_summary(self.draws, self.frame)

    # ---------------------------------------------------------------- io
    def save(self, out_dir: Path | str) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.frame.write_parquet(out / 'frame.parquet')
        np.save(out / 'draws.npy', self.draws.astype('float32'))
        self.skipped.write_parquet(out / 'skipped.parquet')
        self.summary().write_csv(out / 'summary.csv')
        (out / 'meta.json').write_text(json.dumps(
            {'season': self.season, 'week': self.week, **self.meta}, indent=2, default=str))
        return out

    @classmethod
    def load(cls, out_dir: Path | str) -> 'WeekForecast':
        out = Path(out_dir)
        meta = json.loads((out / 'meta.json').read_text())
        frame = pl.read_parquet(out / 'frame.parquet')
        draws = np.load(out / 'draws.npy').astype('float64')
        if draws.shape[1] != frame.height:
            raise ValueError(f'{out}: draws has {draws.shape[1]} columns, frame has {frame.height} rows')
        skipped = pl.read_parquet(out / 'skipped.parquet') if (out / 'skipped.parquet').exists() else frame.head(0)
        return cls(season=int(meta.pop('season')), week=float(meta.pop('week')), frame=frame, draws=draws,
                   skipped=skipped, meta=meta)


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
                  min_train_rows: int = MIN_TRAIN_ROWS) -> WeekForecast:
    """Fit each model on the window before (season, week) and forecast ``frame``.

    history:   processed player-weeks (``data.load_processed``). Only rows
               strictly before the week are used for fitting.
    team_form: ``data.load_team_form``.
    frame:     rows to forecast -- ``upcoming.build_upcoming_frame`` for a
               real week; default: the week's played rows from ``history``
               (a replay, for validation).
    """
    week = float(week)
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
        skipped.append(rows.filter(~pl.Series(ok)).with_columns(
            pl.lit(name).alias('model'), pl.lit('player not in training window / time out of range').alias('reason')))
        draws.append(fm.predict(keep))
        frames.append(keep.with_columns(pl.lit(name).alias('model')))
        fits[name] = {'n_train': fm.n_train, 'n_players': len(fm.prep.players), 'n_scored': keep.height,
                          'n_skipped': int((~ok).sum()), 'fit_time': round(fm.fit_time, 1),
                          'divergences': fm.divergences, 'max_rhat': fm.max_rhat, 'min_ess_bulk': fm.min_ess_bulk}
        log.info('%s: scored %d rows, skipped %d', name, keep.height, (~ok).sum())

    n = min(d.shape[0] for d in draws)
    if any(d.shape[0] != n for d in draws):
        log.warning('models returned different draw counts %s; truncating to %d', [d.shape[0] for d in draws], n)
    all_draws = np.hstack([d[:n] for d in draws])
    all_frame = pl.concat(frames, how='diagonal_relaxed')
    not_covered = frame.filter(~pl.col('position').is_in([p for m in models for p in MODELS[m].positions]))
    skipped_all = pl.concat(
        [*skipped, not_covered.with_columns(pl.lit(None, pl.String).alias('model'),
                                            pl.lit('position not modelled').alias('reason'))],
        how='diagonal_relaxed')
    meta = {'source': source, 'window_seasons': window_seasons, 'settings': asdict(settings), 'fits': fits,
                'n_draws': n, 'has_outcomes': bool(TARGET_COL in all_frame.columns
                                             and all_frame[TARGET_COL].null_count() < all_frame.height)}
    return WeekForecast(season=season, week=week, frame=all_frame, draws=all_draws,
                        skipped=skipped_all.select([c for c in (*SUMMARY_COLS, 'model', 'reason')
                                                    if c in skipped_all.columns]),
                        meta=meta)
