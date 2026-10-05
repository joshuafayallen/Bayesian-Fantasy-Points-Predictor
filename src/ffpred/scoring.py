"""Forecast scoring: CRPS, RMSE/MAE, rank correlation, interval coverage,
and tail calibration (by projection tier) for the decision rules."""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from .config import TARGET_COL


def crps(S: np.ndarray, y: np.ndarray, max_pairs: int = 200, seed: int = 0) -> float:
    """Sample CRPS, E|X - y| - 0.5 E|X - X'|, averaged over rows. The second
    term uses ``max_pairs`` random draw pairs (same estimator as src/)."""
    t1 = np.abs(S - y[None, :]).mean(0)
    i = np.random.default_rng(seed).choice(S.shape[0], size=(2, min(max_pairs, S.shape[0])))
    t2 = np.abs(S[i[0]] - S[i[1]]).mean(0)
    return float((t1 - 0.5 * t2).mean())


def score(y: np.ndarray, S: np.ndarray, baseline: np.ndarray | None = None,
          levels=(0.5, 0.8, 0.9)) -> dict:
    pred = S.mean(0)
    out = dict(n=len(y), rmse=float(np.sqrt(((pred - y) ** 2).mean())),
               mae=float(np.abs(pred - y).mean()), crps=crps(S, y),
               spearman=float(spearmanr(pred, y).statistic) if len(y) > 2 else float('nan'))
    for L in levels:
        lo, hi = np.percentile(S, [100 * (1 - L) / 2, 100 * (1 + L) / 2], axis=0)
        out[f'cov{int(L * 100)}'] = float(((y >= lo) & (y <= hi)).mean())
    out.update(tail_metrics(y, S))
    if baseline is not None:
        out['rmse_base'] = float(np.sqrt(((baseline - y) ** 2).mean()))
    return out


# Projection tiers (expected points) for calibration checks. Fixed cut points
# so tiers mean the same thing across folds and positions.
TIER_EDGES = (6.0, 12.0, 18.0)
TIER_LABELS = ('<6', '6-12', '12-18', '18+')


def tail_metrics(y: np.ndarray, S: np.ndarray) -> dict:
    """How the forecast's tails and spread line up with outcomes.

    below_p10 / above_p90: share of outcomes beyond the 10th / 90th
        percentile (0.10 each when calibrated). The decision rules
        (floors, CVaR, P(clear)) read exactly these tails; central-interval
        coverage can look fine while the two tails are off in opposite
        directions.
    pred_sd:  mean predictive sd per player.
    resid_sd: sd of (outcome - predictive mean). With good means, resid_sd
        ~ pred_sd when the spread is right.
    pit_hist: counts of the PIT value P(draw < outcome) in 10 bins (flat when calibrated)."""
    if len(y) == 0:
        return {}
    p10, p90 = np.percentile(S, [10, 90], axis=0)
    pit = (S < y[None, :]).mean(0)
    return {'below_p10': float((y < p10).mean()), 'above_p90': float((y > p90).mean()),
            'pred_sd': float(S.std(0).mean()), 'resid_sd': float((y - S.mean(0)).std()),
            'pit_hist': np.histogram(pit, bins=10, range=(0, 1))[0].tolist()}


def tier_of(expected: np.ndarray) -> np.ndarray:
    return np.array(TIER_LABELS)[np.digitize(expected, TIER_EDGES)]


def calibration_table(frame: pl.DataFrame, S: np.ndarray, by_position: bool = False) -> pl.DataFrame:
    """Tail calibration by projection tier (and optionally position) for a
    frame with outcomes and its draws."""
    y = frame[TARGET_COL].to_numpy().astype(float)
    tiers = tier_of(S.mean(0))
    groups = [('all', np.ones(len(y), bool))] if not by_position else \
        [(p, (frame['position'] == p).to_numpy()) for p in sorted(frame['position'].unique().to_list())]
    rows = []
    for g, gm in groups:
        for t in TIER_LABELS:
            m = gm & (tiers == t)
            if m.sum() == 0:
                continue
            tm = tail_metrics(y[m], S[:, m])
            tm.pop('pit_hist')
            rows.append({'group': g, 'tier': t, 'n': int(m.sum()), 'mean_expected': float(S[:, m].mean()),
                         'mean_actual': float(y[m].mean()), **tm})
    return pl.DataFrame(rows)


def past_only_baseline(rows: pl.DataFrame, train: pl.DataFrame) -> np.ndarray:
    """Each player's mean points so far this season in ``train`` (falling back
    to the training-set mean) -- the naive reference forecast."""
    b = (train.with_columns((pl.col('player_id') + '_' + pl.col('season').cast(pl.String)).alias('_ps'))
         .group_by('_ps').agg(pl.col(TARGET_COL).mean().alias('_pm')))
    keyed = rows.with_columns((pl.col('player_id') + '_' + pl.col('season').cast(pl.String)).alias('_ps'))
    return (keyed.join(b, on='_ps', how='left', maintain_order='left')['_pm']
            .fill_null(float(train[TARGET_COL].mean())).to_numpy())
