"""Walk-forward backtest for the production models.

One fold = one (model, season, week): fit on the ``WINDOW_SEASONS`` seasons
before the week, forecast that week's played rows, score against what
happened. Each fold is written to its own JSON file in a run directory, so
sweeps resume after an interruption and can run in parallel processes
without clobbering a shared CSV.

Eligibility (which test rows get scored):
  'model'  (default) every row the fitted model can score: the player was in
           the training window and his tenure/age is inside the fitted grid.
  'legacy' src/backtest_harness.py's rule: every categorical in
           [player_id, position, team, tenure, opp_team, season, week,
           player_season] must have appeared in training. het_gp uses none
           of season/week/player_season/team/opp_team, and the rule drops
           all of week 1 and every player's first game of a season. Kept
           only to reproduce old results.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
import polars as pl

from . import data as D
from .config import MAX_WEEK, MIN_TRAIN_ROWS, MODELS, TARGET_COL, WINDOW_SEASONS, SamplerSettings
from .model import fit
from .scoring import TIER_LABELS, calibration_table, past_only_baseline, score

log = logging.getLogger(__name__)

LEGACY_IDX_COLS = ('player_id', 'position', 'team', 'tenure', 'opp_team', 'season', 'week', 'player_season')


def legacy_eligible(train: pl.DataFrame, test: pl.DataFrame) -> np.ndarray:
    """src/backtest_harness.Fold.drop_cold_starts as a mask."""
    test = test if 'player_season' in test.columns else test.with_columns(
        pl.concat_str([pl.col('player_id'), pl.lit('_'), pl.col('season')]).alias('player_season'))
    mask = np.ones(test.height, dtype=bool)
    for c in LEGACY_IDX_COLS:
        if c == 'week':
            vocab = {str(float(w)) for w in range(1, MAX_WEEK + 1)}
        else:
            vocab = set(train[c].cast(pl.String).unique().to_list())
        mask &= test[c].cast(pl.String).is_in(list(vocab)).fill_null(False).to_numpy()
    return mask


def fold_path(run_dir: Path, model: str, season: int, week: float) -> Path:
    return Path(run_dir) / f'{model}__{season}__wk{int(week):02d}.json'


def run_fold(model_name: str, season: int, week: float, history: pl.DataFrame, team_form: pl.DataFrame,
             run_dir: Path | str, settings: SamplerSettings = SamplerSettings(),
             eligibility: str = 'model', window_seasons: int = WINDOW_SEASONS,
             primary_position_scope: str = 'asof', overwrite: bool = False,
             min_train_rows: int = MIN_TRAIN_ROWS) -> dict | None:
    """Fit + score one fold and write it to ``run_dir``. Returns the fold
    record, or None when skipped (already done / no data).

    model_name: a key of config.MODELS, or 'baseline' (each player's mean so
        far this season) for a reference row scored on the same rows the skill
        or QB model would see -- pass ``baseline_for`` via the name
        'baseline:<model>'.
    primary_position_scope: 'asof' uses each player's most frequent position
        up to the target week; 'all' uses every season in ``history``
    """
    if eligibility not in ('model', 'legacy'):
        raise ValueError("eligibility must be 'model' or 'legacy'")
    week = float(week)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    out = fold_path(run_dir, model_name.replace(':', '-'), season, week)
    if out.exists() and not overwrite:
        log.info('skip %s (exists)', out.name)
        return None

    is_baseline = model_name.startswith('baseline')
    cfg = MODELS[model_name.split(':', 1)[1] if is_baseline else model_name]

    df = D.join_team_form(history, team_form)
    ref = df if primary_position_scope == 'all' else df.filter(
        D.before(season, week) | ((pl.col('season') == season) & (pl.col('week') == week)))
    df = D.with_primary_position(df, reference=ref)
    df = df.filter(pl.col('position').is_in(list(cfg.positions)))
    train = D.training_window(df, season, week, window_seasons)
    test = D.week_rows(df, season, week).sort('player_id')
    if test.height == 0 or train.height < min_train_rows:
        log.info('skip %s %d wk%g: no test rows or too little training data', model_name, season, week)
        return None

    if is_baseline:
        keep = test.filter(pl.Series(legacy_eligible(train, test))) if eligibility == 'legacy' else test
        S = past_only_baseline(keep, train)[None, :]
        fit_info = {'fit_time': 0.0, 'n_train': train.height, 'divergences': 0, 'max_rhat': None}
        coefs = None
    else:
        fm = fit(train, cfg, settings)
        ok = fm.prep.eligible(test)
        if eligibility == 'legacy':
            ok &= legacy_eligible(train, test)
        keep = test.filter(pl.Series(ok))
        if keep.height == 0:
            log.info('skip %s %d wk%g: no scoreable rows', model_name, season, week)
            return None
        S = fm.predict(keep)
        fit_info = {'fit_time': fm.fit_time, 'n_train': fm.n_train, 'divergences': fm.divergences,
                        'max_rhat': fm.max_rhat, 'min_ess_bulk': fm.min_ess_bulk}
        coefs = fm.coefs().to_dicts()

    y = keep[TARGET_COL].to_numpy()
    base = past_only_baseline(keep, train)
    metrics = score(y, S, baseline=base)
    by_pos = []
    pos = keep['position'].to_numpy()
    for p_ in np.unique(pos):
        m = pos == p_
        if m.sum() >= 5:
            by_pos.append({'pos': str(p_), **score(y[m], S[:, m], baseline=base[m])})
    # tail calibration by projection tier, per position (pooled across folds in summarize)
    by_tier = calibration_table(keep, S, by_position=True).rename({'group': 'pos'}).to_dicts()
    record = dict(model=model_name, season=season, week=week, method=settings.method,
                  sampler=settings.nuts_sampler if settings.method == 'nuts' else None,
                  eligibility=eligibility, window_seasons=window_seasons,
                  n_test=test.height, n_dropped=test.height - keep.height, **fit_info, **metrics,
                  by_position=by_pos, by_tier=by_tier, coefs=coefs, settings={'draws': settings.draws, 'tune': settings.tune,
                                                                 'chains': settings.chains, 'seed': settings.seed})
    tmp = out.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(record, indent=1, default=float))
    os.replace(tmp, out)
    log.info('%s %d wk%g: n=%d crps=%.3f rmse=%.3f (base %.3f) cov80=%.2f', model_name, season, week,
             metrics['n'], metrics['crps'], metrics['rmse'], metrics['rmse_base'], metrics['cov80'])
    return record


# --------------------------------------------------------------- summarize

def load_run(run_dir: Path | str) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """(folds, by_position, coefs) frames from every fold JSON in run_dir."""
    folds, bypos, coefs = [], [], []
    for f in sorted(Path(run_dir).glob('*.json')):
        r = json.loads(f.read_text())
        key = {'model': r['model'], 'season': r['season'], 'week': r['week'], 'eligibility': r['eligibility']}
        for p_ in r.pop('by_position') or []:
            bypos.append({**key, **p_})
        for c in r.pop('coefs') or []:
            coefs.append({**key, **c})
        r.pop('by_tier', None)
        r.pop('settings', None)
        folds.append(r)
    mk = lambda rows: pl.DataFrame(rows) if rows else pl.DataFrame()  
    return mk(folds), mk(bypos), mk(coefs)


def load_tiers(run_dir: Path | str) -> pl.DataFrame:
    """Per-fold tail calibration rows (model, season, week, pos, tier, n, below_p10, ...)."""
    rows = []
    for f in sorted(Path(run_dir).glob('*.json')):
        r = json.loads(f.read_text())
        for t in r.get('by_tier') or []:
            rows.append({'model': r['model'], 'season': r['season'], 'week': r['week'], **t})
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def tier_summary(tiers: pl.DataFrame, by_position: bool = False) -> pl.DataFrame:
    """Tail calibration pooled over folds, weighted by rows. Calibrated:
    below_p10 ~ above_p90 ~ 0.10 and resid_sd ~ pred_sd in every tier."""
    keys = ['model', 'pos', 'tier'] if by_position else ['model', 'tier']
    w = lambda c: ((pl.col(c) * pl.col('n')).sum() / pl.col('n').sum()).alias(c)  # noqa: E731
    order = {t: i for i, t in enumerate(TIER_LABELS)}
    return (tiers.group_by(keys)
            .agg(pl.col('n').sum(), *[w(c) for c in ('mean_expected', 'mean_actual', 'below_p10', 'above_p90',
                                                     'pred_sd', 'resid_sd')])
            .with_columns(pl.col('tier').replace_strict(order, return_dtype=pl.Int32).alias('_o'))
            .sort([*keys[:-1], '_o']).drop('_o'))


def paired_diff(folds: pl.DataFrame, a: str, b: str, metric: str = 'crps') -> dict | None:
    """Mean +- SE across folds of metric(a) - metric(b), on folds both ran."""
    k = ['season', 'week', 'eligibility']
    x = folds.filter(pl.col('model') == a).select(*k, pl.col(metric).alias('a'), pl.col('n').alias('na'))
    y = folds.filter(pl.col('model') == b).select(*k, pl.col(metric).alias('b'), pl.col('n').alias('nb'))
    j = x.join(y, on=k)
    if j.height == 0:
        return None
    if (j['na'] != j['nb']).any():
        log.warning('%s vs %s: some folds scored different row counts -- not a like-for-like comparison', a, b)
    d = (j['a'] - j['b']).to_numpy()
    se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float('nan')
    return {'a': a, 'b': b, 'metric': metric, 'n_folds': len(d), 'mean_diff': float(d.mean()), 'se': se,
                'folds_a_better': int((d < 0).sum())}


def summarize(run_dir: Path | str) -> pl.DataFrame:
    folds, bypos, _ = load_run(run_dir)
    if folds.height == 0:
        print(f'no folds in {run_dir}')
        return folds
    w = lambda c: (pl.col(c) * pl.col('n')).sum() / pl.col('n').sum() 
    pooled = (folds.group_by('model', 'eligibility')
              .agg(pl.len().alias('folds'), pl.col('n').sum().alias('rows'),
                   *[w(c).alias(c) for c in ('crps', 'rmse', 'mae', 'spearman', 'cov50', 'cov80', 'cov90', 'rmse_base')],
                   pl.col('divergences').sum().alias('divergences'),
                   pl.col('fit_time').mean().alias('mean_fit_s'))
              .sort('model', 'eligibility'))
    with pl.Config(tbl_cols=20, tbl_width_chars=200, float_precision=3):
        print(f'=== pooled (weighted by rows) -- {run_dir}')
        print(pooled)
        if bypos.height:
            print('=== by position')
            print(bypos.group_by('model', 'pos').agg(
                pl.col('n').sum().alias('rows'),
                *[((pl.col(c) * pl.col('n')).sum() / pl.col('n').sum()).alias(c) for c in ('crps', 'rmse', 'cov80')]
            ).sort('model', 'pos'))
        tiers = load_tiers(run_dir)
        if tiers.height:
            tiers = tiers.filter(~pl.col('model').str.starts_with('baseline'))
        if tiers.height:
            print('=== tail calibration by projection tier (calibrated: below_p10 ~ above_p90 ~ 0.10, '
                  'resid_sd ~ pred_sd)')
            print(tier_summary(tiers))
            print(tier_summary(tiers, by_position=True))
        pit = folds.filter(~pl.col('model').str.starts_with('baseline'))
        if 'pit_hist' in pit.columns and pit.height:
            for (m,), g in pit.group_by('model'):
                h = np.sum([x for x in g['pit_hist'].to_list() if x is not None], axis=0)
                print(f'PIT deciles {m}: {(h / h.sum()).round(3).tolist()}  (calibrated: 0.1 each)')
    for m in folds['model'].unique().sort().to_list():
        b = f'baseline:{m}'
        if not m.startswith('baseline') and b in folds['model'].to_list():
            d = paired_diff(folds, m, b)
            if d:
                print(f'{m} vs {b}: dCRPS {d["mean_diff"]:+.3f} +- {d["se"]:.3f} '
                      f'({d["folds_a_better"]}/{d["n_folds"]} folds better)')
    return pooled
