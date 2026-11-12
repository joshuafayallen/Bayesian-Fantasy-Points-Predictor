import numpy as np
import polars as pl
from scipy import stats
from ffpred.config import PATHS
 
RUN = 'scratch_joint_qb_resid_cut'
OUT = PATHS.results_dir / RUN
FOLDS = [(s, w) for s in (2023, 2024, 2025) for w in (4, 8, 12, 16)]
rng = np.random.default_rng(7)
 
 
def load(s, w):
    S = np.load(OUT / f'S_{s}_{w:02d}.npy').astype(np.float64)
    te = pl.read_parquet(OUT / f'te_{s}_{w:02d}.parquet').with_row_index('col')
    return S, te
 
 
def normal_scores(S, y):
    """Actual: Phi^-1 of the PIT. Draws: Phi^-1 of within-column ranks (same transform, so comparable)."""
    n = S.shape[0]
    u = ((S < y).sum(0) + 0.5 * (S == y).sum(0) + 0.5) / (n + 1)        # randomised-free mid PIT
    z_act = stats.norm.ppf(np.clip(u, 1e-3, 1 - 1e-3))
    z_draw = stats.norm.ppf((S.argsort(0).argsort(0) + 1) / (n + 1))
    return z_act, z_draw
 
 
def starters(te, S):
    """Team -> column of its starting QB (top projected QB on that team)."""
    q = te.filter(pl.col('position') == 'QB').with_columns(pl.Series('proj', S[:, te.filter(
        pl.col('position') == 'QB')['col'].to_numpy()].mean(0)))
    return dict(q.sort('proj', descending=True).unique('team', keep='first').select('team', 'col').iter_rows())
 
 
# %% ============================ 1. correlation calibration ============================
pairs = []
for s, w in FOLDS:
    S, te = load(s, w)
    y = te['total_fantasy_points'].to_numpy().astype(float)
    z_act, z_draw = normal_scores(S, y)
    qb = starters(te, S)
    sk = te.filter(pl.col('position') != 'QB')
    proj = S.mean(0)
    # within-team rank among same-position players (WR1, WR2, TE1 ...)
    sk = sk.with_columns(pl.Series('proj', proj[sk['col'].to_numpy()])) \
           .with_columns(pl.col('proj').rank('ordinal', descending=True).over('team', 'position').alias('depth'))
    for c, pos, team, opp, depth in sk.select('col', 'position', 'team', 'opp_team', 'depth').iter_rows():
        for side, t in (('own', team), ('opp', opp)):
            if t in qb:
                q = qb[t]
                pairs.append(dict(season=s, week=w, side=side, position=pos,
                                  role=f"{pos}{min(depth, 3)}{'+' if depth >= 3 else ''}",
                                  z_qb=z_act[q], z_rec=z_act[c],
                                  model_corr=float(np.corrcoef(z_draw[:, q], z_draw[:, c])[0, 1])))
P = pl.DataFrame(pairs)
 
 
def corr_table(df, by):
    return (df.group_by(by)
              .agg(pl.len().alias('n'),
                   pl.corr('z_qb', 'z_rec').alias('actual_corr'),
                   pl.col('model_corr').mean().alias('model_corr'))
              .with_columns((1 / (pl.col('n') - 3).sqrt()).alias('se_actual'))      # Fisher-z approx
              .sort(by))
 
 
with pl.Config(tbl_rows=-1, float_precision=3):
    print('--- own QB: model vs actual correlation (normal scores) ---')
    print(corr_table(P.filter(pl.col('side') == 'own'), ['position']))
    print(corr_table(P.filter(pl.col('side') == 'own'), ['role']))
    print('--- placebo: OPPOSING QB (model says 0) ---')
    print(corr_table(P.filter(pl.col('side') == 'opp'), ['position']))
 
 
# %% ============================ 2. stack calibration: QB + WR1 + TE1 ============================
def stack_cols(te, S, qb):
    sk = te.filter(pl.col('position').is_in(['WR', 'TE'])) \
           .with_columns(pl.Series('proj', S[:, te.filter(pl.col('position').is_in(['WR', 'TE']))['col'].to_numpy()].mean(0)))
    top = sk.sort('proj', descending=True).unique(['team', 'position'], keep='first')
    out = {}
    for team, q in qb.items():
        wr = top.filter((pl.col('team') == team) & (pl.col('position') == 'WR'))['col'].to_list()
        tt = top.filter((pl.col('team') == team) & (pl.col('position') == 'TE'))['col'].to_list()
        if wr and tt:
            out[team] = (q, wr[0], tt[0])
    return out
 
 
rows = []
for s, w in FOLDS:
    S, te = load(s, w)
    y = te['total_fantasy_points'].to_numpy().astype(float)
    qb = starters(te, S)
    for team, (q, wr, tt) in stack_cols(te, S, qb).items():
        actual = y[q] + y[wr] + y[tt]
        T_corr = S[:, q] + S[:, wr] + S[:, tt]
        T_ind = S[rng.permutation(len(S)), q] + S[:, wr] + S[:, tt]       # same marginals, QB link broken
        for label, T in (('correlated', T_corr), ('independent', T_ind)):
            pit = (T < actual).mean()
            rows.append(dict(draws=label, season=s, week=w, team=team, pit=pit,
                             sd=T.std(), crps=float(np.mean(np.abs(T - actual))
                                                    - 0.5 * np.mean(np.abs(T - rng.permutation(T))))))
R = pl.DataFrame(rows)
with pl.Config(float_precision=3):
    print('--- QB + WR1 + TE1 stacks (target cov 0.50 / 0.80 / 0.90, tails 0.10 / 0.10) ---')
    print(R.group_by('draws').agg(
        pl.len().alias('stacks'), pl.col('sd').mean().alias('mean_sd'), pl.col('crps').mean(),
        ((pl.col('pit') > 0.25) & (pl.col('pit') < 0.75)).mean().alias('cov50'),
        ((pl.col('pit') > 0.10) & (pl.col('pit') < 0.90)).mean().alias('cov80'),
        ((pl.col('pit') > 0.05) & (pl.col('pit') < 0.95)).mean().alias('cov90'),
        (pl.col('pit') < 0.10).mean().alias('below_p10'),
        (pl.col('pit') > 0.90).mean().alias('above_p90')).sort('draws'))
    w_ = R.pivot(on='draws', index=['season', 'week', 'team'], values='crps')
    d = (w_['correlated'] - w_['independent']).to_numpy()
    print(f'stack CRPS correlated - independent: {d.mean():+.3f} ± {d.std(ddof=1) / np.sqrt(len(d)):.3f}  '
          f'(negative = correlation helps; {len(d)} stacks)')
 