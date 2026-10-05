"""
Team-context features for the player models: signed Vegas lines and the
off/def team model's pre-game quantities, plus a position-varying effect.

Used from models.py (see the frame_to_D / build_team_mod hooks). Kept separate
so the ablation arms are just different column lists:

    arm 1  current                      (team_form, opp_team_form; raw spread_line)
    arm 2  Vegas sign fix               VEGAS_COLS in place of spread_line
    arm 3  + model context              VEGAS_COLS + CTX_COLS            (position-varying slopes)
    arm 4  + model context, own offense VEGAS_COLS + CTX_COLS_OFF

roll_avg_points_allowed and f_opp stay as they are in every arm: they carry the
opponent matchup (efficiency allowed, by position). The model features added
here carry what those don't: own-team scoring / volume and game script.
"""

import numpy as np
import polars as pl
import pymc as pm

TEAM_PREGAME_PATH = 'processed-data/team_pregame.parquet'

VEGAS_COLS = ['team_spread', 'vegas_team_total']
CTX_COLS = ['exp_pts_for', 'exp_margin']
CTX_COLS_OFF = ['team_off_form', 'exp_margin']


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------
def add_vegas_features(df):
    """spread_line / total_line in the player rows are game-level and
    HOME-perspective (corr with the player's own margin: +0.45 home rows,
    -0.45 away rows), so as a single linear feature their effect cancels.
    Re-express them from the player's team's side."""
    return df.with_columns(
        (pl.col('spread_line') * (2 * pl.col('is_home') - 1)).alias('team_spread'),     # + = player's team favoured
    ).with_columns(
        (pl.col('total_line') / 2 + pl.col('team_spread') / 2).alias('vegas_team_total'),
    )


def join_team_pregame(df, path=TEAM_PREGAME_PATH, cols=None, pregame=None):
    """EXACT join on (season, week, team): team_pregame.parquet is already
    pre-game (built from the filter's predicted state for that week), so no
    as-of shift -- unlike join_team_form, whose values are post-game."""
    pg = pregame if pregame is not None else pl.read_parquet(path)
    cols = cols or ['exp_pts_for', 'exp_pts_against', 'exp_margin', 'exp_total',
                    'exp_pts_for_sd', 'exp_margin_sd', 'team_off_form', 'team_def_form',
                    'opp_off_form', 'opp_def_form', 'params_in_sample']
    key = ['season', 'week', 'team']
    pg = pg.select(*key, *cols).with_columns(pl.col('season').cast(df.schema['season']),
                                             pl.col('week').cast(df.schema['week']))
    n0 = df.height
    out = df.drop([c for c in cols if c in df.columns]).join(pg, on=key, how='left')
    assert out.height == n0, f'team_pregame join changed row count: {n0} -> {out.height}'
    missing = out.select(pl.col(cols[0]).is_null().sum()).item()
    if missing:
        print(f'warning: {missing} player rows have no team_pregame row (team/week not in the schedule?)')
    return out


def ctx_array(frame, scalers, cols):
    """(n_rows, n_cols) standardized with TRAIN scalers (models.fit_scalers)."""
    return np.column_stack([((frame[c] - scalers[c][0]) / scalers[c][1]).to_numpy() for c in cols])


# ---------------------------------------------------------------------------
# effect: slopes that vary by position, partially pooled
# ---------------------------------------------------------------------------
def add_team_context_effects(ctx_data, position_data, prefix='ctx'):
    """sum_k beta[position, k] * x_k, with beta[p, k] = beta_bar[k] + tau[k] * z[p, k].

    Needs 'position' and f'{prefix}_vars' in the model coords. Position-varying
    because game script should push RBs (leading -> run) and QB/WR/TE
    (trailing -> pass) in opposite directions; a single shared slope would
    average that to ~0. Priors are on the standardized-feature / model-y scale
    used by team_mod's beta_team_form ~ N(0, 1)."""
    beta_bar = pm.Normal(f'{prefix}_beta_bar', 0, 0.5, dims=f'{prefix}_vars')
    tau = pm.HalfNormal(f'{prefix}_tau', 0.25, dims=f'{prefix}_vars')
    z = pm.Normal(f'{prefix}_z', 0, 1, dims=('position', f'{prefix}_vars'))
    beta = pm.Deterministic(f'{prefix}_beta', beta_bar + tau * z, dims=('position', f'{prefix}_vars'))
    return (ctx_data * beta[position_data]).sum(axis=-1)
