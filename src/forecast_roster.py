"""
Mechanically extract fantasy point forecasts for a standard roster.

The model gives you a full posterior predictive DISTRIBUTION per player-week,
not a point estimate. That distribution is the whole value proposition -- start/
sit is a question about P(A > B), and tournament vs cash-game strategy is a
question about ceiling vs floor. So the pipeline here is:

    prediction frame -> set_data -> posterior predictive draws
                     -> per-player summary (mean / floor / ceiling)
                     -> lineup optimization
                     -> lineup total distribution

IMPORTANT: keep the draws matrix around. Summarizing to means too early throws
away the correlation structure between players, which is what you need for
head-to-head comparisons and for the distribution of a lineup TOTAL.

NOTE ON POSITIONS: this dataset only contains QB / RB / WR / TE. Kickers and
team defenses aren't modeled, so the "standard roster" here is
QB, RB, RB, WR, WR, TE, FLEX(RB/WR/TE).
"""

import numpy as np
import polars as pl
import pymc as pm
import arviz as az

# standard lineup: position -> number of dedicated slots
ROSTER = {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1}
FLEX_ELIGIBLE = ('RB', 'WR', 'TE')
N_FLEX = 1


# --------------------------------------------------------------- 1. framing

def build_prediction_frame(raw, target_season, target_week, scalers, idxs,
                           std_cols, binary_cols, idx_cols):
    """Rows to forecast, with lag features computed correctly and the SAME
    scalers/encoders used at training time.

    The lag features must come from each player's own previous game -- note the
    .over('player_id'), which the training pipeline is missing.
    """
    lagged = (
        raw
        .sort(['player_id', 'season', 'week'])
        .with_columns(
            pl.col(
                'yards_per_rush_attempt', 'target_share',
                'yards_per_target', 'yards_per_pass_attempt',
            ).shift(1).over('player_id').name.prefix('lag_')      # <- .over()
        )
        .fill_null(0)
    )

    frame = lagged.filter(
        (pl.col('season') == target_season) & (pl.col('week') == target_week)
    )
    if len(frame) == 0:
        raise ValueError(f"no rows for season {target_season} week {target_week}")

    # apply TRAINING scalers -- never refit on the prediction slice
    frame = frame.with_columns([
        ((pl.col(c) - mu) / sd).alias(f"{c}_z") for c, (mu, sd) in scalers.items()
    ])
    # apply TRAINING enums -- categories must line up with the model coords
    frame = frame.with_columns([
        pl.col(col).cast(pl.String).cast(idxs[col]).to_physical().cast(pl.Int32).alias(f"{col}_idx")
        for col in idx_cols
    ])

    null_idx = [c for c in idx_cols if frame[f"{c}_idx"].null_count() > 0]
    if null_idx:
        raise ValueError(
            f"unseen categories in {null_idx} -- these levels weren't in the "
            f"enums the model was built with, so there is no fitted effect for "
            f"them. Rebuild coords including this week, or drop those rows."
        )
    return frame


# ----------------------------------------------------- 2. predictive draws

def forecast(model, idata, frame, cont_cols, binary_cols, seed=41029041):
    """Return (draws, frame) where draws is (n_draws, n_players).

    Rows of `draws` are paired posterior samples -- column i and column j come
    from the SAME posterior draw, so differences/sums across columns correctly
    propagate the model's correlation structure.
    """
    with model:
        pm.set_data({
            'teams_data': frame['team_idx'].to_numpy().squeeze(),
            'player_data': frame['player_idx'].to_numpy().squeeze(),
            'player_season_data': frame['player_season_idx'].to_numpy().squeeze(),
            'position_data': frame['position_idx'].to_numpy().squeeze(),
            'tenure_data': frame['tenure_idx'].to_numpy().squeeze(),
            'games_data': frame['week_idx'].to_numpy().squeeze(),
            'cont_data': frame.select(cont_cols).to_numpy(),
            'bi_data': frame.select(binary_cols).to_numpy().astype(float),
            'obs_data': np.zeros(len(frame)),          # ignored; shape must match
        })
        ppc = pm.sample_posterior_predictive(
            idata, var_names=['y_obs'], predictions=True, random_seed=seed
        )

    draws = (
        ppc.predictions['y_obs']
        .stack(sample=('chain', 'draw'))
        .transpose('sample', ...)
        .values
    )
    return draws, frame


# ------------------------------------------------------ 3. player summaries

def player_summary(draws, frame, hdi_prob=0.8):
    """Per-player forecast table, sorted by expected points.

    floor/ceiling are the 10th/90th percentiles -- for fantasy these matter more
    than the mean. A high-floor player wins cash games; a high-ceiling player
    wins tournaments. Two players with the same mean are not interchangeable.
    """
    hdi = az.hdi(draws[None, ...], hdi_prob=hdi_prob)['x']
    return (
        pl.DataFrame({
            'player': frame['player'],
            'position': frame['position'],
            'team': frame['team'],
            'opp': frame['opp_team'],
            'expected': draws.mean(axis=0),
            'median': np.median(draws, axis=0),
            'sd': draws.std(axis=0),
            'floor_p10': np.percentile(draws, 10, axis=0),
            'ceiling_p90': np.percentile(draws, 90, axis=0),
            'hdi_low': hdi[:, 0],
            'hdi_high': hdi[:, 1],
        })
        .sort('expected', descending=True)
    )


def head_to_head(draws, frame, player_a, player_b):
    """P(player_a outscores player_b) using PAIRED draws.

    Do not compute this from the two marginal summaries -- that ignores the
    shared uncertainty (same team, same game environment, same population
    parameters) and will give you an overconfident answer.
    """
    names = frame['player'].to_list()
    ia, ib = names.index(player_a), names.index(player_b)
    diff = draws[:, ia] - draws[:, ib]
    return {
        'p_a_beats_b': float((diff > 0).mean()),
        'mean_margin': float(diff.mean()),
        'p10_margin': float(np.percentile(diff, 10)),
        'p90_margin': float(np.percentile(diff, 90)),
    }


# --------------------------------------------------- 4. lineup construction

def optimize_lineup(summary, roster=ROSTER, flex_eligible=FLEX_ELIGIBLE, n_flex=N_FLEX,
                    pool=None, objective='expected'):
    """Greedy fill by `objective`, then FLEX from the best remaining.

    Greedy is optimal for this roster shape: slots are independent, so taking
    the top-k at each position and then the best leftover for FLEX maximizes the
    expected total. (That stops being true the moment you add salary caps --
    DFS needs a real knapsack solver.)

    objective: 'expected' (default), 'floor_p10' for safe/cash lineups, or
    'ceiling_p90' for tournament lineups.
    """
    avail = summary if pool is None else summary.filter(pl.col('player').is_in(pool))
    avail = avail.sort(objective, descending=True)

    picks, used = [], set()
    for pos, n_slots in roster.items():
        chosen = (
            avail.filter((pl.col('position') == pos) & (~pl.col('player').is_in(list(used))))
            .head(n_slots)
        )
        for row in chosen.iter_rows(named=True):
            picks.append({**row, 'slot': pos})
            used.add(row['player'])

    flex = (
        avail.filter(
            pl.col('position').is_in(list(flex_eligible))
            & (~pl.col('player').is_in(list(used)))
        )
        .head(n_flex)
    )
    for row in flex.iter_rows(named=True):
        picks.append({**row, 'slot': 'FLEX'})
        used.add(row['player'])

    return pl.DataFrame(picks).select(
        ['slot', 'player', 'position', 'team', 'opp',
         'expected', 'floor_p10', 'ceiling_p90', 'sd']
    )


def lineup_distribution(draws, frame, lineup):
    """Distribution of the lineup TOTAL, summing paired draws.

    Summing per-player means gives the right expected total but tells you
    nothing about spread. Summing the draws row-wise keeps the correlations --
    e.g. a QB and his own WR move together, which widens the total's variance
    relative to a naive independence assumption.
    """
    names = frame['player'].to_list()
    cols = [names.index(p) for p in lineup['player'].to_list()]
    totals = draws[:, cols].sum(axis=1)
    return {
        'expected': float(totals.mean()),
        'median': float(np.median(totals)),
        'sd': float(totals.std()),
        'floor_p10': float(np.percentile(totals, 10)),
        'ceiling_p90': float(np.percentile(totals, 90)),
        'totals': totals,
    }


def p_beats(totals_a, totals_b):
    """P(lineup A beats lineup B). If the two totals come from the same draws
    matrix they're already paired; otherwise this assumes independence, which
    is the right call for an opponent whose roster shares no players."""
    n = min(len(totals_a), len(totals_b))
    return float((totals_a[:n] > totals_b[:n]).mean())


# ------------------------------------------------------------------ example

if __name__ == '__main__':
    import polars.selectors as cs

    # these come from whichever model script you fit -- ff-mlm.py's globals
    from importlib import import_module
    mod = import_module('ff-mlm')          # or ff-mlm-global-age-ar
    model, raw, scalers, idxs = mod.ff_mlm, mod.add_lags, mod.scalers, mod.idxs
    coords = mod.coords

    idata = az.from_netcdf('model-nc/ff_mlm.nc')

    frame = build_prediction_frame(
        raw=mod.raw_data,
        target_season=2025, target_week=18,
        scalers=scalers, idxs=idxs,
        std_cols=mod.std_these, binary_cols=mod.binary_vars, idx_cols=mod.idx_cols,
    )

    draws, frame = forecast(
        model, idata, frame,
        cont_cols=coords['cont_vars'], binary_cols=coords['binary_vars'],
    )
    print(f"draws matrix: {draws.shape}  (posterior draws x players)")

    summary = player_summary(draws, frame)
    print(summary.head(25))

    lineup = optimize_lineup(summary)
    print(lineup)

    dist = lineup_distribution(draws, frame, lineup)
    print(f"lineup total: expected={dist['expected']:.1f}  "
          f"floor(p10)={dist['floor_p10']:.1f}  ceiling(p90)={dist['ceiling_p90']:.1f}")

    # safe vs boom-or-bust construction off the same draws
    safe = optimize_lineup(summary, objective='floor_p10')
    tourney = optimize_lineup(summary, objective='ceiling_p90')
    d_safe = lineup_distribution(draws, frame, safe)
    d_tourney = lineup_distribution(draws, frame, tourney)
    print(f"safe   : E={d_safe['expected']:.1f}  p10={d_safe['floor_p10']:.1f}  p90={d_safe['ceiling_p90']:.1f}")
    print(f"tourney: E={d_tourney['expected']:.1f}  p10={d_tourney['floor_p10']:.1f}  p90={d_tourney['ceiling_p90']:.1f}")
    print(f"P(tourney beats safe) = {p_beats(d_tourney['totals'], d_safe['totals']):.3f}")
