import numpy as np
import polars as pl
import pymc as pm
import arviz as az
import tidydraws as td
 
import models
 
# standard lineup: position -> number of dedicated slots
ROSTER = {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1}
FLEX_ELIGIBLE = ('RB', 'WR', 'TE')
N_FLEX = 1
 
# lag features computed at prediction time -- must match models.STD_COLS'
# lag_* columns (minus lag_target_share, which STD_COLS itself computes as
# a lag of 'target_share'). Kept as an explicit list rather than derived
# from STD_COLS so it's obvious at a glance which RAW columns get shifted.
LAG_SOURCE_COLS = ('yards_per_rush_attempt', 'target_share', 'yards_per_target',
                    'yards_per_pass_attempt', 'avg_depth_of_target')
 
 
# --------------------------------------------------------------- 0. plumbing
 
def resolve_player(frame, ref):
    """Accepts either a player_id or a display name and returns a
    player_id. If `ref` is already a player_id in `frame`, use it directly
    (unambiguous by construction). Otherwise treat it as a display name --
    but only if EXACTLY ONE row in `frame` has that name; a name matching
    more than one player_id (e.g. two real "Josh Allen"s, at different
    positions or different eras) is a hard error rather than silently
    picking one, since that ambiguity is exactly what player_id-based
    matching exists to avoid."""
    if ref in frame['player_id'].to_list():
        return ref
    matches = frame.filter(pl.col('player') == ref)['player_id'].to_list()
    if len(matches) == 0:
        raise ValueError(f"{ref!r} matches no player_id or player name in this frame")
    if len(matches) > 1:
        raise ValueError(
            f"{ref!r} matches {len(matches)} different player_ids ({matches}) -- "
            f"ambiguous display name, pass a player_id instead"
        )
    return matches[0]
 
 
# --------------------------------------------------------------- 1. framing
 
def build_prediction_frame(raw, target_season, target_week, team_form):
    """Rows to forecast for one (season, week): lag features computed from
    each player's own previous game (note the .over('player_id')), plus
    team_form/opp_team_form joined the same way fit_model.py's prep_data
    does it.
 
    Does NOT scale/encode anything -- that's models.frame_to_D's job,
    using the TRAINING-time scalers/idxs/team_form_scaler
    (fit_model.py's prep_data() returns all three; never refit them here).
 
    raw: the same raw ff-processed.parquet frame prep_data() reads.
    team_form: the raw team_form.parquet frame (season, week, team,
        team_form columns) -- NOT team_form_scaler; that's a separate
        (mu, sd) pair from models.fit_team_form_scaler, applied inside
        frame_to_D, not here."""
    lagged = (
        raw
        .sort(['player_id', 'season', 'week'])
        .with_columns(
            pl.col(*LAG_SOURCE_COLS).shift(1).over('player_id').name.prefix('lag_')
        )
        .fill_null(0)
    )
 
    frame = lagged.filter(
        (pl.col('season') == target_season) & (pl.col('week') == target_week)
    )
    if len(frame) == 0:
        raise ValueError(f"no rows for season {target_season} week {target_week}")
 
    frame = frame.join(team_form, on=['season', 'week', 'team'], how='left')
    opp_team_form = team_form.rename({'team': 'opp_team', 'team_form': 'opp_team_form'})
    frame = frame.join(opp_team_form, on=['season', 'week', 'opp_team'], how='left')
 
    n_missing_form = frame['team_form'].null_count()
    n_missing_opp_form = frame['opp_team_form'].null_count()
    if n_missing_form or n_missing_opp_form:
        print(f"warning: {n_missing_form} rows missing team_form, "
              f"{n_missing_opp_form} missing opp_team_form -- filling with 0.0 "
              f"(check target_week's dtype matches team_form.parquet's -- "
              f"week is float64 there, e.g. pass 18.0 not 18)")
    frame = frame.with_columns(
        pl.col('team_form').fill_null(0.0),
        pl.col('opp_team_form').fill_null(0.0),
        # obs_data placeholder -- frame_to_D needs a y_col to build the
        # right-shaped array, but it's never read at prediction time
        # (pm.sample_posterior_predictive with predictions=True ignores
        # observed values entirely).
        pl.lit(0.0).alias('total_fantasy_points'),
    )
    return frame
 
 
# ----------------------------------------------------- 2. predictive draws
 
def forecast(model, model_name, idata, frame, scalers, idxs, season_offsets,
            team_form_scaler=None, seed=41029041):
    """Return (draws, frame) where draws is (n_draws, n_players), columns
    in `frame`'s row order.
 
    Rows of `draws` are paired posterior samples -- column i and column j come
    from the SAME posterior draw, so differences/sums across columns correctly
    propagate the model's correlation structure.
 
    model, model_name: the pm.Model object and its architecture name
        ('ar'/'hs'/'team'/'r2d2'/'bart') -- model_name picks which
        pm.Data keys models.predict_payload builds.
    idata: this model's FITTED idata (az.from_netcdf(...) of whatever
        fit_model.py saved).
    frame: from build_prediction_frame -- NOT yet scaled/encoded.
    scalers, idxs, season_offsets, team_form_scaler: fit_model.py's
        prep_data() return values for the SAME training run that produced
        `idata`. Passing a different run's stats here would silently
        rescale the prediction frame differently than the model was
        trained on."""
    D = models.frame_to_D(frame, scalers, idxs, team_form_scaler=team_form_scaler)
    payload = models.predict_payload(model_name, D, season_offsets)
 
    with model:
        pm.set_data(payload)
        ppc = pm.sample_posterior_predictive(
            idata, var_names=['y_obs'], predictions=True, random_seed=seed
        )
 
    # obs_dim is whichever un-named dimension PyMC/ArviZ assigned y_obs's
    # non-chain/draw axis (e.g. 'y_obs_dim_2') -- looked up at runtime
    # rather than hardcoded, since the exact name depends on how many
    # other dim_N axes already exist in the idata. newdata's row index
    # under that same name is what tidydraws.prediction_draws joins the
    # draws to -- this is the "join player_id/player back onto the draws"
    # step; frame's own columns just travel through the join.
    obs_dim = [d for d in ppc.predictions['y_obs'].dims if d not in ('chain', 'draw')][0]
    newdata = frame.with_row_index(obs_dim)
 
    tidy = td.prediction_draws(
        ppc, newdata=newdata, var_name='y_obs',
        idata_group='predictions', join_on=obs_dim,
    )
 
    # pivot tidy's (chain, draw, obs_dim, ...covariates, y_obs) rows back
    # into the (n_draws, n_players) matrix decision_engine.py's
    # enumeration/MILP machinery is written against -- pandas/xarray-free,
    # and column order is forced to match frame's row order explicitly
    # (pivot's own column order is not guaranteed to be sorted).
    wide = tidy.pivot(on=obs_dim, index=['chain', 'draw'], values='y_obs')
    ordered_cols = [str(i) for i in range(len(frame))]
    draws = wide.select(ordered_cols).to_numpy()
 
    return draws, frame
 
 
# ------------------------------------------------------ 3. player summaries
 
def player_summary(draws, frame, hdi_prob=0.8):
    """Per-player forecast table, sorted by expected points.
 
    floor/ceiling are the 10th/90th percentiles -- for fantasy these matter more
    than the mean. A high-floor player wins cash games; a high-ceiling player
    wins tournaments. Two players with the same mean are not interchangeable.
 
    Carries player_id alongside player (display name) -- every downstream
    function that matches/pools/dedupes players (optimize_lineup,
    lineup_distribution, decision_engine.py) does it on player_id."""
    hdi = az.hdi(draws[None, ...], hdi_prob=hdi_prob)['x']
    return (
        pl.DataFrame({
            'player_id': frame['player_id'],
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
 
    player_a/player_b: player_id or an unambiguous display name (see
    resolve_player).
 
    Do not compute this from the two marginal summaries -- that ignores the
    shared uncertainty (same team, same game environment, same population
    parameters) and will give you an overconfident answer."""
    ids = frame['player_id'].to_list()
    ia = ids.index(resolve_player(frame, player_a))
    ib = ids.index(resolve_player(frame, player_b))
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
    DFS needs a real knapsack solver, or the joint-scenario rules in
    decision_engine.py for any non-linear objective.)
 
    pool: player_ids or display names (see resolve_player); None means
        every player in `summary` is eligible.
    objective: 'expected' (default), 'floor_p10' for safe/cash lineups, or
        'ceiling_p90' for tournament lineups."""
    if pool is None:
        avail = summary
    else:
        pool_ids = [resolve_player(summary, p) for p in pool]
        avail = summary.filter(pl.col('player_id').is_in(pool_ids))
    avail = avail.sort(objective, descending=True)
 
    picks, used = [], set()
    for pos, n_slots in roster.items():
        chosen = (
            avail.filter((pl.col('position') == pos) & (~pl.col('player_id').is_in(list(used))))
            .head(n_slots)
        )
        for row in chosen.iter_rows(named=True):
            picks.append({**row, 'slot': pos})
            used.add(row['player_id'])
 
    flex = (
        avail.filter(
            pl.col('position').is_in(list(flex_eligible))
            & (~pl.col('player_id').is_in(list(used)))
        )
        .head(n_flex)
    )
    for row in flex.iter_rows(named=True):
        picks.append({**row, 'slot': 'FLEX'})
        used.add(row['player_id'])
 
    return pl.DataFrame(picks).select(
        ['slot', 'player_id', 'player', 'position', 'team', 'opp',
         'expected', 'floor_p10', 'ceiling_p90', 'sd']
    )
 
 
def lineup_distribution(draws, frame, lineup):
    """Distribution of the lineup TOTAL, summing paired draws.
 
    lineup: a DataFrame identifying the lineup's players, either by a
    player_id column (e.g. optimize_lineup's output) or a player column
    of display names (resolved via resolve_player -- errors if any name
    is ambiguous in `frame`).
 
    Summing per-player means gives the right expected total but tells you
    nothing about spread. Summing the draws row-wise keeps the correlations --
    e.g. a QB and his own WR move together, which widens the total's variance
    relative to a naive independence assumption."""
    ids = frame['player_id'].to_list()
    ref_col = 'player_id' if 'player_id' in lineup.columns else 'player'
    cols = [ids.index(resolve_player(frame, p)) for p in lineup[ref_col].to_list()]
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
    import fit_model
 
    MODEL_NAME = 'team'
    SHRUNK = False
    TARGET_SEASON, TARGET_WEEK = 2025, 18.0   # float -- see team_form.parquet's dtype
 
    # SAME prep as training -- scalers/idxs/season_offsets/team_form_scaler
    # must come from the run that produced the idata being loaded, or
    # predictions get silently rescaled against the wrong stats. Re-running
    # prep_data() re-derives them exactly (it's deterministic given the
    # same raw data/last_season/shrunk), which is simpler than trying to
    # persist scalers/idxs to disk alongside the .nc file.
    prep = fit_model.prep_data(shrunk=SHRUNK)
    build = fit_model.BUILDERS[MODEL_NAME]
    model = build(prep['D'], prep['coords'], prep['season_offsets'], prep['T'], prep['boundary_cols'],
                  prep['position_of_player'], prep['position_group_counts'], prep['week_grid_vals'])
 
    idata = az.from_netcdf(fit_model.default_out_path(MODEL_NAME, SHRUNK))
 
    raw = pl.read_parquet(fit_model.DATA_PATH)
    team_form = pl.read_parquet(fit_model.TEAM_FORM_PATH)
    frame = build_prediction_frame(raw, TARGET_SEASON, TARGET_WEEK, team_form)
 
    draws, frame = forecast(
        model, MODEL_NAME, idata, frame,
        scalers=prep['scalers'], idxs=prep['idxs'], season_offsets=prep['season_offsets'],
        team_form_scaler=prep['team_form_scaler'],
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