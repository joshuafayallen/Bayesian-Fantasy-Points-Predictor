"""
Standalone team-ability model: a Bradley-Terry-flavored "team strength" signal
built from actual game scores (not fantasy points), estimated as a batched
local-level Kalman filter across all 32 teams via pymc_extras.statespace.

Workflow
--------
1. game-level table (one row per game, real scores) from nflreadpy schedules
2. reshape to team-perspective long format (2 rows per game)
3. fixed-effects adjustment for home-field advantage + rest differential
   (structural.Regression can't take per-team-varying exogenous data, so this
   is netted out before the state-space step instead of inside it)
4. pivot to a (week, team) wide matrix, NaN on byes
5. fit LevelTrend(order=1, innovations_order=1) across all 32 teams at once,
   with save_kalman_filter_outputs_in_idata=True so both the leakage-safe
   `filtered_states` ("ability through week i") and `smoothed_states` come
   out of a single fit.

This produces team_strength[team, week] as a standalone precomputed feature
-- it does not get estimated jointly inside the player-level model, which
sidesteps the team_effect / f_team_form collapsing-to-zero problem we hit
earlier.
"""

import nflreadpy as nfl
import numpy as np
import polars as pl
import pymc as pm
import pytensor.tensor as pt

from pymc_extras.statespace import structural as st

from data_prep import TEAM_ALIASES

SCHEDULE_TEAM_ALIASES = {**TEAM_ALIASES, "STL": "LA"}


# ---------------------------------------------------------------------------
# 1. game-level table
# ---------------------------------------------------------------------------
def load_games(season_lo=2006, season_hi=2025):
    return (
        nfl.load_schedules(seasons=True)
        .filter(
            (pl.col("season").is_between(season_lo, season_hi))
            & (pl.col("game_type") == "REG")
        )
        .select(
            "game_id", "season", "week",
            "home_team", "away_team", "home_score", "away_score",
            "home_rest", "away_rest",
        )
        .unique(subset=["game_id"])
        .with_columns(
            pl.col("home_team").replace(SCHEDULE_TEAM_ALIASES),
            pl.col("away_team").replace(SCHEDULE_TEAM_ALIASES),
        )
        .drop_nulls(["home_score", "away_score"])  # drop any unplayed/future games
    )


# ---------------------------------------------------------------------------
# 2. team-perspective long format
# ---------------------------------------------------------------------------
def to_team_long(games):
    home = games.select(
        pl.col("season"), pl.col("week"),
        pl.col("home_team").alias("team"),
        (pl.col("home_score") - pl.col("away_score")).cast(pl.Float64).alias("margin"),
        (pl.col("home_rest") - pl.col("away_rest")).cast(pl.Float64).alias("rest_diff"),
        pl.lit(1.0).alias("is_home"),
    )
    away = games.select(
        pl.col("season"), pl.col("week"),
        pl.col("away_team").alias("team"),
        (pl.col("away_score") - pl.col("home_score")).cast(pl.Float64).alias("margin"),
        (pl.col("away_rest") - pl.col("home_rest")).cast(pl.Float64).alias("rest_diff"),
        pl.lit(0.0).alias("is_home"),
    )
    return pl.concat([home, away]).sort(["season", "week", "team"])


# ---------------------------------------------------------------------------
# 3. fixed-effects adjustment for home-field advantage + rest
# ---------------------------------------------------------------------------
def adjust_for_home_and_rest(team_long):
    X = team_long.select("is_home", "rest_diff").to_numpy()
    X = np.column_stack([np.ones(len(X)), X])   # intercept, is_home, rest_diff
    y = team_long["margin"].to_numpy()

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    print(f"home_adv ~ {beta[1]:.2f} pts, beta_rest ~ {beta[2]:.3f} pts/day")

    return team_long.with_columns(
        pl.Series("adjusted_margin", y - fitted)
    )


# ---------------------------------------------------------------------------
# 4. pivot to wide (week, team) matrix, global week clock, NaN on byes
# ---------------------------------------------------------------------------
def to_wide_matrix(team_long, teams):
    global_week = (
        team_long.select("season", "week").unique().sort(["season", "week"])
        .with_row_index("global_week")
    )
    tl = team_long.join(global_week, on=["season", "week"])

    n_week = global_week.height
    wide = np.full((n_week, len(teams)), np.nan)
    team_pos = {t: i for i, t in enumerate(teams)}

    for row in tl.iter_rows(named=True):
        wide[row["global_week"], team_pos[row["team"]]] = row["adjusted_margin"]

    return wide, global_week


# ---------------------------------------------------------------------------
# 5. batched local-level state-space fit
# ---------------------------------------------------------------------------
def build_team_strength_model(wide, teams):
    # LevelTrend alone has zero observation noise -- nothing separates
    # genuine week-to-week ability drift from ordinary single-game score
    # variance (which is large, routinely +/-15 pts for a team of constant
    # true strength). Without MeasurementError, sigma_level_trend is forced
    # to absorb ALL of that raw variance, and filtered/smoothed both just
    # track the raw margin exactly. MeasurementError gives the model an
    # explicit obs_cov / H term to soak up game-level noise, so the level
    # state can actually smooth instead of interpolating every observation.
    ss_mod = (
        st.LevelTrend(order=1, innovations_order=1, observed_state_names=teams)
        + st.MeasurementError(observed_state_names=teams)
    ).build()

    with pm.Model(coords=ss_mod.coords) as team_strength_mod:
        pm.Normal(
            "initial_level_trend", 0, 10,
            dims=ss_mod.param_dims["initial_level_trend"],
        )
        pm.HalfNormal(
            "sigma_level_trend", 3.0,
            dims=ss_mod.param_dims["sigma_level_trend"],
        )
        # observation/game-level noise -- expect this to come out MUCH
        # larger than sigma_level_trend if real NFL score margins are as
        # noisy as they look (single-game sd is typically 10-13+ points).
        # Note: parameter name defaults to the component's class name since
        # we didn't pass name= to MeasurementError() -- check
        # ss_mod.param_dims if this ever changes.
        pm.Exponential(
            "sigma_MeasurementError", 1.0,
            dims=ss_mod.param_dims["sigma_MeasurementError"],
        )
        # required by every structural model -- initial state covariance.
        # pt.eye * 10 (diffuse-ish prior) matches the pattern in pymc_extras'
        # own structural/core.py docstring; tune if the level starts off
        # too unconstrained/too tight relative to the data scale.
        pm.Deterministic(
            "P0", pt.eye(ss_mod.k_states) * 10.0,
            dims=ss_mod.param_dims["P0"],
        )
        ss_mod.build_statespace_graph(
            wide,
            missing_fill_value=-9999.0,
            save_kalman_filter_outputs_in_idata=True,
        )

    return team_strength_mod, ss_mod


# ---------------------------------------------------------------------------
# 6. extract filtered ("through week i") team strength as a plain feature,
#    keyed on (season, week, team) so it can be left-joined straight onto
#    train/test in polars -- no need to reproduce global_week indexing
#    inside the player-level PyMC model at all.
# ---------------------------------------------------------------------------
def extract_filtered_team_form(idata, global_week):
    """Posterior-mean filtered team strength per (season, week, team).

    Uses filtered_states, not smoothed_states -- filtered at week i only
    conditions on data through week i, so this is safe to use as a
    pre-game feature in the player-level model with no leakage. This is a
    point-estimate plug-in (posterior mean only); it does not propagate
    the state-space model's own posterior uncertainty into the downstream
    model, which is a real simplification worth flagging, not just a detail.
    """
    filtered = idata.posterior["filtered_states"]
    state_labels = filtered.coords["state"].values
    level_labels = [s for s in state_labels if s.startswith("level[")]

    filtered_level = filtered.sel(state=level_labels).mean(("chain", "draw"))
    df = filtered_level.to_dataframe(name="team_form").reset_index()
    df["team"] = df["state"].str.extract(r"level\[(.*)\]")
    df = df.drop(columns="state").rename(columns={"time": "global_week"})

    team_form = pl.from_pandas(df)

    return (
        team_form.join(
            global_week.select("season", "week", "global_week"),
            on="global_week",
        )
        .select("season", "week", "team", "team_form")
    )


def attach_team_form(frame, team_form):
    """Left-join team_form onto a train/test frame on (season, week, team).

    Left join (not inner) so any (season, week, team) combination missing
    from the state-space fit -- shouldn't happen given both are built off
    the same schedule, but worth not silently dropping rows over -- shows
    up as an explicit null rather than a disappearing row.
    """
    out = frame.join(team_form, on=["season", "week", "team"], how="left")
    n_missing = out["team_form"].null_count()
    if n_missing:
        print(f"warning: {n_missing} rows had no matching team_form value")
    return out


if __name__ == "__main__":
    games = load_games()
    print(f"games: {games.height} rows (expect ~5000-5500 for 2006-2025 REG season)")

    team_long = adjust_for_home_and_rest(to_team_long(games))

    teams = sorted(team_long["team"].unique().to_list())
    print(f"teams: {len(teams)} (expect 32)")

    wide, global_week = to_wide_matrix(team_long, teams)
    print(f"wide matrix: {wide.shape}, {np.isnan(wide).mean():.1%} missing (byes)")

    team_strength_mod, ss_mod = build_team_strength_model(wide, teams)

    with team_strength_mod:
        idata = pm.sample(nuts_sampler="nutpie")

    idata.to_netcdf("model-nc/team_strength.nc")

    # persist the extracted feature separately from the raw idata -- ff-ar.py
    # (and anything else downstream) should never need to touch pymc_extras
    # or re-run this fit just to read team_form back.
    team_form = extract_filtered_team_form(idata, global_week)
    team_form.write_parquet("processed-data/team_form.parquet")
    print(f"wrote {team_form.height} (season, week, team) team_form rows")
