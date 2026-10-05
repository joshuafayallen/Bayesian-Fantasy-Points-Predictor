import os
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ffpred.config import PATHS  # noqa: E402

REAL_DATA = Path(os.environ.get('FFPRED_DATA', PATHS.processed))
REAL_TEAM_FORM = Path(os.environ.get('FFPRED_TEAM_FORM', PATHS.team_form))
needs_real_data = pytest.mark.skipif(not (REAL_DATA.exists() and REAL_TEAM_FORM.exists()),
                                     reason='processed data / team_form not found (set FFPRED_DATA / FFPRED_TEAM_FORM)')
slow = pytest.mark.skipif(os.environ.get('RUN_SLOW_TESTS') != '1', reason='set RUN_SLOW_TESTS=1')

TEAMS = ['AAA', 'BBB', 'CCC', 'DDD']
DEPTH = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1}


def make_league(seasons=(2021, 2022, 2023), weeks=6, seed=0) -> pl.DataFrame:
    """A small processed-data-shaped frame: 4 teams, round-robin-ish games,
    every column the models and the upcoming-frame builder read."""
    rng = np.random.default_rng(seed)
    players = []
    for t in TEAMS:
        for pos, k in DEPTH.items():
            for i in range(k):
                players.append(dict(player_id=f'{t}-{pos}{i}', player=f'{t} {pos}{i}', position=pos, team=t,
                                    skill=rng.normal(0, 3), entry=2019 + i, birth=1995 + i))
    rows = []
    for s in seasons:
        for w in range(1, weeks + 1):
            order = TEAMS[w % 4:] + TEAMS[:w % 4]
            games = [(order[0], order[1]), (order[2], order[3])]
            for home, away in games:
                gid = f'{s}_{w:02d}_{away}_{home}'
                spread, total = rng.normal(0, 4), rng.normal(45, 4)
                grass, indoors = int(rng.random() < .5), int(rng.random() < .3)
                for p in players:
                    if p['team'] not in (home, away):
                        continue
                    is_home = int(p['team'] == home)
                    base = {'QB': 18, 'RB': 10, 'WR': 9, 'TE': 6}[p['position']]
                    rows.append(dict(
                        player_id=p['player_id'], player=p['player'], position=p['position'], team=p['team'],
                        opp_team=away if is_home else home, season=s, week=float(w), game_id=gid,
                        home_team=home, away_team=away, home_score=24, away_score=20, home_rest=7, away_rest=7,
                        total_fantasy_points=float(max(-2, base + p['skill'] + rng.normal(0, 5))),
                        tenure=s - p['entry'], age=s - p['birth'], spread_line=spread, total_line=total,
                        is_home=is_home, is_grass=grass, is_indoors=indoors,
                        div_game=0, era=1, temp=60, wind=8,
                        yards_per_rush_attempt=rng.gamma(4, 1), target_share=rng.beta(2, 8),
                        yards_per_target=rng.gamma(8, 1), yards_per_pass_attempt=rng.gamma(7, 1),
                        avg_depth_of_target=rng.gamma(8, 1), rush_share=rng.beta(2, 8),
                        offense_pct=rng.beta(8, 3), st_pct=rng.beta(1, 8)))
    df = pl.DataFrame(rows).with_columns(pl.col('season').cast(pl.Int32), pl.col('tenure', 'age').cast(pl.Int32))
    # points allowed by opponent/position, then the same derived features as the data build
    allowed = (df.group_by('opp_team', 'season', 'week', 'position')
               .agg(pl.col('total_fantasy_points').sum().alias('points_allowed'))
               .sort('opp_team', 'season', 'position', 'week')
               .with_columns(pl.col('points_allowed').shift(1).rolling_mean(3, min_samples=1)
                             .over('opp_team', 'season', 'position').alias('roll_avg_points_allowed')))
    df = df.join(allowed, on=['opp_team', 'season', 'week', 'position'], how='left').sort('player_id', 'season', 'week')
    lags = {'yards_per_rush_attempt': 'lag_yards_per_rush_attempt', 'target_share': 'lag_target_share',
            'yards_per_target': 'lag_yards_per_target', 'yards_per_pass_attempt': 'lag_yards_per_pass_attempt',
            'avg_depth_of_target': 'lag_avg_depth_of_target', 'rush_share': 'lag_rush_share',
            'offense_pct': 'lag_offense_pct', 'st_pct': 'lag_st_pct'}
    df = df.with_columns([pl.col(s).shift(1).over('player_id').alias(d) for s, d in lags.items()])
    gp = (df.with_columns(pl.col('week').cum_count().over('player_id', 'season').alias('_pg'),
                          (pl.col('week').rank('dense').over('team', 'season')).alias('_tg')))
    df = gp.with_columns((pl.col('_tg').cast(pl.Int32) - pl.col('_pg').cast(pl.Int32)).clip(0, 7)
                         .alias('games_missed_ytd')).drop('_pg', '_tg').fill_null(0)
    return df.with_columns(pl.concat_str([pl.col('player_id'), pl.lit('_'), pl.col('season')]).alias('player_season'))


def make_team_form(df: pl.DataFrame, seed=1) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    tw = df.select('season', 'week', 'team').unique().sort('season', 'week', 'team')
    return tw.with_columns(pl.Series('team_form', rng.normal(0, 3, tw.height)))


@pytest.fixture(scope='session')
def league():
    from ffpred.data import add_vegas_features
    return add_vegas_features(make_league())


@pytest.fixture(scope='session')
def league_team_form(league):
    return make_team_form(league)
