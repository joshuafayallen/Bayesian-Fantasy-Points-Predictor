import nflreadpy as nfl
import polars as pl
import polars.selectors as cs

keep_vegas = ['game_id', 'season','away_team', 'away_score', 'home_team', 'home_score', 'away_rest', 'home_rest', 'total_line', 'spread_line', 'is_grass', 'is_indoors', 'div_game', 'wind_clean', 'temp_clean']



raw_vegas = (
    nfl.load_schedules(seasons = True).filter(
    (pl.col('season').is_between(2006, 2025)) &
    (pl.col('game_type') == 'REG')) 
        .with_columns(
        pl.when(pl.col('surface').str.contains('grass'))
        .then(1)
        .otherwise(0)
        .alias('is_grass'), 
        pl.when(
            pl.col('roof').is_in(['closed', 'dome'])
        )
        .then(1)
        .otherwise(0)
        .alias('is_indoors'))
    .with_columns(
        pl.when(pl.col('is_indoors') == 1)
        .then(pl.lit(68))
        .when(pl.col('temp').is_null())
        .then(pl.lit(60))
        .otherwise(pl.col('temp'))
        .alias('temp_clean'), 
        pl.when(pl.col('is_indoors') == 1)
        .then(pl.lit(0))
        .when(pl.col('wind').is_null())
        .then(pl.lit(8))
        ## there were two values with 70 mph miles per hour
        ## so we are just going to hand correct them
        ## the data for those games were found on PFR 
        .when(pl.col('game_id') == '2016_13_NYG_PIT')
        .then(7)
        .when(pl.col('game_id') == '2008_02_TEN_CIN')
        .then(13)
        .otherwise(pl.col('wind'))
        .alias('wind_clean')
    )
    .select(pl.col(keep_vegas))
    .rename({'temp_clean': 'temp', 'wind_clean': 'wind'})
)

raw_vegas.filter(pl.col('wind') == pl.col('wind').max())

pos = ['TE', 'RB', 'FB', 'QB', 'WR']


raw_rosters = (
    nfl.load_rosters(seasons = [2006, 2007, 2008, 2009, 2010,
    2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021,
    2022, 2023, 2024, 2025])
    .filter(pl.col('position').is_in(pos))
    .with_columns(
        pl.col("birth_date").dt.year().alias('birth_year')
    )
    .sort(['full_name', 'season'])
    .fill_null(strategy = 'forward')
    .with_columns(
        (pl.col('season') - pl.col('birth_year')).alias('age'), 
        (pl.col('season') - pl.col("entry_year")).alias("tenure")
    )
    .select(pl.col('full_name','team', 'season' ,'birth_date' ,'gsis_id', 'tenure', 'age'))
    .filter((pl.col("gsis_id").is_not_null())
    )
    .rename({'gsis_id': 'player_id'})
    .filter(pl.col("full_name") != 'George Farmer') # it seems like the entry year is wrong
    .drop('full_name')
    )




# lets keep some of the diff numbers. Those are just over expectation
# 
raw_fantasy = (
    nfl.load_ff_opportunity(seasons = True, stat_type='weekly')
    .select(
        pl.col('player_id', 'full_name', 'position', 'total_fantasy_points', 'total_fantasy_points_diff_team', 'game_id', 'receptions', 'pass_completions', 'season', 'week'), 
        cs.ends_with('gained'), 
        cs.ends_with('points_diff'),
        cs.ends_with('attempt'),
        pl.col('rec_air_yards')
        
    )
    .filter((pl.col('player_id').is_not_null()) & 
            (pl.col('position').is_in(['TE', 'RB', 'WR', 'QB', 'FB']))) 
    .with_columns(
        pl.col('season').cast(pl.Int32)
    )
)


# raw_rosters can carry more than one row per (player_id, season) -- not
# just for players traded mid-season, but sometimes for the same team too
# (practice-squad/IR roster-snapshot noise). Joined on [player_id, season]
# alone, EVERY one of those candidate rows fans out across that player's
# entire season of game rows, not just the weeks it actually applies to
# (e.g. Randy Moss's 2010 NE/MIN/TEN roster stints all attach to every one
# of his weeks that season, not just the ones he actually played for each
# team). game_id embeds the two teams that actually played that game
# ({season}_{week}_{away}_{home}), so filtering to the roster-team candidate
# that's actually one of those two teams resolves it unambiguously, however
# many spurious candidates raw_rosters produced.
join_covariates = (
    raw_fantasy
    .join(
        raw_rosters,
        on = ['player_id' ,'season']
    )
    .with_columns(
        pl.col('game_id').str.split('_').list.get(2).alias('_away_team'),
        pl.col('game_id').str.split('_').list.get(3).alias('_home_team'),
    )
    .filter(
        (pl.col('team') == pl.col('_away_team')) | (pl.col('team') == pl.col('_home_team'))
    )
    .drop('_away_team', '_home_team')
    # two remaining edge cases the team-match filter can't resolve on its
    # own: (a) a player traded to a team that happens to face their old team
    # the very next week -- both candidates are legitimately "one of the two
    # teams in this game" (e.g. Randy Moss, MIN @ NE, week 8 2010, the week
    # after his trade); (b) raw_rosters occasionally has a literal exact
    # duplicate row for the same (player_id, season, team) with nothing to
    # disambiguate (e.g. Fred Taylor, JAX, week 1 2006). Both are rare and,
    # for (a), genuinely ambiguous from this data alone -- just keep one.
    .unique(subset=['player_id', 'season', 'week'], keep='first')
    .sort(['full_name', 'season', 'week'])
    .join(
        raw_vegas,
        on = 'game_id'
    ))


make_covariate = (
    join_covariates    
    .with_columns(
        pl.when(
            pl.col('team') == pl.col('home_team')
        )
        .then(1)
        .otherwise(0)
        .alias('is_home'), 
        pl.when(pl.col('team') !=  pl.col('home_team'))
        .then(pl.col('home_team'))
        .otherwise(pl.col('away_team'))
        .alias('opp_team')
    )
    .with_columns(
        pl.when(pl.col('is_home') == 1)
        .then(pl.col('home_rest'))
        .otherwise(pl.col('away_rest'))
        .alias('player_rest'), 
        pl.when(
            pl.col('team') != pl.col('home_team')
        )
        .then(pl.col('home_rest'))
        .otherwise(pl.col('away_rest'))
        .alias("opponent_rest"),
        pl.when(pl.col('is_home') == 1)
        .then(pl.col('home_score'))
        .otherwise(pl.col('away_score'))
        .alias('team_score'), 
        pl.when(
            pl.col('team') != pl.col('home_team')
        )
        .then(pl.col('home_score'))
        .otherwise(pl.col('away_score'))
        .alias("opponent_score")
    )
    .with_columns(
        (pl.col('player_rest') - pl.col("opponent_rest")).alias('rest_diff'), 
        # make totals for usage rates we need this at the game team level 
        cs.ends_with('attempt').sum().over(['team', 'game_id']).name.suffix('_total'), 
        pl.col('pass_completions').sum().over(['team', 'game_id']).alias('total_completions')
    )
    .with_columns(
        pl.when(pl.col('rush_attempt') > 0)
        .then(pl.col('rush_yards_gained') / pl.col('rush_attempt'))
        .otherwise(None)
        .alias('yards_per_rush_attempt'),
        pl.when(pl.col('rec_attempt_total') > 0)
        .then(pl.col('rec_attempt') / pl.col('rec_attempt_total'))
        .otherwise(None)
        .alias("target_share"),
        pl.when(pl.col('rush_attempt_total') > 0)
        .then(pl.col('rush_attempt')/pl.col('rush_attempt_total'))
        .otherwise(None)
        .alias('rush_share'),
        pl.when(pl.col('rec_attempt') > 0)
        .then(pl.col('rec_yards_gained') / pl.col('rec_attempt'))
        .otherwise(None)
        .alias('yards_per_target'),
        pl.when(pl.col('pass_attempt_total') > 0)
        .then(pl.col('pass_yards_gained') / pl.col('pass_attempt_total'))
        .otherwise(None)
        .alias('yards_per_pass_attempt'),
        pl.when(pl.col('rec_attempt') > 0)
        .then(pl.col('rec_air_yards') / pl.col('rec_attempt'))
        .otherwise(None)
        .alias('avg_depth_of_target'), 
        
    )
)



weekly = (
    make_covariate
    .group_by(["opp_team", "season", "week", "position"])
    .agg((-pl.col("total_fantasy_points_diff")).sum().alias("points_allowed"))
    .sort(["opp_team", "season", "position", "week"])
    .with_columns(
        pl.col('points_allowed')
        .shift(1)                                    # exclude the current week's own outcome
        .rolling_mean(window_size=3, min_samples=1)
        .over(['opp_team', 'season', 'position'])
        .alias('roll_avg_points_allowed')
        )
)


team_mapping = {
    'SL': 'LA',
    'ARZ': 'ARI',
    'OAK': 'LV',
    'SD': 'LAC',
    'BLT': 'BAL',
    'CLV': 'CLE',
    'HST': 'HOU',
    'STL': 'LA'
}

bring_back_in = (
    make_covariate
    .join( 
        weekly,
    on = ['opp_team', 'season', 'week', 'position'], 
    how = 'left'
    ).drop('season_right')
    .fill_nan(0)
    .with_columns(
        pl.when(pl.col('season') >= 2018)
        .then(1)
        .otherwise(0)
        .alias('era'), 
        pl.concat_str(
            [
                pl.col('player_id'), 
                pl.lit('_'), 
                pl.col('season')
            ]
        ).alias("player_season"), 
        pl.concat_str(
            [pl.col('team'), 
            pl.lit('_'), 
            pl.col('season')]
        )
        .alias('team_season')
        
    )
    .rename({'full_name' : 'player'})
    .with_columns(
        pl.col('team').replace(team_mapping).name.keep(), 
        pl.col('opp_team').replace(team_mapping).name.keep(), 
        pl.col('home_team').replace(team_mapping).name.keep(), 
        pl.col('away_team').replace(team_mapping).name.keep()
    )
    .with_columns(
        pl.col('yards_per_rush_attempt',
                'target_share',
                'yards_per_target',
                'yards_per_pass_attempt',
                'avg_depth_of_target', 'rush_share').shift(1).over('player_id').name.prefix('lag_')
    )
    .fill_null(0)
)

team_games = (
    bring_back_in
    .select(
        pl.col('team', 'season', 'week')
    )
    .unique()
    .sort(['team', 'season', 'week'])
    .with_columns(
        (pl.col('week').cum_count().over(['team', 'season']) - 1).alias('team_games_played_ytd')
    )
)
snaps = nfl.load_snap_counts(True)
ids = pl.read_csv('https://raw.githubusercontent.com/dynastyprocess/data/refs/heads/master/files/db_playerids.csv', null_values='NA')

ids.write_csv('raw-data/nfl-playerids.csv')

add_ids = (
    snaps
    .join(ids, left_on = ['pfr_player_id'], right_on=['pfr_id'])
    .filter(pl.col('position').is_in(['TE', 'RB', 'WR', 'QB', 'FB']))
    .select(pl.col('game_id', 'gsis_id', 'offense_pct' ,'st_pct'))
    
)


with_availability = (
    bring_back_in
    .sort(['player_id', 'season', 'week'])
    .with_columns(
        # this player's own appearances so far this season, before this week
        (pl.col('week').cum_count().over(['player_id', 'season']) - 1).alias('player_games_played_ytd')
    )
    .join(team_games, on=['team', 'season', 'week'], how='left')
    .with_columns(
        # 0 = hasn't missed a single team game this season (fully available)
        # >0 = missed that many of the team's games -- the injury-return proxy
        #
        # cum_count() returns an unsigned int (u32), so this subtraction
        # underflows to ~4.29 billion instead of going negative whenever it's
        # negative -- cast to a signed type first, and clip at 0 as a
        # belt-and-suspenders guard for any other edge case (e.g. a
        # practice-squad promotion mid-week) beyond the join fix above.
        (
            pl.col('team_games_played_ytd').cast(pl.Int32)
            - pl.col('player_games_played_ytd').cast(pl.Int32)
        ).clip(lower_bound=0, upper_bound = 7).alias('games_missed_ytd')
    
    )
    .join(
        add_ids, left_on = ['game_id', 'player_id'], right_on = ['game_id', 'gsis_id']
    )
    .with_columns(
        pl.col("offense_pct", 'st_pct').shift(1).over('player_id').name.prefix('lag_')
    )
    .fill_null(0)

)


with_availability.write_parquet('processed-data/ff-processed.parquet')


