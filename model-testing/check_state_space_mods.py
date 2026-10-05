import sys; sys.path.insert(0, 'src')
import polars as pl
from evaluate_team_models import paired_diff, summarise

cutoffs = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
rows = pl.concat([
    pl.read_parquet(f'results/team_models_thru{c}_rows.parquet')
      .filter(pl.col('season') == c + 1)            # one season ahead only -> each game counted once
      .with_columns(pl.lit(c).alias('cutoff'))
    for c in cutoffs
])



print(summarise(rows))                                # overall, pooled
print(summarise(rows.select(pl.col('model', 'phase','mean' ,'lpd' ,'crps', 'cutoff', 'margin')), ['phase']))                     # does the division gain sit in weeks 1-4?
print(pl.concat([paired_diff(rows, 'division', 'new_offdef').select(pl.col('width50')),
                 paired_diff(rows, 'new_offdef', 'old_leveltrend')]))





# consistency: the division - offdef difference for each held-out season
print(pl.concat([paired_diff(rows.filter(pl.col('cutoff') == c), 'division', 'new_offdef')
                   .with_columns(pl.lit(c + 1).alias('season')) for c in cutoffs]))
