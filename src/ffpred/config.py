"""Constants, paths and model configurations shared by every module.

Only the two production models live here: ``het_gp_vegas`` (RB/WR/TE) and
``het_gp_qb_vegas`` (QB). They are the het_gp configs from ``src/models.py``
with the signed Vegas lines (``team_spread``, ``vegas_team_total``) in place
of the home-perspective ``spread_line``; see the project doc
``claude/team-context-ablation.md`` for why.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------- paths
# src-dev/ffpred/config.py -> project root is two levels up from this package.
PROJECT_ROOT = Path(os.environ.get('FFPRED_ROOT', Path(__file__).resolve().parents[2]))


@dataclass(frozen=True)
class Paths:
    root: Path = PROJECT_ROOT

    @property
    def processed(self) -> Path:
        return self.root / 'processed-data' / 'ff-processed.parquet'

    @property
    def team_form(self) -> Path:
        return self.root / 'processed-data' / 'team_form.parquet'

    @property
    def raw_dir(self) -> Path:
        return self.root / 'raw-data'

    @property
    def results_dir(self) -> Path:
        return self.root / 'results'

    @property
    def forecasts_dir(self) -> Path:
        return self.root / 'forecasts'


PATHS = Paths()

# ---------------------------------------------------------------- calendar
WINDOW_SEASONS = 3      # seasons of history fed to each fit (target season + 2 prior)
MAX_WEEK = 18           # regular-season weeks (17 through 2020, 18 since 2021)
MIN_TRAIN_ROWS = 200

# ------------------------------------------------------------------ lineup
ROSTER = {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1}
FLEX_ELIGIBLE = ('RB', 'WR', 'TE')
N_FLEX = 1

# --------------------------------------------------------------- features
BINARY_VARS = ('is_grass', 'is_indoors', 'is_home', 'div_game', 'era')
VEGAS_COLS = ('team_spread', 'vegas_team_total')

# Columns the modelling code reads from a processed / upcoming frame.
ID_COLS = ('player_id', 'player', 'position', 'team', 'opp_team', 'season', 'week')
TARGET_COL = 'total_fantasy_points'


@dataclass(frozen=True)
class HetConfig:
    """One het_gp model: which positions it covers and how it is shaped."""
    name: str
    positions: tuple[str, ...]
    std_cols: tuple[str, ...]        # standardized linear covariates
    time_col: str                    # 'tenure' (skill) or 'age' (QB)
    asymmetric_bump: bool            # separate rise / decline widths (QB)
    noise_gp: bool                   # GP over time on log sigma (skill) vs one sigma (QB)
    t_peak: tuple[float, float]      # Normal(mean, sd) prior on the aging peak
    season_ell: tuple[float, float]  # maxent InverseGamma range for the season-GP lengthscale
    noise_ell: tuple[float, float] = (4.0, 8.0)

    @property
    def required_cols(self) -> tuple[str, ...]:
        return (*ID_COLS, self.time_col, *self.std_cols, *BINARY_VARS, 'team_form', 'opp_team_form')


HET_GP_VEGAS = HetConfig(
    name='het_gp_vegas',
    positions=('RB', 'TE', 'WR'),
    std_cols=('roll_avg_points_allowed', 'lag_yards_per_rush_attempt', 'lag_yards_per_target',
              'lag_avg_depth_of_target', 'lag_target_share', 'lag_rush_share', 'temp', 'wind',
              'games_missed_ytd', 'lag_offense_pct', 'lag_st_pct', *VEGAS_COLS),
    time_col='tenure', asymmetric_bump=False, noise_gp=True,
    t_peak=(4.0, 1.25), season_ell=(1.5, 3.0), noise_ell=(4.0, 8.0),
)

HET_GP_QB_VEGAS = HetConfig(
    name='het_gp_qb_vegas',
    positions=('QB',),
    std_cols=('roll_avg_points_allowed', 'lag_yards_per_rush_attempt', 'lag_yards_per_pass_attempt',
              'lag_avg_depth_of_target', 'lag_rush_share', 'temp', 'wind', 'games_missed_ytd',
              'lag_offense_pct', *VEGAS_COLS),
    time_col='age', asymmetric_bump=True, noise_gp=False,
    t_peak=(28.5, 3.0), season_ell=(1.0, 4.0),
)

MODELS: dict[str, HetConfig] = {c.name: c for c in (HET_GP_VEGAS, HET_GP_QB_VEGAS)}
# Every fantasy position is covered by exactly one model.
MODEL_FOR_POSITION = {p: c for c in MODELS.values() for p in c.positions}


@dataclass(frozen=True)
class SamplerSettings:
    """How to fit. NUTS via nutpie, 4 chains x 1000 draws / 1000 tune, is the
    production default (the team-context ablation used 2 x 500 / 500).
    Chains run in parallel (cores = chains). ADVI is for smoke runs only."""
    method: str = 'nuts'             # 'nuts' | 'advi'
    nuts_sampler: str = 'nutpie'     # 'nutpie' | 'pymc'
    draws: int = 1000
    tune: int = 1000
    chains: int = 4
    target_accept: float = 0.95
    advi_n: int = 20_000
    advi_draws: int = 500
    seed: int = 0
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.method not in ('nuts', 'advi'):
            raise ValueError(f"method must be 'nuts' or 'advi', got {self.method!r}")
        if self.nuts_sampler not in ('nutpie', 'pymc'):
            raise ValueError(f"nuts_sampler must be 'nutpie' or 'pymc', got {self.nuts_sampler!r}")

    @property
    def n_draws(self) -> int:
        return self.draws * self.chains if self.method == 'nuts' else self.advi_draws
