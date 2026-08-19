"""
Build plausible historical league rosters from real FantasyPros expert
consensus rankings (ECR), via nflreadpy.load_ff_rankings -- actual ADP
instead of either a one-off scraped mock (mock_league.py, 2025 only) or a
trailing-mean proxy (league_validate.draft_league, which can accidentally
draft long-retired players -- see that module's lookback_seasons fix).

IMPORTANT / UNVERIFIED: nflreadpy.load_ff_rankings pulls from
DynastyProcess.com's ECR archive (github.com/dynastyprocess/data). That
network path is BLOCKED in this sandbox, so none of this has been run
against real data -- it's written to the documented schema
(https://nflreadr.nflverse.com/reference/load_ff_rankings.html), not
verified against it. Before trusting a draft built from this, run
`describe_adp_archive()` on your own machine first and eyeball the output:
    - how far back `scrape_date` actually goes (the archive is finite;
      I have no way to know its start date from here)
    - that `pos` contains QB/RB/WR/TE as expected
    - that `player` names are joinable to ff-processed.parquet's `player`
      column (reuses mock_league.resolve_names' Jr./Sr./II/III suffix
      handling, but FantasyPros' naming conventions weren't cross-checked
      against nflreadr's here)
`load_historical_adp` raises a clear, loud error if a season's preseason
window comes up empty rather than silently returning nothing -- if you
hit that, the archive doesn't reach that season and you should fall back
to league_validate.draft_league instead.

Two things this does differently from the earlier drafts:

    1. draft_by_adp     -- ONE overall snake draft ordered strictly by
                           ascending ECR (best-player-available), not
                           independent per-position mini-drafts. This is
                           what "going strictly off ADP" means: a QB rated
                           8th overall goes off the board before a WR
                           rated 15th, same as it would in a real draft.
    2. fill_bench_depth -- after the ADP snake draft, tops up any position
                           below a minimum bench depth by handing out the
                           next-best remaining ADP players at that
                           position. This directly targets the "team only
                           owns 1 QB, a bye strands it" failure mode that
                           showed up validating both the synthetic draft
                           and the real 2025 Yahoo mock -- a real manager
                           would stream a replacement; this just rosters
                           one up front so decision_engine always has a
                           legal lineup to choose from.
"""

import polars as pl

from mock_league import resolve_names

# Enough bench depth per position to survive a bye without stranding a
# required slot (decision_engine.ROSTER needs QB1/RB2/WR2/TE1 + 1 FLEX from
# RB/WR/TE) -- tune these based on what describe_adp_archive/validation
# runs show you actually need.
MIN_BENCH_DEPTH = {'QB': 2, 'RB': 4, 'WR': 4, 'TE': 2}
DEFAULT_ROSTER_SIZE = 14   # matches league_validate.DEFAULT_ROSTER_DEPTH's total


def describe_adp_archive():
    """Run this FIRST, on a machine with real network access, before
    trusting anything else in this module. Prints the archive's actual
    date coverage and a couple of sanity checks so you know what you're
    working with rather than assuming.
    """
    import nflreadpy as nfl
    all_ecr = nfl.load_ff_rankings(type='all')
    print(f"rows: {all_ecr.height}")
    print(f"columns: {all_ecr.columns}")
    if 'scrape_date' in all_ecr.columns:
        print(f"scrape_date range: {all_ecr['scrape_date'].min()} .. {all_ecr['scrape_date'].max()}")
    if 'pos' in all_ecr.columns:
        print(f"position value counts:\n{all_ecr['pos'].value_counts().sort('pos')}")
    if 'ecr_type' in all_ecr.columns:
        print(f"ecr_type value counts:\n{all_ecr['ecr_type'].value_counts()}")
    return all_ecr


def load_historical_adp(season, preseason_window=('-08-01', '-09-05'), min_games=None):
    """Best-effort reconstruction of preseason ADP for `season` from
    DynastyProcess/FantasyPros' full ECR archive.

    Returns a polars DataFrame with columns: player, pos, ecr, bye --
    one row per player, deduplicated to the EARLIEST ECR snapshot inside
    `season`'s preseason window (earliest = least in-season information
    leaked into what's supposed to be a preseason ranking).

    Raises ImportError if nflreadpy isn't installed, ValueError if the
    archive has no rows in that window (most likely because it doesn't go
    back that far, or the window needs widening for whatever FantasyPros
    published that year).
    """
    try:
        import nflreadpy as nfl
    except ImportError as e:
        raise ImportError(
            "nflreadpy is required for ADP-based rosters (already in this "
            "project's pyproject.toml -- pip install nflreadpy if needed)"
        ) from e

    all_ecr = nfl.load_ff_rankings(type='all')
    lo = f"{season}{preseason_window[0]}"
    hi = f"{season}{preseason_window[1]}"
    window = all_ecr.filter(
        (pl.col('scrape_date') >= pl.lit(lo).str.to_date())
        & (pl.col('scrape_date') <= pl.lit(hi).str.to_date())
        & (pl.col('pos').is_in(['QB', 'RB', 'WR', 'TE']))
    )
    if window.height == 0:
        raise ValueError(
            f"no ECR snapshots between {lo} and {hi} in load_ff_rankings(type='all') -- "
            f"either widen `preseason_window`, or DynastyProcess's archive doesn't "
            f"go back to {season} (run describe_adp_archive() to check). Fall back "
            f"to league_validate.draft_league (with a suitable lookback_seasons) instead."
        )

    dedup = (
        window.sort('scrape_date')
        .group_by('player')
        .agg(pl.col('pos').first(), pl.col('ecr').first(),
             pl.col('bye').first() if 'bye' in window.columns else pl.lit(None).alias('bye'),
             pl.col('scrape_date').first())
    )
    return dedup.sort('ecr')


def resolve_adp_to_canonical(adp, data_path='processed-data/ff-processed.parquet'):
    """Map ADP's `player` names onto ff-processed.parquet's `player`
    strings (same Jr./Sr./II/III suffix handling as the Yahoo mock -- see
    mock_league.resolve_names). Drops anything that doesn't resolve and
    prints how many/which, so gaps are visible rather than silent.
    """
    canonical = pl.read_parquet(data_path)['player'].unique().to_list()
    mapping, unmatched = resolve_names(adp['player'].to_list(), canonical)
    if unmatched:
        print(f"WARNING: {len(unmatched)} ADP players could not be matched to "
              f"ff-processed.parquet and will be dropped: {unmatched[:20]}"
              + (" ..." if len(unmatched) > 20 else ""))
    return (
        adp.filter(pl.col('player').is_in(list(mapping)))
        .with_columns(pl.col('player').replace(mapping))
    )


def draft_by_adp(adp, n_teams=12, roster_size=DEFAULT_ROSTER_SIZE, seed=0):
    """Single OVERALL snake draft strictly by ascending ECR -- the
    best-player-available draft a real ADP-following manager would run,
    as opposed to league_validate.draft_league's independent per-position
    mini-drafts. Returns {f'Team{i+1}': [player names]}.
    """
    names = adp.sort('ecr')['player'].to_list()
    need = n_teams * roster_size
    if len(names) < need:
        raise ValueError(f"only {len(names)} ranked players available, need "
                         f"{need} for {n_teams} teams x {roster_size} -- lower "
                         f"roster_size or n_teams")
    names = names[:need]

    team_names = [f'Team{i + 1}' for i in range(n_teams)]
    teams = {t: [] for t in team_names}
    order = []
    for rnd in range(roster_size):
        order.extend(range(n_teams) if rnd % 2 == 0 else range(n_teams - 1, -1, -1))
    for slot_i, team_i in enumerate(order):
        teams[team_names[team_i]].append(names[slot_i])
    return teams


def fill_bench_depth(rosters, adp, min_bench_depth=None):
    """Top up any position below `min_bench_depth` on any team by handing
    out the next-best remaining ADP players at that position, round-robin
    across the teams that need them so a shortfall doesn't concentrate on
    one team. Mutates and returns `rosters`.

    This is what makes ADP-drafted rosters usable for week-by-week
    validation at all: a strict best-player-available draft can easily
    leave some team with only 1 rostered QB if QBs just weren't
    fashionable picks in the relevant rounds, and league_validate.py has
    no waiver-wire model -- a bye/injury at a 1-deep position means a
    skipped matchup, not a bad-but-legal lineup.
    """
    min_bench_depth = min_bench_depth or MIN_BENCH_DEPTH
    pos_of = dict(zip(adp['player'].to_list(), adp['pos'].to_list()))
    drafted = {p for roster in rosters.values() for p in roster}

    for pos, min_n in min_bench_depth.items():
        pool = [p for p in adp.sort('ecr')['player'].to_list()
               if pos_of.get(p) == pos and p not in drafted]
        needy = [t for t, roster in rosters.items()
                if sum(1 for p in roster if pos_of.get(p) == pos) < min_n]
        i = 0
        while needy and i < len(pool):
            for t in list(needy):
                have = sum(1 for p in rosters[t] if pos_of.get(p) == pos)
                if have >= min_n:
                    needy.remove(t)
                    continue
                if i >= len(pool):
                    break
                rosters[t].append(pool[i])
                drafted.add(pool[i])
                i += 1
    return rosters


def rosters_from_adp(season, n_teams=12, roster_size=DEFAULT_ROSTER_SIZE,
                     min_bench_depth=None, seed=0,
                     data_path='processed-data/ff-processed.parquet'):
    """One-call entry point: real historical ADP -> overall snake draft ->
    bye-week bench top-up -> name resolution against ff-processed.parquet.
    See module docstring for the "this is unverified against real data"
    caveat -- run describe_adp_archive() on your machine first.
    """
    adp = load_historical_adp(season)
    adp = resolve_adp_to_canonical(adp, data_path=data_path)
    rosters = draft_by_adp(adp, n_teams=n_teams, roster_size=roster_size, seed=seed)
    rosters = fill_bench_depth(rosters, adp, min_bench_depth=min_bench_depth)
    return rosters
