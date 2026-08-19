"""
Load a REAL drafted league from a scraped mock-draft CSV (e.g.
mock-rosters/yahoo_expert_2025.csv -- a Yahoo staff mock draft, 12 experts,
12 rounds, snake order) and turn it into the same
{manager_name: [player_names]} shape that league_validate.draft_league
produces synthetically, so league_validate.py's scheduling / scoring /
summarize machinery works UNCHANGED -- just fed a real draft board instead
of a trailing-mean ADP proxy.

This is a strictly better roster source than draft_league whenever a real
mock exists for the season you're validating: it's an actual
expert-consensus draft, not an approximation, and (being drafted before
the season) has no lookahead problem either.

The mock CSV's `team` column is the drafting MANAGER's name, not an NFL
team -- don't confuse it with ff-processed.parquet's `team` column.

Player names need light normalization before they'll join against
ff-processed.parquet's `player` column: nflreadr and Yahoo don't always
agree on whether to include a Jr./Sr./II/III/IV suffix (e.g. the mock has
"Travis Etienne Jr.", the data has "Travis Etienne"; the mock has "Luther
Burden", the data has "Luther Burden III"). `resolve_names` tries an exact
match first, then falls back to matching with the suffix stripped from
either side.

A handful of drafted players may have ZERO rows in the season you're
validating even after name resolution (e.g. a season-ending injury before
the mock was even drafted, like Chris Godwin/Joe Mixon/Brandon Aiyuk in the
2025 mock) -- that's not a matching bug, it's a real gap, and downstream
code already handles it: any (season, week) where a player has no row just
drops them from that week's available pool, same as a real bye/injury.

NOTE: a 12-round mock only gives each team 1 QB and 1-2 TE. Real seasons
have byes and injuries, and decision_engine.py has no waiver-wire model --
it only ever picks from the pool you hand it. So on any given week, a
team missing its only QB or TE to a bye/injury can't fill that slot at
all, and league_validate.py will skip that whole matchup rather than
silently play a lineup short a player. Expect week-to-week matchup
coverage to vary quite a bit with a roster this thin -- that's a real
finding about this specific draft's depth, not a bug in the tooling.
"""

import polars as pl

SUFFIXES = (' Jr.', ' Sr.', ' III', ' II', ' IV', ' Jr', ' Sr')


def _strip_suffix(name):
    for suf in SUFFIXES:
        if name.endswith(suf):
            return name[: -len(suf)].strip()
    return name


def resolve_names(mock_names, canonical_names):
    """Map mock-draft player names to the exact strings used in
    ff-processed.parquet's `player` column.

    Returns (mapping, unmatched): mapping is {mock_name: canonical_name}
    for everything resolved (exact match, or match after stripping a
    Jr./Sr./II/III/IV suffix from either side); unmatched lists names that
    need a manual fix (typo, a nickname Yahoo used that nflreadr doesn't,
    etc.) -- print these and eyeball them rather than assuming they're all
    benign.
    """
    canon_set = set(canonical_names)
    stripped_to_canon = {}
    for c in canonical_names:
        stripped_to_canon.setdefault(_strip_suffix(c), c)

    mapping, unmatched = {}, []
    for name in mock_names:
        if name in canon_set:
            mapping[name] = name
        elif _strip_suffix(name) in stripped_to_canon:
            mapping[name] = stripped_to_canon[_strip_suffix(name)]
        else:
            unmatched.append(name)
    return mapping, unmatched


def rosters_from_mock_draft(csv_path, data_path='processed-data/ff-processed.parquet',
                            manager_col='team', player_col='player'):
    """Build a {manager: [canonical_player_names]} roster dict from a
    scraped mock-draft CSV (columns: pick, player, position, team=manager).
    Prints a warning listing any names that couldn't be resolved at all
    (dropped from their team's roster) so gaps are visible, not silent.
    """
    mock = pl.read_csv(csv_path)
    all_players = pl.read_parquet(data_path)['player'].unique().to_list()

    mapping, unmatched = resolve_names(mock[player_col].to_list(), all_players)
    if unmatched:
        print(f"WARNING: {len(unmatched)} mock-draft players could not be matched "
              f"to ff-processed.parquet's player column and will be dropped from "
              f"their team's roster: {unmatched}")

    resolved = mock.filter(pl.col(player_col).is_in(list(mapping))).with_columns(
        pl.col(player_col).replace(mapping).alias('_resolved')
    )

    managers = resolved[manager_col].unique().to_list()
    return {m: resolved.filter(pl.col(manager_col) == m)['_resolved'].to_list() for m in managers}
