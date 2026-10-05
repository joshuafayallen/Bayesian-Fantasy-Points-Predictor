"""League simulation: does a risk-aware start/sit rule win more head-to-head
matchups than "start the projected-best lineup", against real outcomes?

For each (season, week): forecast the week once with both models, then for
every scheduled matchup compute team A's baseline lineup
(maximize_expected) and the rule's lineup from the SAME draws, against the
SAME opponent lineup (team B always plays maximize_expected), and score
both on realized points. The baseline-vs-rule comparison is paired, so it
is tested with McNemar's exact test on the discordant matchups.

Ported from src/league_validate.py + src/mock_league.py. Changes: rosters,
pools and realized points are keyed on player_id (display names aren't
unique); one forecast per week is shared by every rule and cached on disk;
each week's rows go to their own JSON file (resumable, parallel-safe).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import polars as pl

from . import decision_engine as de

# pyrefly: ignore [missing-import]
from .config import TARGET_COL, SamplerSettings

# pyrefly: ignore [missing-import]
from .forecast import WeekForecast, forecast_week

log = logging.getLogger(__name__)

DEFAULT_ROSTER_DEPTH = {'QB': 2, 'RB': 5, 'WR': 5, 'TE': 2}
SUFFIXES = (' Jr.', ' Sr.', ' III', ' II', ' IV', ' Jr', ' Sr')
RISK_RULES = ('chance_constrained', 'cvar_averse', 'head_to_head')


# ----------------------------------------------------------------- rosters

def draft_league(history: pl.DataFrame, n_teams: int = 10, roster_depth: dict | None = None,
                 min_games: int = 3) -> dict[str, list[str]]:
    """Per-position snake drafts by trailing mean points over ``history``
    (which must end before the season being simulated). Returns
    {team_name: [player_id, ...]}."""
    roster_depth = roster_depth or DEFAULT_ROSTER_DEPTH
    agg = (history.group_by('player_id', 'position')
           .agg(pl.col(TARGET_COL).mean().alias('trail_mean'), pl.len().alias('n_games'))
           .filter(pl.col('n_games') >= min_games))
    names = [f'Team{i + 1}' for i in range(n_teams)]
    teams = {t: [] for t in names}
    for pos, k in roster_depth.items():
        pool = agg.filter(pl.col('position') == pos).sort(['trail_mean', 'player_id'], descending=[True, False])
        need = n_teams * k
        if pool.height < need:
            raise ValueError(f'only {pool.height} {pos} with >= {min_games} games; need {need}')
        ids = pool['player_id'].to_list()[:need]
        order = [t for rnd in range(k) for t in (range(n_teams) if rnd % 2 == 0 else range(n_teams - 1, -1, -1))]
        for slot, team_i in enumerate(order):
            teams[names[team_i]].append(ids[slot])
    return teams


def _strip_suffix(name: str) -> str:
    for suf in SUFFIXES:
        if name.endswith(suf):
            return name[: -len(suf)].strip()
    return name


def rosters_from_mock_draft(csv_path: Path | str, history: pl.DataFrame,
                            manager_col: str = 'team', player_col: str = 'player') -> dict[str, list[str]]:
    """{manager: [player_id]} from a mock-draft CSV. Uses its player_id
    column when present (the Yahoo mock has one); otherwise matches names
    (exact, then with Jr./Sr./II/III/IV stripped), refusing ambiguous names."""
    mock = pl.read_csv(csv_path, infer_schema_length=10_000)
    known = set(history['player_id'].unique().to_list())
    if 'player_id' in mock.columns:
        mock = _one_id_per_pick(mock, history, manager_col, player_col)
        unknown = mock.filter(~pl.col('player_id').is_in(list(known)))
        if unknown.height:
            log.warning('%d drafted players have no rows in the data (rookies / no stats): %s',
                        unknown.height, unknown[player_col].to_list()[:15])
        ids = mock.select(manager_col, 'player_id')
    else:
        names = history.select('player_id', 'player').unique()
        names = names.with_columns(pl.col('player').map_elements(_strip_suffix, return_dtype=pl.String).alias('_key'))
        uniq = names.group_by('_key').agg(pl.col('player_id').unique().alias('_ids')).filter(pl.col('_ids').list.len() == 1)
        lookup = dict(zip(uniq['_key'].to_list(), [x[0] for x in uniq['_ids'].to_list()]))
        mock = mock.with_columns(pl.col(player_col).map_elements(lambda n: lookup.get(_strip_suffix(n)),
                                                                return_dtype=pl.String).alias('player_id'))
        miss = mock.filter(pl.col('player_id').is_null())[player_col].to_list()
        if miss:
            log.warning('%d mock-draft names unmatched or ambiguous (dropped): %s', len(miss), miss[:15])
        ids = mock.drop_nulls('player_id').select(manager_col, 'player_id')
    return {m: g['player_id'].to_list() for (m,), g in ids.group_by(manager_col, maintain_order=True)}


def _one_id_per_pick(mock: pl.DataFrame, history: pl.DataFrame, manager_col: str, player_col: str) -> pl.DataFrame:
    """The scraped Yahoo mock was joined to nflverse rosters by NAME, so a pick
    can appear once per same-named player (three Josh Allens, three Kyle
    Williamses). Keep, per pick, the candidate whose position matches and who
    played most recently."""
    key = ['pick'] if 'pick' in mock.columns else [manager_col, player_col]
    seen = history.group_by('player_id').agg(pl.col('season').max().alias('_last'),
                                              pl.col('position').mode().first().alias('_hpos'))
    m = (mock.join(seen, on='player_id', how='left')
         .with_columns((pl.col('_hpos') == pl.col('position')).fill_null(False).alias('_pos_ok'))
         .sort(['_pos_ok', '_last'], descending=True, nulls_last=True)
         .unique(subset=key, keep='first')
         .sort(key))
    if m.height < mock.height:
        log.info('mock draft: %d duplicate name matches collapsed to one player_id per pick', mock.height - m.height)
    return m.drop('_last', '_hpos', '_pos_ok')


def load_rosters(season: int, history: pl.DataFrame, mock_draft_csv=None, n_teams: int = 10,
                 roster_depth: dict | None = None, lookback_seasons: int = 2) -> dict[str, list[str]]:
    if mock_draft_csv:
        return rosters_from_mock_draft(mock_draft_csv, history)
    h = history.filter((pl.col('season') < season) & (pl.col('season') >= season - lookback_seasons))
    if h.height == 0:
        raise ValueError(f'no history in {season - lookback_seasons}-{season - 1} to draft from')
    return draft_league(h, n_teams=n_teams, roster_depth=roster_depth)


def round_robin_schedule(team_names, n_rounds: int | None = None):
    """Circle-method round robin: list of rounds of (team_a, team_b)."""
    teams = list(team_names)
    n = len(teams)
    if n % 2:
        raise ValueError('round_robin_schedule needs an even number of teams')
    full, fixed, rot = [], teams[0], teams[1:]
    for _ in range(n - 1):
        cur = [fixed] + rot
        full.append([(cur[i], cur[n - 1 - i]) for i in range(n // 2)])
        rot = rot[-1:] + rot[:-1]
    return full if n_rounds is None else [full[i % len(full)] for i in range(n_rounds)]


# ----------------------------------------------------------------- scoring

def realized_total(frame: pl.DataFrame, player_ids) -> float:
    """Actual points for a set of players (absent / null = 0: didn't play)."""
    return float(frame.filter(pl.col('player_id').is_in(list(player_ids)))[TARGET_COL].fill_null(0).sum())


def oracle_total(frame: pl.DataFrame, pool) -> float:
    """Best legal lineup from ``pool`` on realized points (hindsight bound)."""
    y = frame[TARGET_COL].fill_null(0).to_numpy()
    return max(float(y[[i for _, i in c]].sum()) for c in de.all_candidate_lineups(frame, pool=pool))


def score_matchups(fc: WeekForecast, rosters: dict, schedule_round, rules=RISK_RULES, rule_kwargs=None):
    """Rows for every (matchup, rule) in this week. Model-agnostic: needs only
    the forecast's frame (with realized points) and draws."""
    rule_kwargs = rule_kwargs or {}
    frame, S = fc.frame, fc.draws
    if TARGET_COL not in frame.columns or frame[TARGET_COL].null_count() == frame.height:
        raise ValueError('forecast frame has no realized points -- league scoring needs a played week')
    available = set(frame['player_id'].to_list())
    rows = []
    for team_a, team_b in schedule_round:
        pool_a = [p for p in rosters[team_a] if p in available]
        pool_b = [p for p in rosters[team_b] if p in available]
        try:
            base_a = de.maximize_expected(S, frame, pool=pool_a)
            base_b = de.maximize_expected(S, frame, pool=pool_b)
        except ValueError as e:      # e.g. the only QB is on bye; no waiver wire is modelled
            log.info('skip %s vs %s %d wk%g: %s', team_a, team_b, fc.season, fc.week, e)
            continue
        ids = lambda c: c.lineup['player_id'].to_list()  # noqa: E731
        base_a_real, base_b_real = realized_total(frame, ids(base_a)), realized_total(frame, ids(base_b))
        oracle_a = oracle_total(frame, pool_a)
        for rule in rules:
            kw = dict(rule_kwargs.get(rule, {}))
            smart = de.decide(rule, S, frame, pool_a, opp_totals=base_b.totals if rule == 'head_to_head' else None, **kw)
            smart_real = realized_total(frame, ids(smart))
            row = {'season': fc.season, 'week': fc.week, 'rule': rule, 'team_a': team_a, 'team_b': team_b,
                       'baseline_a_realized': base_a_real, 'baseline_b_realized': base_b_real,
                       'baseline_a_won': int(base_a_real > base_b_real),
                       'smart_a_realized': smart_real, 'smart_a_won': int(smart_real > base_b_real),
                       'same_lineup_as_baseline': int(set(ids(base_a)) == set(ids(smart))),
                       'baseline_near_tie': bool(base_a.stats.get('near_tie')),
                       'smart_near_tie': bool(smart.stats.get('near_tie')),
                       'predicted_expected_baseline': base_a.stats['expected'],
                       'predicted_expected_smart': smart.stats['expected'],
                       'predicted_p_win_baseline': de.p_beats(base_a.totals, base_b.totals),
                       'predicted_p_win_smart': de.p_beats(smart.totals, base_b.totals),
                       'oracle_a': oracle_a, 'regret_baseline_a': oracle_a - base_a_real,
                       'regret_smart_a': oracle_a - smart_real}
            if rule == 'chance_constrained':
                row.update(floor=smart.stats['floor'], beta=smart.stats['beta'],
                           feasible=bool(smart.stats['feasible']), p_clear=smart.stats['p_clear'],
                           cleared=int(smart_real >= smart.stats['floor']))
            elif rule == 'cvar_averse':
                row.update(target=smart.stats['target'], alpha=smart.stats['alpha'], lam=smart.stats['lam'],
                           cvar=smart.stats['cvar'])
            rows.append(row)
    return rows


def run_league_week(season: int, week: float, history: pl.DataFrame, team_form: pl.DataFrame,
                    rosters: dict, out_dir: Path | str, settings: SamplerSettings = SamplerSettings(),  # noqa: B008
                    rules=RISK_RULES, rule_kwargs=None, overwrite: bool = False):
    """Forecast (season, week) on its played rows (cached under
    out_dir/forecasts), score every matchup for every rule, write one JSON."""
    out_dir = Path(out_dir)
    out = out_dir / f'league__{season}__wk{int(week):02d}.json'
    if out.exists() and not overwrite:
        log.info('skip %s (exists)', out.name)
        return None
    fc_dir = out_dir / 'forecasts' / f'{season}_wk{int(week):02d}'
    if (fc_dir / 'meta.json').exists():
        fc = WeekForecast.load(fc_dir)
    else:
        fc = forecast_week(history, team_form, season, week, settings=settings)
        fc.save(fc_dir)
    names = list(rosters)
    sched = round_robin_schedule(names)
    rows = score_matchups(fc, rosters, sched[int(week - 1) % len(sched)], rules=rules, rule_kwargs=rule_kwargs)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(rows, default=float))
    os.replace(tmp, out)
    log.info('%d wk%g: %d matchup-rule rows -> %s', season, week, len(rows), out)
    return rows


# --------------------------------------------------------------- summarize

def load_league(out_dir: Path | str) -> pl.DataFrame:
    rows = [r for f in sorted(Path(out_dir).glob('league__*.json')) for r in json.loads(f.read_text())]
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def reliability(df: pl.DataFrame, pred: str, outcome: str, n_buckets: int = 5) -> pl.DataFrame:
    """Mean predicted probability vs empirical rate, by predicted-probability quantile bucket."""
    n_buckets = max(1, min(n_buckets, df.height))
    return (df.with_columns(((pl.col(pred).rank('ordinal') - 1) * n_buckets // df.height).alias('bucket'))
            .group_by('bucket').agg(pl.col(pred).mean().alias('mean_predicted'),
                                    pl.col(outcome).mean().alias('empirical_rate'), pl.len().alias('n'))
            .sort('bucket'))


def summarize_league(out_dir: Path | str) -> pl.DataFrame:
    from scipy.stats import binomtest
    df = load_league(out_dir)
    if df.height == 0:
        print(f'no league rows in {out_dir}')
        return df
    out = []
    for (rule,), d in df.group_by('rule', maintain_order=True):
        rescued = d.filter((pl.col('smart_a_won') == 1) & (pl.col('baseline_a_won') == 0)).height
        cost = d.filter((pl.col('smart_a_won') == 0) & (pl.col('baseline_a_won') == 1)).height
        p = binomtest(rescued, rescued + cost, 0.5).pvalue if rescued + cost else float('nan')
        out.append({'rule': rule, 'matchups': d.height, 'weeks': d.select('season', 'week').n_unique(),
                        'win_rate_baseline': d['baseline_a_won'].mean(), 'win_rate_rule': d['smart_a_won'].mean(),
                        'rescued': rescued, 'cost': cost, 'mcnemar_p': p, 'same_lineup': d['same_lineup_as_baseline'].mean(),
                        'regret_baseline': d['regret_baseline_a'].mean(), 'regret_rule': d['regret_smart_a'].mean(),
                        'near_tie_rule': d['smart_near_tie'].mean() if 'smart_near_tie' in d.columns else None})
    summary = pl.DataFrame(out)
    with pl.Config(tbl_cols=20, tbl_width_chars=220, float_precision=3):
        print(summary)
        base = df.unique(subset=['season', 'week', 'team_a'])
        print('--- calibration of predicted P(win), baseline lineups')
        print(reliability(base, 'predicted_p_win_baseline', 'baseline_a_won'))
        cc = df.filter(pl.col('rule') == 'chance_constrained')
        if cc.height and 'p_clear' in cc.columns:
            print(f'--- chance_constrained: promised p_clear {cc["p_clear"].mean():.3f}, '
                  f'realized clear rate {cc["cleared"].mean():.3f}, feasible {cc["feasible"].mean():.1%}')
    return summary
