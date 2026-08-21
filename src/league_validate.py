"""
Ten-team league simulation for validating decision_engine.py's start/sit
calls in a realistic multi-team, head-to-head context.

calibrate_decisions.py validates each rule's PROMISES (does a beta=0.90
chance constraint clear the floor 90% of the time, etc.) against one
simulated roster vs. one simulated "tier-down" opponent. That's a fine
statistical check but a thin economic one: real fantasy football is a
league of ~10 GMs drafting from the SAME player pool and playing each
other head-to-head every week. This module builds that league and asks
the sharper question: does using a risk-aware decision rule instead of
plain "start your projected-best lineup" actually win you more matchups
against a normal manager, all else (the roster, the schedule, the week's
real outcomes) held fixed?

    1. draft_league          -- ONE preseason snake draft (by prior-SEASON
                                 trailing mean, the ADP proxy) fixes 10
                                 rosters for the whole season -- a real
                                 league doesn't re-draft every week, and
                                 using only prior-season data means calling
                                 this once per week of the same season
                                 deterministically reproduces the same 10
                                 rosters with no lookahead into the season
                                 being validated.
    2. round_robin_schedule  -- a real weekly schedule (circle method), so
                                 "your opponent this week" is a specific
                                 other team's roster, not a synthetic
                                 average.
    3. run_league_fold       -- for one real (season, week) and every
                                 scheduled matchup, computes BOTH lineups
                                 for team_a: the naive baseline
                                 (maximize_expected) and the rule under
                                 test -- against the SAME opponent lineup,
                                 the SAME posterior draws, and the SAME
                                 actual outcome once the week resolves.
                                 That pairing is what makes `summarize_league`'s
                                 win-rate comparison a fair, PAIRED test
                                 (McNemar) instead of two noisy independent
                                 averages that happen to be compared.

Everything downstream of "get real posterior draws for a real week" reuses
calibrate_decisions.py's plumbing (fit_predict from sandbox/backtest_harness.py)
and decision_engine.py's rules, unchanged.

Usage (run from the project root):
    python src/league_validate.py 2025 12 --rule cvar_averse
    python src/league_validate.py 2025 14 --rule chance_constrained --beta 0.85
    python src/league_validate.py 2025 12 --rule chance_constrained --model hs
    ...
    python src/league_validate.py --summarize --out results/league_validation.csv

--model selects which of backtest_harness.py's fitted architectures to use
('ar', 'hs', or 'team' -- defaults to 'team', the strongest performer in
results/backtest_walkforward.csv). Was hardcoded to 'ar' in an earlier
version of this script -- results/league_validation.csv rows written before
this change have no `model` column value and are reported separately by
summarize_league() rather than silently pooled with a named model's rows.

Team_b (the "opponent" in every matchup) always plays maximize_expected --
this experiment isolates the value of team_a's smarter rule, holding the
opponent's behavior fixed. Swap roles/re-run with a different --rule to see
each rule's lift against a normal manager.

If you have a real scraped mock draft for the season (see mock_league.py --
e.g. mock-rosters/yahoo_expert_2025.csv, a 12-expert Yahoo staff mock),
pass --mock-draft to use those actual drafted rosters instead of the
synthetic trailing-mean draft_league below. It's strictly better when
available: a real expert-consensus board, not an ADP proxy, and (being
drafted before the season) has no lookahead problem either.

    python src/league_validate.py 2025 12 --rule cvar_averse \\
        --mock-draft mock-rosters/yahoo_expert_2025.csv
"""

import os
import sys
import argparse

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                 # src/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'sandbox'))

import decision_engine as de
from calibrate_decisions import realized_total, oracle_lineup, DEFAULT_ROSTER_DEPTH, WINDOW_SEASONS, DATA_PATH

OUT_CSV = "results/league_validation.csv"


# ------------------------------------------------------------------ league setup

def draft_league(history, n_teams=10, roster_depth=None, min_games=3, seed=0):
    """Snake-draft `n_teams` rosters from `history`, independently per
    position (n_teams parallel single-position snake drafts by trailing
    mean) -- keeps every roster the same SHAPE, and is a reasonable
    stand-in for position-by-position ADP without needing real ADP data.

    `history` must be data STRICTLY BEFORE the season you're about to
    validate (e.g. `df.filter(pl.col('season') < target_season)`) so this
    is a genuine preseason draft, not a mid-season re-draft with
    lookahead. Rookies with no prior-season history can't be drafted --
    same limitation any ADP-free simulation has.

    IMPORTANT: `history` should already be trimmed to a RECENT window
    (a couple of prior seasons), not hand it the entire multi-decade
    history -- `load_rosters` does this trimming for you via
    `lookback_seasons`. Ranking by lifetime average over 15+ seasons will
    happily draft long-retired players whose career average is high but
    who have zero games in the season you're validating (their peak years
    drag the average up even though they haven't played in a decade) --
    exactly the "need 2 RB, have 0" failure mode this causes if skipped.
    """
    roster_depth = roster_depth or DEFAULT_ROSTER_DEPTH
    agg = (
        history.group_by(['player', 'position'])
        .agg(pl.col('total_fantasy_points').mean().alias('trail_mean'),
             pl.len().alias('n_games'))
        .filter(pl.col('n_games') >= min_games)
    )
    team_names = [f'Team{i + 1}' for i in range(n_teams)]
    teams = {t: [] for t in team_names}

    for pos, k in roster_depth.items():
        pool = agg.filter(pl.col('position') == pos).sort('trail_mean', descending=True)
        need = n_teams * k
        if pool.height < need:
            raise ValueError(
                f"only {pool.height} {pos} with >= {min_games} games in this "
                f"history -- need {need} for {n_teams} teams x {k} per team. "
                f"Lower roster_depth['{pos}'], lower min_games, or pick a "
                f"target season with more prior history."
            )
        names = pool['player'].to_list()[:need]
        order = []
        for rnd in range(k):
            order.extend(range(n_teams) if rnd % 2 == 0 else range(n_teams - 1, -1, -1))
        for slot_i, team_i in enumerate(order):
            teams[team_names[team_i]].append(names[slot_i])
    return teams


def load_rosters(season, mock_draft_csv=None, adp_draft=False, n_teams=10, roster_depth=None,
                 seed=0, lookback_seasons=2, data_path=DATA_PATH):
    """Single entry point for "how do we get 10-ish rosters for this
    season", in priority order:

      1. `mock_draft_csv`  -- a real scraped mock draft for this exact
                              season (e.g. mock-rosters/yahoo_expert_2025.csv).
                              Best option when it exists: an actual
                              expert-consensus board, not a proxy, and
                              (being drafted before the season) has no
                              lookahead issue either. Only covers whatever
                              season(s) you've scraped one for.
      2. `adp_draft=True`  -- real historical ADP (FantasyPros ECR via
                              nflreadpy) for ANY season the archive
                              reaches, strictly best-player-available
                              drafted, with a bye-week bench top-up. See
                              adp_league.py -- this path is UNVERIFIED
                              against real data from this environment
                              (network to the data source is blocked
                              here); run adp_league.describe_adp_archive()
                              on your machine first.
      3. (default)         -- synthetic `draft_league`: trailing mean over
                              the `lookback_seasons` seasons immediately
                              before `season` (NOT the entire history --
                              ranking by lifetime average since 2006 would
                              draft long-retired players who have a high
                              career average but zero games left to play).
                              Works for any season, but is the weakest
                              proxy of the three.
    """
    if mock_draft_csv:
        from mock_league import rosters_from_mock_draft
        return rosters_from_mock_draft(mock_draft_csv, data_path=data_path)

    if adp_draft:
        from adp_league import rosters_from_adp
        return rosters_from_adp(season, n_teams=n_teams, seed=seed, data_path=data_path)

    df = pl.read_parquet(data_path)
    draft_history = df.filter(
        (pl.col('season') < season) & (pl.col('season') >= season - lookback_seasons)
    )
    if draft_history.height == 0:
        raise ValueError(f"no history in seasons {season - lookback_seasons}-{season - 1} -- "
                         f"can't draft a league; pick a later target season, raise "
                         f"lookback_seasons, or supply --mock-draft/--adp-draft")
    return draft_league(draft_history, n_teams=n_teams, roster_depth=roster_depth, seed=seed)


def round_robin_schedule(team_names, n_rounds=None):
    """Circle-method round-robin: a list of rounds, each a list of
    (team_a, team_b) tuples, every team appearing exactly once per round.
    Needs an even number of teams (byes not modeled). Cycles through the
    full (n_teams - 1)-round schedule if n_rounds is longer than that."""
    teams = list(team_names)
    n = len(teams)
    if n % 2 != 0:
        raise ValueError("round_robin_schedule needs an even number of teams (byes not modeled)")
    full = []
    fixed, rotating = teams[0], teams[1:]
    for _ in range(n - 1):
        cur = [fixed] + rotating
        full.append([(cur[i], cur[n - 1 - i]) for i in range(n // 2)])
        rotating = rotating[-1:] + rotating[:-1]
    if n_rounds is None:
        return full
    return [full[i % len(full)] for i in range(n_rounds)]


# ------------------------------------------------------------------ one fold

def already_done(out_csv, season, week, rule, model_name=None):
    if not os.path.exists(out_csv):
        return False
    existing = pl.read_csv(out_csv)
    match = existing.filter(
        (pl.col('season') == season) & (pl.col('week') == week) & (pl.col('rule') == rule)
    )
    # older rows (from before `model` was tracked) have a null model column
    # via diagonal_relaxed concat -- only treat a fold as done if it was run
    # under the SAME model, so re-running with a different model_name isn't
    # silently skipped as "already done" against a different model's rows.
    if model_name is not None and 'model' in existing.columns:
        match = match.filter(pl.col('model') == model_name)
    return match.height > 0


def run_league_fold(season, week, rosters, schedule_round, rule='chance_constrained',
                    rule_kwargs=None, seed=0, advi_n=6000, advi_draws=400,
                    data_path=DATA_PATH, model_name='team'):
    """Fit once for this (season, week), then score every scheduled
    matchup under `rule`. Returns a list of row-dicts, or None if this
    week has no data / all cold starts.

    model_name : which of backtest_harness.py's architectures to fit --
        'ar', 'hs', or 'team' (same choices `fit_predict` itself accepts).
        Was hardcoded to 'ar' here; defaulted to 'team' now since it's the
        strongest performer in the walk-forward backtest
        (results/backtest_walkforward.csv) -- but every rule in
        decision_engine.py is model-agnostic (it only consumes `keep`/`S`,
        the frame + posterior predictive draws, never anything
        architecture-specific), so any of the three works here.
    """
    from backtest_harness import fit_predict   # see calibrate_decisions.py's note on this import

    df = pl.read_parquet(data_path)
    train = df.filter(
        (pl.col('season') >= season - (WINDOW_SEASONS - 1)) &
        ((pl.col('season') < season) | ((pl.col('season') == season) & (pl.col('week') < week)))
    )
    test = df.filter((pl.col('season') == season) & (pl.col('week') == week))
    if len(test) == 0 or len(train) < 200:
        return None
    r = fit_predict(model_name, 'advi', train, test, seed=seed, advi_n=advi_n, advi_draws=advi_draws)
    if r is None:
        return None
    keep, S = r['keep'], r['S']

    rows = _score_matchups(keep, S, season, week, rosters, schedule_round,
                           rule=rule, rule_kwargs=rule_kwargs, model_name=model_name)
    return rows


def _score_matchups(keep, S, season, week, rosters, schedule_round, rule='chance_constrained',
                    rule_kwargs=None, model_name=None):
    """Model-agnostic half of one fold -- given real draws `S` + realized
    `keep` for one week, score every scheduled matchup. Split out so it can
    be exercised with fake S/keep in tests without touching pymc.

    model_name is recorded on each output row (not used for anything else
    here) so results from different architectures can be told apart in
    results/league_validation.csv once more than one has been run."""
    rule_kwargs = rule_kwargs or {}
    available = set(keep['player'].to_list())
    rows = []

    for team_a, team_b in schedule_round:
        pool_a = [p for p in rosters[team_a] if p in available]
        pool_b = [p for p in rosters[team_b] if p in available]
        try:
            base_a = de.maximize_expected(S, keep, pool=pool_a)
            base_b = de.maximize_expected(S, keep, pool=pool_b)

            if rule == 'chance_constrained':
                # Default floor used to be 0.85 * expected -- a fixed
                # fraction of the mean that ignores the lineup's own
                # variance entirely. For a lineup total with realistic
                # week-to-week spread, "clear 85% of the mean in 90% of
                # scenarios" is often mathematically unreachable (rough
                # normal-approximation check: hitting 90% coverage at
                # 0.85*mean needs sigma/mean <~ 0.12, tighter than most
                # real lineup totals) -- which is exactly why every one of
                # the 79 chance_constrained rows in the first league
                # validation run (results/league_validation.csv) came back
                # feasible=False. base_a.stats['floor_p10'] is already the
                # baseline lineup's own 10th-percentile total (see
                # _package) -- self-calibrated to THIS lineup's actual
                # spread instead of an arbitrary fraction of the mean, and
                # roughly at the edge of feasibility for the baseline by
                # construction, so chance_constrained has real room to
                # either match it or find something strictly better that
                # still clears it.
                floor = rule_kwargs.get('floor') or base_a.stats['floor_p10']
                beta = rule_kwargs.get('beta', 0.90)
                smart_a = de.chance_constrained(S, keep, floor=floor, beta=beta, pool=pool_a)
                stat_name, stat_val = 'p_clear', smart_a.stats['p_clear']
            elif rule == 'cvar_averse':
                target = rule_kwargs.get('target') or base_a.stats['expected']
                alpha = rule_kwargs.get('alpha', 0.10)
                lam = rule_kwargs.get('lam', 1.0)
                smart_a = de.cvar_averse(S, keep, target=target, alpha=alpha, lam=lam, pool=pool_a)
                stat_name, stat_val = 'cvar', smart_a.stats['cvar']
            elif rule == 'head_to_head':
                smart_a = de.head_to_head_optimize(S, keep, base_b.totals, pool=pool_a)
                stat_name, stat_val = 'p_win', smart_a.stats['p_win']
            else:
                raise ValueError(f"unknown rule {rule!r}")
        except ValueError as e:
            # Most common cause: a team only rosters 1 QB (or 1-2 TE) and
            # that player is on bye/inactive this week, so there's no
            # eligible player left to fill the slot -- no waiver-wire
            # streaming is modeled here, so this matchup just can't be
            # scored this week. Real, expected, and will vary week to week
            # -- don't expect every round to yield a full set of scored
            # matchups, especially with thin (1-2 deep) rosters.
            print(f"  skip {team_a} vs {team_b} wk{week}: {e}")
            continue

        _, oracle_a_val = oracle_lineup(keep, pool_a)

        base_a_players = base_a.lineup['player'].to_list()
        smart_a_players = smart_a.lineup['player'].to_list()
        base_a_realized = realized_total(keep, base_a_players)
        base_b_realized = realized_total(keep, base_b.lineup['player'].to_list())
        smart_a_realized = realized_total(keep, smart_a_players)

        row = dict(
            season=season, week=week, rule=rule, model=model_name, team_a=team_a, team_b=team_b,
            baseline_a_realized=base_a_realized, baseline_b_realized=base_b_realized,
            baseline_a_won=int(base_a_realized > base_b_realized),
            smart_a_realized=smart_a_realized,
            smart_a_won=int(smart_a_realized > base_b_realized),
            # If this is 1, the "smart" rule landed on the identical lineup
            # as the naive baseline -- expected occasionally (sometimes the
            # highest-expectation lineup IS the safest/highest-win-prob one
            # too), but if it's 1 almost every row, the rule's parameters
            # (floor/beta, target/alpha/lam) probably aren't biting: see
            # `feasible` below for chance_constrained specifically.
            same_lineup_as_baseline=int(set(base_a_players) == set(smart_a_players)),
            predicted_expected_baseline=base_a.stats['expected'],
            predicted_stat_name=stat_name, predicted_stat_value=stat_val,
            oracle_a=oracle_a_val,
            regret_baseline_a=oracle_a_val - base_a_realized,
            regret_smart_a=oracle_a_val - smart_a_realized,
        )
        if rule == 'chance_constrained':
            # `feasible=False` means no candidate lineup could clear `floor`
            # in >= beta fraction of scenarios -- the rule fell back to the
            # highest-P(clear) lineup instead, which very often IS the
            # highest-expectation lineup too, hence same_lineup_as_baseline.
            # If this is False most weeks, `floor`/`beta` are set tighter
            # than this roster can support and should be loosened.
            row.update(floor=floor, beta=beta, feasible=smart_a.stats['feasible'])
        elif rule == 'cvar_averse':
            row.update(target=target, alpha=alpha, lam=lam)
        rows.append(row)
    return rows


def run_one_week(season, week, rule='chance_constrained', n_teams=10, roster_depth=None,
                 mock_draft_csv=None, adp_draft=False, lookback_seasons=2, rule_kwargs=None,
                 out_csv=OUT_CSV, seed=0, advi_n=6000, advi_draws=400, data_path=DATA_PATH,
                 model_name='team'):
    if already_done(out_csv, season, week, rule, model_name=model_name):
        print(f"SKIP: {season} wk{week} rule={rule} model={model_name} already in {out_csv}")
        return

    rosters = load_rosters(season, mock_draft_csv=mock_draft_csv, adp_draft=adp_draft,
                           n_teams=n_teams, roster_depth=roster_depth, seed=seed,
                           lookback_seasons=lookback_seasons, data_path=data_path)
    if (mock_draft_csv or adp_draft) and len(rosters) != n_teams:
        print(f"note: draft has {len(rosters)} teams, not --n-teams={n_teams} -- "
              f"using {len(rosters)} (the real draft wins)")
    team_names = list(rosters)
    full_schedule = round_robin_schedule(team_names)
    schedule_round = full_schedule[int(week - 1) % len(full_schedule)]

    rows = run_league_fold(season, week, rosters, schedule_round, rule=rule,
                           rule_kwargs=rule_kwargs, seed=seed, advi_n=advi_n,
                           advi_draws=advi_draws, data_path=data_path, model_name=model_name)
    if not rows:
        print(f"SKIP: no scoreable matchups for {season} wk{week}")
        return

    new = pl.DataFrame(rows)
    if os.path.exists(out_csv):
        existing = pl.read_csv(out_csv)
        pl.concat([existing, new], how='diagonal_relaxed').write_csv(out_csv)
    else:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        new.write_csv(out_csv)
    print(f"wrote {len(rows)} matchups for {season} wk{week} rule={rule} model={model_name} -> {out_csv}")


# --------------------------------------------------------------- summarize

def summarize_league(out_csv=OUT_CSV):
    """Paired win-rate comparison (McNemar) + regret, per (rule, model)
    pair, across every matchup-week written to `out_csv` so far. Rows from
    before `model` was tracked (see already_done) have a null model --
    grouped separately here rather than silently merged with a named
    model's rows, since they were all run against 'ar' (the old hardcoded
    default) and mixing them in would misattribute those results."""
    from scipy.stats import binomtest

    df = pl.read_csv(out_csv)
    has_model = 'model' in df.columns
    group_cols = ['rule', 'model'] if has_model else ['rule']
    groups = (df.select(group_cols).unique().sort(group_cols).rows(named=True))

    for g in groups:
        d = df.filter(pl.all_horizontal([pl.col(k).eq(v) if v is not None else pl.col(k).is_null()
                                         for k, v in g.items()]))
        rule = g['rule']
        model_label = g.get('model') or 'ar (untracked, pre-model-column run)'
        n = d.height
        wr_base = d['baseline_a_won'].mean()
        wr_smart = d['smart_a_won'].mean()
        rescued = d.filter((pl.col('smart_a_won') == 1) & (pl.col('baseline_a_won') == 0)).height
        cost = d.filter((pl.col('smart_a_won') == 0) & (pl.col('baseline_a_won') == 1)).height

        same = d['same_lineup_as_baseline'].mean() if 'same_lineup_as_baseline' in d.columns else None

        print(f"=== rule={rule}  model={model_label}  (n={n} matchups) ===")
        print(f"  baseline (maximize_expected) win rate : {wr_base:.3f}")
        print(f"  {rule} win rate                        : {wr_smart:.3f}   (lift {wr_smart - wr_base:+.3f})")
        print(f"  matchups {rule} WON that baseline would have LOST : {rescued}")
        print(f"  matchups {rule} LOST that baseline would have WON : {cost}")
        if same is not None:
            print(f"  picked the SAME lineup as baseline: {same:.1%} of matchups"
                  + ("  <-- rule parameters may not be biting, see below" if same > 0.8 else ""))
        if rule == 'chance_constrained' and 'feasible' in d.columns:
            print(f"  feasible (cleared floor >= beta): {d['feasible'].mean():.1%} of matchups"
                  + ("  <-- floor/beta likely too tight for this roster, try loosening"
                     if d['feasible'].mean() < 0.5 else ""))
        if rescued + cost > 0:
            pval = binomtest(rescued, rescued + cost, 0.5).pvalue
            print(f"  McNemar exact test (rescued vs. cost, H0: no difference): p={pval:.4f}")
        else:
            print("  no discordant pairs yet -- need more folds")
        print(f"  mean regret  baseline: {d['regret_baseline_a'].mean():.2f}   "
              f"{rule}: {d['regret_smart_a'].mean():.2f}")
        print()


# ------------------------------------------------------------------ main

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('season', type=int, nargs='?')
    p.add_argument('week', type=float, nargs='?')
    p.add_argument('--rule', choices=['chance_constrained', 'cvar_averse', 'head_to_head'],
                  default='chance_constrained')
    p.add_argument('--model', choices=['ar', 'hs', 'team'], default='team',
                  help='which fitted architecture to use (was hardcoded to ar; team_mod is '
                       'currently the strongest performer in results/backtest_walkforward.csv)')
    p.add_argument('--n-teams', type=int, default=12,
                  help='ignored if --mock-draft is given -- team count comes from the draft')
    p.add_argument('--mock-draft', type=str, default=None,
                  help='path to a scraped mock-draft CSV (pick,player,position,team=manager), '
                       'e.g. mock-rosters/yahoo_expert_2025.csv -- use real drafted rosters '
                       'instead of the synthetic trailing-mean draft_league')
    p.add_argument('--adp-draft', action='store_true',
                  help='draft from real historical FantasyPros ADP (nflreadpy) instead of the '
                       'synthetic trailing-mean draft_league -- works for any season the ECR '
                       'archive reaches, unlike --mock-draft which only covers scraped seasons. '
                       'See adp_league.py -- unverified against real data from this sandbox, '
                       'run adp_league.describe_adp_archive() first.')
    p.add_argument('--lookback-seasons', type=int, default=2,
                  help='ignored if --mock-draft/--adp-draft is given -- how many seasons of '
                       'prior history the synthetic draft ranks players over (recent form, '
                       'not lifetime avg)')
    p.add_argument('--floor', type=float, default=None)
    p.add_argument('--beta', type=float, default=0.90)
    p.add_argument('--target', type=float, default=None)
    p.add_argument('--alpha', type=float, default=0.10)
    p.add_argument('--lam', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--advi-n', type=int, default=6000)
    p.add_argument('--advi-draws', type=int, default=400)
    p.add_argument('--out', type=str, default=OUT_CSV)
    p.add_argument('--summarize', action='store_true')
    args = p.parse_args()

    if args.summarize:
        summarize_league(args.out)
        raise SystemExit

    if args.season is None or args.week is None:
        p.error('season and week are required unless --summarize is given')

    rule_kwargs = dict(floor=args.floor, beta=args.beta, target=args.target,
                       alpha=args.alpha, lam=args.lam)
    run_one_week(args.season, args.week, rule=args.rule, n_teams=args.n_teams,
                mock_draft_csv=args.mock_draft, adp_draft=args.adp_draft,
                lookback_seasons=args.lookback_seasons,
                rule_kwargs=rule_kwargs, out_csv=args.out,
                seed=args.seed, advi_n=args.advi_n, advi_draws=args.advi_draws,
                model_name=args.model)
