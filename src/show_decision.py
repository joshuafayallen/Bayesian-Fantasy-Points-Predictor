"""
Ad-hoc viewer: print the decision engine's ACTUAL lineup choices (not just
win/loss aggregates) for a handful of early-season weeks, where every
roster should be fully available (no NFL bye weeks land in weeks 1-3, so
this sidesteps the "not enough QB in pool" skip issue seen in week 12 --
that's a real roster-availability gap, not a decision-engine bug, and is
a separate to-do).

This reuses league_validate.py's own league setup (draft_league /
round_robin_schedule / load_rosters) and decision_engine.py's rules
unchanged -- it just prints the lineup instead of only writing the
aggregate CSV row that run_league_fold/_score_matchups produce.

Usage (from the project root, with the venv that has pymc etc. active):
    python src/show_decision.py 2025 --weeks 1 2 3 --team Team1 \
        --rule chance_constrained --model team

    # cheaper/faster first look: fewer ADVI draws, still a real fit
    python src/show_decision.py 2025 --weeks 1 2 3 --team Team1 \
        --rule chance_constrained --model ar --advi-n 2000 --advi-draws 200

    # write a per-player CSV alongside the console output (one row per
    # player per lineup per week -- baseline AND the rule under test)
    python src/show_decision.py 2025 --weeks 1 2 3 --team Team5 \
        --rule cvar_averse --model team --lam 2 \
        --csv-out results/decision_example.csv
"""
import os
import argparse

import polars as pl

import decision_engine as de
from league_validate import load_rosters, round_robin_schedule, TEAM_FORM_PATH
from calibrate_decisions import DATA_PATH, WINDOW_SEASONS, realized_total


def fit_week(season, week, model_name, advi_n, advi_draws, seed, data_path=DATA_PATH):
    from backtest_harness import fit_predict

    df = pl.read_parquet(data_path)
    if model_name in ('team', 'stack'):
        team_form = pl.read_parquet(TEAM_FORM_PATH)
        opp_team_form = team_form.rename({'team': 'opp_team', 'team_form': 'opp_team_form'})
        df = df.join(team_form, on=['season', 'week', 'team'], how='left')
        df = df.join(opp_team_form, on=['season', 'week', 'opp_team'], how='left')
        df = df.with_columns(pl.col('team_form', 'opp_team_form').fill_null(0.0))

    train = df.filter(
        (pl.col('season') >= season - (WINDOW_SEASONS - 1)) &
        ((pl.col('season') < season) | ((pl.col('season') == season) & (pl.col('week') < week)))
    )
    test = df.filter((pl.col('season') == season) & (pl.col('week') == week))
    if len(test) == 0 or len(train) < 200:
        print(f"  no data / too little training history for wk{week} -- skipping")
        return None
    return fit_predict(model_name, 'advi', train, test, seed=seed, advi_n=advi_n, advi_draws=advi_draws)


def player_expected(keep, S):
    """Per-player expected points -- mean of that player's column across
    the posterior predictive draws S (columns of S line up 1:1 with rows
    of keep, per decision_engine.py's own frame/draws contract). This is
    what each individual player is projected for, as opposed to
    choice.stats['expected'], which is the LINEUP's total expectation."""
    return dict(zip(keep['player'].to_list(), S.mean(axis=0).tolist()))


def describe_lineup(choice, keep, expected_by_player):
    """slot: player (position, team vs opp) -- projected pts, realized pts if known."""
    lines = []
    realized = dict(zip(keep['player'].to_list(), keep['total_fantasy_points'].to_list()))
    for row in choice.lineup.iter_rows(named=True):
        proj = expected_by_player.get(row['player'])
        proj_str = f"{proj:.1f} proj" if proj is not None else "proj n/a"
        pts = realized.get(row['player'])
        pts_str = f"{pts:.1f} pts" if pts is not None else "pts n/a"
        opp = f" vs {row['opp']}" if row.get('opp') else ""
        lines.append(f"    {row['slot']:<6} {row['player']:<24} {row['position']:<4}{opp:<10} "
                     f"{proj_str:<10} {pts_str}")
    return "\n".join(lines)


def lineup_rows(choice, keep, expected_by_player, *, season, week, team, opp_team, rule, lineup_type,
                same_as_baseline, lineup_realized, opp_realized, won, rule_params):
    """One CSV row per rostered slot in this lineup -- lineup_type is
    'baseline' or 'rule' so the two can be told apart (and diffed) once
    loaded back in. rule_params carries whatever rule-specific numbers
    (floor/beta/p_clear, target/alpha/lam/cvar, p_win) apply to THIS
    lineup as a whole -- repeated on every player row for that lineup,
    which is redundant but keeps the CSV flat and easy to filter/pivot.

    predicted_expected_pts is the PLAYER's own expected points (mean of
    their posterior predictive draws) -- lineup_expected is still the
    lineup TOTAL's expectation, kept for backwards compatibility with the
    earlier CSV.

    lineup_realized/opp_realized/won are the ACTUAL outcome -- summed real
    total_fantasy_points for this lineup vs. the opponent's (always
    maximize_expected, same convention as league_validate.py's
    _score_matchups), not the model's predicted expectation. This is what
    decides who actually won the matchup that week."""
    realized = dict(zip(keep['player'].to_list(), keep['total_fantasy_points'].to_list()))
    rows = []
    for row in choice.lineup.iter_rows(named=True):
        out = dict(
            season=season, week=week, team=team, opponent=opp_team, rule=rule,
            lineup_type=lineup_type, same_as_baseline=same_as_baseline,
            slot=row['slot'], player=row['player'], position=row['position'],
            vs=row.get('opp'), predicted_expected_pts=expected_by_player.get(row['player']),
            realized_pts=realized.get(row['player']),
            lineup_expected=choice.stats.get('expected'),
            lineup_realized=lineup_realized, opponent_realized=opp_realized, won=won,
        )
        out.update(rule_params)
        rows.append(out)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('season', type=int)
    p.add_argument('--weeks', type=int, nargs='+', default=[1, 2, 3])
    p.add_argument('--team', type=str, default='Team1', help="which drafted team to show the decision for")
    p.add_argument('--rule', choices=['chance_constrained', 'cvar_averse', 'head_to_head'],
                   default='chance_constrained')
    p.add_argument('--model', choices=['ar', 'hs', 'team'], default='team')
    p.add_argument('--n-teams', type=int, default=10)
    p.add_argument('--lookback-seasons', type=int, default=2)
    p.add_argument('--beta', type=float, default=0.90)
    p.add_argument('--alpha', type=float, default=0.10)
    p.add_argument('--lam', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--advi-n', type=int, default=6000)
    p.add_argument('--advi-draws', type=int, default=400)
    p.add_argument('--csv-out', type=str, default=None,
                   help="if set, write one row per rostered slot per lineup per week "
                        "(baseline AND the rule under test) to this CSV path")
    args = p.parse_args()

    rosters = load_rosters(args.season, n_teams=args.n_teams, lookback_seasons=args.lookback_seasons,
                           seed=args.seed)
    schedule = round_robin_schedule(list(rosters.keys()))

    if args.team not in rosters:
        print(f"'{args.team}' not in drafted rosters: {sorted(rosters)}")
        return
    print(f"{args.team} roster: {rosters[args.team]}\n")

    csv_rows = []

    for week in args.weeks:
        # find this team's opponent in the round-robin for this week
        round_idx = (week - 1) % len(schedule)
        opp = None
        for a, b in schedule[round_idx]:
            if a == args.team:
                opp = b
                break
            if b == args.team:
                opp = a
                break
        print(f"=== week {week}: {args.team} vs {opp} ===")

        r = fit_week(args.season, week, args.model, args.advi_n, args.advi_draws, args.seed)
        if r is None:
            continue
        keep, S = r['keep'], r['S']
        available = set(keep['player'].to_list())
        expected_by_player = player_expected(keep, S)

        pool_a = [pl_ for pl_ in rosters[args.team] if pl_ in available]
        pool_b = [pl_ for pl_ in rosters[opp] if pl_ in available]
        missing_a = [pl_ for pl_ in rosters[args.team] if pl_ not in available]
        if missing_a:
            print(f"  (not in this week's pool, likely bye/inactive/cold-start: {missing_a})")

        try:
            base_a = de.maximize_expected(S, keep, pool=pool_a)
            base_b = de.maximize_expected(S, keep, pool=pool_b)

            if args.rule == 'chance_constrained':
                floor = base_a.stats['floor_p10']
                smart_a = de.chance_constrained(S, keep, floor=floor, beta=args.beta, pool=pool_a)
                rule_params = dict(floor=floor, beta=args.beta,
                                   feasible=smart_a.stats['feasible'], p_clear=smart_a.stats['p_clear'])
                print(f"  rule=chance_constrained  floor={floor:.1f}  beta={args.beta}  "
                     f"feasible={smart_a.stats['feasible']}  p_clear={smart_a.stats['p_clear']:.3f}")
            elif args.rule == 'cvar_averse':
                target = base_a.stats['expected']
                smart_a = de.cvar_averse(S, keep, target=target, alpha=args.alpha, lam=args.lam, pool=pool_a)
                rule_params = dict(target=target, alpha=args.alpha, lam=args.lam, cvar=smart_a.stats['cvar'])
                print(f"  rule=cvar_averse  target={target:.1f}  alpha={args.alpha}  lam={args.lam}  "
                     f"cvar={smart_a.stats['cvar']:.1f}")
            else:
                smart_a = de.head_to_head_optimize(S, keep, base_b.totals, pool=pool_a)
                rule_params = dict(p_win=smart_a.stats['p_win'])
                print(f"  rule=head_to_head  p_win={smart_a.stats['p_win']:.3f}")
        except ValueError as e:
            print(f"  skip: {e}")
            continue

        same = set(base_a.lineup['player']) == set(smart_a.lineup['player'])

        # WINNER RULE: score every lineup on ACTUAL realized points, not the
        # model's predicted expectation -- same as league_validate.py's
        # _score_matchups. The opponent (team_b) always plays
        # maximize_expected, so both team_a lineups are judged against the
        # same fixed opponent total, making baseline-won vs. rule-won a fair
        # paired comparison for this one matchup.
        base_a_realized = realized_total(keep, base_a.lineup['player'].to_list())
        smart_a_realized = realized_total(keep, smart_a.lineup['player'].to_list())
        base_b_realized = realized_total(keep, base_b.lineup['player'].to_list())
        base_a_won = base_a_realized > base_b_realized
        smart_a_won = smart_a_realized > base_b_realized

        print(f"  baseline (maximize_expected)  E[total]={base_a.stats['expected']:.1f}")
        print(describe_lineup(base_a, keep, expected_by_player))
        print(f"  {args.rule}{'  (SAME as baseline)' if same else '  (DIFFERENT from baseline)'}  "
             f"E[total]={smart_a.stats['expected']:.1f}")
        print(describe_lineup(smart_a, keep, expected_by_player))
        print(f"  --- winner (actual points) ---")
        print(f"    {opp} (opponent, maximize_expected) scored {base_b_realized:.1f}")
        print(f"    baseline scored {base_a_realized:.1f}  -> {'WIN' if base_a_won else 'LOSS'}")
        print(f"    {args.rule} scored {smart_a_realized:.1f}  -> {'WIN' if smart_a_won else 'LOSS'}"
             f"{'  (same result as baseline)' if base_a_won == smart_a_won else '  (DIFFERENT result than baseline!)'}")
        print()

        if args.csv_out:
            csv_rows.extend(lineup_rows(
                base_a, keep, expected_by_player, season=args.season, week=week, team=args.team,
                opp_team=opp, rule=args.rule, lineup_type='baseline', same_as_baseline=same,
                lineup_realized=base_a_realized, opp_realized=base_b_realized, won=base_a_won,
                rule_params={}))
            csv_rows.extend(lineup_rows(
                smart_a, keep, expected_by_player, season=args.season, week=week, team=args.team,
                opp_team=opp, rule=args.rule, lineup_type='rule', same_as_baseline=same,
                lineup_realized=smart_a_realized, opp_realized=base_b_realized, won=smart_a_won,
                rule_params=rule_params))

    if args.csv_out:
        if csv_rows:
            os.makedirs(os.path.dirname(args.csv_out) or '.', exist_ok=True)
            pl.DataFrame(csv_rows).write_csv(args.csv_out)
            print(f"wrote {len(csv_rows)} rows -> {args.csv_out}")
        else:
            print(f"no rows to write (every week skipped) -- {args.csv_out} not created")


if __name__ == '__main__':
    main()
