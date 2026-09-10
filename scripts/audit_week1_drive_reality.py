from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import polars as pl
from run_week1_v13_first_sim import (
    GAME_DATE,
    MATCHUPS,
    _defensive_unit,
    _read,
    _situational_context,
    _team_identity,
    _with_event_rush_plan,
)

from monster.feature_compile.health_pools import apply_health_to_skill_pools
from monster.feature_compile.league_units import compile_league_unit_player_map
from monster.feature_compile.reality_inputs import compile_player_reality_inputs
from monster.feature_compile.skill_pools import compile_current_skill_pools
from monster.sim.football_state import PossessionTerminal
from monster.sim.game_loop_v13 import simulate_game
from monster.sim.rushing_roles import sample_event_rush_share_plan
from monster.snapshot.league import compile_team_state_map


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _quantile(values: list[float], q: float) -> float:
    return float(np.quantile(values, q)) if values else 0.0


def _summarize(rows: list[dict[str, object]], *, scope: str) -> dict[str, object]:
    if not rows:
        raise ValueError(f"no drive rows available for {scope}")
    terminals = Counter(str(row["terminal"]) for row in rows)
    drives = len(rows)
    touchdowns = terminals[PossessionTerminal.TOUCHDOWN.value]
    field_goals = terminals[PossessionTerminal.FIELD_GOAL.value]
    missed_field_goals = terminals[PossessionTerminal.MISSED_FIELD_GOAL.value]
    punts = terminals[PossessionTerminal.PUNT.value]
    turnovers = terminals[PossessionTerminal.TURNOVER.value]
    downs = terminals[PossessionTerminal.TURNOVER_ON_DOWNS.value]
    safeties = terminals[PossessionTerminal.SAFETY.value]
    period_ends = terminals[PossessionTerminal.HALFTIME.value] + terminals[PossessionTerminal.END_GAME.value]

    red_zone_reached = [row for row in rows if bool(row["red_zone_entered"])]
    red_zone_snaps = [row for row in rows if bool(row["red_zone_snap_seen"])]
    goal_to_go_snaps = [row for row in rows if bool(row["goal_to_go_snap_seen"])]
    explosive_drives = [row for row in rows if int(row["explosive_plays"]) > 0]
    no_explosive_drives = [row for row in rows if int(row["explosive_plays"]) == 0]
    short_field = [row for row in rows if float(row["start_yardline_100"]) >= 50.0]
    long_field = [row for row in rows if float(row["start_yardline_100"]) < 50.0]
    td_rows = [row for row in rows if str(row["terminal"]) == PossessionTerminal.TOUCHDOWN.value]

    def _td_count(subset: list[dict[str, object]]) -> int:
        return sum(str(row["terminal"]) == PossessionTerminal.TOUCHDOWN.value for row in subset)

    plays = [float(row["scrimmage_plays"]) for row in rows]
    yards = [float(row["net_scrimmage_yards"]) for row in rows]
    first_downs = [float(row["first_downs"]) for row in rows]
    explosives = [float(row["explosive_plays"]) for row in rows]
    points = [float(row["points"]) for row in rows]
    pressures = [float(row["pressured_dropbacks"]) for row in rows]
    sacks = [float(row["sacks"]) for row in rows]
    starts = [float(row["start_yardline_100"]) for row in rows]

    return {
        "scope": scope,
        "drives": drives,
        "touchdown_rate": _safe_rate(touchdowns, drives),
        "field_goal_rate": _safe_rate(field_goals, drives),
        "missed_field_goal_rate": _safe_rate(missed_field_goals, drives),
        "punt_rate": _safe_rate(punts, drives),
        "turnover_rate": _safe_rate(turnovers, drives),
        "turnover_on_downs_rate": _safe_rate(downs, drives),
        "safety_rate": _safe_rate(safeties, drives),
        "period_end_rate": _safe_rate(period_ends, drives),
        "scoring_drive_rate": _safe_rate(touchdowns + field_goals, drives),
        "points_per_drive": _mean(points),
        "red_zone_reach_rate": _safe_rate(len(red_zone_reached), drives),
        "red_zone_snap_rate": _safe_rate(len(red_zone_snaps), drives),
        "goal_to_go_snap_rate": _safe_rate(len(goal_to_go_snaps), drives),
        "td_per_red_zone_reach": _safe_rate(_td_count(red_zone_reached), len(red_zone_reached)),
        "td_per_red_zone_snap": _safe_rate(_td_count(red_zone_snaps), len(red_zone_snaps)),
        "td_per_goal_to_go_snap": _safe_rate(_td_count(goal_to_go_snaps), len(goal_to_go_snaps)),
        "td_without_red_zone_snap_share": _safe_rate(
            sum(not bool(row["red_zone_snap_seen"]) for row in td_rows), len(td_rows)
        ),
        "explosive_drive_rate": _safe_rate(len(explosive_drives), drives),
        "td_given_explosive_drive": _safe_rate(_td_count(explosive_drives), len(explosive_drives)),
        "td_given_no_explosive_drive": _safe_rate(
            _td_count(no_explosive_drives), len(no_explosive_drives)
        ),
        "short_field_start_rate": _safe_rate(len(short_field), drives),
        "td_given_short_field": _safe_rate(_td_count(short_field), len(short_field)),
        "td_given_long_field": _safe_rate(_td_count(long_field), len(long_field)),
        "scrimmage_plays_per_drive": _mean(plays),
        "scrimmage_plays_p50": _quantile(plays, 0.50),
        "scrimmage_plays_p90": _quantile(plays, 0.90),
        "net_yards_per_drive": _mean(yards),
        "net_yards_p50": _quantile(yards, 0.50),
        "net_yards_p90": _quantile(yards, 0.90),
        "first_downs_per_drive": _mean(first_downs),
        "explosive_plays_per_drive": _mean(explosives),
        "pressured_dropbacks_per_drive": _mean(pressures),
        "sacks_per_drive": _mean(sacks),
        "start_yardline_100_mean": _mean(starts),
        "start_yardline_100_p10": _quantile(starts, 0.10),
        "start_yardline_100_p50": _quantile(starts, 0.50),
        "start_yardline_100_p90": _quantile(starts, 0.90),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--personnel", type=Path, required=True)
    parser.add_argument("--player-usage", type=Path, required=True)
    parser.add_argument("--situation-context", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026091021)
    parser.add_argument("--out", type=Path, default=Path("artifacts/week1-drive-reality"))
    args = parser.parse_args()

    policy = _read(args.policy)
    personnel = _read(args.personnel)
    usage = _read(args.player_usage)
    situation_context = _read(args.situation_context)
    league_neutral_pass_rate, situational_pass_rates = _situational_context(situation_context)

    pools = apply_health_to_skill_pools(
        compile_current_skill_pools(personnel, usage, policy=policy),
        personnel,
    )
    reality = compile_player_reality_inputs(personnel, game_date=GAME_DATE)
    units = compile_league_unit_player_map(personnel)
    states = {}
    for away, home in MATCHUPS:
        states.update(compile_team_state_map(policy, {away: home, home: away}))

    teams = {
        team: _team_identity(
            team,
            pools[team],
            reality,
            units[team],
            states[team],
            league_neutral_pass_rate=league_neutral_pass_rate,
            situational_pass_rates=situational_pass_rates,
        )
        for pair in MATCHUPS
        for team in pair
    }
    defenses = {team: _defensive_unit(units[team]) for pair in MATCHUPS for team in pair}

    rows: list[dict[str, object]] = []
    world_rows: list[dict[str, object]] = []
    for game_idx, (away, home) in enumerate(MATCHUPS):
        game = f"{away}@{home}"
        for world in range(args.worlds):
            seed = args.seed + game_idx * 1_000_003 + world
            away_plan = sample_event_rush_share_plan(
                pools[away], rng=np.random.default_rng(seed + 101_003)
            )
            home_plan = sample_event_rush_share_plan(
                pools[home], rng=np.random.default_rng(seed + 202_007)
            )
            result = simulate_game(
                _with_event_rush_plan(teams[away], away_plan),
                _with_event_rush_plan(teams[home], home_plan),
                away_defense=defenses[away],
                home_defense=defenses[home],
                seed=seed,
            )
            for drive_index, trace in enumerate(result.drive_traces):
                row = asdict(trace)
                row["terminal"] = trace.terminal.value
                row.update({"game": game, "world": world, "drive_index": drive_index})
                rows.append(row)
            world_rows.append(
                {
                    "game": game,
                    "world": world,
                    "traced_drives": len(result.drive_traces),
                    "legacy_drive_counter": result.drives,
                    "scoreboard_points": result.final_state.away_score + result.final_state.home_score,
                }
            )

    if not rows:
        raise ValueError("simulation produced no drive traces")

    overall = _summarize(rows, scope="all_week1_simulated_drives")
    by_game = []
    for away, home in MATCHUPS:
        game = f"{away}@{home}"
        by_game.append(_summarize([row for row in rows if row["game"] == game], scope=game))

    args.out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(args.out / "simulated_drive_traces.parquet")
    pl.DataFrame(world_rows).write_csv(args.out / "drive_counts_by_world.csv")
    pl.DataFrame([overall]).write_csv(args.out / "simulated_drive_reality_overall.csv")
    pl.DataFrame(by_game).write_csv(args.out / "simulated_drive_reality_by_game.csv")

    manifest = {
        "artifact": "Monster v1.3 Week 1 Passive Drive Reality Audit",
        "season": 2026,
        "week": 1,
        "games": len(MATCHUPS),
        "worlds_per_game": args.worlds,
        "seed": args.seed,
        "market_blind": True,
        "behavior_changed_by_audit": False,
        "explosive_play_definition": "scrimmage gain >= 15 yards",
        "red_zone_reach_definition": "possession reaches offense-relative yardline >= 80, including scoring plays from outside the red zone",
        "red_zone_snap_definition": "an observed event begins with offense-relative yardline >= 80",
        "legacy_drive_counter_preserved": True,
        "pressure_note": "Pressure is retained as a Monster-only drive diagnostic here; historical pressure comparison requires a separately definition-governed participation join.",
        "files": {
            "raw": "simulated_drive_traces.parquet",
            "world_counts": "drive_counts_by_world.csv",
            "overall": "simulated_drive_reality_overall.csv",
            "by_game": "simulated_drive_reality_by_game.csv",
        },
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest": manifest, "overall": overall}, indent=2))


if __name__ == "__main__":
    main()
