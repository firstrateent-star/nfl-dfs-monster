from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
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
from monster.sim.game_flow_lookup import build_team_game_flow_policy
from monster.sim.game_loop_v13 import simulate_game
from monster.sim.intent_ecology import build_intent_ecology
from monster.sim.rushing_roles import sample_event_rush_share_plan
from monster.snapshot.league import compile_team_state_map


def _rows(path: Path) -> list[dict[str, object]]:
    return (_read(path)).to_dicts()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game-flow-league", type=Path, required=True)
    parser.add_argument("--game-flow-team", type=Path, required=True)
    parser.add_argument("--pass-depth-league", type=Path, required=True)
    parser.add_argument("--pass-depth-team", type=Path, required=True)
    parser.add_argument("--pass-depth-qb", type=Path, required=True)
    parser.add_argument("--pass-depth-outcomes", type=Path, required=True)
    parser.add_argument("--target-depth", type=Path, required=True)
    parser.add_argument("--run-geometry-league", type=Path, required=True)
    parser.add_argument("--run-geometry-team", type=Path, required=True)
    parser.add_argument("--run-geometry-rusher", type=Path, required=True)
    parser.add_argument("--run-geometry-outcomes", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--personnel", type=Path, required=True)
    parser.add_argument("--player-usage", type=Path, required=True)
    parser.add_argument("--situation-context", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=250)
    parser.add_argument("--seed", type=int, default=2026091050)
    parser.add_argument("--out", type=Path, default=Path("artifacts/stage3-drive-survival"))
    args = parser.parse_args()

    policy = _read(args.policy)
    personnel = _read(args.personnel)
    usage = _read(args.player_usage)
    situation_context = _read(args.situation_context)
    league_neutral_pass_rate, situational_pass_rates = _situational_context(situation_context)

    game_flow_league = _rows(args.game_flow_league)
    game_flow_team = _rows(args.game_flow_team)
    pass_league = _rows(args.pass_depth_league)
    pass_team = _rows(args.pass_depth_team)
    pass_qb = _rows(args.pass_depth_qb)
    pass_outcomes = _rows(args.pass_depth_outcomes)
    target_depth = _rows(args.target_depth)
    run_league = _rows(args.run_geometry_league)
    run_team = _rows(args.run_geometry_team)
    run_rusher = _rows(args.run_geometry_rusher)
    run_outcomes = _rows(args.run_geometry_outcomes)

    pools = apply_health_to_skill_pools(
        compile_current_skill_pools(personnel, usage, policy=policy),
        personnel,
    )
    reality = compile_player_reality_inputs(personnel, game_date=GAME_DATE)
    units = compile_league_unit_player_map(personnel)
    states = {}
    for away, home in MATCHUPS:
        states.update(compile_team_state_map(policy, {away: home, home: away}))

    attached_teams: set[str] = set()
    teams = {}
    for pair in MATCHUPS:
        for team in pair:
            identity = _team_identity(
                team,
                pools[team],
                reality,
                units[team],
                states[team],
                league_neutral_pass_rate=league_neutral_pass_rate,
                situational_pass_rates=situational_pass_rates,
            )
            flow_policy = build_team_game_flow_policy(
                team_id=team,
                league_rows=game_flow_league,
                team_rows=game_flow_team,
                team_neutral_rate=pools[team].neutral_pass_rate,
                league_neutral_rate=league_neutral_pass_rate,
            )
            intent = build_intent_ecology(
                team_id=team,
                pass_league_rows=pass_league,
                pass_team_rows=pass_team,
                pass_qb_rows=pass_qb,
                pass_outcome_rows=pass_outcomes,
                target_depth_rows=target_depth,
                run_league_rows=run_league,
                run_team_rows=run_team,
                run_rusher_rows=run_rusher,
                run_outcome_rows=run_outcomes,
            )
            teams[team] = replace(identity, game_flow_policy=flow_policy, intent_ecology=intent)
            attached_teams.add(team)

    if len(attached_teams) != 24:
        raise RuntimeError(f"Stage 3 attached to {len(attached_teams)} teams, expected 24")

    defenses = {team: _defensive_unit(units[team]) for pair in MATCHUPS for team in pair}
    drive_rows: list[dict[str, object]] = []
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
                drive_rows.append(row)
            world_rows.append(
                {
                    "game": game,
                    "world": world,
                    "traced_drives": len(result.drive_traces),
                    "legacy_drive_counter": result.drives,
                    "scoreboard_points": result.final_state.away_score + result.final_state.home_score,
                    "away_points": result.final_state.away_score,
                    "home_points": result.final_state.home_score,
                }
            )

    if not drive_rows:
        raise RuntimeError("Stage 3 drive survival audit produced no traces")

    args.out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(drive_rows).write_parquet(args.out / "stage3_drive_traces.parquet")
    pl.DataFrame(world_rows).write_csv(args.out / "stage3_drive_counts_by_world.csv")

    manifest = {
        "artifact": "Monster v1.3 Stage 3 Drive Survival Shadow Audit",
        "season": 2026,
        "week": 1,
        "games": len(MATCHUPS),
        "worlds_per_game": args.worlds,
        "seed": args.seed,
        "market_blind": True,
        "behavior_changed_by_audit": False,
        "hierarchical_game_flow_active": True,
        "intent_ecology_active": True,
        "stage3_resolution_ecology_active": True,
        "stage3_attached_team_count": len(attached_teams),
        "survival_ledger_active": True,
        "definitions": {
            "series_started": "definition-safe simulated scrimmage snap on first down",
            "series_conversion": "simulated scrimmage gain reaches down-distance without TD/turnover",
            "third_and_long": "third down with distance >= 7 yards",
            "early_down_5plus": "first/second-down scrimmage gain >= 5 yards",
        },
        "files": {
            "drive_traces": "stage3_drive_traces.parquet",
            "world_counts": "stage3_drive_counts_by_world.csv",
        },
        "promotion_status": "SHADOW_DIAGNOSTIC_ONLY",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
