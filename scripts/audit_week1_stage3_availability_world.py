from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import replace
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
from monster.sim.availability_world import sample_team_availability_world
from monster.sim.game_flow_lookup import build_team_game_flow_policy
from monster.sim.game_loop_v13 import simulate_game
from monster.sim.intent_ecology import build_intent_ecology
from monster.sim.rushing_roles import sample_event_rush_share_plan
from monster.snapshot.league import compile_team_state_map


def _rows(path: Path) -> list[dict[str, object]]:
    return _read(path).to_dicts()


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


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
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    policy = _read(args.policy)
    personnel = _read(args.personnel)
    usage = _read(args.player_usage)
    situation_context = _read(args.situation_context)
    league_neutral_pass_rate, situational_pass_rates = _situational_context(situation_context)

    pools = apply_health_to_skill_pools(
        compile_current_skill_pools(personnel, usage, policy=policy), personnel
    )
    reality = compile_player_reality_inputs(personnel, game_date=GAME_DATE)
    units = compile_league_unit_player_map(personnel)
    states = {}
    for away, home in MATCHUPS:
        states.update(compile_team_state_map(policy, {away: home, home: away}))

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

    flow_policy = {}
    intent_ecology = {}
    for pair in MATCHUPS:
        for team in pair:
            flow_policy[team] = build_team_game_flow_policy(
                team_id=team,
                league_rows=game_flow_league,
                team_rows=game_flow_team,
                team_neutral_rate=pools[team].neutral_pass_rate,
                league_neutral_rate=league_neutral_pass_rate,
            )
            intent_ecology[team] = build_intent_ecology(
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

    defenses = {team: _defensive_unit(units[team]) for pair in MATCHUPS for team in pair}
    totals: dict[str, list[float]] = defaultdict(list)
    away_scores: dict[str, list[float]] = defaultdict(list)
    home_scores: dict[str, list[float]] = defaultdict(list)
    active_counts: dict[str, list[float]] = defaultdict(list)
    inactive_counts: dict[str, list[float]] = defaultdict(list)
    qb_starts: dict[str, Counter[str]] = defaultdict(Counter)

    for game_idx, (away, home) in enumerate(MATCHUPS):
        game = f"{away}@{home}"
        for world in range(args.worlds):
            seed = args.seed + game_idx * 1_000_003 + world
            sampled = {}
            teams = {}
            for offset, team in ((401_003, away), (502_009, home)):
                sampled[team] = sample_team_availability_world(
                    pools[team], rng=np.random.default_rng(seed + offset)
                )
                active_counts[team].append(float(len(sampled[team].active_player_ids)))
                inactive_counts[team].append(float(len(sampled[team].inactive_player_ids)))
                qb_starts[team][sampled[team].starting_qb_id] += 1

                # Once availability is sampled, the active player exists with probability 1 in
                # this world. Keep effectiveness-if-active intact but remove the old expected-state
                # ability penalty from active_probability before compiling PlayerIdentity.
                world_reality = {
                    player_id: replace(inputs, active_probability=1.0)
                    for player_id, inputs in reality.items()
                }
                identity = _team_identity(
                    team,
                    sampled[team].pool,
                    world_reality,
                    units[team],
                    states[team],
                    league_neutral_pass_rate=league_neutral_pass_rate,
                    situational_pass_rates=situational_pass_rates,
                )
                teams[team] = replace(
                    identity,
                    game_flow_policy=flow_policy[team],
                    intent_ecology=intent_ecology[team],
                )

            away_plan = sample_event_rush_share_plan(
                sampled[away].pool, rng=np.random.default_rng(seed + 101_003)
            )
            home_plan = sample_event_rush_share_plan(
                sampled[home].pool, rng=np.random.default_rng(seed + 202_007)
            )
            result = simulate_game(
                _with_event_rush_plan(teams[away], away_plan),
                _with_event_rush_plan(teams[home], home_plan),
                away_defense=defenses[away],
                home_defense=defenses[home],
                seed=seed,
            )
            away_points = float(result.final_state.away_score)
            home_points = float(result.final_state.home_score)
            away_scores[game].append(away_points)
            home_scores[game].append(home_points)
            totals[game].append(away_points + home_points)

    game_rows = []
    for away, home in MATCHUPS:
        game = f"{away}@{home}"
        game_rows.append(
            {
                "game": game,
                "worlds": args.worlds,
                "away_points_mean": _mean(away_scores[game]),
                "home_points_mean": _mean(home_scores[game]),
                "total_mean": _mean(totals[game]),
                "away_active_skill_count_mean": _mean(active_counts[away]),
                "home_active_skill_count_mean": _mean(active_counts[home]),
            }
        )

    qb_rows = []
    for team, counts in sorted(qb_starts.items()):
        total = sum(counts.values())
        for player_id, count in counts.most_common():
            qb_rows.append(
                {
                    "team": team,
                    "player_id": player_id,
                    "start_worlds": count,
                    "start_probability": count / total if total else 0.0,
                }
            )

    args.out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(game_rows).write_csv(args.out / "availability_world_game_anatomy.csv")
    pl.DataFrame(qb_rows).write_csv(args.out / "availability_world_qb_starts.csv")
    active_rows = [
        {
            "team": team,
            "active_skill_count_mean": _mean(active_counts[team]),
            "inactive_skill_count_mean": _mean(inactive_counts[team]),
        }
        for team in sorted(active_counts)
    ]
    pl.DataFrame(active_rows).write_csv(args.out / "availability_world_roster_anatomy.csv")

    manifest = {
        "artifact": "Monster v1.3 Stage 3 World-Specific Availability Shadow",
        "market_blind": True,
        "worlds_per_game": args.worlds,
        "seed": args.seed,
        "availability_sampled_once_per_team_game_world": True,
        "inactive_skill_players_removed_from_opportunity_trees": True,
        "active_probability_removed_from_active_world_ability": True,
        "effectiveness_if_active_preserved": True,
        "offensive_skill_personnel_only": True,
        "defensive_and_ol_world_substitution": False,
        "team_level_injury_state_still_present": True,
        "behavior_changed_vs_stage3_control": True,
        "promotion_status": "SHADOW_AVAILABILITY_EXPERIMENT_ONLY",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
