from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import polars as pl

from monster.dfs.fanduel import score_offensive_player_worlds
from monster.dfs.turnovers import attach_turnover_stats, attribute_team_turnovers
from monster.dfs.world_matrix import build_defense_world_rows
from monster.feature_compile.health_pools import apply_health_to_skill_pools
from monster.feature_compile.league_units import compile_league_unit_player_map
from monster.feature_compile.ol_simulation import compile_ol_simulation_context
from monster.feature_compile.skill_pools import (
    compile_current_skill_pools,
    compile_player_physical_inputs,
)
from monster.feature_compile.units import apply_team_unit_effects, compile_team_unit_effects
from monster.sim.pipeline import simulate_monster_game
from monster.snapshot.league import compile_team_state_map
from monster.snapshot.model import GameState

MATCHUPS = (
    ("CHI", "CAR"), ("BUF", "HOU"), ("NO", "DET"), ("CLE", "JAC"),
    ("TB", "CIN"), ("ATL", "PIT"), ("NYJ", "TEN"), ("BAL", "IND"),
    ("ARI", "LAC"), ("WAS", "PHI"), ("MIA", "LV"), ("GB", "MIN"),
)
GAME_DATE = date(2026, 9, 13)


def _read(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path) if path.suffix == ".parquet" else pl.read_csv(path)


def _q(x: np.ndarray, q: float) -> float:
    return float(np.quantile(x.astype(float), q))


def _unit_context(personnel: pl.DataFrame, historical_ol: pl.DataFrame | None):
    unit_map = compile_league_unit_player_map(personnel)
    rows = []
    for team_id, players in unit_map.items():
        effects, _ = compile_team_unit_effects(players)
        rows.append({"team_id": team_id, **effects.__dict__})
    effects_frame = pl.DataFrame(rows)
    ol = compile_ol_simulation_context(personnel, effects_frame, historical_ol)
    return unit_map, {str(row["team_id"]): row for row in ol.to_dicts()}


def _state(base, unit_players, ol_row):
    effects, _ = compile_team_unit_effects(unit_players)
    state = apply_team_unit_effects(base, effects)
    return replace(state, offensive_line_continuity=float(ol_row["offensive_line_continuity"]), offensive_line_uncertainty=float(ol_row["offensive_line_uncertainty"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--personnel", type=Path, required=True)
    parser.add_argument("--player-usage", type=Path, required=True)
    parser.add_argument("--ol-outcomes", type=Path, default=None)
    parser.add_argument("--worlds", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026090817)
    parser.add_argument("--interception-fraction", type=float, default=0.5922258892555923)
    parser.add_argument("--out", type=Path, default=Path("artifacts/week1-fantasy-complete"))
    args = parser.parse_args()

    policy = _read(args.policy)
    personnel = _read(args.personnel)
    usage = _read(args.player_usage)
    historical_ol = _read(args.ol_outcomes) if args.ol_outcomes and args.ol_outcomes.exists() else None
    unit_map, ol_map = _unit_context(personnel, historical_ol)
    pools = apply_health_to_skill_pools(compile_current_skill_pools(personnel, usage, policy=policy), personnel)
    physical_inputs = compile_player_physical_inputs(personnel, game_date=GAME_DATE)

    rows: list[dict] = []
    world_player_ids: list[str] = []
    world_scores: list[np.ndarray] = []
    game_world_rows: list[dict] = []
    conservation_failures = 0
    for idx, (away, home) in enumerate(MATCHUPS):
        game = f"{away}@{home}"
        states = compile_team_state_map(policy, {away: home, home: away})
        result = simulate_monster_game(GameState(game, _state(states[away], unit_map[away], ol_map[away]), _state(states[home], unit_map[home], ol_map[home])), pools[away], pools[home], worlds=args.worlds, seed=args.seed + idx * 10007, player_inputs=physical_inputs)
        game_world_rows.append({
            "game": game, "away_team": away, "home_team": home,
            "away_points": result.game_worlds.away_points, "home_points": result.game_worlds.home_points,
            "away_turnovers": result.game_worlds.away_turnovers, "home_turnovers": result.game_worlds.home_turnovers,
            "away_pass_disruption": result.game_worlds.away_pass_disruption, "home_pass_disruption": result.game_worlds.home_pass_disruption,
            "away_pass_attempts": result.allocation_worlds.away.team_pass_attempts, "home_pass_attempts": result.allocation_worlds.home.team_pass_attempts,
        })
        for side_idx, (team, pool, allocation, team_turnovers) in enumerate(((away, result.snapshot.away_pool, result.allocation_worlds.away, result.game_worlds.away_turnovers), (home, result.snapshot.home_pool, result.allocation_worlds.home, result.game_worlds.home_turnovers))):
            attribution = attribute_team_turnovers(team_turnovers, allocation, pool, interception_fraction=args.interception_fraction, seed=args.seed + idx * 10007 + 900001 + side_idx)
            attach_turnover_stats(allocation, attribution)
            attributed = sum(attribution.interceptions.values()) + sum(attribution.fumbles_lost.values())
            if not np.array_equal(attributed.astype(np.int16), team_turnovers.astype(np.int16)):
                conservation_failures += 1
            for player in pool.players:
                stats = allocation.player_stats[player.player_id]
                fd = score_offensive_player_worlds(stats)
                world_player_ids.append(player.player_id)
                world_scores.append(fd.astype(np.float32))
                rows.append({"game": game, "team_id": team, "player_id": player.player_id, "player": player.display_name, "position": player.position, "active_probability": player.active_probability, "fd_mean": float(fd.mean()), "fd_p20": _q(fd, .20), "fd_p50": _q(fd, .50), "fd_p75": _q(fd, .75), "fd_p90": _q(fd, .90), "fd_p95": _q(fd, .95), "fd_p99": _q(fd, .99), "p20_plus": float((fd >= 20).mean()), "p25_plus": float((fd >= 25).mean()), "p30_plus": float((fd >= 30).mean()), "interceptions_mean": float(stats["interceptions"].mean()), "fumbles_lost_mean": float(stats["fumbles_lost"].mean())})

    game_payload: dict[str, np.ndarray] = {}
    game_keys = ("away_points", "home_points", "away_turnovers", "home_turnovers", "away_pass_disruption", "home_pass_disruption", "away_pass_attempts", "home_pass_attempts")
    for idx, row in enumerate(game_world_rows):
        for key in game_keys:
            value = row[key]
            if value is not None:
                game_payload[f"g{idx}_{key}"] = np.asarray(value)
    game_index = [{"game_index": idx, "game": row["game"], "away_team": row["away_team"], "home_team": row["home_team"]} for idx, row in enumerate(game_world_rows)]
    team_pass_attempts = {}
    for idx, game in enumerate(game_index):
        team_pass_attempts[str(game["away_team"])] = game_payload[f"g{idx}_away_pass_attempts"]
        team_pass_attempts[str(game["home_team"])] = game_payload[f"g{idx}_home_pass_attempts"]
    defense_rows = build_defense_world_rows(game_index=game_index, game_worlds=game_payload, team_pass_attempts=team_pass_attempts, seed=args.seed + 8000001)
    matchup_by_team = {team: f"{away}@{home}" for away, home in MATCHUPS for team in (away, home)}
    for defense in defense_rows:
        fd = defense.scores
        player_id = f"DST:{defense.team_id}"
        world_player_ids.append(player_id)
        world_scores.append(fd.astype(np.float32))
        rows.append({"game": matchup_by_team[defense.team_id], "team_id": defense.team_id, "player_id": player_id, "player": defense.player, "position": "D", "active_probability": 1.0, "fd_mean": float(fd.mean()), "fd_p20": _q(fd, .20), "fd_p50": _q(fd, .50), "fd_p75": _q(fd, .75), "fd_p90": _q(fd, .90), "fd_p95": _q(fd, .95), "fd_p99": _q(fd, .99), "p20_plus": float((fd >= 20).mean()), "p25_plus": float((fd >= 25).mean()), "p30_plus": float((fd >= 30).mean()), "interceptions_mean": None, "fumbles_lost_mean": None})

    args.out.mkdir(parents=True, exist_ok=True)
    frame = pl.DataFrame(rows).sort("fd_mean", descending=True)
    frame.write_csv(args.out / "player_distributions.csv")
    np.savez_compressed(args.out / "player_worlds.npz", player_ids=np.asarray(world_player_ids), fd_points=np.stack(world_scores))
    np.savez_compressed(args.out / "game_worlds.npz", **game_payload)
    (args.out / "game_world_index.json").write_text(json.dumps(game_index, indent=2) + "\n")
    manifest = {"artifact": "Monster Week 1 Fantasy-Complete Player Worlds", "generated_at_utc": datetime.now(UTC).isoformat(), "game_date": GAME_DATE.isoformat(), "worlds_per_game": args.worlds, "games": len(MATCHUPS), "simulated_game_worlds": args.worlds * len(MATCHUPS), "seed": args.seed, "interception_fraction": args.interception_fraction, "turnover_conservation_failures": conservation_failures, "offensive_player_rows": frame.filter(pl.col("position") != "D").height, "defense_rows": frame.filter(pl.col("position") == "D").height, "player_rows": frame.height, "joint_player_worlds_persisted": True, "joint_world_shape": [len(world_player_ids), args.worlds], "game_worlds_persisted": True, "team_pass_attempt_worlds_persisted": True, "defense_worlds_persisted": True, "blocked_kick_points_modeled": False, "market_blind_football": True, "dfs_layer_downstream_only": True}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    print(frame.select("position", "player", "team_id", "fd_mean", "fd_p90", "fd_p99").head(30))
    if conservation_failures:
        raise SystemExit("Turnover attribution failed world-level conservation")
    if len(defense_rows) != 24:
        raise SystemExit("D/ST world construction did not produce 24 teams")


if __name__ == "__main__":
    main()
