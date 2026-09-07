from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from monster.feature_compile.health_pools import apply_health_to_skill_pools
from monster.feature_compile.league_units import compile_league_unit_player_map
from monster.feature_compile.ol_simulation import compile_ol_simulation_context
from monster.feature_compile.skill_pools import compile_current_skill_pools, compile_player_physical_inputs
from monster.feature_compile.units import apply_team_unit_effects, compile_team_unit_effects
from monster.sim.pipeline import simulate_monster_game
from monster.snapshot.league import compile_team_state_map
from monster.snapshot.model import GameState

MATCHUPS = (
    ("CHI", "CAR"), ("BUF", "HOU"), ("NO", "DET"), ("CLE", "JAX"),
    ("TB", "CIN"), ("ATL", "PIT"), ("NYJ", "TEN"), ("BAL", "IND"),
    ("ARI", "LAC"), ("WAS", "PHI"), ("MIA", "LV"), ("GB", "MIN"),
)
GAME_DATE = date(2026, 9, 13)


def _read(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path) if path.suffix == ".parquet" else pl.read_csv(path)


def _unit_context(personnel: pl.DataFrame, historical_ol: pl.DataFrame | None):
    unit_map = compile_league_unit_player_map(personnel)
    rows = []
    for team_id, players in unit_map.items():
        effects, _ = compile_team_unit_effects(players)
        rows.append({"team_id": team_id, **effects.__dict__})
    unit_effects = pl.DataFrame(rows)
    ol = compile_ol_simulation_context(personnel, unit_effects, historical_ol)
    return unit_map, {str(row["team_id"]): row for row in ol.to_dicts()}


def _strengthened_state(base, unit_players, ol_row):
    effects, _ = compile_team_unit_effects(unit_players)
    state = apply_team_unit_effects(base, effects)
    return replace(
        state,
        offensive_line_continuity=float(ol_row["offensive_line_continuity"]),
        offensive_line_uncertainty=float(ol_row["offensive_line_uncertainty"]),
    )


def _player_totals(side, player_id: str) -> tuple[float, float, float]:
    stats = side.player_stats.get(player_id)
    if stats is None:
        return (0.0, 0.0, 0.0)
    return (
        float(np.mean(stats["passing_yards"])),
        float(np.mean(stats["receiving_yards"])),
        float(np.mean(stats["rushing_yards"])),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--control-personnel", type=Path, required=True)
    parser.add_argument("--health-personnel", type=Path, required=True)
    parser.add_argument("--player-usage", type=Path, required=True)
    parser.add_argument("--ol-outcomes", type=Path, default=None)
    parser.add_argument("--worlds", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026090705)
    parser.add_argument("--out", type=Path, default=Path("artifacts/health-pressure-test"))
    args = parser.parse_args()

    policy = _read(args.policy)
    control_personnel = _read(args.control_personnel)
    health_personnel = _read(args.health_personnel)
    usage = _read(args.player_usage)
    historical_ol = _read(args.ol_outcomes) if args.ol_outcomes and args.ol_outcomes.exists() else None

    control_units, control_ol = _unit_context(control_personnel, historical_ol)
    health_units, health_ol = _unit_context(health_personnel, historical_ol)
    control_pools = compile_current_skill_pools(control_personnel, usage, policy=policy)
    health_pools = apply_health_to_skill_pools(
        compile_current_skill_pools(health_personnel, usage, policy=policy), health_personnel
    )
    physical_inputs = compile_player_physical_inputs(control_personnel, game_date=GAME_DATE)

    game_rows: list[dict] = []
    player_rows: list[dict] = []
    for idx, (away, home) in enumerate(MATCHUPS):
        opponents = {away: home, home: away}
        base_states = compile_team_state_map(policy, opponents)
        control_away = _strengthened_state(base_states[away], control_units[away], control_ol[away])
        control_home = _strengthened_state(base_states[home], control_units[home], control_ol[home])
        health_away = _strengthened_state(base_states[away], health_units[away], health_ol[away])
        health_home = _strengthened_state(base_states[home], health_units[home], health_ol[home])
        seed = args.seed + idx * 10_007

        control = simulate_monster_game(
            GameState(f"{away}@{home}-control", control_away, control_home),
            control_pools[away], control_pools[home], worlds=args.worlds, seed=seed,
            player_inputs=physical_inputs,
        )
        health = simulate_monster_game(
            GameState(f"{away}@{home}-health", health_away, health_home),
            health_pools[away], health_pools[home], worlds=args.worlds, seed=seed,
            player_inputs=physical_inputs,
        )
        c_game = control.game_worlds
        h_game = health.game_worlds
        c_total = c_game.away_points + c_game.home_points
        h_total = h_game.away_points + h_game.home_points
        game_rows.append({
            "game": f"{away}@{home}", "seed": seed, "worlds": args.worlds,
            "control_away_mean": float(c_game.away_points.mean()),
            "health_away_mean": float(h_game.away_points.mean()),
            "delta_away_mean": float(h_game.away_points.mean() - c_game.away_points.mean()),
            "control_home_mean": float(c_game.home_points.mean()),
            "health_home_mean": float(h_game.home_points.mean()),
            "delta_home_mean": float(h_game.home_points.mean() - c_game.home_points.mean()),
            "control_total_mean": float(c_total.mean()),
            "health_total_mean": float(h_total.mean()),
            "delta_total_mean": float(h_total.mean() - c_total.mean()),
            "control_total_p90": float(np.quantile(c_total, 0.90)),
            "health_total_p90": float(np.quantile(h_total, 0.90)),
            "delta_total_p90": float(np.quantile(h_total, 0.90) - np.quantile(c_total, 0.90)),
        })

        for team_id, c_pool, h_pool, c_side, h_side in (
            (away, control.snapshot.away_pool, health.snapshot.away_pool, control.allocation_worlds.away, health.allocation_worlds.away),
            (home, control.snapshot.home_pool, health.snapshot.home_pool, control.allocation_worlds.home, health.allocation_worlds.home),
        ):
            names = {p.player_id: (p.display_name, p.position) for p in c_pool.players}
            names.update({p.player_id: (p.display_name, p.position) for p in h_pool.players})
            h_players = {p.player_id: p for p in h_pool.players}
            for player_id, (name, position) in names.items():
                c_pass, c_rec, c_rush = _player_totals(c_side, player_id)
                h_pass, h_rec, h_rush = _player_totals(h_side, player_id)
                hp = h_players.get(player_id)
                player_rows.append({
                    "game": f"{away}@{home}", "team_id": team_id, "player_id": player_id,
                    "player": name, "position": position,
                    "health_active_probability": hp.active_probability if hp else 0.0,
                    "health_effectiveness_if_active": hp.effectiveness_if_active if hp else 0.0,
                    "control_passing_yards_mean": c_pass, "health_passing_yards_mean": h_pass,
                    "delta_passing_yards_mean": h_pass - c_pass,
                    "control_receiving_yards_mean": c_rec, "health_receiving_yards_mean": h_rec,
                    "delta_receiving_yards_mean": h_rec - c_rec,
                    "control_rushing_yards_mean": c_rush, "health_rushing_yards_mean": h_rush,
                    "delta_rushing_yards_mean": h_rush - c_rush,
                    "combined_absolute_yardage_shift": abs(h_pass-c_pass)+abs(h_rec-c_rec)+abs(h_rush-c_rush),
                })

    games = pl.DataFrame(game_rows)
    players = pl.DataFrame(player_rows)
    args.out.mkdir(parents=True, exist_ok=True)
    games.write_csv(args.out / "game_health_ab.csv")
    players.write_csv(args.out / "player_health_ab.csv")
    players.sort("combined_absolute_yardage_shift", descending=True).head(60).write_csv(
        args.out / "top_player_health_changes.csv"
    )

    max_total = float(games.select(pl.col("delta_total_mean").abs().max()).item())
    max_team = float(
        games.select(pl.max_horizontal(pl.col("delta_away_mean").abs(), pl.col("delta_home_mean").abs()).max()).item()
    )
    max_player = float(players.select(pl.col("combined_absolute_yardage_shift").max()).item())
    manifest = {
        "artifact": "Monster Week 1 Health Matched-Seed A/B",
        "worlds_per_game": args.worlds,
        "base_seed": args.seed,
        "games": [f"{a}@{h}" for a, h in MATCHUPS],
        "control": "same football model with current health evidence neutralized by baseline personnel",
        "health": "explicit P(active), effectiveness-if-active, roster/OL participation, and verified Week 1 health evidence",
        "max_abs_game_total_mean_shift": max_total,
        "max_abs_team_points_mean_shift": max_team,
        "max_abs_player_combined_yardage_mean_shift": max_player,
        "guardrail_pass": max_total <= 5.0 and max_team <= 4.0 and max_player <= 35.0,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(games.sort("delta_total_mean", descending=True))
    print(players.sort("combined_absolute_yardage_shift", descending=True).head(30))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
