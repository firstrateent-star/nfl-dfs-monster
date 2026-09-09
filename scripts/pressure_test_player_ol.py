from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

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
    ("BUF", "HOU"),
    ("GB", "MIN"),
    ("BAL", "IND"),
    ("TB", "CIN"),
)
GAME_DATE = date(2026, 9, 13)


def _read(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path) if path.suffix == ".parquet" else pl.read_csv(path)


def _mean(array: np.ndarray) -> float:
    return float(np.mean(array.astype(float)))


def _q(array: np.ndarray, q: float) -> float:
    return float(np.quantile(array.astype(float), q))


def _team_summary(side) -> dict[str, float]:
    total_receiving = np.sum(
        np.column_stack([stats["receiving_yards"] for stats in side.player_stats.values()]),
        axis=1,
    )
    total_rushing = np.sum(
        np.column_stack([stats["rushing_yards"] for stats in side.player_stats.values()]),
        axis=1,
    )
    return {
        "plays_mean": _mean(side.team_plays),
        "dropbacks_mean": _mean(side.team_dropbacks),
        "sacks_mean": _mean(side.team_sacks),
        "pass_attempts_mean": _mean(side.team_pass_attempts),
        "targets_mean": _mean(side.team_targets),
        "rush_attempts_mean": _mean(side.team_rush_attempts),
        "receiving_yards_mean": _mean(total_receiving),
        "receiving_yards_p90": _q(total_receiving, 0.90),
        "rushing_yards_mean": _mean(total_rushing),
        "rushing_yards_p90": _q(total_rushing, 0.90),
        "pass_disruption_mean": _mean(side.pass_disruption),
        "pass_disruption_p90": _q(side.pass_disruption, 0.90),
        "run_efficiency_mean": _mean(side.run_efficiency),
        "run_efficiency_p10": _q(side.run_efficiency, 0.10),
        "run_efficiency_p90": _q(side.run_efficiency, 0.90),
    }


def _player_rows(game: str, team_id: str, pool, control_side, strengthened_side) -> list[dict]:
    rows: list[dict] = []
    for player in pool.players:
        a = control_side.player_stats[player.player_id]
        b = strengthened_side.player_stats[player.player_id]
        row = {
            "game": game,
            "team_id": team_id,
            "player_id": player.player_id,
            "player": player.display_name,
            "position": player.position,
            "active_probability": player.active_probability,
            "role_uncertainty": player.role_uncertainty,
        }
        for stat in (
            "pass_attempts",
            "passing_yards",
            "passing_tds",
            "targets",
            "receptions",
            "receiving_yards",
            "receiving_tds",
            "rush_attempts",
            "rushing_yards",
            "rushing_tds",
        ):
            a_mean = _mean(a[stat])
            b_mean = _mean(b[stat])
            a_p90 = _q(a[stat], 0.90)
            b_p90 = _q(b[stat], 0.90)
            row[f"control_{stat}_mean"] = a_mean
            row[f"strengthened_{stat}_mean"] = b_mean
            row[f"delta_{stat}_mean"] = b_mean - a_mean
            row[f"control_{stat}_p90"] = a_p90
            row[f"strengthened_{stat}_p90"] = b_p90
            row[f"delta_{stat}_p90"] = b_p90 - a_p90
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--personnel", type=Path, required=True)
    parser.add_argument("--player-usage", type=Path, required=True)
    parser.add_argument("--ol-outcomes", type=Path, default=None)
    parser.add_argument("--worlds", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026090704)
    parser.add_argument("--out", type=Path, default=Path("artifacts/player-ol-pressure-test"))
    args = parser.parse_args()

    policy = _read(args.policy)
    personnel = _read(args.personnel)
    player_usage = _read(args.player_usage)
    historical_ol = _read(args.ol_outcomes) if args.ol_outcomes and args.ol_outcomes.exists() else None

    unit_map = compile_league_unit_player_map(personnel)
    unit_rows: list[dict] = []
    for team_id, players in unit_map.items():
        effects, _ = compile_team_unit_effects(players)
        unit_rows.append({"team_id": team_id, **effects.__dict__})
    unit_effects = pl.DataFrame(unit_rows)
    ol_context = compile_ol_simulation_context(personnel, unit_effects, historical_ol)
    context_map = {str(row["team_id"]): row for row in ol_context.to_dicts()}

    pools = compile_current_skill_pools(personnel, player_usage, policy=policy)
    physical_inputs = compile_player_physical_inputs(personnel, game_date=GAME_DATE)

    team_rows: list[dict] = []
    player_rows: list[dict] = []
    for game_index, (away, home) in enumerate(MATCHUPS):
        game_name = f"{away}@{home}"
        if away not in pools or home not in pools:
            raise ValueError(f"Missing current skill pool for {game_name}")

        states = compile_team_state_map(policy, {away: home, home: away})
        away_effects, _ = compile_team_unit_effects(unit_map[away])
        home_effects, _ = compile_team_unit_effects(unit_map[home])
        away_state = apply_team_unit_effects(states[away], away_effects)
        home_state = apply_team_unit_effects(states[home], home_effects)

        seed = args.seed + game_index * 10_000
        control = simulate_monster_game(
            GameState(game_id=f"{game_name}-control", away=away_state, home=home_state),
            pools[away],
            pools[home],
            worlds=args.worlds,
            seed=seed,
            player_inputs=physical_inputs,
        )

        away_ctx = context_map[away]
        home_ctx = context_map[home]
        strengthened_away = replace(
            away_state,
            offensive_line_continuity=float(away_ctx["offensive_line_continuity"]),
            offensive_line_uncertainty=float(away_ctx["offensive_line_uncertainty"]),
        )
        strengthened_home = replace(
            home_state,
            offensive_line_continuity=float(home_ctx["offensive_line_continuity"]),
            offensive_line_uncertainty=float(home_ctx["offensive_line_uncertainty"]),
        )
        strengthened = simulate_monster_game(
            GameState(
                game_id=f"{game_name}-strengthened",
                away=strengthened_away,
                home=strengthened_home,
            ),
            pools[away],
            pools[home],
            worlds=args.worlds,
            seed=seed,
            player_inputs=physical_inputs,
        )

        for team_id, control_side, strengthened_side, ctx in (
            (away, control.allocation_worlds.away, strengthened.allocation_worlds.away, away_ctx),
            (home, control.allocation_worlds.home, strengthened.allocation_worlds.home, home_ctx),
        ):
            a = _team_summary(control_side)
            b = _team_summary(strengthened_side)
            row = {
                "game": game_name,
                "team_id": team_id,
                "seed": seed,
                "worlds": args.worlds,
                "ol_continuity": float(ctx["offensive_line_continuity"]),
                "ol_uncertainty": float(ctx["offensive_line_uncertainty"]),
                "pass_protection_effect": float(ctx.get("pass_protection_effect") or 0.0),
                "run_block_effect": float(ctx.get("run_block_effect") or 0.0),
            }
            for key, value in a.items():
                row[f"control_{key}"] = value
            for key, value in b.items():
                row[f"strengthened_{key}"] = value
                row[f"delta_{key}"] = value - a[key]
            team_rows.append(row)

        player_rows.extend(
            _player_rows(
                game_name,
                away,
                control.snapshot.away_pool,
                control.allocation_worlds.away,
                strengthened.allocation_worlds.away,
            )
        )
        player_rows.extend(
            _player_rows(
                game_name,
                home,
                control.snapshot.home_pool,
                control.allocation_worlds.home,
                strengthened.allocation_worlds.home,
            )
        )

    team_result = pl.DataFrame(team_rows)
    player_result = pl.DataFrame(player_rows)
    args.out.mkdir(parents=True, exist_ok=True)
    team_result.write_csv(args.out / "team_pathway_ab.csv")
    player_result.write_csv(args.out / "player_pathway_ab.csv")
    ol_context.write_csv(args.out / "ol_simulation_context.csv")

    active_players = player_result.filter(
        (pl.col("control_pass_attempts_mean") >= 5)
        | (pl.col("control_targets_mean") >= 2)
        | (pl.col("control_rush_attempts_mean") >= 3)
    )
    top_changes = active_players.with_columns(
        (
            pl.col("delta_passing_yards_mean").abs()
            + pl.col("delta_receiving_yards_mean").abs()
            + pl.col("delta_rushing_yards_mean").abs()
        ).alias("absolute_yardage_mean_shift")
    ).sort("absolute_yardage_mean_shift", descending=True)
    top_changes.head(40).write_csv(args.out / "top_player_changes.csv")

    max_sack_shift = float(team_result.select(pl.col("delta_sacks_mean").abs().max()).item())
    max_attempt_shift = float(
        team_result.select(pl.col("delta_pass_attempts_mean").abs().max()).item()
    )
    max_player_yard_shift = float(
        top_changes.select(pl.col("absolute_yardage_mean_shift").max()).item()
        if top_changes.height
        else 0.0
    )
    manifest = {
        "artifact": "Monster Player-Level OL Pathway A/B",
        "worlds_per_game": args.worlds,
        "base_seed": args.seed,
        "games": [f"{away}@{home}" for away, home in MATCHUPS],
        "control": "current unit capability; OL continuity/uncertainty neutral",
        "strengthened": "same inputs plus measured OL continuity/uncertainty propagated into score, sacks, targets, pass yards, and rush efficiency",
        "historical_ol_outcomes_production_authority": 0.0,
        "max_abs_team_sack_mean_delta": max_sack_shift,
        "max_abs_team_pass_attempt_mean_delta": max_attempt_shift,
        "max_active_player_combined_yardage_mean_shift": max_player_yard_shift,
        "guardrail_pass": max_sack_shift <= 0.35 and max_attempt_shift <= 1.0 and max_player_yard_shift <= 8.0,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(team_result)
    print(top_changes.head(25))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
