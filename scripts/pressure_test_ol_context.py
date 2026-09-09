from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl

from monster.feature_compile.league_units import compile_league_unit_player_map
from monster.feature_compile.ol_simulation import compile_ol_simulation_context
from monster.feature_compile.units import apply_team_unit_effects, compile_team_unit_effects
from monster.sim.game import simulate_game
from monster.snapshot.league import compile_team_state_map
from monster.snapshot.model import GameState


MATCHUPS = (
    ("BUF", "HOU"),
    ("GB", "MIN"),
    ("BAL", "IND"),
    ("TB", "CIN"),
)


def _summarize(worlds) -> dict[str, float]:
    total = worlds.total.astype(float)
    return {
        "mean_total": float(np.mean(total)),
        "p10_total": float(np.quantile(total, 0.10)),
        "p90_total": float(np.quantile(total, 0.90)),
        "away_mean": float(np.mean(worlds.away_points)),
        "home_mean": float(np.mean(worlds.home_points)),
        "away_td_mean": float(np.mean(worlds.away_touchdowns)),
        "home_td_mean": float(np.mean(worlds.home_touchdowns)),
        "away_turnover_mean": float(np.mean(worlds.away_turnovers)),
        "home_turnover_mean": float(np.mean(worlds.home_turnovers)),
        "p60_plus": float(np.mean(total >= 60.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--personnel", type=Path, required=True)
    parser.add_argument("--ol-outcomes", type=Path, default=None)
    parser.add_argument("--worlds", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026090703)
    parser.add_argument("--out", type=Path, default=Path("artifacts/ol-pressure-test"))
    args = parser.parse_args()

    policy = pl.read_parquet(args.policy) if args.policy.suffix == ".parquet" else pl.read_csv(args.policy)
    personnel = (
        pl.read_parquet(args.personnel)
        if args.personnel.suffix == ".parquet"
        else pl.read_csv(args.personnel)
    )
    historical = None
    if args.ol_outcomes is not None and args.ol_outcomes.exists():
        historical = (
            pl.read_parquet(args.ol_outcomes)
            if args.ol_outcomes.suffix == ".parquet"
            else pl.read_csv(args.ol_outcomes)
        )

    unit_map = compile_league_unit_player_map(personnel)
    unit_rows: list[dict] = []
    for team_id, players in unit_map.items():
        effects, _ = compile_team_unit_effects(players)
        unit_rows.append({"team_id": team_id, **effects.__dict__})
    unit_effects = pl.DataFrame(unit_rows)
    context = compile_ol_simulation_context(personnel, unit_effects, historical)
    context_map = {str(r["team_id"]): r for r in context.to_dicts()}

    rows: list[dict] = []
    for game_index, (away, home) in enumerate(MATCHUPS):
        states = compile_team_state_map(policy, {away: home, home: away})
        away_effects, _ = compile_team_unit_effects(unit_map[away])
        home_effects, _ = compile_team_unit_effects(unit_map[home])
        away_state = apply_team_unit_effects(states[away], away_effects)
        home_state = apply_team_unit_effects(states[home], home_effects)

        seed = args.seed + game_index * 10_000
        control = simulate_game(
            GameState(game_id=f"{away}@{home}-control", away=away_state, home=home_state),
            worlds=args.worlds,
            seed=seed,
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
        strengthened = simulate_game(
            GameState(
                game_id=f"{away}@{home}-strengthened",
                away=strengthened_away,
                home=strengthened_home,
            ),
            worlds=args.worlds,
            seed=seed,
        )

        a = _summarize(control)
        b = _summarize(strengthened)
        row = {
            "game": f"{away}@{home}",
            "seed": seed,
            "worlds": args.worlds,
            "away_ol_continuity": float(away_ctx["offensive_line_continuity"]),
            "home_ol_continuity": float(home_ctx["offensive_line_continuity"]),
            "away_ol_uncertainty": float(away_ctx["offensive_line_uncertainty"]),
            "home_ol_uncertainty": float(home_ctx["offensive_line_uncertainty"]),
            "away_pass_protection_effect": float(away_ctx.get("pass_protection_effect") or 0.0),
            "home_pass_protection_effect": float(home_ctx.get("pass_protection_effect") or 0.0),
            "away_run_block_effect": float(away_ctx.get("run_block_effect") or 0.0),
            "home_run_block_effect": float(home_ctx.get("run_block_effect") or 0.0),
        }
        for key, value in a.items():
            row[f"control_{key}"] = value
        for key, value in b.items():
            row[f"strengthened_{key}"] = value
            row[f"delta_{key}"] = value - a[key]
        rows.append(row)

    result = pl.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    result.write_csv(args.out / "ol_pressure_test.csv")
    context.write_csv(args.out / "ol_simulation_context.csv")

    max_abs_mean_delta = float(result.select(pl.col("delta_mean_total").abs().max()).item())
    manifest = {
        "artifact": "Monster OL Context Pressure Test",
        "worlds_per_game": args.worlds,
        "base_seed": args.seed,
        "games": [f"{a}@{h}" for a, h in MATCHUPS],
        "control": "current unit capability only; OL continuity/uncertainty neutral",
        "strengthened": "same unit capability plus current OL continuity/uncertainty",
        "historical_ol_outcomes_production_authority": 0.0,
        "max_abs_mean_total_delta": max_abs_mean_delta,
        "guardrail_pass": max_abs_mean_delta <= 1.5,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(result)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
