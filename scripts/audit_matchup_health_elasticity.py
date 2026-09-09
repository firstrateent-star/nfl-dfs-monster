from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl

from monster.sim.game import simulate_game
from monster.snapshot.model import GameState, TeamState


def _base_game() -> GameState:
    away = TeamState(team_id="A", opponent_id="H", neutral_pass_rate=0.56)
    home = TeamState(team_id="H", opponent_id="A", neutral_pass_rate=0.56)
    return GameState(game_id="elasticity", away=away, home=home, dome=True)


def _shock(game: GameState, name: str, magnitude: float) -> GameState:
    """Apply a controlled mechanism shock to the away team only.

    Magnitude is mechanism-space, not a claimed real-world injury value. The purpose is
    to measure the simulator's elasticity before any historical calibration is attempted.
    Negative magnitude always represents a deterioration for the away team.
    """
    away = game.away
    home = game.home
    m = float(magnitude)

    if name == "qb_capability":
        away = replace(away, injury_effect=away.injury_effect + m)
    elif name == "ol_pass_protection":
        away = replace(away, pass_protection_effect=away.pass_protection_effect + m)
    elif name == "ol_run_blocking":
        away = replace(away, run_block_effect=away.run_block_effect + m)
    elif name == "edge_pass_rush":
        away = replace(away, pass_rush_effect=away.pass_rush_effect + m)
    elif name == "db_coverage":
        away = replace(away, coverage_effect=away.coverage_effect + m)
    elif name == "run_defense":
        away = replace(away, run_defense_effect=away.run_defense_effect + m)
    elif name == "skill_capability":
        away = replace(away, injury_effect=away.injury_effect + 0.5 * m)
    elif name == "ol_combined":
        away = replace(
            away,
            pass_protection_effect=away.pass_protection_effect + m,
            run_block_effect=away.run_block_effect + 0.75 * m,
        )
    elif name == "defense_combined":
        away = replace(
            away,
            pass_rush_effect=away.pass_rush_effect + m,
            coverage_effect=away.coverage_effect + 0.75 * m,
            run_defense_effect=away.run_defense_effect + 0.60 * m,
        )
    else:
        raise ValueError(name)
    return replace(game, away=away, home=home)


def _mean(x: np.ndarray) -> float:
    return float(np.mean(x))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worlds", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=2026090801)
    parser.add_argument("--out", type=Path, default=Path("artifacts/matchup-health-elasticity"))
    args = parser.parse_args()

    base = _base_game()
    control = simulate_game(base, args.worlds, args.seed)

    scenarios = [
        ("qb_capability", -0.02),
        ("qb_capability", -0.05),
        ("ol_pass_protection", -0.02),
        ("ol_pass_protection", -0.05),
        ("ol_run_blocking", -0.02),
        ("ol_run_blocking", -0.05),
        ("ol_combined", -0.02),
        ("ol_combined", -0.05),
        ("edge_pass_rush", -0.02),
        ("edge_pass_rush", -0.05),
        ("db_coverage", -0.02),
        ("db_coverage", -0.05),
        ("run_defense", -0.02),
        ("run_defense", -0.05),
        ("defense_combined", -0.02),
        ("defense_combined", -0.05),
        ("skill_capability", -0.02),
        ("skill_capability", -0.05),
    ]

    rows: list[dict[str, float | int | str]] = []
    control_away = _mean(control.away_points)
    control_home = _mean(control.home_points)
    control_total = _mean(control.total)
    control_margin = _mean(control.away_points - control.home_points)
    control_away_disruption = _mean(control.away_pass_disruption)
    control_home_disruption = _mean(control.home_pass_disruption)
    control_away_run = _mean(control.away_run_efficiency)
    control_home_run = _mean(control.home_run_efficiency)

    for mechanism, magnitude in scenarios:
        world = simulate_game(_shock(base, mechanism, magnitude), args.worlds, args.seed)
        away_mean = _mean(world.away_points)
        home_mean = _mean(world.home_points)
        total_mean = _mean(world.total)
        margin_mean = _mean(world.away_points - world.home_points)
        rows.append(
            {
                "mechanism": mechanism,
                "shock": magnitude,
                "worlds": args.worlds,
                "away_points_delta": away_mean - control_away,
                "home_points_delta": home_mean - control_home,
                "game_total_delta": total_mean - control_total,
                "away_margin_delta": margin_mean - control_margin,
                "away_pass_disruption_delta": _mean(world.away_pass_disruption) - control_away_disruption,
                "home_pass_disruption_delta": _mean(world.home_pass_disruption) - control_home_disruption,
                "away_run_efficiency_delta": _mean(world.away_run_efficiency) - control_away_run,
                "home_run_efficiency_delta": _mean(world.home_run_efficiency) - control_home_run,
                "away_points_per_minus_0_01": (away_mean - control_away) / (abs(magnitude) / 0.01),
                "home_points_per_minus_0_01": (home_mean - control_home) / (abs(magnitude) / 0.01),
            }
        )

    frame = pl.DataFrame(rows).sort(["mechanism", "shock"])

    monotonic_checks = []
    for mechanism in sorted(set(frame["mechanism"].to_list())):
        sub = frame.filter(pl.col("mechanism") == mechanism).sort("shock", descending=True)
        weak = sub.row(0, named=True)
        strong = sub.row(1, named=True)
        if mechanism in {"edge_pass_rush", "db_coverage", "run_defense", "defense_combined"}:
            metric = "home_points_delta"
            monotonic = float(strong[metric]) >= float(weak[metric]) - 0.05
        else:
            metric = "away_points_delta"
            monotonic = float(strong[metric]) <= float(weak[metric]) + 0.05
        monotonic_checks.append({"mechanism": mechanism, "metric": metric, "monotonic": monotonic})

    check_frame = pl.DataFrame(monotonic_checks)
    manifest = {
        "artifact": "Monster Controlled Matchup/Health Elasticity Audit",
        "purpose": "Measure whether existing game mechanisms react sensibly to controlled personnel/unit deterioration before building deeper health replacement-state logic.",
        "worlds_per_scenario": args.worlds,
        "seed": args.seed,
        "shock_units_are_calibration_free": True,
        "claim": "These are simulator elasticities, not historical estimates of injury value.",
        "monotonic_mechanism_checks": int(check_frame.filter(pl.col("monotonic")).height),
        "total_mechanism_checks": int(check_frame.height),
        "monotonic_pass": bool(check_frame["monotonic"].all()),
        "next_if_sensible": "Pivot to matchup interaction; retain health as a later evidence-driven reality-state modifier.",
        "market_blind": True,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    frame.write_csv(args.out / "elasticity.csv")
    check_frame.write_csv(args.out / "monotonic_checks.csv")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(frame)
    print(check_frame)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
