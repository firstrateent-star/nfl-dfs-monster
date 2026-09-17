from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl

import audit_week1_v13_drive_survival as drive
import run_week1_v13_integrated as integrated
from monster.sim.current_role_guard_v635 import (
    configure_current_skill_roles_v635,
    sample_rush_share_plan_v635,
)
from monster.sim.game_loop_v13 import simulate_game
from monster.sim.rich_identity import RichPlayerIdentity


def _with_qb_channels(team, *, execution: float, mobility: float):
    qb_id = team.quarterback.player_id
    qb = team.quarterback
    if isinstance(qb, RichPlayerIdentity):
        qb_new = replace(
            qb,
            qb_execution_skill=float(np.clip(execution, -1.0, 1.0)),
            mobility_skill=float(np.clip(mobility, -1.0, 1.0)),
        )
    else:
        # Compatibility fallback for a player without rich evidence.
        qb_new = replace(
            qb,
            efficiency=float(np.clip(1.0 + 0.18 * execution, 0.70, 1.30)),
            explosive=float(np.clip(1.0 + 0.18 * mobility, 0.70, 1.30)),
        )

    def swap(player):
        return qb_new if player.player_id == qb_id else player

    return replace(
        team,
        quarterback=qb_new,
        rushers=tuple(swap(player) for player in team.rushers),
        receivers=tuple(swap(player) for player in team.receivers),
    )


def _apply_rush_plan(team, plan):
    return integrated._with_event_rush_plan(team, plan)


def _simulate(
    away_team,
    home_team,
    *,
    away_defense,
    home_defense,
    ecology,
    seed: int,
):
    return simulate_game(
        away_team,
        home_team,
        away_defense=away_defense,
        home_defense=home_defense,
        seed=seed,
        chaos_ecology=ecology,
    )


def _qb_passing_yards(result, qb_id: str) -> float:
    box = result.player_stats.get(qb_id)
    return 0.0 if box is None else float(box.passing_yards)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--personnel", type=Path, required=True)
    parser.add_argument("--player-usage", type=Path, required=True)
    parser.add_argument("--situation-context", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=20)
    parser.add_argument("--seed", type=int, default=6351701)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    personnel = pl.read_parquet(args.personnel)
    configure_current_skill_roles_v635(personnel)
    pools, teams, defenses, _states, ecology = drive._build_current_world_inputs(
        policy_path=args.policy,
        personnel_path=args.personnel,
        player_usage_path=args.player_usage,
        situation_context_path=args.situation_context,
    )

    rows: list[dict[str, object]] = []
    for game_idx, (away, home) in enumerate(integrated.MATCHUPS):
        for world in range(args.worlds):
            seed = args.seed + game_idx * 1_000_003 + world
            away_plan = sample_rush_share_plan_v635(
                pools[away], rng=np.random.default_rng(seed + 101_003)
            )
            home_plan = sample_rush_share_plan_v635(
                pools[home], rng=np.random.default_rng(seed + 202_007)
            )
            away_base = _apply_rush_plan(teams[away], away_plan)
            home_base = _apply_rush_plan(teams[home], home_plan)

            variants = {
                "baseline": (away_base, home_base, None),
                f"{away}_qb_low": (
                    _with_qb_channels(away_base, execution=-0.80, mobility=-0.50),
                    home_base,
                    away,
                ),
                f"{away}_qb_high": (
                    _with_qb_channels(away_base, execution=0.80, mobility=0.50),
                    home_base,
                    away,
                ),
                f"{home}_qb_low": (
                    away_base,
                    _with_qb_channels(home_base, execution=-0.80, mobility=-0.50),
                    home,
                ),
                f"{home}_qb_high": (
                    away_base,
                    _with_qb_channels(home_base, execution=0.80, mobility=0.50),
                    home,
                ),
            }
            for variant, (away_team, home_team, changed_team) in variants.items():
                result = _simulate(
                    away_team,
                    home_team,
                    away_defense=defenses[away],
                    home_defense=defenses[home],
                    ecology=ecology,
                    seed=seed,
                )
                rows.append(
                    {
                        "game": f"{away}@{home}",
                        "world": world,
                        "seed": seed,
                        "variant": variant,
                        "changed_team": changed_team,
                        "away_points": result.final_state.away_score,
                        "home_points": result.final_state.home_score,
                        "away_qb_yards": _qb_passing_yards(
                            result, away_team.quarterback.player_id
                        ),
                        "home_qb_yards": _qb_passing_yards(
                            result, home_team.quarterback.player_id
                        ),
                    }
                )

    frame = pl.DataFrame(rows)
    frame.write_csv(args.out / "v635_full_game_qb_counterfactual_worlds.csv")

    summaries: list[dict[str, object]] = []
    for team in sorted(teams):
        matchup = next(
            (pair for pair in integrated.MATCHUPS if team in pair),
            None,
        )
        if matchup is None:
            continue
        away, home = matchup
        game = f"{away}@{home}"
        sub = frame.filter(pl.col("game") == game)
        low = sub.filter(pl.col("variant") == f"{team}_qb_low")
        high = sub.filter(pl.col("variant") == f"{team}_qb_high")
        if team == away:
            points_col = "away_points"
            yards_col = "away_qb_yards"
        else:
            points_col = "home_points"
            yards_col = "home_qb_yards"
        low_points = float(low.get_column(points_col).mean())
        high_points = float(high.get_column(points_col).mean())
        low_yards = float(low.get_column(yards_col).mean())
        high_yards = float(high.get_column(yards_col).mean())
        summaries.append(
            {
                "team_id": team,
                "game": game,
                "low_qb_points_mean": low_points,
                "high_qb_points_mean": high_points,
                "points_high_minus_low": high_points - low_points,
                "low_qb_passing_yards_mean": low_yards,
                "high_qb_passing_yards_mean": high_yards,
                "passing_yards_high_minus_low": high_yards - low_yards,
            }
        )

    summary = pl.DataFrame(summaries).sort("team_id")
    summary.write_csv(args.out / "v635_full_game_qb_counterfactual_summary.csv")
    point_deltas = summary.get_column("points_high_minus_low").to_numpy()
    yard_deltas = summary.get_column("passing_yards_high_minus_low").to_numpy()
    report = {
        "experiment": "v6.3.5-full-game-qb-counterfactual",
        "teams": summary.height,
        "worlds_per_game": args.worlds,
        "shared_seeds_across_variants": True,
        "teams_with_positive_points_response": int(np.sum(point_deltas > 0.0)),
        "teams_with_positive_passing_yards_response": int(np.sum(yard_deltas > 0.0)),
        "median_points_high_minus_low": float(np.median(point_deltas)),
        "mean_points_high_minus_low": float(np.mean(point_deltas)),
        "median_passing_yards_high_minus_low": float(np.median(yard_deltas)),
        "mean_passing_yards_high_minus_low": float(np.mean(yard_deltas)),
        "interpretation": (
            "This is a propagation test, not a calibration target. It asks whether changing only "
            "QB execution/mobility, under common random numbers and fixed teammates/opponent, "
            "survives into passing production and team scoring."
        ),
        "direct_score_adjustment": False,
        "reality_outcomes_used": False,
    }
    (args.out / "v635_full_game_qb_counterfactual.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
