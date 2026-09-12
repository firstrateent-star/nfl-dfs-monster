from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

import audit_week1_v13_drive_survival as drive
import run_week1_v13_integrated as integrated
from monster.sim.football_state import FootballState
from monster.sim.matchup_kernel import resolve_pass_matchup, resolve_run_matchup
from monster.sim.play_kernel import PlayerIdentity, TeamIdentity, _dropback_probability


STATES = (
    ("neutral_1_10_25", 1, 10.0, 25.0, 3000, 0),
    ("neutral_2_6_40", 2, 6.0, 40.0, 2400, 0),
    ("neutral_3_7_50", 3, 7.0, 50.0, 1800, 0),
    ("redzone_1_10", 1, 10.0, 82.0, 1500, 0),
    ("late_trailing", 2, 8.0, 55.0, 420, -10),
    ("late_leading", 2, 8.0, 55.0, 420, 10),
)


def _state(team: str, opponent: str, spec: tuple[Any, ...]) -> FootballState:
    name, down, distance, yardline, seconds, margin = spec
    away_score = 20 if margin >= 0 else 20 - margin
    home_score = 20 - margin if margin >= 0 else 20
    return FootballState(
        possession=team,
        defense=opponent,
        quarter=4 if seconds <= 900 else 2,
        seconds_remaining=int(seconds),
        yardline_100=float(yardline),
        down=int(down),
        distance=float(distance),
        away_score=int(away_score),
        home_score=int(home_score),
        away_team_id=team,
        home_team_id=opponent,
    )


def _top_receiver(team: TeamIdentity) -> PlayerIdentity:
    return max(team.receivers, key=lambda p: p.usage_weight)


def _top_rusher(team: TeamIdentity) -> PlayerIdentity:
    return max(team.rushers, key=lambda p: p.usage_weight)


def _with_qb_efficiency(team: TeamIdentity, value: float) -> TeamIdentity:
    qb_id = team.quarterback.player_id
    qb = replace(team.quarterback, efficiency=value)
    rushers = tuple(replace(p, efficiency=value) if p.player_id == qb_id else p for p in team.rushers)
    receivers = tuple(replace(p, efficiency=value) if p.player_id == qb_id else p for p in team.receivers)
    return replace(team, quarterback=qb, rushers=rushers, receivers=receivers)


def _with_receiver_efficiency(team: TeamIdentity, player_id: str, value: float) -> TeamIdentity:
    receivers = tuple(replace(p, efficiency=value) if p.player_id == player_id else p for p in team.receivers)
    return replace(team, receivers=receivers)


def _with_rusher_efficiency(team: TeamIdentity, player_id: str, value: float) -> TeamIdentity:
    rushers = tuple(replace(p, efficiency=value) if p.player_id == player_id else p for p in team.rushers)
    return replace(team, rushers=rushers)


def _pass_proxy(team: TeamIdentity, defense: Any) -> dict[str, float]:
    target = _top_receiver(team)
    matchup = resolve_pass_matchup(
        target,
        defense,
        pass_protection=team.pass_protection,
        # This intentionally mirrors the current production call site.
        quarterback_efficiency=team.pass_efficiency,
    )
    return {
        "completion": float(matchup.completion_probability),
        "interception": float(matchup.interception_probability),
        "pressure": float(matchup.pressure_probability),
        "yards_multiplier": float(matchup.yards_multiplier),
        "qb_read_quality": float(matchup.qb_read_quality),
    }


def _run_proxy(team: TeamIdentity, defense: Any) -> dict[str, float]:
    rusher = _top_rusher(team)
    matchup = resolve_run_matchup(rusher, defense, run_blocking=team.run_blocking)
    return {
        "stuff": float(matchup.stuff_probability),
        "yards_multiplier": float(matchup.yards_multiplier),
        "runner_edge": float(matchup.runner_edge),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--personnel", type=Path, required=True)
    parser.add_argument("--player-usage", type=Path, required=True)
    parser.add_argument("--situation-context", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    pools, teams, defenses, _states, _ecology = drive._build_current_world_inputs(
        policy_path=args.policy,
        personnel_path=args.personnel,
        player_usage_path=args.player_usage,
        situation_context_path=args.situation_context,
    )

    rows: list[dict[str, Any]] = []
    sensitivity: list[dict[str, Any]] = []
    for away, home in integrated.MATCHUPS:
        for team_id, opponent_id in ((away, home), (home, away)):
            team = teams[team_id]
            defense = defenses[opponent_id]
            receiver = _top_receiver(team)
            rusher = _top_rusher(team)
            base_pass = _pass_proxy(team, defense)
            base_run = _run_proxy(team, defense)
            rows.append(
                {
                    "team_id": team_id,
                    "opponent_id": opponent_id,
                    "qb_efficiency": float(team.quarterback.efficiency),
                    "qb_explosive": float(team.quarterback.explosive),
                    "team_pass_efficiency": float(team.pass_efficiency),
                    "team_rush_efficiency": float(team.rush_efficiency),
                    "top_receiver_efficiency": float(receiver.efficiency),
                    "top_receiver_explosive": float(receiver.explosive),
                    "top_rusher_efficiency": float(rusher.efficiency),
                    "pass_completion_proxy": base_pass["completion"],
                    "pass_pressure_proxy": base_pass["pressure"],
                    "pass_yards_multiplier_proxy": base_pass["yards_multiplier"],
                    "pass_qb_read_quality_proxy": base_pass["qb_read_quality"],
                    "run_stuff_proxy": base_run["stuff"],
                    "run_yards_multiplier_proxy": base_run["yards_multiplier"],
                }
            )

            for label, qb_eff in (("qb_low", 0.80), ("qb_high", 1.20)):
                variant = _with_qb_efficiency(team, qb_eff)
                outcome = _pass_proxy(variant, defense)
                sensitivity.append(
                    {
                        "team_id": team_id,
                        "opponent_id": opponent_id,
                        "test": label,
                        "changed_component": "quarterback.efficiency_only",
                        "input_value": qb_eff,
                        "completion_delta": outcome["completion"] - base_pass["completion"],
                        "pressure_delta": outcome["pressure"] - base_pass["pressure"],
                        "yards_multiplier_delta": outcome["yards_multiplier"] - base_pass["yards_multiplier"],
                        "qb_read_quality_delta": outcome["qb_read_quality"] - base_pass["qb_read_quality"],
                    }
                )

            for label, receiver_eff in (("receiver_low", 0.80), ("receiver_high", 1.20)):
                variant = _with_receiver_efficiency(team, receiver.player_id, receiver_eff)
                outcome = _pass_proxy(variant, defense)
                sensitivity.append(
                    {
                        "team_id": team_id,
                        "opponent_id": opponent_id,
                        "test": label,
                        "changed_component": "top_receiver.efficiency_only",
                        "input_value": receiver_eff,
                        "completion_delta": outcome["completion"] - base_pass["completion"],
                        "pressure_delta": outcome["pressure"] - base_pass["pressure"],
                        "yards_multiplier_delta": outcome["yards_multiplier"] - base_pass["yards_multiplier"],
                        "qb_read_quality_delta": outcome["qb_read_quality"] - base_pass["qb_read_quality"],
                    }
                )

            for label, team_eff in (("team_pass_low", 0.80), ("team_pass_high", 1.20)):
                variant = replace(team, pass_efficiency=team_eff)
                outcome = _pass_proxy(variant, defense)
                sensitivity.append(
                    {
                        "team_id": team_id,
                        "opponent_id": opponent_id,
                        "test": label,
                        "changed_component": "team.pass_efficiency_only",
                        "input_value": team_eff,
                        "completion_delta": outcome["completion"] - base_pass["completion"],
                        "pressure_delta": outcome["pressure"] - base_pass["pressure"],
                        "yards_multiplier_delta": outcome["yards_multiplier"] - base_pass["yards_multiplier"],
                        "qb_read_quality_delta": outcome["qb_read_quality"] - base_pass["qb_read_quality"],
                    }
                )

            for label, rusher_eff in (("rusher_low", 0.80), ("rusher_high", 1.20)):
                variant = _with_rusher_efficiency(team, rusher.player_id, rusher_eff)
                outcome = _run_proxy(variant, defense)
                sensitivity.append(
                    {
                        "team_id": team_id,
                        "opponent_id": opponent_id,
                        "test": label,
                        "changed_component": "top_rusher.efficiency_only",
                        "input_value": rusher_eff,
                        "completion_delta": None,
                        "pressure_delta": None,
                        "yards_multiplier_delta": outcome["yards_multiplier"] - base_run["yards_multiplier"],
                        "qb_read_quality_delta": None,
                    }
                )

            for spec in STATES:
                state = _state(team_id, opponent_id, spec)
                sensitivity.append(
                    {
                        "team_id": team_id,
                        "opponent_id": opponent_id,
                        "test": f"dropback_{spec[0]}",
                        "changed_component": "game_flow_state",
                        "input_value": None,
                        "completion_delta": None,
                        "pressure_delta": None,
                        "yards_multiplier_delta": None,
                        "qb_read_quality_delta": float(_dropback_probability(state, team)),
                    }
                )

    identity = pl.DataFrame(rows)
    sens = pl.DataFrame(sensitivity)
    identity.write_csv(args.out / "identity_transmission.csv")
    sens.write_csv(args.out / "component_sensitivity.csv")

    component_summary = (
        sens.filter(pl.col("changed_component") != "game_flow_state")
        .group_by(["changed_component", "test"])
        .agg(
            pl.col("completion_delta").abs().mean().alias("mean_abs_completion_delta"),
            pl.col("pressure_delta").abs().mean().alias("mean_abs_pressure_delta"),
            pl.col("yards_multiplier_delta").abs().mean().alias("mean_abs_yards_multiplier_delta"),
            pl.col("qb_read_quality_delta").abs().mean().alias("mean_abs_qb_read_quality_delta"),
        )
        .sort(["changed_component", "test"])
    )
    component_summary.write_csv(args.out / "component_sensitivity_summary.csv")

    state_dispersion = (
        sens.filter(pl.col("changed_component") == "game_flow_state")
        .group_by("test")
        .agg(
            pl.col("qb_read_quality_delta").mean().alias("mean_dropback_probability"),
            pl.col("qb_read_quality_delta").std().alias("team_sd_dropback_probability"),
            (pl.col("qb_read_quality_delta").max() - pl.col("qb_read_quality_delta").min()).alias("team_range_dropback_probability"),
        )
        .sort("test")
    )
    state_dispersion.write_csv(args.out / "game_flow_identity_dispersion.csv")

    input_output = identity.select(
        "qb_efficiency",
        "team_pass_efficiency",
        "top_receiver_efficiency",
        "top_rusher_efficiency",
        "pass_completion_proxy",
        "pass_qb_read_quality_proxy",
        "run_yards_multiplier_proxy",
    ).corr()
    input_output.write_csv(args.out / "identity_correlation_matrix.csv")

    qb_effect = sens.filter(pl.col("changed_component") == "quarterback.efficiency_only")
    qb_completion_authority = float(qb_effect.get_column("completion_delta").abs().max())
    qb_read_authority = float(qb_effect.get_column("qb_read_quality_delta").abs().max())
    manifest = {
        "experiment": "MON-LEDGER-003-IDENTITY-TRANSMISSION",
        "center": "player skill and team identity must causally survive into football outcomes",
        "games": len(integrated.MATCHUPS),
        "teams": identity.height,
        "qb_efficiency_counterfactual_max_abs_completion_delta": qb_completion_authority,
        "qb_efficiency_counterfactual_max_abs_qb_read_delta": qb_read_authority,
        "runtime_authority_observation": {
            "pass_matchup_quarterback_efficiency_argument": "TeamIdentity.pass_efficiency",
            "compiled_qb_efficiency_is_direct_argument": False,
        },
        "governance": {
            "stage": "LAB",
            "production_promoted": False,
            "football_coefficients_changed_for_experiment": False,
            "market_inputs_used": False,
            "counterfactuals_change_identity_inputs_only": True,
        },
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(component_summary)
    print(state_dispersion)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
