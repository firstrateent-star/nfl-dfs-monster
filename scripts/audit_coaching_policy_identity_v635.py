from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

import audit_week1_v13_drive_survival as drive
import run_week1_v13_integrated as integrated
from monster.sim.football_state import FootballState
from monster.sim.play_kernel import _dropback_probability


STATE_SPECS = (
    ("neutral_1_10_25", 1, 10.0, 25.0, 3000, 0),
    ("neutral_2_6_40", 2, 6.0, 40.0, 2400, 0),
    ("neutral_3_7_50", 3, 7.0, 50.0, 1800, 0),
    ("redzone_1_10", 1, 10.0, 82.0, 1500, 0),
    ("late_trailing", 2, 8.0, 55.0, 420, -10),
    ("late_leading", 2, 8.0, 55.0, 420, 10),
)


def _state(team: str, opponent: str, spec: tuple[object, ...]) -> FootballState:
    _name, down, distance, yardline, seconds, margin = spec
    margin = int(margin)
    away_score = 20 if margin >= 0 else 20 - margin
    home_score = 20 - margin if margin >= 0 else 20
    return FootballState(
        possession=team,
        defense=opponent,
        quarter=4 if int(seconds) <= 900 else 2,
        seconds_remaining=int(seconds),
        yardline_100=float(yardline),
        down=int(down),
        distance=float(distance),
        away_score=int(away_score),
        home_score=int(home_score),
        away_team_id=team,
        home_team_id=opponent,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--personnel", type=Path, required=True)
    parser.add_argument("--player-usage", type=Path, required=True)
    parser.add_argument("--situation-context", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    _pools, teams, _defenses, _states, _ecology = drive._build_current_world_inputs(
        policy_path=args.policy,
        personnel_path=args.personnel,
        player_usage_path=args.player_usage,
        situation_context_path=args.situation_context,
    )

    opponent = {
        away: home
        for away, home in integrated.MATCHUPS
    } | {
        home: away
        for away, home in integrated.MATCHUPS
    }

    rows: list[dict[str, object]] = []
    for team_id in sorted(teams):
        team = teams[team_id]
        for spec in STATE_SPECS:
            state_name = str(spec[0])
            probability = float(
                _dropback_probability(
                    _state(team_id, opponent[team_id], spec),
                    team,
                )
            )
            rows.append(
                {
                    "team_id": team_id,
                    "opponent_id": opponent[team_id],
                    "state": state_name,
                    "dropback_probability": probability,
                    "neutral_pass_rate": float(team.neutral_pass_rate),
                    "has_game_flow_policy": team.game_flow_policy is not None,
                }
            )

    frame = pl.DataFrame(rows).sort(["state", "team_id"])
    frame.write_csv(args.out / "v635_coaching_policy_identity_matrix.csv")

    summaries: list[dict[str, object]] = []
    for state in [str(spec[0]) for spec in STATE_SPECS]:
        values = (
            frame.filter(pl.col("state") == state)
            .get_column("dropback_probability")
            .to_numpy()
        )
        summaries.append(
            {
                "state": state,
                "team_mean": float(np.mean(values)),
                "team_sd": float(np.std(values, ddof=1)),
                "team_min": float(np.min(values)),
                "team_max": float(np.max(values)),
                "team_range": float(np.max(values) - np.min(values)),
            }
        )
    summary = pl.DataFrame(summaries)
    summary.write_csv(args.out / "v635_coaching_policy_identity_summary.csv")

    report = {
        "experiment": "v6.3.5-coaching-policy-identity",
        "teams": len(teams),
        "states": len(STATE_SPECS),
        "all_teams_have_game_flow_policy": bool(
            frame.get_column("has_game_flow_policy").all()
        ),
        "mean_state_team_sd": float(summary.get_column("team_sd").mean()),
        "mean_state_team_range": float(summary.get_column("team_range").mean()),
        "minimum_state_team_range": float(summary.get_column("team_range").min()),
        "maximum_state_team_range": float(summary.get_column("team_range").max()),
        "interpretation": (
            "This measures the policy that actually chooses pass/run intent under identical "
            "football states. It is diagnostic only; no target dispersion is imposed."
        ),
    }
    (args.out / "v635_coaching_policy_identity.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
