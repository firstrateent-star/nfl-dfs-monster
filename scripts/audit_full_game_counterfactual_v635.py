from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl
import run_reality_loop_v2_smoke as v61
import run_week1_v13_integrated as integrated

from monster.sim.play_kernel import PassResult, PlayType
from monster.sim.rich_identity import RichPlayerIdentity
from runtime_v635_composer import (
    build_week1_runtime_inputs_v635,
    write_runtime_fingerprint_v635,
)


def _with_qb_channels(team, *, execution: float, mobility: float):
    """Intervene only on the live rich-QB execution/mobility channels."""
    qb_id = team.quarterback.player_id
    qb = team.quarterback
    if not isinstance(qb, RichPlayerIdentity):
        raise RuntimeError(
            f"v6.3.5 counterfactual requires RichPlayerIdentity for {team.team_id} QB {qb_id}"
        )
    qb_new = replace(
        qb,
        qb_execution_skill=float(np.clip(execution, -1.0, 1.0)),
        mobility_skill=float(np.clip(mobility, -1.0, 1.0)),
    )

    def swap(player):
        return qb_new if player.player_id == qb_id else player

    return replace(
        team,
        quarterback=qb_new,
        rushers=tuple(swap(player) for player in team.rushers),
        receivers=tuple(swap(player) for player in team.receivers),
    )


def _safe_ratio(numerator: float, denominator: float) -> float:
    return 0.0 if denominator <= 0.0 else float(numerator / denominator)


def _team_result_metrics(result, *, team_id: str, qb_id: str, is_away: bool) -> dict[str, float]:
    box = result.player_stats.get(qb_id)
    pass_attempts = 0 if box is None else int(box.pass_attempts)
    completions = 0 if box is None else int(box.completions)
    interceptions = 0 if box is None else int(box.interceptions)
    passing_yards = 0.0 if box is None else float(box.passing_yards)

    qb_events = [
        event
        for event in result.plays
        if event.play_type == PlayType.PASS and event.passer_id == qb_id
    ]
    sacks = sum(event.pass_result == PassResult.SACK for event in qb_events)
    scrambles = sum(event.pass_result == PassResult.SCRAMBLE for event in qb_events)
    dropbacks = pass_attempts + sacks + scrambles
    read_events = [
        event
        for event in qb_events
        if event.pass_result not in {PassResult.SACK, PassResult.SCRAMBLE}
    ]
    read_quality = (
        float(np.mean([event.qb_read_quality for event in read_events]))
        if read_events
        else 0.0
    )
    pressure_rate = _safe_ratio(
        sum(bool(event.pressured) for event in qb_events),
        max(dropbacks, 1),
    )

    team_drives = [
        trace for trace in result.drive_traces if trace.offense_team_id == team_id
    ]
    drives = len(team_drives)
    drive_points = sum(float(trace.points) for trace in team_drives)
    drive_yards = sum(float(trace.net_scrimmage_yards) for trace in team_drives)
    first_downs = sum(int(trace.first_downs) for trace in team_drives)
    survival_4plus = sum(int(trace.scrimmage_plays >= 4) for trace in team_drives)
    red_zone = sum(bool(trace.red_zone_entered) for trace in team_drives)
    drive_turnovers = sum(int(trace.turnovers) for trace in team_drives)

    points = (
        float(result.final_state.away_score)
        if is_away
        else float(result.final_state.home_score)
    )
    return {
        "points": points,
        "passing_yards": passing_yards,
        "pass_attempts": float(pass_attempts),
        "completion_rate": _safe_ratio(completions, pass_attempts),
        "interception_rate": _safe_ratio(interceptions, pass_attempts),
        "sack_rate": _safe_ratio(sacks, dropbacks),
        "pressure_rate": pressure_rate,
        "qb_read_quality": read_quality,
        "points_per_drive": _safe_ratio(drive_points, drives),
        "yards_per_drive": _safe_ratio(drive_yards, drives),
        "first_downs_per_drive": _safe_ratio(first_downs, drives),
        "drive_survival_4plus_rate": _safe_ratio(survival_4plus, drives),
        "red_zone_entry_rate": _safe_ratio(red_zone, drives),
        "drive_turnover_rate": _safe_ratio(drive_turnovers, drives),
    }


def _paired_summary(frame: pl.DataFrame, metric: str) -> dict[str, float | int]:
    pivot = (
        frame.select(["team_id", "world", "variant_level", metric])
        .pivot(
            values=metric,
            index=["team_id", "world"],
            on="variant_level",
            aggregate_function="first",
        )
        .drop_nulls(["low", "high"])
        .with_columns((pl.col("high") - pl.col("low")).alias("delta"))
    )
    by_team = pivot.group_by("team_id").agg(pl.col("delta").mean().alias("delta")).sort("team_id")
    deltas = by_team.get_column("delta").to_numpy()
    return {
        "teams": by_team.height,
        "teams_positive": int(np.sum(deltas > 0.0)),
        "teams_negative": int(np.sum(deltas < 0.0)),
        "mean_delta": float(np.mean(deltas)),
        "median_delta": float(np.median(deltas)),
        "p10_delta": float(np.quantile(deltas, 0.10)),
        "p90_delta": float(np.quantile(deltas, 0.90)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--personnel", type=Path, required=True)
    parser.add_argument("--player-usage", type=Path, required=True)
    parser.add_argument("--situation-context", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=40)
    parser.add_argument("--seed", type=int, default=6351701)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    runtime = build_week1_runtime_inputs_v635(
        policy_path=args.policy,
        personnel_path=args.personnel,
        player_usage_path=args.player_usage,
        situation_context_path=args.situation_context,
    )
    fingerprint = write_runtime_fingerprint_v635(args.out)
    pools = runtime["pools"]
    teams = runtime["teams"]
    defenses = runtime["defenses"]
    ecology = runtime["ecology"]
    live = v61.runner.integrated

    rows: list[dict[str, object]] = []
    for game_idx, (away, home) in enumerate(integrated.MATCHUPS):
        for world in range(args.worlds):
            seed = args.seed + game_idx * 1_000_003 + world

            # Use the exact production role-world sampler and role application. One role world
            # is shared by low/high QB variants so the intervention changes QB identity only.
            away_plan = live.sample_event_rush_share_plan(
                pools[away], rng=np.random.default_rng(seed + 101_003)
            )
            home_plan = live.sample_event_rush_share_plan(
                pools[home], rng=np.random.default_rng(seed + 202_007)
            )
            away_base = live._with_event_rush_plan(teams[away], away_plan)
            home_base = live._with_event_rush_plan(teams[home], home_plan)

            for changed_team in (away, home):
                for level, execution, mobility in (
                    ("low", -0.80, -0.50),
                    ("high", 0.80, 0.50),
                ):
                    if changed_team == away:
                        away_team = _with_qb_channels(
                            away_base,
                            execution=execution,
                            mobility=mobility,
                        )
                        home_team = home_base
                    else:
                        away_team = away_base
                        home_team = _with_qb_channels(
                            home_base,
                            execution=execution,
                            mobility=mobility,
                        )

                    result = live.simulate_game(
                        away_team,
                        home_team,
                        away_defense=defenses[away],
                        home_defense=defenses[home],
                        seed=seed,
                        chaos_ecology=ecology,
                    )
                    team_identity = away_team if changed_team == away else home_team
                    metrics = _team_result_metrics(
                        result,
                        team_id=changed_team,
                        qb_id=team_identity.quarterback.player_id,
                        is_away=changed_team == away,
                    )
                    rows.append(
                        {
                            "game": f"{away}@{home}",
                            "world": world,
                            "seed": seed,
                            "team_id": changed_team,
                            "variant_level": level,
                            **metrics,
                        }
                    )

    frame = pl.DataFrame(rows).sort(["game", "world", "team_id", "variant_level"])
    frame.write_csv(args.out / "v635_full_game_qb_counterfactual_worlds.csv")

    metric_directions = {
        # Mechanism/play/drive outcomes where higher is the expected direction.
        "qb_read_quality": "higher",
        "completion_rate": "higher",
        "points_per_drive": "higher",
        "yards_per_drive": "higher",
        "first_downs_per_drive": "higher",
        "drive_survival_4plus_rate": "higher",
        "red_zone_entry_rate": "higher",
        "points": "higher",
        "passing_yards": "diagnostic",
        "pass_attempts": "diagnostic",
        "pressure_rate": "diagnostic",
        # Lower is directionally favorable.
        "sack_rate": "lower",
        "interception_rate": "lower",
        "drive_turnover_rate": "lower",
    }

    summaries: dict[str, dict[str, float | int | str]] = {}
    for metric, direction in metric_directions.items():
        summary = _paired_summary(frame, metric)
        if direction == "lower":
            summary["directionally_favorable_mean_delta"] = -float(summary["mean_delta"])
            summary["directionally_favorable_median_delta"] = -float(summary["median_delta"])
        elif direction == "higher":
            summary["directionally_favorable_mean_delta"] = float(summary["mean_delta"])
            summary["directionally_favorable_median_delta"] = float(summary["median_delta"])
        summary["expected_direction"] = direction
        summaries[metric] = summary

    # Hierarchical realism gate: strict at the nearest mechanism, statistical at drive/game scale.
    mechanism_pass = (
        summaries["qb_read_quality"]["teams_positive"] == len(teams)
        and float(summaries["qb_read_quality"]["median_delta"]) > 0.05
    )
    play_pass = (
        float(summaries["completion_rate"]["median_delta"]) > 0.0
        and float(summaries["sack_rate"]["median_delta"]) <= 0.0
    )
    drive_pass = (
        float(summaries["points_per_drive"]["median_delta"]) > 0.0
        and float(summaries["first_downs_per_drive"]["median_delta"]) > 0.0
    )
    game_pass = (
        float(summaries["points"]["median_delta"]) > 0.0
        and int(summaries["points"]["teams_positive"]) >= 15
    )

    report = {
        "experiment": "v6.3.5-hierarchical-full-game-qb-counterfactual",
        "runtime_hash": fingerprint.runtime_hash,
        "teams": len(teams),
        "worlds_per_game": args.worlds,
        "shared_seeds_across_variants": True,
        "production_role_world_shared_across_qb_variants": True,
        "metrics": summaries,
        "hierarchical_gate": {
            "mechanism": mechanism_pass,
            "play": play_pass,
            "drive": drive_pass,
            "game": game_pass,
            "overall": mechanism_pass and play_pass and drive_pass and game_pass,
        },
        "interpretation": (
            "Strict monotonicity is required nearest the intervention. Full-game box-score "
            "statistics are allowed to move non-monotonically because improved execution changes "
            "score, pace, play mix and possession feedback. The promotion gate therefore follows "
            "mechanism -> play -> drive -> game rather than requiring passing yards to rise for "
            "every team."
        ),
        "direct_score_adjustment": False,
        "reality_outcomes_used": False,
    }
    (args.out / "v635_full_game_qb_counterfactual.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
