from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl
import run_week1_v13_first_sim as baseline

import monster.sim.game_loop_v13 as game_loop
from monster.sim.game_flow_lookup import build_team_game_flow_policy
from monster.sim.intent_ecology import build_intent_ecology
from monster.sim.play_kernel import PassResult, PlayType


def _read(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path) if path.suffix == ".parquet" else pl.read_csv(path)


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _summarize_pass(store: dict[str, dict[str, object]]) -> pl.DataFrame:
    rows = []
    for category, bucket in sorted(store.items()):
        intents = int(bucket["intents"])
        throws = int(bucket["throws"])
        completions = int(bucket["completions"])
        interceptions = int(bucket["interceptions"])
        sacks = int(bucket["sacks"])
        scrambles = int(bucket["scrambles"])
        completed_yards = list(bucket["completed_yards"])
        air_yards = list(bucket["air_yards"])
        rows.append(
            {
                "category": category,
                "intent_dropbacks": intents,
                "throws": throws,
                "completions": completions,
                "interceptions": interceptions,
                "sacks": sacks,
                "scrambles": scrambles,
                "completion_rate_on_throws": _safe_rate(completions, throws),
                "interception_rate_on_throws": _safe_rate(interceptions, throws),
                "sack_rate_on_intents": _safe_rate(sacks, intents),
                "scramble_rate_on_intents": _safe_rate(scrambles, intents),
                "air_yards_mean_on_throws": _mean(air_yards),
                "completed_yards_mean": _mean(completed_yards),
                "negative_completion_rate": (
                    float(np.mean(np.asarray(completed_yards) < 0.0)) if completed_yards else None
                ),
                "completed_gain_15_plus_rate": (
                    float(np.mean(np.asarray(completed_yards) >= 15.0)) if completed_yards else None
                ),
                "completed_gain_20_plus_rate": (
                    float(np.mean(np.asarray(completed_yards) >= 20.0)) if completed_yards else None
                ),
                "completed_gain_40_plus_rate": (
                    float(np.mean(np.asarray(completed_yards) >= 40.0)) if completed_yards else None
                ),
            }
        )
    return pl.DataFrame(rows)


def _summarize_run(store: dict[str, dict[str, object]]) -> pl.DataFrame:
    rows = []
    for category, bucket in sorted(store.items()):
        yards = np.asarray(list(bucket["yards"]), dtype=float)
        attempts = len(yards)
        rows.append(
            {
                "category": category,
                "attempts": attempts,
                "yards_mean": float(yards.mean()) if attempts else None,
                "yards_p10": float(np.quantile(yards, 0.10)) if attempts else None,
                "yards_p50": float(np.quantile(yards, 0.50)) if attempts else None,
                "yards_p90": float(np.quantile(yards, 0.90)) if attempts else None,
                "yards_p99": float(np.quantile(yards, 0.99)) if attempts else None,
                "negative_rate": float(np.mean(yards < 0.0)) if attempts else None,
                "zero_rate": float(np.mean(yards == 0.0)) if attempts else None,
                "loss_2_plus_rate": float(np.mean(yards <= -2.0)) if attempts else None,
                "loss_5_plus_rate": float(np.mean(yards <= -5.0)) if attempts else None,
                "explosive_10_rate": float(np.mean(yards >= 10.0)) if attempts else None,
                "explosive_15_rate": float(np.mean(yards >= 15.0)) if attempts else None,
                "explosive_20_rate": float(np.mean(yards >= 20.0)) if attempts else None,
            }
        )
    return pl.DataFrame(rows)


def _summarize_pressure(store: dict[tuple[str, bool], dict[str, int]]) -> pl.DataFrame:
    rows = []
    for (category, pressured), bucket in sorted(store.items()):
        intents = int(bucket["intents"])
        throws = int(bucket["throws"])
        rows.append(
            {
                "category": category,
                "pressured": pressured,
                "intents": intents,
                "throws": throws,
                "completions": int(bucket["completions"]),
                "interceptions": int(bucket["interceptions"]),
                "sacks": int(bucket["sacks"]),
                "scrambles": int(bucket["scrambles"]),
                "completion_rate_on_throws": _safe_rate(int(bucket["completions"]), throws),
                "interception_rate_on_throws": _safe_rate(int(bucket["interceptions"]), throws),
                "sack_rate_on_intents": _safe_rate(int(bucket["sacks"]), intents),
                "scramble_rate_on_intents": _safe_rate(int(bucket["scrambles"]), intents),
            }
        )
    return pl.DataFrame(rows)


def _summarize_run_contact(store: dict[str, dict[str, list[float]]]) -> pl.DataFrame:
    rows = []
    for category, bucket in sorted(store.items()):
        yards = np.asarray(bucket["yards"], dtype=float)
        before = np.asarray(bucket["before_contact"], dtype=float)
        after = np.asarray(bucket["after_contact"], dtype=float)
        attempts = len(yards)
        explosive = yards >= 20.0 if attempts else np.asarray([], dtype=bool)
        rows.append(
            {
                "category": category,
                "attempts": attempts,
                "yards_before_contact_mean": float(before.mean()) if attempts else None,
                "yards_after_contact_mean": float(after.mean()) if attempts else None,
                "second_level_rate": float(np.mean(before >= 4.0)) if attempts else None,
                "after_contact_3plus_rate": float(np.mean(after >= 3.0)) if attempts else None,
                "breakaway_20plus_rate": float(np.mean(explosive)) if attempts else None,
                "breakaway_20plus_yards_mean": float(yards[explosive].mean()) if explosive.any() else None,
                "breakaway_20plus_before_contact_mean": (
                    float(before[explosive].mean()) if explosive.any() else None
                ),
                "breakaway_20plus_after_contact_mean": (
                    float(after[explosive].mean()) if explosive.any() else None
                ),
            }
        )
    return pl.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
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
    parser.add_argument("--out", type=Path, required=True)
    known, remaining = parser.parse_known_args()

    game_flow_league = _read(known.game_flow_league).to_dicts()
    game_flow_team = _read(known.game_flow_team).to_dicts()
    pass_league = _read(known.pass_depth_league).to_dicts()
    pass_team = _read(known.pass_depth_team).to_dicts()
    pass_qb = _read(known.pass_depth_qb).to_dicts()
    pass_outcomes = _read(known.pass_depth_outcomes).to_dicts()
    target_depth = _read(known.target_depth).to_dicts()
    run_league = _read(known.run_geometry_league).to_dicts()
    run_team = _read(known.run_geometry_team).to_dicts()
    run_rusher = _read(known.run_geometry_rusher).to_dicts()
    run_outcomes = _read(known.run_geometry_outcomes).to_dicts()

    original_team_identity = baseline._team_identity
    original_scrimmage_play = game_loop.simulate_scrimmage_play
    pass_trace: dict[str, dict[str, object]] = defaultdict(
        lambda: {
            "intents": 0,
            "throws": 0,
            "completions": 0,
            "interceptions": 0,
            "sacks": 0,
            "scrambles": 0,
            "air_yards": [],
            "completed_yards": [],
        }
    )
    run_trace: dict[str, dict[str, object]] = defaultdict(lambda: {"yards": []})
    pressure_trace: dict[tuple[str, bool], dict[str, int]] = defaultdict(
        lambda: {
            "intents": 0,
            "throws": 0,
            "completions": 0,
            "interceptions": 0,
            "sacks": 0,
            "scrambles": 0,
        }
    )
    run_contact_trace: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"yards": [], "before_contact": [], "after_contact": []}
    )
    attached_teams: set[str] = set()

    def team_identity_with_stage3(
        team_id,
        pool,
        reality,
        unit_players,
        state,
        *,
        league_neutral_pass_rate,
        situational_pass_rates,
    ):
        identity = original_team_identity(
            team_id,
            pool,
            reality,
            unit_players,
            state,
            league_neutral_pass_rate=league_neutral_pass_rate,
            situational_pass_rates=situational_pass_rates,
        )
        flow_policy = build_team_game_flow_policy(
            team_id=team_id,
            league_rows=game_flow_league,
            team_rows=game_flow_team,
            team_neutral_rate=pool.neutral_pass_rate,
            league_neutral_rate=league_neutral_pass_rate,
        )
        intent = build_intent_ecology(
            team_id=team_id,
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
        attached_teams.add(team_id)
        return replace(identity, game_flow_policy=flow_policy, intent_ecology=intent)

    def traced_scrimmage_play(state, offense, defense_strength, rng, defense=None):
        event = original_scrimmage_play(
            state,
            offense,
            defense_strength,
            rng,
            defense=defense,
        )
        if event.play_type == PlayType.PASS and event.pass_depth_category is not None:
            bucket = pass_trace[event.pass_depth_category]
            bucket["intents"] = int(bucket["intents"]) + 1
            conditioned = pressure_trace[(event.pass_depth_category, bool(event.pressured))]
            conditioned["intents"] += 1
            if event.pass_result == PassResult.SACK:
                bucket["sacks"] = int(bucket["sacks"]) + 1
                conditioned["sacks"] += 1
            elif event.pass_result == PassResult.SCRAMBLE:
                bucket["scrambles"] = int(bucket["scrambles"]) + 1
                conditioned["scrambles"] += 1
            elif event.pass_result in {
                PassResult.COMPLETE,
                PassResult.INCOMPLETE,
                PassResult.INTERCEPTION,
            }:
                bucket["throws"] = int(bucket["throws"]) + 1
                conditioned["throws"] += 1
                cast_air = bucket["air_yards"]
                assert isinstance(cast_air, list)
                cast_air.append(float(event.air_yards))
                if event.pass_result == PassResult.COMPLETE:
                    bucket["completions"] = int(bucket["completions"]) + 1
                    conditioned["completions"] += 1
                    cast_yards = bucket["completed_yards"]
                    assert isinstance(cast_yards, list)
                    cast_yards.append(float(event.yards))
                elif event.pass_result == PassResult.INTERCEPTION:
                    bucket["interceptions"] = int(bucket["interceptions"]) + 1
                    conditioned["interceptions"] += 1
        elif event.play_type == PlayType.RUN and event.run_geometry_category is not None:
            yards = run_trace[event.run_geometry_category]["yards"]
            assert isinstance(yards, list)
            yards.append(float(event.yards))
            contact = run_contact_trace[event.run_geometry_category]
            contact["yards"].append(float(event.yards))
            contact["before_contact"].append(float(event.yards_before_contact))
            contact["after_contact"].append(float(event.yards_after_contact))
        return event

    baseline._team_identity = team_identity_with_stage3
    game_loop.simulate_scrimmage_play = traced_scrimmage_play
    sys.argv = [sys.argv[0], *remaining, "--out", str(known.out)]
    try:
        baseline.main()
    finally:
        game_loop.simulate_scrimmage_play = original_scrimmage_play
        baseline._team_identity = original_team_identity

    if len(attached_teams) != 24:
        raise RuntimeError(f"Stage 3 attached to {len(attached_teams)} teams, expected 24")
    if not pass_trace or not run_trace:
        raise RuntimeError("Stage 3 traces are empty; intent/resolution bridge was not exercised")

    pass_df = _summarize_pass(pass_trace)
    run_df = _summarize_run(run_trace)
    pressure_df = _summarize_pressure(pressure_trace)
    run_contact_df = _summarize_run_contact(run_contact_trace)
    pass_df.write_csv(known.out / "stage3_pass_depth_anatomy.csv")
    run_df.write_csv(known.out / "stage3_run_geometry_anatomy.csv")
    pressure_df.write_csv(known.out / "stage3_pressure_depth_anatomy.csv")
    run_contact_df.write_csv(known.out / "stage3_run_second_level_anatomy.csv")

    manifest_path = known.out / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(
        {
            "hierarchical_game_flow_active": True,
            "game_flow_decision_scope": "top_level_run_vs_dropback_only",
            "intent_ecology_active": True,
            "intent_oos_gate_required": True,
            "pass_depth_intent_active": True,
            "run_geometry_intent_active": True,
            "target_depth_compatibility_active": True,
            "rusher_geometry_compatibility_active": True,
            "stage3_resolution_ecology_active": True,
            "signed_pass_geometry_active": True,
            "depth_double_penalty_guard_active": True,
            "stage3_attached_team_count": len(attached_teams),
            "stage3_pass_depth_anatomy": "stage3_pass_depth_anatomy.csv",
            "stage3_run_geometry_anatomy": "stage3_run_geometry_anatomy.csv",
            "stage3_pressure_depth_anatomy": "stage3_pressure_depth_anatomy.csv",
            "stage3_run_second_level_anatomy": "stage3_run_second_level_anatomy.csv",
            "pressure_depth_trace_behavioral_authority": False,
            "run_second_level_trace_behavioral_authority": False,
            "promotion_status": "SHADOW_STAGE3_NOT_PROMOTED",
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(pass_df)
    print(run_df)
    print(pressure_df)
    print(run_contact_df)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
