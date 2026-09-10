from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import polars as pl
import run_week1_v13_first_sim as baseline

import monster.sim.game_loop_v13 as game_loop
from monster.sim.game_flow import derive_game_flow_state
from monster.sim.game_flow_brain import decide_game_flow
from monster.sim.game_flow_lookup import build_team_game_flow_policy
from monster.sim.game_flow_trace import GameFlowTraceRecorder
from monster.sim.play_kernel import PlayType


def _read(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path) if path.suffix == ".parquet" else pl.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--game-flow-league", type=Path, required=True)
    parser.add_argument("--game-flow-team", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    known, remaining = parser.parse_known_args()

    league_rows = _read(known.game_flow_league).to_dicts()
    team_rows = _read(known.game_flow_team).to_dicts()
    original_team_identity = baseline._team_identity
    original_scrimmage_play = game_loop.simulate_scrimmage_play
    trace = GameFlowTraceRecorder()

    def team_identity_with_game_flow(
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
            league_rows=league_rows,
            team_rows=team_rows,
            team_neutral_rate=pool.neutral_pass_rate,
            league_neutral_rate=league_neutral_pass_rate,
        )
        return replace(identity, game_flow_policy=flow_policy)

    def traced_scrimmage_play(state, offense, defense_strength, rng, defense=None):
        flow = derive_game_flow_state(state)
        decision = None
        if offense.game_flow_policy is not None:
            decision = decide_game_flow(
                flow,
                offense.game_flow_policy.evidence_for(flow),
            )
        event = original_scrimmage_play(
            state,
            offense,
            defense_strength,
            rng,
            defense=defense,
        )
        if decision is not None and event.play_type in {PlayType.RUN, PlayType.PASS}:
            away = state.away_team_id or "away"
            home = state.home_team_id or "home"
            trace.observe(
                game=f"{away}@{home}",
                offense=offense.team_id,
                flow=flow,
                is_dropback=event.play_type == PlayType.PASS,
                predicted_probability=decision.dropback_probability,
                league_prior=decision.trace.league_rate,
            )
        return event

    baseline._team_identity = team_identity_with_game_flow
    game_loop.simulate_scrimmage_play = traced_scrimmage_play
    sys.argv = [sys.argv[0], *remaining, "--out", str(known.out)]
    try:
        baseline.main()
    finally:
        game_loop.simulate_scrimmage_play = original_scrimmage_play
        baseline._team_identity = original_team_identity

    trace_df = pl.DataFrame(trace.rows())
    trace_path = known.out / "game_flow_situational_call_trace.csv"
    trace_df.write_csv(trace_path)
    aggregate = trace_df.filter(pl.col("game") == "ALL")
    traced_samples = int(aggregate.get_column("samples").sum()) if aggregate.height else 0
    weighted_sampler_error = (
        float(
            (
                aggregate.get_column("sampler_calibration_error")
                * aggregate.get_column("samples")
            ).sum()
            / traced_samples
        )
        if traced_samples
        else 0.0
    )

    manifest_path = known.out / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(
        {
            "hierarchical_game_flow_active": True,
            "game_flow_decision_scope": "top_level_run_vs_dropback_only",
            "game_flow_resolution_unchanged": True,
            "game_flow_team_style_shrunk": True,
            "game_flow_league_context_shrunk": True,
            "game_flow_personnel_adjustment_authority": 0.0,
            "game_flow_opponent_adjustment_authority": 0.0,
            "game_flow_environment_adjustment_authority": 0.0,
            "game_flow_venue_adjustment_authority": 0.0,
            "game_flow_adaptation_authority": 0.0,
            "situational_call_trace_active": True,
            "situational_call_trace_samples": traced_samples,
            "situational_call_weighted_sampler_error": weighted_sampler_error,
            "promotion_status": "SHADOW_GAME_FLOW_STAGE_2_NOT_PROMOTED",
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
