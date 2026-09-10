from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import polars as pl
import run_week1_v13_first_sim as baseline
import run_week1_v13_stage3_shadow as stage3

from monster.feature_compile.v13_identity_authority import apply_v13_team_identity_authority


def _arg_path(flag: str) -> Path:
    try:
        index = sys.argv.index(flag)
    except ValueError as exc:
        raise ValueError(f"Missing required argument {flag}") from exc
    if index + 1 >= len(sys.argv):
        raise ValueError(f"Missing value for {flag}")
    return Path(sys.argv[index + 1])


def main() -> None:
    original = baseline._team_identity
    traces: dict[str, object] = {}

    def identity_with_authority(
        team_id,
        pool,
        reality,
        unit_players,
        state,
        *,
        league_neutral_pass_rate,
        situational_pass_rates,
    ):
        identity = original(
            team_id,
            pool,
            reality,
            unit_players,
            state,
            league_neutral_pass_rate=league_neutral_pass_rate,
            situational_pass_rates=situational_pass_rates,
        )
        enhanced, trace = apply_v13_team_identity_authority(
            identity,
            pool=pool,
            reality=reality,
            unit_players=unit_players,
            state=state,
        )
        traces[team_id] = trace
        return enhanced

    baseline._team_identity = identity_with_authority
    try:
        stage3.main()
    finally:
        baseline._team_identity = original

    out = _arg_path("--out")
    if len(traces) != 24:
        raise RuntimeError(f"Identity authority attached to {len(traces)} teams, expected 24")

    trace_frame = pl.DataFrame([asdict(traces[team]) for team in sorted(traces)])
    trace_frame.write_csv(out / "identity_authority_trace.csv")

    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(
        {
            "rich_identity_authority_active": True,
            "identity_authority_scope": [
                "position_specific_player_capability",
                "qb_execution_to_pass_resolution",
                "receiver_capability_to_target_resolution",
                "rusher_capability_to_run_resolution",
                "live_team_run_context",
            ],
            "identity_evidence_sources": [
                "physical_measurements",
                "combine_speed",
                "madden_position_attributes",
                "current_health_effectiveness",
                "current_availability_expected_state",
                "unit_continuity",
            ],
            "direct_point_adjustment": False,
            "future_extension_seam": "PlayerIdentityChannels + InteractionRegistry",
            "identity_authority_trace": "identity_authority_trace.csv",
            "promotion_status": "SHADOW_IDENTITY_AUTHORITY_NOT_PROMOTED",
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(trace_frame)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
