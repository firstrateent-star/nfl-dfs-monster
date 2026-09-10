from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import polars as pl

import run_week1_v13_first_sim as baseline
from monster.sim.game_flow_lookup import build_team_game_flow_policy


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

    baseline._team_identity = team_identity_with_game_flow
    sys.argv = [sys.argv[0], *remaining, "--out", str(known.out)]
    baseline.main()

    manifest_path = known.out / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(
        {
            "hierarchical_game_flow_active": True,
            "game_flow_decision_scope": "top_level_run_vs_dropback_only",
            "game_flow_resolution_unchanged": True,
            "game_flow_team_style_shrunk": True,
            "game_flow_personnel_adjustment_authority": 0.0,
            "game_flow_opponent_adjustment_authority": 0.0,
            "game_flow_environment_adjustment_authority": 0.0,
            "game_flow_venue_adjustment_authority": 0.0,
            "game_flow_adaptation_authority": 0.0,
            "promotion_status": "SHADOW_GAME_FLOW_STAGE_2_NOT_PROMOTED",
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
