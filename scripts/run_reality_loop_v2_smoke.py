from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl
import run_week1_v13_dispersion_test as runner

from monster.sim import (
    game_loop_v13,
    intent_ecology,
    matchup_kernel,
    play_kernel,
    progressive_skill_tail_v2,
    resolution_ecology,
)
from monster.sim.clock_ecology_v2 import sample_snap_cadence_v2
from monster.sim.full_world_telemetry_v5 import FullWorldTelemetryV5
from monster.sim.game_flow_lookup import build_team_game_flow_policy
from monster.sim.interaction_topology_v4b import snap_responsibility_key
from monster.sim.opportunity_skill_tail_v3 import OpportunitySkillTailV3
from monster.sim.player_skill_telemetry_v2 import PlayerSkillTelemetryV2
from monster.sim.progressive_skill_tail_v2 import (
    resolve_run_contact_progressive_skill_v2,
    resolve_run_ecology_progressive_skill_v2,
    sample_yac_progressive_skill_v2,
)
from monster.sim.reality_snap_v5 import annotate_event, prepare_snap_world
from monster.sim.snap_ecology import register_team_units
from monster.sim.snap_ecology_v2 import resolve_pass_snap_v2, resolve_run_snap_v2

_NATIVE_ATTACH_INTENT = runner.integrated._attach_historical_intent_ecology
_NATIVE_SIMULATE_GAME = runner.integrated.simulate_game
_NATIVE_SIMULATE_SCRIMMAGE_PLAY = play_kernel.simulate_scrimmage_play
_NATIVE_ENHANCED_TEAM_IDENTITY = runner.enhanced_team_identity
_NATIVE_ENHANCED_DEFENSIVE_UNIT = runner.enhanced_defensive_unit
_NATIVE_FIELD_READ_TARGET = play_kernel._field_read_target
_NATIVE_CHOOSE_TARGET_FOR_DEPTH = intent_ecology.choose_target_for_depth
_NATIVE_CHOOSE_RUSHER_FOR_GEOMETRY = intent_ecology.choose_rusher_for_geometry
_NATIVE_PASS_INTERACTION = progressive_skill_tail_v2._pass_interaction
_NATIVE_RUN_SKILL_EDGE = progressive_skill_tail_v2._run_skill_edge
_TELEMETRY = FullWorldTelemetryV5()
_SKILL_TELEMETRY = PlayerSkillTelemetryV2()
_OPPORTUNITY_TAIL = OpportunitySkillTailV3()


def _attach_reality_loop_policy(teams: dict, policy_dir: Path) -> dict:
    enriched = _NATIVE_ATTACH_INTENT(teams, policy_dir)
    league_path = policy_dir / "game_flow_league.parquet"
    team_path = policy_dir / "game_flow_team.parquet"
    if not league_path.exists() or not team_path.exists():
        raise FileNotFoundError(
            "Reality Loop v2 requires game-flow policy inputs alongside intent ecology: "
            f"{league_path}, {team_path}"
        )
    league_rows = pl.read_parquet(league_path).to_dicts()
    team_rows = pl.read_parquet(team_path).to_dicts()
    return {
        team_id: replace(
            team,
            game_flow_policy=build_team_game_flow_policy(
                team_id=team_id,
                league_rows=league_rows,
                team_rows=team_rows,
                team_neutral_rate=team.neutral_pass_rate,
                league_neutral_rate=team.league_neutral_pass_rate,
            ),
        )
        for team_id, team in enriched.items()
    }


def _team_identity_with_skill_telemetry(*args, **kwargs):
    team = _NATIVE_ENHANCED_TEAM_IDENTITY(*args, **kwargs)
    _SKILL_TELEMETRY.register_team_identity(team)
    unit_players = args[3] if len(args) > 3 else kwargs.get("unit_players", ())
    # Production v4B had the OL duel machinery but never registered the five actual linemen.
    # Registering here activates individual pass/run blocking in the exact runner used for worlds.
    register_team_units(team.team_id, unit_players)
    return team


def _defensive_unit_with_skill_telemetry(*args, **kwargs):
    defense = _NATIVE_ENHANCED_DEFENSIVE_UNIT(*args, **kwargs)
    _SKILL_TELEMETRY.register_defensive_unit(defense)
    return defense


def _simulate_game_with_telemetry(*args, **kwargs):
    result = _NATIVE_SIMULATE_GAME(*args, **kwargs)
    _TELEMETRY.capture(result)
    return result


def _simulate_scrimmage_play_with_snap_world(
    state,
    offense,
    defense_strength,
    rng,
    defense=None,
):
    responsibility_key = snap_responsibility_key(
        offense_team_id=state.possession,
        defense_team_id=state.defense,
        quarter=state.quarter,
        seconds_remaining=state.seconds_remaining,
        down=state.down,
        distance=state.distance,
        yardline_100=state.yardline_100,
        offense_score=state.away_score,
        defense_score=state.home_score,
    )
    active_offense, active_defense, _ = prepare_snap_world(
        state=state,
        offense=offense,
        defense=defense,
        responsibility_key=responsibility_key,
    )
    event = _NATIVE_SIMULATE_SCRIMMAGE_PLAY(
        state,
        active_offense,
        defense_strength,
        rng,
        defense=active_defense,
    )
    annotate_event(event, responsibility_key=responsibility_key)
    return event


def _choose_target_with_opportunity(players, ecology, category, rng, *, shrinkage_samples=45.0):
    selected = _NATIVE_CHOOSE_TARGET_FOR_DEPTH(
        players,
        ecology,
        category,
        rng,
        shrinkage_samples=shrinkage_samples,
    )
    _OPPORTUNITY_TAIL.register_pass(selected, players)
    return selected


def _field_read_target_with_opportunity(*args, **kwargs):
    target, matchup = _NATIVE_FIELD_READ_TARGET(*args, **kwargs)
    offense = args[0] if args else kwargs["offense"]
    _OPPORTUNITY_TAIL.register_pass(target, offense.receivers)
    return target, matchup


def _choose_rusher_with_opportunity(players, ecology, category, rng, *, shrinkage_samples=60.0):
    selected = _NATIVE_CHOOSE_RUSHER_FOR_GEOMETRY(
        players,
        ecology,
        category,
        rng,
        shrinkage_samples=shrinkage_samples,
    )
    _OPPORTUNITY_TAIL.register_run(selected, players)
    return selected


def _pass_interaction_with_opportunity(receiver_explosiveness: float, coverage_strength: float) -> float:
    base = _NATIVE_PASS_INTERACTION(receiver_explosiveness, coverage_strength)
    return float(np.clip(base * _OPPORTUNITY_TAIL.pass_multiplier(), 0.55, 1.90))


def _run_skill_edge_with_opportunity(*, explosiveness: float, runner_power: float, tackling: float) -> float:
    base = _NATIVE_RUN_SKILL_EDGE(
        explosiveness=explosiveness,
        runner_power=runner_power,
        tackling=tackling,
    )
    return float(np.clip(base * _OPPORTUNITY_TAIL.run_multiplier(), 0.55, 1.90))


def _first_out_path() -> Path:
    if "--first-out" in sys.argv:
        index = sys.argv.index("--first-out")
        if index + 1 < len(sys.argv):
            return Path(sys.argv[index + 1])
    return Path("artifacts/reality-loop-v2-smoke")


def configure_reality_loop_v2() -> None:
    """Activate the Reality Loop causal seams while preserving calibrated v2/v3 priors."""
    resolution_ecology._COARSE_MATCHUP_AUTHORITY = 0.35
    resolution_ecology._COARSE_RUN_MATCHUP_AUTHORITY = 0.25
    runner._PASS_MATCHUP_AUTHORITY_OVERRIDE = 0.35

    runner.enhanced_team_identity = _team_identity_with_skill_telemetry
    runner.enhanced_defensive_unit = _defensive_unit_with_skill_telemetry
    runner.integrated._attach_historical_intent_ecology = _attach_reality_loop_policy
    runner.integrated.simulate_game = _simulate_game_with_telemetry

    # game_loop_v13 imported the scrimmage function by name at module import, so patch that
    # production global explicitly. Each snap now receives one 11v11 world before play resolution.
    game_loop_v13.simulate_scrimmage_play = _simulate_scrimmage_play_with_snap_world

    matchup_kernel.resolve_pass_snap = resolve_pass_snap_v2
    matchup_kernel.resolve_run_snap = resolve_run_snap_v2

    # V3 opportunity × skill × matchup remains downstream of the v5 participant world.
    intent_ecology.choose_target_for_depth = _choose_target_with_opportunity
    intent_ecology.choose_rusher_for_geometry = _choose_rusher_with_opportunity
    play_kernel._field_read_target = _field_read_target_with_opportunity
    progressive_skill_tail_v2._pass_interaction = _pass_interaction_with_opportunity
    progressive_skill_tail_v2._run_skill_edge = _run_skill_edge_with_opportunity

    resolution_ecology.resolve_run_ecology = resolve_run_ecology_progressive_skill_v2
    resolution_ecology.sample_yac = sample_yac_progressive_skill_v2
    play_kernel.resolve_run_contact = resolve_run_contact_progressive_skill_v2
    play_kernel._sample_snap_cadence = sample_snap_cadence_v2


def main() -> None:
    configure_reality_loop_v2()
    runner.main()
    out = _first_out_path()
    _TELEMETRY.write(out)
    _SKILL_TELEMETRY.write(out)
    _OPPORTUNITY_TAIL.write(out)
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        import json

        manifest = json.loads(manifest_path.read_text())
        manifest.update(
            {
                "snap_world_v5_active": True,
                "per_world_snap_participant_telemetry": True,
                "actual_11v11_personnel_packages_active": True,
                "alignment_roles_active": True,
                "defensive_shell_and_rush_plan_active": True,
                "production_individual_ol_registration_active": True,
                "stronger_local_run_interaction_active": True,
            }
        )
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
