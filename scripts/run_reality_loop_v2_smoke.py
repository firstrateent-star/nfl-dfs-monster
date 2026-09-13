from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl
import run_week1_v13_dispersion_test as runner

from monster.sim import (
    intent_ecology,
    matchup_kernel,
    play_kernel,
    progressive_skill_tail_v2,
    resolution_ecology,
)
from monster.sim.clock_ecology_v2 import sample_snap_cadence_v2
from monster.sim.full_world_telemetry_v2 import FullWorldTelemetryV2
from monster.sim.game_flow_lookup import build_team_game_flow_policy
from monster.sim.opportunity_skill_tail_v3 import OpportunitySkillTailV3
from monster.sim.player_skill_telemetry_v2 import PlayerSkillTelemetryV2
from monster.sim.progressive_skill_tail_v2 import (
    resolve_run_contact_progressive_skill_v2,
    resolve_run_ecology_progressive_skill_v2,
    sample_yac_progressive_skill_v2,
)
from monster.sim.snap_ecology_v2 import resolve_pass_snap_v2, resolve_run_snap_v2

_NATIVE_ATTACH_INTENT = runner.integrated._attach_historical_intent_ecology
_NATIVE_SIMULATE_GAME = runner.integrated.simulate_game
_NATIVE_ENHANCED_TEAM_IDENTITY = runner.enhanced_team_identity
_NATIVE_ENHANCED_DEFENSIVE_UNIT = runner.enhanced_defensive_unit
_NATIVE_FIELD_READ_TARGET = play_kernel._field_read_target
_NATIVE_CHOOSE_TARGET_FOR_DEPTH = intent_ecology.choose_target_for_depth
_NATIVE_CHOOSE_RUSHER_FOR_GEOMETRY = intent_ecology.choose_rusher_for_geometry
_NATIVE_PASS_INTERACTION = progressive_skill_tail_v2._pass_interaction
_NATIVE_RUN_SKILL_EDGE = progressive_skill_tail_v2._run_skill_edge
_TELEMETRY = FullWorldTelemetryV2()
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
    return team


def _defensive_unit_with_skill_telemetry(*args, **kwargs):
    defense = _NATIVE_ENHANCED_DEFENSIVE_UNIT(*args, **kwargs)
    _SKILL_TELEMETRY.register_defensive_unit(defense)
    return defense


def _simulate_game_with_telemetry(*args, **kwargs):
    result = _NATIVE_SIMULATE_GAME(*args, **kwargs)
    _TELEMETRY.capture(result)
    return result


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
    """Activate the Reality Loop v2 causal seams without mutating stable v1.3 defaults."""
    resolution_ecology._COARSE_MATCHUP_AUTHORITY = 0.35
    resolution_ecology._COARSE_RUN_MATCHUP_AUTHORITY = 0.25
    runner._PASS_MATCHUP_AUTHORITY_OVERRIDE = 0.35

    runner.enhanced_team_identity = _team_identity_with_skill_telemetry
    runner.enhanced_defensive_unit = _defensive_unit_with_skill_telemetry
    runner.integrated._attach_historical_intent_ecology = _attach_reality_loop_policy
    runner.integrated.simulate_game = _simulate_game_with_telemetry

    matchup_kernel.resolve_pass_snap = resolve_pass_snap_v2
    matchup_kernel.resolve_run_snap = resolve_run_snap_v2

    # V3 experiment: opportunity alone has zero tail authority. It only amplifies a signed rich
    # skill edge; the existing progressive resolver still owns historical priors and defender
    # matchup. This isolates opportunity × skill × matchup rather than adding offense globally.
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


if __name__ == "__main__":
    main()
