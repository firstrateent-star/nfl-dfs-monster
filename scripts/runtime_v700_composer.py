from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import run_reality_loop_v2_smoke as v61
import runtime_v638_composer as v638

from monster.reality.opportunity import (
    choose_rusher_for_geometry_v7,
    choose_target_for_depth_v7,
    field_read_target_v7,
    install_participation_first_opportunity_v7,
)
from monster.sim import (
    game_loop_v13,
    intent_ecology,
    play_kernel,
    progressive_skill_tail_v2,
    reality_v62,
)
from monster.sim.current_role_guard_v635 import sample_target_share_plan_v635
from monster.sim.current_role_guard_v638 import sample_rush_share_plan_v638


@dataclass(frozen=True)
class RuntimeFingerprintV700:
    version: str
    base_architecture: str
    base_runtime_hash: str
    scrimmage_runtime: str
    target_participation_sampler: str
    rush_participation_sampler: str
    target_assignment_runtime: str
    rusher_assignment_runtime: str
    field_read_runtime: str
    usage_based_opportunity_tail_live: bool
    score_authority: str
    market_inputs_to_football: bool
    direct_fantasy_inputs_to_football: bool
    direct_score_adjustment: bool
    runtime_hash: str


def _callable_id(fn: Callable[..., Any]) -> str:
    return (
        f"{getattr(fn, '__module__', '<unknown>')}."
        f"{getattr(fn, '__qualname__', getattr(fn, '__name__', '<unknown>'))}"
    )


def _fingerprint_payload() -> dict[str, object]:
    base = v638.runtime_fingerprint_v638()
    return {
        "version": "v7.0.0-shadow-participation-first",
        "base_architecture": "frozen-v6.3.8",
        "base_runtime_hash": base.runtime_hash,
        "scrimmage_runtime": _callable_id(game_loop_v13.simulate_scrimmage_play),
        "target_participation_sampler": _callable_id(reality_v62.sample_target_share_plan),
        "rush_participation_sampler": _callable_id(reality_v62.sample_event_rush_share_plan),
        "target_assignment_runtime": _callable_id(intent_ecology.choose_target_for_depth),
        "rusher_assignment_runtime": _callable_id(intent_ecology.choose_rusher_for_geometry),
        "field_read_runtime": _callable_id(play_kernel._field_read_target),
        "usage_based_opportunity_tail_live": False,
        "score_authority": "monster.sim.game_loop_v13 event-derived football state",
        "market_inputs_to_football": False,
        "direct_fantasy_inputs_to_football": False,
        "direct_score_adjustment": False,
    }


def runtime_fingerprint_v700() -> RuntimeFingerprintV700:
    payload = _fingerprint_payload()
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return RuntimeFingerprintV700(**payload, runtime_hash=digest)


def compose_v700_runtime() -> None:
    """Install v7 participation-first opportunity on top of frozen v6.3.8."""

    v638.compose_v638_runtime()
    install_participation_first_opportunity_v7()

    # v6's opportunity-skill tail reads pre-sampled usage_weight after participation.
    # Restore the underlying player-skill interactions so this v7 experiment removes
    # that second post-participation share authority rather than leaving it hidden.
    progressive_skill_tail_v2._pass_interaction = v61._NATIVE_PASS_INTERACTION
    progressive_skill_tail_v2._run_skill_edge = v61._NATIVE_RUN_SKILL_EDGE


def _runtime_integrity_errors() -> list[str]:
    errors: list[str] = []
    if game_loop_v13.simulate_scrimmage_play is not v61._simulate_scrimmage_play_with_snap_world:
        errors.append("v7 must inherit the production 11v11 snap-world wrapper")
    if reality_v62.sample_target_share_plan is not sample_target_share_plan_v635:
        errors.append("v7 shadow changed the frozen target participation prior")
    if reality_v62.sample_event_rush_share_plan is not sample_rush_share_plan_v638:
        errors.append("v7 shadow changed the frozen rushing participation prior")
    if intent_ecology.choose_target_for_depth is not choose_target_for_depth_v7:
        errors.append("participation-first target assignment is not active")
    if intent_ecology.choose_rusher_for_geometry is not choose_rusher_for_geometry_v7:
        errors.append("participation-first rusher assignment is not active")
    if play_kernel._field_read_target is not field_read_target_v7:
        errors.append("participation-first field read is not active")
    if progressive_skill_tail_v2._pass_interaction is not v61._NATIVE_PASS_INTERACTION:
        errors.append("usage-based receiving opportunity tail remains live")
    if progressive_skill_tail_v2._run_skill_edge is not v61._NATIVE_RUN_SKILL_EDGE:
        errors.append("usage-based rushing opportunity tail remains live")
    return errors


def assert_v700_runtime() -> RuntimeFingerprintV700:
    errors = _runtime_integrity_errors()
    if errors:
        raise RuntimeError("v7.0 runtime integrity failure: " + "; ".join(errors))
    return runtime_fingerprint_v700()


def build_week1_runtime_inputs_v700(
    *,
    policy_path: Path,
    personnel_path: Path,
    player_usage_path: Path,
    situation_context_path: Path,
):
    runtime = v638.build_week1_runtime_inputs_v638(
        policy_path=policy_path,
        personnel_path=personnel_path,
        player_usage_path=player_usage_path,
        situation_context_path=situation_context_path,
    )
    install_participation_first_opportunity_v7()
    progressive_skill_tail_v2._pass_interaction = v61._NATIVE_PASS_INTERACTION
    progressive_skill_tail_v2._run_skill_edge = v61._NATIVE_RUN_SKILL_EDGE
    runtime["fingerprint"] = assert_v700_runtime()
    return runtime


def write_runtime_fingerprint_v700(out: Path) -> RuntimeFingerprintV700:
    fingerprint = assert_v700_runtime()
    out.mkdir(parents=True, exist_ok=True)
    (out / "runtime_fingerprint_v700.json").write_text(
        json.dumps(asdict(fingerprint), indent=2) + "\n",
        encoding="utf-8",
    )
    return fingerprint
