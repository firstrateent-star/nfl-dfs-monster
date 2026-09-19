from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import run_reality_loop_v2_smoke as v61
import run_reality_loop_v634 as v634
import runtime_v701_composer as v701

from monster.reality.defensive_attribution_v72 import (
    attribute_defensive_box_score_v72,
)
from monster.reality.live_state_v72 import (
    configure_base_scrimmage_hook_v72,
    configure_live_state_v72,
    simulate_scrimmage_play_v72,
    write_live_state_telemetry_v72,
)
from monster.reality.participation_authority_v72 import (
    install_participation_authority_v72,
    register_team_units_v72,
    weighted_without_replacement_v72,
)
from monster.reality.qb_rush_authority_v72 import (
    choose_rusher_for_geometry_v72,
)
from monster.reality.receiver_topology_v72 import (
    choose_target_for_depth_v72,
    field_read_target_v72,
)
from monster.reality.special_teams_identity_v72 import (
    capture_special_teams_v72,
    configure_base_special_teams_hooks_v72,
    configure_special_teams_identities_v72,
    kickoff_loop_v72,
    simulate_field_goal_v72,
    simulate_kickoff_v72,
    simulate_punt_v72,
    write_special_teams_telemetry_v72,
)
from monster.sim import game_loop_v13, intent_ecology, play_kernel, reality_snap_v5


@dataclass(frozen=True)
class RuntimeFingerprintV720:
    version: str
    base_architecture: str
    base_runtime_hash: str
    ol_registration_runtime: str
    snap_participation_runtime: str
    designed_qb_run_runtime: str
    target_concept_runtime: str
    field_read_runtime: str
    defensive_attribution_runtime: str
    live_state_runtime: str
    punt_runtime: str
    field_goal_runtime: str
    kickoff_runtime: str
    scoreboard_authority: str
    market_inputs_to_football: bool
    direct_fantasy_inputs_to_football: bool
    direct_score_adjustment: bool
    runtime_hash: str


_BASE_SIMULATE_GAME: Callable[..., Any] | None = None


def _callable_id(fn: Callable[..., Any]) -> str:
    return (
        f"{getattr(fn, '__module__', '<unknown>')}."
        f"{getattr(fn, '__qualname__', getattr(fn, '__name__', '<unknown>'))}"
    )


def install_current_role_guard_v720(personnel) -> None:
    install_participation_authority_v72(personnel)
    configure_special_teams_identities_v72(personnel)
    configure_live_state_v72()


def _simulate_game_v720(*args, **kwargs):
    if _BASE_SIMULATE_GAME is None:
        raise RuntimeError("v7.2 base game runtime has not been configured")
    result = _BASE_SIMULATE_GAME(*args, **kwargs)
    capture_special_teams_v72(result)
    return result


def _fingerprint_payload() -> dict[str, object]:
    base = v701.runtime_fingerprint_v701()
    integrated = v61.runner.integrated
    return {
        "version": "v7.2.0-shadow-reality-allocation",
        "base_architecture": "v7.0.1-shadow-qb-rush-family-authority",
        "base_runtime_hash": base.runtime_hash,
        "ol_registration_runtime": _callable_id(v61.register_team_units),
        "snap_participation_runtime": _callable_id(
            reality_snap_v5._weighted_without_replacement
        ),
        "designed_qb_run_runtime": _callable_id(
            intent_ecology.choose_rusher_for_geometry
        ),
        "target_concept_runtime": _callable_id(
            intent_ecology.choose_target_for_depth
        ),
        "field_read_runtime": _callable_id(play_kernel._field_read_target),
        "defensive_attribution_runtime": _callable_id(
            integrated.attribute_defensive_box_score
        ),
        "live_state_runtime": _callable_id(game_loop_v13.simulate_scrimmage_play),
        "punt_runtime": _callable_id(game_loop_v13.simulate_punt),
        "field_goal_runtime": _callable_id(game_loop_v13.simulate_field_goal),
        "kickoff_runtime": _callable_id(game_loop_v13.simulate_kickoff),
        "scoreboard_authority": "monster.sim.game_loop_v13 event-derived football state",
        "market_inputs_to_football": False,
        "direct_fantasy_inputs_to_football": False,
        "direct_score_adjustment": False,
    }


def runtime_fingerprint_v720() -> RuntimeFingerprintV720:
    payload = _fingerprint_payload()
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return RuntimeFingerprintV720(**payload, runtime_hash=digest)


def compose_v720_runtime() -> None:
    """Compose V7.2 on top of the frozen V7.1 shadow.

    The order is deliberate:
    current participation -> QB concept entry -> receiver topology ->
    defensive attribution -> live state -> special-team identities.
    """

    global _BASE_SIMULATE_GAME

    v701.compose_v701_runtime()
    integrated = v61.runner.integrated

    # The v6.3.4 pool compiler calls this global after personnel is read.
    v634.install_current_role_guard = install_current_role_guard_v720

    # Current depth owns the exact OL seats; stale snap history remains evidence
    # inside the selected current seat instead of choosing the five-man line.
    v61.register_team_units = register_team_units_v72

    # Snap-level skill/defensive participation now consumes the same current
    # authority registry and the live in-game availability multiplier.
    reality_snap_v5._weighted_without_replacement = weighted_without_replacement_v72

    # Opportunity is conditional on participation and concept.
    intent_ecology.choose_rusher_for_geometry = choose_rusher_for_geometry_v72
    intent_ecology.choose_target_for_depth = choose_target_for_depth_v72
    play_kernel._field_read_target = field_read_target_v72

    # Box attribution consumes the already-realized snap topology.
    integrated.attribute_defensive_box_score = attribute_defensive_box_score_v72

    # Install live state around, not inside, the inherited V7.1 snap/play engine.
    if game_loop_v13.simulate_scrimmage_play is not simulate_scrimmage_play_v72:
        configure_base_scrimmage_hook_v72(game_loop_v13.simulate_scrimmage_play)
    game_loop_v13.simulate_scrimmage_play = simulate_scrimmage_play_v72

    # Preserve whichever v6.3.x special-team ecology V7.1 composed, then add
    # player identity/capability arguments on top of it.
    if game_loop_v13.simulate_punt is not simulate_punt_v72:
        configure_base_special_teams_hooks_v72(
            punt=game_loop_v13.simulate_punt,
            field_goal=game_loop_v13.simulate_field_goal,
            kickoff=game_loop_v13.simulate_kickoff,
            kickoff_loop=game_loop_v13._kickoff,
        )
    game_loop_v13.simulate_punt = simulate_punt_v72
    game_loop_v13.simulate_field_goal = simulate_field_goal_v72
    game_loop_v13.simulate_kickoff = simulate_kickoff_v72
    game_loop_v13._kickoff = kickoff_loop_v72

    if integrated.simulate_game is not _simulate_game_v720:
        _BASE_SIMULATE_GAME = integrated.simulate_game
    integrated.simulate_game = _simulate_game_v720


def _runtime_integrity_errors() -> list[str]:
    integrated = v61.runner.integrated
    errors: list[str] = []
    if v61.register_team_units is not register_team_units_v72:
        errors.append("seat-aware OL registration is not active")
    if reality_snap_v5._weighted_without_replacement is not weighted_without_replacement_v72:
        errors.append("current-role snap participation is not active")
    if intent_ecology.choose_rusher_for_geometry is not choose_rusher_for_geometry_v72:
        errors.append("QB concept-entry rushing runtime is not active")
    if intent_ecology.choose_target_for_depth is not choose_target_for_depth_v72:
        errors.append("receiver concept topology is not active")
    if play_kernel._field_read_target is not field_read_target_v72:
        errors.append("topology-aware field read is not active")
    if integrated.attribute_defensive_box_score is not attribute_defensive_box_score_v72:
        errors.append("causal defensive attribution is not active")
    if game_loop_v13.simulate_scrimmage_play is not simulate_scrimmage_play_v72:
        errors.append("live mutation wrapper is not active")
    if game_loop_v13.simulate_punt is not simulate_punt_v72:
        errors.append("punter identity runtime is not active")
    if game_loop_v13.simulate_field_goal is not simulate_field_goal_v72:
        errors.append("kicker identity runtime is not active")
    if game_loop_v13.simulate_kickoff is not simulate_kickoff_v72:
        errors.append("kickoff identity runtime is not active")
    if game_loop_v13._kickoff is not kickoff_loop_v72:
        errors.append("kickoff context runtime is not active")
    if integrated.simulate_game is not _simulate_game_v720:
        errors.append("V7.2 game telemetry wrapper is not active")
    return errors


def assert_v720_runtime() -> RuntimeFingerprintV720:
    errors = _runtime_integrity_errors()
    if errors:
        raise RuntimeError("v7.2 runtime integrity failure: " + "; ".join(errors))
    return runtime_fingerprint_v720()


def write_runtime_fingerprint_v720(out: Path) -> RuntimeFingerprintV720:
    fingerprint = assert_v720_runtime()
    out.mkdir(parents=True, exist_ok=True)
    (out / "runtime_fingerprint_v720.json").write_text(
        json.dumps(asdict(fingerprint), indent=2) + "\n",
        encoding="utf-8",
    )
    write_live_state_telemetry_v72(out)
    write_special_teams_telemetry_v72(out)
    return fingerprint
