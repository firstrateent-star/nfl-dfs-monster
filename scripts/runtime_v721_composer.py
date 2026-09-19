from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import run_reality_loop_v2_smoke as v61
import runtime_v720_composer as v720

from monster.reality.game_script_v721 import (
    configure_v721_hooks,
    defensive_intent_v721,
    policy_for_state_v721,
    reset_v721_state,
    simulate_scrimmage_play_v721,
    team_identity_v721,
    write_game_script_telemetry_v721,
)
from monster.sim import game_loop_v13, play_kernel, reality_snap_v5


@dataclass(frozen=True)
class RuntimeFingerprintV721:
    version: str
    base_architecture: str
    base_runtime_hash: str
    situation_clock_runtime: str
    scrimmage_clock_runtime: str
    defensive_script_runtime: str
    team_identity_runtime: str
    timeout_state_active: bool
    two_minute_cadence_active: bool
    four_minute_drain_active: bool
    garbage_defense_active: bool
    blowout_preservation_active: bool
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
    base = v720.runtime_fingerprint_v720()
    return {
        "version": "v7.2.1-shadow-game-script-clock",
        "base_architecture": "v7.2.0-shadow-reality-allocation",
        "base_runtime_hash": base.runtime_hash,
        "situation_clock_runtime": _callable_id(play_kernel._policy_for_state),
        "scrimmage_clock_runtime": _callable_id(game_loop_v13.simulate_scrimmage_play),
        "defensive_script_runtime": _callable_id(reality_snap_v5._intent_for_snap),
        "team_identity_runtime": _callable_id(v61.runner.enhanced_team_identity),
        "timeout_state_active": True,
        "two_minute_cadence_active": True,
        "four_minute_drain_active": True,
        "garbage_defense_active": True,
        "blowout_preservation_active": True,
        "market_inputs_to_football": False,
        "direct_fantasy_inputs_to_football": False,
        "direct_score_adjustment": False,
    }


def runtime_fingerprint_v721() -> RuntimeFingerprintV721:
    payload = _fingerprint_payload()
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return RuntimeFingerprintV721(**payload, runtime_hash=digest)


def compose_v721_runtime() -> None:
    """Add game-script/clock causality around the frozen V7.2 football runtime."""

    v720.compose_v720_runtime()
    reset_v721_state()

    configure_v721_hooks(
        policy=play_kernel._policy_for_state,
        scrimmage=game_loop_v13.simulate_scrimmage_play,
        defensive_intent=reality_snap_v5._intent_for_snap,
        team_identity=v61.runner.enhanced_team_identity,
    )

    # The pass/run chooser remains V7.2's hierarchical game-flow brain. This hook
    # changes only clock urgency/cadence on the legacy SituationPolicy path used
    # by the snap cadence sampler.
    play_kernel._policy_for_state = policy_for_state_v721

    # Wrap the complete V7.2 live-state/snap-world play so timeouts and
    # preservation alter opportunity supply without bypassing participation.
    game_loop_v13.simulate_scrimmage_play = simulate_scrimmage_play_v721

    # Extend late multi-score defensive posture without reading the realized
    # offensive play call.
    reality_snap_v5._intent_for_snap = defensive_intent_v721

    # Register the full current QB depth identities while the production runner
    # constructs each team. This makes terminal blowout QB preservation causal.
    v61.runner.enhanced_team_identity = team_identity_v721


def _runtime_integrity_errors() -> list[str]:
    errors: list[str] = []
    if play_kernel._policy_for_state is not policy_for_state_v721:
        errors.append("V7.2.1 clock-aware situation policy is not active")
    if game_loop_v13.simulate_scrimmage_play is not simulate_scrimmage_play_v721:
        errors.append("V7.2.1 scrimmage clock wrapper is not active")
    if reality_snap_v5._intent_for_snap is not defensive_intent_v721:
        errors.append("V7.2.1 late defensive script is not active")
    if v61.runner.enhanced_team_identity is not team_identity_v721:
        errors.append("V7.2.1 QB-depth registration is not active")
    return errors


def assert_v721_runtime() -> RuntimeFingerprintV721:
    errors = _runtime_integrity_errors()
    if errors:
        raise RuntimeError("v7.2.1 runtime integrity failure: " + "; ".join(errors))
    return runtime_fingerprint_v721()


def write_runtime_fingerprint_v721(out: Path) -> RuntimeFingerprintV721:
    fingerprint = assert_v721_runtime()
    out.mkdir(parents=True, exist_ok=True)
    (out / "runtime_fingerprint_v721.json").write_text(
        json.dumps(asdict(fingerprint), indent=2) + "\n",
        encoding="utf-8",
    )
    write_game_script_telemetry_v721(out)
    return fingerprint
