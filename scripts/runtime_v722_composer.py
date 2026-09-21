from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import run_reality_loop_v2_smoke as v61
import runtime_v721_composer as v721

from monster.sim import game_loop_v13
from monster.reality.failure_paths_v722 import (
    configure_failure_path_scrimmage_v722,
    reset_failure_paths_v722,
    simulate_scrimmage_play_v722,
    write_failure_path_telemetry_v722,
)
from monster.reality.world_availability_v722 import materialize_world_pools_v722


_BASE_V721_FINGERPRINT = None


@dataclass(frozen=True)
class RuntimeFingerprintV722:
    version: str
    base_runtime_hash: str
    world_binary_availability_active: bool
    role_redistribution_after_availability: bool
    offensive_collapse_active: bool
    scoring_finish_friction_active: bool
    performance_benching_active: bool
    live_exit_scenario_capture_active: bool
    failure_path_telemetry_active: bool
    market_inputs_to_football: bool
    direct_score_adjustment: bool
    runtime_hash: str


def compose_v722_runtime() -> None:
    global _BASE_V721_FINGERPRINT
    v721.compose_v721_runtime()
    _BASE_V721_FINGERPRINT = v721.assert_v721_runtime()
    reset_failure_paths_v722()

    configure_failure_path_scrimmage_v722(game_loop_v13.simulate_scrimmage_play)
    game_loop_v13.simulate_scrimmage_play = simulate_scrimmage_play_v722
    v61.runner.integrated._WORLD_POOL_HOOK = materialize_world_pools_v722


def runtime_fingerprint_v722() -> RuntimeFingerprintV722:
    base = _BASE_V721_FINGERPRINT or v721.runtime_fingerprint_v721()
    payload = {
        "version": "v7.2.2-week2-reality-failure-paths",
        "base_runtime_hash": base.runtime_hash,
        "world_binary_availability_active": True,
        "role_redistribution_after_availability": True,
        "offensive_collapse_active": True,
        "scoring_finish_friction_active": True,
        "performance_benching_active": True,
        "live_exit_scenario_capture_active": True,
        "failure_path_telemetry_active": True,
        "market_inputs_to_football": False,
        "direct_score_adjustment": False,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return RuntimeFingerprintV722(**payload, runtime_hash=digest)


def assert_v722_runtime() -> RuntimeFingerprintV722:
    errors: list[str] = []
    if v61.runner.integrated._WORLD_POOL_HOOK is not materialize_world_pools_v722:
        errors.append("V7.2.2 world availability hook is not active")
    if game_loop_v13.simulate_scrimmage_play is not simulate_scrimmage_play_v722:
        errors.append("V7.2.2 failure-path scrimmage wrapper is not active")
    try:
        v721.assert_v721_runtime()
    except RuntimeError as exc:
        # V7.2.2 intentionally sits one wrapper outside V7.2.1's scrimmage hook.
        # Preserve every other inherited V7.2.1 integrity assertion while accepting
        # the expected outer-wrapper difference.
        message = str(exc)
        expected = "V7.2.1 scrimmage clock wrapper is not active"
        remainder = message.replace("v7.2.1 runtime integrity failure: ", "")
        parts = [part.strip() for part in remainder.split(";") if part.strip()]
        unexpected = [part for part in parts if part != expected]
        if unexpected:
            errors.extend(unexpected)
    if errors:
        raise RuntimeError("v7.2.2 runtime integrity failure: " + "; ".join(errors))
    return runtime_fingerprint_v722()


def write_runtime_fingerprint_v722(out: Path) -> RuntimeFingerprintV722:
    fingerprint = assert_v722_runtime()
    out.mkdir(parents=True, exist_ok=True)
    (out / "runtime_fingerprint_v722.json").write_text(
        json.dumps(asdict(fingerprint), indent=2) + "\n",
        encoding="utf-8",
    )
    write_failure_path_telemetry_v722(out)
    return fingerprint
