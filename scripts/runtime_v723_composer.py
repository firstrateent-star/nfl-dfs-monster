from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import run_reality_loop_v2_smoke as v61
import runtime_v722_composer as v722

from monster.reality import failure_paths_v722
from monster.reality.failure_paths_v723 import (
    configure_scrimmage_v723,
    materialize_world_pools_v723,
    reset_v723_state,
    simulate_scrimmage_play_v723,
    update_qb_performance_v723,
    write_calibration_telemetry_v723,
)
from monster.sim import game_loop_v13


_BASE_V722_FINGERPRINT = None


@dataclass(frozen=True)
class RuntimeFingerprintV723:
    version: str
    base_runtime_hash: str
    calibrated_left_tail_active: bool
    explicit_drive_stall_active: bool
    scoring_event_authority_preserved: bool
    tightened_qb_benching_active: bool
    upper_worlds_unmodified_by_tail_sampler: bool
    market_inputs_to_football: bool
    direct_score_adjustment: bool
    runtime_hash: str


def compose_v723_runtime() -> None:
    global _BASE_V722_FINGERPRINT

    v722.compose_v722_runtime()
    _BASE_V722_FINGERPRINT = v722.assert_v722_runtime()
    reset_v723_state()

    # Reuse V7.2.2's availability world, then recalibrate only the failure state.
    v61.runner.integrated._WORLD_POOL_HOOK = materialize_world_pools_v723

    # Tighten the performance-removal policy used inside the inherited V7.2.2 wrapper.
    failure_paths_v722._update_qb_performance_v722 = update_qb_performance_v723

    # Add causal finishing failures outside the complete V7.2.2 snap runtime.
    configure_scrimmage_v723(game_loop_v13.simulate_scrimmage_play)
    game_loop_v13.simulate_scrimmage_play = simulate_scrimmage_play_v723


def runtime_fingerprint_v723() -> RuntimeFingerprintV723:
    base = _BASE_V722_FINGERPRINT or v722.runtime_fingerprint_v722()
    payload = {
        "version": "v7.2.3-left-tail-drive-resolution-calibration",
        "base_runtime_hash": base.runtime_hash,
        "calibrated_left_tail_active": True,
        "explicit_drive_stall_active": True,
        "scoring_event_authority_preserved": True,
        "tightened_qb_benching_active": True,
        "upper_worlds_unmodified_by_tail_sampler": True,
        "market_inputs_to_football": False,
        "direct_score_adjustment": False,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return RuntimeFingerprintV723(**payload, runtime_hash=digest)


def assert_v723_runtime() -> RuntimeFingerprintV723:
    errors: list[str] = []
    if v61.runner.integrated._WORLD_POOL_HOOK is not materialize_world_pools_v723:
        errors.append("V7.2.3 calibrated world hook is not active")
    if game_loop_v13.simulate_scrimmage_play is not simulate_scrimmage_play_v723:
        errors.append("V7.2.3 drive-finishing wrapper is not active")
    if failure_paths_v722._update_qb_performance_v722 is not update_qb_performance_v723:
        errors.append("V7.2.3 tightened QB bench policy is not active")

    try:
        v722.assert_v722_runtime()
    except RuntimeError as exc:
        message = str(exc).replace("v7.2.2 runtime integrity failure: ", "")
        expected = {
            "V7.2.2 world availability hook is not active",
            "V7.2.2 failure-path scrimmage wrapper is not active",
        }
        unexpected = [
            part.strip()
            for part in message.split(";")
            if part.strip() and part.strip() not in expected
        ]
        errors.extend(unexpected)

    if errors:
        raise RuntimeError("v7.2.3 runtime integrity failure: " + "; ".join(errors))
    return runtime_fingerprint_v723()


def write_runtime_fingerprint_v723(out: Path) -> RuntimeFingerprintV723:
    fingerprint = assert_v723_runtime()
    out.mkdir(parents=True, exist_ok=True)
    (out / "runtime_fingerprint_v723.json").write_text(
        json.dumps(asdict(fingerprint), indent=2) + "\n",
        encoding="utf-8",
    )
    write_calibration_telemetry_v723(out)
    return fingerprint
