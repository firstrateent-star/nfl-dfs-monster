from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import run_reality_loop_v2_smoke as v61
import runtime_v723_composer as v723

from monster.reality import failure_paths_v722, failure_paths_v723
from monster.reality.failure_paths_v724 import (
    _stall_probability_v724,
    install_drive_consequence_v724,
    materialize_world_pools_v724,
    reset_v724_state,
    write_consequence_telemetry_v724,
)
from monster.sim import game_loop_v13

_BASE_V723_FINGERPRINT = None


@dataclass(frozen=True)
class RuntimeFingerprintV724:
    version: str
    base_runtime_hash: str
    left_tail_mass_unchanged_from_v723: bool
    mode_gated_drive_consequence_active: bool
    normal_world_finishing_protected: bool
    hard_collapse_finishing_amplified: bool
    scoring_event_authority_preserved: bool
    tightened_qb_benching_preserved: bool
    market_inputs_to_football: bool
    direct_score_adjustment: bool
    runtime_hash: str


def compose_v724_runtime() -> None:
    global _BASE_V723_FINGERPRINT

    v723.compose_v723_runtime()
    _BASE_V723_FINGERPRINT = v723.assert_v723_runtime()
    reset_v724_state()

    # Preserve V7.2.3's exact collapse sampler. Only remap sampled mode to
    # scoring-territory consequence after the inherited world is materialized.
    v61.runner.integrated._WORLD_POOL_HOOK = materialize_world_pools_v724
    install_drive_consequence_v724()


def runtime_fingerprint_v724() -> RuntimeFingerprintV724:
    base = _BASE_V723_FINGERPRINT or v723.runtime_fingerprint_v723()
    payload = {
        "version": "v7.2.4-collapse-consequence-calibration",
        "base_runtime_hash": base.runtime_hash,
        "left_tail_mass_unchanged_from_v723": True,
        "mode_gated_drive_consequence_active": True,
        "normal_world_finishing_protected": True,
        "hard_collapse_finishing_amplified": True,
        "scoring_event_authority_preserved": True,
        "tightened_qb_benching_preserved": True,
        "market_inputs_to_football": False,
        "direct_score_adjustment": False,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return RuntimeFingerprintV724(**payload, runtime_hash=digest)


def assert_v724_runtime() -> RuntimeFingerprintV724:
    errors: list[str] = []
    if v61.runner.integrated._WORLD_POOL_HOOK is not materialize_world_pools_v724:
        errors.append("V7.2.4 consequence-calibrated world hook is not active")
    if failure_paths_v723._stall_probability is not _stall_probability_v724:
        errors.append("V7.2.4 mode-gated stall policy is not active")
    if game_loop_v13.simulate_scrimmage_play is not failure_paths_v723.simulate_scrimmage_play_v723:
        errors.append("V7.2.3 event-conserving finishing wrapper is not preserved")
    if (
        failure_paths_v722._update_qb_performance_v722
        is not failure_paths_v723.update_qb_performance_v723
    ):
        errors.append("V7.2.3 tightened QB bench policy is not preserved")

    if errors:
        raise RuntimeError("v7.2.4 runtime integrity failure: " + "; ".join(errors))
    return runtime_fingerprint_v724()


def write_runtime_fingerprint_v724(out: Path) -> RuntimeFingerprintV724:
    fingerprint = assert_v724_runtime()
    out.mkdir(parents=True, exist_ok=True)
    (out / "runtime_fingerprint_v724.json").write_text(
        json.dumps(asdict(fingerprint), indent=2) + "\n",
        encoding="utf-8",
    )

    # Persist all three observability layers so the next reality comparison can
    # distinguish inherited V7.2.2 state, V7.2.3 sampling, and V7.2.4 consequence.
    failure_paths_v722.write_failure_path_telemetry_v722(out)
    failure_paths_v723.write_calibration_telemetry_v723(out)
    write_consequence_telemetry_v724(out)
    return fingerprint
