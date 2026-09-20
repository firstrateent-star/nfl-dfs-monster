from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import run_reality_loop_v2_smoke as v61
import runtime_v721_composer as v721

from monster.reality.world_availability_v722 import materialize_world_pools_v722


@dataclass(frozen=True)
class RuntimeFingerprintV722:
    version: str
    base_runtime_hash: str
    world_binary_availability_active: bool
    role_redistribution_after_availability: bool
    market_inputs_to_football: bool
    runtime_hash: str


def compose_v722_runtime() -> None:
    v721.compose_v721_runtime()
    v61.runner.integrated._WORLD_POOL_HOOK = materialize_world_pools_v722


def runtime_fingerprint_v722() -> RuntimeFingerprintV722:
    base = v721.runtime_fingerprint_v721()
    payload = {
        "version": "v7.2.2-world-binary-availability",
        "base_runtime_hash": base.runtime_hash,
        "world_binary_availability_active": True,
        "role_redistribution_after_availability": True,
        "market_inputs_to_football": False,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return RuntimeFingerprintV722(**payload, runtime_hash=digest)


def assert_v722_runtime() -> RuntimeFingerprintV722:
    if v61.runner.integrated._WORLD_POOL_HOOK is not materialize_world_pools_v722:
        raise RuntimeError("V7.2.2 world availability hook is not active")
    v721.assert_v721_runtime()
    return runtime_fingerprint_v722()


def write_runtime_fingerprint_v722(out: Path) -> RuntimeFingerprintV722:
    fingerprint = assert_v722_runtime()
    out.mkdir(parents=True, exist_ok=True)
    (out / "runtime_fingerprint_v722.json").write_text(
        json.dumps(asdict(fingerprint), indent=2) + "\n",
        encoding="utf-8",
    )
    return fingerprint
