from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import run_reality_loop_v2_smoke as v61
import runtime_v722_composer as v722

from monster.reality.world_premise_v723 import (
    materialize_world_defense_v723,
    materialize_world_pools_v723,
    materialize_world_team_v723,
    reset_world_premise_v723,
    write_world_premise_telemetry_v723,
)


@dataclass(frozen=True)
class RuntimeFingerprintV723:
    version: str
    base_runtime_hash: str
    world_binary_availability_active: bool
    existing_role_uncertainty_reused: bool
    coherent_game_day_latents_active: bool
    offense_mechanism_routing_active: bool
    defense_mechanism_routing_active: bool
    direct_score_adjustment: bool
    direct_fantasy_adjustment: bool
    market_inputs_to_football: bool
    runtime_hash: str


def compose_v723_runtime() -> None:
    """Add epistemic world premises above the V7.2.2/V7.2.1 football runtime."""

    v722.compose_v722_runtime()
    reset_world_premise_v723()
    integrated = v61.runner.integrated
    integrated._WORLD_POOL_HOOK = materialize_world_pools_v723
    integrated._WORLD_TEAM_HOOK = materialize_world_team_v723
    integrated._WORLD_DEFENSE_HOOK = materialize_world_defense_v723


def runtime_fingerprint_v723() -> RuntimeFingerprintV723:
    base = v722.runtime_fingerprint_v722()
    payload = {
        "version": "v7.2.3-shadow-assumption-regimes",
        "base_runtime_hash": base.runtime_hash,
        "world_binary_availability_active": True,
        "existing_role_uncertainty_reused": True,
        "coherent_game_day_latents_active": True,
        "offense_mechanism_routing_active": True,
        "defense_mechanism_routing_active": True,
        "direct_score_adjustment": False,
        "direct_fantasy_adjustment": False,
        "market_inputs_to_football": False,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return RuntimeFingerprintV723(**payload, runtime_hash=digest)


def assert_v723_runtime() -> RuntimeFingerprintV723:
    integrated = v61.runner.integrated
    errors: list[str] = []
    if integrated._WORLD_POOL_HOOK is not materialize_world_pools_v723:
        errors.append("V7.2.3 world-pool premise hook is not active")
    if integrated._WORLD_TEAM_HOOK is not materialize_world_team_v723:
        errors.append("V7.2.3 world-team premise hook is not active")
    if integrated._WORLD_DEFENSE_HOOK is not materialize_world_defense_v723:
        errors.append("V7.2.3 world-defense premise hook is not active")
    if errors:
        raise RuntimeError("v7.2.3 runtime integrity failure: " + "; ".join(errors))
    v722.assert_v722_runtime()
    return runtime_fingerprint_v723()


def write_runtime_fingerprint_v723(out: Path) -> RuntimeFingerprintV723:
    fingerprint = assert_v723_runtime()
    out.mkdir(parents=True, exist_ok=True)
    (out / "runtime_fingerprint_v723.json").write_text(
        json.dumps(asdict(fingerprint), indent=2) + "\n",
        encoding="utf-8",
    )
    write_world_premise_telemetry_v723(out)
    return fingerprint
