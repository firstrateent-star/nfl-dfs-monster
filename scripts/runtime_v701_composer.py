from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import run_reality_loop_v2_smoke as v61
import runtime_v700_composer as v700

from monster.reality import qb_rush_runtime
from monster.sim import reality_v62


@dataclass(frozen=True)
class RuntimeFingerprintV701:
    version: str
    base_architecture: str
    base_runtime_hash: str
    role_world_sampler: str
    role_world_applier: str
    qb_total_rush_share_reserves_designed_mass: bool
    qb_designed_run_authority: str
    scramble_authority: str
    kneel_authority: str
    target_world_changed: bool
    new_random_draws: bool
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
    integrated = v61.runner.integrated
    base = v700.runtime_fingerprint_v700()
    return {
        "version": "v7.0.1-shadow-qb-rush-family-authority",
        "base_architecture": "v7.0.0-participation-first-shadow",
        "base_runtime_hash": base.runtime_hash,
        "role_world_sampler": _callable_id(integrated.sample_event_rush_share_plan),
        "role_world_applier": _callable_id(integrated._with_event_rush_plan),
        "qb_total_rush_share_reserves_designed_mass": False,
        "qb_designed_run_authority": (
            "current QB presence + historical non-dropback run-geometry evidence"
        ),
        "scramble_authority": "dropback QB response only",
        "kneel_authority": "not yet promoted; remains outside competitive QB rush family",
        "target_world_changed": False,
        "new_random_draws": False,
        "market_inputs_to_football": False,
        "direct_fantasy_inputs_to_football": False,
        "direct_score_adjustment": False,
    }


def runtime_fingerprint_v701() -> RuntimeFingerprintV701:
    payload = _fingerprint_payload()
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return RuntimeFingerprintV701(**payload, runtime_hash=digest)


def compose_v701_runtime() -> None:
    """Install QB rush-family separation on top of the v7.0 participation shadow."""

    v700.compose_v700_runtime()
    integrated = v61.runner.integrated

    if integrated.sample_event_rush_share_plan is not qb_rush_runtime.sample_role_world_v701:
        qb_rush_runtime.configure_base_role_hooks(
            sampler=integrated.sample_event_rush_share_plan,
            applier=integrated._with_event_rush_plan,
        )

    integrated.sample_event_rush_share_plan = qb_rush_runtime.sample_role_world_v701
    integrated._with_event_rush_plan = qb_rush_runtime.apply_role_world_v701


def _runtime_integrity_errors() -> list[str]:
    errors: list[str] = []
    integrated = v61.runner.integrated
    if integrated.sample_event_rush_share_plan is not qb_rush_runtime.sample_role_world_v701:
        errors.append("v7.1 QB-separated role sampler is not active")
    if integrated._with_event_rush_plan is not qb_rush_runtime.apply_role_world_v701:
        errors.append("v7.1 QB-separated role applier is not active")
    if not qb_rush_runtime.base_role_hooks_ready():
        errors.append("v7.1 lost the inherited role-world runtime")
    if reality_v62.sample_target_share_plan.__name__ != "sample_target_share_plan_v635":
        errors.append("v7.1 changed the frozen target participation prior")
    return errors


def assert_v701_runtime() -> RuntimeFingerprintV701:
    errors = _runtime_integrity_errors()
    if errors:
        raise RuntimeError("v7.1 runtime integrity failure: " + "; ".join(errors))
    return runtime_fingerprint_v701()


def build_week1_runtime_inputs_v701(
    *,
    policy_path: Path,
    personnel_path: Path,
    player_usage_path: Path,
    situation_context_path: Path,
):
    runtime = v700.build_week1_runtime_inputs_v700(
        policy_path=policy_path,
        personnel_path=personnel_path,
        player_usage_path=player_usage_path,
        situation_context_path=situation_context_path,
    )
    compose_v701_runtime()
    runtime["fingerprint"] = assert_v701_runtime()
    return runtime


def write_runtime_fingerprint_v701(out: Path) -> RuntimeFingerprintV701:
    fingerprint = assert_v701_runtime()
    out.mkdir(parents=True, exist_ok=True)
    (out / "runtime_fingerprint_v701.json").write_text(
        json.dumps(asdict(fingerprint), indent=2) + "\n",
        encoding="utf-8",
    )
    return fingerprint
