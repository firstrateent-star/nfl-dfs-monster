from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import runtime_v725_composer as v725

from monster.sim import resolution_ecology
from monster.sim.intent_ecology import PassDepthOutcome, RunGeometryOutcome

_BASE_V725_FINGERPRINT = None


@dataclass(frozen=True)
class RuntimeFingerprintV726:
    version: str
    base_runtime_hash: str
    early_down_run_survival_active: bool
    early_down_negative_completion_priors_active: bool
    touchdown_conversion_observer_preserved: bool
    possession_geometry_v725_preserved: bool
    field_goal_ecology_v725_preserved: bool
    v724_collapse_sampler_preserved: bool
    third_down_execution_directly_adjusted: bool
    scoring_event_authority_preserved: bool
    market_inputs_to_football: bool
    direct_score_adjustment: bool
    runtime_hash: str


def compose_v726_runtime() -> None:
    global _BASE_V725_FINGERPRINT
    v725.compose_v725_runtime()
    _BASE_V725_FINGERPRINT = v725.assert_v725_runtime()


def runtime_fingerprint_v726() -> RuntimeFingerprintV726:
    base = _BASE_V725_FINGERPRINT or v725.runtime_fingerprint_v725()
    payload = {
        "version": "v7.2.6-series-survival-ecology",
        "base_runtime_hash": base.runtime_hash,
        "early_down_run_survival_active": True,
        "early_down_negative_completion_priors_active": True,
        "touchdown_conversion_observer_preserved": True,
        "possession_geometry_v725_preserved": True,
        "field_goal_ecology_v725_preserved": True,
        "v724_collapse_sampler_preserved": True,
        "third_down_execution_directly_adjusted": False,
        "scoring_event_authority_preserved": True,
        "market_inputs_to_football": False,
        "direct_score_adjustment": False,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return RuntimeFingerprintV726(**payload, runtime_hash=digest)


def assert_v726_runtime() -> RuntimeFingerprintV726:
    errors: list[str] = []
    run_fields = RunGeometryOutcome.__dataclass_fields__
    pass_fields = PassDepthOutcome.__dataclass_fields__
    if "early_down_gain_5plus_rate" not in run_fields:
        errors.append("early-down run five-plus prior is missing")
    if "early_down_negative_completion_rate" not in pass_fields:
        errors.append("early-down pass loss prior is missing")

    run_signature = inspect.signature(resolution_ecology.resolve_run_ecology)
    yac_signature = inspect.signature(resolution_ecology.sample_yac)
    if "early_down" not in run_signature.parameters:
        errors.append("early-down run resolver switch is missing")
    if "early_down" not in yac_signature.parameters:
        errors.append("early-down YAC resolver switch is missing")

    v725.assert_v725_runtime()
    if errors:
        raise RuntimeError("v7.2.6 runtime integrity failure: " + "; ".join(errors))
    return runtime_fingerprint_v726()


def write_runtime_fingerprint_v726(out: Path) -> RuntimeFingerprintV726:
    fingerprint = assert_v726_runtime()
    out.mkdir(parents=True, exist_ok=True)
    (out / "runtime_fingerprint_v726.json").write_text(
        json.dumps(asdict(fingerprint), indent=2) + "\n",
        encoding="utf-8",
    )
    return fingerprint
