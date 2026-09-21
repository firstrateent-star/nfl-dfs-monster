from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import polars as pl
import run_reality_loop_v2_smoke as v61
import runtime_v724_composer as v724

from monster.reality import special_teams_identity_v72
from monster.sim import game_loop_v13, resolution_ecology
from monster.sim.possession_ecology_v725 import (
    configure_possession_ecology_v725,
    reset_possession_ecology_v725,
    resolve_turnover_return_v725,
    simulate_field_goal_v725,
    simulate_kickoff_v725,
    write_possession_ecology_telemetry_v725,
)

_BASE_V724_FINGERPRINT = None
_BASE_LOAD_CHAOS = None

PASS_MATCHUP_AUTHORITY_V725 = 0.70
RUN_MATCHUP_AUTHORITY_V725 = 0.72


@dataclass(frozen=True)
class RuntimeFingerprintV725:
    version: str
    base_runtime_hash: str
    drive_conversion_observer_definition_repaired: bool
    empirical_field_goal_ecology_active: bool
    empirical_kickoff_geometry_active: bool
    empirical_fumble_return_geometry_active: bool
    punt_ecology_changed: bool
    pass_matchup_authority: float
    run_matchup_authority: float
    v724_collapse_sampler_preserved: bool
    scoring_event_authority_preserved: bool
    market_inputs_to_football: bool
    direct_score_adjustment: bool
    runtime_hash: str


def _load_chaos_ecology_v725():
    if _BASE_LOAD_CHAOS is None:
        raise RuntimeError("V7.2.5 base chaos loader is not configured")
    ecology = _BASE_LOAD_CHAOS()
    player_usage = v61.runner._argument_path(
        "--player-usage", "artifacts/league-policy/player_usage.parquet"
    )
    path = player_usage.parent / "chaos_ecology.parquet"
    row = None
    if path.exists():
        rows = pl.read_parquet(path).to_dicts()
        row = rows[0] if rows else None
    configure_possession_ecology_v725(row)
    return ecology


def compose_v725_runtime() -> None:
    global _BASE_V724_FINGERPRINT, _BASE_LOAD_CHAOS

    v724.compose_v724_runtime()
    _BASE_V724_FINGERPRINT = v724.assert_v724_runtime()
    reset_possession_ecology_v725()

    if v61.runner._load_chaos_ecology is not _load_chaos_ecology_v725:
        _BASE_LOAD_CHAOS = v61.runner._load_chaos_ecology
    v61.runner._load_chaos_ecology = _load_chaos_ecology_v725

    # Preserve V7.2's current kicker/returner identity wrapper and replace only
    # the historical football ecology beneath that wrapper.
    special_teams_identity_v72._BASE_FIELD_GOAL = simulate_field_goal_v725
    special_teams_identity_v72._BASE_KICKOFF = simulate_kickoff_v725
    game_loop_v13.resolve_turnover_return = resolve_turnover_return_v725

    # Historical play-outcome profiles remain the marginal prior. Current 11v11
    # matchup evidence still reorders worlds, but with less authority to distort
    # league-wide negative-play/drive-survival anatomy.
    resolution_ecology._SNAP_MATCHUP_AUTHORITY = PASS_MATCHUP_AUTHORITY_V725
    resolution_ecology._SNAP_RUN_MATCHUP_AUTHORITY = RUN_MATCHUP_AUTHORITY_V725


def runtime_fingerprint_v725() -> RuntimeFingerprintV725:
    base = _BASE_V724_FINGERPRINT or v724.runtime_fingerprint_v724()
    payload = {
        "version": "v7.2.5-possession-ecology",
        "base_runtime_hash": base.runtime_hash,
        "drive_conversion_observer_definition_repaired": True,
        "empirical_field_goal_ecology_active": True,
        "empirical_kickoff_geometry_active": True,
        "empirical_fumble_return_geometry_active": True,
        "punt_ecology_changed": False,
        "pass_matchup_authority": float(resolution_ecology._SNAP_MATCHUP_AUTHORITY),
        "run_matchup_authority": float(resolution_ecology._SNAP_RUN_MATCHUP_AUTHORITY),
        "v724_collapse_sampler_preserved": True,
        "scoring_event_authority_preserved": True,
        "market_inputs_to_football": False,
        "direct_score_adjustment": False,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return RuntimeFingerprintV725(**payload, runtime_hash=digest)


def assert_v725_runtime() -> RuntimeFingerprintV725:
    errors: list[str] = []
    if v61.runner._load_chaos_ecology is not _load_chaos_ecology_v725:
        errors.append("V7.2.5 policy ecology loader is not active")
    if special_teams_identity_v72._BASE_FIELD_GOAL is not simulate_field_goal_v725:
        errors.append("V7.2.5 empirical field-goal ecology is not beneath V7.2 identity")
    if special_teams_identity_v72._BASE_KICKOFF is not simulate_kickoff_v725:
        errors.append("V7.2.5 empirical kickoff geometry is not beneath V7.2 identity")
    if game_loop_v13.resolve_turnover_return is not resolve_turnover_return_v725:
        errors.append("V7.2.5 fumble-return geometry is not active")
    if abs(resolution_ecology._SNAP_MATCHUP_AUTHORITY - PASS_MATCHUP_AUTHORITY_V725) > 1e-12:
        errors.append("V7.2.5 pass matchup authority is not active")
    if abs(resolution_ecology._SNAP_RUN_MATCHUP_AUTHORITY - RUN_MATCHUP_AUTHORITY_V725) > 1e-12:
        errors.append("V7.2.5 run matchup authority is not active")
    if errors:
        raise RuntimeError("v7.2.5 runtime integrity failure: " + "; ".join(errors))
    return runtime_fingerprint_v725()


def write_runtime_fingerprint_v725(out: Path) -> RuntimeFingerprintV725:
    fingerprint = assert_v725_runtime()
    out.mkdir(parents=True, exist_ok=True)
    (out / "runtime_fingerprint_v725.json").write_text(
        json.dumps(asdict(fingerprint), indent=2) + "\n",
        encoding="utf-8",
    )
    write_possession_ecology_telemetry_v725(out)
    return fingerprint
