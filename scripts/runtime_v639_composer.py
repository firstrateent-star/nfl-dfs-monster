from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import run_reality_loop_v2_smoke as v61
import run_reality_loop_v634 as v634
import run_week1_v13_root_cause_experiment as root
import runtime_v635_composer as v635_runtime

from monster.sim import game_loop_v13, play_kernel, reality_v62, resolution_ecology
from monster.sim.current_role_guard_v635 import sample_target_share_plan_v635
from monster.sim.current_role_guard_v639 import (
    install_current_role_guard_v639,
    sample_rush_share_plan_v639,
)


@dataclass(frozen=True)
class RuntimeFingerprintV639:
    version: str
    game_flow_policy_composer: str
    scrimmage_runtime: str
    team_identity_runtime: str
    pool_compiler_runtime: str
    target_role_sampler: str
    rush_role_sampler: str
    field_read_runtime: str
    live_pass_matchup_authority: float
    live_run_matchup_authority: float
    stale_coarse_matchup_authority_is_live: bool
    score_authority: str
    market_inputs_to_football: bool
    direct_fantasy_inputs_to_football: bool
    runtime_hash: str


def _callable_id(fn: Callable[..., Any]) -> str:
    return (
        f"{getattr(fn, '__module__', '<unknown>')}."
        f"{getattr(fn, '__qualname__', getattr(fn, '__name__', '<unknown>'))}"
    )


def _fingerprint_payload() -> dict[str, object]:
    integrated = v61.runner.integrated
    return {
        "version": "v6.3.9",
        "game_flow_policy_composer": _callable_id(
            integrated._attach_historical_intent_ecology
        ),
        "scrimmage_runtime": _callable_id(game_loop_v13.simulate_scrimmage_play),
        "team_identity_runtime": _callable_id(v61.runner.enhanced_team_identity),
        "pool_compiler_runtime": _callable_id(integrated.compile_current_skill_pools),
        "target_role_sampler": _callable_id(reality_v62.sample_target_share_plan),
        "rush_role_sampler": _callable_id(reality_v62.sample_event_rush_share_plan),
        "field_read_runtime": _callable_id(play_kernel._field_read_target),
        "live_pass_matchup_authority": float(
            resolution_ecology._SNAP_MATCHUP_AUTHORITY
        ),
        "live_run_matchup_authority": float(
            resolution_ecology._SNAP_RUN_MATCHUP_AUTHORITY
        ),
        "stale_coarse_matchup_authority_is_live": False,
        "score_authority": "monster.sim.game_loop_v13 event-derived football state",
        "market_inputs_to_football": False,
        "direct_fantasy_inputs_to_football": False,
    }


def runtime_fingerprint_v639() -> RuntimeFingerprintV639:
    payload = _fingerprint_payload()
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return RuntimeFingerprintV639(**payload, runtime_hash=digest)


def _runtime_integrity_errors(*, require_role_guard: bool) -> list[str]:
    integrated = v61.runner.integrated
    errors: list[str] = []

    if integrated._attach_historical_intent_ecology is not v61._attach_reality_loop_policy:
        errors.append("game-flow policy composer is not Reality Loop v2 production policy")
    if game_loop_v13.simulate_scrimmage_play is not v61._simulate_scrimmage_play_with_snap_world:
        errors.append("game_loop_v13 is not using the production per-snap participant world")
    if play_kernel._field_read_target is not v61._field_read_target_with_opportunity:
        errors.append("production opportunity-aware field read wrapper is not active")
    if not hasattr(resolution_ecology, "_SNAP_MATCHUP_AUTHORITY"):
        errors.append("live pass matchup authority is unavailable")
    if not hasattr(resolution_ecology, "_SNAP_RUN_MATCHUP_AUTHORITY"):
        errors.append("live run matchup authority is unavailable")

    if require_role_guard:
        if reality_v62.sample_target_share_plan is not sample_target_share_plan_v635:
            errors.append("v6.3.5 target role sampler changed inside v6.3.9")
        if reality_v62.sample_event_rush_share_plan is not sample_rush_share_plan_v639:
            errors.append("v6.3.9 rush role sampler is not active after personnel compile")
    return errors


def compose_v639_runtime() -> None:
    """Install frozen v6.3.6 identity plus calibrated v6.3.9 gadget recurrence."""

    v634.install_current_role_guard = install_current_role_guard_v639
    v634.apply_matchup_identity_authority_v634 = (
        v635_runtime._apply_identity_authority_v635
    )
    v634.configure_reality_loop_v634()

    v61.runner._PASS_MATCHUP_AUTHORITY_OVERRIDE = None

    errors = _runtime_integrity_errors(require_role_guard=False)
    if errors:
        raise RuntimeError(
            "v6.3.9 static runtime integrity failure: " + "; ".join(errors)
        )


def assert_v639_runtime() -> RuntimeFingerprintV639:
    errors = _runtime_integrity_errors(require_role_guard=True)
    if errors:
        raise RuntimeError("v6.3.9 runtime integrity failure: " + "; ".join(errors))
    return runtime_fingerprint_v639()


def assert_team_runtime_v639(teams: dict[str, object]) -> None:
    missing_flow = [
        team_id
        for team_id, team in teams.items()
        if getattr(team, "game_flow_policy", None) is None
    ]
    missing_intent = [
        team_id
        for team_id, team in teams.items()
        if getattr(team, "intent_ecology", None) is None
    ]
    if missing_flow or missing_intent:
        raise RuntimeError(
            "v6.3.9 constructed-team integrity failure: "
            f"missing_game_flow_policy={missing_flow}; missing_intent_ecology={missing_intent}"
        )


def build_week1_runtime_inputs_v639(
    *,
    policy_path: Path,
    personnel_path: Path,
    player_usage_path: Path,
    situation_context_path: Path,
):
    compose_v639_runtime()
    integrated = v61.runner.integrated

    policy = integrated._read(policy_path)
    personnel = integrated._read(personnel_path)
    usage = integrated._read(player_usage_path)
    situation_context = integrated._read(situation_context_path)
    league_neutral_pass_rate, situational_pass_rates = integrated._situational_context(
        situation_context
    )

    pools = integrated.apply_health_to_skill_pools(
        integrated.compile_current_skill_pools(personnel, usage, policy=policy),
        personnel,
    )
    reality = integrated.compile_player_reality_inputs(
        personnel,
        game_date=integrated.GAME_DATE,
    )
    units = integrated.compile_league_unit_player_map(personnel)

    states: dict[str, object] = {}
    for away, home in integrated.MATCHUPS:
        states.update(integrated.compile_team_state_map(policy, {away: home, home: away}))

    teams = {
        team: v61.runner.enhanced_team_identity(
            team,
            pools[team],
            reality,
            units[team],
            states[team],
            league_neutral_pass_rate=league_neutral_pass_rate,
            situational_pass_rates=situational_pass_rates,
        )
        for pair in integrated.MATCHUPS
        for team in pair
    }
    teams = integrated._attach_historical_intent_ecology(
        teams,
        player_usage_path.parent,
    )
    defenses = {
        team: v61.runner.enhanced_defensive_unit(units[team])
        for pair in integrated.MATCHUPS
        for team in pair
    }
    assert_team_runtime_v639(teams)
    fingerprint = assert_v639_runtime()
    ecology = root._load_chaos_ecology(player_usage_path)
    return {
        "fingerprint": fingerprint,
        "personnel": personnel,
        "pools": pools,
        "teams": teams,
        "defenses": defenses,
        "states": states,
        "units": units,
        "ecology": ecology,
        "league_neutral_pass_rate": league_neutral_pass_rate,
        "situational_pass_rates": situational_pass_rates,
    }


def write_runtime_fingerprint_v639(out: Path) -> RuntimeFingerprintV639:
    fingerprint = assert_v639_runtime()
    out.mkdir(parents=True, exist_ok=True)
    (out / "runtime_fingerprint_v639.json").write_text(
        json.dumps(asdict(fingerprint), indent=2) + "\n",
        encoding="utf-8",
    )
    return fingerprint
