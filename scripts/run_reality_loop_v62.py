from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import run_reality_loop_v2_smoke as v61

from monster.sim import intent_ecology, play_kernel, resolution_ecology
from monster.sim.reality_v62 import (
    RoleWorldPlan,
    apply_chaos_environment,
    apply_game_environment,
    apply_role_world,
    choose_rusher_role_authoritative,
    choose_target_role_authoritative,
    complete_team_receivers,
    filter_roster_truth,
    sample_game_environment,
    sample_role_world_plan,
)

_NATIVE_CONFIGURE_V61 = v61.configure_reality_loop_v2
_NATIVE_LOST_FUMBLE = play_kernel._lost_fumble_probability


def _depth_throw_probabilities_v62(*args, **kwargs):
    """Rebalance turnover composition: less INT forcing, more live-ball fumble opportunity."""

    result = v61._NATIVE_DEPTH_THROW_PROBABILITIES(*args, **kwargs)
    return replace(
        result,
        interception=float(np.clip(result.interception * 1.05, 0.001, 0.12)),
    )


def _lost_fumble_probability_v62(*args, **kwargs):
    probability = _NATIVE_LOST_FUMBLE(*args, **kwargs)
    return float(np.clip(probability * 1.25, 0.001, 0.05))


def configure_reality_loop_v62() -> None:
    """Layer v6.2 role reality and world-state uncertainty over the validated v6.1 engine."""

    _NATIVE_CONFIGURE_V61()

    integrated = v61.runner.integrated
    native_team_identity = v61.runner.enhanced_team_identity
    native_compile_pools = integrated.compile_current_skill_pools
    native_apply_rush_plan = integrated._with_event_rush_plan
    native_simulate_game = integrated.simulate_game

    def compile_current_skill_pools_v62(personnel, historical_usage, **kwargs):
        filtered = filter_roster_truth(personnel)
        return native_compile_pools(filtered, historical_usage, **kwargs)

    def enhanced_team_identity_v62(*args, **kwargs):
        team = native_team_identity(*args, **kwargs)
        pool = args[1] if len(args) > 1 else kwargs["pool"]
        reality = args[2] if len(args) > 2 else kwargs["reality"]
        unit_players = args[3] if len(args) > 3 else kwargs.get("unit_players", ())
        return complete_team_receivers(team, pool, reality, unit_players)

    def sample_role_world(pool, *, rng, **_kwargs):
        return sample_role_world_plan(pool, rng=rng)

    def apply_role_plan(team, plan):
        base = native_apply_rush_plan(team, plan)
        if isinstance(plan, RoleWorldPlan):
            return apply_role_world(base, plan)
        return base

    def choose_target(players, ecology, category, rng, *, shrinkage_samples=45.0):
        selected = choose_target_role_authoritative(
            players,
            ecology,
            category,
            rng,
            shrinkage_samples=shrinkage_samples,
        )
        v61._OPPORTUNITY_TAIL.register_pass(selected, players)
        return selected

    def choose_rusher(players, ecology, category, rng, *, shrinkage_samples=60.0):
        selected = choose_rusher_role_authoritative(
            players,
            ecology,
            category,
            rng,
            shrinkage_samples=shrinkage_samples,
        )
        v61._OPPORTUNITY_TAIL.register_run(selected, players)
        return selected

    def simulate_game_v62(*args, **kwargs):
        if len(args) >= 2:
            away, home = args[0], args[1]
            remaining = args[2:]
        else:
            away = kwargs.pop("away")
            home = kwargs.pop("home")
            remaining = ()
        seed = int(kwargs.get("seed", 1))
        environment = sample_game_environment(seed)
        away, home = apply_game_environment(away, home, environment)
        kwargs["penalty_rate"] = environment.penalty_rate
        if "chaos_ecology" in kwargs:
            kwargs["chaos_ecology"] = apply_chaos_environment(
                kwargs["chaos_ecology"], environment
            )
        return native_simulate_game(away, home, *remaining, **kwargs)

    integrated.compile_current_skill_pools = compile_current_skill_pools_v62
    v61.runner.enhanced_team_identity = enhanced_team_identity_v62
    integrated.sample_event_rush_share_plan = sample_role_world
    integrated._with_event_rush_plan = apply_role_plan
    integrated.simulate_game = simulate_game_v62

    intent_ecology.choose_target_for_depth = choose_target
    intent_ecology.choose_rusher_for_geometry = choose_rusher

    resolution_ecology.depth_throw_probabilities = _depth_throw_probabilities_v62
    play_kernel._lost_fumble_probability = _lost_fumble_probability_v62


def _record_v62_manifest() -> None:
    out = v61._first_out_path()
    manifest_path = out / "manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text())
    manifest.update(
        {
            "reality_loop_v62_active": True,
            "kickoff_roster_truth_filter_active": True,
            "role_world_sampled_before_game_world": True,
            "current_role_authority_over_historical_actor_volume": True,
            "historical_depth_geometry_used_as_compatibility_only": True,
            "zero_history_current_receiver_candidate_reserve_active": True,
            "persistent_game_environment_world_active": True,
            "persistent_penalty_environment_active": True,
            "persistent_chaos_tail_environment_active": True,
            "v62_interception_multiplier": 1.05,
            "v62_lost_fumble_multiplier": 1.25,
            "v62_principle": (
                "Current roster and sampled current role decide who participates. Historical "
                "intent conditions how a current participant is used; it cannot restore an old "
                "team's opportunity hierarchy. Game-tail variance remains football-causal."
            ),
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    original = v61.configure_reality_loop_v2
    try:
        v61.configure_reality_loop_v2 = configure_reality_loop_v62
        v61.main()
    finally:
        v61.configure_reality_loop_v2 = original
    _record_v62_manifest()


if __name__ == "__main__":
    main()
