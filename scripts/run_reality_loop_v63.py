from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
import run_reality_loop_v2_smoke as v61
import run_reality_loop_v62 as v62

from monster.sim import (
    chaos_ecology,
    game_loop_v13,
    play_kernel,
    progressive_skill_tail_v2,
    reality_snap_v5,
    special_teams_v13,
)
from monster.sim.reality_v63 import (
    apply_game_environment_v63,
    cadence_seconds_v63,
    explosive_environment_multiplier,
    role_aware_offense_skill_players,
    sample_game_environment_v63,
)
from monster.sim.return_tail_v631 import (
    apply_chaos_environment_v631,
    resolve_turnover_return_v631,
    sample_return_yards_v631,
)

_NATIVE_SAMPLE_ROLE_WORLD = v62.sample_role_world_plan


class TargetRolePlanTelemetryV63:
    """Aggregate the sampled receiving role world before snap/game randomness."""

    def __init__(self) -> None:
        self._values: dict[tuple[str, str], list[float]] = defaultdict(list)
        self._meta: dict[tuple[str, str], tuple[str, str, float, float, float]] = {}

    def register(self, pool, plan) -> None:
        target_plan = getattr(plan, "target_plan", {})
        for player in pool.players:
            key = (str(pool.team_id), str(player.player_id))
            self._values[key].append(float(target_plan.get(player.player_id, 0.0)))
            self._meta[key] = (
                str(player.display_name),
                str(player.position),
                float(player.target_share),
                float(player.active_probability),
                float(player.role_uncertainty),
            )

    def write(self, out: Path) -> None:
        rows = []
        for (team, player_id), values in self._values.items():
            arr = np.asarray(values, dtype=float)
            player, position, base_share, active_probability, uncertainty = self._meta[
                (team, player_id)
            ]
            rows.append(
                {
                    "team": team,
                    "player_id": player_id,
                    "player": player,
                    "position": position,
                    "base_target_share": base_share,
                    "active_probability": active_probability,
                    "role_uncertainty": uncertainty,
                    "target_plan_share_mean": float(arr.mean()),
                    "target_plan_share_p50": float(np.quantile(arr, 0.50)),
                    "target_plan_share_p90": float(np.quantile(arr, 0.90)),
                    "target_plan_participation_probability": float(np.mean(arr > 0.0)),
                }
            )
        if not rows:
            return
        out.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(rows).sort(
            ["team", "target_plan_share_mean"], descending=[False, True]
        ).write_csv(out / "target_role_plan_audit_v63.csv")


_TARGET_ROLE_TELEMETRY = TargetRolePlanTelemetryV63()


def _sample_role_world_v63(pool, *, rng, **kwargs):
    plan = _NATIVE_SAMPLE_ROLE_WORLD(pool, rng=rng, **kwargs)
    _TARGET_ROLE_TELEMETRY.register(pool, plan)
    return plan


def configure_reality_loop_v63() -> None:
    """Activate v6.3 role/snap authority, historical return geometry and persistent tails."""

    v62.sample_game_environment = sample_game_environment_v63
    v62.apply_game_environment = apply_game_environment_v63
    v62.apply_chaos_environment = apply_chaos_environment_v631
    v62.sample_role_world_plan = _sample_role_world_v63
    v62.configure_reality_loop_v62()

    reality_snap_v5._offense_skill_players = role_aware_offense_skill_players

    # Keep v6.2 as a clean control: the stable chaos sampler accepts the richer prior columns but
    # ignores them. Only the v6.3 shadow swaps in the empirical 20/40/60/80-yard survivor sampler.
    chaos_ecology.sample_return_yards = sample_return_yards_v631
    game_loop_v13.sample_return_yards = sample_return_yards_v631
    game_loop_v13.resolve_turnover_return = resolve_turnover_return_v631
    special_teams_v13.sample_return_yards = sample_return_yards_v631

    native_pass_interaction = progressive_skill_tail_v2._pass_interaction
    native_run_skill_edge = progressive_skill_tail_v2._run_skill_edge
    native_snap_cadence = play_kernel._sample_snap_cadence

    def pass_interaction_v63(receiver_explosiveness: float, coverage_strength: float) -> float:
        base = native_pass_interaction(receiver_explosiveness, coverage_strength)
        return float(
            np.clip(base * explosive_environment_multiplier(exponent=0.58), 0.52, 2.02)
        )

    def run_skill_edge_v63(*, explosiveness: float, runner_power: float, tackling: float) -> float:
        base = native_run_skill_edge(
            explosiveness=explosiveness,
            runner_power=runner_power,
            tackling=tackling,
        )
        return float(
            np.clip(base * explosive_environment_multiplier(exponent=0.62), 0.52, 2.02)
        )

    def snap_cadence_v63(*args, **kwargs) -> int:
        return cadence_seconds_v63(native_snap_cadence(*args, **kwargs))

    progressive_skill_tail_v2._pass_interaction = pass_interaction_v63
    progressive_skill_tail_v2._run_skill_edge = run_skill_edge_v63
    play_kernel._sample_snap_cadence = snap_cadence_v63


def _record_v63_manifest(out: Path) -> None:
    manifest_path = out / "manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text())
    manifest.update(
        {
            "reality_loop_v63_active": True,
            "role_to_snap_merge_v63_active": True,
            "dual_role_player_snap_weight_bug_fixed": True,
            "target_role_plan_telemetry_v63_active": True,
            "role_snap_opportunity_audit_v63_expected": True,
            "historical_return_survivor_bands_v631_active": True,
            "historical_return_survivor_thresholds_yards": [20, 40, 60, 80],
            "turnover_return_continuation_tail_v63_active": True,
            "field_leverage_return_geometry_v63_active": True,
            "return_touchdown_probability_sampled_directly": False,
            "persistent_explosive_environment_v63_active": True,
            "persistent_tempo_environment_v63_active": True,
            "direct_score_tail_sampling": False,
            "v63_principle": (
                "Current role controls snap presence before historical geometry can shape the "
                "specific opportunity. Return distance is sampled from historical 20/40/60/80+ "
                "survivor geometry, while the live field decides whether that distance scores. "
                "Game tails emerge from persistent execution, tempo, open-field and chaos states "
                "rather than score or fantasy targets."
            ),
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    original = v61.configure_reality_loop_v2
    try:
        v61.configure_reality_loop_v2 = configure_reality_loop_v63
        v61.main()
    finally:
        v61.configure_reality_loop_v2 = original
    out = v61._first_out_path()
    _TARGET_ROLE_TELEMETRY.write(out)
    _record_v63_manifest(out)


if __name__ == "__main__":
    main()