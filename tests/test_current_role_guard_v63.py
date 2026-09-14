from __future__ import annotations

import numpy as np
import polars as pl

from monster.sim.current_role_guard_v63 import (
    configure_current_receiving_roles,
    receiver_candidate_ids_current,
    sample_target_share_plan_current,
)
from monster.snapshot.player import PlayerState, TeamPlayerPool


def _personnel() -> pl.DataFrame:
    names = ["starter", *[f"hist{i}" for i in range(10)]]
    return pl.DataFrame(
        {
            "gsis_id": names,
            "display_name": names,
            "position": ["TE", *(["WR"] * 10)],
            "status": ["ACT"] * 11,
            "depth_rank": [1, *([2] * 10)],
            "conditional_offense_snap_share": [0.72, *([0.55] * 10)],
            "game_day_active_probability": [1.0] * 11,
        }
    )


def test_current_starter_survives_candidate_cutoff_despite_low_history():
    configure_current_receiving_roles(_personnel())
    players = [
        PlayerState("starter", "Current Starter", "TE", "T", target_share=0.001, role_uncertainty=0.25)
    ] + [
        PlayerState(f"hist{i}", f"History {i}", "WR", "T", target_share=0.099, role_uncertainty=0.10)
        for i in range(10)
    ]
    pool = TeamPlayerPool(team_id="T", players=tuple(players))
    assert "starter" in receiver_candidate_ids_current(pool, max_candidates=8)


def test_active_current_starter_is_present_in_every_role_world():
    configure_current_receiving_roles(_personnel())
    pool = TeamPlayerPool(
        team_id="T",
        players=(
            PlayerState("starter", "Current Starter", "TE", "T", target_share=0.01, role_uncertainty=0.25),
            *tuple(
                PlayerState(f"hist{i}", f"History {i}", "WR", "T", target_share=0.099, role_uncertainty=0.10)
                for i in range(10)
            ),
        ),
    )
    rng = np.random.default_rng(6304)
    shares = [sample_target_share_plan_current(pool, rng=rng).get("starter", 0.0) for _ in range(1000)]
    assert np.all(np.asarray(shares) > 0.0)
    assert float(np.mean(shares)) > 0.005


def test_history_still_controls_share_not_starter_flag():
    personnel = pl.DataFrame(
        {
            "gsis_id": ["starter", "alpha"],
            "display_name": ["Starter", "Alpha"],
            "position": ["WR", "WR"],
            "status": ["ACT", "ACT"],
            "depth_rank": [1, 2],
            "conditional_offense_snap_share": [0.70, 0.68],
            "game_day_active_probability": [1.0, 1.0],
        }
    )
    configure_current_receiving_roles(personnel)
    pool = TeamPlayerPool(
        team_id="T",
        players=(
            PlayerState("starter", "Starter", "WR", "T", target_share=0.08, role_uncertainty=0.10),
            PlayerState("alpha", "Alpha", "WR", "T", target_share=0.45, role_uncertainty=0.10),
        ),
    )
    rng = np.random.default_rng(6305)
    plans = [sample_target_share_plan_current(pool, rng=rng) for _ in range(1500)]
    starter_mean = float(np.mean([plan.get("starter", 0.0) for plan in plans]))
    alpha_mean = float(np.mean([plan.get("alpha", 0.0) for plan in plans]))
    assert starter_mean > 0.0
    assert alpha_mean > starter_mean
