from __future__ import annotations

import numpy as np

from monster.feature_compile.units import UnitPlayerInputs
from monster.sim.unit_availability_world import sample_unit_availability_world


def test_inactive_unit_player_is_removed_and_active_exposure_reconserved() -> None:
    players = (
        UnitPlayerInputs(
            player_id="lt-out",
            position="LT",
            offense_snap_share=0.95,
            active_probability=0.0,
            madden_pass_block=95.0,
        ),
        UnitPlayerInputs(
            player_id="lt-backup",
            position="LT",
            offense_snap_share=0.55,
            active_probability=1.0,
            madden_pass_block=70.0,
        ),
        UnitPlayerInputs(
            player_id="rg",
            position="RG",
            offense_snap_share=0.90,
            active_probability=1.0,
            madden_pass_block=82.0,
        ),
    )
    world = sample_unit_availability_world(players, rng=np.random.default_rng(1))
    assert "lt-out" in world.inactive_player_ids
    assert "lt-backup" in world.active_player_ids
    assert all(player.active_probability == 1.0 for player in world.players)
    assert world.offense_snap_equivalents > 0.0


def test_unit_world_preserves_effectiveness_if_active() -> None:
    players = (
        UnitPlayerInputs(
            player_id="cb",
            position="CB",
            defense_snap_share=0.95,
            active_probability=1.0,
            effectiveness_if_active=0.72,
            madden_coverage=92.0,
        ),
    )
    world = sample_unit_availability_world(players, rng=np.random.default_rng(2))
    assert world.players[0].effectiveness_if_active == 0.72
    assert world.players[0].active_probability == 1.0


def test_all_missed_unit_candidates_receive_deterministic_fallback() -> None:
    players = (
        UnitPlayerInputs(
            player_id="edge1",
            position="EDGE",
            defense_snap_share=0.80,
            active_probability=0.0,
        ),
        UnitPlayerInputs(
            player_id="edge2",
            position="EDGE",
            defense_snap_share=0.60,
            active_probability=0.0,
        ),
    )
    world = sample_unit_availability_world(players, rng=np.random.default_rng(3))
    assert len(world.active_player_ids) == 1
    assert world.active_player_ids[0] == "edge1"
