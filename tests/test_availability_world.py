from __future__ import annotations

import numpy as np
import pytest

from monster.sim.availability_world import (
    availability_world_from_active_ids,
    sample_team_availability_world,
)
from monster.snapshot.player import PlayerState, TeamPlayerPool


def _player(
    player_id: str,
    position: str,
    *,
    active_probability: float,
    target_share: float = 0.0,
    rush_share: float = 0.0,
    qb_pass_share: float = 0.0,
) -> PlayerState:
    return PlayerState(
        player_id=player_id,
        display_name=player_id,
        position=position,
        team_id="T",
        active_probability=active_probability,
        effectiveness_if_active=0.82,
        target_share=target_share,
        rush_share=rush_share,
        red_zone_target_share=target_share,
        red_zone_rush_share=rush_share,
        receiving_td_share=target_share,
        rushing_td_share=rush_share,
        qb_pass_share=qb_pass_share,
    )


def test_inactive_player_is_removed_and_target_mass_is_conserved() -> None:
    pool = TeamPlayerPool(
        team_id="T",
        players=(
            _player("qb1", "QB", active_probability=1.0, rush_share=0.1, qb_pass_share=1.0),
            _player("wr1", "WR", active_probability=1.0, target_share=0.7),
            _player("wr2", "WR", active_probability=0.0, target_share=0.3),
            _player("rb1", "RB", active_probability=1.0, target_share=0.1, rush_share=0.9),
        ),
    )
    world = sample_team_availability_world(pool, rng=np.random.default_rng(1))
    assert "wr2" in world.inactive_player_ids
    assert "wr2" not in world.active_player_ids
    assert abs(sum(player.target_share for player in world.pool.players) - 1.0) < 1e-12
    assert abs(sum(player.rush_share for player in world.pool.players) - 1.0) < 1e-12
    assert all(player.active_probability == 1.0 for player in world.pool.players)


def test_qb_fallback_guarantees_one_active_quarterback() -> None:
    pool = TeamPlayerPool(
        team_id="T",
        players=(
            _player("qb1", "QB", active_probability=0.0, qb_pass_share=0.9),
            _player("qb2", "QB", active_probability=0.0, qb_pass_share=0.1),
            _player("wr1", "WR", active_probability=1.0, target_share=1.0),
        ),
    )
    world = sample_team_availability_world(pool, rng=np.random.default_rng(2))
    assert world.starting_qb_id == "qb1"
    assert "qb1" in world.active_player_ids
    assert sum(player.qb_pass_share for player in world.pool.players) == 1.0


def test_effectiveness_is_preserved_conditional_on_active() -> None:
    pool = TeamPlayerPool(
        team_id="T",
        players=(
            _player("qb1", "QB", active_probability=1.0, qb_pass_share=1.0),
            _player("wr1", "WR", active_probability=1.0, target_share=1.0),
        ),
    )
    world = sample_team_availability_world(pool, rng=np.random.default_rng(3))
    assert all(player.effectiveness_if_active == 0.82 for player in world.pool.players)


def test_shared_active_ids_drive_skill_usage_without_resampling() -> None:
    pool = TeamPlayerPool(
        team_id="T",
        players=(
            _player("qb1", "QB", active_probability=0.2, qb_pass_share=1.0),
            _player("wr1", "WR", active_probability=0.2, target_share=0.7),
            _player("wr2", "WR", active_probability=0.9, target_share=0.3),
            _player("rb1", "RB", active_probability=0.4, rush_share=1.0),
        ),
    )
    world = availability_world_from_active_ids(
        pool,
        active_player_ids={"qb1", "wr2", "rb1"},
    )
    assert set(world.active_player_ids) == {"qb1", "wr2", "rb1"}
    assert "wr1" in world.inactive_player_ids
    assert sum(player.target_share for player in world.pool.players) == 1.0
    assert sum(player.rush_share for player in world.pool.players) == 1.0
    assert all(player.active_probability == 1.0 for player in world.pool.players)


def test_shared_active_ids_reject_world_without_quarterback() -> None:
    pool = TeamPlayerPool(
        team_id="T",
        players=(
            _player("qb1", "QB", active_probability=1.0, qb_pass_share=1.0),
            _player("wr1", "WR", active_probability=1.0, target_share=1.0),
        ),
    )
    with pytest.raises(ValueError, match="no active quarterback"):
        availability_world_from_active_ids(pool, active_player_ids={"wr1"})
