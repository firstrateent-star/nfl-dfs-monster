from __future__ import annotations

from monster.reality.world_availability_v722 import (
    active_this_world_v722,
    begin_world_availability_v722,
    materialize_world_pools_v722,
)
from monster.snapshot.player import PlayerState, TeamPlayerPool


def test_world_availability_is_binary_and_stable_within_world():
    begin_world_availability_v722(seed=12345, game="AAA@BBB")
    first = active_this_world_v722("player-x", 0.58)
    second = active_this_world_v722("player-x", 0.58)
    assert first in {True, False}
    assert second is first


def test_world_pool_zeroes_inactive_and_renormalizes_active_roles():
    players = (
        PlayerState(
            "qb1", "QB1", "QB", "AAA",
            rush_share=0.10, qb_pass_share=1.0, active_probability=1.0,
        ),
        PlayerState(
            "wr1", "WR1", "WR", "AAA",
            target_share=0.60, receiving_td_share=0.60,
            red_zone_target_share=0.60, active_probability=1.0,
        ),
        PlayerState(
            "wr2", "WR2", "WR", "AAA",
            target_share=0.40, receiving_td_share=0.40,
            red_zone_target_share=0.40, active_probability=0.0,
        ),
        PlayerState(
            "rb1", "RB1", "RB", "AAA",
            target_share=0.20, rush_share=0.90,
            red_zone_target_share=0.20, red_zone_rush_share=1.0,
            receiving_td_share=0.20, rushing_td_share=1.0,
            active_probability=1.0,
        ),
    )
    pool = TeamPlayerPool(team_id="AAA", players=players)
    world = materialize_world_pools_v722(
        {"AAA": pool}, seed=999, game="AAA@BBB"
    )["AAA"]
    by_id = {player.player_id: player for player in world.players}

    assert by_id["wr2"].active_probability == 0.0
    assert by_id["wr2"].target_share == 0.0
    assert abs(
        sum(
            p.target_share
            for p in world.players
            if p.position in {"RB", "WR", "TE"}
        )
        - 1.0
    ) < 1e-9
    assert abs(sum(p.rush_share for p in world.players) - 1.0) < 1e-9
