from __future__ import annotations

import numpy as np

from monster.dfs.turnovers import attribute_team_turnovers
from monster.sim.allocation import TeamAllocationWorlds
from monster.snapshot.player import PlayerState, TeamPlayerPool


def _allocation() -> tuple[TeamAllocationWorlds, TeamPlayerPool]:
    worlds = 4
    players = (
        PlayerState("QB1", "QB One", "QB", "A", qb_pass_share=1.0),
        PlayerState("RB1", "RB One", "RB", "A"),
        PlayerState("WR1", "WR One", "WR", "A"),
    )
    stats = {}
    for player in players:
        stats[player.player_id] = {
            "pass_attempts": np.zeros(worlds, dtype=np.float32),
            "rush_attempts": np.zeros(worlds, dtype=np.float32),
            "receptions": np.zeros(worlds, dtype=np.float32),
        }
    stats["QB1"]["pass_attempts"][:] = [30, 35, 25, 40]
    stats["QB1"]["rush_attempts"][:] = [3, 4, 2, 5]
    stats["RB1"]["rush_attempts"][:] = [15, 20, 12, 18]
    stats["RB1"]["receptions"][:] = [3, 2, 4, 1]
    stats["WR1"]["receptions"][:] = [5, 7, 3, 8]
    zeros = np.zeros(worlds, dtype=np.float32)
    allocation = TeamAllocationWorlds(
        player_stats=stats,
        team_plays=zeros,
        team_dropbacks=zeros,
        team_sacks=zeros,
        team_pass_attempts=zeros,
        team_targets=zeros,
        team_rush_attempts=zeros,
        passing_tds=zeros,
        rushing_tds=zeros,
        pass_disruption=np.ones(worlds, dtype=np.float32),
        run_efficiency=np.ones(worlds, dtype=np.float32),
    )
    return allocation, TeamPlayerPool("A", players)


def test_turnover_reservoir_is_conserved_world_by_world() -> None:
    allocation, pool = _allocation()
    turnovers = np.array([0, 1, 2, 3], dtype=np.int16)
    result = attribute_team_turnovers(
        turnovers, allocation, pool, interception_fraction=0.65, seed=77
    )
    player_total = sum(result.interceptions.values()) + sum(result.fumbles_lost.values())
    np.testing.assert_array_equal(player_total, turnovers)
    np.testing.assert_array_equal(result.team_interceptions + result.team_fumbles_lost, turnovers)


def test_interceptions_are_qb_only() -> None:
    allocation, pool = _allocation()
    result = attribute_team_turnovers(
        np.array([2, 2, 2, 2], dtype=np.int16),
        allocation,
        pool,
        interception_fraction=1.0,
        seed=9,
    )
    assert result.interceptions["RB1"].sum() == 0
    assert result.interceptions["WR1"].sum() == 0
    assert result.interceptions["QB1"].sum() == 8
