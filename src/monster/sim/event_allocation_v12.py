from __future__ import annotations

import numpy as np

from monster.sim.allocation import GameAllocationWorlds, _allocate_team
from monster.sim.event_coupling import couple_touchdowns_to_events
from monster.sim.game import GameWorlds
from monster.snapshot.player import TeamPlayerPool


def allocate_event_game_players(
    game_worlds: GameWorlds,
    away_pool: TeamPlayerPool,
    home_pool: TeamPlayerPool,
    *,
    seed: int,
) -> GameAllocationWorlds:
    """Allocate players from the exact finite play/scoring supply created by v1.2.

    v1.2 does not resample team play volume or touchdown type downstream. The game engine owns
    those facts because they emerged from possessions and scrimmage events. Player allocation
    only resolves who participated in the already-simulated football.
    """
    required = (
        game_worlds.away_plays,
        game_worlds.home_plays,
        game_worlds.away_passing_touchdowns,
        game_worlds.home_passing_touchdowns,
        game_worlds.away_rushing_touchdowns,
        game_worlds.home_rushing_touchdowns,
    )
    if any(value is None for value in required):
        raise ValueError("event-first allocation requires exact play and typed touchdown supply")

    rng = np.random.default_rng(seed)
    worlds = len(game_worlds.away_points)
    away_disruption = (
        game_worlds.away_pass_disruption
        if game_worlds.away_pass_disruption is not None
        else np.ones(worlds, dtype=np.float32)
    )
    home_disruption = (
        game_worlds.home_pass_disruption
        if game_worlds.home_pass_disruption is not None
        else np.ones(worlds, dtype=np.float32)
    )
    away_run_eff = (
        game_worlds.away_run_efficiency
        if game_worlds.away_run_efficiency is not None
        else np.ones(worlds, dtype=np.float32)
    )
    home_run_eff = (
        game_worlds.home_run_efficiency
        if game_worlds.home_run_efficiency is not None
        else np.ones(worlds, dtype=np.float32)
    )

    away = _allocate_team(
        rng,
        away_pool,
        game_worlds.away_points,
        game_worlds.home_points,
        game_worlds.away_plays,
        game_worlds.away_touchdowns,
        away_disruption,
        away_run_eff,
    )
    home = _allocate_team(
        rng,
        home_pool,
        game_worlds.home_points,
        game_worlds.away_points,
        game_worlds.home_plays,
        game_worlds.home_touchdowns,
        home_disruption,
        home_run_eff,
    )

    # The event engine, not the allocator, owns whether each touchdown was a pass or rush.
    away.passing_tds[:] = game_worlds.away_passing_touchdowns
    away.rushing_tds[:] = game_worlds.away_rushing_touchdowns
    home.passing_tds[:] = game_worlds.home_passing_touchdowns
    home.rushing_tds[:] = game_worlds.home_rushing_touchdowns

    allocation = GameAllocationWorlds(away=away, home=home)
    return couple_touchdowns_to_events(
        allocation,
        away_pool,
        home_pool,
        seed=seed + 4_000_037,
    )
