from __future__ import annotations

from dataclasses import replace
from hashlib import blake2b

import numpy as np

from monster.snapshot.player import TeamPlayerPool

_WORLD_SEED: int | None = None
_WORLD_GAME: str | None = None
_WORLD_ACTIVE: dict[str, bool] = {}


def _draw(seed: int, player_id: str, probability: float) -> bool:
    p = float(np.clip(probability, 0.0, 1.0))
    if p <= 0.0:
        return False
    if p >= 1.0:
        return True
    digest = blake2b(
        f"v722-world-active:{seed}:{player_id}".encode("utf-8"),
        digest_size=8,
    ).digest()
    u = (int.from_bytes(digest, "big") + 0.5) / (2**64)
    return bool(u < p)


def begin_world_availability_v722(*, seed: int, game: str | None = None) -> None:
    global _WORLD_SEED, _WORLD_GAME, _WORLD_ACTIVE
    _WORLD_SEED = int(seed)
    _WORLD_GAME = game
    _WORLD_ACTIVE = {}


def clear_world_availability_v722() -> None:
    global _WORLD_SEED, _WORLD_GAME, _WORLD_ACTIVE
    _WORLD_SEED = None
    _WORLD_GAME = None
    _WORLD_ACTIVE = {}


def world_context_active_v722() -> bool:
    return _WORLD_SEED is not None


def active_this_world_v722(
    player_id: str,
    active_probability: float,
) -> bool | None:
    if _WORLD_SEED is None:
        return None
    key = str(player_id)
    if key not in _WORLD_ACTIVE:
        _WORLD_ACTIVE[key] = _draw(_WORLD_SEED, key, active_probability)
    return _WORLD_ACTIVE[key]


def _renormalized(
    players,
    active: list[bool],
    attr: str,
    *,
    positions: set[str] | None = None,
) -> list[float]:
    raw = []
    for player, is_active in zip(players, active):
        allowed = positions is None or player.position.upper() in positions
        raw.append(float(getattr(player, attr)) if is_active and allowed else 0.0)
    total = float(sum(raw))
    if total <= 0.0:
        return [0.0 for _ in raw]
    return [value / total for value in raw]


def materialize_world_pools_v722(
    pools: dict[str, TeamPlayerPool],
    *,
    seed: int,
    game: str,
) -> dict[str, TeamPlayerPool]:
    """Materialize one binary game-day availability state for the entire game world.

    Pregame uncertainty is sampled once per player per world. If active, the player keeps
    conditional football effectiveness and competes for a renormalized active-team role.
    If inactive, all skill opportunity shares become zero for that world.
    """

    begin_world_availability_v722(seed=seed, game=game)
    result: dict[str, TeamPlayerPool] = {}

    for team_id, pool in pools.items():
        players = list(pool.players)
        active = [
            bool(active_this_world_v722(player.player_id, player.active_probability))
            for player in players
        ]

        # An NFL offense must field a quarterback. If all QB availability draws fail,
        # promote the most likely listed QB rather than silently reusing an inactive QB1.
        qb_indices = [
            idx for idx, player in enumerate(players)
            if player.position.upper() == "QB"
        ]
        if qb_indices and not any(active[idx] for idx in qb_indices):
            winner = max(
                qb_indices,
                key=lambda idx: (
                    players[idx].active_probability,
                    players[idx].qb_pass_share,
                    -idx,
                ),
            )
            active[winner] = True
            _WORLD_ACTIVE[players[winner].player_id] = True

        changed = any(
            (player.active_probability < 0.999999) or not is_active
            for player, is_active in zip(players, active)
        )
        if not changed:
            result[team_id] = pool
            continue

        target = _renormalized(players, active, "target_share", positions={"RB", "WR", "TE"})
        rush = _renormalized(players, active, "rush_share", positions={"QB", "RB", "WR", "TE"})
        rz_target = _renormalized(
            players, active, "red_zone_target_share", positions={"RB", "WR", "TE"}
        )
        rz_rush = _renormalized(
            players, active, "red_zone_rush_share", positions={"QB", "RB", "WR", "TE"}
        )
        rec_td = _renormalized(
            players, active, "receiving_td_share", positions={"RB", "WR", "TE"}
        )
        rush_td = _renormalized(
            players, active, "rushing_td_share", positions={"QB", "RB", "WR", "TE"}
        )
        qb = _renormalized(players, active, "qb_pass_share", positions={"QB"})

        world_players = tuple(
            replace(
                player,
                target_share=target[idx],
                rush_share=rush[idx],
                red_zone_target_share=rz_target[idx],
                red_zone_rush_share=rz_rush[idx],
                receiving_td_share=rec_td[idx],
                rushing_td_share=rush_td[idx],
                qb_pass_share=qb[idx],
                active_probability=1.0 if active[idx] else 0.0,
                rush_role_probability=(
                    player.rush_role_probability if active[idx] else 0.0
                ),
            )
            for idx, player in enumerate(players)
        )
        result[team_id] = replace(pool, players=world_players)

    return result
