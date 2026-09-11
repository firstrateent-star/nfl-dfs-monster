from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from monster.snapshot.player import PlayerState, TeamPlayerPool


@dataclass(frozen=True)
class AvailabilityWorld:
    """One sampled game-day personnel state for a team.

    Availability decides whether a player exists in this world. Conditional effectiveness
    remains attached to the player and may affect only worlds in which that player is active.
    This module is intentionally independent of scoring and market information.
    """

    team_id: str
    active_player_ids: tuple[str, ...]
    inactive_player_ids: tuple[str, ...]
    starting_qb_id: str
    pool: TeamPlayerPool


def _normalize(values: list[float]) -> list[float]:
    arr = np.clip(np.asarray(values, dtype=float), 0.0, None)
    total = float(arr.sum())
    if total <= 0.0:
        return [0.0 for _ in values]
    return (arr / total).tolist()


def _redistribute(players: list[PlayerState], field: str, *, eligible_positions: set[str]) -> list[float]:
    raw = [
        float(getattr(player, field)) if player.position.upper() in eligible_positions else 0.0
        for player in players
    ]
    return _normalize(raw)


def availability_world_from_active_ids(
    pool: TeamPlayerPool,
    *,
    active_player_ids: set[str] | frozenset[str] | tuple[str, ...],
) -> AvailabilityWorld:
    """Derive the skill opportunity world from an already-sampled full-roster active set.

    This is the coherence bridge between all-player unit availability and skill opportunity.
    It performs no new Bernoulli draw: the same active IDs that rebuild OL/defense also decide
    who can receive targets/carries. At least one active QB must already exist in the supplied
    roster state; otherwise the caller has produced an invalid full-roster world.
    """

    if not pool.players:
        raise ValueError("cannot derive availability for an empty team pool")

    active_set = set(active_player_ids)
    active_players = [player for player in pool.players if player.player_id in active_set]
    if not active_players:
        raise ValueError(f"{pool.team_id} full-roster world has no active skill players")

    active_qbs = [player for player in active_players if player.position.upper() == "QB"]
    if not active_qbs:
        raise ValueError(f"{pool.team_id} full-roster world has no active quarterback")
    starter = max(
        active_qbs,
        key=lambda player: (float(player.qb_pass_share), float(player.active_probability)),
    )

    target_share = _redistribute(active_players, "target_share", eligible_positions={"RB", "WR", "TE"})
    rush_share = _redistribute(active_players, "rush_share", eligible_positions={"QB", "RB", "WR", "TE"})
    rz_target_share = _redistribute(
        active_players, "red_zone_target_share", eligible_positions={"RB", "WR", "TE"}
    )
    rz_rush_share = _redistribute(
        active_players, "red_zone_rush_share", eligible_positions={"QB", "RB", "WR", "TE"}
    )
    rec_td_share = _redistribute(
        active_players, "receiving_td_share", eligible_positions={"RB", "WR", "TE"}
    )
    rush_td_share = _redistribute(
        active_players, "rushing_td_share", eligible_positions={"QB", "RB", "WR", "TE"}
    )

    qb_weights = [
        float(player.qb_pass_share) if player.position.upper() == "QB" else 0.0
        for player in active_players
    ]
    if sum(qb_weights) <= 0.0:
        qb_weights = [1.0 if player.player_id == starter.player_id else 0.0 for player in active_players]
    qb_share = _normalize(qb_weights)

    sampled = []
    for idx, player in enumerate(active_players):
        sampled.append(
            replace(
                player,
                target_share=target_share[idx],
                rush_share=rush_share[idx],
                red_zone_target_share=rz_target_share[idx],
                red_zone_rush_share=rz_rush_share[idx],
                receiving_td_share=rec_td_share[idx],
                rushing_td_share=rush_td_share[idx],
                qb_pass_share=qb_share[idx],
                active_probability=1.0,
            )
        )

    active_ids = tuple(player.player_id for player in sampled)
    inactive_ids = tuple(
        player.player_id for player in pool.players if player.player_id not in active_set
    )
    return AvailabilityWorld(
        team_id=pool.team_id,
        active_player_ids=active_ids,
        inactive_player_ids=inactive_ids,
        starting_qb_id=starter.player_id,
        pool=replace(pool, players=tuple(sampled)),
    )


def sample_team_availability_world(
    pool: TeamPlayerPool,
    *,
    rng: np.random.Generator,
) -> AvailabilityWorld:
    """Sample active skill personnel once per game world and conserve role-share mass.

    This legacy skill-only entry point remains available for isolated experiments. Integrated
    v1.3 worlds should prefer the full-roster unit sampler and then call
    ``availability_world_from_active_ids`` so OL, defense and skill usage share one reality.
    """
    if not pool.players:
        raise ValueError("cannot sample availability for an empty team pool")

    players = list(pool.players)
    active_mask = np.asarray(
        [rng.random() < float(np.clip(player.active_probability, 0.0, 1.0)) for player in players],
        dtype=bool,
    )

    qb_indices = [idx for idx, player in enumerate(players) if player.position.upper() == "QB"]
    if not qb_indices:
        raise ValueError(f"{pool.team_id} has no quarterback candidates")
    if not any(active_mask[idx] for idx in qb_indices):
        fallback = max(
            qb_indices,
            key=lambda idx: (
                float(players[idx].qb_pass_share),
                float(players[idx].active_probability),
            ),
        )
        active_mask[fallback] = True

    active_ids = {
        player.player_id for idx, player in enumerate(players) if active_mask[idx]
    }
    return availability_world_from_active_ids(pool, active_player_ids=active_ids)
