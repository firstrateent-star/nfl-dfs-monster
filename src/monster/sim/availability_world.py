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


def sample_team_availability_world(
    pool: TeamPlayerPool,
    *,
    rng: np.random.Generator,
) -> AvailabilityWorld:
    """Sample active personnel once per game world and conserve role-share mass.

    Rules:
    - availability is sampled once for the game, not independently by play;
    - at least one QB must be active, using the highest qb_pass_share/current availability as
      a deterministic fallback if all Bernoulli samples miss;
    - inactive players are removed from every offensive opportunity tree;
    - target/rush/red-zone/TD shares are renormalized only among active eligible players;
    - effectiveness_if_active is preserved and is not multiplied into availability.
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

    active_players = [player for idx, player in enumerate(players) if active_mask[idx]]
    active_qbs = [player for player in active_players if player.position.upper() == "QB"]
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
        player.player_id for idx, player in enumerate(players) if not active_mask[idx]
    )
    sampled_pool = replace(pool, players=tuple(sampled))
    return AvailabilityWorld(
        team_id=pool.team_id,
        active_player_ids=active_ids,
        inactive_player_ids=inactive_ids,
        starting_qb_id=starter.player_id,
        pool=sampled_pool,
    )
