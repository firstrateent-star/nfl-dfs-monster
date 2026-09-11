from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from monster.feature_compile.units import UnitPlayerInputs


@dataclass(frozen=True)
class UnitAvailabilityWorld:
    """One sampled all-player unit state for a team game world."""

    active_player_ids: tuple[str, ...]
    inactive_player_ids: tuple[str, ...]
    players: tuple[UnitPlayerInputs, ...]
    offense_snap_equivalents: float
    defense_snap_equivalents: float
    special_teams_snap_equivalents: float


def _conserve(values: list[float], target: float = 11.0) -> list[float]:
    arr = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    total = float(arr.sum())
    if total <= 0.0:
        return [0.0 for _ in values]
    return np.clip(arr * (target / total), 0.0, 0.995).tolist()


def _ensure_unit_presence(
    mask: np.ndarray,
    players: tuple[UnitPlayerInputs, ...],
    field: str,
) -> None:
    candidates = [idx for idx, player in enumerate(players) if float(getattr(player, field)) > 0.0]
    if not candidates or any(mask[idx] for idx in candidates):
        return
    fallback = max(
        candidates,
        key=lambda idx: (
            float(players[idx].active_probability),
            float(getattr(players[idx], field)),
        ),
    )
    mask[fallback] = True


def _ensure_position_presence(
    mask: np.ndarray,
    players: tuple[UnitPlayerInputs, ...],
    *,
    positions: set[str],
    share_field: str,
) -> None:
    """Guarantee a required football position without sampling a second roster reality."""

    candidates = [
        idx
        for idx, player in enumerate(players)
        if player.position.upper() in positions and float(getattr(player, share_field)) > 0.0
    ]
    if not candidates or any(mask[idx] for idx in candidates):
        return
    fallback = max(
        candidates,
        key=lambda idx: (
            float(players[idx].active_probability),
            float(getattr(players[idx], share_field)),
        ),
    )
    mask[fallback] = True


def sample_unit_availability_world(
    players: tuple[UnitPlayerInputs, ...],
    *,
    rng: np.random.Generator,
) -> UnitAvailabilityWorld:
    """Sample one active roster and rebuild conditional 11-man unit exposure.

    Input players must use *conditional* snap shares: how much the player would participate
    if active. Availability is sampled once for the game world, inactive players are removed,
    active_probability becomes 1.0, and offense/defense/special-teams exposure is reconserved
    among the surviving players. Effectiveness-if-active remains attached to the active player.

    The same sampled active-player set is intended to drive skill opportunity, OL, defense and
    special teams. A quarterback is therefore guaranteed inside this full-roster draw rather
    than by a separate skill-only sampler, preventing contradictory personnel worlds.
    """

    if not players:
        return UnitAvailabilityWorld((), (), (), 0.0, 0.0, 0.0)

    active = np.asarray(
        [rng.random() < float(np.clip(player.active_probability, 0.0, 1.0)) for player in players],
        dtype=bool,
    )
    _ensure_position_presence(
        active,
        players,
        positions={"QB"},
        share_field="offense_snap_share",
    )
    for field in (
        "offense_snap_share",
        "defense_snap_share",
        "special_teams_snap_share",
    ):
        _ensure_unit_presence(active, players, field)

    survivors = [player for idx, player in enumerate(players) if active[idx]]
    offense = _conserve([player.offense_snap_share for player in survivors])
    defense = _conserve([player.defense_snap_share for player in survivors])
    special = _conserve([player.special_teams_snap_share for player in survivors])

    sampled = tuple(
        replace(
            player,
            offense_snap_share=offense[idx],
            defense_snap_share=defense[idx],
            special_teams_snap_share=special[idx],
            active_probability=1.0,
        )
        for idx, player in enumerate(survivors)
    )
    active_ids = tuple(player.player_id for player in sampled)
    inactive_ids = tuple(
        player.player_id for idx, player in enumerate(players) if not active[idx]
    )
    return UnitAvailabilityWorld(
        active_player_ids=active_ids,
        inactive_player_ids=inactive_ids,
        players=sampled,
        offense_snap_equivalents=float(sum(player.offense_snap_share for player in sampled)),
        defense_snap_equivalents=float(sum(player.defense_snap_share for player in sampled)),
        special_teams_snap_equivalents=float(
            sum(player.special_teams_snap_share for player in sampled)
        ),
    )
