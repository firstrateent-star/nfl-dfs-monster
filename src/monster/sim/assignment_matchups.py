from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from monster.sim.matchup_kernel import DefensiveIdentity
from monster.sim.play_kernel import PlayerIdentity


@dataclass(frozen=True)
class CoverageAssignment:
    receiver_id: str
    defender_id: str | None
    receiver_share: float
    defender_exposure: float
    matchup_weight: float
    authority: float = 0.0


@dataclass(frozen=True)
class RushAssignment:
    rusher_id: str
    blocker_id: str | None
    rusher_exposure: float
    blocker_exposure: float
    matchup_weight: float
    authority: float = 0.0


def _weights(values: list[float]) -> np.ndarray:
    arr = np.clip(np.asarray(values, dtype=float), 0.0, None)
    total = float(arr.sum())
    if total <= 0.0:
        return np.full(len(arr), 1.0 / len(arr)) if len(arr) else np.asarray([], dtype=float)
    return arr / total


def build_shadow_coverage_assignments(
    receivers: tuple[PlayerIdentity, ...],
    defenders: tuple[DefensiveIdentity, ...],
) -> tuple[CoverageAssignment, ...]:
    """Build a zero-authority coverage topology from exposure, not a production matchup model.

    The topology guarantees that all active receiving opportunity has an addressable defensive
    counterpart where coverage personnel exist. It intentionally does not yet claim route-side,
    alignment, man/zone responsibility, help coverage, or motion knowledge.
    """
    if not receivers:
        return ()
    receiver_weights = _weights([player.usage_weight for player in receivers])
    if not defenders:
        return tuple(
            CoverageAssignment(
                receiver_id=receiver.player_id,
                defender_id=None,
                receiver_share=float(receiver_weights[idx]),
                defender_exposure=0.0,
                matchup_weight=float(receiver_weights[idx]),
            )
            for idx, receiver in enumerate(receivers)
        )

    defender_weights = _weights([defender.snap_weight for defender in defenders])
    assignments = []
    for ridx, receiver in enumerate(receivers):
        # Stable cyclic pairing spreads top receiving exposure across the available coverage
        # population without pretending we know actual alignment assignments yet.
        didx = ridx % len(defenders)
        assignments.append(
            CoverageAssignment(
                receiver_id=receiver.player_id,
                defender_id=defenders[didx].player_id,
                receiver_share=float(receiver_weights[ridx]),
                defender_exposure=float(defender_weights[didx]),
                matchup_weight=float(receiver_weights[ridx] * defender_weights[didx]),
            )
        )
    return tuple(assignments)


def build_shadow_rush_assignments(
    rushers: tuple[DefensiveIdentity, ...],
    blocker_ids: tuple[str, ...],
    blocker_exposures: tuple[float, ...],
) -> tuple[RushAssignment, ...]:
    """Build a zero-authority trench assignment topology for later OL↔rusher resolution."""
    if not rushers:
        return ()
    rush_weights = _weights([rusher.snap_weight for rusher in rushers])
    if not blocker_ids:
        return tuple(
            RushAssignment(
                rusher_id=rusher.player_id,
                blocker_id=None,
                rusher_exposure=float(rush_weights[idx]),
                blocker_exposure=0.0,
                matchup_weight=float(rush_weights[idx]),
            )
            for idx, rusher in enumerate(rushers)
        )
    if len(blocker_ids) != len(blocker_exposures):
        raise ValueError("blocker ids and exposures must be aligned")
    block_weights = _weights(list(blocker_exposures))
    return tuple(
        RushAssignment(
            rusher_id=rusher.player_id,
            blocker_id=blocker_ids[idx % len(blocker_ids)],
            rusher_exposure=float(rush_weights[idx]),
            blocker_exposure=float(block_weights[idx % len(blocker_ids)]),
            matchup_weight=float(rush_weights[idx] * block_weights[idx % len(blocker_ids)]),
        )
        for idx, rusher in enumerate(rushers)
    )
