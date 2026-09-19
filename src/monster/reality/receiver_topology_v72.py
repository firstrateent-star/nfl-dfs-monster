from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from monster.reality.opportunity import participation_assignment_weights
from monster.reality.participation_authority_v72 import (
    participation_multiplier,
    starter_authority,
)
from monster.sim import play_kernel
from monster.sim.matchup_kernel import resolve_pass_matchup
from monster.sim.reality_snap_v5 import snap_world
from monster.sim.rich_identity import qb_execution_skill

if TYPE_CHECKING:
    from monster.sim.matchup_kernel import DefensiveUnit, PassMatchup
    from monster.sim.play_kernel import PlayerIdentity


_POSITION_DEPTH_COMPATIBILITY = {
    "behind_los": {"RB": 1.35, "FB": 1.20, "WR": 1.05, "TE": 0.82},
    "short_0_5": {"RB": 1.12, "FB": 1.02, "WR": 1.08, "TE": 1.18},
    "short_6_9": {"RB": 0.92, "FB": 0.78, "WR": 1.12, "TE": 1.15},
    "intermediate_10_19": {"RB": 0.72, "FB": 0.62, "WR": 1.20, "TE": 1.12},
    "deep_20_39": {"RB": 0.52, "FB": 0.45, "WR": 1.35, "TE": 0.92},
    "bomb_40_plus": {"RB": 0.35, "FB": 0.30, "WR": 1.48, "TE": 0.70},
}

_LAST_ROUTE_FAMILY_BY_ECOLOGY: dict[int, str] = {}


_ALIGNMENT_DEPTH_COMPATIBILITY = {
    "behind_los": {
        "BACK": 1.55,
        "HBACK": 1.35,
        "SLOT": 1.10,
        "Y": 0.82,
        "F": 0.92,
        "X": 0.88,
        "Z": 0.92,
    },
    "short_0_5": {
        "BACK": 1.18,
        "HBACK": 1.20,
        "SLOT": 1.30,
        "Y": 1.22,
        "F": 1.20,
        "X": 0.96,
        "Z": 1.02,
    },
    "short_6_9": {
        "BACK": 0.90,
        "HBACK": 1.00,
        "SLOT": 1.28,
        "Y": 1.18,
        "F": 1.18,
        "X": 1.02,
        "Z": 1.08,
    },
    "intermediate_10_19": {
        "BACK": 0.62,
        "HBACK": 0.78,
        "SLOT": 1.18,
        "Y": 1.12,
        "F": 1.08,
        "X": 1.24,
        "Z": 1.25,
    },
    "deep_20_39": {
        "BACK": 0.42,
        "HBACK": 0.56,
        "SLOT": 0.96,
        "Y": 0.88,
        "F": 0.90,
        "X": 1.42,
        "Z": 1.38,
    },
    "bomb_40_plus": {
        "BACK": 0.28,
        "HBACK": 0.35,
        "SLOT": 0.78,
        "Y": 0.62,
        "F": 0.65,
        "X": 1.55,
        "Z": 1.48,
    },
}


def _position_compatibility(player: object, category: str) -> float:
    position = str(getattr(player, "position", "")).upper()
    return float(
        _POSITION_DEPTH_COMPATIBILITY.get(category, {}).get(position, 1.0)
    )


def _alignment_map(responsibility_key: str) -> dict[str, str]:
    world = snap_world(responsibility_key)
    if world is None:
        return {}
    result: dict[str, str] = {}
    for item in world.offense_alignment:
        if ":" not in item:
            continue
        player_id, role = item.split(":", 1)
        result[player_id] = role
    return result


def alignment_compatibility_v72(
    player_id: str,
    category: str,
    responsibility_key: str,
) -> float:
    role = _alignment_map(responsibility_key).get(str(player_id), "")
    return float(
        _ALIGNMENT_DEPTH_COMPATIBILITY.get(category, {}).get(role, 1.0)
    )


def choose_target_for_depth_v72(
    players: Sequence[object],
    ecology: object,
    category: str,
    rng: np.random.Generator,
    *,
    shrinkage_samples: float = 45.0,
):
    """Concept preference among actual participants, current roles and historical routes."""

    if not players:
        raise ValueError("receiver participant set cannot be empty")
    _LAST_ROUTE_FAMILY_BY_ECOLOGY[id(ecology)] = str(category)
    weights = participation_assignment_weights(
        players,
        category=category,
        attempts=getattr(ecology, "target_depth_attempts", {}),
        shrinkage_samples=shrinkage_samples,
    )
    modifiers = np.asarray(
        [
            _position_compatibility(player, category)
            * (0.82 + 0.36 * starter_authority(str(getattr(player, "player_id", ""))))
            * np.sqrt(
                max(
                    participation_multiplier(
                        str(getattr(player, "player_id", "")),
                        unit="offense",
                    ),
                    0.03,
                )
            )
            for player in players
        ],
        dtype=float,
    )
    weights = weights * modifiers
    if not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
        weights = np.ones(len(players), dtype=float)
    weights /= weights.sum()
    return players[int(rng.choice(len(players), p=weights))]


def field_read_target_v72(
    offense: object,
    defense: DefensiveUnit,
    rng: np.random.Generator,
    *,
    preferred: PlayerIdentity | None = None,
    fatigue: dict[str, float],
    responsibility_key: str = "static",
) -> tuple[PlayerIdentity, PassMatchup]:
    """Resolve a read through route family, alignment, current role and matchup."""

    candidates: list[tuple[PlayerIdentity, PassMatchup, float]] = []
    qb_fatigue = play_kernel._fatigue_factor(fatigue, offense.quarterback.player_id)
    line_fatigue = play_kernel._fatigue_factor(fatigue, f"line:{offense.team_id}")
    quarterback_efficiency = qb_execution_skill(offense.quarterback) * qb_fatigue

    ecology = getattr(offense, "intent_ecology", None)
    category = (
        "short_0_5"
        if ecology is None
        else _LAST_ROUTE_FAMILY_BY_ECOLOGY.get(id(ecology), "short_0_5")
    )

    for receiver in offense.receivers:
        matchup = resolve_pass_matchup(
            receiver,
            defense,
            pass_protection=offense.pass_protection * line_fatigue,
            quarterback_efficiency=quarterback_efficiency,
            responsibility_key=responsibility_key,
        )
        receiver_fatigue = play_kernel._fatigue_factor(fatigue, receiver.player_id)
        expected_gain = (
            matchup.completion_probability
            * matchup.yards_multiplier
            * receiver.explosive
            * receiver_fatigue
        )
        topology = alignment_compatibility_v72(
            receiver.player_id,
            category,
            responsibility_key,
        )
        current_role = np.sqrt(
            max(
                participation_multiplier(receiver.player_id, unit="offense"),
                0.03,
            )
        )
        read_score = (
            max(expected_gain, 0.02)
            * max(matchup.qb_read_quality, 0.55)
            * topology
            * current_role
        )
        if preferred is not None and receiver.player_id == preferred.player_id:
            read_score *= 1.28
        candidates.append((receiver, matchup, read_score))

    if not candidates:
        raise ValueError("pass play requires at least one eligible receiver")
    weights = np.asarray([max(item[2], 1e-5) for item in candidates], dtype=float)
    weights /= weights.sum()
    index = int(rng.choice(len(candidates), p=weights))
    receiver, matchup, _ = candidates[index]
    return receiver, matchup


def last_route_family_v72(ecology: object) -> str | None:
    return _LAST_ROUTE_FAMILY_BY_ECOLOGY.get(id(ecology))
