from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class CoverageParticipants:
    primary_defender_id: str | None
    safety_defender_id: str | None
    bracket_defender_id: str | None


@dataclass(frozen=True)
class RunParticipants:
    box_defender_id: str | None
    pursuit_defender_id: str | None


def _player_id(player: object) -> str:
    return str(getattr(player, "player_id", ""))


def _position(player: object) -> str:
    return str(getattr(player, "position", "")).upper()


def _snap_weight(player: object) -> float:
    return float(np.clip(getattr(player, "snap_weight", 0.0) or 0.0, 0.0, 1.0))


def _stable_unit_interval(key: str) -> float:
    digest = blake2b(key.encode("utf-8"), digest_size=8).digest()
    integer = int.from_bytes(digest, "big", signed=False)
    return (integer + 0.5) / (2**64)


def _choose_by_exposure(
    players: tuple[object, ...],
    *,
    key: str,
    position_weights: dict[str, float] | None = None,
) -> object | None:
    """Choose responsibility from participation/role evidence, never player skill.

    This is deterministic for a given responsibility key so field-read evaluation does not
    consume extra randomness or let candidate ordering change the simulated world. Skill is
    intentionally excluded from the selection weights; it belongs in the downstream duel.
    """
    if not players:
        return None

    ordered = tuple(sorted(players, key=lambda player: (_player_id(player), _position(player))))
    weights = []
    for player in ordered:
        role_weight = 1.0
        if position_weights is not None:
            role_weight = float(position_weights.get(_position(player), 0.10))
        weights.append(max(_snap_weight(player), 0.001) * max(role_weight, 0.0))

    arr = np.asarray(weights, dtype=float)
    if not np.isfinite(arr).all() or float(arr.sum()) <= 0.0:
        arr = np.ones(len(ordered), dtype=float)
    arr /= arr.sum()
    cutoff = _stable_unit_interval(key)
    index = int(np.searchsorted(np.cumsum(arr), cutoff, side="right"))
    index = min(index, len(ordered) - 1)
    return ordered[index]


def _coverage_role_weights(target_position: str) -> dict[str, float]:
    position = target_position.upper()
    if position == "TE":
        return {
            "CB": 0.55,
            "DB": 0.85,
            "S": 1.25,
            "FS": 1.15,
            "SS": 1.25,
            "LB": 1.20,
            "ILB": 1.20,
            "MLB": 1.20,
            "OLB": 1.05,
        }
    if position in {"RB", "FB"}:
        return {
            "CB": 0.45,
            "DB": 0.80,
            "S": 1.15,
            "FS": 1.05,
            "SS": 1.15,
            "LB": 1.30,
            "ILB": 1.30,
            "MLB": 1.30,
            "OLB": 1.20,
        }
    return {
        "CB": 1.35,
        "DB": 1.15,
        "S": 0.85,
        "FS": 0.80,
        "SS": 0.80,
        "LB": 0.38,
        "ILB": 0.30,
        "MLB": 0.30,
        "OLB": 0.42,
    }


def _coverage_exposure_order(
    defenders: tuple[object, ...],
    *,
    target_position: str,
) -> tuple[object, ...]:
    role_weights = _coverage_role_weights(target_position)
    return tuple(
        sorted(
            defenders,
            key=lambda player: (
                -_snap_weight(player) * role_weights.get(_position(player), 0.10),
                _player_id(player),
            ),
        )
    )


def _usage_responsibility_slot(target: object, available: int) -> int:
    """Map offensive opportunity hierarchy to defender exposure hierarchy.

    This is a temporary alignment proxy until route/alignment data is promoted. It uses only
    offensive role/opportunity and defensive participation, never defensive skill.
    """
    if available <= 1:
        return 0
    usage = float(np.clip(getattr(target, "usage_weight", 0.0) or 0.0, 0.0, 1.0))
    position = _position(target)
    if position == "WR":
        if usage >= 0.32:
            return 0
        if usage >= 0.24:
            return min(1, available - 1)
        if usage >= 0.14:
            return min(2, available - 1)
    if position == "TE" and usage >= 0.18:
        return 0
    if position in {"RB", "FB"} and usage >= 0.14:
        return 0
    return min(
        int(_stable_unit_interval(f"coverage-slot:{_player_id(target)}") * available),
        available - 1,
    )


def choose_coverage_participants(
    *,
    target: object,
    defenders: tuple[object, ...],
) -> CoverageParticipants:
    """Assign primary/help coverage from role and exposure before evaluating skill."""
    target_id = _player_id(target)
    target_position = _position(target)
    ordered = _coverage_exposure_order(defenders, target_position=target_position)
    primary = None if not ordered else ordered[_usage_responsibility_slot(target, len(ordered))]
    primary_id = None if primary is None else _player_id(primary)

    remaining = tuple(player for player in defenders if _player_id(player) != primary_id)
    safety_pool = tuple(
        player for player in remaining if _position(player) in {"S", "FS", "SS", "DB"}
    )
    safety = _choose_by_exposure(safety_pool, key=f"safety-help:{target_id}")
    safety_id = None if safety is None else _player_id(safety)

    bracket_pool = tuple(player for player in remaining if _player_id(player) != safety_id)
    bracket = None
    if float(getattr(target, "explosive", 1.0)) > 1.02 and bracket_pool:
        bracket = _choose_by_exposure(
            bracket_pool,
            key=f"bracket:{target_id}",
            position_weights=_coverage_role_weights(target_position),
        )
    bracket_id = None if bracket is None else _player_id(bracket)
    return CoverageParticipants(primary_id, safety_id, bracket_id)


def choose_pass_rushers(
    rushers: tuple[object, ...],
    *,
    count: int = 5,
) -> tuple[object, ...]:
    """Choose the rush population from role/exposure only, not pass-rush rating."""
    role_priority = {
        "EDGE": 1.20,
        "DE": 1.15,
        "DT": 1.10,
        "DL": 1.05,
        "NT": 0.95,
        "OLB": 0.82,
        "LB": 0.62,
        "ILB": 0.52,
        "MLB": 0.52,
    }
    return tuple(
        sorted(
            rushers,
            key=lambda player: (
                -_snap_weight(player) * role_priority.get(_position(player), 0.55),
                _player_id(player),
            ),
        )[: max(count, 0)]
    )


def pair_pass_rushers_to_blockers(
    rushers: tuple[object, ...],
    blockers: tuple[object, ...],
) -> tuple[tuple[object, object], ...]:
    """Pair rushers to plausible OL roles without consulting either player's skill rating."""
    if not rushers or not blockers:
        return ()

    tackles = tuple(
        blocker for blocker in blockers if _position(blocker) in {"LT", "RT", "T", "OT"}
    )
    interior = tuple(
        blocker for blocker in blockers if _position(blocker) in {"LG", "RG", "G", "OG", "C"}
    )
    all_blockers = tuple(
        sorted(blockers, key=lambda player: (_position(player), _player_id(player)))
    )
    usage: dict[str, int] = {_player_id(blocker): 0 for blocker in all_blockers}
    pairs: list[tuple[object, object]] = []

    for rusher in rushers:
        position = _position(rusher)
        if position in {"EDGE", "DE", "OLB"} and tackles:
            pool = tackles
        elif position in {"DT", "DL", "NT", "ILB", "MLB", "LB"} and interior:
            pool = interior
        else:
            pool = all_blockers
        blocker = min(
            pool,
            key=lambda candidate: (
                usage[_player_id(candidate)],
                -_snap_weight(candidate),
                _player_id(candidate),
            ),
        )
        usage[_player_id(blocker)] += 1
        pairs.append((rusher, blocker))
    return tuple(pairs)


def choose_protection_helper(
    protectors: tuple[object, ...],
    *,
    responsibility_key: str,
) -> object | None:
    """Select a helper from snap exposure; blocking skill only affects the help after selection."""
    return _choose_by_exposure(protectors, key=f"protection-help:{responsibility_key}")


def _run_front_role_weights(rusher_position: str) -> dict[str, float]:
    if rusher_position.upper() == "QB":
        return {
            "EDGE": 1.05,
            "DE": 1.00,
            "DT": 0.90,
            "DL": 0.90,
            "NT": 0.85,
            "LB": 1.25,
            "ILB": 1.20,
            "MLB": 1.20,
            "OLB": 1.25,
        }
    return {
        "EDGE": 0.95,
        "DE": 1.00,
        "DT": 1.12,
        "DL": 1.08,
        "NT": 1.12,
        "LB": 1.20,
        "ILB": 1.25,
        "MLB": 1.25,
        "OLB": 1.10,
    }


def _pursuit_role_weights(rusher_position: str) -> dict[str, float]:
    if rusher_position.upper() == "QB":
        return {
            "CB": 0.85,
            "DB": 1.00,
            "S": 1.20,
            "FS": 1.15,
            "SS": 1.20,
            "LB": 1.20,
            "ILB": 1.10,
            "MLB": 1.10,
            "OLB": 1.30,
            "EDGE": 0.85,
        }
    return {
        "CB": 0.75,
        "DB": 1.00,
        "S": 1.30,
        "FS": 1.25,
        "SS": 1.30,
        "LB": 1.20,
        "ILB": 1.12,
        "MLB": 1.12,
        "OLB": 1.20,
        "EDGE": 0.55,
        "DE": 0.45,
        "DT": 0.20,
        "DL": 0.25,
    }


def choose_run_participants(
    *,
    rusher: object,
    front: tuple[object, ...],
    coverage: tuple[object, ...],
) -> RunParticipants:
    """Assign box and pursuit responsibility without reading the eventual rushing outcome."""
    rusher_id = _player_id(rusher)
    rusher_position = _position(rusher)
    box = _choose_by_exposure(
        front,
        key=f"run-box:{rusher_id}",
        position_weights=_run_front_role_weights(rusher_position),
    )
    pursuit_pool = coverage if coverage else front
    pursuit = _choose_by_exposure(
        pursuit_pool,
        key=f"run-pursuit:{rusher_id}",
        position_weights=_pursuit_role_weights(rusher_position),
    )
    return RunParticipants(
        None if box is None else _player_id(box),
        None if pursuit is None else _player_id(pursuit),
    )


def exposure_weighted_mean(players: Iterable[object], attribute: str) -> float:
    rows = tuple(players)
    if not rows:
        return 1.0
    weights = np.asarray(
        [max(_snap_weight(player), 0.001) for player in rows], dtype=float
    )
    values = np.asarray(
        [float(getattr(player, attribute, 1.0)) for player in rows], dtype=float
    )
    return float(np.average(values, weights=weights))


def effective_participant_count(player_ids: Iterable[str | None]) -> float:
    counts: dict[str, int] = {}
    total = 0
    for player_id in player_ids:
        if not player_id:
            continue
        counts[player_id] = counts.get(player_id, 0) + 1
        total += 1
    if total == 0:
        return 0.0
    shares = np.asarray([count / total for count in counts.values()], dtype=float)
    return float(1.0 / np.sum(np.square(shares)))
