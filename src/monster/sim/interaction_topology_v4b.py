from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import blake2b
from math import exp, log

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


def stable_unit_interval(key: str) -> float:
    digest = blake2b(key.encode("utf-8"), digest_size=8).digest()
    integer = int.from_bytes(digest, "big", signed=False)
    return (integer + 0.5) / (2**64)


def snap_responsibility_key(
    *,
    offense_team_id: str,
    defense_team_id: str,
    quarter: int,
    seconds_remaining: int,
    down: int,
    distance: float,
    yardline_100: float,
    offense_score: int,
    defense_score: int,
) -> str:
    """Build one stable defensive world identifier for the current scrimmage snap."""
    return (
        f"{offense_team_id}>{defense_team_id}:q{quarter}:t{seconds_remaining}:"
        f"d{down}:{distance:.2f}:y{yardline_100:.2f}:s{offense_score}-{defense_score}"
    )


def _weighted_index(weights: np.ndarray, *, key: str) -> int:
    arr = np.asarray(weights, dtype=float)
    if not np.isfinite(arr).all() or float(arr.sum()) <= 0.0:
        arr = np.ones(len(arr), dtype=float)
    arr /= arr.sum()
    cutoff = stable_unit_interval(key)
    index = int(np.searchsorted(np.cumsum(arr), cutoff, side="right"))
    return min(index, len(arr) - 1)


def _choose_by_exposure(
    players: tuple[object, ...],
    *,
    key: str,
    position_weights: dict[str, float] | None = None,
) -> object | None:
    if not players:
        return None
    ordered = tuple(sorted(players, key=lambda player: (_player_id(player), _position(player))))
    weights = []
    for player in ordered:
        role_weight = (
            1.0
            if position_weights is None
            else float(position_weights.get(_position(player), 0.10))
        )
        weights.append(max(_snap_weight(player), 0.001) * max(role_weight, 0.001))
    return ordered[_weighted_index(np.asarray(weights, dtype=float), key=key)]


def _weighted_without_replacement(
    players: tuple[object, ...],
    *,
    count: int,
    key: str,
    position_weights: dict[str, float] | None = None,
) -> tuple[object, ...]:
    ranked: list[tuple[float, str, object]] = []
    for player in players:
        role_weight = (
            1.0
            if position_weights is None
            else float(position_weights.get(_position(player), 0.10))
        )
        weight = max(_snap_weight(player), 0.001) * max(role_weight, 0.001)
        u = max(stable_unit_interval(f"{key}:{_player_id(player)}"), 1e-12)
        ranked.append((-log(u) / weight, _player_id(player), player))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return tuple(item[2] for item in ranked[: max(count, 0)])


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
        int(stable_unit_interval(f"coverage-slot:{_player_id(target)}") * available),
        available - 1,
    )


def choose_coverage_participants(
    *,
    target: object,
    defenders: tuple[object, ...],
    responsibility_key: str,
) -> CoverageParticipants:
    """Rotate plausible coverage responsibility by snap; skill is excluded from selection."""
    target_id = _player_id(target)
    target_position = _position(target)
    ordered = _coverage_exposure_order(defenders, target_position=target_position)
    primary = None
    if ordered:
        base_slot = _usage_responsibility_slot(target, len(ordered))
        role_weights = _coverage_role_weights(target_position)
        weights = np.asarray(
            [
                max(_snap_weight(defender), 0.001)
                * role_weights.get(_position(defender), 0.10)
                * exp(-0.65 * abs(index - base_slot))
                for index, defender in enumerate(ordered)
            ],
            dtype=float,
        )
        primary = ordered[
            _weighted_index(
                weights,
                key=f"coverage-primary:{responsibility_key}:{target_id}",
            )
        ]
    primary_id = None if primary is None else _player_id(primary)

    remaining = tuple(player for player in defenders if _player_id(player) != primary_id)
    safety_pool = tuple(
        player for player in remaining if _position(player) in {"S", "FS", "SS", "DB"}
    )
    safety = _choose_by_exposure(
        safety_pool,
        key=f"safety:{responsibility_key}:{target_id}",
    )
    safety_id = None if safety is None else _player_id(safety)

    bracket_pool = tuple(player for player in remaining if _player_id(player) != safety_id)
    bracket = None
    if float(getattr(target, "explosive", 1.0)) > 1.02 and bracket_pool:
        bracket = _choose_by_exposure(
            bracket_pool,
            key=f"bracket:{responsibility_key}:{target_id}",
            position_weights=_coverage_role_weights(target_position),
        )
    return CoverageParticipants(
        primary_id,
        safety_id,
        None if bracket is None else _player_id(bracket),
    )


def choose_pass_rushers(
    rushers: tuple[object, ...],
    *,
    count: int,
    responsibility_key: str,
) -> tuple[object, ...]:
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
    return _weighted_without_replacement(
        rushers,
        count=count,
        key=f"pass-rush:{responsibility_key}",
        position_weights=role_priority,
    )


def choose_protection_helper(
    protectors: tuple[object, ...],
    *,
    responsibility_key: str,
) -> object | None:
    return _choose_by_exposure(protectors, key=f"protection:{responsibility_key}")


def _run_front_role_weights(
    rusher_position: str,
    run_geometry: str | None,
) -> dict[str, float]:
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
    if run_geometry == "interior":
        return {
            "EDGE": 0.55,
            "DE": 0.75,
            "DT": 1.35,
            "DL": 1.20,
            "NT": 1.40,
            "LB": 1.25,
            "ILB": 1.35,
            "MLB": 1.35,
            "OLB": 0.85,
        }
    if run_geometry in {"left_edge", "right_edge", "left_offtackle", "right_offtackle"}:
        return {
            "EDGE": 1.35,
            "DE": 1.25,
            "DT": 0.75,
            "DL": 0.90,
            "NT": 0.55,
            "LB": 1.10,
            "ILB": 0.95,
            "MLB": 0.95,
            "OLB": 1.35,
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


def _pursuit_role_weights(
    rusher_position: str,
    run_geometry: str | None,
) -> dict[str, float]:
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
    edge_bonus = (
        1.15
        if run_geometry in {"left_edge", "right_edge", "left_offtackle", "right_offtackle"}
        else 1.0
    )
    return {
        "CB": 0.75 * edge_bonus,
        "DB": 1.00 * edge_bonus,
        "S": 1.30,
        "FS": 1.25,
        "SS": 1.30,
        "LB": 1.20,
        "ILB": 1.12,
        "MLB": 1.12,
        "OLB": 1.20 * edge_bonus,
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
    responsibility_key: str,
    run_geometry: str | None,
) -> RunParticipants:
    rusher_id = _player_id(rusher)
    rusher_position = _position(rusher)
    box = _choose_by_exposure(
        front,
        key=f"run-box:{responsibility_key}:{run_geometry}:{rusher_id}",
        position_weights=_run_front_role_weights(rusher_position, run_geometry),
    )
    pursuit_pool = coverage if coverage else front
    pursuit = _choose_by_exposure(
        pursuit_pool,
        key=f"run-pursuit:{responsibility_key}:{run_geometry}:{rusher_id}",
        position_weights=_pursuit_role_weights(rusher_position, run_geometry),
    )
    return RunParticipants(
        None if box is None else _player_id(box),
        None if pursuit is None else _player_id(pursuit),
    )


def choose_run_blockers(
    blockers: tuple[object, ...],
    protectors: tuple[object, ...],
    *,
    run_geometry: str | None,
) -> tuple[object, ...]:
    by_position: dict[str, list[object]] = {}
    for blocker in blockers:
        by_position.setdefault(_position(blocker), []).append(blocker)

    def positions(*names: str) -> list[object]:
        selected: list[object] = []
        for name in names:
            selected.extend(by_position.get(name, ()))
        return selected

    if run_geometry in {"interior", "qb_sneak"}:
        selected = positions("LG", "G", "OG", "C", "RG")
    elif run_geometry in {"left_offtackle", "left_edge"}:
        selected = positions("LT", "T", "OT", "LG", "G", "OG")
    elif run_geometry in {"right_offtackle", "right_edge"}:
        selected = positions("RT", "T", "OT", "RG", "G", "OG")
    else:
        selected = list(blockers)

    if run_geometry in {"left_edge", "right_edge", "left_offtackle", "right_offtackle"}:
        selected.extend(
            protector
            for protector in protectors
            if _position(protector) in {"TE", "FB"}
        )
    if not selected:
        selected = list(blockers)
    return tuple(selected)


def exposure_weighted_mean(players: Iterable[object], attribute: str) -> float:
    rows = tuple(players)
    if not rows:
        return 1.0
    weights = np.asarray(
        [max(_snap_weight(player), 0.001) for player in rows],
        dtype=float,
    )
    values = np.asarray(
        [float(getattr(player, attribute, 1.0)) for player in rows],
        dtype=float,
    )
    return float(np.average(values, weights=weights))
