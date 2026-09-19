from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable

from monster.sim.play_kernel import PassResult, PlayEvent, PlayType


class QBRushFamily(StrEnum):
    SNEAK = "sneak"
    DESIGNED_KEEPER = "designed_keeper"
    PRESSURE_SCRAMBLE = "pressure_scramble"
    COVERAGE_SCRAMBLE = "coverage_scramble"
    KNEEL = "kneel"


@dataclass(frozen=True)
class QBRushFamilySummary:
    family: QBRushFamily
    attempts: int
    yards: float
    touchdowns: int
    fumbles_lost: int


def classify_qb_rush_event(
    event: PlayEvent,
    *,
    quarterback_id: str,
) -> QBRushFamily | None:
    """Classify an observed QB rushing event without changing play generation."""

    if event.pass_result == PassResult.SCRAMBLE and (
        event.rusher_id == quarterback_id or event.passer_id == quarterback_id
    ):
        return (
            QBRushFamily.PRESSURE_SCRAMBLE
            if event.pressured
            else QBRushFamily.COVERAGE_SCRAMBLE
        )

    if event.play_type != PlayType.RUN or event.rusher_id != quarterback_id:
        return None

    category = str(event.run_geometry_category or "").lower()
    if category == "qb_sneak":
        return QBRushFamily.SNEAK
    if category in {"kneel", "qb_kneel"}:
        return QBRushFamily.KNEEL
    return QBRushFamily.DESIGNED_KEEPER


def summarize_qb_rush_families(
    events: Iterable[PlayEvent],
    *,
    quarterback_id: str,
) -> tuple[QBRushFamilySummary, ...]:
    rows: dict[QBRushFamily, list[float | int]] = {}
    for event in events:
        family = classify_qb_rush_event(event, quarterback_id=quarterback_id)
        if family is None:
            continue
        row = rows.setdefault(family, [0, 0.0, 0, 0])
        row[0] = int(row[0]) + 1
        row[1] = float(row[1]) + float(event.yards)
        row[2] = int(row[2]) + int(bool(event.touchdown))
        row[3] = int(row[3]) + int(event.fumbler_id == quarterback_id and event.turnover)

    return tuple(
        QBRushFamilySummary(
            family=family,
            attempts=int(values[0]),
            yards=float(values[1]),
            touchdowns=int(values[2]),
            fumbles_lost=int(values[3]),
        )
        for family, values in sorted(rows.items(), key=lambda item: item[0].value)
    )
