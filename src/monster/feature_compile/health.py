from __future__ import annotations

from typing import Any

import polars as pl

_REPORT_AVAILABILITY = {
    "out": 0.01,
    "doubtful": 0.20,
    "questionable": 0.68,
}
_REPORT_EFFECTIVENESS = {
    "out": 0.88,
    "doubtful": 0.91,
    "questionable": 0.96,
}
_REPORT_UNCERTAINTY = {
    "out": 0.03,
    "doubtful": 0.18,
    "questionable": 0.22,
}
_PRACTICE_AVAILABILITY = {
    "did not participate": 0.72,
    "dnp": 0.72,
    "limited": 0.86,
    "limited participation": 0.86,
    "full": 0.985,
    "full participation": 0.985,
}
_PRACTICE_EFFECTIVENESS = {
    "did not participate": 0.94,
    "dnp": 0.94,
    "limited": 0.97,
    "limited participation": 0.97,
    "full": 0.995,
    "full participation": 0.995,
}
_PRACTICE_UNCERTAINTY = {
    "did not participate": 0.18,
    "dnp": 0.18,
    "limited": 0.14,
    "limited participation": 0.14,
    "full": 0.06,
    "full participation": 0.06,
}


def _finite(value: Any, default: float) -> float:
    if value is None or value == "":
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if result == result else default


def _normal(value: Any) -> str:
    return str(value or "").strip().lower()


def health_state_from_evidence(row: dict[str, Any]) -> tuple[float, float, float]:
    """Convert available injury/practice evidence into separate health dimensions.

    Returns P(active), conditional effectiveness if active, and health uncertainty.
    Roster/status availability remains an independent prior and is combined conservatively.
    """
    base_availability = _finite(row.get("game_day_active_probability"), 0.985)
    availability = base_availability
    effectiveness = 1.0
    uncertainty = _finite(row.get("participation_uncertainty"), 0.08)

    report_status = _normal(
        row.get("report_status")
        or row.get("game_status")
        or row.get("injury_status")
    )
    if report_status in _REPORT_AVAILABILITY:
        availability = min(availability, _REPORT_AVAILABILITY[report_status])
        effectiveness = min(effectiveness, _REPORT_EFFECTIVENESS[report_status])
        uncertainty = max(uncertainty, _REPORT_UNCERTAINTY[report_status])

    practice_status = _normal(
        row.get("practice_status")
        or row.get("practice_participation")
        or row.get("practice")
    )
    if practice_status in _PRACTICE_AVAILABILITY:
        availability = min(availability, _PRACTICE_AVAILABILITY[practice_status])
        effectiveness = min(effectiveness, _PRACTICE_EFFECTIVENESS[practice_status])
        uncertainty = max(uncertainty, _PRACTICE_UNCERTAINTY[practice_status])

    return (
        float(max(0.0, min(1.0, availability))),
        float(max(0.25, min(1.05, effectiveness))),
        float(max(0.02, min(0.35, uncertainty))),
    )


def apply_health_state(personnel: pl.DataFrame) -> pl.DataFrame:
    """Attach health dimensions to personnel without collapsing role and capability."""
    rows = []
    for row in personnel.to_dicts():
        availability, effectiveness, uncertainty = health_state_from_evidence(row)
        row["health_availability_probability"] = availability
        row["health_effectiveness_if_active"] = effectiveness
        row["health_uncertainty"] = uncertainty
        rows.append(row)
    return pl.DataFrame(rows, schema=personnel.schema | {
        "health_availability_probability": pl.Float64,
        "health_effectiveness_if_active": pl.Float64,
        "health_uncertainty": pl.Float64,
    })
