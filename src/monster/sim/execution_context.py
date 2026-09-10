from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ExecutionMechanism(StrEnum):
    PASS_PROTECTION = "pass_protection"
    QB_RESPONSE = "qb_response"
    ROUTE_SEPARATION = "route_separation"
    CATCHPOINT = "catchpoint"
    RUN_PENETRATION = "run_penetration"
    RUN_CONTACT = "run_contact"
    OPEN_FIELD_PURSUIT = "open_field_pursuit"
    BALL_SECURITY = "ball_security"
    KICK_FLIGHT = "kick_flight"


@dataclass(frozen=True)
class ExecutionContext:
    """Mechanism-routable execution evidence for one snap.

    Values are normalized evidence signals, not a combined success score. The runtime must
    ask for a particular mechanism's factors, preserving causal jurisdiction.
    """

    individual_matchup: float = 0.0
    unit_matchup: float = 0.0
    concept_difficulty: float = 0.0
    offense_scheme_fit: float = 0.0
    defense_counter_fit: float = 0.0
    offense_health: float = 0.0
    defense_health: float = 0.0
    weather_severity: float = 0.0
    venue_noise: float = 0.0
    fatigue: float = 0.0
    adaptation: float = 0.0

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if not -1.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be between -1 and 1")


_MECHANISM_FACTORS: dict[ExecutionMechanism, tuple[str, ...]] = {
    ExecutionMechanism.PASS_PROTECTION: (
        "individual_matchup",
        "unit_matchup",
        "concept_difficulty",
        "offense_scheme_fit",
        "defense_counter_fit",
        "offense_health",
        "defense_health",
        "venue_noise",
        "fatigue",
        "adaptation",
    ),
    ExecutionMechanism.QB_RESPONSE: (
        "individual_matchup",
        "concept_difficulty",
        "offense_health",
        "weather_severity",
        "venue_noise",
        "fatigue",
        "adaptation",
    ),
    ExecutionMechanism.ROUTE_SEPARATION: (
        "individual_matchup",
        "concept_difficulty",
        "offense_scheme_fit",
        "defense_counter_fit",
        "offense_health",
        "defense_health",
        "weather_severity",
        "fatigue",
        "adaptation",
    ),
    ExecutionMechanism.CATCHPOINT: (
        "individual_matchup",
        "concept_difficulty",
        "offense_health",
        "defense_health",
        "weather_severity",
        "fatigue",
    ),
    ExecutionMechanism.RUN_PENETRATION: (
        "individual_matchup",
        "unit_matchup",
        "concept_difficulty",
        "offense_scheme_fit",
        "defense_counter_fit",
        "offense_health",
        "defense_health",
        "venue_noise",
        "fatigue",
        "adaptation",
    ),
    ExecutionMechanism.RUN_CONTACT: (
        "individual_matchup",
        "concept_difficulty",
        "offense_health",
        "defense_health",
        "weather_severity",
        "fatigue",
    ),
    ExecutionMechanism.OPEN_FIELD_PURSUIT: (
        "individual_matchup",
        "unit_matchup",
        "offense_health",
        "defense_health",
        "weather_severity",
        "fatigue",
    ),
    ExecutionMechanism.BALL_SECURITY: (
        "individual_matchup",
        "offense_health",
        "defense_health",
        "weather_severity",
        "fatigue",
    ),
    ExecutionMechanism.KICK_FLIGHT: (
        "concept_difficulty",
        "offense_health",
        "weather_severity",
    ),
}


def factors_for_mechanism(
    context: ExecutionContext,
    mechanism: ExecutionMechanism,
) -> dict[str, float]:
    """Return only evidence with declared jurisdiction for one execution mechanism."""

    return {name: float(getattr(context, name)) for name in _MECHANISM_FACTORS[mechanism]}
