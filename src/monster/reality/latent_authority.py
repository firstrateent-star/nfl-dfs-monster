from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from monster.reality.world_state import GameDayLatents


class LatentMechanism(StrEnum):
    QB_READ = "qb_read"
    THROW_EXECUTION = "throw_execution"
    PRESSURE_RESPONSE = "pressure_response"
    PASS_PROTECTION = "pass_protection"
    ROUTE_EXECUTION = "route_execution"
    CATCHPOINT = "catchpoint"
    YAC = "yac"
    RUN_BLOCKING = "run_blocking"
    RUN_CONTACT = "run_contact"
    OPEN_FIELD = "open_field"
    PASS_RUSH = "pass_rush"
    COVERAGE = "coverage"
    RUN_FIT = "run_fit"
    TACKLING = "tackling"
    FIELD_GOAL = "field_goal"
    PUNT = "punt"
    KICK_RETURN = "kick_return"
    PUNT_RETURN = "punt_return"


@dataclass(frozen=True)
class LatentAuthoritySpec:
    factor: str
    mechanisms: frozenset[LatentMechanism]
    max_authority: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.max_authority <= 1.0:
            raise ValueError("latent max authority must be between zero and one")


@dataclass(frozen=True)
class RoutedLatent:
    team_id: str
    factor: str
    mechanism: LatentMechanism
    standardized_value: float
    requested_authority: float
    applied_authority: float

    @property
    def effective_value(self) -> float:
        return self.standardized_value * self.applied_authority


LATENT_AUTHORITY: dict[str, LatentAuthoritySpec] = {
    "qb_execution": LatentAuthoritySpec(
        "qb_execution",
        frozenset(
            {
                LatentMechanism.QB_READ,
                LatentMechanism.THROW_EXECUTION,
                LatentMechanism.PRESSURE_RESPONSE,
            }
        ),
        0.35,
    ),
    "pass_protection": LatentAuthoritySpec(
        "pass_protection",
        frozenset({LatentMechanism.PASS_PROTECTION}),
        0.30,
    ),
    "receiver_execution": LatentAuthoritySpec(
        "receiver_execution",
        frozenset(
            {
                LatentMechanism.ROUTE_EXECUTION,
                LatentMechanism.CATCHPOINT,
                LatentMechanism.YAC,
            }
        ),
        0.30,
    ),
    "run_blocking": LatentAuthoritySpec(
        "run_blocking",
        frozenset({LatentMechanism.RUN_BLOCKING}),
        0.30,
    ),
    "ballcarrier_execution": LatentAuthoritySpec(
        "ballcarrier_execution",
        frozenset({LatentMechanism.RUN_CONTACT, LatentMechanism.OPEN_FIELD}),
        0.30,
    ),
    "pass_rush": LatentAuthoritySpec(
        "pass_rush",
        frozenset({LatentMechanism.PASS_RUSH}),
        0.30,
    ),
    "coverage_execution": LatentAuthoritySpec(
        "coverage_execution",
        frozenset({LatentMechanism.COVERAGE}),
        0.30,
    ),
    "run_fit": LatentAuthoritySpec(
        "run_fit",
        frozenset({LatentMechanism.RUN_FIT}),
        0.30,
    ),
    "tackling": LatentAuthoritySpec(
        "tackling",
        frozenset({LatentMechanism.TACKLING}),
        0.30,
    ),
    "special_teams_execution": LatentAuthoritySpec(
        "special_teams_execution",
        frozenset(
            {
                LatentMechanism.FIELD_GOAL,
                LatentMechanism.PUNT,
                LatentMechanism.KICK_RETURN,
                LatentMechanism.PUNT_RETURN,
            }
        ),
        0.25,
    ),
}


def route_team_latent(
    latents: GameDayLatents,
    *,
    team_id: str,
    factor: str,
    mechanism: LatentMechanism,
    requested_authority: float,
) -> RoutedLatent:
    """Route one latent only to a football mechanism with declared jurisdiction."""

    spec = LATENT_AUTHORITY.get(factor)
    if spec is None:
        raise KeyError(f"unknown v7 latent factor: {factor}")
    if mechanism not in spec.mechanisms:
        raise ValueError(f"{factor} has no jurisdiction over {mechanism.value}")
    if requested_authority < 0.0:
        raise ValueError("requested latent authority cannot be negative")

    value = latents.value(f"{team_id}.{factor}")
    applied = min(float(requested_authority), spec.max_authority)
    return RoutedLatent(
        team_id=team_id,
        factor=factor,
        mechanism=mechanism,
        standardized_value=float(value),
        requested_authority=float(requested_authority),
        applied_authority=applied,
    )
