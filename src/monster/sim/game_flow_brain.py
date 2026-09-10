from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, log

import numpy as np

from monster.sim.game_flow import GameFlowState


def _clip_probability(value: float) -> float:
    return float(np.clip(float(value), 0.01, 0.99))


def _logit(probability: float) -> float:
    p = _clip_probability(probability)
    return log(p / (1.0 - p))


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = exp(-value)
        return 1.0 / (1.0 + z)
    z = exp(value)
    return z / (1.0 + z)


@dataclass(frozen=True)
class RateEvidence:
    """Observed rate with sample size for hierarchical shrinkage."""

    rate: float
    samples: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.rate <= 1.0:
            raise ValueError("rate must be between 0 and 1")
        if self.samples < 0:
            raise ValueError("samples cannot be negative")


@dataclass(frozen=True)
class DecisionAdjustment:
    """One audited layer's bounded contribution to play-choice log odds.

    The value is deliberately expressed in log-odds space so independent evidence layers
    compose without directly pretending to be play success. A zero adjustment is neutral.
    """

    source: str
    dropback_logit_shift: float = 0.0
    authority: float = 0.0
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.authority <= 1.0:
            raise ValueError("authority must be between 0 and 1")
        if not np.isfinite(self.dropback_logit_shift):
            raise ValueError("dropback_logit_shift must be finite")

    @property
    def authorized_shift(self) -> float:
        return float(self.dropback_logit_shift * self.authority)


@dataclass(frozen=True)
class HierarchicalFlowEvidence:
    """Evidence layers for one pre-snap GameFlowState.

    `league_context` is the broad situation prior. `team_context` is the current
    team/coach-style proxy in the same or compatible situation and is shrunk toward the
    league prior. Other layers must arrive as separately audited adjustments.
    """

    league_context: RateEvidence
    team_context: RateEvidence | None = None
    team_neutral_rate: float | None = None
    league_neutral_rate: float | None = None
    shrinkage_samples: float = 80.0
    personnel: DecisionAdjustment = field(
        default_factory=lambda: DecisionAdjustment("personnel")
    )
    opponent: DecisionAdjustment = field(
        default_factory=lambda: DecisionAdjustment("opponent")
    )
    environment: DecisionAdjustment = field(
        default_factory=lambda: DecisionAdjustment("environment")
    )
    venue: DecisionAdjustment = field(
        default_factory=lambda: DecisionAdjustment("venue")
    )
    adaptation: DecisionAdjustment = field(
        default_factory=lambda: DecisionAdjustment("adaptation")
    )

    def __post_init__(self) -> None:
        if self.shrinkage_samples <= 0:
            raise ValueError("shrinkage_samples must be positive")
        for rate in (self.team_neutral_rate, self.league_neutral_rate):
            if rate is not None and not 0.0 <= rate <= 1.0:
                raise ValueError("neutral rates must be between 0 and 1")


@dataclass(frozen=True)
class GameFlowDecisionTrace:
    league_rate: float
    team_context_rate: float | None
    team_context_authority: float
    neutral_identity_logit_shift: float
    personnel_logit_shift: float
    opponent_logit_shift: float
    environment_logit_shift: float
    venue_logit_shift: float
    adaptation_logit_shift: float
    total_logit_shift: float
    final_dropback_probability: float


@dataclass(frozen=True)
class GameFlowDecision:
    flow: GameFlowState
    dropback_probability: float
    trace: GameFlowDecisionTrace


def _team_context_authority(samples: int, shrinkage_samples: float) -> float:
    if samples <= 0:
        return 0.0
    return float(samples / (samples + shrinkage_samples))


def _shrunken_team_context_rate(
    league: RateEvidence,
    team: RateEvidence | None,
    *,
    shrinkage_samples: float,
) -> tuple[float, float]:
    if team is None or team.samples <= 0:
        return league.rate, 0.0
    authority = _team_context_authority(team.samples, shrinkage_samples)
    rate = league.rate + authority * (team.rate - league.rate)
    return _clip_probability(rate), authority


def _neutral_identity_shift(
    team_neutral_rate: float | None,
    league_neutral_rate: float | None,
) -> float:
    if team_neutral_rate is None or league_neutral_rate is None:
        return 0.0
    # A broad team identity is lower authority than an exact situation cell. Bound it so
    # a team's season-wide preference cannot erase the state-specific football prior.
    return float(
        np.clip(
            0.35 * (_logit(team_neutral_rate) - _logit(league_neutral_rate)),
            -0.35,
            0.35,
        )
    )


def decide_game_flow(
    flow: GameFlowState,
    evidence: HierarchicalFlowEvidence,
) -> GameFlowDecision:
    """Compose audited decision layers without assigning execution success.

    This function chooses only the probability of dropback-family intent. It does not
    choose pass depth, run concept, target, carrier, or outcome. Those remain later
    tactical and execution layers.
    """

    shrunken_rate, team_authority = _shrunken_team_context_rate(
        evidence.league_context,
        evidence.team_context,
        shrinkage_samples=evidence.shrinkage_samples,
    )
    identity_shift = _neutral_identity_shift(
        evidence.team_neutral_rate,
        evidence.league_neutral_rate,
    )
    personnel_shift = evidence.personnel.authorized_shift
    opponent_shift = evidence.opponent.authorized_shift
    environment_shift = evidence.environment.authorized_shift
    venue_shift = evidence.venue.authorized_shift
    adaptation_shift = evidence.adaptation.authorized_shift

    raw_shift = (
        identity_shift
        + personnel_shift
        + opponent_shift
        + environment_shift
        + venue_shift
        + adaptation_shift
    )
    # Multiple individually reasonable layers must not combine into an absurd strategic
    # override. Wider behavior can earn authority later through explicit OOS evidence.
    total_shift = float(np.clip(raw_shift, -1.10, 1.10))
    final_probability = _clip_probability(_sigmoid(_logit(shrunken_rate) + total_shift))

    trace = GameFlowDecisionTrace(
        league_rate=evidence.league_context.rate,
        team_context_rate=None if evidence.team_context is None else evidence.team_context.rate,
        team_context_authority=team_authority,
        neutral_identity_logit_shift=identity_shift,
        personnel_logit_shift=personnel_shift,
        opponent_logit_shift=opponent_shift,
        environment_logit_shift=environment_shift,
        venue_logit_shift=venue_shift,
        adaptation_logit_shift=adaptation_shift,
        total_logit_shift=total_shift,
        final_dropback_probability=final_probability,
    )
    return GameFlowDecision(flow=flow, dropback_probability=final_probability, trace=trace)
