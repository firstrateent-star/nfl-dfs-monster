from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from monster.interaction_registry import InteractionRegistry
from monster.registry import FeatureRegistry


class EntityScope(StrEnum):
    PLAYER = "player"
    POSITION_GROUP = "position_group"
    UNIT = "unit"
    TEAM = "team"
    GAME = "game"
    ENVIRONMENT = "environment"
    COACHING = "coaching"


class InteractionScale(StrEnum):
    ONE_V_ONE = "1v1"
    SMALL_GROUP = "small_group"
    UNIT = "unit"
    ELEVEN_V_ELEVEN = "11v11"
    GAME_STATE = "game_state"


class PlayPhase(StrEnum):
    PRE_SNAP = "pre_snap"
    INITIAL_CONTACT = "initial_contact"
    DEVELOPMENT = "development"
    RESOLUTION = "resolution"
    POST_PLAY = "post_play"


@dataclass(frozen=True)
class MetricObservation:
    """One evidence value attached to a real football entity.

    Values remain in their native/compiled units. The interaction runtime deliberately
    does not normalize or blend unrelated metrics; the owning football mechanism is
    responsible for interpreting the evidence it explicitly requests.
    """

    feature_name: str
    entity_id: str
    entity_scope: EntityScope
    value: float
    confidence: float = 1.0
    source: str | None = None


@dataclass(frozen=True)
class InteractionContext:
    mechanism: str
    phase: PlayPhase
    scale: InteractionScale
    participant_ids: frozenset[str] = frozenset()


@dataclass(frozen=True)
class RoutedMetric:
    observation: MetricObservation
    authority: float
    production_weight: float
    confidence: float

    @property
    def effective_authority(self) -> float:
        return self.authority * self.production_weight * self.confidence


@dataclass(frozen=True)
class RoutingDecision:
    accepted: tuple[RoutedMetric, ...]
    rejected: tuple[tuple[MetricObservation, str], ...]


class InteractionMetricRouter:
    """Gate evidence into only the football interactions where it has jurisdiction."""

    def __init__(
        self,
        feature_registry: FeatureRegistry,
        interaction_registry: InteractionRegistry,
    ) -> None:
        self.features = feature_registry
        self.interactions = interaction_registry

    def route(
        self,
        observations: tuple[MetricObservation, ...],
        context: InteractionContext,
        *,
        production_upstream: bool = True,
    ) -> RoutingDecision:
        accepted: list[RoutedMetric] = []
        rejected: list[tuple[MetricObservation, str]] = []

        for observation in observations:
            feature = self.features.specs.get(observation.feature_name)
            if feature is None:
                rejected.append((observation, "unknown_feature"))
                continue
            interaction = self.interactions.specs.get(observation.feature_name)
            if interaction is None:
                rejected.append((observation, "no_interaction_jurisdiction"))
                continue
            if production_upstream and feature.market_derived:
                rejected.append((observation, "market_derived_forbidden_upstream"))
                continue
            if production_upstream and (
                feature.authority <= 0.0
                or feature.production_weight <= 0.0
                or interaction.audit_only
            ):
                rejected.append((observation, "no_production_authority"))
                continue
            if observation.entity_scope.value not in interaction.entity_scopes:
                rejected.append((observation, "entity_scope_mismatch"))
                continue
            if context.scale.value not in interaction.interaction_scales:
                rejected.append((observation, "interaction_scale_mismatch"))
                continue
            if context.phase.value not in interaction.phases:
                rejected.append((observation, "play_phase_mismatch"))
                continue
            if context.mechanism not in interaction.mechanisms:
                rejected.append((observation, "mechanism_mismatch"))
                continue
            if context.participant_ids and observation.entity_id not in context.participant_ids:
                rejected.append((observation, "entity_not_participating"))
                continue

            confidence = min(max(float(observation.confidence), 0.0), 1.0)
            accepted.append(
                RoutedMetric(
                    observation=observation,
                    authority=float(feature.authority),
                    production_weight=float(feature.production_weight),
                    confidence=confidence,
                )
            )

        return RoutingDecision(tuple(accepted), tuple(rejected))
