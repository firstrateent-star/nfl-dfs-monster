from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Any, Mapping

from monster.interaction_registry import InteractionRegistry
from monster.registry import FeatureRegistry
from monster.sim.interaction_runtime import EntityScope, MetricObservation


@dataclass(frozen=True)
class ObservationCompileResult:
    observations: tuple[MetricObservation, ...]
    ignored: tuple[tuple[str, str], ...]


def compile_metric_observations(
    *,
    entity_id: str,
    entity_scope: EntityScope,
    values: Mapping[str, Any],
    feature_registry: FeatureRegistry,
    interaction_registry: InteractionRegistry,
    confidence_by_feature: Mapping[str, float] | None = None,
    source_by_feature: Mapping[str, str] | None = None,
) -> ObservationCompileResult:
    """Turn an arbitrary reality record into governed interaction observations.

    The adapter intentionally accepts a broad mapping so new data sources can expose new
    columns without changing the runtime. A value only becomes routable when its name exists
    in both registries. Everything else remains visible in ``ignored`` for provenance/audit.
    """
    confidence_by_feature = confidence_by_feature or {}
    source_by_feature = source_by_feature or {}
    observations: list[MetricObservation] = []
    ignored: list[tuple[str, str]] = []

    for feature_name, value in values.items():
        if value is None:
            ignored.append((feature_name, "missing_value"))
            continue
        if feature_name not in feature_registry.specs:
            ignored.append((feature_name, "ungoverned_feature"))
            continue
        if feature_name not in interaction_registry.specs:
            ignored.append((feature_name, "no_interaction_jurisdiction"))
            continue
        if isinstance(value, bool):
            numeric = float(value)
        elif isinstance(value, Real):
            numeric = float(value)
        else:
            ignored.append((feature_name, "non_numeric_value"))
            continue

        observations.append(
            MetricObservation(
                feature_name=feature_name,
                entity_id=entity_id,
                entity_scope=entity_scope,
                value=numeric,
                confidence=float(confidence_by_feature.get(feature_name, 1.0)),
                source=source_by_feature.get(feature_name),
            )
        )

    return ObservationCompileResult(tuple(observations), tuple(ignored))
