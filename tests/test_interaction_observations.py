from __future__ import annotations

from monster.feature_compile.interaction_observations import compile_metric_observations
from monster.interaction_registry import InteractionRegistry
from monster.registry import FeatureRegistry
from monster.sim.interaction_runtime import EntityScope


def test_observation_compiler_admits_only_governed_routable_metrics() -> None:
    features = FeatureRegistry.load()
    interactions = InteractionRegistry.load(feature_registry=features)
    result = compile_metric_observations(
        entity_id="wr-1",
        entity_scope=EntityScope.PLAYER,
        values={
            "height_in": 74.0,
            "wingspan_in": 80.0,
            "future_unregistered_metric": 123.0,
            "madden_catching": None,
            "team_name": "Example",
        },
        feature_registry=features,
        interaction_registry=interactions,
        confidence_by_feature={"wingspan_in": 0.75},
        source_by_feature={"wingspan_in": "combine"},
    )

    assert [item.feature_name for item in result.observations] == ["height_in", "wingspan_in"]
    wingspan = result.observations[1]
    assert wingspan.confidence == 0.75
    assert wingspan.source == "combine"
    assert ("future_unregistered_metric", "ungoverned_feature") in result.ignored
    assert ("madden_catching", "missing_value") in result.ignored
    assert ("team_name", "ungoverned_feature") in result.ignored


def test_boolean_metric_is_encoded_without_custom_plumbing() -> None:
    features = FeatureRegistry.load()
    interactions = InteractionRegistry.load(feature_registry=features)
    result = compile_metric_observations(
        entity_id="game-1",
        entity_scope=EntityScope.ENVIRONMENT,
        values={"dome_flag": True},
        feature_registry=features,
        interaction_registry=interactions,
    )

    assert len(result.observations) == 1
    assert result.observations[0].value == 1.0
