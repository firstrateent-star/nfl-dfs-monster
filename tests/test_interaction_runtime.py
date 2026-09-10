from __future__ import annotations

import pytest

from monster.interaction_registry import InteractionRegistry
from monster.registry import FeatureRegistry
from monster.sim.interaction_runtime import (
    EntityScope,
    InteractionContext,
    InteractionMetricRouter,
    InteractionScale,
    MetricObservation,
    PlayPhase,
)


def _router() -> InteractionMetricRouter:
    features = FeatureRegistry.load()
    interactions = InteractionRegistry.load(feature_registry=features)
    return InteractionMetricRouter(features, interactions)


def test_registered_metric_routes_only_into_its_football_jurisdiction() -> None:
    router = _router()
    observation = MetricObservation(
        feature_name="madden_pass_rush",
        entity_id="edge-1",
        entity_scope=EntityScope.PLAYER,
        value=92.0,
        confidence=0.8,
        source="madden",
    )
    context = InteractionContext(
        mechanism="pressure",
        phase=PlayPhase.DEVELOPMENT,
        scale=InteractionScale.UNIT,
        participant_ids=frozenset({"edge-1", "lt-1", "qb-1"}),
    )
    decision = router.route((observation,), context)

    assert len(decision.accepted) == 1
    assert not decision.rejected
    assert decision.accepted[0].effective_authority == pytest.approx(0.25 * 0.8)


def test_wrong_interaction_scale_is_rejected_instead_of_silently_blended() -> None:
    router = _router()
    observation = MetricObservation(
        feature_name="wingspan_in",
        entity_id="wr-1",
        entity_scope=EntityScope.PLAYER,
        value=81.0,
    )
    context = InteractionContext(
        mechanism="catchpoint",
        phase=PlayPhase.RESOLUTION,
        scale=InteractionScale.ELEVEN_V_ELEVEN,
        participant_ids=frozenset({"wr-1"}),
    )
    decision = router.route((observation,), context)

    assert not decision.accepted
    assert decision.rejected[0][1] == "interaction_scale_mismatch"


def test_nonparticipant_metric_cannot_affect_current_snap() -> None:
    router = _router()
    observation = MetricObservation(
        feature_name="madden_coverage",
        entity_id="cb-bench",
        entity_scope=EntityScope.PLAYER,
        value=96.0,
    )
    context = InteractionContext(
        mechanism="coverage",
        phase=PlayPhase.DEVELOPMENT,
        scale=InteractionScale.ONE_V_ONE,
        participant_ids=frozenset({"wr-1", "cb-1"}),
    )
    decision = router.route((observation,), context)

    assert not decision.accepted
    assert decision.rejected[0][1] == "entity_not_participating"


def test_market_metric_is_forbidden_upstream_but_available_to_audit() -> None:
    router = _router()
    observation = MetricObservation(
        feature_name="sportsbook_spread",
        entity_id="game-1",
        entity_scope=EntityScope.GAME,
        value=-3.5,
    )
    context = InteractionContext(
        mechanism="market_audit",
        phase=PlayPhase.POST_PLAY,
        scale=InteractionScale.GAME_STATE,
        participant_ids=frozenset({"game-1"}),
    )

    upstream = router.route((observation,), context, production_upstream=True)
    assert not upstream.accepted
    assert upstream.rejected[0][1] == "market_derived_forbidden_upstream"

    audit = router.route((observation,), context, production_upstream=False)
    assert len(audit.accepted) == 1


def test_zero_authority_experimental_metric_stays_shadow_only() -> None:
    router = _router()
    observation = MetricObservation(
        feature_name="natal_sun_longitude",
        entity_id="qb-1",
        entity_scope=EntityScope.PLAYER,
        value=210.0,
    )
    context = InteractionContext(
        mechanism="experimental_shadow",
        phase=PlayPhase.PRE_SNAP,
        scale=InteractionScale.GAME_STATE,
        participant_ids=frozenset({"qb-1"}),
    )
    decision = router.route((observation,), context)

    assert not decision.accepted
    assert decision.rejected[0][1] == "no_production_authority"


def test_unknown_feature_is_explicitly_rejected() -> None:
    router = _router()
    observation = MetricObservation(
        feature_name="future_metric_not_registered",
        entity_id="p1",
        entity_scope=EntityScope.PLAYER,
        value=1.0,
    )
    context = InteractionContext(
        mechanism="pressure",
        phase=PlayPhase.DEVELOPMENT,
        scale=InteractionScale.ONE_V_ONE,
    )
    decision = router.route((observation,), context)

    assert not decision.accepted
    assert decision.rejected[0][1] == "unknown_feature"
