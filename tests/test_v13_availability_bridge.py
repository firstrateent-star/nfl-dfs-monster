from __future__ import annotations

from monster.feature_compile.mechanisms import PlayerMechanismInputs
from monster.feature_compile.v13_identity_bridge import compile_v13_player_identity


def _inputs(*, active_probability: float, effectiveness_if_active: float) -> PlayerMechanismInputs:
    return PlayerMechanismInputs(
        active_probability=active_probability,
        effectiveness_if_active=effectiveness_if_active,
    )


def test_sampled_availability_does_not_double_penalize_active_player() -> None:
    low_availability = _inputs(active_probability=0.25, effectiveness_if_active=0.80)
    certain = _inputs(active_probability=1.0, effectiveness_if_active=0.80)

    low_identity, low_trace = compile_v13_player_identity(
        player_id="p",
        name="P",
        position="WR",
        usage_weight=1.0,
        inputs=low_availability,
        availability_already_sampled=True,
    )
    certain_identity, certain_trace = compile_v13_player_identity(
        player_id="p",
        name="P",
        position="WR",
        usage_weight=1.0,
        inputs=certain,
        availability_already_sampled=True,
    )

    assert low_identity.efficiency == certain_identity.efficiency
    assert low_trace.health == certain_trace.health == 0.80


def test_legacy_expected_state_path_remains_unchanged_until_promotion() -> None:
    low, _ = compile_v13_player_identity(
        player_id="p",
        name="P",
        position="WR",
        usage_weight=1.0,
        inputs=_inputs(active_probability=0.50, effectiveness_if_active=0.80),
    )
    certain, _ = compile_v13_player_identity(
        player_id="p",
        name="P",
        position="WR",
        usage_weight=1.0,
        inputs=_inputs(active_probability=1.0, effectiveness_if_active=0.80),
    )
    assert low.efficiency < certain.efficiency
