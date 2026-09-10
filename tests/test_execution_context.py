import pytest

from monster.sim.execution_context import (
    ExecutionContext,
    ExecutionMechanism,
    factors_for_mechanism,
)


def test_catchpoint_cannot_see_unit_or_venue_noise() -> None:
    context = ExecutionContext(
        individual_matchup=0.4,
        unit_matchup=0.9,
        concept_difficulty=0.3,
        weather_severity=0.2,
        venue_noise=0.8,
    )
    factors = factors_for_mechanism(context, ExecutionMechanism.CATCHPOINT)
    assert factors["individual_matchup"] == 0.4
    assert factors["concept_difficulty"] == 0.3
    assert factors["weather_severity"] == 0.2
    assert "unit_matchup" not in factors
    assert "venue_noise" not in factors


def test_pass_protection_can_see_trench_and_communication_context() -> None:
    context = ExecutionContext(
        individual_matchup=-0.2,
        unit_matchup=0.5,
        offense_scheme_fit=0.3,
        defense_counter_fit=-0.4,
        venue_noise=0.6,
        fatigue=0.2,
    )
    factors = factors_for_mechanism(context, ExecutionMechanism.PASS_PROTECTION)
    assert factors["unit_matchup"] == 0.5
    assert factors["offense_scheme_fit"] == 0.3
    assert factors["defense_counter_fit"] == -0.4
    assert factors["venue_noise"] == 0.6
    assert factors["fatigue"] == 0.2


def test_ball_security_does_not_receive_scheme_or_venue_by_default() -> None:
    context = ExecutionContext(
        defense_counter_fit=0.9,
        venue_noise=0.9,
        weather_severity=0.7,
        fatigue=0.4,
    )
    factors = factors_for_mechanism(context, ExecutionMechanism.BALL_SECURITY)
    assert factors["weather_severity"] == 0.7
    assert factors["fatigue"] == 0.4
    assert "defense_counter_fit" not in factors
    assert "venue_noise" not in factors


def test_execution_signals_are_bounded_evidence_not_probabilities() -> None:
    with pytest.raises(ValueError):
        ExecutionContext(individual_matchup=1.01)
