from __future__ import annotations

import numpy as np

from monster.sim.defensive_intent import (
    CoverageShell,
    DefensiveTacticalPrior,
    RushPlan,
    sample_defensive_intent,
)


def test_shadow_defensive_intent_has_zero_authority() -> None:
    intent = sample_defensive_intent(DefensiveTacticalPrior(), rng=np.random.default_rng(11))
    assert intent.authority == 0.0
    assert intent.coverage_shell in set(CoverageShell)
    assert intent.rush_plan in set(RushPlan)


def test_short_yardage_increases_box_aggression_same_random_world() -> None:
    prior = DefensiveTacticalPrior()
    ordinary = sample_defensive_intent(prior, rng=np.random.default_rng(22), short_yardage=False)
    short = sample_defensive_intent(prior, rng=np.random.default_rng(22), short_yardage=True)
    assert short.coverage_shell == ordinary.coverage_shell
    assert short.rush_plan == ordinary.rush_plan
    assert short.box_aggression > ordinary.box_aggression


def test_defense_does_not_require_realized_offensive_play_call() -> None:
    # The public signature intentionally accepts strategic context only; a future production
    # bridge must preserve this simultaneous-information boundary.
    intent = sample_defensive_intent(
        DefensiveTacticalPrior(),
        rng=np.random.default_rng(33),
        late_lead=True,
        qb_run_threat=0.7,
    )
    assert 0.0 <= intent.box_aggression <= 1.0
