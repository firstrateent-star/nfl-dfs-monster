from __future__ import annotations

import numpy as np

from monster.sim.play_anatomy import (
    CLEAN_SCRAMBLE_RATE,
    PRESSURED_SACK_RATE,
    PRESSURED_SCRAMBLE_RATE,
    CatchpointResult,
    ContactResult,
    QBResponse,
    condition_throw_probabilities,
    resolve_catchpoint,
    resolve_qb_response,
    resolve_run_contact,
)


def test_clean_dropback_can_create_scramble_or_throw() -> None:
    rng = np.random.default_rng(1)
    outcomes = {
        resolve_qb_response(
            pressured=False,
            mobility=1.0,
            pocket_skill=1.0,
            rng=rng,
        )
        for _ in range(500)
    }
    assert QBResponse.SCRAMBLE in outcomes
    assert QBResponse.THROW in outcomes
    assert QBResponse.SACK not in outcomes


def test_neutral_qb_response_matches_empirical_branch_rates() -> None:
    n = 50_000
    pressure_rng = np.random.default_rng(11)
    pressured = [
        resolve_qb_response(
            pressured=True,
            mobility=1.0,
            pocket_skill=1.0,
            rng=pressure_rng,
        )
        for _ in range(n)
    ]
    clean_rng = np.random.default_rng(12)
    clean = [
        resolve_qb_response(
            pressured=False,
            mobility=1.0,
            pocket_skill=1.0,
            rng=clean_rng,
        )
        for _ in range(n)
    ]
    assert abs(pressured.count(QBResponse.SACK) / n - PRESSURED_SACK_RATE) < 0.01
    assert abs(pressured.count(QBResponse.SCRAMBLE) / n - PRESSURED_SCRAMBLE_RATE) < 0.01
    assert abs(clean.count(QBResponse.SCRAMBLE) / n - CLEAN_SCRAMBLE_RATE) < 0.01


def test_pocket_skill_reduces_sacks_and_mobility_increases_scrambles() -> None:
    n = 20_000
    weak_pocket_rng = np.random.default_rng(13)
    strong_pocket_rng = np.random.default_rng(13)
    weak_pocket = [
        resolve_qb_response(
            pressured=True,
            mobility=1.0,
            pocket_skill=0.80,
            rng=weak_pocket_rng,
        )
        for _ in range(n)
    ]
    strong_pocket = [
        resolve_qb_response(
            pressured=True,
            mobility=1.0,
            pocket_skill=1.20,
            rng=strong_pocket_rng,
        )
        for _ in range(n)
    ]
    assert strong_pocket.count(QBResponse.SACK) < weak_pocket.count(QBResponse.SACK)

    low_mobility_rng = np.random.default_rng(14)
    high_mobility_rng = np.random.default_rng(14)
    low_mobility = [
        resolve_qb_response(
            pressured=False,
            mobility=0.80,
            pocket_skill=1.0,
            rng=low_mobility_rng,
        )
        for _ in range(n)
    ]
    high_mobility = [
        resolve_qb_response(
            pressured=False,
            mobility=1.20,
            pocket_skill=1.0,
            rng=high_mobility_rng,
        )
        for _ in range(n)
    ]
    assert high_mobility.count(QBResponse.SCRAMBLE) > low_mobility.count(QBResponse.SCRAMBLE)


def test_pressure_can_create_sacks_scrambles_and_throws() -> None:
    rng = np.random.default_rng(2)
    outcomes = {
        resolve_qb_response(pressured=True, mobility=1.1, pocket_skill=1.0, rng=rng)
        for _ in range(200)
    }
    assert QBResponse.SACK in outcomes
    assert QBResponse.SCRAMBLE in outcomes
    assert QBResponse.THROW in outcomes


def test_pressure_conditioning_changes_throw_quality_without_moving_baseline() -> None:
    base_completion = 0.64
    base_interception = 0.022
    pressured_completion, pressured_interception = condition_throw_probabilities(
        completion_probability=base_completion,
        interception_probability=base_interception,
        pressured=True,
    )
    clean_completion, clean_interception = condition_throw_probabilities(
        completion_probability=base_completion,
        interception_probability=base_interception,
        pressured=False,
    )
    assert pressured_completion < base_completion < clean_completion
    assert pressured_interception > base_interception > clean_interception

    pressured_throw_share = 4207 / 17454
    clean_throw_share = 13247 / 17454
    weighted_completion = (
        pressured_throw_share * pressured_completion
        + clean_throw_share * clean_completion
    )
    weighted_interception = (
        pressured_throw_share * pressured_interception
        + clean_throw_share * clean_interception
    )
    assert abs(weighted_completion - base_completion) < 1e-6
    assert abs(weighted_interception - base_interception) < 1e-6


def test_catchpoint_has_catch_drop_breakup_and_interception_paths() -> None:
    rng = np.random.default_rng(3)
    outcomes = {
        resolve_catchpoint(
            catch_skill=1.0,
            coverage_strength=1.0,
            ball_hawk=1.0,
            air_yards=12.0,
            rng=rng,
        )
        for _ in range(1200)
    }
    assert CatchpointResult.CATCH in outcomes
    assert CatchpointResult.DROP in outcomes
    assert CatchpointResult.BREAKUP in outcomes
    assert CatchpointResult.INTERCEPTION in outcomes


def test_matchup_probabilities_govern_catchpoint_outcomes() -> None:
    n = 50_000
    rng = np.random.default_rng(31)
    outcomes = [
        resolve_catchpoint(
            catch_skill=1.25,
            coverage_strength=0.75,
            ball_hawk=1.25,
            air_yards=8.0,
            completion_probability=0.62,
            interception_probability=0.025,
            rng=rng,
        )
        for _ in range(n)
    ]
    catch_rate = outcomes.count(CatchpointResult.CATCH) / n
    interception_rate = outcomes.count(CatchpointResult.INTERCEPTION) / n
    assert abs(catch_rate - 0.62) < 0.01
    assert abs(interception_rate - 0.025) < 0.005


def test_supplied_matchup_probabilities_override_legacy_trait_path() -> None:
    n = 20_000
    low_rng = np.random.default_rng(32)
    high_rng = np.random.default_rng(32)
    low = [
        resolve_catchpoint(
            catch_skill=1.0,
            coverage_strength=1.0,
            ball_hawk=1.0,
            air_yards=8.0,
            completion_probability=0.45,
            interception_probability=0.01,
            rng=low_rng,
        )
        for _ in range(n)
    ]
    high = [
        resolve_catchpoint(
            catch_skill=1.0,
            coverage_strength=1.0,
            ball_hawk=1.0,
            air_yards=8.0,
            completion_probability=0.75,
            interception_probability=0.05,
            rng=high_rng,
        )
        for _ in range(n)
    ]
    assert high.count(CatchpointResult.CATCH) > low.count(CatchpointResult.CATCH)
    assert high.count(CatchpointResult.INTERCEPTION) > low.count(CatchpointResult.INTERCEPTION)


def test_run_contact_can_stuff_tackle_or_break_tackle() -> None:
    rng = np.random.default_rng(4)
    outcomes = {
        resolve_run_contact(
            penetration_probability=0.25,
            runner_power=1.0,
            tackling=1.0,
            explosiveness=1.0,
            rng=rng,
        ).contact
        for _ in range(800)
    }
    assert ContactResult.STUFF in outcomes
    assert ContactResult.TACKLED in outcomes
    assert ContactResult.BROKEN_TACKLE in outcomes


def test_stronger_runner_increases_broken_tackle_frequency() -> None:
    weak_rng = np.random.default_rng(5)
    strong_rng = np.random.default_rng(5)
    weak = [
        resolve_run_contact(
            penetration_probability=0.15,
            runner_power=0.75,
            tackling=1.1,
            explosiveness=1.0,
            rng=weak_rng,
        ).contact
        for _ in range(3000)
    ]
    strong = [
        resolve_run_contact(
            penetration_probability=0.15,
            runner_power=1.25,
            tackling=0.9,
            explosiveness=1.0,
            rng=strong_rng,
        ).contact
        for _ in range(3000)
    ]
    assert strong.count(ContactResult.BROKEN_TACKLE) > weak.count(ContactResult.BROKEN_TACKLE)
