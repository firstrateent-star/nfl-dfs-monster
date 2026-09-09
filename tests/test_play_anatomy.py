from __future__ import annotations

import numpy as np

from monster.sim.play_anatomy import (
    CatchpointResult,
    ContactResult,
    QBResponse,
    resolve_catchpoint,
    resolve_qb_response,
    resolve_run_contact,
)


def test_no_pressure_produces_throw_response() -> None:
    assert resolve_qb_response(
        pressured=False, mobility=1.0, pocket_skill=1.0, rng=np.random.default_rng(1)
    ) == QBResponse.THROW


def test_pressure_can_create_sacks_scrambles_and_throws() -> None:
    rng = np.random.default_rng(2)
    outcomes = {
        resolve_qb_response(pressured=True, mobility=1.1, pocket_skill=1.0, rng=rng)
        for _ in range(200)
    }
    assert QBResponse.SACK in outcomes
    assert QBResponse.SCRAMBLE in outcomes
    assert QBResponse.THROW in outcomes


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
