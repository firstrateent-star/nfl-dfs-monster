from __future__ import annotations

from types import SimpleNamespace

import pytest

from monster.sim.opportunity_skill_tail_v3 import (
    OpportunitySkillTailV3,
    opportunity_skill_multiplier,
    relative_opportunity,
)


def _player(*, player_id: str, usage: float, skill: float):
    return SimpleNamespace(
        player_id=player_id,
        name=player_id,
        position="WR",
        usage_weight=usage,
        speed_skill=skill,
        route_separation_skill=skill,
        open_field_skill=skill,
        catchpoint_skill=skill,
        rush_creation_skill=skill,
        runner_power_skill=skill,
    )


def test_usage_alone_cannot_create_tail_authority() -> None:
    neutral = _player(player_id="neutral", usage=0.90, skill=0.0)
    assert opportunity_skill_multiplier(
        neutral,
        family="receiving",
        relative_opportunity_value=4.0,
    ) == pytest.approx(1.0)
    assert opportunity_skill_multiplier(
        neutral,
        family="designed_run",
        relative_opportunity_value=0.25,
    ) == pytest.approx(1.0)


def test_featured_role_amplifies_signed_skill_not_usage_itself() -> None:
    elite = _player(player_id="elite", usage=0.50, skill=0.70)
    constrained = _player(player_id="constrained", usage=0.50, skill=-0.70)

    elite_low = opportunity_skill_multiplier(
        elite,
        family="receiving",
        relative_opportunity_value=0.50,
    )
    elite_high = opportunity_skill_multiplier(
        elite,
        family="receiving",
        relative_opportunity_value=2.50,
    )
    constrained_low = opportunity_skill_multiplier(
        constrained,
        family="receiving",
        relative_opportunity_value=0.50,
    )
    constrained_high = opportunity_skill_multiplier(
        constrained,
        family="receiving",
        relative_opportunity_value=2.50,
    )

    assert elite_high > elite_low > 1.0
    assert constrained_high < constrained_low < 1.0


def test_relative_opportunity_is_team_rotation_relative() -> None:
    featured = _player(player_id="featured", usage=0.40, skill=0.50)
    role = _player(player_id="role", usage=0.10, skill=0.50)
    pool = (featured, role)

    assert relative_opportunity(featured, pool) > 1.0
    assert relative_opportunity(role, pool) < 1.0

    engine = OpportunitySkillTailV3()
    engine.register_pass(featured, pool)
    featured_multiplier = engine.pass_multiplier()
    engine.register_pass(role, pool)
    role_multiplier = engine.pass_multiplier()
    assert featured_multiplier > role_multiplier > 1.0
