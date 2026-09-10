from __future__ import annotations

from monster.sim.system_identity import shadow_system_identity


def test_shadow_system_identity_cannot_double_count_team_policy() -> None:
    identity = shadow_system_identity("T", continuity=0.72, head_coach_id="coach")
    assert identity.production_authority == 0.0
    assert identity.play_call_authority == 0.0
    assert identity.fourth_down_authority == 0.0


def test_change_uncertainty_increases_when_continuity_falls() -> None:
    stable = shadow_system_identity("T", continuity=0.9)
    changed = shadow_system_identity("T", continuity=0.2)
    assert changed.offensive_change_uncertainty > stable.offensive_change_uncertainty
    assert changed.defensive_change_uncertainty > stable.defensive_change_uncertainty
