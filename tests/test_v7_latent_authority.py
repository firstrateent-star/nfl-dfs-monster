from __future__ import annotations

import pytest

from monster.reality.game_latents import sample_game_day_latents
from monster.reality.latent_authority import (
    LATENT_AUTHORITY,
    LatentMechanism,
    route_team_latent,
)
from monster.reality.world_state import WorldKey


def _latents():
    world = WorldKey(game_id="AWY@HME", world_index=0, seed=7006001)
    return sample_game_day_latents(world, away_team_id="AWY", home_team_id="HME")


def test_qb_execution_routes_only_to_declared_local_mechanism() -> None:
    routed = route_team_latent(
        _latents(),
        team_id="AWY",
        factor="qb_execution",
        mechanism=LatentMechanism.QB_READ,
        requested_authority=0.20,
    )
    assert routed.applied_authority == pytest.approx(0.20)
    assert routed.standardized_value == pytest.approx(
        _latents().value("AWY.qb_execution")
    )


def test_latent_authority_is_bounded_by_registry() -> None:
    routed = route_team_latent(
        _latents(),
        team_id="AWY",
        factor="pass_protection",
        mechanism=LatentMechanism.PASS_PROTECTION,
        requested_authority=0.95,
    )
    assert routed.applied_authority == LATENT_AUTHORITY["pass_protection"].max_authority
    assert routed.applied_authority < 1.0


def test_latent_cannot_cross_mechanism_jurisdiction() -> None:
    with pytest.raises(ValueError):
        route_team_latent(
            _latents(),
            team_id="AWY",
            factor="qb_execution",
            mechanism=LatentMechanism.FIELD_GOAL,
            requested_authority=0.10,
        )


def test_latent_registry_contains_no_score_or_fantasy_mechanism() -> None:
    names = {mechanism.value for spec in LATENT_AUTHORITY.values() for mechanism in spec.mechanisms}
    assert "score" not in names
    assert "points" not in names
    assert "fantasy" not in names
    assert "dfs" not in names
