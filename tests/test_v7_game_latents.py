from __future__ import annotations

import numpy as np
import pytest

from monster.reality.game_latents import (
    sample_game_day_latents,
    team_latent_vector,
)
from monster.reality.world_state import WorldKey


def _world(index: int, seed: int) -> WorldKey:
    return WorldKey(game_id="AWY@HME", world_index=index, seed=seed)


def test_game_day_latents_are_deterministic_for_world_key() -> None:
    first = sample_game_day_latents(_world(4, 7004001), away_team_id="AWY", home_team_id="HME")
    second = sample_game_day_latents(_world(4, 7004001), away_team_id="AWY", home_team_id="HME")
    assert first == second


def test_distinct_world_indices_get_distinct_latent_conditions() -> None:
    first = sample_game_day_latents(_world(4, 7004001), away_team_id="AWY", home_team_id="HME")
    second = sample_game_day_latents(_world(5, 7004001), away_team_id="AWY", home_team_id="HME")
    assert first != second


def test_latent_world_has_complete_team_mechanism_vectors() -> None:
    latents = sample_game_day_latents(_world(0, 7004002), away_team_id="AWY", home_team_id="HME")
    away = team_latent_vector(latents, "AWY")
    home = team_latent_vector(latents, "HME")
    expected = {
        "qb_execution",
        "pass_protection",
        "receiver_execution",
        "run_blocking",
        "ballcarrier_execution",
        "pass_rush",
        "coverage_execution",
        "run_fit",
        "tackling",
        "special_teams_execution",
    }
    assert set(away) == expected
    assert set(home) == expected
    assert all(-2.75 <= value <= 2.75 for value in away.values())
    assert all(-2.75 <= value <= 2.75 for value in home.values())


def test_latent_sampler_creates_positive_within_unit_dependence_without_score_output() -> None:
    qb: list[float] = []
    protection: list[float] = []
    receiver: list[float] = []
    pass_rush: list[float] = []
    coverage: list[float] = []

    for index in range(600):
        latents = sample_game_day_latents(
            _world(index, 7004003),
            away_team_id="AWY",
            home_team_id="HME",
        )
        away = team_latent_vector(latents, "AWY")
        qb.append(away["qb_execution"])
        protection.append(away["pass_protection"])
        receiver.append(away["receiver_execution"])
        pass_rush.append(away["pass_rush"])
        coverage.append(away["coverage_execution"])

    assert np.corrcoef(qb, protection)[0, 1] > 0.25
    assert np.corrcoef(qb, receiver)[0, 1] > 0.25
    assert np.corrcoef(pass_rush, coverage)[0, 1] > 0.25


def test_latent_sampler_rejects_world_team_mismatch() -> None:
    with pytest.raises(ValueError):
        sample_game_day_latents(_world(0, 7004004), away_team_id="XXX", home_team_id="HME")
