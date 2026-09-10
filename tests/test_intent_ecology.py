from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from monster.sim.categorical_policy import CategoryEvidence, shrink_distribution
from monster.sim.game_flow import GameFlowState
from monster.sim.intent_ecology import (
    IntentEcology,
    PassDepthOutcome,
    choose_rusher_for_geometry,
    choose_target_for_depth,
    feasible_pass_probabilities,
    sample_air_yards,
)


def _flow(*, yardline: float = 50.0) -> GameFlowState:
    return GameFlowState(
        down=2,
        distance=6.0,
        yardline=yardline,
        quarter=2,
        seconds_remaining=1800,
        seconds_remaining_in_period=900,
        score_margin=0,
        yards_to_goal=max(100.0 - yardline, 0.0),
        tags=frozenset(),
    )


def test_sparse_categorical_child_shrinks_toward_parent() -> None:
    parent = np.asarray([0.5, 0.5])
    child = CategoryEvidence({"a": 1, "b": 0}, 1)
    shrunk = shrink_distribution(parent, child, categories=("a", "b"), shrinkage_samples=99)
    assert np.allclose(shrunk, [0.505, 0.495])


def test_pass_geometry_masks_physically_impossible_bombs() -> None:
    class Policy:
        categories = (
            "behind_los",
            "short_0_5",
            "short_6_9",
            "intermediate_10_19",
            "deep_20_39",
            "bomb_40_plus",
        )

        def probabilities_for(self, flow, *, actor_id=None):
            return np.asarray([0.05, 0.10, 0.10, 0.15, 0.20, 0.40])

    ecology = IntentEcology(
        pass_depth=Policy(),  # type: ignore[arg-type]
        run_geometry=Policy(),  # type: ignore[arg-type]
        pass_outcomes={},
        run_outcomes={},
        target_depth_attempts={},
        rusher_geometry_attempts={},
    )
    probs = feasible_pass_probabilities(ecology, _flow(yardline=95.0), quarterback_id="QB")
    assert probs[-1] == 0.0
    assert probs[-2] == 0.0
    assert np.isclose(probs.sum(), 1.0)


def test_signed_air_yards_preserve_behind_los_geometry() -> None:
    profile = PassDepthOutcome(
        category="behind_los",
        attempts=100,
        completion_rate=0.8,
        interception_rate=0.005,
        touchdown_rate=0.01,
        air_yards_mean=-3.5,
        air_yards_sd=1.0,
        yac_mean_completed=5.0,
        yac_sd_completed=3.0,
        negative_completion_rate=0.03,
        zero_completion_rate=0.02,
    )
    rng = np.random.default_rng(7)
    values = [sample_air_yards("behind_los", profile, yards_to_goal=50.0, rng=rng) for _ in range(100)]
    assert max(values) < 0.0
    assert min(values) >= -12.0


def test_target_and_rusher_compatibility_can_refine_but_not_create_players() -> None:
    players = [
        SimpleNamespace(player_id="A", usage_weight=0.7),
        SimpleNamespace(player_id="B", usage_weight=0.3),
    ]

    class Policy:
        categories = ("x",)

        def probabilities_for(self, flow, *, actor_id=None):
            return np.asarray([1.0])

        def sample(self, flow, rng, *, actor_id=None):
            return "x"

    ecology = IntentEcology(
        pass_depth=Policy(),  # type: ignore[arg-type]
        run_geometry=Policy(),  # type: ignore[arg-type]
        pass_outcomes={},
        run_outcomes={},
        target_depth_attempts={("B", "x"): 100},
        rusher_geometry_attempts={("B", "x"): 100},
    )
    rng = np.random.default_rng(11)
    targets = [choose_target_for_depth(players, ecology, "x", rng).player_id for _ in range(300)]
    rushers = [choose_rusher_for_geometry(players, ecology, "x", rng).player_id for _ in range(300)]
    assert targets.count("B") > targets.count("A")
    assert rushers.count("B") > rushers.count("A")
    assert set(targets) == {"A", "B"}
    assert set(rushers) == {"A", "B"}
