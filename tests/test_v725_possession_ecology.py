from __future__ import annotations

import numpy as np
import polars as pl

from monster.feature_compile.chaos_priors import compile_chaos_ecology
from monster.sim.chaos_ecology import DEFAULT_CHAOS_ECOLOGY
from monster.sim.possession_ecology_v725 import (
    active_field_goal_ecology_v725,
    configure_possession_ecology_v725,
    simulate_field_goal_v725,
)


def test_v725_field_goal_policy_compiles_unblocked_distance_evidence() -> None:
    pbp = pl.DataFrame(
        {
            "play_type": ["field_goal"] * 6,
            "field_goal_attempt": [1] * 6,
            "field_goal_result": ["made", "made", "missed", "blocked", "made", "missed"],
            "kick_distance": [45.0, 46.0, 47.0, 48.0, 55.0, 56.0],
        }
    )
    row = compile_chaos_ecology(pbp).to_dicts()[0]
    assert row["v725_fg_40_49_attempts"] == 4
    assert row["v725_fg_40_49_unblocked_attempts"] == 3
    assert 0.0 < row["v725_fg_40_49_make_rate_unblocked"] < 1.0


def test_v725_field_goal_runtime_consumes_compiled_policy() -> None:
    configure_possession_ecology_v725(
        {
            "v725_fg_40_49_attempts": 300,
            "v725_fg_40_49_unblocked_attempts": 295,
            "v725_fg_40_49_make_rate_unblocked": 0.81,
        }
    )
    bucket = active_field_goal_ecology_v725().buckets[2]
    assert bucket.make_rate_unblocked == 0.81
    event = simulate_field_goal_v725(
        np.random.default_rng(725),
        distance=45.0,
        kicking_skill=1.0,
        ecology=DEFAULT_CHAOS_ECOLOGY,
    )
    assert event.kick_distance == 45.0
    assert event.made in {True, False}


def test_v725_missing_policy_keeps_safe_field_goal_defaults() -> None:
    configure_possession_ecology_v725(None)
    buckets = active_field_goal_ecology_v725().buckets
    assert buckets[0].make_rate_unblocked == 0.985
    assert buckets[-1].make_rate_unblocked == 0.480
