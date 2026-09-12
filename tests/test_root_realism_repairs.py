from __future__ import annotations

from collections import Counter

import numpy as np

from monster.sim.football_state import FootballState
from monster.sim.intent_ecology import RunGeometryOutcome
from monster.sim.matchup_kernel import DefensiveUnit
from monster.sim.play_kernel import (
    PlayType,
    PlayerIdentity,
    TeamIdentity,
    _field_read_target,
    choose_play_type,
)
from monster.sim.resolution_ecology import resolve_run_ecology


def _team(*, qb_efficiency: float = 1.0) -> TeamIdentity:
    quarterback = PlayerIdentity(
        "qb",
        "QB",
        "QB",
        usage_weight=1.0,
        efficiency=qb_efficiency,
        explosive=1.0,
    )
    rusher = PlayerIdentity("rb", "RB", "RB", usage_weight=1.0)
    receiver = PlayerIdentity("wr", "WR", "WR", usage_weight=1.0)
    return TeamIdentity(
        team_id="away",
        quarterback=quarterback,
        rushers=(rusher,),
        receivers=(receiver,),
        neutral_pass_rate=0.56,
        pass_efficiency=1.0,
    )


def test_qb_player_efficiency_reaches_pass_matchup() -> None:
    defense = DefensiveUnit(front=(), coverage=())
    low_target, low = _field_read_target(
        _team(qb_efficiency=0.80),
        defense,
        np.random.default_rng(11),
        fatigue={},
    )
    high_target, high = _field_read_target(
        _team(qb_efficiency=1.20),
        defense,
        np.random.default_rng(11),
        fatigue={},
    )

    assert low_target.player_id == high_target.player_id == "wr"
    assert high.qb_read_quality > low.qb_read_quality
    assert high.completion_probability > low.completion_probability
    assert high.interception_probability < low.interception_probability


def test_fourth_down_runtime_samples_empirical_decision_mix() -> None:
    state = FootballState(
        possession="away",
        defense="home",
        yardline_100=65.0,
        down=4,
        distance=2.0,
    )
    team = _team()
    decisions = Counter(
        choose_play_type(state, team, np.random.default_rng(seed))
        for seed in range(600)
    )
    go = decisions[PlayType.RUN] + decisions[PlayType.PASS]

    # This state has a 2025 prior of 70.8% go, 27.0% FG and 2.2% punt. We intentionally
    # use broad deterministic gates: the invariant is that the live runtime samples the
    # distribution rather than collapsing the state to its modal GO decision.
    assert go > 350
    assert decisions[PlayType.FIELD_GOAL] > 100
    assert decisions[PlayType.PUNT] > 0


def test_routine_run_sampler_has_no_winsorized_boundary_mass() -> None:
    profile = RunGeometryOutcome(
        category="interior",
        attempts=5000,
        yards_mean=4.20,
        yards_sd=4.80,
        negative_rate=0.12,
        zero_rate=0.03,
        loss_2_plus_rate=0.055,
        loss_5_plus_rate=0.012,
        explosive_10_rate=0.105,
        explosive_15_rate=0.050,
        explosive_20_rate=0.025,
        touchdown_rate=0.025,
        fumble_lost_rate=0.008,
        yards_p10=-1.0,
        yards_p50=4.0,
        yards_p90=10.0,
        yards_p99=28.0,
    )
    rng = np.random.default_rng(2026190921)
    values = np.asarray(
        [
            resolve_run_ecology(
                profile,
                matchup_stuff_probability=0.18,
                matchup_yards_multiplier=1.0,
                runner_power=1.0,
                tackling=1.0,
                explosiveness=1.0,
                rng=rng,
            ).total_yards
            for _ in range(20_000)
        ]
    )

    assert not np.any(values == 0.1)
    assert not np.any(values == 9.999)
    assert abs(float(values.mean()) - profile.yards_mean) < 0.35
