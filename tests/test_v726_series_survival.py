from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from monster.dfs.fanduel import FANDUEL_KICKER_SCORING, score_kicker_player_worlds
from monster.reality.special_teams_identity_v72 import (
    configure_base_special_teams_hooks_v72,
    configure_special_teams_identities_v72,
    simulate_try_v72,
)
from monster.sim.game_loop_v13 import PlayerBoxScore, _record_kicking_events
from monster.sim.intent_ecology import PassDepthOutcome, RunGeometryOutcome
from monster.sim.rules_v13 import TryEvent, TryResult, simulate_try
from monster.sim.special_teams_v13 import (
    SpecialTeamsEvent,
    SpecialTeamsType,
    simulate_field_goal,
    simulate_kickoff,
    simulate_punt,
)
from monster.sim.resolution_ecology import resolve_run_ecology, sample_yac


def _run_profile() -> RunGeometryOutcome:
    return RunGeometryOutcome(
        category="interior",
        attempts=5000,
        yards_mean=4.25,
        yards_sd=5.1,
        negative_rate=0.17,
        zero_rate=0.08,
        loss_2_plus_rate=0.07,
        loss_5_plus_rate=0.015,
        gain_5plus_rate=0.34,
        explosive_10_rate=0.105,
        explosive_15_rate=0.047,
        explosive_20_rate=0.024,
        touchdown_rate=0.03,
        fumble_lost_rate=0.006,
        yards_p10=-1.0,
        yards_p50=3.0,
        yards_p90=9.0,
        yards_p99=26.0,
        explosive_40_rate=0.004,
        yards_40plus_mean=48.0,
        early_down_attempts=3200,
        early_down_yards_mean=4.45,
        early_down_yards_sd=5.0,
        early_down_negative_rate=0.145,
        early_down_zero_rate=0.07,
        early_down_gain_5plus_rate=0.39,
        early_down_explosive_10_rate=0.11,
        early_down_explosive_15_rate=0.05,
        early_down_explosive_20_rate=0.025,
    )


def _pass_profile() -> PassDepthOutcome:
    return PassDepthOutcome(
        category="behind_los",
        attempts=1500,
        completion_rate=0.83,
        interception_rate=0.004,
        touchdown_rate=0.025,
        air_yards_mean=-1.8,
        air_yards_sd=1.1,
        yac_mean_completed=5.0,
        yac_sd_completed=4.5,
        negative_completion_rate=0.24,
        zero_completion_rate=0.08,
        yards_mean_completed=3.2,
        yards_sd_completed=5.0,
        gain_5plus_completion_rate=0.33,
        gain_10plus_completion_rate=0.12,
        gain_15plus_completion_rate=0.05,
        gain_20plus_completion_rate=0.025,
        early_down_attempts=1050,
        early_down_completion_attempts=870,
        early_down_negative_completion_rate=0.13,
        early_down_zero_completion_rate=0.07,
    )


def test_v726_early_down_run_branch_preserves_five_plus_frequency() -> None:
    rng = np.random.default_rng(72601)
    profile = _run_profile()
    samples = [
        resolve_run_ecology(
            profile,
            matchup_stuff_probability=0.18,
            matchup_yards_multiplier=1.0,
            runner_power=1.0,
            tackling=1.0,
            explosiveness=1.0,
            rng=rng,
            early_down=True,
        ).total_yards
        for _ in range(8000)
    ]
    gain5 = sum(yards >= 5.0 for yards in samples) / len(samples)
    negative = sum(yards < 0.0 for yards in samples) / len(samples)

    assert 0.35 <= gain5 <= 0.43
    assert 0.11 <= negative <= 0.18


def test_v726_early_down_screen_losses_use_early_down_prior() -> None:
    profile = _pass_profile()
    rng_early = np.random.default_rng(72602)
    rng_all = np.random.default_rng(72603)

    def negative_rate(rng: np.random.Generator, early_down: bool) -> float:
        negatives = 0
        samples = 7000
        for _ in range(samples):
            air = -2.0
            yac = sample_yac(
                profile,
                receiver_explosiveness=1.0,
                coverage_strength=1.0,
                rng=rng,
                air_yards=air,
                early_down=early_down,
            )
            negatives += int(air + yac < 0.0)
        return negatives / samples

    early = negative_rate(rng_early, True)
    all_downs = negative_rate(rng_all, False)
    assert early < all_downs
    assert 0.12 <= early <= 0.19
    assert 0.20 <= all_downs <= 0.28


def test_v726_non_early_run_path_remains_available() -> None:
    rng = np.random.default_rng(72604)
    outcome = resolve_run_ecology(
        _run_profile(),
        matchup_stuff_probability=0.18,
        matchup_yards_multiplier=1.0,
        runner_power=1.0,
        tackling=1.0,
        explosiveness=1.0,
        rng=rng,
        early_down=False,
    )
    assert np.isfinite(outcome.total_yards)



def test_v726_kicker_events_survive_into_player_box_and_downstream_scoring() -> None:
    stats: dict[str, PlayerBoxScore] = {}
    special = [
        SpecialTeamsEvent(
            SpecialTeamsType.FIELD_GOAL,
            kick_distance=37.0,
            made=True,
            kicker_id="k",
        ),
        SpecialTeamsEvent(
            SpecialTeamsType.FIELD_GOAL,
            kick_distance=47.0,
            made=True,
            kicker_id="k",
        ),
        SpecialTeamsEvent(
            SpecialTeamsType.FIELD_GOAL,
            kick_distance=55.0,
            made=True,
            kicker_id="k",
        ),
        SpecialTeamsEvent(
            SpecialTeamsType.FIELD_GOAL,
            kick_distance=52.0,
            made=False,
            kicker_id="k",
        ),
    ]
    tries = [
        TryEvent(
            TryResult.PAT_GOOD,
            1,
            kicking_team_id="A",
            kicker_id="k",
            kick_distance=33.0,
        ),
        TryEvent(
            TryResult.PAT_MISS,
            0,
            kicking_team_id="A",
            kicker_id="k",
            kick_distance=33.0,
        ),
    ]

    _record_kicking_events(stats, special, tries)
    box = stats["k"]
    assert box.field_goal_attempts == 4
    assert box.field_goals_made == 3
    assert box.field_goals_made_0_39 == 1
    assert box.field_goals_made_40_49 == 1
    assert box.field_goals_made_50_plus == 1
    assert box.extra_point_attempts == 2
    assert box.extra_points_made == 1

    fd = score_kicker_player_worlds(
        {
            stat: np.asarray([float(getattr(box, stat))])
            for stat in FANDUEL_KICKER_SCORING
        }
    )
    assert fd[0] == pytest.approx(13.0)


def test_v726_pat_runtime_assigns_current_kicker_identity() -> None:
    personnel = pl.DataFrame(
        [
            {
                "gsis_id": "k",
                "team_id": "A",
                "position": "K",
                "depth_position": "K",
                "depth_rank": 1,
                "conditional_special_teams_snap_share": 0.90,
                "game_day_active_probability": 1.0,
                "madden_kick_power": 90.0,
                "madden_kick_accuracy": 92.0,
            }
        ]
    )
    configure_special_teams_identities_v72(personnel)
    configure_base_special_teams_hooks_v72(
        punt=simulate_punt,
        field_goal=simulate_field_goal,
        kickoff=simulate_kickoff,
        kickoff_loop=lambda state, *args, **kwargs: state,
        try_play=simulate_try,
    )
    event = simulate_try_v72(
        np.random.default_rng(72605),
        go_for_two=False,
        kicking_team_id="A",
        kicking_skill=1.0,
        offense_skill=1.0,
        defense_skill=1.0,
    )
    assert event.kicking_team_id == "A"
    assert event.kicker_id == "k"
    assert event.kick_distance == 33.0
    assert event.result in {TryResult.PAT_GOOD, TryResult.PAT_MISS}
