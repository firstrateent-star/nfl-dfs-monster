from __future__ import annotations

import numpy as np

from monster.sim.chaos_ecology import DEFAULT_CHAOS_ECOLOGY, sample_return_yards
from monster.sim.play_kernel import PlayerIdentity, TeamIdentity
from monster.sim.reality_v63 import (
    GameEnvironmentV63,
    apply_chaos_environment_v63,
    cadence_seconds_v63,
    role_aware_offense_skill_players,
    sample_game_environment_v63,
    sample_return_yards_v63,
    set_active_game_environment,
    snap_presence_weights,
)


def _dual_role_offense() -> TeamIdentity:
    lead_receiver = PlayerIdentity("lead", "Lead Back", "RB", usage_weight=0.06)
    backup_receiver = PlayerIdentity("backup", "Backup Back", "RB", usage_weight=0.18)
    receivers = (
        lead_receiver,
        backup_receiver,
        PlayerIdentity("wr1", "WR1", "WR", usage_weight=0.42),
        PlayerIdentity("wr2", "WR2", "WR", usage_weight=0.27),
        PlayerIdentity("wr3", "WR3", "WR", usage_weight=0.18),
        PlayerIdentity("te1", "TE1", "TE", usage_weight=0.22),
        PlayerIdentity("te2", "TE2", "TE", usage_weight=0.08),
    )
    rushers = (
        PlayerIdentity("lead", "Lead Back", "RB", usage_weight=0.70),
        PlayerIdentity("backup", "Backup Back", "RB", usage_weight=0.20),
    )
    return TeamIdentity(
        team_id="TST",
        quarterback=PlayerIdentity("qb", "QB", "QB"),
        rushers=rushers,
        receivers=receivers,
    )


def test_dual_role_back_keeps_rushing_authority_in_snap_presence_weight():
    weights = snap_presence_weights(_dual_role_offense())
    assert weights["lead"] > weights["backup"] * 2.0
    assert weights["lead"] > 0.50


def test_feature_back_is_actually_on_field_more_often_after_role_merge():
    offense = _dual_role_offense()
    lead = 0
    backup = 0
    trials = 600
    for index in range(trials):
        selected = role_aware_offense_skill_players(offense, "11", f"snap:{index}")
        ids = {player.player_id for player in selected}
        lead += int("lead" in ids)
        backup += int("backup" in ids)
    assert lead / trials > 0.68
    assert lead > backup * 1.8


def test_v63_environment_is_centered_and_wider_in_real_mechanisms():
    environments = [sample_game_environment_v63(seed) for seed in range(6300, 10300)]
    away = np.asarray([item.away_execution for item in environments])
    explosive = np.asarray([item.explosive_factor for item in environments])
    tempo = np.asarray([item.tempo_factor for item in environments])
    assert 0.97 < away.mean() < 1.03
    assert 0.97 < explosive.mean() < 1.03
    assert 0.98 < tempo.mean() < 1.02
    assert np.quantile(away, 0.05) < 0.87
    assert np.quantile(away, 0.95) > 1.13
    assert np.quantile(explosive, 0.05) < 0.88
    assert np.quantile(explosive, 0.95) > 1.12
    assert all(0.032 <= item.penalty_rate <= 0.120 for item in environments)


def test_persistent_tempo_changes_clock_cadence_not_points():
    fast = GameEnvironmentV63(1.0, 1.0, 1.0, 0.07, 1.0, 1.0, 1.15)
    slow = GameEnvironmentV63(1.0, 1.0, 1.0, 0.07, 1.0, 1.0, 0.87)
    set_active_game_environment(fast)
    fast_seconds = cadence_seconds_v63(35)
    set_active_game_environment(slow)
    slow_seconds = cadence_seconds_v63(35)
    set_active_game_environment(None)
    assert fast_seconds < 35 < slow_seconds
    assert not hasattr(fast, "points")


def test_chaos_environment_changes_live_ball_geometry_not_touchdowns_directly():
    high = GameEnvironmentV63(1.0, 1.0, 1.0, 0.07, 1.80, 1.15, 1.0)
    low = GameEnvironmentV63(1.0, 1.0, 1.0, 0.07, 0.60, 0.90, 1.0)
    high_ecology = apply_chaos_environment_v63(DEFAULT_CHAOS_ECOLOGY, high)
    low_ecology = apply_chaos_environment_v63(DEFAULT_CHAOS_ECOLOGY, low)
    assert high_ecology.interception_40_plus_rate > low_ecology.interception_40_plus_rate
    assert high_ecology.interception_zero_return_rate < low_ecology.interception_zero_return_rate
    assert not hasattr(high_ecology, "touchdown_rate")


def test_breakaway_return_branch_has_more_far_continuation_than_v62_shape():
    set_active_game_environment(None)
    base_rng = np.random.default_rng(63031)
    v63_rng = np.random.default_rng(63031)
    base = np.asarray(
        [
            sample_return_yards(
                mean=11.5,
                sd=12.0,
                zero_rate=0.22,
                forty_plus_rate=0.045,
                return_skill=1.0,
                rng=base_rng,
                maximum=100.0,
            )
            for _ in range(12000)
        ]
    )
    candidate = np.asarray(
        [
            sample_return_yards_v63(
                mean=11.5,
                sd=12.0,
                zero_rate=0.22,
                forty_plus_rate=0.045,
                return_skill=1.0,
                rng=v63_rng,
                maximum=100.0,
            )
            for _ in range(12000)
        ]
    )
    assert np.mean(candidate >= 60.0) > np.mean(base >= 60.0) * 1.15
    assert np.quantile(candidate, 0.50) < 15.0
