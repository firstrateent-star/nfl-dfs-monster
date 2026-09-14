from __future__ import annotations

import numpy as np

from monster.sim.chaos_ecology import sample_return_yards
from monster.sim.play_kernel import PlayerIdentity, TeamIdentity
from monster.sim.reality_v63 import (
    GameEnvironmentV63,
    cadence_seconds_v63,
    role_aware_offense_skill_players,
    sample_game_environment_v63,
    sample_return_yards_v63,
    set_active_game_environment,
    snap_presence_weights,
)


def _offense() -> TeamIdentity:
    qb = PlayerIdentity("q", "Q", "QB", usage_weight=0.10)
    lead_rush = PlayerIdentity("lead", "Lead", "RB", usage_weight=0.72)
    pass_rush = PlayerIdentity("pass", "Pass", "RB", usage_weight=0.18)
    lead_recv = PlayerIdentity("lead", "Lead", "RB", usage_weight=0.07)
    pass_recv = PlayerIdentity("pass", "Pass", "RB", usage_weight=0.22)
    return TeamIdentity(
        team_id="TST",
        quarterback=qb,
        rushers=(lead_rush, pass_rush, qb),
        receivers=(
            lead_recv,
            pass_recv,
            PlayerIdentity("te", "TE", "TE", usage_weight=0.16),
            PlayerIdentity("w1", "W1", "WR", usage_weight=0.27),
            PlayerIdentity("w2", "W2", "WR", usage_weight=0.18),
            PlayerIdentity("w3", "W3", "WR", usage_weight=0.10),
        ),
    )


def test_dual_role_rb_presence_uses_rush_and_target_roles():
    weights = snap_presence_weights(_offense())
    assert weights["lead"] > 2.0 * weights["pass"]


def test_lead_back_wins_one_rb_snap_seat_more_often():
    offense = _offense()
    lead = 0
    receiving = 0
    for index in range(1200):
        selected = role_aware_offense_skill_players(offense, "11", f"snap-{index}")
        ids = {player.player_id for player in selected}
        lead += int("lead" in ids)
        receiving += int("pass" in ids)
    assert lead > receiving * 1.75


def test_v63_environment_is_centered_broad_and_correlated():
    environments = [sample_game_environment_v63(seed) for seed in range(6300, 9300)]
    common = np.asarray([item.common_execution for item in environments])
    away = np.asarray([item.away_execution for item in environments])
    home = np.asarray([item.home_execution for item in environments])
    explosive = np.asarray([item.explosive_factor for item in environments])
    tempo = np.asarray([item.tempo_factor for item in environments])

    assert 0.96 < common.mean() < 1.03
    assert np.quantile(common, 0.05) < 0.90
    assert np.quantile(common, 0.95) > 1.10
    assert np.corrcoef(away, home)[0, 1] > 0.25
    assert np.corrcoef(common, explosive)[0, 1] > 0.25
    assert tempo.std(ddof=1) > 0.035


def test_tempo_changes_clock_causally_not_score_directly():
    fast = GameEnvironmentV63(1.0, 1.0, 1.0, 0.07, 1.0, 1.0, 1.15)
    slow = GameEnvironmentV63(1.0, 1.0, 1.0, 0.07, 1.0, 1.0, 0.87)
    set_active_game_environment(fast)
    fast_seconds = cadence_seconds_v63(32)
    set_active_game_environment(slow)
    slow_seconds = cadence_seconds_v63(32)
    set_active_game_environment(None)
    assert fast_seconds < 32 < slow_seconds
    assert not hasattr(fast, "points")


def test_return_continuation_has_heavier_sixty_plus_tail_than_v62_geometry():
    set_active_game_environment(None)
    kwargs = {
        "mean": 11.5,
        "sd": 12.0,
        "zero_rate": 0.22,
        "forty_plus_rate": 0.07,
        "return_skill": 1.0,
        "maximum": 100.0,
    }
    base_rng = np.random.default_rng(6311)
    new_rng = np.random.default_rng(6311)
    base = np.asarray([sample_return_yards(rng=base_rng, **kwargs) for _ in range(20000)])
    candidate = np.asarray(
        [sample_return_yards_v63(rng=new_rng, **kwargs) for _ in range(20000)]
    )
    assert np.mean(candidate >= 60.0) > np.mean(base >= 60.0) * 1.20
    assert np.mean(candidate) < np.mean(base) + 3.0
