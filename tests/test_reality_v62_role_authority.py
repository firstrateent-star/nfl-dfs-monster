from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import polars as pl

from monster.sim.play_kernel import PlayerIdentity, TeamIdentity
from monster.sim.reality_v62 import (
    apply_game_environment,
    choose_rusher_role_authoritative,
    choose_target_role_authoritative,
    filter_roster_truth,
    sample_game_environment,
    sample_target_share_plan,
)
from monster.snapshot.player import PlayerState, TeamPlayerPool


def test_roster_truth_blocks_cut_and_low_probability_reserve_states():
    personnel = pl.DataFrame(
        {
            "display_name": ["Active", "Cut", "PS", "Elevated"],
            "status": ["ACT", "CUT", "DEV", "DEV"],
            "game_day_active_probability": [0.99, 0.99, 0.04, 0.95],
        }
    )
    filtered = filter_roster_truth(personnel)
    assert filtered.get_column("display_name").to_list() == ["Active", "Elevated"]


def test_current_rush_role_beats_old_geometry_volume():
    players = (
        PlayerIdentity("lead", "Lead", "RB", usage_weight=0.70),
        PlayerIdentity("old", "Old", "RB", usage_weight=0.30),
    )
    ecology = SimpleNamespace(
        run_geometry=SimpleNamespace(categories=("interior", "edge")),
        rusher_geometry_attempts={
            ("lead", "interior"): 2,
            ("lead", "edge"): 98,
            ("old", "interior"): 900,
            ("old", "edge"): 100,
        },
    )
    rng = np.random.default_rng(6201)
    lead = sum(
        choose_rusher_role_authoritative(players, ecology, "interior", rng).player_id == "lead"
        for _ in range(5000)
    )
    assert lead / 5000 > 0.55


def test_current_target_role_beats_old_depth_volume():
    players = (
        PlayerIdentity("alpha", "Alpha", "WR", usage_weight=0.64),
        PlayerIdentity("old", "Old", "WR", usage_weight=0.36),
    )
    ecology = SimpleNamespace(
        pass_depth=SimpleNamespace(categories=("short", "deep")),
        target_depth_attempts={
            ("alpha", "short"): 95,
            ("alpha", "deep"): 5,
            ("old", "short"): 50,
            ("old", "deep"): 950,
        },
    )
    rng = np.random.default_rng(6202)
    alpha = sum(
        choose_target_role_authoritative(players, ecology, "deep", rng).player_id == "alpha"
        for _ in range(5000)
    )
    assert alpha / 5000 > 0.50


def test_zero_history_current_receiver_can_enter_role_world():
    pool = TeamPlayerPool(
        team_id="TST",
        players=(
            PlayerState("a", "Alpha", "WR", "TST", target_share=0.60, role_uncertainty=0.08),
            PlayerState("b", "Beta", "WR", "TST", target_share=0.25, role_uncertainty=0.10),
            PlayerState("c", "Current TE", "TE", "TST", target_share=0.0, role_uncertainty=0.30),
            PlayerState("d", "Back", "RB", "TST", target_share=0.15, role_uncertainty=0.12),
        ),
    )
    rng = np.random.default_rng(6203)
    shares = [sample_target_share_plan(pool, rng=rng).get("c", 0.0) for _ in range(1500)]
    assert np.mean(shares) > 0.01
    assert np.mean(np.asarray(shares) > 0.0) > 0.25


def test_game_environment_is_centered_but_has_persistent_tail_width():
    environments = [sample_game_environment(seed) for seed in range(5000, 9000)]
    away = np.asarray([item.away_execution for item in environments])
    common = np.asarray([item.common_execution for item in environments])
    assert 0.97 < away.mean() < 1.03
    assert np.quantile(away, 0.05) < 0.88
    assert np.quantile(away, 0.95) > 1.12
    assert np.quantile(common, 0.05) < 0.91
    assert np.quantile(common, 0.95) > 1.09
    assert all(0.035 <= item.penalty_rate <= 0.115 for item in environments)


def test_game_environment_changes_mechanisms_not_points_directly():
    team = TeamIdentity(
        team_id="A",
        quarterback=PlayerIdentity("q", "Q", "QB"),
        rushers=(PlayerIdentity("r", "R", "RB"),),
        receivers=(PlayerIdentity("w", "W", "WR"),),
        pass_efficiency=1.0,
        rush_efficiency=1.0,
        pass_protection=1.0,
        run_blocking=1.0,
    )
    environment = sample_game_environment(99)
    away, home = apply_game_environment(team, team, environment)
    assert away.pass_efficiency != 1.0 or away.rush_efficiency != 1.0
    assert home.pass_efficiency != 1.0 or home.rush_efficiency != 1.0
    assert not hasattr(environment, "points")
