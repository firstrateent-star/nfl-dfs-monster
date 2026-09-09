from __future__ import annotations

import numpy as np
import polars as pl

from monster.feature_compile.participation import compile_snap_priors
from monster.feature_compile.units import (
    UnitPlayerInputs,
    apply_team_unit_effects,
    compile_team_unit_effects,
)
from monster.sim.game import simulate_game
from monster.snapshot.model import GameState, TeamState


def test_snap_priors_use_recent_games_and_normalize_percent_scale():
    df = pl.DataFrame(
        {
            "season": [2025] * 4,
            "week": [1, 2, 3, 4],
            "team": ["A"] * 4,
            "position": ["EDGE"] * 4,
            "pfr_player_id": ["edge1"] * 4,
            "offense_pct": [0.0, 0.0, 0.0, 0.0],
            "defense_pct": [50.0, 60.0, 70.0, 80.0],
            "st_pct": [10.0, 10.0, 10.0, 10.0],
        }
    )
    result = compile_snap_priors(df, recent_games=3).row(0, named=True)
    assert np.isclose(result["defense_snap_share"], 0.70)
    assert np.isclose(result["special_teams_snap_share"], 0.10)
    assert result["snap_games_observed"] == 3


def test_high_snap_defender_has_more_unit_weight_than_low_snap_defender():
    high = UnitPlayerInputs(
        "high",
        "EDGE",
        defense_snap_share=0.90,
        madden_pass_rush=95.0,
    )
    low = UnitPlayerInputs(
        "low",
        "EDGE",
        defense_snap_share=0.20,
        madden_pass_rush=95.0,
    )
    weak = UnitPlayerInputs(
        "weak",
        "EDGE",
        defense_snap_share=0.90,
        madden_pass_rush=65.0,
    )
    high_effect, _ = compile_team_unit_effects((high, weak))
    low_effect, _ = compile_team_unit_effects((low, weak))
    assert high_effect.pass_rush_effect > low_effect.pass_rush_effect


def test_unit_effects_are_neutral_when_capability_data_are_missing():
    players = (
        UnitPlayerInputs("ol", "OT", offense_snap_share=0.95),
        UnitPlayerInputs("cb", "CB", defense_snap_share=0.98),
        UnitPlayerInputs("st", "LB", special_teams_snap_share=0.70),
    )
    effects, trace = compile_team_unit_effects(players)
    assert effects.pass_protection_effect == 0.0
    assert effects.run_block_effect == 0.0
    assert effects.pass_rush_effect == 0.0
    assert effects.coverage_effect == 0.0
    assert effects.run_defense_effect == 0.0
    assert effects.special_teams_effect == 0.0
    assert trace.players_with_offense_weight == 1
    assert trace.players_with_defense_weight == 1
    assert trace.players_with_special_teams_weight == 1


def test_stronger_defensive_personnel_can_reduce_opponent_scoring_with_same_seed():
    offense = TeamState("A", "H", td_drive_rate=0.24, fg_drive_rate=0.14)
    neutral_defense = TeamState("H", "A")
    strong_defense = apply_team_unit_effects(
        neutral_defense,
        compile_team_unit_effects(
            (
                UnitPlayerInputs(
                    "edge",
                    "EDGE",
                    defense_snap_share=0.95,
                    madden_pass_rush=97.0,
                ),
                UnitPlayerInputs(
                    "cb",
                    "CB",
                    defense_snap_share=0.98,
                    madden_coverage=96.0,
                ),
                UnitPlayerInputs(
                    "lb",
                    "LB",
                    defense_snap_share=0.90,
                    madden_tackle=94.0,
                ),
            )
        )[0],
    )
    neutral = simulate_game(GameState("A@H", offense, neutral_defense), 40_000, 1234)
    strong = simulate_game(GameState("A@H", offense, strong_defense), 40_000, 1234)
    assert strong.away_points.mean() < neutral.away_points.mean()
