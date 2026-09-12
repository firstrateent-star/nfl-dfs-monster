from __future__ import annotations

import polars as pl

from monster.feature_compile.mechanisms import PlayerMechanismInputs
from monster.feature_compile.units import UnitPlayerInputs
from monster.ingest.madden_canonical import canonicalize_madden_attribute_mirror
from monster.ingest.madden_official import attach_all_madden_attributes
from monster.sim.dispersion_bridge import enhanced_team_identity
from monster.snapshot.model import TeamState
from monster.snapshot.player import PlayerState, TeamPlayerPool


def test_public_madden_attribute_mirror_becomes_canonical_mechanism_evidence() -> None:
    mirror = pl.DataFrame(
        {
            "player_id": [20948],
            "full_name": ["Joe Burrow"],
            "position": ["QB"],
            "team_name": ["Cincinnati Bengals"],
            "speed_rating": [83],
            "accel_rating": [85],
            "catch_rating": [19],
            "route_run_short_rating": [20],
            "route_run_med_rating": [30],
            "route_run_deep_rating": [15],
            "throw_acc_short_rating": [98],
            "throw_acc_mid_rating": [99],
            "throw_acc_deep_rating": [95],
            "power_moves_rating": [27],
            "finesse_moves_rating": [10],
            "block_shed_rating": [29],
            "pursuit_rating": [38],
            "play_rec_rating": [86],
            "man_cover_rating": [10],
            "zone_cover_rating": [33],
            "press_rating": [25],
            "tackle_rating": [36],
        }
    )
    canonical = canonicalize_madden_attribute_mirror(mirror)
    row = canonical.row(0, named=True)
    assert row["madden_speed"] == 83.0
    assert row["madden_throw_accuracy"] == 292.0 / 3.0
    assert row["madden_route_running"] == 65.0 / 3.0
    assert row["madden_match_name_key"] == "joeburrow"

    personnel = pl.DataFrame(
        {
            "display_name": ["Joe Burrow"],
            "team_id": ["CIN"],
            "position": ["QB"],
        }
    )
    attached = attach_all_madden_attributes(personnel, canonical)
    assert attached["madden_speed"][0] == 83.0
    assert attached["madden_throw_accuracy"][0] == 292.0 / 3.0
    assert attached["madden_official_match_type"][0] == "team_name"


def test_dispersion_bridge_limits_full_roster_target_competition_and_uses_rich_identity() -> None:
    players = (
        PlayerState("qb", "QB", "QB", "AAA", qb_pass_share=1.0),
        PlayerState("star", "Star", "WR", "AAA", target_share=0.34),
        PlayerState("wr2", "WR2", "WR", "AAA", target_share=0.22),
        PlayerState("wr3", "WR3", "WR", "AAA", target_share=0.15),
        PlayerState("te1", "TE1", "TE", "AAA", target_share=0.12),
        PlayerState("rb1", "RB1", "RB", "AAA", target_share=0.09, rush_share=0.70),
        PlayerState("wr4", "WR4", "WR", "AAA", target_share=0.05),
        PlayerState("fringe", "Fringe", "WR", "AAA", target_share=0.01),
    )
    pool = TeamPlayerPool(team_id="AAA", players=players, neutral_pass_rate=0.60)
    reality = {
        player.player_id: PlayerMechanismInputs(
            madden_speed=96.0 if player.player_id == "star" else 80.0,
            madden_acceleration=95.0 if player.player_id == "star" else 80.0,
            madden_route_running=97.0 if player.player_id == "star" else 78.0,
            madden_catching=99.0 if player.player_id == "star" else 78.0,
        )
        for player in players
    }
    units = tuple(
        UnitPlayerInputs(
            player_id=player.player_id,
            position=player.position,
            offense_snap_share=0.90 if player.player_id in {"qb", "star"} else 0.55,
            madden_speed=96.0 if player.player_id == "star" else 80.0,
            madden_acceleration=95.0 if player.player_id == "star" else 80.0,
            madden_route_running=97.0 if player.player_id == "star" else 78.0,
            madden_catching=99.0 if player.player_id == "star" else 78.0,
            madden_release=96.0 if player.player_id == "star" else 78.0,
            madden_awareness=90.0,
        )
        for player in players
    )
    state = TeamState(
        team_id="AAA",
        opponent_id="BBB",
        offensive_epa_per_play=0.12,
        offensive_success_rate=0.49,
        offensive_explosive_rate=0.14,
    )
    identity = enhanced_team_identity(
        "AAA",
        pool,
        reality,
        units,
        state,
        league_neutral_pass_rate=0.56,
        situational_pass_rates=(0.55,) * 12,
    )

    receiver_ids = {player.player_id for player in identity.receivers}
    assert len(identity.receivers) == 6
    assert "fringe" not in receiver_ids
    assert "star" in receiver_ids
    assert identity.pass_efficiency > 1.0
    star = next(player for player in identity.receivers if player.player_id == "star")
    wr4 = next(player for player in identity.receivers if player.player_id == "wr4")
    assert star.usage_weight > wr4.usage_weight
    assert star.efficiency > wr4.efficiency
