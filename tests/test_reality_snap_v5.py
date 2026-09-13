from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from monster.sim.football_state import FootballState
from monster.sim.matchup_kernel import DefensiveIdentity, DefensiveUnit
from monster.sim.play_kernel import PlayerIdentity, TeamIdentity
from monster.sim.reality_snap_v5 import prepare_snap_world
from monster.sim.snap_ecology import register_team_units


def _offense() -> TeamIdentity:
    qb = PlayerIdentity("qb", "QB", "QB", usage_weight=1.0, efficiency=1.05, explosive=1.05)
    rb1 = PlayerIdentity("rb1", "RB1", "RB", usage_weight=0.65, efficiency=1.04)
    rb2 = PlayerIdentity("rb2", "RB2", "RB", usage_weight=0.22, efficiency=0.98)
    te1 = PlayerIdentity("te1", "TE1", "TE", usage_weight=0.42, efficiency=1.02)
    te2 = PlayerIdentity("te2", "TE2", "TE", usage_weight=0.16, efficiency=0.96)
    wr1 = PlayerIdentity("wr1", "WR1", "WR", usage_weight=0.80, efficiency=1.10, explosive=1.10)
    wr2 = PlayerIdentity("wr2", "WR2", "WR", usage_weight=0.62, efficiency=1.04)
    wr3 = PlayerIdentity("wr3", "WR3", "WR", usage_weight=0.34, efficiency=1.00)
    wr4 = PlayerIdentity("wr4", "WR4", "WR", usage_weight=0.14, efficiency=0.95)
    return TeamIdentity(
        team_id="OFF",
        quarterback=qb,
        rushers=(rb1, rb2, qb),
        receivers=(wr1, wr2, wr3, wr4, te1, te2, rb1, rb2),
        neutral_pass_rate=0.59,
    )


def _unit_players() -> tuple[object, ...]:
    rows = []
    for idx, position in enumerate(("LT", "LG", "C", "RG", "RT")):
        rows.append(
            SimpleNamespace(
                player_id=f"ol{idx}",
                position=position,
                offense_snap_share=0.95 - 0.01 * idx,
                pass_block_signal=0.08 * (idx - 2),
                run_block_signal=0.06 * (2 - idx),
                madden_pass_block=80 + idx,
                madden_run_block=79 + idx,
                madden_awareness=81,
                madden_stamina=88,
            )
        )
    rows.extend(
        [
            SimpleNamespace(
                player_id="te1",
                position="TE",
                offense_snap_share=0.75,
                madden_pass_block=72,
                madden_run_block=75,
                madden_route_running=80,
                madden_catching=82,
            ),
            SimpleNamespace(
                player_id="rb1",
                position="RB",
                offense_snap_share=0.62,
                madden_pass_block=66,
                madden_run_block=62,
                madden_route_running=72,
                madden_catching=78,
            ),
        ]
    )
    return tuple(rows)


def _defense(skill_flip: bool = False) -> DefensiveUnit:
    players = []
    specs = (
        ("edge1", "EDGE", 0.88),
        ("edge2", "EDGE", 0.84),
        ("dt1", "DT", 0.80),
        ("dt2", "DT", 0.72),
        ("lb1", "LB", 0.86),
        ("lb2", "LB", 0.77),
        ("lb3", "LB", 0.63),
        ("cb1", "CB", 0.92),
        ("cb2", "CB", 0.88),
        ("cb3", "CB", 0.71),
        ("s1", "S", 0.91),
        ("s2", "S", 0.83),
        ("db6", "DB", 0.52),
    )
    for idx, (pid, pos, snap) in enumerate(specs):
        skill = (1.28 - 0.04 * idx) if skill_flip else (0.78 + 0.04 * idx)
        players.append(
            DefensiveIdentity(
                pid,
                pid,
                pos,
                coverage=skill,
                pass_rush=1.35 - skill / 3.0,
                run_defense=skill,
                tackling=skill,
                snap_weight=snap,
            )
        )
    front = tuple(p for p in players if p.position in {"EDGE", "DE", "DT", "NT", "DL", "LB", "ILB", "MLB", "OLB"})
    coverage = tuple(p for p in players if p.position in {"CB", "DB", "S", "FS", "SS", "LB", "ILB", "MLB", "OLB"})
    return DefensiveUnit(front=front, coverage=coverage)


def _state() -> FootballState:
    return FootballState(
        possession="OFF",
        defense="DEF",
        quarter=2,
        seconds_remaining=1733,
        yardline_100=43.0,
        down=2,
        distance=6.0,
        away_score=7,
        home_score=10,
        away_team_id="OFF",
        home_team_id="DEF",
    )


def test_v5_snap_world_is_exact_11v11_with_real_ol() -> None:
    offense = _offense()
    register_team_units("OFF", _unit_players())
    _, _, world = prepare_snap_world(
        state=_state(),
        offense=offense,
        defense=_defense(),
        responsibility_key="exact-11v11",
    )
    assert len(world.offense_participant_ids) == 11
    assert len(set(world.offense_participant_ids)) == 11
    assert len(world.defense_participant_ids) == 11
    assert len(set(world.defense_participant_ids)) == 11
    assert {"ol0", "ol1", "ol2", "ol3", "ol4"}.issubset(world.offense_participant_ids)
    assert len(world.active_receiver_ids) <= 5


def test_v5_participation_does_not_depend_on_defender_skill() -> None:
    offense = _offense()
    register_team_units("OFF", _unit_players())
    _, _, base = prepare_snap_world(
        state=_state(), offense=offense, defense=_defense(False), responsibility_key="skill-blind"
    )
    _, _, flipped = prepare_snap_world(
        state=_state(), offense=offense, defense=_defense(True), responsibility_key="skill-blind"
    )
    assert base.defense_participant_ids == flipped.defense_participant_ids
    assert base.rush_participant_ids == flipped.rush_participant_ids
    assert base.defense_alignment == flipped.defense_alignment


def test_v5_same_snap_has_one_shared_defensive_call() -> None:
    offense = _offense()
    register_team_units("OFF", _unit_players())
    _, _, first = prepare_snap_world(
        state=_state(), offense=offense, defense=_defense(), responsibility_key="shared-intent"
    )
    _, _, second = prepare_snap_world(
        state=_state(), offense=offense, defense=_defense(), responsibility_key="shared-intent"
    )
    assert first.offense_package == second.offense_package
    assert first.defense_package == second.defense_package
    assert first.defensive_intent == second.defensive_intent
    assert first.offense_alignment == second.offense_alignment
    assert first.defense_alignment == second.defense_alignment


def test_v5_long_yardage_can_create_subpackage_without_reading_play_result() -> None:
    offense = _offense()
    register_team_units("OFF", _unit_players())
    state = replace(_state(), down=3, distance=11.0)
    _, _, world = prepare_snap_world(
        state=state,
        offense=offense,
        defense=_defense(),
        responsibility_key="third-and-long",
    )
    assert world.defense_package in {"nickel", "dime", "base"}
    assert len(world.rush_participant_ids) in {4, 5}
    assert world.defensive_intent.authority > 0.0
