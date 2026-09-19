from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import polars as pl

from monster.feature_compile.units import UnitPlayerInputs
from monster.reality import live_state_v72
from monster.reality.defensive_attribution_v72 import (
    attribute_defensive_box_score_v72,
)
from monster.reality.participation_authority_v72 import (
    configure_participation_authority_v72,
    register_team_units_v72,
)
from monster.reality.qb_rush_authority_v72 import (
    choose_rusher_for_geometry_v72,
    designed_qb_entry_probability_v72,
)
from monster.reality.receiver_topology_v72 import (
    alignment_compatibility_v72,
    choose_target_for_depth_v72,
)
from monster.reality.special_teams_identity_v72 import (
    configure_base_special_teams_hooks_v72,
    configure_special_teams_identities_v72,
    set_play_context_v72,
    simulate_field_goal_v72,
    simulate_kickoff_v72,
    simulate_punt_v72,
)
from monster.sim.defensive_intent import CoverageShell, DefensiveIntent, RushPlan
from monster.sim.matchup_kernel import DefensiveIdentity, DefensiveUnit
from monster.sim.play_kernel import PassResult, PlayerIdentity, PlayEvent, PlayType, TeamIdentity
from monster.sim.reality_snap_v5 import (
    _EVENT_META,
    _WORLD_BY_KEY,
    SnapWorldV5,
)


def _personnel(rows: list[dict]) -> pl.DataFrame:
    defaults = {
        "position_group": "",
        "depth_position": "",
        "depth_rank": 0,
        "conditional_offense_snap_share": 0.0,
        "conditional_defense_snap_share": 0.0,
        "conditional_special_teams_snap_share": 0.0,
        "game_day_active_probability": 1.0,
        "status": "ACT",
        "madden_injury": 80.0,
        "madden_stamina": 82.0,
        "madden_return": None,
        "madden_kick_power": None,
        "madden_kick_accuracy": None,
    }
    return pl.DataFrame([{**defaults, **row} for row in rows])


def test_v72_offensive_line_uses_current_depth_seats_before_stale_snap_share() -> None:
    personnel = _personnel(
        [
            {
                "gsis_id": "lt1",
                "team_id": "BUF",
                "position": "OL",
                "position_group": "OL",
                "depth_position": "LT",
                "depth_rank": 1,
                "conditional_offense_snap_share": 1.0,
            },
            {
                "gsis_id": "lt2",
                "team_id": "BUF",
                "position": "OL",
                "position_group": "OL",
                "depth_position": "LT",
                "depth_rank": 2,
                "conditional_offense_snap_share": 0.99,
            },
            {
                "gsis_id": "lg1",
                "team_id": "BUF",
                "position": "OL",
                "position_group": "OL",
                "depth_position": "LG",
                "depth_rank": 1,
                "conditional_offense_snap_share": 0.35,
            },
            {
                "gsis_id": "c1",
                "team_id": "BUF",
                "position": "OL",
                "position_group": "OL",
                "depth_position": "C",
                "depth_rank": 1,
                "conditional_offense_snap_share": 0.88,
            },
            {
                "gsis_id": "rg1",
                "team_id": "BUF",
                "position": "OL",
                "position_group": "OL",
                "depth_position": "RG",
                "depth_rank": 1,
                "conditional_offense_snap_share": 0.92,
            },
            {
                "gsis_id": "rt1",
                "team_id": "BUF",
                "position": "OL",
                "position_group": "OL",
                "depth_position": "RT",
                "depth_rank": 1,
                "conditional_offense_snap_share": 0.62,
            },
        ]
    )
    configure_participation_authority_v72(personnel)
    units = tuple(
        UnitPlayerInputs(
            row["gsis_id"],
            "OL",
            offense_snap_share=float(row["conditional_offense_snap_share"]),
        )
        for row in personnel.to_dicts()
    )
    profile = register_team_units_v72("BUF", units)
    seats = {blocker.position: blocker.player_id for blocker in profile.offensive_line}
    assert seats == {
        "LT": "lt1",
        "LG": "lg1",
        "C": "c1",
        "RG": "rg1",
        "RT": "rt1",
    }
    assert "lt2" not in {blocker.player_id for blocker in profile.offensive_line}


def test_v72_qb_designed_run_is_entry_probability_not_equal_actor_weight() -> None:
    qb = PlayerIdentity("qb", "QB", "QB")
    rb = PlayerIdentity("rb", "RB", "RB")
    ecology = SimpleNamespace(
        rusher_geometry_attempts={
            ("qb", "left_edge"): 5,
            ("rb", "left_edge"): 95,
        }
    )
    probability = designed_qb_entry_probability_v72(
        (qb, rb),
        ecology,
        "left_edge",
    )
    assert 0.04 <= probability <= 0.07

    rng = np.random.default_rng(72001)
    qb_count = 0
    for _ in range(6000):
        chosen = choose_rusher_for_geometry_v72(
            (qb, rb),
            ecology,
            "left_edge",
            rng,
        )
        qb_count += chosen.player_id == "qb"
    assert 0.035 <= qb_count / 6000 <= 0.075


def test_v72_deep_route_family_prefers_wr_over_rb_without_usage_share() -> None:
    personnel = _personnel(
        [
            {
                "gsis_id": "wr",
                "team_id": "T",
                "position": "WR",
                "position_group": "WR",
                "depth_rank": 1,
                "conditional_offense_snap_share": 0.9,
            },
            {
                "gsis_id": "rb",
                "team_id": "T",
                "position": "RB",
                "position_group": "RB",
                "depth_rank": 1,
                "conditional_offense_snap_share": 0.9,
            },
        ]
    )
    configure_participation_authority_v72(personnel)
    wr = PlayerIdentity("wr", "WR", "WR", usage_weight=0.01)
    rb = PlayerIdentity("rb", "RB", "RB", usage_weight=0.99)
    ecology = SimpleNamespace(target_depth_attempts={})
    rng = np.random.default_rng(72002)
    counts = {"wr": 0, "rb": 0}
    for _ in range(3000):
        target = choose_target_for_depth_v72(
            (wr, rb),
            ecology,
            "deep_20_39",
            rng,
        )
        counts[target.player_id] += 1
    assert counts["wr"] > 2 * counts["rb"]


def test_v72_alignment_compatibility_reads_realized_snap_alignment() -> None:
    intent = DefensiveIntent(
        coverage_shell=CoverageShell.TWO_HIGH,
        rush_plan=RushPlan.FOUR,
        box_aggression=0.5,
    )
    _WORLD_BY_KEY["k"] = SnapWorldV5(
        responsibility_key="k",
        offense_package="11",
        defense_package="nickel",
        offense_participant_ids=("qb", "x", "slot"),
        defense_participant_ids=(),
        offense_alignment=("qb:QB", "x:X", "slot:SLOT"),
        defense_alignment=(),
        active_receiver_ids=frozenset({"x", "slot"}),
        active_rusher_ids=frozenset(),
        rush_participant_ids=frozenset(),
        defensive_intent=intent,
        defense=None,
    )
    assert (
        alignment_compatibility_v72("x", "deep_20_39", "k")
        > alignment_compatibility_v72("slot", "deep_20_39", "k")
    )
    _WORLD_BY_KEY.pop("k", None)


def test_v72_defensive_attribution_uses_actual_rusher_and_participants() -> None:
    edge = DefensiveIdentity("edge", "EDGE", "EDGE", pass_rush=1.2)
    cb = DefensiveIdentity("cb", "CB", "CB", coverage=1.2)
    bench = DefensiveIdentity("bench", "BENCH", "EDGE", pass_rush=1.4)
    defense = DefensiveUnit(front=(edge, bench), coverage=(cb,))
    event = PlayEvent(
        play_type=PlayType.PASS,
        elapsed_seconds=5,
        passer_id="qb",
        pass_result=PassResult.SACK,
        pressured=True,
    )
    _EVENT_META[id(event)] = {
        "defense_participant_ids": ("edge", "cb"),
        "primary_rusher_id": "edge",
    }
    stats = attribute_defensive_box_score_v72((event,), defense, seed=7)
    assert stats["edge"].pressures == 1
    assert stats["edge"].sacks == 1
    assert stats["edge"].defensive_snaps == 1
    assert stats["cb"].defensive_snaps == 1
    assert stats["bench"].defensive_snaps == 0
    _EVENT_META.pop(id(event), None)


def test_v72_live_qb_exit_promotes_current_backup() -> None:
    personnel = _personnel(
        [
            {
                "gsis_id": "qb1",
                "team_id": "T",
                "position": "QB",
                "position_group": "QB",
                "depth_rank": 1,
                "conditional_offense_snap_share": 1.0,
            },
            {
                "gsis_id": "qb2",
                "team_id": "T",
                "position": "QB",
                "position_group": "QB",
                "depth_rank": 2,
                "conditional_offense_snap_share": 0.05,
            },
            {
                "gsis_id": "wr",
                "team_id": "T",
                "position": "WR",
                "position_group": "WR",
                "depth_rank": 1,
                "conditional_offense_snap_share": 1.0,
            },
        ]
    )
    configure_participation_authority_v72(personnel)
    live_state_v72.configure_live_state_v72()
    qb1 = PlayerIdentity("qb1", "QB1", "QB")
    qb2 = PlayerIdentity("qb2", "QB2", "QB")
    wr = PlayerIdentity("wr", "WR", "WR")
    live_state_v72.register_team_identities_v72(
        "T",
        {"qb1": qb1, "qb2": qb2, "wr": wr},
    )
    offense = TeamIdentity("T", qb1, (qb1,), (wr,))
    state = SimpleNamespace(
        away_team_id="T",
        home_team_id="X",
        possession="T",
        defense="X",
    )
    rng = np.random.default_rng(72003)
    world = live_state_v72._world(state, rng)
    player_state = live_state_v72._player_state(world, "qb1")
    player_state.status = "out"
    player_state.multiplier = 0.0
    active, _ = live_state_v72.apply_live_state_v72(
        offense,
        None,
        state=state,
        rng=rng,
    )
    assert active.quarterback.player_id == "qb2"


def test_v72_special_teams_assigns_kicker_punter_and_returner_ids() -> None:
    personnel = _personnel(
        [
            {
                "gsis_id": "k",
                "team_id": "A",
                "position": "K",
                "position_group": "SPEC",
                "depth_rank": 1,
                "conditional_special_teams_snap_share": 0.8,
                "madden_kick_power": 90,
                "madden_kick_accuracy": 90,
            },
            {
                "gsis_id": "p",
                "team_id": "A",
                "position": "P",
                "position_group": "SPEC",
                "depth_rank": 1,
                "conditional_special_teams_snap_share": 0.8,
                "madden_kick_power": 88,
                "madden_kick_accuracy": 82,
            },
            {
                "gsis_id": "ret",
                "team_id": "B",
                "position": "WR",
                "position_group": "WR",
                "depth_rank": 2,
                "conditional_special_teams_snap_share": 0.7,
                "madden_return": 92,
            },
        ]
    )
    configure_participation_authority_v72(personnel)
    configure_special_teams_identities_v72(personnel)

    captured: dict[str, dict] = {}

    def fake_punt(rng, **kwargs):
        captured["punt"] = kwargs
        return kwargs

    def fake_fg(rng, **kwargs):
        captured["fg"] = kwargs
        return kwargs

    def fake_kickoff(rng, **kwargs):
        captured["kickoff"] = kwargs
        return kwargs

    configure_base_special_teams_hooks_v72(
        punt=fake_punt,
        field_goal=fake_fg,
        kickoff=fake_kickoff,
        kickoff_loop=lambda state, *args, **kwargs: state,
    )
    set_play_context_v72("A", "B")
    rng = np.random.default_rng(72004)
    simulate_punt_v72(rng)
    simulate_field_goal_v72(rng, distance=43.0)
    simulate_kickoff_v72(rng)

    assert captured["punt"]["punter_id"] == "p"
    assert captured["punt"]["returner_id"] == "ret"
    assert captured["fg"]["kicker_id"] == "k"
    assert captured["kickoff"]["kicker_id"] == "k"
    assert captured["kickoff"]["returner_id"] == "ret"
