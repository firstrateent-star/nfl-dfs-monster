from __future__ import annotations

from monster.feature_compile.mechanisms import PlayerMechanismInputs
from monster.feature_compile.units import UnitPlayerInputs
from monster.feature_compile.v13_identity_bridge import compile_v13_player_identity
from monster.sim.matchup_kernel import (
    DefensiveIdentity,
    DefensiveUnit,
    resolve_pass_matchup,
    resolve_run_matchup,
)
from monster.sim.rich_identity import RichPlayerIdentity


def _defense() -> DefensiveUnit:
    return DefensiveUnit(
        front=(
            DefensiveIdentity(
                player_id="edge",
                name="Edge",
                position="EDGE",
                pass_rush=1.08,
                run_defense=1.06,
                tackling=1.05,
            ),
        ),
        coverage=(
            DefensiveIdentity(
                player_id="cb",
                name="Corner",
                position="CB",
                coverage=1.02,
                tackling=1.02,
                ball_hawk=1.01,
                speed=1.02,
            ),
        ),
    )


def test_rich_capability_bridge_preserves_mechanism_channels() -> None:
    identity, trace = compile_v13_player_identity(
        player_id="wr-rich",
        name="Rich WR",
        position="WR",
        usage_weight=1.0,
        inputs=PlayerMechanismInputs(),
        capability_inputs=UnitPlayerInputs(
            player_id="wr-rich",
            position="WR",
            madden_speed=97.0,
            madden_acceleration=95.0,
            madden_route_running=94.0,
            madden_catching=93.0,
            madden_release=92.0,
            madden_catch_in_traffic=90.0,
        ),
    )

    assert isinstance(identity, RichPlayerIdentity)
    assert trace.rich_capability_active
    assert identity.evidence_fields >= 5
    assert identity.speed_skill > 0.0
    assert identity.route_separation_skill > 0.0
    assert identity.catchpoint_skill > 0.0


def test_receiver_skill_changes_owned_pass_mechanisms() -> None:
    defense = _defense()
    weak = RichPlayerIdentity(
        player_id="wr-low",
        name="Low",
        position="WR",
        route_separation_skill=-1.0,
        catchpoint_skill=-1.0,
        speed_skill=-1.0,
        open_field_skill=-1.0,
    )
    elite = RichPlayerIdentity(
        player_id="wr-high",
        name="High",
        position="WR",
        route_separation_skill=1.0,
        catchpoint_skill=1.0,
        speed_skill=1.0,
        open_field_skill=1.0,
    )

    weak_matchup = resolve_pass_matchup(
        weak,
        defense,
        pass_protection=1.0,
        quarterback_efficiency=1.0,
    )
    elite_matchup = resolve_pass_matchup(
        elite,
        defense,
        pass_protection=1.0,
        quarterback_efficiency=1.0,
    )

    assert elite_matchup.local_separation_edge > weak_matchup.local_separation_edge
    assert elite_matchup.completion_probability > weak_matchup.completion_probability
    assert elite_matchup.yards_multiplier > weak_matchup.yards_multiplier


def test_rusher_skill_changes_owned_run_mechanisms() -> None:
    defense = _defense()
    weak = RichPlayerIdentity(
        player_id="rb-low",
        name="Low",
        position="RB",
        rush_creation_skill=-1.0,
        runner_power_skill=-1.0,
        open_field_skill=-1.0,
        speed_skill=-1.0,
    )
    elite = RichPlayerIdentity(
        player_id="rb-high",
        name="High",
        position="RB",
        rush_creation_skill=1.0,
        runner_power_skill=1.0,
        open_field_skill=1.0,
        speed_skill=1.0,
    )

    weak_matchup = resolve_run_matchup(weak, defense, run_blocking=1.0)
    elite_matchup = resolve_run_matchup(elite, defense, run_blocking=1.0)

    assert elite_matchup.runner_edge > weak_matchup.runner_edge
    assert elite_matchup.stuff_probability < weak_matchup.stuff_probability
    assert elite_matchup.yards_multiplier > weak_matchup.yards_multiplier
