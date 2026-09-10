from __future__ import annotations

from monster.feature_compile.mechanisms import PlayerMechanismInputs
from monster.feature_compile.units import UnitPlayerInputs
from monster.feature_compile.v13_identity_authority import apply_v13_team_identity_authority
from monster.feature_compile.v13_identity_bridge import compile_v13_player_identity
from monster.sim.play_kernel import PlayerIdentity, TeamIdentity
from monster.snapshot.model import TeamState
from monster.snapshot.player import PlayerState, TeamPlayerPool


def _capability(player_id: str, position: str, rating: float) -> UnitPlayerInputs:
    kwargs = {
        "player_id": player_id,
        "position": position,
        "madden_speed": rating,
        "madden_acceleration": rating,
        "madden_awareness": rating,
    }
    if position == "QB":
        kwargs.update(
            {
                "madden_throw_accuracy": rating,
                "madden_throw_power": rating,
                "madden_throw_under_pressure": rating,
            }
        )
    elif position in {"WR", "TE"}:
        kwargs.update(
            {
                "madden_route_running": rating,
                "madden_catching": rating,
                "madden_release": rating,
            }
        )
    elif position == "RB":
        kwargs.update(
            {
                "madden_carrying": rating,
                "madden_break_tackle": rating,
                "madden_ball_carrier_vision": rating,
            }
        )
    return UnitPlayerInputs(**kwargs)


def test_rich_qb_capability_changes_qb_identity() -> None:
    physical = PlayerMechanismInputs(
        active_probability=1.0,
        effectiveness_if_active=1.0,
        forty_time=4.65,
    )
    low, low_trace = compile_v13_player_identity(
        player_id="qb",
        name="QB",
        position="QB",
        usage_weight=1.0,
        inputs=physical,
        capability_inputs=_capability("qb", "QB", 68.0),
    )
    high, high_trace = compile_v13_player_identity(
        player_id="qb",
        name="QB",
        position="QB",
        usage_weight=1.0,
        inputs=physical,
        capability_inputs=_capability("qb", "QB", 96.0),
    )
    assert high.efficiency > low.efficiency
    assert high_trace.primary_skill > low_trace.primary_skill
    assert high_trace.rich_capability_active is True


def test_rich_receiver_and_runner_capability_have_position_specific_effects() -> None:
    physical = PlayerMechanismInputs(active_probability=1.0, effectiveness_if_active=1.0)
    low_wr, _ = compile_v13_player_identity(
        player_id="wr",
        name="WR",
        position="WR",
        usage_weight=1.0,
        inputs=physical,
        capability_inputs=_capability("wr", "WR", 70.0),
    )
    high_wr, _ = compile_v13_player_identity(
        player_id="wr",
        name="WR",
        position="WR",
        usage_weight=1.0,
        inputs=physical,
        capability_inputs=_capability("wr", "WR", 94.0),
    )
    low_rb, _ = compile_v13_player_identity(
        player_id="rb",
        name="RB",
        position="RB",
        usage_weight=1.0,
        inputs=physical,
        capability_inputs=_capability("rb", "RB", 70.0),
    )
    high_rb, _ = compile_v13_player_identity(
        player_id="rb",
        name="RB",
        position="RB",
        usage_weight=1.0,
        inputs=physical,
        capability_inputs=_capability("rb", "RB", 94.0),
    )
    assert high_wr.efficiency > low_wr.efficiency
    assert high_rb.efficiency > low_rb.efficiency
    assert high_rb.turnover_security > low_rb.turnover_security


def test_team_identity_authority_connects_qb_and_live_team_run_context() -> None:
    qb_state = PlayerState("qb", "QB", "QB", "A", qb_pass_share=1.0)
    rb_state = PlayerState("rb", "RB", "RB", "A", rush_share=0.8)
    wr_state = PlayerState("wr", "WR", "WR", "A", target_share=0.8)
    pool = TeamPlayerPool("A", (qb_state, rb_state, wr_state))
    base = TeamIdentity(
        team_id="A",
        quarterback=PlayerIdentity("qb", "QB", "QB"),
        rushers=(PlayerIdentity("rb", "RB", "RB"),),
        receivers=(PlayerIdentity("wr", "WR", "WR"),),
        pass_efficiency=1.0,
        rush_efficiency=1.0,
    )
    reality = {
        player_id: PlayerMechanismInputs(
            active_probability=1.0,
            effectiveness_if_active=1.0,
        )
        for player_id in ("qb", "rb", "wr")
    }
    units = (
        _capability("qb", "QB", 96.0),
        _capability("rb", "RB", 92.0),
        _capability("wr", "WR", 92.0),
    )
    state = TeamState(
        team_id="A",
        opponent_id="B",
        offensive_epa_per_play=0.18,
        offensive_success_rate=0.50,
        offensive_explosive_rate=0.14,
    )

    enhanced, trace = apply_v13_team_identity_authority(
        base,
        pool=pool,
        reality=reality,
        unit_players=units,
        state=state,
    )

    assert enhanced.pass_efficiency > base.pass_efficiency
    assert enhanced.rush_efficiency > base.rush_efficiency
    assert enhanced.receivers[0].efficiency > 1.0
    assert enhanced.rushers[0].efficiency > 1.0
    assert trace.live_rush_context > 1.0
    assert trace.players_with_capability_evidence == 3
