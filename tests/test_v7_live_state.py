from __future__ import annotations

import pytest

from monster.reality.availability import (
    ExitReason,
    GameAvailabilityState,
)
from monster.reality.ledger import LedgerEventKind, LedgerFidelity, ParticipantSnapshot
from monster.reality.native_ledger import NativeRealityLedgerBuilder
from monster.reality.world_state import PregameWorld, TeamPregameState, WorldKey
from monster.sim.football_state import FootballState, apply_scrimmage_yards
from monster.sim.play_kernel import PlayEvent, PlayType


def _world() -> PregameWorld:
    return PregameWorld(
        key=WorldKey(game_id="AWY@HME", world_index=7, seed=7002001),
        away=TeamPregameState(
            team_id="AWY",
            starting_qb_id="awy-qb1",
            active_player_ids=("awy-qb1", "awy-qb2", "awy-rb"),
        ),
        home=TeamPregameState(
            team_id="HME",
            starting_qb_id="hme-qb1",
            active_player_ids=("hme-qb1", "hme-qb2", "hme-rb"),
        ),
    )


def test_live_qb_exit_can_atomically_promote_active_backup() -> None:
    state = GameAvailabilityState.from_pregame(_world())
    changed = state.exit_player(
        team_id="AWY",
        player_id="awy-qb1",
        reason=ExitReason.INJURY,
        quarter=2,
        seconds_remaining=947,
        replacement_qb_id="awy-qb2",
    )

    assert state.away.current_qb_id == "awy-qb1"
    assert changed.away.current_qb_id == "awy-qb2"
    assert "awy-qb1" not in changed.away.active_player_ids
    assert "awy-qb1" in changed.away.exited_player_ids
    assert changed.transitions[-1].replacement_player_id == "awy-qb2"


def test_live_qb_replacement_must_already_exist_in_pregame_world() -> None:
    state = GameAvailabilityState.from_pregame(_world())
    with pytest.raises(ValueError):
        state.exit_player(
            team_id="AWY",
            player_id="awy-qb1",
            reason=ExitReason.CONCUSSION,
            quarter=1,
            seconds_remaining=3200,
            replacement_qb_id="street-free-agent",
        )


def test_non_qb_exit_cannot_smuggle_in_qb_replacement() -> None:
    state = GameAvailabilityState.from_pregame(_world())
    with pytest.raises(ValueError):
        state.exit_player(
            team_id="AWY",
            player_id="awy-rb",
            reason=ExitReason.INJURY,
            quarter=3,
            seconds_remaining=1200,
            replacement_qb_id="awy-qb2",
        )


def test_native_ledger_requires_exact_11v11_and_persists_pre_post_state() -> None:
    world = _world().key
    builder = NativeRealityLedgerBuilder(world=world, source_runtime="native-test")
    before = FootballState(
        possession="AWY",
        defense="HME",
        quarter=1,
        seconds_remaining=3500,
        yardline_100=25.0,
        down=1,
        distance=10.0,
        away_team_id="AWY",
        home_team_id="HME",
    )
    event = PlayEvent(
        play_type=PlayType.RUN,
        elapsed_seconds=28,
        yards=6.0,
        rusher_id="awy-rb",
    )
    after = apply_scrimmage_yards(before, event.yards, event.elapsed_seconds)
    participants = ParticipantSnapshot(
        offense_player_ids=tuple(f"o{i}" for i in range(11)),
        defense_player_ids=tuple(f"d{i}" for i in range(11)),
    )
    builder.record_snap(
        pre_state=before,
        post_state=after,
        event=event,
        participants=participants,
        drive_index=0,
    )
    ledger = builder.close_game(
        final_state=after,
        drives=1,
        went_to_overtime=False,
    )

    snap = ledger.snap_records[0]
    assert snap.fidelity == LedgerFidelity.NATIVE
    assert snap.pre_state is not None
    assert snap.post_state is not None
    assert snap.pre_state.down == 1
    assert snap.post_state.down == 2
    assert snap.post_state.yardline_100 == pytest.approx(31.0)


def test_native_ledger_rejects_partial_participant_world() -> None:
    builder = NativeRealityLedgerBuilder(world=_world().key, source_runtime="native-test")
    state = FootballState(
        possession="AWY",
        defense="HME",
        away_team_id="AWY",
        home_team_id="HME",
    )
    event = PlayEvent(play_type=PlayType.RUN, elapsed_seconds=20, yards=2.0, rusher_id="awy-rb")
    with pytest.raises(ValueError):
        builder.record_snap(
            pre_state=state,
            post_state=apply_scrimmage_yards(state, 2.0, 20),
            event=event,
            participants=ParticipantSnapshot(
                offense_player_ids=("one",),
                defense_player_ids=("two",),
            ),
            drive_index=0,
        )


def test_native_ledger_records_qb_exit_as_first_class_transition() -> None:
    availability = GameAvailabilityState.from_pregame(_world())
    changed = availability.exit_player(
        team_id="AWY",
        player_id="awy-qb1",
        reason=ExitReason.CONCUSSION,
        quarter=2,
        seconds_remaining=811,
        replacement_qb_id="awy-qb2",
    )
    transition = changed.transitions[-1]

    builder = NativeRealityLedgerBuilder(world=_world().key, source_runtime="native-test")
    builder.record_availability_transition(transition)
    final_state = FootballState(
        possession="AWY",
        defense="HME",
        quarter=2,
        seconds_remaining=811,
        away_team_id="AWY",
        home_team_id="HME",
    )
    ledger = builder.close_game(
        final_state=final_state,
        drives=1,
        went_to_overtime=False,
    )

    assert len(ledger.availability_records) == 1
    record = ledger.availability_records[0]
    assert record.event_kind == LedgerEventKind.AVAILABILITY_TRANSITION
    assert record.fidelity == LedgerFidelity.NATIVE
    assert record.availability_transition is not None
    assert record.availability_transition.player_id == "awy-qb1"
    assert record.availability_transition.replacement_player_id == "awy-qb2"
    assert record.availability_transition.reason == "concussion"
