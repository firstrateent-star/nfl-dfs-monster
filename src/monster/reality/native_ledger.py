from __future__ import annotations

from dataclasses import dataclass

from monster.reality.availability import AvailabilityTransition
from monster.reality.ledger import (
    AvailabilityTransitionSnapshot,
    FootballStateSnapshot,
    LedgerEventKind,
    LedgerFidelity,
    LedgerRecord,
    ParticipantSnapshot,
    PlayOutcomeSnapshot,
    RealityLedger,
)
from monster.reality.world_state import WorldKey
from monster.sim.football_state import FootballState
from monster.sim.play_kernel import PassResult, PlayEvent, PlayType


def _state_snapshot(state: FootballState) -> FootballStateSnapshot:
    return FootballStateSnapshot(
        possession=state.possession,
        defense=state.defense,
        quarter=int(state.quarter),
        seconds_remaining=int(state.seconds_remaining),
        yardline_100=float(state.yardline_100),
        down=int(state.down),
        distance=float(state.distance),
        away_score=int(state.away_score),
        home_score=int(state.home_score),
    )


def _play_snapshot(event: PlayEvent) -> PlayOutcomeSnapshot:
    play_type = event.play_type.value if isinstance(event.play_type, PlayType) else str(event.play_type)
    pass_result = (
        event.pass_result.value
        if isinstance(event.pass_result, PassResult)
        else (None if event.pass_result is None else str(event.pass_result))
    )
    return PlayOutcomeSnapshot(
        play_type=play_type,
        pass_result=pass_result,
        yards=float(event.yards),
        elapsed_seconds=int(event.elapsed_seconds),
        passer_id=event.passer_id,
        target_id=event.target_id,
        rusher_id=event.rusher_id,
        fumbler_id=event.fumbler_id,
        primary_defender_id=event.primary_defender_id,
        touchdown=bool(event.touchdown),
        turnover=bool(event.turnover),
        pressured=bool(event.pressured),
        stuffed=bool(event.stuffed),
        air_yards=float(event.air_yards),
        yards_after_catch=float(event.yards_after_catch),
        yards_before_contact=float(event.yards_before_contact),
        yards_after_contact=float(event.yards_after_contact),
        pass_depth_category=event.pass_depth_category,
        run_geometry_category=event.run_geometry_category,
    )


@dataclass
class NativeRealityLedgerBuilder:
    """Mutable writer whose completed artifact is an immutable RealityLedger."""

    world: WorldKey
    source_runtime: str

    def __post_init__(self) -> None:
        self._records: list[LedgerRecord] = []
        self._closed = False

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("native ledger is already closed")

    def record_snap(
        self,
        *,
        pre_state: FootballState,
        post_state: FootballState,
        event: PlayEvent,
        participants: ParticipantSnapshot,
        drive_index: int,
    ) -> None:
        self._require_open()
        if not participants.exact_11v11:
            raise ValueError("native v7 snap records require exact 11v11 participants")
        self._records.append(
            LedgerRecord(
                sequence=len(self._records),
                world_id=self.world.world_id,
                event_kind=LedgerEventKind.SNAP_RESULT,
                fidelity=LedgerFidelity.NATIVE,
                source_runtime=self.source_runtime,
                drive_index=drive_index,
                offense_team_id=pre_state.possession,
                defense_team_id=pre_state.defense,
                pre_state=_state_snapshot(pre_state),
                post_state=_state_snapshot(post_state),
                participants=participants,
                play=_play_snapshot(event),
            )
        )

    def record_availability_transition(
        self,
        transition: AvailabilityTransition,
    ) -> None:
        self._require_open()
        self._records.append(
            LedgerRecord(
                sequence=len(self._records),
                world_id=self.world.world_id,
                event_kind=LedgerEventKind.AVAILABILITY_TRANSITION,
                fidelity=LedgerFidelity.NATIVE,
                source_runtime=self.source_runtime,
                offense_team_id=transition.team_id,
                availability_transition=AvailabilityTransitionSnapshot(
                    team_id=transition.team_id,
                    player_id=transition.player_id,
                    reason=transition.reason.value,
                    quarter=int(transition.quarter),
                    seconds_remaining=int(transition.seconds_remaining),
                    replacement_player_id=transition.replacement_player_id,
                ),
            )
        )

    def close_game(
        self,
        *,
        final_state: FootballState,
        drives: int,
        went_to_overtime: bool,
    ) -> RealityLedger:
        self._require_open()
        self._records.append(
            LedgerRecord(
                sequence=len(self._records),
                world_id=self.world.world_id,
                event_kind=LedgerEventKind.GAME_FINAL,
                fidelity=LedgerFidelity.NATIVE,
                source_runtime=self.source_runtime,
                post_state=_state_snapshot(final_state),
                payload=(
                    ("went_to_overtime", str(bool(went_to_overtime)).lower()),
                    ("drives", str(int(drives))),
                ),
            )
        )
        self._closed = True
        ledger = RealityLedger(world=self.world, records=tuple(self._records))
        ledger.assert_contract()
        return ledger
