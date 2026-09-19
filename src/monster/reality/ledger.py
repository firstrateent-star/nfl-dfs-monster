from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from monster.reality.world_state import WorldKey
from monster.sim.play_kernel import PassResult, PlayType
from monster.sim.reality_snap_v5 import event_metadata

if TYPE_CHECKING:
    from monster.sim.game_loop_v13 import GameResultV13
    from monster.sim.play_kernel import TeamIdentity


class LedgerFidelity(StrEnum):
    """How completely a record was observed at generation time."""

    NATIVE = "native"
    V6_ADAPTER = "v6_adapter"


class LedgerEventKind(StrEnum):
    SNAP_RESULT = "snap_result"
    DRIVE_TERMINAL = "drive_terminal"
    GAME_FINAL = "game_final"


@dataclass(frozen=True)
class ParticipantSnapshot:
    offense_package: str | None = None
    defense_package: str | None = None
    offense_player_ids: tuple[str, ...] = ()
    defense_player_ids: tuple[str, ...] = ()
    offense_alignment: tuple[str, ...] = ()
    defense_alignment: tuple[str, ...] = ()
    planned_rusher_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.offense_player_ids) != len(set(self.offense_player_ids)):
            raise ValueError("offensive snap participants must be unique")
        if len(self.defense_player_ids) != len(set(self.defense_player_ids)):
            raise ValueError("defensive snap participants must be unique")

    @property
    def exact_11v11(self) -> bool:
        return (
            len(self.offense_player_ids) == 11
            and len(self.defense_player_ids) == 11
            and len(set(self.offense_player_ids)) == 11
            and len(set(self.defense_player_ids)) == 11
        )


@dataclass(frozen=True)
class FootballStateSnapshot:
    possession: str | None = None
    defense: str | None = None
    quarter: int | None = None
    seconds_remaining: int | None = None
    yardline_100: float | None = None
    down: int | None = None
    distance: float | None = None
    away_score: int | None = None
    home_score: int | None = None


@dataclass(frozen=True)
class PlayOutcomeSnapshot:
    play_type: str
    pass_result: str | None = None
    yards: float = 0.0
    elapsed_seconds: int = 0
    passer_id: str | None = None
    target_id: str | None = None
    rusher_id: str | None = None
    fumbler_id: str | None = None
    primary_defender_id: str | None = None
    touchdown: bool = False
    turnover: bool = False
    pressured: bool = False
    stuffed: bool = False
    air_yards: float = 0.0
    yards_after_catch: float = 0.0
    yards_before_contact: float = 0.0
    yards_after_contact: float = 0.0
    pass_depth_category: str | None = None
    run_geometry_category: str | None = None


@dataclass(frozen=True)
class DriveTerminalSnapshot:
    offense_team_id: str
    defense_team_id: str
    terminal: str
    points: int
    scrimmage_plays: int
    net_scrimmage_yards: float
    first_downs: int
    start_quarter: int
    start_seconds_remaining: int
    start_yardline_100: float
    end_quarter: int
    end_seconds_remaining: int
    end_yardline_100: float


@dataclass(frozen=True)
class LedgerRecord:
    sequence: int
    world_id: str
    event_kind: LedgerEventKind
    fidelity: LedgerFidelity
    source_runtime: str
    drive_index: int | None = None
    offense_team_id: str | None = None
    defense_team_id: str | None = None
    pre_state: FootballStateSnapshot | None = None
    post_state: FootballStateSnapshot | None = None
    participants: ParticipantSnapshot | None = None
    play: PlayOutcomeSnapshot | None = None
    drive_terminal: DriveTerminalSnapshot | None = None
    payload: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class RealityLedger:
    world: WorldKey
    records: tuple[LedgerRecord, ...]
    schema_version: str = "v7.0.0"

    def __post_init__(self) -> None:
        expected = list(range(len(self.records)))
        observed = [record.sequence for record in self.records]
        if observed != expected:
            raise ValueError("ledger records must be contiguous and ordered from sequence zero")
        if any(record.world_id != self.world.world_id for record in self.records):
            raise ValueError("every ledger record must belong to the same world")

    def digest(self) -> str:
        body = json.dumps(
            {
                "schema_version": self.schema_version,
                "world": asdict(self.world),
                "records": [asdict(record) for record in self.records],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(body.encode("utf-8")).hexdigest()

    @property
    def snap_records(self) -> tuple[LedgerRecord, ...]:
        return tuple(
            record
            for record in self.records
            if record.event_kind == LedgerEventKind.SNAP_RESULT
        )

    @property
    def drive_records(self) -> tuple[LedgerRecord, ...]:
        return tuple(
            record
            for record in self.records
            if record.event_kind == LedgerEventKind.DRIVE_TERMINAL
        )

    def assert_contract(self) -> None:
        if not self.records:
            raise AssertionError("a completed game ledger cannot be empty")
        if self.records[-1].event_kind != LedgerEventKind.GAME_FINAL:
            raise AssertionError("the final ledger record must close the game")
        for record in self.snap_records:
            if record.play is None:
                raise AssertionError("snap-result records require a play outcome")
            if record.participants is not None:
                participants = record.participants
                if participants.offense_player_ids and len(participants.offense_player_ids) != 11:
                    raise AssertionError("observed offensive participants must be exact 11")
                if participants.defense_player_ids and len(participants.defense_player_ids) != 11:
                    raise AssertionError("observed defensive participants must be exact 11")


def _player_team_map(away: TeamIdentity, home: TeamIdentity) -> dict[str, str]:
    out: dict[str, str] = {}
    for team in (away, home):
        players = (team.quarterback, *team.rushers, *team.receivers)
        for player in players:
            out.setdefault(player.player_id, team.team_id)
    return out


def _play_offense(event: Any, player_teams: dict[str, str]) -> str | None:
    for player_id in (event.passer_id, event.rusher_id, event.target_id):
        if player_id is not None and player_id in player_teams:
            return player_teams[player_id]
    return None


def _participant_snapshot(event: Any) -> ParticipantSnapshot | None:
    meta = event_metadata(event)
    if not meta:
        return None
    return ParticipantSnapshot(
        offense_package=meta.get("offense_package"),
        defense_package=meta.get("defense_package"),
        offense_player_ids=tuple(meta.get("offense_participant_ids", ())),
        defense_player_ids=tuple(meta.get("defense_participant_ids", ())),
        offense_alignment=tuple(meta.get("offense_alignment", ())),
        defense_alignment=tuple(meta.get("defense_alignment", ())),
        planned_rusher_ids=tuple(meta.get("planned_rusher_ids", ())),
    )


def _play_snapshot(event: Any) -> PlayOutcomeSnapshot:
    pass_result = event.pass_result
    if isinstance(pass_result, PassResult):
        pass_result_value: str | None = pass_result.value
    else:
        pass_result_value = None if pass_result is None else str(pass_result)
    play_type = event.play_type
    play_type_value = play_type.value if isinstance(play_type, PlayType) else str(play_type)
    return PlayOutcomeSnapshot(
        play_type=play_type_value,
        pass_result=pass_result_value,
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


def ledger_from_v6_result(
    *,
    world: WorldKey,
    result: GameResultV13,
    away: TeamIdentity,
    home: TeamIdentity,
    source_runtime: str,
) -> RealityLedger:
    """Adapt frozen-v6 output to the v7 ledger contract without changing football.

    v6 did not natively persist every pre/post-snap state. Adapted snap records are
    therefore explicitly marked V6_ADAPTER. Native v7 execution will fill those
    states at generation time instead of reconstructing them afterward.
    """

    records: list[LedgerRecord] = []
    player_teams = _player_team_map(away, home)

    for event in result.plays:
        offense = _play_offense(event, player_teams)
        defense = None
        if offense == away.team_id:
            defense = home.team_id
        elif offense == home.team_id:
            defense = away.team_id
        records.append(
            LedgerRecord(
                sequence=len(records),
                world_id=world.world_id,
                event_kind=LedgerEventKind.SNAP_RESULT,
                fidelity=LedgerFidelity.V6_ADAPTER,
                source_runtime=source_runtime,
                offense_team_id=offense,
                defense_team_id=defense,
                participants=_participant_snapshot(event),
                play=_play_snapshot(event),
            )
        )

    for drive_index, trace in enumerate(result.drive_traces):
        records.append(
            LedgerRecord(
                sequence=len(records),
                world_id=world.world_id,
                event_kind=LedgerEventKind.DRIVE_TERMINAL,
                fidelity=LedgerFidelity.V6_ADAPTER,
                source_runtime=source_runtime,
                drive_index=drive_index,
                offense_team_id=trace.offense_team_id,
                defense_team_id=trace.defense_team_id,
                drive_terminal=DriveTerminalSnapshot(
                    offense_team_id=trace.offense_team_id,
                    defense_team_id=trace.defense_team_id,
                    terminal=getattr(trace.terminal, "value", str(trace.terminal)),
                    points=int(trace.points),
                    scrimmage_plays=int(trace.scrimmage_plays),
                    net_scrimmage_yards=float(trace.net_scrimmage_yards),
                    first_downs=int(trace.first_downs),
                    start_quarter=int(trace.start_quarter),
                    start_seconds_remaining=int(trace.start_seconds_remaining),
                    start_yardline_100=float(trace.start_yardline_100),
                    end_quarter=int(trace.end_quarter),
                    end_seconds_remaining=int(trace.end_seconds_remaining),
                    end_yardline_100=float(trace.end_yardline_100),
                ),
            )
        )

    final = result.final_state
    records.append(
        LedgerRecord(
            sequence=len(records),
            world_id=world.world_id,
            event_kind=LedgerEventKind.GAME_FINAL,
            fidelity=LedgerFidelity.V6_ADAPTER,
            source_runtime=source_runtime,
            post_state=FootballStateSnapshot(
                possession=final.possession,
                defense=final.defense,
                quarter=int(final.quarter),
                seconds_remaining=int(final.seconds_remaining),
                yardline_100=float(final.yardline_100),
                down=int(final.down),
                distance=float(final.distance),
                away_score=int(final.away_score),
                home_score=int(final.home_score),
            ),
            payload=(
                ("went_to_overtime", str(bool(result.went_to_overtime)).lower()),
                ("drives", str(int(result.drives))),
            ),
        )
    )

    ledger = RealityLedger(world=world, records=tuple(records))
    ledger.assert_contract()
    return ledger
