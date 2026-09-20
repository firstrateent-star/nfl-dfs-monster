from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import blake2b

import numpy as np
import polars as pl

from monster.sim import current_role_guard_v638 as v638
from monster.sim import snap_ecology
from monster.reality.world_availability_v722 import active_this_world_v722


@dataclass(frozen=True)
class CurrentParticipationRoleV72:
    player_id: str
    team_id: str
    position: str
    position_group: str
    depth_position: str
    depth_rank: int
    conditional_offense_snap_share: float
    conditional_defense_snap_share: float
    conditional_special_teams_snap_share: float
    active_probability: float
    status: str
    madden_injury: float | None
    madden_stamina: float | None
    madden_return: float | None
    madden_kick_power: float | None
    madden_kick_accuracy: float | None


_ROLES: dict[str, CurrentParticipationRoleV72] = {}
_ENABLED = False

OL_SEATS = ("LT", "LG", "C", "RG", "RT")


def _number(row: dict, key: str, default: float = 0.0) -> float:
    value = row.get(key)
    try:
        return default if value is None else float(value)
    except (TypeError, ValueError):
        return default


def _optional_number(row: dict, key: str) -> float | None:
    value = row.get(key)
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def configure_participation_authority_v72(personnel: pl.DataFrame) -> None:
    global _ENABLED, _ROLES
    roles: dict[str, CurrentParticipationRoleV72] = {}
    for row in personnel.to_dicts():
        player_id = str(
            row.get("gsis_id")
            or row.get("pfr_id")
            or row.get("display_name")
            or ""
        )
        if not player_id:
            continue
        try:
            depth_rank = int(float(row.get("depth_rank") or 0))
        except (TypeError, ValueError):
            depth_rank = 0
        roles[player_id] = CurrentParticipationRoleV72(
            player_id=player_id,
            team_id=str(row.get("team_id") or ""),
            position=str(row.get("position") or "").upper(),
            position_group=str(row.get("position_group") or "").upper(),
            depth_position=str(row.get("depth_position") or "").upper(),
            depth_rank=depth_rank,
            conditional_offense_snap_share=float(
                np.clip(_number(row, "conditional_offense_snap_share"), 0.0, 1.0)
            ),
            conditional_defense_snap_share=float(
                np.clip(_number(row, "conditional_defense_snap_share"), 0.0, 1.0)
            ),
            conditional_special_teams_snap_share=float(
                np.clip(_number(row, "conditional_special_teams_snap_share"), 0.0, 1.0)
            ),
            active_probability=float(
                np.clip(_number(row, "game_day_active_probability", 1.0), 0.0, 1.0)
            ),
            status=str(row.get("status") or "").upper(),
            madden_injury=_optional_number(row, "madden_injury"),
            madden_stamina=_optional_number(row, "madden_stamina"),
            madden_return=_optional_number(row, "madden_return"),
            madden_kick_power=_optional_number(row, "madden_kick_power"),
            madden_kick_accuracy=_optional_number(row, "madden_kick_accuracy"),
        )
    _ROLES = roles
    _ENABLED = True


def install_participation_authority_v72(personnel: pl.DataFrame) -> None:
    """Preserve inherited V7.1 role priors and add one current participation authority."""

    v638.install_current_role_guard_v638(personnel)
    configure_participation_authority_v72(personnel)


def role_for(player_id: str) -> CurrentParticipationRoleV72 | None:
    return _ROLES.get(str(player_id))


def enabled() -> bool:
    return _ENABLED


def _depth_multiplier(rank: int) -> float:
    if rank == 1:
        return 1.34
    if rank == 2:
        return 0.86
    if rank == 3:
        return 0.58
    if rank >= 4:
        return 0.38
    return 0.78


def participation_multiplier(
    player_id: str,
    *,
    unit: str,
) -> float:
    """Current role authority conditional on the player being in the candidate universe."""

    role = role_for(player_id)
    if role is None:
        return 1.0
    if role.status != "ACT" or role.active_probability < 0.10:
        return 0.01
    world_active = active_this_world_v722(player_id, role.active_probability)
    if world_active is False:
        return 0.01
    availability = 1.0 if world_active is True else role.active_probability
    share = {
        "offense": role.conditional_offense_snap_share,
        "defense": role.conditional_defense_snap_share,
        "special": role.conditional_special_teams_snap_share,
    }.get(unit, 0.0)
    current = 0.62 * _depth_multiplier(role.depth_rank) + 0.38 * np.sqrt(
        max(share, 0.0)
    )
    return float(np.clip(availability * current, 0.03, 1.55))


def starter_authority(player_id: str, *, unit: str = "offense") -> float:
    role = role_for(player_id)
    if role is None or role.status != "ACT":
        return 0.0
    world_active = active_this_world_v722(player_id, role.active_probability)
    if world_active is False:
        return 0.0
    availability = 1.0 if world_active is True else role.active_probability
    share = (
        role.conditional_offense_snap_share
        if unit == "offense"
        else role.conditional_defense_snap_share
    )
    starter = 1.0 if role.depth_rank == 1 else 0.0
    return float(
        np.clip(
            availability * (0.72 * starter + 0.28 * np.sqrt(max(share, 0.0))),
            0.0,
            1.0,
        )
    )


def canonical_ol_seat(player_id: str, fallback_position: str = "OL") -> str:
    role = role_for(player_id)
    if role is not None and role.depth_position in OL_SEATS:
        return role.depth_position
    position = str(fallback_position).upper()
    return position if position in OL_SEATS else "OL"


def _ol_candidate_key(player: object, seat: str) -> tuple[float, float, str]:
    player_id = str(getattr(player, "player_id", ""))
    role = role_for(player_id)
    if role is None:
        active = 1.0
    else:
        world_active = active_this_world_v722(player_id, role.active_probability)
        availability = 1.0 if world_active is True else (0.0 if world_active is False else role.active_probability)
        active = float(role.status == "ACT") * availability
    rank = 99 if role is None or role.depth_rank <= 0 else role.depth_rank
    seat_match = 1.0 if canonical_ol_seat(player_id, getattr(player, "position", "")) == seat else 0.0
    snap = float(getattr(player, "offense_snap_share", 0.0) or 0.0)
    # Seat ownership is lexicographically stronger than stale snap history.
    authority = 10.0 * seat_match + 3.0 * active + (2.0 if rank == 1 else 0.0)
    return (-authority, -snap, player_id)


def _choose_offensive_line(players: Iterable[object]) -> tuple[object, ...]:
    rows = [
        player
        for player in players
        if str(getattr(player, "position", "")).upper()
        in {"LT", "LG", "C", "RG", "RT", "T", "OT", "G", "OG", "OL"}
    ]
    if not rows:
        return ()

    chosen: list[object] = []
    chosen_ids: set[str] = set()
    for seat in OL_SEATS:
        candidates = [
            player
            for player in rows
            if str(getattr(player, "player_id", "")) not in chosen_ids
            and canonical_ol_seat(
                str(getattr(player, "player_id", "")),
                str(getattr(player, "position", "")),
            )
            == seat
        ]
        if not candidates:
            continue
        selected = min(candidates, key=lambda player: _ol_candidate_key(player, seat))
        chosen.append(selected)
        chosen_ids.add(str(getattr(selected, "player_id", "")))

    if len(chosen) < 5:
        remaining = [
            player
            for player in rows
            if str(getattr(player, "player_id", "")) not in chosen_ids
        ]
        remaining.sort(
            key=lambda player: (
                -starter_authority(str(getattr(player, "player_id", ""))),
                -float(getattr(player, "offense_snap_share", 0.0) or 0.0),
                str(getattr(player, "player_id", "")),
            )
        )
        chosen.extend(remaining[: 5 - len(chosen)])
    return tuple(chosen[:5])


def register_team_units_v72(
    team_id: str,
    players: Iterable[object],
) -> snap_ecology.TeamSnapProfile:
    """Register exact OL seats before individual trench physics are compiled."""

    rows = list(players)
    for row in rows:
        player_id = str(getattr(row, "player_id", ""))
        if player_id:
            snap_ecology._PLAYER_TEAM[player_id] = team_id

    ol = _choose_offensive_line(rows)
    blockers = tuple(
        snap_ecology.BlockerProfile(
            player_id=str(getattr(row, "player_id", "")),
            position=canonical_ol_seat(
                str(getattr(row, "player_id", "")),
                str(getattr(row, "position", "OL")),
            ),
            pass_block=snap_ecology._signal(
                getattr(row, "pass_block_signal", None),
                getattr(row, "madden_pass_block", None),
            ),
            run_block=snap_ecology._signal(
                getattr(row, "run_block_signal", None),
                getattr(row, "madden_run_block", None),
            ),
            awareness=snap_ecology._rating(getattr(row, "madden_awareness", None)),
            stamina=snap_ecology._rating(
                getattr(row, "madden_stamina", None),
                82.0,
                10.0,
            ),
            snap_weight=float(
                np.clip(getattr(row, "offense_snap_share", 0.0) or 0.0, 0.0, 1.0)
            ),
        )
        for row in ol
    )

    helpers = [
        row
        for row in rows
        if str(getattr(row, "position", "")).upper() in {"RB", "FB", "TE"}
        and float(getattr(row, "offense_snap_share", 0.0) or 0.0) > 0.01
    ]
    protectors = tuple(
        snap_ecology.ProtectorProfile(
            player_id=str(getattr(row, "player_id", "")),
            position=str(getattr(row, "position", "")),
            pass_block=snap_ecology._rating(getattr(row, "madden_pass_block", None)),
            run_block=snap_ecology._rating(getattr(row, "madden_run_block", None)),
            receiving_value=float(
                np.mean(
                    [
                        snap_ecology._rating(
                            getattr(row, "madden_route_running", None)
                        ),
                        snap_ecology._rating(getattr(row, "madden_catching", None)),
                    ]
                )
            ),
            snap_weight=float(
                np.clip(getattr(row, "offense_snap_share", 0.0) or 0.0, 0.0, 1.0)
            ),
        )
        for row in helpers
    )
    profile = snap_ecology.TeamSnapProfile(team_id, blockers, protectors)
    snap_ecology._TEAM_PROFILES[team_id] = profile
    return profile


def weighted_without_replacement_v72(
    players: tuple[object, ...],
    count: int,
    *,
    key: str,
    exposure: bool = False,
) -> tuple[object, ...]:
    """Use current role as participation authority without replacing football skill."""

    if count <= 0 or not players:
        return ()
    unit = "defense" if exposure else "offense"
    ranked: list[tuple[float, str, object]] = []
    for player in players:
        player_id = str(getattr(player, "player_id", ""))
        inherited = (
            max(float(getattr(player, "snap_weight", 0.0) or 0.0), 0.001)
            if exposure
            else max(float(getattr(player, "usage_weight", 0.0) or 0.0), 0.001)
        )
        authority = participation_multiplier(player_id, unit=unit)
        try:
            from monster.reality.live_state_v72 import participation_multiplier_live

            live = participation_multiplier_live(player_id)
        except ImportError:
            live = 1.0
        weight = max(inherited * authority * live, 1e-6)
        digest = blake2b(
            f"{key}:{player_id}".encode(),
            digest_size=8,
        ).digest()
        u = max((int.from_bytes(digest, "big") + 0.5) / (2**64), 1e-12)
        ranked.append((-np.log(u) / weight, player_id, player))
    ranked.sort(key=lambda row: (row[0], row[1]))
    return tuple(row[2] for row in ranked[: min(count, len(ranked))])
