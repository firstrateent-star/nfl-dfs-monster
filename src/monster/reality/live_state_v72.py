from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import blake2b
from pathlib import Path

import numpy as np
import polars as pl

from monster.reality.participation_authority_v72 import (
    participation_multiplier,
    role_for,
)
from monster.sim.matchup_kernel import DefensiveUnit
from monster.sim.play_kernel import PlayerIdentity, TeamIdentity
from monster.sim.reality_snap_v5 import event_metadata


@dataclass
class LivePlayerStateV72:
    status: str = "active"
    multiplier: float = 1.0
    mutation_snap_key: str = ""


@dataclass
class LiveWorldStateV72:
    game: str
    players: dict[str, LivePlayerStateV72]


_ENABLED = False
_TEAM_IDENTITIES: dict[str, dict[str, PlayerIdentity]] = {}
_WORLD_STATES: dict[int, LiveWorldStateV72] = {}
_ACTIVE_WORLD_KEY: int | None = None
_MUTATION_ROWS: list[dict[str, object]] = []


_EXIT_RISK = {
    "QB": 0.00080,
    "RB": 0.00070,
    "FB": 0.00055,
    "WR": 0.00055,
    "TE": 0.00055,
    "OL": 0.00025,
    "LT": 0.00025,
    "LG": 0.00025,
    "C": 0.00025,
    "RG": 0.00025,
    "RT": 0.00025,
    "DL": 0.00045,
    "DE": 0.00045,
    "DT": 0.00045,
    "NT": 0.00045,
    "EDGE": 0.00045,
    "LB": 0.00045,
    "ILB": 0.00045,
    "OLB": 0.00045,
    "MLB": 0.00045,
    "CB": 0.00040,
    "DB": 0.00040,
    "S": 0.00040,
    "FS": 0.00040,
    "SS": 0.00040,
}
_LIMITED_MULTIPLIER = 0.56


def configure_live_state_v72() -> None:
    global _ENABLED
    _ENABLED = True


def enabled() -> bool:
    return _ENABLED


def register_team_identities_v72(
    team_id: str,
    identities: dict[str, PlayerIdentity],
) -> None:
    if not _ENABLED:
        return
    _TEAM_IDENTITIES[str(team_id)] = dict(identities)


def _world_key(rng: np.random.Generator) -> int:
    return id(rng)


def _game_id(state: object) -> str:
    return (
        f"{getattr(state, 'away_team_id', '')}@"
        f"{getattr(state, 'home_team_id', '')}"
    )


def _world(state: object, rng: np.random.Generator) -> LiveWorldStateV72:
    global _ACTIVE_WORLD_KEY
    key = _world_key(rng)
    game = _game_id(state)
    current = _WORLD_STATES.get(key)
    # A new RNG object is normally created per world. The explicit game check
    # protects against Python id reuse in long benchmark processes.
    if current is None or current.game != game:
        current = LiveWorldStateV72(game=game, players={})
        _WORLD_STATES[key] = current
    _ACTIVE_WORLD_KEY = key
    return current


def _player_state(world: LiveWorldStateV72, player_id: str) -> LivePlayerStateV72:
    return world.players.setdefault(str(player_id), LivePlayerStateV72())


def participation_multiplier_live(player_id: str) -> float:
    if not _ENABLED or _ACTIVE_WORLD_KEY is None:
        return 1.0
    world = _WORLD_STATES.get(_ACTIVE_WORLD_KEY)
    if world is None:
        return 1.0
    state = world.players.get(str(player_id))
    return 1.0 if state is None else float(state.multiplier)


def _is_out(world: LiveWorldStateV72, player_id: str) -> bool:
    return _player_state(world, player_id).status == "out"


def _role_sort(player: PlayerIdentity) -> tuple[float, str]:
    role = role_for(player.player_id)
    rank = 99 if role is None or role.depth_rank <= 0 else role.depth_rank
    current = participation_multiplier(player.player_id, unit="offense")
    return (rank - 0.25 * current, player.player_id)


def _backup_qb(
    team_id: str,
    world: LiveWorldStateV72,
    starter_id: str,
) -> PlayerIdentity | None:
    candidates = [
        player
        for player in _TEAM_IDENTITIES.get(str(team_id), {}).values()
        if player.position.upper() == "QB"
        and player.player_id != starter_id
        and not _is_out(world, player.player_id)
    ]
    if not candidates:
        return None
    return sorted(candidates, key=_role_sort)[0]


def _refill_receivers(
    team_id: str,
    receivers: tuple[PlayerIdentity, ...],
    world: LiveWorldStateV72,
    *,
    minimum: int = 6,
) -> tuple[PlayerIdentity, ...]:
    kept = [
        player
        for player in receivers
        if not _is_out(world, player.player_id)
    ]
    ids = {player.player_id for player in kept}
    candidates = [
        player
        for player in _TEAM_IDENTITIES.get(str(team_id), {}).values()
        if player.position.upper() in {"RB", "FB", "WR", "TE"}
        and player.player_id not in ids
        and not _is_out(world, player.player_id)
    ]
    candidates.sort(key=_role_sort)
    kept.extend(candidates[: max(minimum - len(kept), 0)])
    return tuple(kept)


def apply_live_state_v72(
    offense: TeamIdentity,
    defense: DefensiveUnit | None,
    *,
    state: object,
    rng: np.random.Generator,
) -> tuple[TeamIdentity, DefensiveUnit | None]:
    if not _ENABLED:
        return offense, defense
    world = _world(state, rng)

    quarterback = offense.quarterback
    if _is_out(world, quarterback.player_id):
        replacement = _backup_qb(offense.team_id, world, quarterback.player_id)
        if replacement is not None:
            quarterback = replacement

    receivers = _refill_receivers(offense.team_id, offense.receivers, world)
    rushers = tuple(
        player
        for player in offense.rushers
        if not _is_out(world, player.player_id)
        and player.player_id != offense.quarterback.player_id
    )
    if all(player.player_id != quarterback.player_id for player in rushers):
        rushers = (*rushers, replace(quarterback, usage_weight=0.001))

    active_offense = replace(
        offense,
        quarterback=quarterback,
        receivers=receivers or offense.receivers,
        rushers=rushers or (quarterback,),
    )

    if defense is None:
        return active_offense, None

    active_front = tuple(
        player for player in defense.front if not _is_out(world, player.player_id)
    )
    active_coverage = tuple(
        player for player in defense.coverage if not _is_out(world, player.player_id)
    )
    active_defense = replace(
        defense,
        front=active_front or defense.front,
        coverage=active_coverage or defense.coverage,
    )
    return active_offense, active_defense


def _unit_position(player_id: str) -> str:
    role = role_for(player_id)
    if role is None:
        return ""
    if role.position_group:
        return role.position_group
    return role.position


def _risk_modifier(player_id: str) -> float:
    role = role_for(player_id)
    if role is None:
        return 1.0
    injury = 80.0 if role.madden_injury is None else role.madden_injury
    stamina = 82.0 if role.madden_stamina is None else role.madden_stamina
    injury_factor = float(np.clip(np.exp((80.0 - injury) / 35.0), 0.65, 1.55))
    stamina_factor = float(np.clip(np.exp((82.0 - stamina) / 55.0), 0.78, 1.32))
    return injury_factor * stamina_factor


def _stable_uniform(snap_key: str, player_id: str, channel: str) -> float:
    digest = blake2b(
        f"{snap_key}:{player_id}:{channel}".encode("utf-8"),
        digest_size=8,
    ).digest()
    return (int.from_bytes(digest, "big") + 0.5) / (2**64)


def observe_live_mutations_v72(
    event: object,
    *,
    state: object,
    rng: np.random.Generator,
) -> None:
    if not _ENABLED:
        return
    world = _world(state, rng)
    meta = event_metadata(event)
    snap_key = str(meta.get("snap_key") or "")
    if not snap_key:
        return
    participants = tuple(
        dict.fromkeys(
            [
                *[str(x) for x in meta.get("offense_participant_ids", ())],
                *[str(x) for x in meta.get("defense_participant_ids", ())],
            ]
        )
    )
    for player_id in participants:
        current = _player_state(world, player_id)
        if current.status == "out":
            continue
        position = _unit_position(player_id)
        exit_risk = _EXIT_RISK.get(position, 0.00035) * _risk_modifier(player_id)
        limited_risk = min(exit_risk * 2.2, 0.0045)
        draw = _stable_uniform(snap_key, player_id, "availability")

        new_status = ""
        new_multiplier = current.multiplier
        if draw < exit_risk:
            new_status = "out"
            new_multiplier = 0.0
        elif current.status == "active" and draw < exit_risk + limited_risk:
            new_status = "limited"
            new_multiplier = _LIMITED_MULTIPLIER

        if not new_status:
            continue
        current.status = new_status
        current.multiplier = new_multiplier
        current.mutation_snap_key = snap_key
        role = role_for(player_id)
        _MUTATION_ROWS.append(
            {
                "game": world.game,
                "snap_key": snap_key,
                "player_id": player_id,
                "team": "" if role is None else role.team_id,
                "position": position,
                "mutation": new_status,
                "participation_multiplier_after": new_multiplier,
                "exit_risk": exit_risk,
                "limited_risk": limited_risk,
            }
        )


def mutation_rows_v72() -> list[dict[str, object]]:
    return list(_MUTATION_ROWS)


def write_live_state_telemetry_v72(out: Path) -> None:
    if not _MUTATION_ROWS:
        return
    out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(_MUTATION_ROWS).write_csv(out / "live_mutations_v72.csv")
