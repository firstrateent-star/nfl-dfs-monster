from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from monster.feature_compile.v13_identity_bridge import compile_v13_player_identity
from monster.reality.availability import TeamLiveRoster
from monster.sim.play_kernel import PlayerIdentity, TeamIdentity


@dataclass(frozen=True)
class TeamIdentityCatalog:
    """Pregame catalog of compiled player capability identities for one team."""

    team_id: str
    players: tuple[PlayerIdentity, ...]

    def __post_init__(self) -> None:
        ids = [player.player_id for player in self.players]
        if len(ids) != len(set(ids)):
            raise ValueError("identity catalog player ids must be unique")

    def player(self, player_id: str) -> PlayerIdentity:
        for player in self.players:
            if player.player_id == player_id:
                return player
        raise KeyError(f"{player_id} is not in {self.team_id} identity catalog")


def compile_team_identity_catalog(
    *,
    team_id: str,
    pool: Any,
    reality: dict[str, Any],
    unit_players: tuple[Any, ...],
) -> TeamIdentityCatalog:
    """Compile every current player identity before a starter is selected.

    Usage values here are evidence needed by legacy PlayerIdentity construction.
    v7 does not interpret them as final opportunity ownership after participation.
    """

    capability = {player.player_id: player for player in unit_players}
    compiled: list[PlayerIdentity] = []
    for player in pool.players:
        usage = (
            float(player.qb_pass_share)
            if str(player.position).upper() == "QB"
            else max(float(player.target_share), float(player.rush_share), 0.001)
        )
        inputs = reality.get(player.player_id)
        if inputs is None:
            identity = PlayerIdentity(
                player_id=player.player_id,
                name=player.display_name,
                position=player.position,
                usage_weight=usage,
            )
        else:
            identity, _ = compile_v13_player_identity(
                player_id=player.player_id,
                name=player.display_name,
                position=player.position,
                usage_weight=usage,
                inputs=inputs,
                capability_inputs=capability.get(player.player_id),
            )
        compiled.append(identity)
    return TeamIdentityCatalog(team_id=team_id, players=tuple(compiled))


def apply_live_roster_identity(
    *,
    team: TeamIdentity,
    catalog: TeamIdentityCatalog,
    roster: TeamLiveRoster,
) -> TeamIdentity:
    """Project live availability onto a team identity without cloning starter skill."""

    if team.team_id != catalog.team_id or team.team_id != roster.team_id:
        raise ValueError("team, catalog and live roster must refer to the same team")
    if roster.current_qb_id is None:
        raise ValueError("live team requires a current QB before a snap can be created")

    active = set(roster.active_player_ids)
    quarterback = catalog.player(roster.current_qb_id)
    if quarterback.position.upper() != "QB":
        raise ValueError("current_qb_id must resolve to a QB identity")

    rushers = tuple(player for player in team.rushers if player.player_id in active)
    receivers = tuple(player for player in team.receivers if player.player_id in active)

    if not receivers:
        raise ValueError("live team has no eligible receiving identities")
    if not rushers:
        # Scrambles and sneaks use the quarterback identity directly. A native v7
        # designed-run engine will separately decide whether a QB is a legal
        # designed-run participant instead of inventing RB workload here.
        rushers = (quarterback,)

    return replace(
        team,
        quarterback=quarterback,
        rushers=rushers,
        receivers=receivers,
    )
