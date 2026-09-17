from __future__ import annotations

from dataclasses import dataclass, replace
from math import exp, log

import numpy as np

from monster.feature_compile.mechanisms import PlayerMechanismInputs
from monster.feature_compile.units import UnitPlayerInputs
from monster.feature_compile.v13_identity_bridge import compile_v13_player_identity
from monster.sim.play_kernel import PlayerIdentity, TeamIdentity
from monster.snapshot.model import TeamState
from monster.snapshot.player import TeamPlayerPool


@dataclass(frozen=True)
class MatchupIdentityAuthorityConfigV634:
    """Bounded causal authority for the v6.3.4 shadow.

    The coefficients change football mechanisms, never score or fantasy output directly.
    """

    qb_to_pass_efficiency: float = 0.80
    runner_to_rush_efficiency: float = 0.55
    pass_protection_amplification: float = 1.35
    run_block_amplification: float = 1.35
    pass_matchup_relative_authority: float = 0.50
    run_matchup_relative_authority: float = 0.45


DEFAULT_V634_AUTHORITY = MatchupIdentityAuthorityConfigV634()


@dataclass(frozen=True)
class MatchupIdentityTraceV634:
    team_id: str
    quarterback_id: str
    quarterback_efficiency: float
    pass_efficiency_before: float
    pass_efficiency_after: float
    weighted_runner_efficiency: float
    rush_efficiency_before: float
    rush_efficiency_after: float
    pass_protection_before: float
    pass_protection_after: float
    run_blocking_before: float
    run_blocking_after: float
    mean_receiver_efficiency: float
    mean_receiver_route_skill: float
    neutral_pass_rate: float
    situational_pass_rate_sd: float
    situational_pass_rate_range: float
    rich_players: int
    players_with_capability_evidence: int


def _relative(value: float, authority: float, *, low: float, high: float) -> float:
    value = max(float(value), 1e-6)
    return float(np.clip(exp(authority * log(value)), low, high))


def _amplify_unit(value: float, authority: float) -> float:
    return float(np.clip(1.0 + authority * (float(value) - 1.0), 0.86, 1.14))


def _weighted_efficiency(players: tuple[PlayerIdentity, ...]) -> float:
    if not players:
        return 1.0
    weights = np.asarray([max(float(p.usage_weight), 0.001) for p in players], dtype=float)
    values = np.asarray([max(float(p.efficiency), 0.35) for p in players], dtype=float)
    weights /= weights.sum()
    # A weighted geometric mean respects relative football skill without allowing one reserve
    # to drag the entire offense or one star to become a direct team-points multiplier.
    return float(exp(np.sum(weights * np.log(values))))


def apply_matchup_identity_authority_v634(
    identity: TeamIdentity,
    *,
    pool: TeamPlayerPool,
    reality: dict[str, PlayerMechanismInputs],
    unit_players: tuple[UnitPlayerInputs, ...],
    state: TeamState,
    config: MatchupIdentityAuthorityConfigV634 = DEFAULT_V634_AUTHORITY,
    availability_already_sampled: bool = False,
) -> tuple[TeamIdentity, MatchupIdentityTraceV634]:
    """Reconnect current player identity to team and 1v1 football mechanisms.

    v6.3 already contains rich local duels. This layer repairs the upstream authority seam:
    the actual QB and runner identities materially bend the team's efficiency prior before the
    snap, while the rich player objects themselves remain intact for route, catch, rush,
    open-field and turnover mechanisms downstream.
    """

    current = {
        player.player_id: player
        for player in (identity.quarterback, *identity.rushers, *identity.receivers)
    }
    capability = {player.player_id: player for player in unit_players}
    compiled: dict[str, PlayerIdentity] = {}
    evidence_count = 0

    for player in pool.players:
        existing = current.get(player.player_id)
        if existing is None:
            usage = (
                max(float(player.qb_pass_share), 0.001)
                if player.position.upper() == "QB"
                else max(float(player.target_share), float(player.rush_share), 0.001)
            )
        else:
            usage = max(float(existing.usage_weight), 0.001)
        rich, trace = compile_v13_player_identity(
            player_id=player.player_id,
            name=player.display_name,
            position=player.position,
            usage_weight=usage,
            inputs=reality.get(player.player_id, PlayerMechanismInputs()),
            availability_already_sampled=availability_already_sampled,
            capability_inputs=capability.get(player.player_id),
        )
        compiled[player.player_id] = rich
        evidence_count += int(trace.evidence_fields > 0)

    qb = compiled.get(identity.quarterback.player_id, identity.quarterback)

    # Use the newly compiled rich object itself, not only its legacy aggregate fields, so the
    # Madden/physical mechanism channels survive into the downstream 1v1 engine.
    rushers = tuple(
        replace(
            compiled.get(player.player_id, player),
            usage_weight=player.usage_weight,
        )
        for player in identity.rushers
    )
    receivers = tuple(
        replace(
            compiled.get(player.player_id, player),
            usage_weight=player.usage_weight,
        )
        for player in identity.receivers
    )

    runner_eff = _weighted_efficiency(rushers)
    pass_eff = float(
        np.clip(
            identity.pass_efficiency
            * _relative(qb.efficiency, config.qb_to_pass_efficiency, low=0.78, high=1.24),
            0.55,
            1.52,
        )
    )
    rush_eff = float(
        np.clip(
            identity.rush_efficiency
            * _relative(
                runner_eff,
                config.runner_to_rush_efficiency,
                low=0.82,
                high=1.20,
            ),
            0.62,
            1.45,
        )
    )

    pass_protection = _amplify_unit(
        identity.pass_protection, config.pass_protection_amplification
    )
    run_blocking = _amplify_unit(
        identity.run_blocking, config.run_block_amplification
    )

    enhanced = replace(
        identity,
        quarterback=qb,
        rushers=rushers,
        receivers=receivers,
        pass_efficiency=pass_eff,
        rush_efficiency=rush_eff,
        pass_protection=pass_protection,
        run_blocking=run_blocking,
    )
    trace = MatchupIdentityTraceV634(
        team_id=identity.team_id,
        quarterback_id=qb.player_id,
        quarterback_efficiency=float(qb.efficiency),
        pass_efficiency_before=float(identity.pass_efficiency),
        pass_efficiency_after=pass_eff,
        weighted_runner_efficiency=runner_eff,
        rush_efficiency_before=float(identity.rush_efficiency),
        rush_efficiency_after=rush_eff,
        pass_protection_before=float(identity.pass_protection),
        pass_protection_after=pass_protection,
        run_blocking_before=float(identity.run_blocking),
        run_blocking_after=run_blocking,
        mean_receiver_efficiency=(
            float(np.mean([p.efficiency for p in receivers])) if receivers else 1.0
        ),
        mean_receiver_route_skill=(
            float(np.mean([float(getattr(p, "route_separation_skill", 0.0)) for p in receivers]))
            if receivers
            else 0.0
        ),
        neutral_pass_rate=float(identity.neutral_pass_rate),
        situational_pass_rate_sd=(
            float(np.std(np.asarray(identity.situational_pass_rates, dtype=float), ddof=0))
            if identity.situational_pass_rates
            else 0.0
        ),
        situational_pass_rate_range=(
            float(
                max(identity.situational_pass_rates)
                - min(identity.situational_pass_rates)
            )
            if identity.situational_pass_rates
            else 0.0
        ),
        rich_players=len(compiled),
        players_with_capability_evidence=evidence_count,
    )
    return enhanced, trace
