from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from monster.feature_compile.mechanisms import PlayerMechanismInputs
from monster.feature_compile.units import UnitPlayerInputs
from monster.feature_compile.v13_identity_bridge import compile_v13_player_identity
from monster.sim.play_kernel import PlayerIdentity, TeamIdentity
from monster.snapshot.model import TeamState
from monster.snapshot.player import TeamPlayerPool


@dataclass(frozen=True)
class V13TeamIdentityAuthorityTrace:
    team_id: str
    quarterback_id: str
    quarterback_efficiency: float
    historical_pass_efficiency_before_qb: float
    pass_efficiency_after_qb: float
    historical_rush_efficiency_before_live_prior: float
    live_rush_context: float
    rush_efficiency_after_live_prior: float
    mean_receiver_efficiency: float
    mean_rusher_efficiency: float
    rich_players: int
    players_with_capability_evidence: int


def _usage(player, existing: PlayerIdentity | None) -> float:
    if existing is not None:
        return existing.usage_weight
    if player.position == "QB":
        return max(float(player.qb_pass_share), 0.001)
    return max(float(player.target_share), float(player.rush_share), 0.001)


def _live_rush_context(state: TeamState) -> float:
    """Recover team-specific run context from live historical football evidence.

    v1.3 previously referenced ``state.offense_strength`` here, but the current policy
    compiler never populates that field. Use the already-live, market-blind team evidence
    that is actually present in TeamState instead of silently collapsing every team to zero.
    This context bends run ecology; it never directly adds points.
    """

    value = (
        1.0
        + 0.65 * float(state.offensive_epa_per_play)
        + 0.45 * (float(state.offensive_success_rate) - 0.44)
        + 0.30 * (float(state.offensive_explosive_rate) - 0.10)
    )
    return float(np.clip(value, 0.82, 1.18))


def _amplify_unit(value: float, *, authority: float = 1.55) -> float:
    """Increase real unit differentiation around neutral without inventing a new baseline."""
    return float(np.clip(1.0 + authority * (float(value) - 1.0), 0.86, 1.14))


def apply_v13_team_identity_authority(
    identity: TeamIdentity,
    *,
    pool: TeamPlayerPool,
    reality: dict[str, PlayerMechanismInputs],
    unit_players: tuple[UnitPlayerInputs, ...],
    state: TeamState,
    availability_already_sampled: bool = False,
) -> tuple[TeamIdentity, V13TeamIdentityAuthorityTrace]:
    """Attach rich player capability to v1.3 without bypassing football mechanisms.

    This is the promotion seam for current and future identity evidence. Sources compile into
    player capability channels first; only then do they alter QB execution, receiver/rusher
    resolution and team run context. The function intentionally does not touch play calling,
    possessions or score directly.
    """

    current = {
        player.player_id: player
        for player in (identity.quarterback, *identity.rushers, *identity.receivers)
    }
    capability = {player.player_id: player for player in unit_players}
    compiled: dict[str, PlayerIdentity] = {}
    evidence_count = 0

    for player in pool.players:
        inputs = reality.get(player.player_id, PlayerMechanismInputs())
        existing = current.get(player.player_id)
        rich, trace = compile_v13_player_identity(
            player_id=player.player_id,
            name=player.display_name,
            position=player.position,
            usage_weight=_usage(player, existing),
            inputs=inputs,
            availability_already_sampled=availability_already_sampled,
            capability_inputs=capability.get(player.player_id),
        )
        compiled[player.player_id] = rich
        evidence_count += int(trace.evidence_fields > 0)

    qb = compiled.get(identity.quarterback.player_id, identity.quarterback)

    rushers = tuple(
        replace(
            rusher,
            efficiency=compiled.get(rusher.player_id, rusher).efficiency,
            explosive=compiled.get(rusher.player_id, rusher).explosive,
            turnover_security=compiled.get(rusher.player_id, rusher).turnover_security,
        )
        for rusher in identity.rushers
    )
    receivers = tuple(
        replace(
            receiver,
            efficiency=compiled.get(receiver.player_id, receiver).efficiency,
            explosive=compiled.get(receiver.player_id, receiver).explosive,
            turnover_security=compiled.get(receiver.player_id, receiver).turnover_security,
        )
        for receiver in identity.receivers
    )

    # Game Flow decides what the offense attempts. QB identity has stronger authority over
    # whether the attempted pass actually works, while the historical/team prior still anchors
    # the league-scale environment.
    pass_efficiency = float(
        np.clip(identity.pass_efficiency * qb.efficiency, 0.55, 1.52)
    )

    # Historical team rushing quality and individual runner identity now receive enough
    # authority to separate good/bad run environments without directly changing points.
    live_rush = _live_rush_context(state)
    rush_efficiency = float(
        np.clip(identity.rush_efficiency * live_rush, 0.62, 1.45)
    )

    enhanced = replace(
        identity,
        quarterback=qb,
        rushers=rushers,
        receivers=receivers,
        pass_efficiency=pass_efficiency,
        rush_efficiency=rush_efficiency,
        pass_protection=_amplify_unit(identity.pass_protection),
        run_blocking=_amplify_unit(identity.run_blocking),
    )
    trace = V13TeamIdentityAuthorityTrace(
        team_id=identity.team_id,
        quarterback_id=qb.player_id,
        quarterback_efficiency=qb.efficiency,
        historical_pass_efficiency_before_qb=identity.pass_efficiency,
        pass_efficiency_after_qb=pass_efficiency,
        historical_rush_efficiency_before_live_prior=identity.rush_efficiency,
        live_rush_context=live_rush,
        rush_efficiency_after_live_prior=rush_efficiency,
        mean_receiver_efficiency=(
            float(np.mean([player.efficiency for player in receivers])) if receivers else 1.0
        ),
        mean_rusher_efficiency=(
            float(np.mean([player.efficiency for player in rushers])) if rushers else 1.0
        ),
        rich_players=len(compiled),
        players_with_capability_evidence=evidence_count,
    )
    return enhanced, trace
