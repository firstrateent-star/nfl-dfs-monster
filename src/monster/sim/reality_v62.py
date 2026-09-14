from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from math import exp
from typing import Any

import numpy as np
import polars as pl

from monster.feature_compile.v13_identity_bridge import compile_v13_player_identity
from monster.sim.chaos_ecology import ChaosEcology
from monster.sim.play_kernel import PlayerIdentity, TeamIdentity
from monster.sim.rushing_roles import sample_event_rush_share_plan
from monster.snapshot.player import TeamPlayerPool

BLOCKED_ROSTER_STATUSES = {"CUT", "RET"}
CONDITIONAL_ROSTER_STATUSES = {"DEV", "IR", "RES", "NFI", "PUP"}
TARGET_ROLE_RESERVE = 0.14
MAX_RECEIVER_CANDIDATES = 8
MAX_ACTIVE_RECEIVERS = 6


class RoleWorldPlan(dict[str, float]):
    """One world-stable current-role state for rush and receiving opportunity."""

    def __init__(self, rush_plan: Mapping[str, float], target_plan: Mapping[str, float]):
        super().__init__((str(key), float(value)) for key, value in rush_plan.items())
        self.target_plan = {str(key): float(value) for key, value in target_plan.items()}


def filter_roster_truth(personnel: pl.DataFrame) -> pl.DataFrame:
    """Remove clearly unavailable roster states before old usage can create opportunity.

    Current game-day overrides remain authoritative: practice-squad/reserve states can re-enter
    only when the health layer has explicit evidence that active probability is at least 50%.
    CUT/RET states are hard boundaries.
    """

    if "status" not in personnel.columns or "game_day_active_probability" not in personnel.columns:
        return personnel
    status = pl.col("status").cast(pl.Utf8).fill_null("").str.to_uppercase()
    active = pl.col("game_day_active_probability").fill_null(0.0).cast(pl.Float64)
    hard_block = status.is_in(sorted(BLOCKED_ROSTER_STATUSES))
    conditional_block = status.is_in(sorted(CONDITIONAL_ROSTER_STATUSES)) & (active < 0.50)
    return personnel.filter(~hard_block & ~conditional_block)


def _receiving_skill(player: Any) -> float:
    catch = float(np.clip(getattr(player, "catch_rate", 0.65) / 0.65, 0.70, 1.30))
    ypr = float(np.clip(getattr(player, "yards_per_reception", 10.5) / 10.5, 0.70, 1.35))
    effectiveness = float(np.clip(getattr(player, "effectiveness_if_active", 1.0), 0.50, 1.20))
    return float(np.clip(effectiveness * catch**0.45 * ypr**0.25, 0.60, 1.45))


def receiver_candidate_ids(
    pool: TeamPlayerPool,
    *,
    max_candidates: int = MAX_RECEIVER_CANDIDATES,
) -> tuple[str, ...]:
    """Choose current receiving candidates without requiring prior-team target volume."""

    eligible = [
        player
        for player in pool.players
        if player.position.upper() in {"RB", "WR", "TE"} and player.active_probability >= 0.10
    ]
    if not eligible:
        return ()
    ranked = sorted(
        eligible,
        key=lambda player: (
            max(float(player.target_share), 0.0) ** 0.70
            + 0.035 * (0.35 + float(player.role_uncertainty)) * _receiving_skill(player)
        )
        * float(np.clip(player.active_probability, 0.0, 1.0)),
        reverse=True,
    )
    return tuple(player.player_id for player in ranked[:max_candidates])


def _cap_simplex(values: np.ndarray, cap: float) -> np.ndarray:
    """Cap one share without losing probability mass."""

    out = np.clip(values.astype(float), 0.0, None)
    if out.sum() <= 0:
        return np.ones(len(out), dtype=float) / max(len(out), 1)
    out /= out.sum()
    for _ in range(8):
        above = out > cap
        if not above.any():
            break
        excess = float((out[above] - cap).sum())
        out[above] = cap
        below = ~above
        room = np.clip(cap - out[below], 0.0, None)
        if room.sum() <= 0:
            break
        out[below] += excess * room / room.sum()
    return out / out.sum()


def sample_target_share_plan(
    pool: TeamPlayerPool,
    *,
    rng: np.random.Generator,
    max_active_receivers: int = MAX_ACTIVE_RECEIVERS,
) -> dict[str, float]:
    """Sample a current-team route/target role world before pass outcomes are simulated.

    Historical/current target share remains evidence, but a finite reserve is assigned from
    present-day availability, role uncertainty and receiving capability. This prevents a new
    current-team contributor from becoming impossible merely because his prior-team target
    history maps poorly to the new offense.
    """

    candidate_ids = set(receiver_candidate_ids(pool))
    candidates = [
        player
        for player in pool.players
        if player.player_id in candidate_ids
        and player.position.upper() in {"RB", "WR", "TE"}
    ]
    if not candidates:
        return {}

    active = [
        player
        for player in candidates
        if rng.random() < float(np.clip(player.active_probability, 0.0, 1.0))
    ]
    if not active:
        active = [max(candidates, key=lambda player: player.active_probability)]

    strength = np.asarray(
        [
            max(float(player.target_share), 0.0) ** 0.70
            + TARGET_ROLE_RESERVE
            / max(len(active), 1)
            * (0.35 + float(player.role_uncertainty))
            * _receiving_skill(player)
            for player in active
        ],
        dtype=float,
    )
    sigma = np.asarray(
        [0.10 + 0.90 * float(np.clip(player.role_uncertainty, 0.02, 0.35)) for player in active],
        dtype=float,
    )
    latent = np.log(np.clip(strength, 1e-7, None)) + rng.normal(0.0, sigma)
    keep_n = min(max_active_receivers, len(active))
    order = np.argsort(latent)[::-1][:keep_n]
    selected = [active[int(index)] for index in order]

    centers = np.asarray(
        [
            max(float(player.target_share), 0.0) ** 0.72
            + TARGET_ROLE_RESERVE
            / max(len(selected), 1)
            * (0.40 + float(player.role_uncertainty))
            * _receiving_skill(player)
            for player in selected
        ],
        dtype=float,
    )
    centers /= centers.sum()
    centers = _cap_simplex(centers, 0.42)

    mean_uncertainty = float(np.mean([player.role_uncertainty for player in selected]))
    concentration = float(np.clip(52.0 - 70.0 * mean_uncertainty, 18.0, 48.0))
    shares = rng.dirichlet(np.clip(centers * concentration, 0.20, None))
    shares = _cap_simplex(shares, 0.48)
    return {
        player.player_id: float(shares[index])
        for index, player in enumerate(selected)
        if shares[index] > 0.0
    }


def sample_role_world_plan(
    pool: TeamPlayerPool,
    *,
    rng: np.random.Generator,
) -> RoleWorldPlan:
    """Sample rushing and receiving role uncertainty once, upstream of game randomness."""

    rush_plan = sample_event_rush_share_plan(pool, rng=rng)
    target_plan = sample_target_share_plan(pool, rng=rng)
    return RoleWorldPlan(rush_plan, target_plan)


def complete_team_receivers(
    team: TeamIdentity,
    pool: TeamPlayerPool,
    reality: Mapping[str, Any],
    unit_players: Sequence[Any],
) -> TeamIdentity:
    """Make plausible current receivers available before the per-world role draw.

    The stable v6.1 rotation requires non-zero historical/current target share. v6.2 instead
    admits a bounded current candidate set and lets the role world decide who actually runs
    meaningful routes.
    """

    capability = {player.player_id: player for player in unit_players}
    states = {player.player_id: player for player in pool.players}
    candidates = receiver_candidate_ids(pool)
    compiled: dict[str, PlayerIdentity] = {receiver.player_id: receiver for receiver in team.receivers}

    for player_id in candidates:
        if player_id in compiled:
            continue
        player = states[player_id]
        usage = max(float(player.target_share), 0.008)
        inputs = reality.get(player_id)
        if inputs is None:
            identity = PlayerIdentity(
                player_id=player_id,
                name=player.display_name,
                position=player.position,
                usage_weight=usage,
            )
        else:
            identity, _ = compile_v13_player_identity(
                player_id=player_id,
                name=player.display_name,
                position=player.position,
                usage_weight=usage,
                inputs=inputs,
                capability_inputs=capability.get(player_id),
            )
        compiled[player_id] = identity

    receivers = tuple(compiled[player_id] for player_id in candidates if player_id in compiled)
    return replace(team, receivers=receivers or team.receivers)


def apply_role_world(team: TeamIdentity, plan: RoleWorldPlan) -> TeamIdentity:
    """Apply the sampled role state as the authoritative participant/opportunity layer."""

    rushers = tuple(
        replace(rusher, usage_weight=float(plan[rusher.player_id]))
        for rusher in team.rushers
        if plan.get(rusher.player_id, 0.0) > 0.0
    )
    target_plan = plan.target_plan
    receivers = tuple(
        replace(receiver, usage_weight=float(target_plan[receiver.player_id]))
        for receiver in team.receivers
        if target_plan.get(receiver.player_id, 0.0) > 0.0
    )
    return replace(
        team,
        rushers=rushers or team.rushers,
        receivers=receivers or team.receivers,
    )


def _conditional_compatibility(
    players: Sequence[Any],
    categories: Sequence[str],
    attempts: Mapping[tuple[str, str], int],
    category: str,
    *,
    shrinkage_samples: float,
    max_authority: float,
    relative_low: float,
    relative_high: float,
) -> np.ndarray:
    base = np.asarray(
        [max(float(getattr(player, "usage_weight", 0.0)), 0.001) for player in players],
        dtype=float,
    )
    base /= base.sum()

    totals = np.asarray(
        [
            sum(max(int(attempts.get((str(player.player_id), cat), 0)), 0) for cat in categories)
            for player in players
        ],
        dtype=float,
    )
    category_counts = np.asarray(
        [max(int(attempts.get((str(player.player_id), category), 0)), 0) for player in players],
        dtype=float,
    )
    population_total = float(totals.sum())
    if population_total <= 0.0:
        return base
    population_rate = float(category_counts.sum() / population_total)
    if population_rate <= 1e-8:
        return base

    compatibility = np.ones(len(players), dtype=float)
    for index, total in enumerate(totals):
        if total <= 0:
            continue
        actor_rate = float(category_counts[index] / total)
        relative = float(np.clip(actor_rate / population_rate, relative_low, relative_high))
        authority = min(float(total / (total + shrinkage_samples)), max_authority)
        compatibility[index] = (1.0 - authority) + authority * relative

    weights = base * compatibility
    return weights / weights.sum()


def choose_target_role_authoritative(
    players: Sequence[Any],
    ecology: Any,
    category: str,
    rng: np.random.Generator,
    *,
    shrinkage_samples: float = 45.0,
):
    """Use history as depth compatibility, never as an old-team target-share allocator."""

    if not players:
        raise ValueError("receiver pool cannot be empty")
    weights = _conditional_compatibility(
        players,
        ecology.pass_depth.categories,
        ecology.target_depth_attempts,
        category,
        shrinkage_samples=shrinkage_samples,
        max_authority=0.22,
        relative_low=0.55,
        relative_high=1.65,
    )
    return players[int(rng.choice(len(players), p=weights))]


def choose_rusher_role_authoritative(
    players: Sequence[Any],
    ecology: Any,
    category: str,
    rng: np.random.Generator,
    *,
    shrinkage_samples: float = 60.0,
):
    """Current role chooses the carrier; old geometry history only shapes compatibility."""

    if not players:
        raise ValueError("rusher pool cannot be empty")
    weights = _conditional_compatibility(
        players,
        ecology.run_geometry.categories,
        ecology.rusher_geometry_attempts,
        category,
        shrinkage_samples=shrinkage_samples,
        max_authority=0.28,
        relative_low=0.60,
        relative_high=1.55,
    )
    return players[int(rng.choice(len(players), p=weights))]


class GameEnvironment:
    def __init__(
        self,
        *,
        away_execution: float,
        home_execution: float,
        common_execution: float,
        penalty_rate: float,
        chaos_factor: float,
    ):
        self.away_execution = float(away_execution)
        self.home_execution = float(home_execution)
        self.common_execution = float(common_execution)
        self.penalty_rate = float(penalty_rate)
        self.chaos_factor = float(chaos_factor)


def sample_game_environment(seed: int) -> GameEnvironment:
    """Sample persistent game-level execution/chaos state without directly sampling points."""

    rng = np.random.default_rng(int(seed) + 6_200_911)
    tail_world = rng.random() < 0.20
    common_sigma = 0.145 if tail_world else 0.065
    common_z = float(rng.normal())
    common = exp(common_sigma * common_z - 0.5 * common_sigma**2)

    team_sigma = 0.085
    away = common * exp(team_sigma * float(rng.normal()) - 0.5 * team_sigma**2)
    home = common * exp(team_sigma * float(rng.normal()) - 0.5 * team_sigma**2)

    penalty_sigma = 0.28
    penalty = 0.070 * exp(
        penalty_sigma * float(rng.normal()) - 0.5 * penalty_sigma**2
    )
    chaos_sigma = 0.32
    chaos = exp(chaos_sigma * float(rng.normal()) - 0.5 * chaos_sigma**2)

    return GameEnvironment(
        away_execution=float(np.clip(away, 0.76, 1.30)),
        home_execution=float(np.clip(home, 0.76, 1.30)),
        common_execution=float(np.clip(common, 0.82, 1.22)),
        penalty_rate=float(np.clip(penalty, 0.035, 0.115)),
        chaos_factor=float(np.clip(chaos, 0.55, 1.85)),
    )


def apply_game_environment(
    away: TeamIdentity,
    home: TeamIdentity,
    environment: GameEnvironment,
) -> tuple[TeamIdentity, TeamIdentity]:
    """Apply persistent execution state to football mechanisms, centered near neutral."""

    def apply(team: TeamIdentity, factor: float) -> TeamIdentity:
        return replace(
            team,
            pass_efficiency=float(np.clip(team.pass_efficiency * factor, 0.64, 1.48)),
            rush_efficiency=float(
                np.clip(team.rush_efficiency * factor**0.78, 0.68, 1.38)
            ),
            pass_protection=float(
                np.clip(team.pass_protection * factor**0.16, 0.88, 1.12)
            ),
            run_blocking=float(
                np.clip(team.run_blocking * factor**0.14, 0.88, 1.12)
            ),
        )

    return apply(away, environment.away_execution), apply(home, environment.home_execution)


def apply_chaos_environment(
    ecology: ChaosEcology,
    environment: GameEnvironment,
) -> ChaosEcology:
    """Widen rare-event return geometry without sampling a touchdown directly."""

    factor = environment.chaos_factor
    tail = float(np.sqrt(factor))
    return replace(
        ecology,
        interception_return_sd=float(np.clip(ecology.interception_return_sd * tail, 1.0, 40.0)),
        interception_40_plus_rate=float(
            np.clip(ecology.interception_40_plus_rate * factor, 0.0, 0.25)
        ),
        fumble_return_sd=float(np.clip(ecology.fumble_return_sd * tail, 1.0, 35.0)),
        fumble_40_plus_rate=float(
            np.clip(ecology.fumble_40_plus_rate * factor, 0.0, 0.20)
        ),
        punt_40_plus_rate=float(np.clip(ecology.punt_40_plus_rate * factor, 0.0, 0.20)),
        kickoff_40_plus_rate=float(
            np.clip(ecology.kickoff_40_plus_rate * factor, 0.0, 0.20)
        ),
        punt_muff_rate=float(np.clip(ecology.punt_muff_rate * tail, 0.0, 0.08)),
        kickoff_muff_rate=float(np.clip(ecology.kickoff_muff_rate * tail, 0.0, 0.05)),
        blocked_punt_rate=float(np.clip(ecology.blocked_punt_rate * tail, 0.0, 0.08)),
        blocked_field_goal_rate=float(
            np.clip(ecology.blocked_field_goal_rate * tail, 0.0, 0.08)
        ),
    )
