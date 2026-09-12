from __future__ import annotations

from dataclasses import replace
from math import exp
from typing import Any

import numpy as np

from monster.feature_compile.units import compile_team_unit_effects
from monster.feature_compile.v13_identity_bridge import compile_v13_player_identity
from monster.sim.matchup_kernel import (
    LEAGUE_PRESSURE_RATE,
    DefensiveIdentity,
    DefensiveUnit,
)
from monster.sim.play_kernel import PlayerIdentity, TeamIdentity

# _team_identity is evaluated for every team before the integrated runner builds defenses.
# Recording state by player lets the legacy one-argument _defensive_unit seam recover the
# correct market-blind team defensive context without changing the stable integrated runner.
_STATE_BY_PLAYER: dict[str, Any] = {}


def _rating(value: float | None, center: float = 78.0, scale: float = 12.0) -> float:
    if value is None:
        return 1.0
    return float(np.clip(1.0 + 0.22 * np.tanh((float(value) - center) / scale), 0.75, 1.25))


def _team_context(state: Any) -> tuple[float, float]:
    """Turn independent historical efficiency dimensions into offense identity.

    EPA, success rate and explosive rate are separate football facts. Combining their
    deviations preserves real team shape instead of asking one regressed feature to carry
    the entire offense. The exponential form is symmetric around neutral and remains bounded.
    """
    epa = float(getattr(state, "offensive_epa_per_play", 0.0) or 0.0)
    success = float(getattr(state, "offensive_success_rate", 0.44) or 0.44)
    explosive = float(getattr(state, "offensive_explosive_rate", 0.10) or 0.10)
    injury = float(getattr(state, "injury_effect", 0.0) or 0.0)
    weather = float(getattr(state, "weather_effect", 0.0) or 0.0)
    physical = float(getattr(state, "physical_madden_effect", 0.0) or 0.0)

    pass_context = exp(
        1.25 * epa
        + 1.25 * (success - 0.44)
        + 1.90 * (explosive - 0.10)
        + physical
        + injury
        + 0.35 * weather
    )
    rush_context = exp(
        0.70 * epa
        + 0.90 * (success - 0.44)
        + 0.95 * (explosive - 0.10)
        + injury
        + 0.20 * weather
    )
    return float(np.clip(pass_context, 0.70, 1.38)), float(np.clip(rush_context, 0.76, 1.28))


def _receiver_rotation(pool: Any) -> tuple[Any, ...]:
    """Keep the realistic primary receiving rotation instead of the entire skill roster.

    A pass snap can expose at most five eligible receivers. We retain one rotational seat
    across the game by allowing six season-level candidates, but fringe roster players no
    longer compete on every single QB read. Current target share already incorporates current
    snap role and historical target tendency, so it is the clean ranking signal here.
    """
    eligible = sorted(
        (
            player
            for player in pool.players
            if player.position in {"RB", "WR", "TE"} and player.target_share > 0.001
        ),
        key=lambda player: player.target_share,
        reverse=True,
    )
    primary = [player for player in eligible if player.target_share >= 0.02][:6]
    minimum = min(5, len(eligible))
    if len(primary) < minimum:
        selected = {player.player_id for player in primary}
        primary.extend(player for player in eligible if player.player_id not in selected)
    return tuple(primary[: max(minimum, min(6, len(primary)))])


def enhanced_team_identity(
    team_id: str,
    pool: Any,
    reality: dict[str, Any],
    unit_players: tuple[Any, ...],
    state: Any,
    *,
    league_neutral_pass_rate: float,
    situational_pass_rates: tuple[float, ...],
) -> TeamIdentity:
    """Compile stronger but bounded current-player and team football identity."""
    capability = {player.player_id: player for player in unit_players}
    for player in unit_players:
        _STATE_BY_PLAYER[player.player_id] = state

    compiled: dict[str, PlayerIdentity] = {}
    for player in pool.players:
        usage = (
            player.qb_pass_share
            if player.position == "QB"
            else max(player.target_share, player.rush_share, 0.001)
        )
        inputs = reality.get(player.player_id)
        if inputs is None:
            compiled[player.player_id] = PlayerIdentity(
                player.player_id,
                player.display_name,
                player.position,
                usage_weight=usage,
            )
        else:
            compiled[player.player_id], _ = compile_v13_player_identity(
                player_id=player.player_id,
                name=player.display_name,
                position=player.position,
                usage_weight=usage,
                inputs=inputs,
                capability_inputs=capability.get(player.player_id),
            )

    qbs = [player for player in pool.players if player.position == "QB"]
    if not qbs:
        raise ValueError(f"{team_id} has no quarterback in current pool")
    qb_state = max(qbs, key=lambda player: player.qb_pass_share)
    quarterback = compiled[qb_state.player_id]

    rushers = tuple(
        replace(
            compiled[player.player_id],
            usage_weight=max(player.rush_share, 0.001),
        )
        for player in pool.players
        if player.rush_share > 0.001
    )

    receiver_rotation = _receiver_rotation(pool)
    receivers = tuple(
        replace(
            compiled[player.player_id],
            # The legacy full-field read subsequently applies a 0.62 exponent, which flattened
            # real target hierarchy. A modest pre-sharpening restores current-role authority
            # while the bounded matchup/read terms still decide the actual throw.
            usage_weight=max(player.target_share, 0.001) ** 1.25,
        )
        for player in receiver_rotation
    )

    effects, _ = compile_team_unit_effects(unit_players)
    pass_efficiency, rush_efficiency = _team_context(state)

    return TeamIdentity(
        team_id=team_id,
        quarterback=quarterback,
        rushers=rushers or (quarterback,),
        receivers=receivers,
        neutral_pass_rate=pool.neutral_pass_rate,
        pass_efficiency=pass_efficiency,
        rush_efficiency=rush_efficiency,
        # Individual OL/DL duels now carry rich Madden evidence. Keep aggregate unit effects
        # deliberately lower-authority so the same evidence is not charged twice.
        pass_protection=float(np.clip(1.0 + 0.70 * effects.pass_protection_effect, 0.93, 1.07)),
        run_blocking=float(np.clip(1.0 + 0.70 * effects.run_block_effect, 0.93, 1.07)),
        field_goal_skill=float(np.clip(1.0 + effects.special_teams_effect, 0.92, 1.08)),
        punt_skill=float(np.clip(1.0 + effects.special_teams_effect, 0.92, 1.08)),
        league_neutral_pass_rate=league_neutral_pass_rate,
        situational_pass_rates=situational_pass_rates,
    )


def enhanced_defensive_unit(players: tuple[Any, ...]) -> DefensiveUnit:
    """Combine player-v-player evidence with market-blind team defensive identity."""
    state = next(
        (
            _STATE_BY_PLAYER.get(player.player_id)
            for player in players
            if player.player_id in _STATE_BY_PLAYER
        ),
        None,
    )
    if state is None:
        epa_allowed = 0.0
        explosive_allowed = 0.10
        sack_rate = 0.07
        hit_rate = 0.18
    else:
        epa_allowed = float(getattr(state, "defensive_epa_allowed_per_play", 0.0) or 0.0)
        explosive_allowed = float(getattr(state, "defensive_explosive_rate_allowed", 0.10) or 0.10)
        sack_rate = float(getattr(state, "defensive_sack_rate", 0.07) or 0.07)
        hit_rate = float(getattr(state, "defensive_qb_hit_rate", 0.18) or 0.18)

    coverage_context = float(
        np.clip(exp(-1.15 * epa_allowed - 1.55 * (explosive_allowed - 0.10)), 0.78, 1.28)
    )
    rush_context = float(
        np.clip(
            exp(2.60 * (sack_rate - 0.07) + 1.10 * (hit_rate - 0.18) - 0.35 * epa_allowed),
            0.80,
            1.27,
        )
    )
    run_context = float(np.clip(exp(-0.75 * epa_allowed), 0.84, 1.20))

    front: list[DefensiveIdentity] = []
    coverage: list[DefensiveIdentity] = []
    for player in players:
        if player.defense_snap_share < 0.03:
            continue
        position = player.position.upper()
        snap_weight = float(
            np.clip(
                player.defense_snap_share
                * player.active_probability
                * player.effectiveness_if_active,
                0.001,
                1.10,
            )
        )
        cov = float(
            np.clip(_rating(player.madden_coverage) * coverage_context**0.55, 0.68, 1.38)
        )
        rush = float(
            np.clip(_rating(player.madden_pass_rush) * rush_context**0.60, 0.68, 1.38)
        )
        tackle = float(
            np.clip(_rating(player.madden_tackle) * run_context**0.45, 0.70, 1.34)
        )
        speed = _rating(player.madden_speed, center=85.0, scale=8.0)
        return_rating = _rating(player.madden_return, center=78.0, scale=12.0)
        returning = float(np.clip(0.60 * speed + 0.40 * return_rating, 0.72, 1.30))
        item = DefensiveIdentity(
            player_id=player.player_id,
            name=player.player_id,
            position=position,
            coverage=cov,
            pass_rush=rush,
            run_defense=tackle,
            tackling=tackle,
            ball_hawk=cov,
            speed=speed,
            returning=returning,
            snap_weight=snap_weight,
        )
        if position in {"DE", "DT", "NT", "DL", "EDGE", "LB", "ILB", "OLB", "MLB"}:
            front.append(item)
        if position in {"CB", "DB", "S", "FS", "SS", "LB", "ILB", "OLB", "MLB"}:
            coverage.append(item)

    return DefensiveUnit(
        front=tuple(front),
        coverage=tuple(coverage),
        pressure_rate=float(np.clip(LEAGUE_PRESSURE_RATE * rush_context, 0.18, 0.45)),
        run_stuff_rate=float(np.clip(0.18 * run_context, 0.12, 0.26)),
    )
