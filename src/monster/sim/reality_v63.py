from __future__ import annotations

from dataclasses import dataclass, replace
from math import exp, sqrt
from typing import Any

import numpy as np

from monster.sim import reality_snap_v5
from monster.sim.chaos_ecology import (
    ChaosEcology,
    ReturnEvent,
    ReturnKind,
    _returner,
)
from monster.sim.football_state import FootballState, mirror_field
from monster.sim.play_kernel import PassResult, PlayEvent, TeamIdentity


@dataclass(frozen=True)
class GameEnvironmentV63:
    """Persistent football environment for one simulated game world.

    The state changes execution, open-field conditions, tempo and live-ball chaos. It never
    samples points, touchdowns or fantasy outcomes directly.
    """

    away_execution: float
    home_execution: float
    common_execution: float
    penalty_rate: float
    chaos_factor: float
    explosive_factor: float
    tempo_factor: float


_ACTIVE_ENVIRONMENT: GameEnvironmentV63 | None = None


def set_active_game_environment(environment: GameEnvironmentV63 | None) -> None:
    global _ACTIVE_ENVIRONMENT
    _ACTIVE_ENVIRONMENT = environment


def active_game_environment() -> GameEnvironmentV63 | None:
    return _ACTIVE_ENVIRONMENT


def _correlated_normal(
    base_z: float,
    *,
    correlation: float,
    rng: np.random.Generator,
) -> float:
    rho = float(np.clip(correlation, -0.98, 0.98))
    return float(rho * base_z + sqrt(max(1.0 - rho * rho, 0.0)) * rng.normal())


def sample_game_environment_v63(seed: int) -> GameEnvironmentV63:
    """Sample persistent execution, tempo and open-field conditions for one game.

    A minority of worlds receive a broader latent state. The lognormal draws are centered at
    one, so the mechanism primarily widens coherent game paths instead of shifting the scoring
    center. The common latent is shared by both teams; team-specific execution remains separate.
    """

    rng = np.random.default_rng(int(seed) + 6_300_913)
    tail_world = bool(rng.random() < 0.24)
    common_sigma = 0.18 if tail_world else 0.065
    common_z = float(rng.normal())
    common = exp(common_sigma * common_z - 0.5 * common_sigma**2)

    team_sigma = 0.09
    away = common * exp(team_sigma * float(rng.normal()) - 0.5 * team_sigma**2)
    home = common * exp(team_sigma * float(rng.normal()) - 0.5 * team_sigma**2)

    explosive_sigma = 0.20 if tail_world else 0.085
    explosive_z = _correlated_normal(common_z, correlation=0.68, rng=rng)
    explosive = exp(explosive_sigma * explosive_z - 0.5 * explosive_sigma**2)

    tempo_sigma = 0.10 if tail_world else 0.045
    tempo_z = _correlated_normal(common_z, correlation=0.45, rng=rng)
    tempo = exp(tempo_sigma * tempo_z - 0.5 * tempo_sigma**2)

    penalty_sigma = 0.30
    penalty = 0.070 * exp(penalty_sigma * float(rng.normal()) - 0.5 * penalty_sigma**2)
    chaos_sigma = 0.44 if tail_world else 0.34
    chaos = exp(chaos_sigma * float(rng.normal()) - 0.5 * chaos_sigma**2)

    environment = GameEnvironmentV63(
        away_execution=float(np.clip(away, 0.72, 1.38)),
        home_execution=float(np.clip(home, 0.72, 1.38)),
        common_execution=float(np.clip(common, 0.78, 1.28)),
        penalty_rate=float(np.clip(penalty, 0.032, 0.120)),
        chaos_factor=float(np.clip(chaos, 0.48, 2.10)),
        explosive_factor=float(np.clip(explosive, 0.72, 1.38)),
        tempo_factor=float(np.clip(tempo, 0.86, 1.16)),
    )
    set_active_game_environment(environment)
    return environment


def apply_game_environment_v63(
    away: TeamIdentity,
    home: TeamIdentity,
    environment: GameEnvironmentV63,
) -> tuple[TeamIdentity, TeamIdentity]:
    """Apply persistent game state to execution mechanisms, not the scoreboard."""

    def apply(team: TeamIdentity, factor: float) -> TeamIdentity:
        return replace(
            team,
            pass_efficiency=float(np.clip(team.pass_efficiency * factor, 0.62, 1.52)),
            rush_efficiency=float(np.clip(team.rush_efficiency * factor**0.82, 0.66, 1.42)),
            pass_protection=float(np.clip(team.pass_protection * factor**0.18, 0.86, 1.14)),
            run_blocking=float(np.clip(team.run_blocking * factor**0.16, 0.86, 1.14)),
        )

    return apply(away, environment.away_execution), apply(home, environment.home_execution)


def apply_chaos_environment_v63(
    ecology: ChaosEcology,
    environment: GameEnvironmentV63,
) -> ChaosEcology:
    """Let coherent live-ball worlds alter return lanes and rare-event breadth.

    The historical ecology remains the center. High-chaos worlds create more continuation after
    a turnover/return and low-chaos worlds create less; no touchdown probability is introduced.
    """

    factor = environment.chaos_factor
    spread = float(sqrt(factor))
    zero_adjust = float(np.clip(factor**-0.16, 0.82, 1.18))
    event_adjust = float(np.clip(factor**0.34, 0.78, 1.34))
    return replace(
        ecology,
        interception_zero_return_rate=float(
            np.clip(ecology.interception_zero_return_rate * zero_adjust, 0.02, 0.80)
        ),
        interception_return_sd=float(
            np.clip(ecology.interception_return_sd * spread, 1.0, 44.0)
        ),
        interception_40_plus_rate=float(
            np.clip(ecology.interception_40_plus_rate * factor**0.82, 0.0, 0.30)
        ),
        fumble_zero_return_rate=float(
            np.clip(ecology.fumble_zero_return_rate * zero_adjust, 0.08, 0.92)
        ),
        fumble_return_sd=float(np.clip(ecology.fumble_return_sd * spread, 1.0, 40.0)),
        fumble_40_plus_rate=float(
            np.clip(ecology.fumble_40_plus_rate * factor**0.82, 0.0, 0.25)
        ),
        punt_40_plus_rate=float(
            np.clip(ecology.punt_40_plus_rate * factor**0.72, 0.0, 0.24)
        ),
        kickoff_40_plus_rate=float(
            np.clip(ecology.kickoff_40_plus_rate * factor**0.72, 0.0, 0.24)
        ),
        punt_muff_rate=float(np.clip(ecology.punt_muff_rate * event_adjust, 0.0, 0.08)),
        kickoff_muff_rate=float(
            np.clip(ecology.kickoff_muff_rate * event_adjust, 0.0, 0.05)
        ),
        blocked_punt_rate=float(
            np.clip(ecology.blocked_punt_rate * event_adjust, 0.0, 0.08)
        ),
        blocked_field_goal_rate=float(
            np.clip(ecology.blocked_field_goal_rate * event_adjust, 0.0, 0.08)
        ),
    )


def _role_presence_weight(
    *,
    position: str,
    target_weight: float,
    rush_weight: float,
) -> float:
    """Convert separate target/rush roles into one pre-play snap-participation weight.

    v6.2 stored the receiver copy first when a player appeared in both rotations, which meant an
    RB's target role could silently replace his sampled rushing role before the play was even
    chosen. This merge keeps both roles alive. RB/FB presence is driven mostly by the stable rush
    hierarchy; WR/TE presence remains driven mostly by receiving role.
    """

    target = max(float(target_weight), 0.0)
    rush = max(float(rush_weight), 0.0)
    pos = str(position).upper()
    if pos in {"RB", "FB"}:
        value = 0.74 * rush + 0.26 * target
    elif pos == "TE":
        value = 0.90 * target + 0.10 * rush
    elif pos == "WR":
        value = 0.95 * target + 0.05 * rush
    else:
        value = max(target, rush)
    return float(max(value, 0.001))


def snap_presence_weights(offense: Any) -> dict[str, float]:
    """Expose the exact merged skill-player participation weights used by v6.3."""

    receivers = {str(player.player_id): player for player in tuple(offense.receivers)}
    rushers = {str(player.player_id): player for player in tuple(offense.rushers)}
    ordered_ids = list(receivers)
    ordered_ids.extend(player_id for player_id in rushers if player_id not in receivers)
    out: dict[str, float] = {}
    for player_id in ordered_ids:
        receiver = receivers.get(player_id)
        rusher = rushers.get(player_id)
        template = receiver if receiver is not None else rusher
        if template is None:
            continue
        out[player_id] = _role_presence_weight(
            position=str(template.position),
            target_weight=0.0 if receiver is None else float(receiver.usage_weight),
            rush_weight=0.0 if rusher is None else float(rusher.usage_weight),
        )
    return out


def role_aware_offense_skill_players(
    offense: Any,
    package: str,
    key: str,
) -> tuple[object, ...]:
    """Field five non-QB skill players using merged current-world role authority."""

    receivers = {str(player.player_id): player for player in tuple(offense.receivers)}
    rushers = {str(player.player_id): player for player in tuple(offense.rushers)}
    ordered_ids = list(receivers)
    ordered_ids.extend(player_id for player_id in rushers if player_id not in receivers)
    presence = snap_presence_weights(offense)

    players: list[object] = []
    for player_id in ordered_ids:
        template = receivers.get(player_id) or rushers.get(player_id)
        if template is None or str(template.position).upper() not in {"RB", "FB", "TE", "WR"}:
            continue
        players.append(replace(template, usage_weight=presence[player_id]))

    rb_need, te_need, wr_need = reality_snap_v5._skill_counts(package)
    selected: list[object] = []
    selected_ids: set[str] = set()
    groups = (
        ({"RB", "FB"}, rb_need, "rb"),
        ({"TE"}, te_need, "te"),
        ({"WR"}, wr_need, "wr"),
    )
    for positions, count, label in groups:
        pool = tuple(
            player
            for player in players
            if str(player.position).upper() in positions
            and str(player.player_id) not in selected_ids
        )
        picks = reality_snap_v5._weighted_without_replacement(
            pool,
            count,
            key=f"{key}:{label}:v63-role",
        )
        selected.extend(picks)
        selected_ids.update(str(player.player_id) for player in picks)

    if len(selected) < 5:
        pool = tuple(
            player for player in players if str(player.player_id) not in selected_ids
        )
        picks = reality_snap_v5._weighted_without_replacement(
            pool,
            5 - len(selected),
            key=f"{key}:fallback:v63-role",
        )
        selected.extend(picks)
        selected_ids.update(str(player.player_id) for player in picks)

    if len(selected) < 5:
        raise RuntimeError(
            "V6.3 snap package cannot field five unique non-QB skill participants: "
            f"package={package}, available={sorted(ordered_ids)}"
        )
    return tuple(selected[:5])


def explosive_environment_multiplier(*, exponent: float = 0.58) -> float:
    environment = active_game_environment()
    if environment is None:
        return 1.0
    return float(np.clip(environment.explosive_factor**float(exponent), 0.82, 1.22))


def cadence_seconds_v63(base_seconds: int) -> int:
    """Translate persistent tempo into live-clock cadence without creating plays directly."""

    environment = active_game_environment()
    if environment is None:
        return int(base_seconds)
    return int(np.clip(round(float(base_seconds) / environment.tempo_factor), 5.0, 50.0))


def sample_return_yards_v63(
    *,
    mean: float,
    sd: float,
    zero_rate: float,
    forty_plus_rate: float,
    return_skill: float,
    rng: np.random.Generator,
    maximum: float = 100.0,
) -> float:
    """Resolve return yards with a more realistic continuation tail.

    v6.2 had a valid 40+ branch but its post-40 continuation used an exponential mean near ten
    yards, making 60-100 yard returns too difficult even when a breakaway lane had already been
    earned. v6.3 lengthens only that continuation. The return must still physically cover the
    remaining field to score.
    """

    if maximum <= 0.0:
        return 0.0
    if rng.random() < float(np.clip(zero_rate, 0.0, 0.98)):
        return 0.0

    skill = float(np.clip(return_skill, 0.72, 1.32))
    environment = active_game_environment()
    lane_factor = 1.0
    if environment is not None:
        lane_factor = float(
            np.clip(
                environment.explosive_factor**0.20 * environment.chaos_factor**0.10,
                0.84,
                1.22,
            )
        )

    if rng.random() < float(np.clip(forty_plus_rate, 0.0, 0.30)):
        continuation = rng.gamma(
            shape=1.30,
            scale=max(11.5 * skill * lane_factor, 2.0),
        )
        return float(np.clip(40.0 + continuation, 40.0, maximum))

    live_mean = max(mean * skill, 0.5)
    live_sd = max(sd * sqrt(skill), 0.75)
    shape = max((live_mean / live_sd) ** 2, 0.25)
    scale = max((live_sd**2) / live_mean, 0.05)
    return float(np.clip(rng.gamma(shape, scale), 0.0, maximum))


def resolve_turnover_return_v63(
    before: FootballState,
    event: PlayEvent,
    *,
    defense: Any,
    rng: np.random.Generator,
    ecology: ChaosEcology,
) -> ReturnEvent:
    """Resolve live-ball turnover geometry with field leverage and returner skill."""

    if not event.turnover:
        raise ValueError("turnover return resolution requires a turnover event")

    interception = event.pass_result == PassResult.INTERCEPTION
    kind = ReturnKind.INTERCEPTION if interception else ReturnKind.FUMBLE
    if interception:
        raw_spot = before.yardline_100 + float(event.air_yards)
        if raw_spot >= 100.0:
            return ReturnEvent(
                kind=kind,
                original_offense_team_id=before.possession,
                return_team_id=before.defense,
                returner_id=event.primary_defender_id,
                change_spot_yardline_100=100.0,
                return_yards=0.0,
                receiving_yardline_100=20.0,
                touchdown=False,
                touchback=True,
            )
        spot = float(np.clip(raw_spot, 1.0, 99.0))
        zero_rate = ecology.interception_zero_return_rate
        mean = ecology.interception_return_mean
        sd = ecology.interception_return_sd
        forty = ecology.interception_40_plus_rate
    else:
        spot = float(np.clip(before.yardline_100 + float(event.yards), 1.0, 99.0))
        zero_rate = ecology.fumble_zero_return_rate
        mean = ecology.fumble_return_mean
        sd = ecology.fumble_return_sd
        forty = ecology.fumble_40_plus_rate

    defender = _returner(defense, event.primary_defender_id)
    returner_id = None if defender is None else defender.player_id
    return_skill = 1.0 if defender is None else float(defender.returning)
    start = mirror_field(spot)
    distance_to_goal = 100.0 - start

    field_leverage = float(
        np.clip(1.0 + 0.16 * (1.0 - distance_to_goal / 100.0), 1.0, 1.16)
    )
    adjusted_zero = float(
        np.clip(
            zero_rate * (1.0 - 0.20 * (1.0 - distance_to_goal / 100.0)),
            0.0,
            0.95,
        )
    )
    yards = sample_return_yards_v63(
        mean=mean,
        sd=sd,
        zero_rate=adjusted_zero,
        forty_plus_rate=forty,
        return_skill=return_skill * field_leverage,
        rng=rng,
        maximum=max(distance_to_goal, 0.0),
    )
    receiving = start + yards
    touchdown = receiving >= 100.0 - 1e-9
    if touchdown:
        receiving = 100.0
        yards = distance_to_goal
    return ReturnEvent(
        kind=kind,
        original_offense_team_id=before.possession,
        return_team_id=before.defense,
        returner_id=returner_id,
        change_spot_yardline_100=spot,
        return_yards=float(yards),
        receiving_yardline_100=float(receiving),
        touchdown=touchdown,
    )
