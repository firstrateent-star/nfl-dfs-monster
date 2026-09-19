from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from hashlib import blake2b
from pathlib import Path

import numpy as np
import polars as pl

from monster.feature_compile.v13_identity_bridge import compile_v13_player_identity
from monster.sim import reality_snap_v5
from monster.sim.clock import seconds_remaining_in_quarter
from monster.sim.defensive_intent import DefensiveTacticalPrior, sample_defensive_intent
from monster.sim.play_kernel import PassResult, PlayerIdentity, PlayEvent, PlayType, TeamIdentity


@dataclass
class ClockWorldV721:
    game: str
    half: int
    timeouts: dict[str, int] = field(default_factory=dict)


_BASE_POLICY: Callable | None = None
_BASE_SCRIMMAGE: Callable | None = None
_BASE_DEFENSIVE_INTENT: Callable | None = None
_BASE_TEAM_IDENTITY: Callable | None = None

_CLOCK_WORLDS: dict[int, ClockWorldV721] = {}
_QB_DEPTH: dict[str, tuple[PlayerIdentity, ...]] = {}
_SCRIPT_ROWS: list[dict[str, object]] = []
_PRESERVATION_ROWS: list[dict[str, object]] = []


def _game_id(state: object) -> str:
    return (
        f"{getattr(state, 'away_team_id', '')}@"
        f"{getattr(state, 'home_team_id', '')}"
    )


def _half(state: object) -> int:
    return 1 if int(getattr(state, "quarter", 1)) <= 2 else 2


def _clock_world(state: object, rng: np.random.Generator) -> ClockWorldV721:
    key = id(rng)
    game = _game_id(state)
    half = _half(state)
    current = _CLOCK_WORLDS.get(key)
    if current is None or current.game != game:
        current = ClockWorldV721(game=game, half=half)
        _CLOCK_WORLDS[key] = current
    if current.half != half:
        current.half = half
        current.timeouts = {}
    for team in (
        str(getattr(state, "away_team_id", "") or ""),
        str(getattr(state, "home_team_id", "") or ""),
    ):
        if team:
            current.timeouts.setdefault(team, 3)
    return current


def _stable_uniform(*parts: object) -> float:
    digest = blake2b(
        ":".join(str(part) for part in parts).encode("utf-8"),
        digest_size=8,
    ).digest()
    return (int.from_bytes(digest, "big") + 0.5) / (2**64)


def _hurry_floor(state: object) -> float:
    quarter = int(getattr(state, "quarter", 1))
    clock = seconds_remaining_in_quarter(int(getattr(state, "seconds_remaining", 3600)))
    margin = int(getattr(state, "score_margin_for_offense", 0))

    if quarter == 2 and clock <= 120:
        urgency = float(np.clip((120.0 - clock) / 105.0, 0.0, 1.0))
        if margin <= 8:
            return 0.48 + 0.42 * urgency
        return 0.12 + 0.18 * urgency

    if quarter == 4:
        if margin < 0 and clock <= 240:
            urgency = float(np.clip((240.0 - clock) / 210.0, 0.0, 1.0))
            return 0.52 + 0.40 * urgency
        if margin == 0 and clock <= 120:
            urgency = float(np.clip((120.0 - clock) / 105.0, 0.0, 1.0))
            return 0.42 + 0.38 * urgency
    return 0.0


def policy_for_state_v721(state, offense):
    if _BASE_POLICY is None:
        raise RuntimeError("V7.2.1 base situation policy is not configured")
    base = _BASE_POLICY(state, offense)
    clock = seconds_remaining_in_quarter(int(state.seconds_remaining))
    margin = state.score_margin_for_offense
    hurry = max(float(base.hurry_probability), _hurry_floor(state))

    # A four-minute offense with a meaningful lead should consume the play clock
    # rather than inherit generic hurry behavior from another state layer.
    if state.quarter == 4 and clock <= 240 and margin >= 7:
        hurry = min(hurry, 0.04)

    return replace(base, hurry_probability=float(np.clip(hurry, 0.02, 0.95)))


def _clock_would_run(event: PlayEvent) -> bool:
    if event.touchdown or event.turnover:
        return False
    if event.play_type == PlayType.RUN:
        return True
    if event.play_type != PlayType.PASS:
        return False
    return event.pass_result in {
        PassResult.COMPLETE,
        PassResult.SACK,
        PassResult.SCRAMBLE,
    }


def _timeout_probability(
    state: object,
    *,
    timeout_team: str,
    offense_team: str,
) -> float:
    quarter = int(getattr(state, "quarter", 1))
    clock = seconds_remaining_in_quarter(int(getattr(state, "seconds_remaining", 3600)))
    margin = int(getattr(state, "score_margin_for_offense", 0))
    offense_timeout = timeout_team == offense_team

    if quarter == 2 and offense_timeout and clock <= 90 and margin <= 8:
        if clock <= 30:
            return 0.96
        if clock <= 60:
            return 0.82
        return 0.56

    if quarter == 4 and offense_timeout and margin <= 0:
        if clock <= 60:
            return 0.97
        if clock <= 120:
            return 0.88
        if clock <= 180:
            return 0.58
        if margin <= -9 and clock <= 240:
            return 0.42

    # The defense owns the timeout when the offense is protecting a close lead.
    if quarter == 4 and not offense_timeout and 1 <= margin <= 16 and clock <= 180:
        if clock <= 60:
            return 0.96
        if clock <= 120:
            return 0.84
        return 0.58

    return 0.0


def _play_duration(event: PlayEvent) -> int:
    if event.play_type == PlayType.PASS:
        if event.pass_result == PassResult.SACK:
            return 7
        if event.pass_result == PassResult.SCRAMBLE:
            return 7
        return 6
    if event.play_type == PlayType.RUN:
        return 7
    return min(max(int(event.elapsed_seconds), 3), 10)


def _apply_timeout_clock(
    state: object,
    offense: TeamIdentity,
    event: PlayEvent,
    rng: np.random.Generator,
) -> tuple[PlayEvent, str | None]:
    if not _clock_would_run(event):
        return event, None

    world = _clock_world(state, rng)
    offense_team = str(offense.team_id)
    defense_team = str(getattr(state, "defense", ""))
    candidates = (offense_team, defense_team)

    chosen: str | None = None
    chosen_p = 0.0
    for team in candidates:
        if world.timeouts.get(team, 0) <= 0:
            continue
        probability = _timeout_probability(
            state,
            timeout_team=team,
            offense_team=offense_team,
        )
        if probability <= chosen_p:
            continue
        draw = _stable_uniform(
            world.game,
            world.half,
            int(getattr(state, "seconds_remaining", 0)),
            int(getattr(state, "down", 1)),
            offense_team,
            team,
            "timeout",
        )
        if draw < probability:
            chosen = team
            chosen_p = probability

    if chosen is None:
        return event, None

    world.timeouts[chosen] = max(world.timeouts.get(chosen, 0) - 1, 0)
    elapsed = min(int(event.elapsed_seconds), _play_duration(event))
    adjusted = replace(event, elapsed_seconds=elapsed)

    # V7.2's snap topology is keyed by object identity. Preserve the same metadata
    # when the clock wrapper creates an immutable event copy.
    meta = reality_snap_v5.event_metadata(event)
    if meta:
        reality_snap_v5._EVENT_META[id(adjusted)] = dict(meta)

    return adjusted, chosen


def register_qb_depth_v721(
    team_id: str,
    pool,
    reality,
) -> None:
    rows: list[PlayerIdentity] = []
    for player in pool.players:
        if player.position.upper() != "QB":
            continue
        inputs = reality.get(player.player_id)
        if inputs is None:
            identity = PlayerIdentity(
                player.player_id,
                player.display_name,
                player.position,
                usage_weight=max(float(player.qb_pass_share), 0.001),
            )
        else:
            identity, _ = compile_v13_player_identity(
                player_id=player.player_id,
                name=player.display_name,
                position=player.position,
                usage_weight=max(float(player.qb_pass_share), 0.001),
                inputs=inputs,
            )
        rows.append(identity)

    def key(identity: PlayerIdentity) -> tuple[int, float, str]:
        from monster.reality.participation_authority_v72 import role_for

        role = role_for(identity.player_id)
        rank = 99 if role is None or role.depth_rank <= 0 else role.depth_rank
        active = 0.0 if role is None else role.active_probability
        return rank, -active, identity.player_id

    rows.sort(key=key)
    _QB_DEPTH[str(team_id)] = tuple(rows)


def team_identity_v721(*args, **kwargs):
    if _BASE_TEAM_IDENTITY is None:
        raise RuntimeError("V7.2.1 base team identity is not configured")
    team = _BASE_TEAM_IDENTITY(*args, **kwargs)
    team_id = str(args[0] if args else kwargs["team_id"])
    pool = args[1] if len(args) > 1 else kwargs["pool"]
    reality = args[2] if len(args) > 2 else kwargs["reality"]
    register_qb_depth_v721(team_id, pool, reality)
    return team


def _backup_qb(team_id: str, starter_id: str) -> PlayerIdentity | None:
    from monster.reality.participation_authority_v72 import role_for

    candidates = []
    for identity in _QB_DEPTH.get(str(team_id), ()):
        if identity.player_id == starter_id:
            continue
        role = role_for(identity.player_id)
        if role is None or role.status != "ACT" or role.active_probability < 0.50:
            continue
        candidates.append(identity)
    return candidates[0] if candidates else None


def _preservation_strength(state: object) -> float:
    if int(getattr(state, "quarter", 1)) != 4:
        return 0.0
    clock = seconds_remaining_in_quarter(int(getattr(state, "seconds_remaining", 3600)))
    margin = int(getattr(state, "score_margin_for_offense", 0))
    if margin < 17 or clock > 480:
        return 0.0
    margin_signal = float(np.clip((margin - 17.0) / 14.0, 0.0, 1.0))
    time_signal = float(np.clip((480.0 - clock) / 360.0, 0.0, 1.0))
    return float(np.clip(0.30 + 0.35 * margin_signal + 0.35 * time_signal, 0.0, 0.92))


def _reweight_for_preservation(
    players: tuple[PlayerIdentity, ...],
    strength: float,
) -> tuple[PlayerIdentity, ...]:
    if strength <= 0.0 or len(players) <= 1:
        return players
    weights = np.asarray([max(float(p.usage_weight), 0.001) for p in players], dtype=float)
    order = np.argsort(weights)[::-1]
    top_n = 1 if len(players) < 4 else 2
    protected = {int(i) for i in order[:top_n]}
    adjusted = weights.copy()
    for i in range(len(players)):
        if i in protected:
            adjusted[i] *= 1.0 - 0.62 * strength
        else:
            adjusted[i] *= 1.0 + 0.42 * strength
    if adjusted.sum() > 0:
        adjusted *= weights.sum() / adjusted.sum()
    return tuple(
        replace(player, usage_weight=float(adjusted[i]))
        for i, player in enumerate(players)
    )


def apply_preservation_v721(
    offense: TeamIdentity,
    state: object,
    rng: np.random.Generator | None = None,
) -> TeamIdentity:
    strength = _preservation_strength(state)
    if strength <= 0.0:
        return offense

    quarterback = offense.quarterback
    backup = _backup_qb(offense.team_id, quarterback.player_id)
    period_clock = seconds_remaining_in_quarter(int(getattr(state, "seconds_remaining", 3600)))
    margin = int(getattr(state, "score_margin_for_offense", 0))

    # QB removal requires a very strong terminal state; skill-player workload can
    # taper earlier. This avoids pretending every 17-point fourth-quarter lead is over.
    qb_replaced = False
    if backup is not None and margin >= 24 and period_clock <= 360:
        quarterback = backup
        qb_replaced = True

    receivers = _reweight_for_preservation(offense.receivers, strength)
    rushers = _reweight_for_preservation(offense.rushers, strength)

    _PRESERVATION_ROWS.append(
        {
            "game": _game_id(state),
            "world_key": None if rng is None else id(rng),
            "team": offense.team_id,
            "quarter": int(getattr(state, "quarter", 1)),
            "seconds_remaining": int(getattr(state, "seconds_remaining", 0)),
            "period_clock": period_clock,
            "margin": margin,
            "strength": strength,
            "qb_replaced": qb_replaced,
            "starter_qb": offense.quarterback.player_id,
            "active_qb": quarterback.player_id,
        }
    )
    return replace(
        offense,
        quarterback=quarterback,
        receivers=receivers,
        rushers=rushers,
    )


def defensive_intent_v721(key: str, *, state: object, offense: object):
    if _BASE_DEFENSIVE_INTENT is None:
        raise RuntimeError("V7.2.1 base defensive intent is not configured")
    base = _BASE_DEFENSIVE_INTENT(key, state=state, offense=offense)

    quarter = int(getattr(state, "quarter", 1))
    clock = seconds_remaining_in_quarter(int(getattr(state, "seconds_remaining", 3600)))
    margin = int(getattr(state, "score_margin_for_offense", 0))
    if quarter != 4 or margin > -9 or clock > 600:
        return base

    qb_threat = float(
        np.clip(
            (float(getattr(offense.quarterback, "explosive", 1.0)) - 0.82) / 0.40,
            0.0,
            1.0,
        )
    )
    sampled = sample_defensive_intent(
        DefensiveTacticalPrior(),
        rng=np.random.default_rng(
            int.from_bytes(
                blake2b(f"v721-garbage-defense:{key}".encode(), digest_size=8).digest(),
                "big",
            )
            % (2**63 - 1)
        ),
        short_yardage=float(getattr(state, "distance", 10.0)) <= 3.0,
        late_lead=True,
        qb_run_threat=qb_threat,
    )
    severity = float(
        np.clip(
            0.28
            + 0.08 * ((-margin - 9) / 8.0)
            + 0.08 * ((600 - clock) / 600.0),
            0.28,
            0.42,
        )
    )
    return replace(sampled, authority=max(base.authority, severity))


def simulate_scrimmage_play_v721(
    state,
    offense,
    defense_strength,
    rng,
    defense=None,
):
    if _BASE_SCRIMMAGE is None:
        raise RuntimeError("V7.2.1 base scrimmage runtime is not configured")

    scripted_offense = apply_preservation_v721(offense, state, rng)
    event = _BASE_SCRIMMAGE(
        state,
        scripted_offense,
        defense_strength,
        rng,
        defense=defense,
    )
    adjusted, timeout_team = _apply_timeout_clock(
        state,
        scripted_offense,
        event,
        rng,
    )

    world = _clock_world(state, rng)
    _SCRIPT_ROWS.append(
        {
            "game": world.game,
            "world_key": id(rng),
            "quarter": int(getattr(state, "quarter", 1)),
            "seconds_remaining": int(getattr(state, "seconds_remaining", 0)),
            "period_clock": seconds_remaining_in_quarter(
                int(getattr(state, "seconds_remaining", 0))
            ),
            "offense": scripted_offense.team_id,
            "margin": int(getattr(state, "score_margin_for_offense", 0)),
            "down": int(getattr(state, "down", 1)),
            "distance": float(getattr(state, "distance", 10.0)),
            "play_type": adjusted.play_type.value,
            "pass_result": (
                "" if adjusted.pass_result is None else adjusted.pass_result.value
            ),
            "elapsed_before_clock_management": int(event.elapsed_seconds),
            "elapsed_after_clock_management": int(adjusted.elapsed_seconds),
            "timeout_team": "" if timeout_team is None else timeout_team,
            "offense_timeouts_remaining": world.timeouts.get(
                scripted_offense.team_id, 0
            ),
            "defense_timeouts_remaining": world.timeouts.get(
                str(getattr(state, "defense", "")), 0
            ),
            "hurry_probability": policy_for_state_v721(
                state, scripted_offense
            ).hurry_probability,
            "preservation_strength": _preservation_strength(state),
        }
    )
    return adjusted


def configure_v721_hooks(
    *,
    policy: Callable,
    scrimmage: Callable,
    defensive_intent: Callable,
    team_identity: Callable,
) -> None:
    global _BASE_POLICY, _BASE_SCRIMMAGE, _BASE_DEFENSIVE_INTENT, _BASE_TEAM_IDENTITY
    # Runtime composers can be invoked more than once inside audit processes.
    # Never capture our own wrapper as its inherited base.
    if policy is not policy_for_state_v721:
        _BASE_POLICY = policy
    if scrimmage is not simulate_scrimmage_play_v721:
        _BASE_SCRIMMAGE = scrimmage
    if defensive_intent is not defensive_intent_v721:
        _BASE_DEFENSIVE_INTENT = defensive_intent
    if team_identity is not team_identity_v721:
        _BASE_TEAM_IDENTITY = team_identity


def reset_v721_state() -> None:
    _CLOCK_WORLDS.clear()
    _QB_DEPTH.clear()
    _SCRIPT_ROWS.clear()
    _PRESERVATION_ROWS.clear()


def script_rows_v721() -> list[dict[str, object]]:
    return list(_SCRIPT_ROWS)


def write_game_script_telemetry_v721(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if _SCRIPT_ROWS:
        pl.DataFrame(_SCRIPT_ROWS).write_csv(out / "game_script_v721.csv")
    if _PRESERVATION_ROWS:
        pl.DataFrame(_PRESERVATION_ROWS).write_csv(
            out / "preservation_v721.csv"
        )
