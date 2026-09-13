from __future__ import annotations

from dataclasses import dataclass, field, replace
from hashlib import blake2b
from typing import Any

import numpy as np

from monster.sim.defensive_intent import (
    CoverageShell,
    DefensiveIntent,
    DefensiveTacticalPrior,
    RushPlan,
    sample_defensive_intent,
)
from monster.sim.matchup_kernel import DefensiveUnit
from monster.sim.snap_ecology import team_profile_for_player


@dataclass(frozen=True)
class SnapWorldV5:
    responsibility_key: str
    offense_package: str
    defense_package: str
    offense_participant_ids: tuple[str, ...]
    defense_participant_ids: tuple[str, ...]
    offense_alignment: tuple[str, ...]
    defense_alignment: tuple[str, ...]
    active_receiver_ids: frozenset[str]
    active_rusher_ids: frozenset[str]
    rush_participant_ids: frozenset[str]
    defensive_intent: DefensiveIntent
    defense: DefensiveUnit | None


@dataclass
class SnapEvidenceV5:
    pass_by_target: dict[str, dict[str, Any]] = field(default_factory=dict)
    run: dict[str, Any] = field(default_factory=dict)
    trench_duels: tuple[str, ...] = ()


_WORLD_BY_KEY: dict[str, SnapWorldV5] = {}
_EVIDENCE_BY_KEY: dict[str, SnapEvidenceV5] = {}
_EVENT_META: dict[int, dict[str, Any]] = {}


def _stable_seed(key: str) -> int:
    digest = blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False) % (2**63 - 1)


def _weight(player: object) -> float:
    return max(float(getattr(player, "usage_weight", 0.0) or 0.0), 0.001)


def _snap_weight(player: object) -> float:
    return max(float(getattr(player, "snap_weight", 0.0) or 0.0), 0.001)


def _pid(player: object) -> str:
    return str(getattr(player, "player_id", ""))


def _pos(player: object) -> str:
    return str(getattr(player, "position", "")).upper()


def _weighted_without_replacement(
    players: tuple[object, ...],
    count: int,
    *,
    key: str,
    exposure: bool = False,
) -> tuple[object, ...]:
    if count <= 0 or not players:
        return ()
    ranked: list[tuple[float, str, object]] = []
    for player in players:
        w = _snap_weight(player) if exposure else _weight(player)
        u = max(
            (int.from_bytes(blake2b(f"{key}:{_pid(player)}".encode(), digest_size=8).digest(), "big") + 0.5)
            / (2**64),
            1e-12,
        )
        ranked.append((-np.log(u) / w, _pid(player), player))
    ranked.sort(key=lambda row: (row[0], row[1]))
    return tuple(row[2] for row in ranked[: min(count, len(ranked))])


def _package_probabilities(*, down: int, distance: float, neutral_pass_rate: float) -> tuple[tuple[str, ...], np.ndarray]:
    packages = ("11", "12", "21", "10", "13")
    weights = np.asarray([0.56, 0.20, 0.08, 0.12, 0.04], dtype=float)
    if down >= 3 and distance >= 7.0:
        weights *= np.asarray([1.15, 0.55, 0.35, 1.85, 0.20])
    elif distance <= 2.0:
        weights *= np.asarray([0.80, 1.35, 1.55, 0.45, 1.60])
    pass_tilt = float(np.clip((neutral_pass_rate - 0.56) / 0.15, -1.0, 1.0))
    weights[0] *= 1.0 + 0.10 * pass_tilt
    weights[3] *= 1.0 + 0.22 * pass_tilt
    weights[1] *= 1.0 - 0.08 * pass_tilt
    weights[2] *= 1.0 - 0.12 * pass_tilt
    weights /= weights.sum()
    return packages, weights


def _select_package(key: str, *, down: int, distance: float, neutral_pass_rate: float) -> str:
    packages, weights = _package_probabilities(
        down=down,
        distance=distance,
        neutral_pass_rate=neutral_pass_rate,
    )
    rng = np.random.default_rng(_stable_seed(f"package:{key}"))
    return str(rng.choice(packages, p=weights))


def _skill_counts(package: str) -> tuple[int, int, int]:
    return {
        "10": (1, 0, 4),
        "11": (1, 1, 3),
        "12": (1, 2, 2),
        "13": (1, 3, 1),
        "21": (2, 1, 2),
    }.get(package, (1, 1, 3))


def _offense_skill_players(offense: object, package: str, key: str) -> tuple[object, ...]:
    unique: dict[str, object] = {}
    for player in (*tuple(getattr(offense, "receivers", ())), *tuple(getattr(offense, "rushers", ()))):
        unique.setdefault(_pid(player), player)
    players = tuple(unique.values())
    rb_need, te_need, wr_need = _skill_counts(package)
    selected: list[object] = []
    selected_ids: set[str] = set()
    groups = (
        ({"RB", "FB"}, rb_need, "rb"),
        ({"TE"}, te_need, "te"),
        ({"WR"}, wr_need, "wr"),
    )
    for positions, count, label in groups:
        pool = tuple(player for player in players if _pos(player) in positions and _pid(player) not in selected_ids)
        picks = _weighted_without_replacement(pool, count, key=f"{key}:{label}")
        selected.extend(picks)
        selected_ids.update(_pid(player) for player in picks)
    if len(selected) < 5:
        pool = tuple(player for player in players if _pid(player) not in selected_ids)
        picks = _weighted_without_replacement(pool, 5 - len(selected), key=f"{key}:fallback")
        selected.extend(picks)
    return tuple(selected[:5])


def _offense_alignment(skill: tuple[object, ...], blockers: tuple[object, ...], quarterback_id: str) -> tuple[str, ...]:
    rows = [f"{quarterback_id}:QB"]
    rows.extend(f"{_pid(blocker)}:{_pos(blocker)}" for blocker in blockers)
    wr = [player for player in skill if _pos(player) == "WR"]
    te = [player for player in skill if _pos(player) == "TE"]
    rb = [player for player in skill if _pos(player) in {"RB", "FB"}]
    wr_roles = ("X", "Z", "SLOT", "WR4")
    te_roles = ("Y", "F", "TE3")
    rb_roles = ("BACK", "HBACK")
    rows.extend(f"{_pid(player)}:{wr_roles[min(idx, len(wr_roles)-1)]}" for idx, player in enumerate(wr))
    rows.extend(f"{_pid(player)}:{te_roles[min(idx, len(te_roles)-1)]}" for idx, player in enumerate(te))
    rows.extend(f"{_pid(player)}:{rb_roles[min(idx, len(rb_roles)-1)]}" for idx, player in enumerate(rb))
    return tuple(rows)


def _defense_counts(package: str, *, down: int, distance: float) -> tuple[str, int, int, int]:
    if package == "10" or (down >= 3 and distance >= 8.0):
        return "dime", 4, 1, 6
    if package == "11":
        return "nickel", 4, 2, 5
    return "base", 4, 3, 4


def _defense_participants(
    defense: DefensiveUnit,
    *,
    package: str,
    down: int,
    distance: float,
    key: str,
) -> tuple[str, tuple[object, ...], tuple[str, ...]]:
    defense_package, dl_need, lb_need, db_need = _defense_counts(package, down=down, distance=distance)
    all_players: dict[str, object] = {}
    for player in (*defense.front, *defense.coverage):
        all_players.setdefault(_pid(player), player)
    values = tuple(all_players.values())
    dl = tuple(player for player in values if _pos(player) in {"EDGE", "DE", "DT", "NT", "DL"})
    lb = tuple(player for player in values if _pos(player) in {"LB", "ILB", "MLB", "OLB"})
    db = tuple(player for player in values if _pos(player) in {"CB", "DB", "S", "FS", "SS"})
    selected: list[object] = []
    selected_ids: set[str] = set()
    for pool, need, label in ((dl, dl_need, "dl"), (lb, lb_need, "lb"), (db, db_need, "db")):
        pool = tuple(player for player in pool if _pid(player) not in selected_ids)
        picks = _weighted_without_replacement(pool, need, key=f"{key}:def:{label}", exposure=True)
        selected.extend(picks)
        selected_ids.update(_pid(player) for player in picks)
    if len(selected) < 11:
        pool = tuple(player for player in values if _pid(player) not in selected_ids)
        selected.extend(_weighted_without_replacement(pool, 11 - len(selected), key=f"{key}:def:fallback", exposure=True))
    selected = selected[:11]
    alignment: list[str] = []
    dl_rows = [player for player in selected if _pos(player) in {"EDGE", "DE", "DT", "NT", "DL"}]
    lb_rows = [player for player in selected if _pos(player) in {"LB", "ILB", "MLB", "OLB"}]
    db_rows = [player for player in selected if _pos(player) in {"CB", "DB", "S", "FS", "SS"}]
    dl_roles = ("EDGE_L", "DT_L", "DT_R", "EDGE_R")
    lb_roles = ("MIKE", "WILL", "SAM")
    db_roles = ("CB1", "CB2", "NICKEL", "FS", "SS", "DIME")
    alignment.extend(f"{_pid(player)}:{dl_roles[min(idx, len(dl_roles)-1)]}" for idx, player in enumerate(dl_rows))
    alignment.extend(f"{_pid(player)}:{lb_roles[min(idx, len(lb_roles)-1)]}" for idx, player in enumerate(lb_rows))
    alignment.extend(f"{_pid(player)}:{db_roles[min(idx, len(db_roles)-1)]}" for idx, player in enumerate(db_rows))
    return defense_package, tuple(selected), tuple(alignment)


def _intent_for_snap(key: str, *, state: object, offense: object) -> DefensiveIntent:
    qb_threat = float(np.clip((float(getattr(offense.quarterback, "explosive", 1.0)) - 0.82) / 0.40, 0.0, 1.0))
    late_lead = bool(
        int(getattr(state, "quarter", 1)) >= 4
        and int(getattr(state, "seconds_remaining", 3600)) <= 360
        and float(getattr(state, "score_margin_for_offense", 0.0)) <= -7.0
    )
    sampled = sample_defensive_intent(
        DefensiveTacticalPrior(),
        rng=np.random.default_rng(_stable_seed(f"intent:{key}")),
        short_yardage=float(getattr(state, "distance", 10.0)) <= 3.0,
        late_lead=late_lead,
        qb_run_threat=qb_threat,
    )
    # Low but real authority: identity/skill remains dominant until team-specific tactical priors
    # are compiled from historical coverage/rush-plan data.
    return replace(sampled, authority=0.24)


def prepare_snap_world(
    *,
    state: object,
    offense: object,
    defense: DefensiveUnit | None,
    responsibility_key: str,
) -> tuple[object, DefensiveUnit | None, SnapWorldV5]:
    package = _select_package(
        responsibility_key,
        down=int(getattr(state, "down", 1)),
        distance=float(getattr(state, "distance", 10.0)),
        neutral_pass_rate=float(getattr(offense, "neutral_pass_rate", 0.56)),
    )
    skill = _offense_skill_players(offense, package, responsibility_key)
    active_ids = frozenset(_pid(player) for player in skill)
    receivers = tuple(player for player in getattr(offense, "receivers", ()) if _pid(player) in active_ids)
    rushers = tuple(player for player in getattr(offense, "rushers", ()) if _pid(player) in active_ids or _pos(player) == "QB")
    if not receivers:
        receivers = tuple(getattr(offense, "receivers", ()))
    if not rushers:
        rushers = tuple(getattr(offense, "rushers", ()))
    active_offense = replace(offense, receivers=receivers, rushers=rushers)

    profile = team_profile_for_player(str(getattr(offense.quarterback, "player_id", "")))
    blockers = () if profile is None else profile.offensive_line
    offense_participants = (str(getattr(offense.quarterback, "player_id", "")),) + tuple(
        _pid(blocker) for blocker in blockers
    ) + tuple(_pid(player) for player in skill)
    offense_alignment = _offense_alignment(skill, blockers, str(getattr(offense.quarterback, "player_id", "")))

    intent = _intent_for_snap(responsibility_key, state=state, offense=offense)
    active_defense = defense
    defense_package = "unknown"
    defense_ids: tuple[str, ...] = ()
    defense_alignment: tuple[str, ...] = ()
    rush_ids: frozenset[str] = frozenset()
    if defense is not None:
        defense_package, selected, defense_alignment = _defense_participants(
            defense,
            package=package,
            down=int(getattr(state, "down", 1)),
            distance=float(getattr(state, "distance", 10.0)),
            key=responsibility_key,
        )
        defense_ids = tuple(_pid(player) for player in selected)
        selected_ids = frozenset(defense_ids)
        active_front = tuple(player for player in defense.front if _pid(player) in selected_ids)
        active_coverage = tuple(player for player in defense.coverage if _pid(player) in selected_ids)
        active_defense = replace(defense, front=active_front, coverage=active_coverage)
        rush_count = 5 if intent.rush_plan == RushPlan.BLITZ else 4
        rush_pool = active_front
        rush_selected = _weighted_without_replacement(
            rush_pool,
            min(rush_count, len(rush_pool)),
            key=f"{responsibility_key}:rush-plan:{intent.rush_plan.value}",
            exposure=True,
        )
        rush_ids = frozenset(_pid(player) for player in rush_selected)

    world = SnapWorldV5(
        responsibility_key=responsibility_key,
        offense_package=package,
        defense_package=defense_package,
        offense_participant_ids=tuple(pid for pid in offense_participants if pid),
        defense_participant_ids=defense_ids,
        offense_alignment=offense_alignment,
        defense_alignment=defense_alignment,
        active_receiver_ids=frozenset(_pid(player) for player in receivers),
        active_rusher_ids=frozenset(_pid(player) for player in rushers),
        rush_participant_ids=rush_ids,
        defensive_intent=intent,
        defense=active_defense,
    )
    _WORLD_BY_KEY[responsibility_key] = world
    _EVIDENCE_BY_KEY[responsibility_key] = SnapEvidenceV5()
    return active_offense, active_defense, world


def snap_world(responsibility_key: str) -> SnapWorldV5 | None:
    return _WORLD_BY_KEY.get(responsibility_key)


def planned_rushers(responsibility_key: str, fallback: tuple[object, ...]) -> tuple[object, ...]:
    world = snap_world(responsibility_key)
    if world is None or not world.rush_participant_ids:
        return fallback
    selected = tuple(player for player in fallback if _pid(player) in world.rush_participant_ids)
    return selected or fallback


def record_pass_resolution(responsibility_key: str, target_id: str, **values: Any) -> None:
    evidence = _EVIDENCE_BY_KEY.setdefault(responsibility_key, SnapEvidenceV5())
    evidence.pass_by_target[target_id] = dict(values)


def record_run_resolution(responsibility_key: str, **values: Any) -> None:
    evidence = _EVIDENCE_BY_KEY.setdefault(responsibility_key, SnapEvidenceV5())
    evidence.run = dict(values)


def record_trench_duels(responsibility_key: str, duels: tuple[object, ...]) -> None:
    evidence = _EVIDENCE_BY_KEY.setdefault(responsibility_key, SnapEvidenceV5())
    evidence.trench_duels = tuple(
        f"{getattr(duel, 'blocker_id', '')}>{getattr(duel, 'rusher_id', '')}"
        for duel in duels
    )


def annotate_event(event: object, *, responsibility_key: str) -> None:
    world = _WORLD_BY_KEY.get(responsibility_key)
    evidence = _EVIDENCE_BY_KEY.get(responsibility_key, SnapEvidenceV5())
    if world is None:
        return
    target_id = str(getattr(event, "target_id", "") or "")
    pass_evidence = evidence.pass_by_target.get(target_id, {})
    values = {
        "snap_key": responsibility_key,
        "offense_package": world.offense_package,
        "defense_package": world.defense_package,
        "coverage_shell": world.defensive_intent.coverage_shell.value,
        "rush_plan": world.defensive_intent.rush_plan.value,
        "box_aggression": world.defensive_intent.box_aggression,
        "intent_authority": world.defensive_intent.authority,
        "offense_participant_ids": world.offense_participant_ids,
        "defense_participant_ids": world.defense_participant_ids,
        "offense_alignment": world.offense_alignment,
        "defense_alignment": world.defense_alignment,
        "planned_rusher_ids": tuple(sorted(world.rush_participant_ids)),
        "primary_rusher_id": pass_evidence.get("primary_rusher_id"),
        "safety_defender_id": pass_evidence.get("safety_defender_id"),
        "bracket_defender_id": pass_evidence.get("bracket_defender_id"),
        "pursuit_defender_id": evidence.run.get("pursuit_defender_id"),
        "run_blocker_ids": evidence.run.get("run_blocker_ids", ()),
        "trench_duels": evidence.trench_duels,
    }
    _EVENT_META[id(event)] = values
    _WORLD_BY_KEY.pop(responsibility_key, None)
    _EVIDENCE_BY_KEY.pop(responsibility_key, None)


def event_metadata(event: object) -> dict[str, Any]:
    return _EVENT_META.get(id(event), {})


def coverage_intent_adjustments(responsibility_key: str) -> tuple[float, float, float]:
    world = snap_world(responsibility_key)
    if world is None:
        return 1.0, 1.0, 1.0
    intent = world.defensive_intent
    authority = intent.authority
    if intent.coverage_shell == CoverageShell.TWO_HIGH:
        return 1.0 + 0.30 * authority, 1.0 + 0.18 * authority, 1.0 + 0.12 * authority
    if intent.coverage_shell == CoverageShell.SOFT_ZONE:
        return 1.0 + 0.15 * authority, 1.0 + 0.08 * authority, 1.0 + 0.35 * authority
    if intent.coverage_shell == CoverageShell.MAN:
        return 1.0 - 0.12 * authority, 1.0 - 0.20 * authority, 1.0 - 0.35 * authority
    return 1.0, 1.0, 1.0


def pressure_plan_multiplier(responsibility_key: str) -> float:
    world = snap_world(responsibility_key)
    if world is None:
        return 1.0
    intent = world.defensive_intent
    if intent.rush_plan == RushPlan.BLITZ:
        return 1.0 + 0.14 * intent.authority
    if intent.rush_plan == RushPlan.CONTAIN:
        return 1.0 - 0.08 * intent.authority
    if intent.rush_plan == RushPlan.SIMULATED_PRESSURE:
        return 1.0 + 0.05 * intent.authority
    return 1.0


def run_fit_multiplier(responsibility_key: str) -> float:
    world = snap_world(responsibility_key)
    if world is None:
        return 1.0
    intent = world.defensive_intent
    centered = 2.0 * (intent.box_aggression - 0.5)
    return float(np.clip(1.0 + 0.09 * intent.authority * centered, 0.94, 1.06))
