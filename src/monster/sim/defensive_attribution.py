from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from monster.sim.matchup_kernel import DefensiveIdentity, DefensiveUnit
from monster.sim.play_kernel import PassResult, PlayEvent, PlayType, RunLane


@dataclass
class AttributedDefensiveBoxScore:
    defensive_snaps: int = 0
    pressures: int = 0
    sacks: int = 0
    interceptions: int = 0
    tackles: int = 0
    stuffs: int = 0
    forced_fumbles: int = 0


def _unique_defenders(defense: DefensiveUnit) -> tuple[DefensiveIdentity, ...]:
    seen: set[str] = set()
    out: list[DefensiveIdentity] = []
    for defender in defense.front + defense.coverage:
        if defender.player_id not in seen:
            seen.add(defender.player_id)
            out.append(defender)
    return tuple(out)


def _sample(
    defenders: tuple[DefensiveIdentity, ...],
    rng: np.random.Generator,
    *,
    trait: str,
    position_multipliers: dict[str, float] | None = None,
) -> DefensiveIdentity | None:
    if not defenders:
        return None
    weights = []
    for defender in defenders:
        multiplier = 1.0
        if position_multipliers is not None:
            multiplier = position_multipliers.get(defender.position, 0.45)
        trait_value = max(float(getattr(defender, trait)), 0.05)
        weights.append(max(defender.snap_weight, 0.001) * trait_value * multiplier)
    arr = np.asarray(weights, dtype=float)
    if not np.isfinite(arr).all() or arr.sum() <= 0.0:
        arr = np.ones(len(defenders), dtype=float)
    arr /= arr.sum()
    return defenders[int(rng.choice(len(defenders), p=arr))]


def _pressure_defender(defense: DefensiveUnit, rng: np.random.Generator) -> DefensiveIdentity | None:
    return _sample(
        defense.front,
        rng,
        trait="pass_rush",
        position_multipliers={
            "DE": 1.35,
            "EDGE": 1.35,
            "OLB": 1.12,
            "DT": 1.0,
            "DL": 1.0,
            "NT": 0.75,
            "LB": 0.55,
        },
    )


def _coverage_defender(defense: DefensiveUnit, rng: np.random.Generator) -> DefensiveIdentity | None:
    return _sample(
        defense.coverage,
        rng,
        trait="coverage",
        position_multipliers={
            "CB": 1.25,
            "DB": 1.15,
            "S": 1.10,
            "FS": 1.10,
            "SS": 1.10,
            "LB": 0.75,
            "OLB": 0.65,
        },
    )


def _run_tackler(
    defense: DefensiveUnit,
    event: PlayEvent,
    rng: np.random.Generator,
) -> DefensiveIdentity | None:
    if event.stuffed or event.yards <= 2.0:
        candidates = defense.front + defense.coverage
        multipliers = {
            "DT": 1.35,
            "DL": 1.30,
            "NT": 1.35,
            "DE": 1.15,
            "EDGE": 1.15,
            "LB": 1.25,
            "ILB": 1.35,
            "MLB": 1.35,
            "OLB": 1.0,
            "S": 0.55,
            "FS": 0.50,
            "SS": 0.70,
            "CB": 0.35,
        }
    elif event.yards >= 10.0 or event.run_lane == RunLane.OUTSIDE:
        candidates = defense.coverage + defense.front
        multipliers = {
            "CB": 1.05,
            "DB": 1.05,
            "S": 1.30,
            "FS": 1.25,
            "SS": 1.30,
            "LB": 1.05,
            "ILB": 0.95,
            "OLB": 1.05,
            "DE": 0.55,
            "EDGE": 0.60,
            "DT": 0.25,
            "DL": 0.30,
        }
    else:
        candidates = defense.front + defense.coverage
        multipliers = {
            "LB": 1.35,
            "ILB": 1.45,
            "MLB": 1.45,
            "OLB": 1.10,
            "S": 0.95,
            "SS": 1.0,
            "FS": 0.85,
            "DE": 0.85,
            "EDGE": 0.85,
            "DT": 0.65,
            "DL": 0.65,
            "CB": 0.50,
        }
    return _sample(candidates, rng, trait="tackling", position_multipliers=multipliers)


def _completion_tackler(
    defense: DefensiveUnit,
    event: PlayEvent,
    rng: np.random.Generator,
) -> DefensiveIdentity | None:
    short = event.yards <= 8.0
    multipliers = {
        "CB": 1.20,
        "DB": 1.15,
        "S": 1.20,
        "FS": 1.15,
        "SS": 1.20,
        "LB": 1.05 if short else 0.70,
        "ILB": 1.05 if short else 0.65,
        "OLB": 0.95 if short else 0.65,
        "DE": 0.25,
        "EDGE": 0.30,
        "DT": 0.10,
        "DL": 0.10,
    }
    return _sample(
        defense.coverage + defense.front,
        rng,
        trait="tackling",
        position_multipliers=multipliers,
    )


def attribute_defensive_box_score(
    plays: tuple[PlayEvent, ...],
    defense: DefensiveUnit,
    *,
    seed: int,
) -> dict[str, AttributedDefensiveBoxScore]:
    """Credit defensive events through role-specific causal paths.

    This does not alter the play outcome. It distributes already-generated defensive events
    among plausible participating defenders rather than assigning every event on a play to a
    single representative defender.
    """
    rng = np.random.default_rng(seed)
    defenders = _unique_defenders(defense)
    stats = {d.player_id: AttributedDefensiveBoxScore() for d in defenders}
    scrimmage_snaps = sum(p.play_type in {PlayType.RUN, PlayType.PASS} for p in plays)
    for defender in defenders:
        stats[defender.player_id].defensive_snaps = int(
            np.clip(round(scrimmage_snaps * max(defender.snap_weight, 0.0)), 0, scrimmage_snaps)
        )

    for event in plays:
        if event.play_type not in {PlayType.RUN, PlayType.PASS}:
            continue

        pressure_credit: DefensiveIdentity | None = None
        if event.pressured:
            pressure_credit = _pressure_defender(defense, rng)
            if pressure_credit is not None:
                stats[pressure_credit.player_id].pressures += 1
        if event.pass_result == PassResult.SACK:
            sack_credit = pressure_credit or _pressure_defender(defense, rng)
            if sack_credit is not None:
                stats[sack_credit.player_id].sacks += 1

        if event.pass_result == PassResult.INTERCEPTION:
            interceptor = _coverage_defender(defense, rng)
            if interceptor is not None:
                stats[interceptor.player_id].interceptions += 1

        tackler: DefensiveIdentity | None = None
        if event.rusher_id is not None:
            tackler = _run_tackler(defense, event, rng)
        elif event.pass_result == PassResult.COMPLETE:
            tackler = _completion_tackler(defense, event, rng)
        if tackler is not None:
            stats[tackler.player_id].tackles += 1
            if event.stuffed:
                stats[tackler.player_id].stuffs += 1

        if event.fumbler_id is not None:
            forcer = pressure_credit or tackler
            if forcer is None:
                forcer = _run_tackler(defense, event, rng)
            if forcer is not None:
                stats[forcer.player_id].forced_fumbles += 1

    return stats
