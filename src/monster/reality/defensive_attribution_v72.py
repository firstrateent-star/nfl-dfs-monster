from __future__ import annotations

import numpy as np

from monster.sim import defensive_attribution as base
from monster.sim.matchup_kernel import DefensiveIdentity, DefensiveUnit
from monster.sim.play_kernel import PassResult, PlayEvent, PlayType
from monster.sim.reality_snap_v5 import event_metadata


def _by_id(
    defense: DefensiveUnit,
) -> dict[str, DefensiveIdentity]:
    return {
        defender.player_id: defender
        for defender in (*defense.front, *defense.coverage)
    }


def _valid(
    player_id: str | None,
    defenders: dict[str, DefensiveIdentity],
) -> DefensiveIdentity | None:
    if player_id is None:
        return None
    return defenders.get(str(player_id))


def attribute_defensive_box_score_v72(
    plays: tuple[PlayEvent, ...],
    defense: DefensiveUnit,
    *,
    seed: int,
) -> dict[str, base.AttributedDefensiveBoxScore]:
    """Attribute generated events to the defenders who actually owned the snap.

    V7.1 re-sampled attribution after the game, which could award a sack,
    pressure, tackle or interception to a defender who was not the causal
    participant on that play. V7.2 consumes the existing snap-world metadata
    first and uses the inherited role-weighted sampler only as a fail-safe.
    """

    rng = np.random.default_rng(seed)
    defenders = _by_id(defense)
    stats = {
        player_id: base.AttributedDefensiveBoxScore()
        for player_id in defenders
    }

    for event in plays:
        if event.play_type not in {PlayType.RUN, PlayType.PASS}:
            continue
        meta = event_metadata(event)
        participant_ids = tuple(
            str(player_id)
            for player_id in meta.get("defense_participant_ids", ())
            if str(player_id) in defenders
        )
        if participant_ids:
            for player_id in participant_ids:
                stats[player_id].defensive_snaps += 1
        else:
            for defender in defenders.values():
                if rng.random() < float(np.clip(defender.snap_weight, 0.0, 1.0)):
                    stats[defender.player_id].defensive_snaps += 1

        pressure_credit = _valid(meta.get("primary_rusher_id"), defenders)
        if event.pressured:
            pressure_credit = pressure_credit or base._pressure_defender(defense, rng)
            if pressure_credit is not None:
                stats[pressure_credit.player_id].pressures += 1

        if event.pass_result == PassResult.SACK:
            sack_credit = (
                pressure_credit
                or _valid(meta.get("primary_rusher_id"), defenders)
                or base._pressure_defender(defense, rng)
            )
            if sack_credit is not None:
                stats[sack_credit.player_id].sacks += 1

        if event.pass_result == PassResult.INTERCEPTION:
            interceptor = (
                _valid(event.primary_defender_id, defenders)
                or _valid(meta.get("safety_defender_id"), defenders)
                or _valid(meta.get("bracket_defender_id"), defenders)
                or base._coverage_defender(defense, rng)
            )
            if interceptor is not None:
                stats[interceptor.player_id].interceptions += 1

        tackler: DefensiveIdentity | None = None
        if event.rusher_id is not None:
            if event.stuffed or event.yards <= 2.0:
                tackler = _valid(event.primary_defender_id, defenders)
            if tackler is None:
                tackler = _valid(meta.get("pursuit_defender_id"), defenders)
            if tackler is None:
                tackler = base._run_tackler(defense, event, rng)
        elif event.pass_result == PassResult.COMPLETE:
            tackler = (
                _valid(event.primary_defender_id, defenders)
                or _valid(meta.get("safety_defender_id"), defenders)
                or base._completion_tackler(defense, event, rng)
            )

        if tackler is not None:
            stats[tackler.player_id].tackles += 1
            if event.stuffed:
                stats[tackler.player_id].stuffs += 1

        if event.fumbler_id is not None:
            forcer = pressure_credit or tackler
            if forcer is None:
                forcer = (
                    _valid(meta.get("pursuit_defender_id"), defenders)
                    or base._run_tackler(defense, event, rng)
                )
            if forcer is not None:
                stats[forcer.player_id].forced_fumbles += 1

    return stats
