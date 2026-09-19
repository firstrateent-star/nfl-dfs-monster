from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from monster.reality.opportunity import participation_assignment_weights

# MONSTER's market-blind 2022-2025 QB-family audit puts designed non-sneak
# QB carries at roughly 1.49 per start. Relative to ordinary team rushing
# volume, 5.5% is a deliberately weak population prior; direct team/QB
# geometry evidence quickly dominates it.
LEAGUE_DESIGNED_QB_RUN_SHARE = 0.055
QB_ENTRY_PRIOR_SAMPLES = 12.0


def designed_qb_entry_probability_v72(
    players: Sequence[object],
    ecology: object,
    category: str,
) -> float:
    if category == "qb_sneak":
        return 1.0
    qbs = [
        player for player in players if str(getattr(player, "position", "")).upper() == "QB"
    ]
    if not qbs:
        return 0.0

    attempts = getattr(ecology, "rusher_geometry_attempts", {})
    total = sum(
        max(int(value), 0)
        for (actor_id, concept), value in attempts.items()
        if str(concept) == category
    )
    qb_attempts = sum(
        max(int(attempts.get((str(getattr(qb, "player_id", "")), category), 0)), 0)
        for qb in qbs
    )

    if total <= 0 and qb_attempts <= 0:
        return 0.0
    posterior = (
        qb_attempts + LEAGUE_DESIGNED_QB_RUN_SHARE * QB_ENTRY_PRIOR_SAMPLES
    ) / (total + QB_ENTRY_PRIOR_SAMPLES)
    # Even true option/QB-run teams should not let a generic geometry bucket
    # become a majority-QB carry bucket.
    return float(np.clip(posterior, 0.0, 0.45))


def _non_qb_eligible(
    players: Sequence[object],
    ecology: object,
    category: str,
) -> tuple[object, ...]:
    attempts = getattr(ecology, "rusher_geometry_attempts", {})
    eligible: list[object] = []
    for player in players:
        position = str(getattr(player, "position", "")).upper()
        if position == "QB":
            continue
        evidence = int(
            attempts.get((str(getattr(player, "player_id", "")), category), 0)
        )
        if position in {"RB", "FB"} or evidence > 0:
            eligible.append(player)
    if eligible:
        return tuple(eligible)
    return tuple(
        player
        for player in players
        if str(getattr(player, "position", "")).upper() != "QB"
    )


def choose_rusher_for_geometry_v72(
    players: Sequence[object],
    ecology: object,
    category: str,
    rng: np.random.Generator,
    *,
    shrinkage_samples: float = 60.0,
):
    """Choose the actor after deciding whether the QB concept actually entered."""

    if not players:
        raise ValueError("rusher participant set cannot be empty")

    qbs = tuple(
        player
        for player in players
        if str(getattr(player, "position", "")).upper() == "QB"
    )
    if category == "qb_sneak":
        if not qbs:
            raise ValueError("qb_sneak requires a quarterback participant")
        return qbs[0]

    non_qbs = _non_qb_eligible(players, ecology, category)
    if not non_qbs:
        return qbs[0] if qbs else players[0]

    qb_probability = designed_qb_entry_probability_v72(players, ecology, category)
    if qbs and float(rng.random()) < qb_probability:
        if len(qbs) == 1:
            return qbs[0]
        qb_weights = participation_assignment_weights(
            qbs,
            category=category,
            attempts=getattr(ecology, "rusher_geometry_attempts", {}),
            shrinkage_samples=shrinkage_samples,
        )
        return qbs[int(rng.choice(len(qbs), p=qb_weights))]

    weights = participation_assignment_weights(
        non_qbs,
        category=category,
        attempts=getattr(ecology, "rusher_geometry_attempts", {}),
        shrinkage_samples=shrinkage_samples,
    )
    return non_qbs[int(rng.choice(len(non_qbs), p=weights))]
