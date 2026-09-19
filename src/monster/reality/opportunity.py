from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np

from monster.sim import play_kernel
from monster.sim.matchup_kernel import resolve_pass_matchup
from monster.sim.rich_identity import qb_execution_skill

if TYPE_CHECKING:
    from monster.sim.intent_ecology import IntentEcology
    from monster.sim.matchup_kernel import DefensiveUnit, PassMatchup
    from monster.sim.play_kernel import PlayerIdentity


def participation_assignment_weights(
    players: Sequence[object],
    *,
    category: str,
    attempts: Mapping[tuple[str, str], int],
    shrinkage_samples: float,
) -> np.ndarray:
    """Assignment prior conditional on already being a participant.

    Unlike the v6 compatibility path, this fallback does not read a player's
    pre-sampled usage_weight. Historical concept evidence can tilt assignment
    among the players actually on the snap; absent evidence, those participants
    begin equal.
    """

    if not players:
        return np.asarray([], dtype=float)
    base = np.full(len(players), 1.0 / len(players), dtype=float)
    observed = np.asarray(
        [
            max(int(attempts.get((str(player.player_id), category), 0)), 0)
            for player in players
        ],
        dtype=float,
    )
    total = float(observed.sum())
    if total <= 0.0:
        return base
    observed /= total
    authority = float(np.clip(total / (total + max(shrinkage_samples, 1e-9)), 0.0, 1.0))
    weights = (1.0 - authority) * base + authority * observed
    return weights / weights.sum()


def choose_target_for_depth_v7(
    players: Sequence[object],
    ecology: IntentEcology,
    category: str,
    rng: np.random.Generator,
    *,
    shrinkage_samples: float = 45.0,
):
    """Choose a concept-compatible target from the receivers on this snap."""

    if not players:
        raise ValueError("receiver participant set cannot be empty")
    weights = participation_assignment_weights(
        players,
        category=category,
        attempts=ecology.target_depth_attempts,
        shrinkage_samples=shrinkage_samples,
    )
    return players[int(rng.choice(len(players), p=weights))]


def eligible_rushers_for_geometry_v7(
    players: Sequence[object],
    ecology: IntentEcology,
    category: str,
) -> tuple[object, ...]:
    """Resolve ballcarrier eligibility after personnel participation exists.

    RB/FB participants are ordinary designed-run candidates. QB/WR/TE players
    enter non-sneak designed-run families only when the historical concept table
    contains direct evidence for that player and geometry.
    """

    if category == "qb_sneak":
        qbs = tuple(player for player in players if str(player.position).upper() == "QB")
        return qbs

    eligible: list[object] = []
    for player in players:
        position = str(player.position).upper()
        evidence = int(ecology.rusher_geometry_attempts.get((str(player.player_id), category), 0))
        if position in {"RB", "FB"} or evidence > 0:
            eligible.append(player)

    if eligible:
        return tuple(eligible)
    non_qbs = tuple(player for player in players if str(player.position).upper() != "QB")
    return non_qbs or tuple(players)


def choose_rusher_for_geometry_v7(
    players: Sequence[object],
    ecology: IntentEcology,
    category: str,
    rng: np.random.Generator,
    *,
    shrinkage_samples: float = 60.0,
):
    """Choose a designed-run ballcarrier from the actual snap participants."""

    eligible = eligible_rushers_for_geometry_v7(players, ecology, category)
    if not eligible:
        raise ValueError("rusher participant set cannot be empty")
    if category == "qb_sneak":
        return eligible[0]

    weights = participation_assignment_weights(
        eligible,
        category=category,
        attempts=ecology.rusher_geometry_attempts,
        shrinkage_samples=shrinkage_samples,
    )
    return eligible[int(rng.choice(len(eligible), p=weights))]


def field_read_target_v7(
    offense: object,
    defense: DefensiveUnit,
    rng: np.random.Generator,
    *,
    preferred: PlayerIdentity | None = None,
    fatigue: dict[str, float],
    responsibility_key: str = "static",
) -> tuple[PlayerIdentity, PassMatchup]:
    """Resolve the QB read from live participants, matchup and concept.

    v6 multiplied every read by usage_weight**0.62 even after the snap-world
    participant set was known. v7 removes that preallocated-share term. The
    concept-level preferred target remains a bounded nudge, while actual player
    skill, matchup, QB read quality and fatigue decide the live field read.
    """

    candidates: list[tuple[PlayerIdentity, PassMatchup, float]] = []
    qb_fatigue = play_kernel._fatigue_factor(fatigue, offense.quarterback.player_id)
    line_fatigue = play_kernel._fatigue_factor(fatigue, f"line:{offense.team_id}")
    quarterback_efficiency = qb_execution_skill(offense.quarterback) * qb_fatigue

    for receiver in offense.receivers:
        matchup = resolve_pass_matchup(
            receiver,
            defense,
            pass_protection=offense.pass_protection * line_fatigue,
            quarterback_efficiency=quarterback_efficiency,
            responsibility_key=responsibility_key,
        )
        receiver_fatigue = play_kernel._fatigue_factor(fatigue, receiver.player_id)
        expected_gain = (
            matchup.completion_probability
            * matchup.yards_multiplier
            * receiver.explosive
            * receiver_fatigue
        )
        read_score = max(expected_gain, 0.02) * max(matchup.qb_read_quality, 0.55)
        if preferred is not None and receiver.player_id == preferred.player_id:
            read_score *= 1.35
        candidates.append((receiver, matchup, read_score))

    if not candidates:
        raise ValueError("pass play requires at least one eligible receiver")
    weights = np.asarray([max(item[2], 1e-5) for item in candidates], dtype=float)
    weights /= weights.sum()
    index = int(rng.choice(len(candidates), p=weights))
    receiver, matchup, _ = candidates[index]
    return receiver, matchup


def install_participation_first_opportunity_v7() -> None:
    """Install only the v7 shadow opportunity seams into an already-composed runtime."""

    from monster.sim import intent_ecology

    intent_ecology.choose_target_for_depth = choose_target_for_depth_v7
    intent_ecology.choose_rusher_for_geometry = choose_rusher_for_geometry_v7
    play_kernel._field_read_target = field_read_target_v7
