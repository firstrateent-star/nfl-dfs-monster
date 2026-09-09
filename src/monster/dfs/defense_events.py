from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DefenseEventWorlds:
    sacks: np.ndarray
    defensive_touchdowns: np.ndarray
    special_teams_touchdowns: np.ndarray
    safeties: np.ndarray


# 2022-25 NFL league-average anchors, derived before any market/salary/ownership layer.
# Sacks per team-game: 2.4, 2.6, 2.4, 2.4. Scoring return TDs per team-game:
# INT TD .07/.08/.05/.05, fumble TD .05/.03/.04/.03, PR TD .01/.01/.01/.03,
# KR TD .01/.01/.01/.01, safety .02/.03/.03/.02.
SACKS_PER_TEAM_GAME = 2.45
DEFENSIVE_TD_PER_TEAM_GAME = 0.10
SPECIAL_TEAMS_TD_PER_TEAM_GAME = 0.025
SAFETY_PER_TEAM_GAME = 0.025


def simulate_defense_events(
    *,
    opponent_pass_attempts: np.ndarray,
    opponent_turnovers: np.ndarray,
    pass_disruption: np.ndarray,
    seed: int,
) -> DefenseEventWorlds:
    """Generate missing D/ST events inside the same football worlds.

    The mechanism is market blind. Sack opportunity scales with opponent dropbacks and the
    already-certified pass-disruption state. Defensive return TD opportunity is conditional on
    the world's conserved opponent turnovers rather than sampled independently. Rare special-
    teams TDs and safeties use historical league-average Bernoulli anchors until richer special
    teams state earns authority.
    """
    attempts = np.asarray(opponent_pass_attempts, dtype=float)
    turnovers = np.asarray(opponent_turnovers, dtype=np.int16)
    disruption = np.asarray(pass_disruption, dtype=float)
    if not (attempts.shape == turnovers.shape == disruption.shape):
        raise ValueError("D/ST event inputs must share one correlated world shape")
    rng = np.random.default_rng(seed)

    # 2022-25 average pass attempts are ~32.95/team-game. Convert the 2.45 sack anchor to a
    # per-dropback intensity, then let certified pass disruption move that intensity by world.
    sack_rate = SACKS_PER_TEAM_GAME / (32.95 + SACKS_PER_TEAM_GAME)
    sack_lambda = np.clip(attempts * sack_rate * disruption, 0.0, 7.5)
    sacks = rng.poisson(sack_lambda).astype(np.int16)

    # Historical defensive return TDs are about .10/team-game while turnovers are ~1.25/game.
    # Conditionalizing on the conserved turnover reservoir preserves zero-TD worlds when the
    # defense generated no takeaway and naturally gives multi-takeaway worlds more return paths.
    return_td_per_turnover = DEFENSIVE_TD_PER_TEAM_GAME / 1.25
    defensive_touchdowns = rng.binomial(
        turnovers, np.clip(return_td_per_turnover * disruption, 0.035, 0.14)
    ).astype(np.int16)
    special_teams_touchdowns = rng.binomial(
        1, SPECIAL_TEAMS_TD_PER_TEAM_GAME, attempts.shape
    ).astype(np.int16)
    safeties = rng.binomial(1, SAFETY_PER_TEAM_GAME, attempts.shape).astype(np.int16)
    return DefenseEventWorlds(
        sacks=sacks,
        defensive_touchdowns=defensive_touchdowns,
        special_teams_touchdowns=special_teams_touchdowns,
        safeties=safeties,
    )
