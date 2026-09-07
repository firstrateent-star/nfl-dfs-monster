from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from monster.snapshot.model import GameState, TeamState


@dataclass
class GameWorlds:
    away_points: np.ndarray
    home_points: np.ndarray
    away_drives: np.ndarray
    home_drives: np.ndarray
    away_touchdowns: np.ndarray
    home_touchdowns: np.ndarray
    away_field_goals: np.ndarray
    home_field_goals: np.ndarray
    away_turnovers: np.ndarray
    home_turnovers: np.ndarray
    away_pass_disruption: np.ndarray | None = None
    home_pass_disruption: np.ndarray | None = None
    away_run_efficiency: np.ndarray | None = None
    home_run_efficiency: np.ndarray | None = None

    @property
    def total(self) -> np.ndarray:
        return self.away_points + self.home_points


def _mean_one_lognormal(rng: np.random.Generator, sigma: float, n: int) -> np.ndarray:
    return np.exp(rng.normal(-0.5 * sigma * sigma, sigma, n))


def _clip_probability(value: np.ndarray | float, low: float, high: float) -> np.ndarray:
    return np.clip(value, low, high)


def _blend_rate(offense: float, defense_allowed: float, league_anchor: float) -> float:
    return float(0.46 * offense + 0.46 * defense_allowed + 0.08 * league_anchor)


def _simulate_team_drives(
    rng: np.random.Generator,
    team: TeamState,
    opponent: TeamState,
    worlds: int,
    shared_environment: np.ndarray,
    home: bool,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    # Observed drives/game already contains most realized pace information. Pace therefore
    # acts only as a small centered contextual adjustment rather than multiplying possession
    # volume a second time.
    drive_mu = 0.50 * team.drives_per_game + 0.50 * opponent.drives_per_game
    relative_pace = (team.pace_factor + opponent.pace_factor) / 2.0
    drive_mu *= np.clip(1.0 + 0.25 * (relative_pace - 1.0), 0.94, 1.06)
    drive_mu += 0.12 if home else -0.12
    drive_mu = float(np.clip(drive_mu, 7.5, 14.0))
    drives = rng.poisson(np.clip(drive_mu * shared_environment, 6.0, 16.0)).astype(np.int16)

    td_base = _blend_rate(team.td_drive_rate, opponent.defensive_td_drive_rate_allowed, 0.22)
    fg_base = _blend_rate(team.fg_drive_rate, opponent.defensive_fg_drive_rate_allowed, 0.14)
    to_base = _blend_rate(team.turnover_drive_rate, opponent.defensive_takeaway_drive_rate, 0.11)

    pass_matchup = (
        team.pass_protection_effect
        - opponent.pass_rush_effect
        - 0.70 * opponent.coverage_effect
    )
    run_matchup = team.run_block_effect - opponent.run_defense_effect
    unit_matchup = team.neutral_pass_rate * pass_matchup + (1.0 - team.neutral_pass_rate) * run_matchup

    if team.offensive_line_uncertainty <= 0.0:
        line_state = np.zeros(worlds, dtype=np.float32)
    else:
        line_sigma = float(
            np.clip(
                team.offensive_line_uncertainty
                + 0.035 * (1.0 - team.offensive_line_continuity),
                0.0,
                0.20,
            )
        )
        line_state = rng.normal(0.0, line_sigma, worlds).astype(np.float32)

    pass_disruption = np.clip(
        1.0
        + 1.10
        * (
            opponent.pass_rush_effect
            + 0.60 * opponent.coverage_effect
            - team.pass_protection_effect
        )
        - 0.80 * line_state,
        0.78,
        1.22,
    ).astype(np.float32)
    run_efficiency = np.clip(
        1.0 + 0.90 * run_matchup + 0.60 * line_state,
        0.84,
        1.16,
    ).astype(np.float32)

    quality_base = (
        1.00
        + 0.55 * team.offensive_epa_per_play
        + 0.45 * opponent.defensive_epa_allowed_per_play
        + 0.22 * (team.offensive_explosive_rate - 0.10)
        - 0.18 * (opponent.defensive_explosive_rate_allowed - 0.10)
        + 0.15 * (team.red_zone_td_rate - 0.55)
        + unit_matchup
        + 0.25 * team.special_teams_effect
        + team.injury_effect
        + team.weather_effect
        + team.physical_madden_effect
    )
    quality = np.clip(quality_base + line_state, 0.72, 1.30)
    epistemic_sigma = float(
        np.clip(
            team.uncertainty
            + 0.50 * team.coaching_entropy
            + 0.08 * (1.0 - team.continuity),
            0.05,
            0.30,
        )
    )
    state = _mean_one_lognormal(rng, epistemic_sigma, worlds)

    turnover_multiplier = np.clip(1.0 + 0.55 * (pass_disruption - 1.0), 0.86, 1.14)

    td_p = _clip_probability(td_base * quality * state, 0.07, 0.48)
    to_p = _clip_probability(
        (to_base * turnover_multiplier) / np.sqrt(np.maximum(quality * state, 0.35)),
        0.035,
        0.24,
    )
    fg_p = _clip_probability(
        fg_base
        * (1.0 + team.special_teams_effect)
        * (0.92 + 0.08 * quality)
        * np.sqrt(state),
        0.05,
        0.28,
    )

    touchdowns = rng.binomial(drives, td_p).astype(np.int16)
    remaining = drives - touchdowns
    turnovers = rng.binomial(remaining, to_p).astype(np.int16)
    remaining = remaining - turnovers
    fg_conditional = _clip_probability(
        fg_p / np.maximum(1.0 - td_p - to_p, 0.30), 0.05, 0.45
    )
    field_goals = rng.binomial(remaining, fg_conditional).astype(np.int16)
    two_point = (
        rng.binomial(touchdowns, 0.055) * rng.binomial(1, 0.47, worlds)
    ).astype(np.int16)
    safety = (rng.binomial(1, 0.012, worlds) * 2).astype(np.int16)
    points = (7 * touchdowns + 3 * field_goals + two_point + safety).astype(np.int16)
    return (
        points,
        drives,
        touchdowns,
        field_goals,
        turnovers,
        pass_disruption,
        run_efficiency,
    )


def simulate_game(game: GameState, worlds: int, seed: int) -> GameWorlds:
    """Vectorized market-blind drive simulator."""
    rng = np.random.default_rng(seed)
    shared = _mean_one_lognormal(rng, 0.08 if game.dome else 0.11, worlds)
    away = _simulate_team_drives(rng, game.away, game.home, worlds, shared, False)
    home = _simulate_team_drives(rng, game.home, game.away, worlds, shared, True)
    return GameWorlds(
        away_points=away[0],
        home_points=home[0],
        away_drives=away[1],
        home_drives=home[1],
        away_touchdowns=away[2],
        home_touchdowns=home[2],
        away_field_goals=away[3],
        home_field_goals=home[3],
        away_turnovers=away[4],
        home_turnovers=home[4],
        away_pass_disruption=away[5],
        home_pass_disruption=home[5],
        away_run_efficiency=away[6],
        home_run_efficiency=home[6],
    )
