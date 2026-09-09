from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from monster.sim.game import GameWorlds, _mean_one_lognormal, _simulate_shared_possessions
from monster.snapshot.model import GameState, TeamState
from monster.snapshot.player import TeamPlayerPool


@dataclass(frozen=True)
class DriveEvents:
    plays: np.ndarray
    touchdowns: np.ndarray
    passing_touchdowns: np.ndarray
    rushing_touchdowns: np.ndarray
    field_goals: np.ndarray
    turnovers: np.ndarray
    punts: np.ndarray
    end_field_position: np.ndarray


def _matchup_state(
    rng: np.random.Generator,
    team: TeamState,
    opponent: TeamState,
    worlds: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    line_sigma = float(
        np.clip(
            team.offensive_line_uncertainty
            + 0.035 * (1.0 - team.offensive_line_continuity),
            0.0,
            0.20,
        )
    )
    line_state = (
        rng.normal(0.0, line_sigma, worlds).astype(np.float32)
        if line_sigma > 0
        else np.zeros(worlds, dtype=np.float32)
    )
    pass_disruption = np.clip(
        1.0
        + 1.10
        * (opponent.pass_rush_effect + 0.60 * opponent.coverage_effect - team.pass_protection_effect)
        - 0.80 * line_state,
        0.78,
        1.22,
    ).astype(np.float32)
    run_efficiency = np.clip(
        1.0 + 0.90 * (team.run_block_effect - opponent.run_defense_effect) + 0.60 * line_state,
        0.84,
        1.16,
    ).astype(np.float32)
    epistemic_sigma = float(
        np.clip(
            team.uncertainty + 0.50 * team.coaching_entropy + 0.08 * (1.0 - team.continuity),
            0.05,
            0.30,
        )
    )
    state = _mean_one_lognormal(rng, epistemic_sigma, worlds).astype(np.float32)
    return pass_disruption, run_efficiency, state


def _simulate_drive(
    rng: np.random.Generator,
    team: TeamState,
    opponent: TeamState,
    pool: TeamPlayerPool,
    active: np.ndarray,
    score_pressure: np.ndarray,
    pass_disruption: np.ndarray,
    run_efficiency: np.ndarray,
    quality_state: np.ndarray,
) -> DriveEvents:
    """Simulate a possession from field position through a terminal football event.

    The scoreboard is not an input. A drive ends because its finite sequence of plays reaches
    the end zone, attempts a field goal, turns the ball over, punts, or exhausts its play cap.
    """
    worlds = len(active)
    field = np.full(worlds, 25.0, dtype=np.float32)
    down = np.ones(worlds, dtype=np.int8)
    distance = np.full(worlds, 10.0, dtype=np.float32)
    alive = active.copy()
    plays = np.zeros(worlds, dtype=np.int16)
    touchdowns = np.zeros(worlds, dtype=np.int8)
    pass_tds = np.zeros(worlds, dtype=np.int8)
    rush_tds = np.zeros(worlds, dtype=np.int8)
    field_goals = np.zeros(worlds, dtype=np.int8)
    turnovers = np.zeros(worlds, dtype=np.int8)
    punts = np.zeros(worlds, dtype=np.int8)

    base_quality = np.clip(
        1.0
        + 0.50 * team.offensive_epa_per_play
        + 0.38 * opponent.defensive_epa_allowed_per_play
        + 0.22 * (team.offensive_success_rate - 0.44)
        + 0.12 * (team.offensive_explosive_rate - opponent.defensive_explosive_rate_allowed)
        + team.injury_effect
        + team.weather_effect
        + team.physical_madden_effect,
        0.78,
        1.24,
    )
    pass_rate = np.clip(
        pool.neutral_pass_rate + pool.script_pass_sensitivity * score_pressure,
        0.34,
        0.78,
    )
    turnover_play_p = np.clip(
        (team.turnover_drive_rate + opponent.defensive_takeaway_drive_rate) / 2.0 / 6.1,
        0.006,
        0.045,
    )

    # A hard cap prevents pathological possessions while normal drives terminate naturally.
    for _ in range(20):
        if not alive.any():
            break

        # Fourth down is a football decision before another scrimmage play. In plausible FG
        # range, attempt the kick; otherwise punt. This keeps score downstream of field position.
        fourth = alive & (down >= 4)
        fg_try = fourth & (field >= 58.0)
        if fg_try.any():
            kick_distance = 117.0 - field
            make_p = np.clip(
                0.96 - 0.0105 * np.maximum(kick_distance - 32.0, 0.0)
                + 0.12 * team.special_teams_effect,
                0.38,
                0.97,
            )
            made = fg_try & (rng.random(worlds) < make_p)
            field_goals[made] = 1
            # A missed field goal is still a terminal scoring attempt, not a punt.
            alive[fg_try] = False

        punt = fourth & ~fg_try
        punts[punt] = 1
        alive[punt] = False
        if not alive.any():
            break

        current = alive.copy()
        plays[current] += 1
        is_pass = current & (rng.random(worlds) < pass_rate)
        is_run = current & ~is_pass

        # Turnovers arise from actual scrimmage plays and terminate the possession.
        to_multiplier = np.clip(1.0 + 0.55 * (pass_disruption - 1.0), 0.82, 1.20)
        turnover = current & (rng.random(worlds) < turnover_play_p * to_multiplier)
        turnovers[turnover] = 1
        alive[turnover] = False
        is_pass &= ~turnover
        is_run &= ~turnover

        sack_p = np.clip(pool.sack_rate * pass_disruption, 0.01, 0.22)
        sack = is_pass & (rng.random(worlds) < sack_p)
        completed_pass = is_pass & ~sack & (
            rng.random(worlds)
            < np.clip(0.62 * base_quality * quality_state / np.sqrt(pass_disruption), 0.38, 0.78)
        )

        yards = np.zeros(worlds, dtype=np.float32)
        if completed_pass.any():
            pass_mean = np.clip(
                10.4
                * base_quality
                * quality_state
                * (1.0 - 0.28 * (pass_disruption - 1.0)),
                6.0,
                16.5,
            )
            pass_yards = rng.gamma(1.55, pass_mean / 1.55)
            explosive = completed_pass & (
                rng.random(worlds)
                < np.clip(
                    0.10
                    + 0.65 * (team.offensive_explosive_rate - 0.10)
                    - 0.45 * (opponent.defensive_explosive_rate_allowed - 0.10),
                    0.035,
                    0.24,
                )
            )
            pass_yards = np.where(explosive, pass_yards + rng.gamma(2.0, 6.0, worlds), pass_yards)
            yards[completed_pass] = pass_yards[completed_pass].astype(np.float32)
        yards[sack] = -rng.uniform(4.0, 10.0, worlds)[sack].astype(np.float32)

        if is_run.any():
            run_mean = np.clip(4.25 * base_quality * quality_state * run_efficiency, 2.5, 7.0)
            run_yards = rng.gamma(1.7, run_mean / 1.7)
            stuffed = is_run & (rng.random(worlds) < np.clip(0.17 / run_efficiency, 0.08, 0.28))
            run_yards = np.where(stuffed, rng.uniform(-2.0, 1.0, worlds), run_yards)
            yards[is_run] = run_yards[is_run].astype(np.float32)

        old_distance = distance.copy()
        field[current] += yards[current]
        field[current] = np.clip(field[current], 1.0, 100.0)

        td = current & (field >= 100.0)
        touchdowns[td] = 1
        pass_tds[td & completed_pass] = 1
        rush_tds[td & is_run] = 1
        # A sack cannot score an offensive TD. Rare completion/run crossing is fully typed.
        alive[td] = False

        continuing = current & ~turnover & ~td
        first_down = continuing & (yards >= old_distance)
        down[first_down] = 1
        distance[first_down] = np.minimum(10.0, 100.0 - field[first_down])

        failed = continuing & ~first_down
        down[failed] += 1
        distance[failed] = np.clip(old_distance[failed] - yards[failed], 1.0, 30.0)

    # Possessions that hit the safety cap end as non-scoring drives. They are tracked as punts
    # for terminal accounting rather than being allowed to create phantom score.
    punts[alive] = 1
    alive[:] = False
    return DriveEvents(
        plays=plays,
        touchdowns=touchdowns,
        passing_touchdowns=pass_tds,
        rushing_touchdowns=rush_tds,
        field_goals=field_goals,
        turnovers=turnovers,
        punts=punts,
        end_field_position=field,
    )


def simulate_event_game(
    game: GameState,
    away_pool: TeamPlayerPool,
    home_pool: TeamPlayerPool,
    worlds: int,
    seed: int,
) -> GameWorlds:
    """Monster v1.2: possessions -> finite plays -> field state -> scoring events -> score."""
    rng = np.random.default_rng(seed)
    shared = _mean_one_lognormal(rng, 0.08 if game.dome else 0.11, worlds)
    away_drives, home_drives = _simulate_shared_possessions(rng, game, worlds, shared)

    away_disruption, away_run_eff, away_state = _matchup_state(rng, game.away, game.home, worlds)
    home_disruption, home_run_eff, home_state = _matchup_state(rng, game.home, game.away, worlds)

    away_points = np.zeros(worlds, dtype=np.int16)
    home_points = np.zeros(worlds, dtype=np.int16)
    away_td = np.zeros(worlds, dtype=np.int16)
    home_td = np.zeros(worlds, dtype=np.int16)
    away_pass_td = np.zeros(worlds, dtype=np.int16)
    home_pass_td = np.zeros(worlds, dtype=np.int16)
    away_rush_td = np.zeros(worlds, dtype=np.int16)
    home_rush_td = np.zeros(worlds, dtype=np.int16)
    away_fg = np.zeros(worlds, dtype=np.int16)
    home_fg = np.zeros(worlds, dtype=np.int16)
    away_to = np.zeros(worlds, dtype=np.int16)
    home_to = np.zeros(worlds, dtype=np.int16)
    away_plays = np.zeros(worlds, dtype=np.int16)
    home_plays = np.zeros(worlds, dtype=np.int16)

    max_drives = int(max(away_drives.max(), home_drives.max()))
    for drive_idx in range(max_drives):
        away_active = away_drives > drive_idx
        if away_active.any():
            a = _simulate_drive(
                rng,
                game.away,
                game.home,
                away_pool,
                away_active,
                home_points.astype(float) - away_points.astype(float),
                away_disruption,
                away_run_eff,
                away_state,
            )
            away_plays += a.plays
            away_td += a.touchdowns
            away_pass_td += a.passing_touchdowns
            away_rush_td += a.rushing_touchdowns
            away_fg += a.field_goals
            away_to += a.turnovers
            away_points += (7 * a.touchdowns + 3 * a.field_goals).astype(np.int16)

        home_active = home_drives > drive_idx
        if home_active.any():
            h = _simulate_drive(
                rng,
                game.home,
                game.away,
                home_pool,
                home_active,
                away_points.astype(float) - home_points.astype(float),
                home_disruption,
                home_run_eff,
                home_state,
            )
            home_plays += h.plays
            home_td += h.touchdowns
            home_pass_td += h.passing_touchdowns
            home_rush_td += h.rushing_touchdowns
            home_fg += h.field_goals
            home_to += h.turnovers
            home_points += (7 * h.touchdowns + 3 * h.field_goals).astype(np.int16)

    # Score is a pure accounting identity over simulated scoring events.
    assert np.array_equal(away_points, 7 * away_td + 3 * away_fg)
    assert np.array_equal(home_points, 7 * home_td + 3 * home_fg)
    assert np.array_equal(away_td, away_pass_td + away_rush_td)
    assert np.array_equal(home_td, home_pass_td + home_rush_td)

    return GameWorlds(
        away_points=away_points,
        home_points=home_points,
        away_drives=away_drives,
        home_drives=home_drives,
        away_touchdowns=away_td,
        home_touchdowns=home_td,
        away_field_goals=away_fg,
        home_field_goals=home_fg,
        away_turnovers=away_to,
        home_turnovers=home_to,
        away_pass_disruption=away_disruption,
        home_pass_disruption=home_disruption,
        away_run_efficiency=away_run_eff,
        home_run_efficiency=home_run_eff,
        away_plays=away_plays,
        home_plays=home_plays,
        away_passing_touchdowns=away_pass_td,
        home_passing_touchdowns=home_pass_td,
        away_rushing_touchdowns=away_rush_td,
        home_rushing_touchdowns=home_rush_td,
    )
