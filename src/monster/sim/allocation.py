from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from monster.sim.game import GameWorlds
from monster.snapshot.player import PlayerState, TeamPlayerPool

STAT_KEYS = (
    "targets",
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "rush_attempts",
    "rushing_yards",
    "rushing_tds",
    "pass_attempts",
    "completions",
    "passing_yards",
    "passing_tds",
)


@dataclass
class TeamAllocationWorlds:
    player_stats: dict[str, dict[str, np.ndarray]]
    team_plays: np.ndarray
    team_dropbacks: np.ndarray
    team_sacks: np.ndarray
    team_pass_attempts: np.ndarray
    team_targets: np.ndarray
    team_rush_attempts: np.ndarray
    passing_tds: np.ndarray
    rushing_tds: np.ndarray
    pass_disruption: np.ndarray
    run_efficiency: np.ndarray


@dataclass
class GameAllocationWorlds:
    away: TeamAllocationWorlds
    home: TeamAllocationWorlds


def _mean_one_lognormal(
    rng: np.random.Generator, sigma: float, worlds: int
) -> np.ndarray:
    return np.exp(rng.normal(-0.5 * sigma * sigma, sigma, worlds))


def _normalized_base(values: np.ndarray) -> np.ndarray:
    values = np.clip(values.astype(float), 0.0, None)
    total = values.sum()
    if total <= 0:
        return np.repeat(1.0 / len(values), len(values))
    return values / total


def _sample_role_shares(
    rng: np.random.Generator,
    players: tuple[PlayerState, ...],
    base_values: np.ndarray,
    worlds: int,
) -> np.ndarray:
    """Sample uncertain role shares without inventing opportunity for structural zeros."""
    raw = np.clip(base_values.astype(float), 0.0, None)
    positive = raw > 0.0
    if not positive.any():
        base = np.repeat(1.0 / len(raw), len(raw))
        eligible = np.ones(len(raw), dtype=bool)
    else:
        base = raw / raw.sum()
        eligible = positive

    mean_uncertainty = float(np.mean([p.role_uncertainty for p in players]))
    concentration = float(np.clip(1.0 / max(mean_uncertainty**2, 0.0025), 10.0, 250.0))
    weights = np.zeros((worlds, len(players)), dtype=float)
    eligible_shapes = np.clip(base[eligible] * concentration, 0.05, None)
    weights[:, eligible] = rng.gamma(
        shape=eligible_shapes,
        scale=1.0,
        size=(worlds, int(eligible.sum())),
    )

    active_probability = np.array([p.active_probability for p in players], dtype=float)
    effectiveness = np.array([p.effectiveness_if_active for p in players], dtype=float)
    active = rng.random((worlds, len(players))) < np.clip(active_probability, 0.0, 1.0)
    weights *= active
    weights *= np.clip(effectiveness, 0.25, 1.25)

    row_sum = weights.sum(axis=1)
    empty = row_sum <= 0
    if empty.any():
        fallback = int(np.argmax(base))
        weights[empty, fallback] = 1.0
        row_sum = weights.sum(axis=1)
    return weights / row_sum[:, None]


def _allocate_integer_counts(
    rng: np.random.Generator, counts: np.ndarray, shares: np.ndarray
) -> np.ndarray:
    """Vectorized sequential-binomial allocation that conserves every world total."""
    worlds, n_players = shares.shape
    out = np.zeros((worlds, n_players), dtype=np.int16)
    remaining = counts.astype(np.int64).copy()
    remaining_share = np.ones(worlds, dtype=float)
    for idx in range(n_players - 1):
        probability = np.divide(
            shares[:, idx],
            np.maximum(remaining_share, 1e-12),
            out=np.zeros(worlds, dtype=float),
            where=remaining_share > 0,
        )
        probability = np.clip(probability, 0.0, 1.0)
        draw = rng.binomial(remaining, probability)
        out[:, idx] = draw.astype(np.int16)
        remaining -= draw
        remaining_share -= shares[:, idx]
    out[:, -1] = remaining.astype(np.int16)
    return out


def _gamma_sum(
    rng: np.random.Generator,
    counts: np.ndarray,
    mean_per_event: float,
    shape_per_event: float = 2.2,
) -> np.ndarray:
    counts_float = counts.astype(float)
    shape = np.maximum(counts_float * shape_per_event, 1e-6)
    scale = max(mean_per_event, 0.1) / shape_per_event
    values = rng.gamma(shape, scale)
    values[counts == 0] = 0.0
    return values.astype(np.float32)


def _empty_player_stats(worlds: int) -> dict[str, np.ndarray]:
    return {key: np.zeros(worlds, dtype=np.float32) for key in STAT_KEYS}


def _shared_game_play_budget(
    rng: np.random.Generator,
    away_pool: TeamPlayerPool,
    home_pool: TeamPlayerPool,
    away_drives: np.ndarray,
    home_drives: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample one finite game play budget, then allocate it between the two offenses.

    Historical 2025 REG games averaged 121.1 offensive scrimmage plays with an 8.56-play
    standard deviation. Team drive count correlated only ~0.20 with team play count, so drive
    worlds should shape the split modestly rather than mechanically multiplying plays per drive.
    The matchup-specific center still comes from each team's inherited plays/drive identity.
    """
    worlds = len(away_drives)

    away_raw = away_drives.astype(float) * away_pool.plays_per_drive
    home_raw = home_drives.astype(float) * home_pool.plays_per_drive
    away_center = float(np.mean(away_raw))
    home_center = float(np.mean(home_raw))
    game_center = float(np.clip(away_center + home_center, 105.0, 137.0))

    # One 60-minute game owns the variance budget. The normal is empirical, bounded only at
    # implausible extremes; it replaces two independent team Poisson play lotteries.
    total_plays = np.rint(rng.normal(game_center, 8.56, worlds))
    total_plays = np.clip(total_plays, 90, 150).astype(np.int16)

    base_away_share = away_center / max(game_center, 1e-9)
    total_drives = np.maximum(away_drives + home_drives, 1).astype(float)
    away_drive_share = away_drives.astype(float) / total_drives

    # Drive count contains some information about play share, but only weakly. A small latent
    # possession-efficiency term allows real team play imbalance while conserving the game total.
    drive_signal = 0.20 * (away_drive_share - 0.50)
    split_sigma = float(
        np.clip(
            0.035 + 0.10 * (away_pool.play_volume_uncertainty + home_pool.play_volume_uncertainty - 0.12),
            0.025,
            0.055,
        )
    )
    latent_split = rng.normal(0.0, split_sigma, worlds)
    away_share = np.clip(base_away_share + drive_signal + latent_split, 0.30, 0.70)

    away_plays = rng.binomial(total_plays.astype(np.int64), away_share).astype(np.int16)
    home_plays = (total_plays - away_plays).astype(np.int16)
    return away_plays, home_plays


def _allocate_team(
    rng: np.random.Generator,
    pool: TeamPlayerPool,
    team_points: np.ndarray,
    opponent_points: np.ndarray,
    team_plays: np.ndarray,
    team_touchdowns: np.ndarray,
    pass_disruption: np.ndarray,
    run_efficiency: np.ndarray,
) -> TeamAllocationWorlds:
    if not pool.players:
        raise ValueError(f"Team {pool.team_id} has no players in its allocation pool")

    worlds = len(team_points)
    score_pressure = opponent_points.astype(float) - team_points.astype(float)
    pass_rate = np.clip(
        pool.neutral_pass_rate + pool.script_pass_sensitivity * score_pressure,
        0.34,
        0.78,
    )
    team_dropbacks = rng.binomial(team_plays, pass_rate).astype(np.int16)

    sack_probability = np.clip(pool.sack_rate * pass_disruption, 0.01, 0.22)
    team_sacks = rng.binomial(team_dropbacks, sack_probability).astype(np.int16)
    team_pass_attempts = (team_dropbacks - team_sacks).astype(np.int16)
    targetable_rate = np.clip(
        pool.targetable_dropback_rate * (1.0 - 0.10 * (pass_disruption - 1.0)),
        0.78,
        0.99,
    )
    team_targets = rng.binomial(team_pass_attempts, targetable_rate).astype(np.int16)
    team_rush_attempts = (team_plays - team_dropbacks).astype(np.int16)

    passing_tds = rng.binomial(
        team_touchdowns, np.clip(pool.pass_td_share, 0.20, 0.90)
    ).astype(np.int16)
    rushing_tds = (team_touchdowns - passing_tds).astype(np.int16)

    players = pool.players
    target_base = np.array([p.target_share for p in players], dtype=float)
    rush_base = np.array([p.rush_share for p in players], dtype=float)
    rec_td_base = np.array(
        [max(p.receiving_td_share, p.red_zone_target_share) for p in players], dtype=float
    )
    rush_td_base = np.array(
        [max(p.rushing_td_share, p.red_zone_rush_share) for p in players], dtype=float
    )

    target_shares = _sample_role_shares(rng, players, target_base, worlds)
    rush_shares = _sample_role_shares(rng, players, rush_base, worlds)
    rec_td_shares = _sample_role_shares(rng, players, rec_td_base, worlds)
    rush_td_shares = _sample_role_shares(rng, players, rush_td_base, worlds)

    targets = _allocate_integer_counts(rng, team_targets, target_shares)
    rushes = _allocate_integer_counts(rng, team_rush_attempts, rush_shares)
    rec_tds = _allocate_integer_counts(rng, passing_tds, rec_td_shares)
    rush_tds = _allocate_integer_counts(rng, rushing_tds, rush_td_shares)

    pass_yard_multiplier = np.clip(1.0 - 0.32 * (pass_disruption - 1.0), 0.90, 1.10)

    player_stats: dict[str, dict[str, np.ndarray]] = {}
    for idx, player in enumerate(players):
        stats = _empty_player_stats(worlds)
        catch_probability = np.clip(
            player.catch_rate
            * player.catchpoint_modifier
            * player.effectiveness_if_active
            * (1.0 - 0.12 * (pass_disruption - 1.0)),
            0.25,
            0.93,
        )
        receptions = rng.binomial(targets[:, idx], catch_probability).astype(np.int16)
        receiving_yards = _gamma_sum(
            rng,
            receptions,
            player.yards_per_reception * player.explosive_modifier,
        )
        receiving_yards = (receiving_yards * pass_yard_multiplier).astype(np.float32)
        rush_mean = (
            player.yards_per_carry
            * player.rushing_efficiency_modifier
            * player.effectiveness_if_active
        )
        rushing_yards = _gamma_sum(rng, rushes[:, idx], max(rush_mean, 0.5), 2.8)
        rushing_yards = (rushing_yards * run_efficiency).astype(np.float32)

        stats["targets"] = targets[:, idx].astype(np.float32)
        stats["receptions"] = receptions.astype(np.float32)
        stats["receiving_yards"] = receiving_yards
        stats["receiving_tds"] = rec_tds[:, idx].astype(np.float32)
        stats["rush_attempts"] = rushes[:, idx].astype(np.float32)
        stats["rushing_yards"] = rushing_yards
        stats["rushing_tds"] = rush_tds[:, idx].astype(np.float32)
        player_stats[player.player_id] = stats

    quarterback_indices = [idx for idx, player in enumerate(players) if player.position == "QB"]
    if quarterback_indices:
        qb_base = np.array([players[idx].qb_pass_share for idx in quarterback_indices], dtype=float)
        if qb_base.sum() <= 0:
            qb_base = np.zeros(len(quarterback_indices), dtype=float)
            qb_base[0] = 1.0
        qb_shares = _sample_role_shares(
            rng,
            tuple(players[idx] for idx in quarterback_indices),
            qb_base,
            worlds,
        )
        qb_pass_attempts = _allocate_integer_counts(rng, team_pass_attempts, qb_shares)
        total_receptions = np.sum(
            np.column_stack(
                [player_stats[player.player_id]["receptions"] for player in players]
            ),
            axis=1,
        ).astype(np.int16)
        qb_completions = _allocate_integer_counts(rng, total_receptions, qb_shares)
        qb_passing_tds = _allocate_integer_counts(rng, passing_tds, qb_shares)
        total_receiving_yards = np.sum(
            np.column_stack(
                [player_stats[player.player_id]["receiving_yards"] for player in players]
            ),
            axis=1,
        )
        for local_idx, player_idx in enumerate(quarterback_indices):
            player = players[player_idx]
            stats = player_stats[player.player_id]
            stats["pass_attempts"] = qb_pass_attempts[:, local_idx].astype(np.float32)
            stats["completions"] = qb_completions[:, local_idx].astype(np.float32)
            stats["passing_yards"] = (total_receiving_yards * qb_shares[:, local_idx]).astype(
                np.float32
            )
            stats["passing_tds"] = qb_passing_tds[:, local_idx].astype(np.float32)

    return TeamAllocationWorlds(
        player_stats=player_stats,
        team_plays=team_plays,
        team_dropbacks=team_dropbacks,
        team_sacks=team_sacks,
        team_pass_attempts=team_pass_attempts,
        team_targets=team_targets,
        team_rush_attempts=team_rush_attempts,
        passing_tds=passing_tds,
        rushing_tds=rushing_tds,
        pass_disruption=pass_disruption,
        run_efficiency=run_efficiency,
    )


def allocate_game_players(
    game_worlds: GameWorlds,
    away_pool: TeamPlayerPool,
    home_pool: TeamPlayerPool,
    seed: int,
) -> GameAllocationWorlds:
    """Allocate one simulated game's finite football supply to player stat worlds."""
    rng = np.random.default_rng(seed)
    worlds = len(game_worlds.away_points)
    away_pass_disruption = (
        game_worlds.away_pass_disruption
        if game_worlds.away_pass_disruption is not None
        else np.ones(worlds, dtype=np.float32)
    )
    home_pass_disruption = (
        game_worlds.home_pass_disruption
        if game_worlds.home_pass_disruption is not None
        else np.ones(worlds, dtype=np.float32)
    )
    away_run_efficiency = (
        game_worlds.away_run_efficiency
        if game_worlds.away_run_efficiency is not None
        else np.ones(worlds, dtype=np.float32)
    )
    home_run_efficiency = (
        game_worlds.home_run_efficiency
        if game_worlds.home_run_efficiency is not None
        else np.ones(worlds, dtype=np.float32)
    )

    away_plays, home_plays = _shared_game_play_budget(
        rng,
        away_pool,
        home_pool,
        game_worlds.away_drives,
        game_worlds.home_drives,
    )

    away = _allocate_team(
        rng,
        away_pool,
        game_worlds.away_points,
        game_worlds.home_points,
        away_plays,
        game_worlds.away_touchdowns,
        away_pass_disruption,
        away_run_efficiency,
    )
    home = _allocate_team(
        rng,
        home_pool,
        game_worlds.home_points,
        game_worlds.away_points,
        home_plays,
        game_worlds.home_touchdowns,
        home_pass_disruption,
        home_run_efficiency,
    )
    return GameAllocationWorlds(away=away, home=home)
