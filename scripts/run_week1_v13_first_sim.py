from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from monster.feature_compile.health_pools import apply_health_to_skill_pools
from monster.feature_compile.league_units import compile_league_unit_player_map
from monster.feature_compile.reality_inputs import compile_player_reality_inputs
from monster.feature_compile.skill_pools import compile_current_skill_pools
from monster.feature_compile.units import compile_team_unit_effects
from monster.feature_compile.v13_identity_bridge import compile_v13_player_identity
from monster.sim.event_ledger import assert_event_conservation, summarize_game
from monster.sim.game_loop_v13 import simulate_regulation_game
from monster.sim.matchup_kernel import DefensiveIdentity, DefensiveUnit
from monster.sim.play_kernel import PlayerIdentity, TeamIdentity
from monster.snapshot.league import compile_team_state_map

MATCHUPS = (
    ("CHI", "CAR"),
    ("BUF", "HOU"),
    ("NO", "DET"),
    ("CLE", "JAC"),
    ("TB", "CIN"),
    ("ATL", "PIT"),
    ("NYJ", "TEN"),
    ("BAL", "IND"),
    ("ARI", "LAC"),
    ("WAS", "PHI"),
    ("MIA", "LV"),
    ("GB", "MIN"),
)
GAME_DATE = date(2026, 9, 13)
_DISTANCE_BUCKETS = ("short", "medium", "long")


def _read(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path) if path.suffix == ".parquet" else pl.read_csv(path)


def _rating(value: float | None, center: float = 78.0, scale: float = 12.0) -> float:
    if value is None:
        return 1.0
    return float(
        np.clip(
            1.0 + 0.22 * np.tanh((float(value) - center) / scale),
            0.75,
            1.25,
        )
    )


def _situational_context(context: pl.DataFrame) -> tuple[float, tuple[float, ...]]:
    rows = {
        (int(row["down"]), str(row["distance_bucket"])): float(row["pass_rate"])
        for row in context.to_dicts()
    }
    expected = {(down, bucket) for down in range(1, 5) for bucket in _DISTANCE_BUCKETS}
    missing = expected.difference(rows)
    if missing:
        raise ValueError(f"situational pass context missing cells: {sorted(missing)}")
    league_values = context.get_column("league_neutral_pass_rate").drop_nulls().to_list()
    if not league_values:
        raise ValueError("situational pass context missing league neutral pass rate")
    rates = tuple(rows[(down, bucket)] for down in range(1, 5) for bucket in _DISTANCE_BUCKETS)
    return float(league_values[0]), rates


def _defensive_unit(players) -> DefensiveUnit:
    front = []
    coverage = []
    for player in players:
        if player.defense_snap_share < 0.03:
            continue
        position = player.position.upper()
        snap_weight = float(
            np.clip(
                player.defense_snap_share
                * player.active_probability
                * player.effectiveness_if_active,
                0.001,
                1.10,
            )
        )
        item = DefensiveIdentity(
            player_id=player.player_id,
            name=player.player_id,
            position=position,
            coverage=_rating(player.madden_coverage),
            pass_rush=_rating(player.madden_pass_rush),
            run_defense=_rating(player.madden_tackle),
            tackling=_rating(player.madden_tackle),
            ball_hawk=_rating(player.madden_coverage),
            snap_weight=snap_weight,
        )
        if position in {"DE", "DT", "NT", "DL", "EDGE", "LB", "ILB", "OLB", "MLB"}:
            front.append(item)
        if position in {"CB", "DB", "S", "FS", "SS", "LB", "ILB", "OLB", "MLB"}:
            coverage.append(item)
    return DefensiveUnit(front=tuple(front), coverage=tuple(coverage))


def _team_identity(
    team_id: str,
    pool,
    reality,
    unit_players,
    state,
    *,
    league_neutral_pass_rate: float,
    situational_pass_rates: tuple[float, ...],
) -> TeamIdentity:
    compiled: dict[str, PlayerIdentity] = {}
    for player in pool.players:
        usage = (
            player.qb_pass_share
            if player.position == "QB"
            else max(player.target_share, player.rush_share, 0.001)
        )
        inputs = reality.get(player.player_id)
        if inputs is None:
            compiled[player.player_id] = PlayerIdentity(
                player.player_id,
                player.display_name,
                player.position,
                usage_weight=usage,
            )
        else:
            compiled[player.player_id], _ = compile_v13_player_identity(
                player_id=player.player_id,
                name=player.display_name,
                position=player.position,
                usage_weight=usage,
                inputs=inputs,
            )

    qbs = [player for player in pool.players if player.position == "QB"]
    if not qbs:
        raise ValueError(f"{team_id} has no quarterback in current pool")
    qb_state = max(qbs, key=lambda player: player.qb_pass_share)
    qb = compiled[qb_state.player_id]

    rushers = tuple(
        PlayerIdentity(
            compiled[player.player_id].player_id,
            compiled[player.player_id].name,
            compiled[player.player_id].position,
            usage_weight=max(player.rush_share, 0.001),
            efficiency=compiled[player.player_id].efficiency,
            explosive=compiled[player.player_id].explosive,
            turnover_security=compiled[player.player_id].turnover_security,
        )
        for player in pool.players
        if player.rush_share > 0.001
    )
    receivers = tuple(
        PlayerIdentity(
            compiled[player.player_id].player_id,
            compiled[player.player_id].name,
            compiled[player.player_id].position,
            usage_weight=max(player.target_share, 0.001),
            efficiency=compiled[player.player_id].efficiency,
            explosive=compiled[player.player_id].explosive,
            turnover_security=compiled[player.player_id].turnover_security,
        )
        for player in pool.players
        if player.position in {"RB", "WR", "TE"} and player.target_share > 0.001
    )

    effects, _ = compile_team_unit_effects(unit_players)
    pass_efficiency = float(
        np.clip(
            1.0
            + 0.55 * state.offensive_epa_per_play
            + state.physical_madden_effect
            + state.injury_effect
            + 0.35 * state.weather_effect,
            0.72,
            1.28,
        )
    )
    rush_efficiency = float(
        np.clip(
            1.0
            + 0.35 * state.offense_strength
            + state.injury_effect
            + 0.20 * state.weather_effect,
            0.75,
            1.25,
        )
    )
    return TeamIdentity(
        team_id=team_id,
        quarterback=qb,
        rushers=rushers or (qb,),
        receivers=receivers,
        neutral_pass_rate=pool.neutral_pass_rate,
        pass_efficiency=pass_efficiency,
        rush_efficiency=rush_efficiency,
        pass_protection=float(np.clip(1.0 + effects.pass_protection_effect, 0.90, 1.10)),
        run_blocking=float(np.clip(1.0 + effects.run_block_effect, 0.90, 1.10)),
        field_goal_skill=float(np.clip(1.0 + effects.special_teams_effect, 0.92, 1.08)),
        punt_skill=float(np.clip(1.0 + effects.special_teams_effect, 0.92, 1.08)),
        league_neutral_pass_rate=league_neutral_pass_rate,
        situational_pass_rates=situational_pass_rates,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--personnel", type=Path, required=True)
    parser.add_argument("--player-usage", type=Path, required=True)
    parser.add_argument("--situation-context", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=250)
    parser.add_argument("--seed", type=int, default=2026090913)
    parser.add_argument("--out", type=Path, default=Path("artifacts/week1-v13-first-sim"))
    args = parser.parse_args()

    policy = _read(args.policy)
    personnel = _read(args.personnel)
    usage = _read(args.player_usage)
    situation_context = _read(args.situation_context)
    league_neutral_pass_rate, situational_pass_rates = _situational_context(situation_context)

    pools = apply_health_to_skill_pools(
        compile_current_skill_pools(personnel, usage, policy=policy),
        personnel,
    )
    reality = compile_player_reality_inputs(personnel, game_date=GAME_DATE)
    units = compile_league_unit_player_map(personnel)
    states = {}
    for away, home in MATCHUPS:
        states.update(compile_team_state_map(policy, {away: home, home: away}))

    teams = {
        team: _team_identity(
            team,
            pools[team],
            reality,
            units[team],
            states[team],
            league_neutral_pass_rate=league_neutral_pass_rate,
            situational_pass_rates=situational_pass_rates,
        )
        for pair in MATCHUPS
        for team in pair
    }
    defenses = {
        team: _defensive_unit(units[team])
        for pair in MATCHUPS
        for team in pair
    }

    game_rows = []
    player_acc = defaultdict(lambda: defaultdict(list))
    anatomy_acc = defaultdict(lambda: defaultdict(list))
    outcome_acc = defaultdict(lambda: {"away_wins": 0, "home_wins": 0, "ties": 0})

    for game_idx, (away, home) in enumerate(MATCHUPS):
        game = f"{away}@{home}"
        for world in range(args.worlds):
            seed = args.seed + game_idx * 1_000_003 + world
            result = simulate_regulation_game(
                teams[away],
                teams[home],
                away_defense=defenses[away],
                home_defense=defenses[home],
                seed=seed,
            )
            assert_event_conservation(result)
            summary = summarize_game(result)
            for key, value in asdict(summary).items():
                anatomy_acc[game][key].append(value)
            if summary.away_points > summary.home_points:
                outcome_acc[game]["away_wins"] += 1
            elif summary.home_points > summary.away_points:
                outcome_acc[game]["home_wins"] += 1
            else:
                outcome_acc[game]["ties"] += 1
            for player_id, box in result.player_stats.items():
                for key, value in asdict(box).items():
                    player_acc[(game, player_id)][key].append(value)

    for game, metrics in anatomy_acc.items():
        away, home = game.split("@")
        row = {"game": game, "worlds": args.worlds}
        for key, values in metrics.items():
            arr = np.asarray(values, dtype=float)
            row[f"{key}_mean"] = float(arr.mean())
            if key in {"away_points", "home_points", "snaps", "scrimmage_plays", "drives"}:
                row[f"{key}_p10"] = float(np.quantile(arr, 0.10))
                row[f"{key}_p50"] = float(np.quantile(arr, 0.50))
                row[f"{key}_p90"] = float(np.quantile(arr, 0.90))

        row["completion_percentage"] = row["completions_mean"] / row["pass_attempts_mean"]
        row["sack_rate"] = row["sacks_mean"] / row["dropbacks_mean"]
        row["scramble_rate"] = row["scrambles_mean"] / row["dropbacks_mean"]
        row["interception_rate"] = row["interceptions_mean"] / row["pass_attempts_mean"]
        outcomes = outcome_acc[game]
        row["away"] = away
        row["home"] = home
        row["total_mean"] = row["away_points_mean"] + row["home_points_mean"]
        row["margin_mean"] = row["away_points_mean"] - row["home_points_mean"]
        row["away_win_probability"] = outcomes["away_wins"] / args.worlds
        row["home_win_probability"] = outcomes["home_wins"] / args.worlds
        row["tie_probability"] = outcomes["ties"] / args.worlds
        row["projected_winner"] = (
            away if row["away_win_probability"] > row["home_win_probability"] else home
        )
        row["projected_away_score"] = round(row["away_points_mean"])
        row["projected_home_score"] = round(row["home_points_mean"])
        row["projected_score"] = (
            f'{away} {row["projected_away_score"]} - {home} {row["projected_home_score"]}'
        )
        row["winner_probability"] = max(
            row["away_win_probability"], row["home_win_probability"]
        )
        game_rows.append(row)

    names = {player.player_id: player.display_name for pool in pools.values() for player in pool.players}
    positions = {player.player_id: player.position for pool in pools.values() for player in pool.players}
    player_rows = []
    for (game, player_id), metrics in player_acc.items():
        row = {
            "game": game,
            "player_id": player_id,
            "player": names.get(player_id, player_id),
            "position": positions.get(player_id, ""),
        }
        for key, values in metrics.items():
            arr = np.asarray(values, dtype=float)
            row[f"{key}_mean"] = float(arr.mean())
            row[f"{key}_p90"] = float(np.quantile(arr, 0.90))
        player_rows.append(row)

    args.out.mkdir(parents=True, exist_ok=True)
    game_df = pl.DataFrame(game_rows).sort("total_mean", descending=True)
    game_df.write_csv(args.out / "game_distributions.csv")
    pl.DataFrame(player_rows).write_csv(args.out / "player_distributions.csv")

    projection_cols = [
        "game",
        "projected_winner",
        "winner_probability",
        "projected_score",
        "projected_away_score",
        "projected_home_score",
        "away_win_probability",
        "home_win_probability",
        "tie_probability",
        "total_mean",
        "margin_mean",
        "away_points_p10",
        "away_points_p50",
        "away_points_p90",
        "home_points_p10",
        "home_points_p50",
        "home_points_p90",
    ]
    game_df.select(projection_cols).write_csv(args.out / "projected_scores_and_outcomes.csv")

    anatomy_cols = [
        "game",
        "scrimmage_plays_mean",
        "pass_plays_mean",
        "pass_attempts_mean",
        "dropbacks_mean",
        "run_plays_mean",
        "rush_attempts_mean",
        "scrambles_mean",
        "sacks_mean",
        "completions_mean",
        "completion_percentage",
        "interceptions_mean",
        "interception_rate",
        "fumbles_lost_mean",
        "sack_rate",
        "scramble_rate",
        "punts_mean",
        "field_goal_attempts_mean",
        "field_goals_made_mean",
        "touchdowns_mean",
        "drives_mean",
        "total_mean",
    ]
    game_df.select(anatomy_cols).write_csv(args.out / "football_anatomy.csv")
    situation_context.write_csv(args.out / "situational_pass_context.csv")
    pl.DataFrame(
        [
            {
                "team": team,
                "team_neutral_pass_rate": teams[team].neutral_pass_rate,
                "league_neutral_pass_rate": league_neutral_pass_rate,
                "team_neutral_deviation": teams[team].neutral_pass_rate
                - league_neutral_pass_rate,
            }
            for team in sorted(teams)
        ]
    ).write_csv(args.out / "team_play_call_inputs.csv")

    manifest = {
        "model": "Monster v1.3 Full-Reality event-by-event shadow",
        "week": 1,
        "season": 2026,
        "worlds_per_game": args.worlds,
        "seed": args.seed,
        "games": len(MATCHUPS),
        "market_blind_football": True,
        "scoreboard_event_derived": True,
        "player_stats_event_derived": True,
        "definition_safe_anatomy": True,
        "empirical_situational_pass_context_active": True,
        "defensive_identity_active": True,
        "full_reality_player_bridge_active": True,
        "unit_bridge_active": True,
        "projection_summary": "projected_scores_and_outcomes.csv",
        "football_anatomy": "football_anatomy.csv",
        "situational_pass_context": "situational_pass_context.csv",
        "team_play_call_inputs": "team_play_call_inputs.csv",
        "promotion_status": "SHADOW_FIRST_SIMULATION_NOT_PROMOTED",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(game_df.select(["game", "projected_winner", "winner_probability", "projected_score"]))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
