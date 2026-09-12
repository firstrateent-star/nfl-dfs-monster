from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import polars as pl

from run_week1_v13_box_scores import (
    _aggregate_player_rows,
    _defensive_plays,
    _name_map,
    _position_map,
    _quantiles,
    _representative_world,
    _team_map,
)
from run_week1_v13_first_sim import (
    GAME_DATE,
    MATCHUPS,
    _defensive_unit,
    _read,
    _situational_context,
    _team_identity,
    _with_event_rush_plan,
)
from monster.dfs.fanduel import FANDUEL_SCORING, score_offensive_player_worlds
from monster.feature_compile.health_pools import apply_health_to_skill_pools
from monster.feature_compile.league_units import compile_league_unit_player_map
from monster.feature_compile.reality_inputs import compile_player_reality_inputs
from monster.feature_compile.skill_pools import compile_current_skill_pools
from monster.sim.defensive_attribution import attribute_defensive_box_score
from monster.sim.event_ledger import assert_event_conservation, summarize_game
from monster.sim.game_loop_v13 import DefensiveBoxScore, PlayerBoxScore, simulate_game
from monster.sim.intent_ecology import build_intent_ecology
from monster.sim.rushing_roles import sample_event_rush_share_plan
from monster.snapshot.league import compile_team_state_map


def _progress(
    completed: int,
    total: int,
    *,
    game: str,
    world: int,
    worlds: int,
    started: float,
) -> None:
    elapsed = max(time.monotonic() - started, 1e-9)
    rate = completed / elapsed
    remaining = (total - completed) / rate if rate > 0 else 0.0
    pct = 100.0 * completed / total
    width = 20
    filled = int(width * completed / total)
    bar = "█" * filled + "░" * (width - filled)
    print(
        f"MONSTER [{bar}] {pct:5.1f}% | {completed:,}/{total:,} worlds | "
        f"{game} {world:,}/{worlds:,} | {rate:.2f} worlds/s | ETA {remaining/60:.1f}m",
        flush=True,
    )


def _attach_historical_intent_ecology(teams: dict, policy_dir: Path) -> dict:
    """Attach real historical intent/resolution priors without changing team identities.

    The policy build already separates *what an offense tries* from *whether it succeeds*.
    Here we activate that evidence in the integrated simulator: contextual pass depth and run
    geometry come from 2025 regular-season play-by-play, while the existing current-player,
    Madden, protection, coverage and game-state mechanisms retain bounded matchup authority.
    """
    stems = (
        "pass_depth_league",
        "pass_depth_team",
        "pass_depth_qb",
        "pass_depth_outcomes",
        "target_depth",
        "run_geometry_league",
        "run_geometry_team",
        "run_geometry_rusher",
        "run_geometry_outcomes",
    )
    frames = {}
    for stem in stems:
        path = policy_dir / f"{stem}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"historical intent ecology input missing: {path}")
        frames[stem] = pl.read_parquet(path).to_dicts()

    return {
        team_id: replace(
            team,
            intent_ecology=build_intent_ecology(
                team_id=team_id,
                pass_league_rows=frames["pass_depth_league"],
                pass_team_rows=frames["pass_depth_team"],
                pass_qb_rows=frames["pass_depth_qb"],
                pass_outcome_rows=frames["pass_depth_outcomes"],
                target_depth_rows=frames["target_depth"],
                run_league_rows=frames["run_geometry_league"],
                run_team_rows=frames["run_geometry_team"],
                run_rusher_rows=frames["run_geometry_rusher"],
                run_outcome_rows=frames["run_geometry_outcomes"],
            ),
        )
        for team_id, team in teams.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--personnel", type=Path, required=True)
    parser.add_argument("--player-usage", type=Path, required=True)
    parser.add_argument("--situation-context", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026190921)
    parser.add_argument("--first-out", type=Path, required=True)
    parser.add_argument("--box-out", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    policy = _read(args.policy)
    personnel = _read(args.personnel)
    usage = _read(args.player_usage)
    situation_context = _read(args.situation_context)
    league_neutral_pass_rate, situational_pass_rates = _situational_context(situation_context)
    pools = apply_health_to_skill_pools(
        compile_current_skill_pools(personnel, usage, policy=policy), personnel
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
    teams = _attach_historical_intent_ecology(teams, args.player_usage.parent)
    defenses = {team: _defensive_unit(units[team]) for pair in MATCHUPS for team in pair}
    names = _name_map(personnel, pools)
    positions = _position_map(personnel, pools)
    player_teams = _team_map(pools, units)
    pool_players = {
        (team_id, player.player_id): player
        for team_id, pool in pools.items()
        for player in pool.players
    }

    anatomy_acc = defaultdict(lambda: defaultdict(list))
    outcome_acc = defaultdict(lambda: {"away_wins": 0, "home_wins": 0, "ties": 0})
    rush_plan_acc = defaultdict(list)
    offensive_acc = defaultdict(lambda: defaultdict(list))
    defensive_acc = defaultdict(lambda: defaultdict(list))
    game_acc = defaultdict(lambda: defaultdict(list))
    game_world_rows: dict[str, list[dict]] = defaultdict(list)
    representative_offense: dict[tuple[str, int, str], dict] = {}
    representative_defense: dict[tuple[str, int, str], dict] = {}

    started = time.monotonic()
    total_worlds = len(MATCHUPS) * args.worlds
    completed = 0

    for game_idx, (away, home) in enumerate(MATCHUPS):
        game = f"{away}@{home}"
        offense_ids = {
            team: tuple(p.player_id for p in pools[team].players) for team in (away, home)
        }
        defense_ids = {
            team: tuple(p.player_id for p in units[team] if p.defense_snap_share >= 0.03)
            for team in (away, home)
        }
        print(f"Starting {game}: {args.worlds:,} worlds", flush=True)
        for world in range(args.worlds):
            seed = args.seed + game_idx * 1_000_003 + world
            away_plan = sample_event_rush_share_plan(
                pools[away], rng=np.random.default_rng(seed + 101_003)
            )
            home_plan = sample_event_rush_share_plan(
                pools[home], rng=np.random.default_rng(seed + 202_007)
            )
            for team_id, plan in ((away, away_plan), (home, home_plan)):
                for player in pools[team_id].players:
                    rush_plan_acc[(game, team_id, player.player_id)].append(
                        float(plan.get(player.player_id, 0.0))
                    )

            result = simulate_game(
                _with_event_rush_plan(teams[away], away_plan),
                _with_event_rush_plan(teams[home], home_plan),
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

            team_box = {}
            for team, opponent, attribution_offset in (
                (away, home, 31_337),
                (home, away, 47_771),
            ):
                boxes = {
                    pid: result.player_stats.get(pid, PlayerBoxScore())
                    for pid in offense_ids[team]
                }
                pass_yards = sum(x.passing_yards for x in boxes.values())
                rush_yards = sum(x.rushing_yards for x in boxes.values())
                team_box[team] = {
                    "pass_yards": float(pass_yards),
                    "rush_yards": float(rush_yards),
                    "total_yards": float(pass_yards + rush_yards),
                    "turnovers": float(
                        sum(x.interceptions + x.fumbles_lost for x in boxes.values())
                    ),
                }
                for player_id, box in boxes.items():
                    data = asdict(box)
                    for stat, value in data.items():
                        offensive_acc[(game, player_id)][stat].append(float(value))
                    representative_offense[(game, world, player_id)] = data

                attributed = attribute_defensive_box_score(
                    _defensive_plays(result.plays, opponent, player_teams),
                    defenses[team],
                    seed=seed + attribution_offset,
                )
                for player_id in defense_ids[team]:
                    box = attributed.get(player_id, DefensiveBoxScore())
                    data = asdict(box)
                    for stat, value in data.items():
                        defensive_acc[(game, player_id)][stat].append(float(value))
                    representative_defense[(game, world, player_id)] = data

            away_points = float(result.final_state.away_score)
            home_points = float(result.final_state.home_score)
            for team, points, opponent_points in (
                (away, away_points, home_points),
                (home, home_points, away_points),
            ):
                values = {
                    "points": points,
                    "points_allowed": opponent_points,
                    **team_box[team],
                }
                for stat, value in values.items():
                    game_acc[(game, team)][stat].append(float(value))
            game_world_rows[game].append(
                {
                    "world": world,
                    "away_points": away_points,
                    "home_points": home_points,
                    "total_yards": team_box[away]["total_yards"]
                    + team_box[home]["total_yards"],
                }
            )

            completed += 1
            if (
                completed == total_worlds
                or (world + 1) == args.worlds
                or completed % max(args.progress_every, 1) == 0
            ):
                _progress(
                    completed,
                    total_worlds,
                    game=game,
                    world=world + 1,
                    worlds=args.worlds,
                    started=started,
                )

    game_rows = []
    for game, metrics in anatomy_acc.items():
        away, home = game.split("@")
        row = {"game": game, "worlds": args.worlds}
        for key, values in metrics.items():
            arr = np.asarray(values, dtype=float)
            row[f"{key}_mean"] = float(arr.mean())
            row[f"{key}_sd"] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
            if key in {"away_points", "home_points", "snaps", "scrimmage_plays", "drives"}:
                row[f"{key}_p10"] = float(np.quantile(arr, 0.10))
                row[f"{key}_p50"] = float(np.quantile(arr, 0.50))
                row[f"{key}_p90"] = float(np.quantile(arr, 0.90))
        row["completion_percentage"] = row["completions_mean"] / row["pass_attempts_mean"]
        row["sack_rate"] = row["sacks_mean"] / row["dropbacks_mean"]
        row["scramble_rate"] = row["scrambles_mean"] / row["dropbacks_mean"]
        row["interception_rate"] = row["interceptions_mean"] / row["pass_attempts_mean"]
        outcomes = outcome_acc[game]
        row.update(
            {
                "away": away,
                "home": home,
                "total_mean": row["away_points_mean"] + row["home_points_mean"],
                "margin_mean": row["away_points_mean"] - row["home_points_mean"],
                "away_win_probability": outcomes["away_wins"] / args.worlds,
                "home_win_probability": outcomes["home_wins"] / args.worlds,
                "tie_probability": outcomes["ties"] / args.worlds,
                "overtime_probability": row["went_to_overtime_mean"],
            }
        )
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

    first_player_rows = []
    player_world_rows = []
    for (game, player_id), metrics in offensive_acc.items():
        arrays = {key: np.asarray(values, dtype=float) for key, values in metrics.items()}
        fd_scores = score_offensive_player_worlds(
            {stat: arrays[stat] for stat in FANDUEL_SCORING}
        )
        row = {
            "game": game,
            "player_id": player_id,
            "player": names.get(player_id, player_id),
            "position": positions.get(player_id, ""),
        }
        for key, arr in arrays.items():
            row[f"{key}_mean"] = float(arr.mean())
            row[f"{key}_p90"] = float(np.quantile(arr, 0.90))
        row.update(
            {
                "fanduel_mean": float(fd_scores.mean()),
                "fanduel_p50": float(np.quantile(fd_scores, 0.50)),
                "fanduel_p75": float(np.quantile(fd_scores, 0.75)),
                "fanduel_p90": float(np.quantile(fd_scores, 0.90)),
                "fanduel_p95": float(np.quantile(fd_scores, 0.95)),
                "fanduel_p99": float(np.quantile(fd_scores, 0.99)),
                "fanduel_15_plus_probability": float(np.mean(fd_scores >= 15.0)),
                "fanduel_20_plus_probability": float(np.mean(fd_scores >= 20.0)),
                "fanduel_25_plus_probability": float(np.mean(fd_scores >= 25.0)),
                "fanduel_30_plus_probability": float(np.mean(fd_scores >= 30.0)),
            }
        )
        first_player_rows.append(row)
        for world, fd_points in enumerate(fd_scores):
            world_row = {
                "game": game,
                "world": world,
                "player_id": player_id,
                "player": names.get(player_id, player_id),
                "position": positions.get(player_id, ""),
                "fanduel_points": float(fd_points),
            }
            for key, arr in arrays.items():
                world_row[key] = float(arr[world])
            player_world_rows.append(world_row)

    rush_plan_rows = []
    for (game, team_id, player_id), values in rush_plan_acc.items():
        player = pool_players[(team_id, player_id)]
        arr = np.asarray(values, dtype=float)
        rush_plan_rows.append(
            {
                "game": game,
                "team": team_id,
                "player_id": player_id,
                "player": player.display_name,
                "position": player.position,
                "base_rush_share": float(player.rush_share),
                "rush_role_probability": float(player.rush_role_probability),
                "active_probability": float(player.active_probability),
                "role_uncertainty": float(player.role_uncertainty),
                "plan_share_mean": float(arr.mean()),
                "plan_share_p50": float(np.quantile(arr, 0.50)),
                "plan_share_p90": float(np.quantile(arr, 0.90)),
                "plan_participation_probability": float(np.mean(arr > 0.0)),
            }
        )

    args.first_out.mkdir(parents=True, exist_ok=True)
    game_df = pl.DataFrame(game_rows).sort("total_mean", descending=True)
    player_df = pl.DataFrame(first_player_rows).sort("fanduel_mean", descending=True)
    world_df = pl.DataFrame(player_world_rows).sort(["game", "world", "player_id"])
    rush_plan_df = pl.DataFrame(rush_plan_rows).sort(
        ["team", "plan_share_mean"], descending=[False, True]
    )
    game_df.write_csv(args.first_out / "game_distributions.csv")
    player_df.write_csv(args.first_out / "player_distributions.csv")
    world_df.write_csv(args.first_out / "player_world_fanduel.csv")
    rush_plan_df.write_csv(args.first_out / "rushing_role_plan_audit.csv")
    game_df.select(
        [
            "game",
            "projected_winner",
            "winner_probability",
            "projected_score",
            "projected_away_score",
            "projected_home_score",
            "away_win_probability",
            "home_win_probability",
            "tie_probability",
            "overtime_probability",
            "total_mean",
            "margin_mean",
            "away_points_p10",
            "away_points_p50",
            "away_points_p90",
            "home_points_p10",
            "home_points_p50",
            "home_points_p90",
        ]
    ).write_csv(args.first_out / "projected_scores_and_outcomes.csv")
    game_df.select(
        [
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
            "went_to_overtime_mean",
            "overtime_touchdowns_without_try_mean",
            "total_mean",
        ]
    ).write_csv(args.first_out / "football_anatomy.csv")
    situation_context.write_csv(args.first_out / "situational_pass_context.csv")
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
    ).write_csv(args.first_out / "team_play_call_inputs.csv")
    first_manifest = {
        "model": "Monster v1.3 Full-Reality integrated event-by-event simulation",
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
        "historical_intent_ecology_active": True,
        "historical_intent_ecology_scope": "2025 regular-season NFL play-by-play",
        "pass_depth_contextual_by_team_and_qb": True,
        "target_depth_compatibility_active": True,
        "depth_specific_air_yards_and_yac_active": True,
        "run_geometry_ecology_active": True,
        "defensive_identity_active": True,
        "full_reality_player_bridge_active": True,
        "unit_bridge_active": True,
        "stable_event_rushing_roles_active": True,
        "rushing_role_plan_audit_active": True,
        "complete_game_simulation_active": True,
        "regular_season_overtime_active": True,
        "overtime_rule_version": "2026_rule_16_regular_season_10min_both_possessions",
        "game_distribution_standard_deviations_active": True,
        "zero_inclusive_player_worlds": True,
        "fanduel_scoring_downstream_only": True,
        "box_scores_materialized_from_same_worlds": True,
        "duplicate_box_score_resimulation": False,
        "progress_telemetry_active": True,
        "promotion_status": "SHADOW_FIRST_SIMULATION_NOT_PROMOTED",
    }
    (args.first_out / "manifest.json").write_text(json.dumps(first_manifest, indent=2))

    offense_rows = _aggregate_player_rows(
        offensive_acc, names, positions, player_teams, side="offense"
    )
    defense_rows = _aggregate_player_rows(
        defensive_acc, names, positions, player_teams, side="defense"
    )
    team_rows = []
    for (game, team), metrics in game_acc.items():
        row = {"game": game, "team": team}
        for stat, values in metrics.items():
            for label, value in _quantiles(values).items():
                row[f"{stat}_{label}"] = value
        team_rows.append(row)

    representative_rows = []
    for game, worlds in game_world_rows.items():
        representative_world = _representative_world(worlds)
        away, home = game.split("@")
        world_summary = next(x for x in worlds if x["world"] == representative_world)
        representative_rows.append(
            {
                "game": game,
                "world": representative_world,
                "record_type": "game",
                "team": "",
                "player_id": "",
                "player": "",
                "position": "",
                "away_points": world_summary["away_points"],
                "home_points": world_summary["home_points"],
            }
        )
        for team in (away, home):
            for player_id in [p.player_id for p in pools[team].players]:
                representative_rows.append(
                    {
                        "game": game,
                        "world": representative_world,
                        "record_type": "offense",
                        "team": team,
                        "player_id": player_id,
                        "player": names.get(player_id, player_id),
                        "position": positions.get(player_id, ""),
                        **representative_offense[(game, representative_world, player_id)],
                    }
                )
            for player_id in [p.player_id for p in units[team] if p.defense_snap_share >= 0.03]:
                representative_rows.append(
                    {
                        "game": game,
                        "world": representative_world,
                        "record_type": "defense",
                        "team": team,
                        "player_id": player_id,
                        "player": names.get(player_id, player_id),
                        "position": positions.get(player_id, ""),
                        **representative_defense[(game, representative_world, player_id)],
                    }
                )

    args.box_out.mkdir(parents=True, exist_ok=True)
    offense_df = pl.DataFrame(offense_rows).sort(["game", "team", "position", "player"])
    defense_df = pl.DataFrame(defense_rows).sort(
        ["game", "team", "tackles_mean"], descending=[False, False, True]
    )
    team_df = pl.DataFrame(team_rows).sort(["game", "team"])
    representative_df = pl.DataFrame(representative_rows).sort(
        ["game", "record_type", "team", "player"]
    )
    offense_df.write_csv(args.box_out / "offensive_player_box_score_distributions.csv")
    defense_df.write_csv(args.box_out / "defensive_player_box_score_distributions.csv")
    team_df.write_csv(args.box_out / "team_box_score_distributions.csv")
    representative_df.write_csv(args.box_out / "representative_world_box_scores.csv")
    game_summaries = []
    for game in sorted(game_world_rows):
        away, home = game.split("@")
        game_summaries.append(
            {
                "game": game,
                "representative_world": _representative_world(game_world_rows[game]),
                "away": away,
                "home": home,
                "teams": [row for row in team_rows if row["game"] == game],
                "offense": [row for row in offense_rows if row["game"] == game],
                "defense": [row for row in defense_rows if row["game"] == game],
            }
        )
    (args.box_out / "simulated_game_box_scores.json").write_text(
        json.dumps(game_summaries, indent=2)
    )
    box_manifest = {
        "model": "Monster v1.3 same-world box-score observability layer",
        "season": 2026,
        "week": 1,
        "worlds_per_game": args.worlds,
        "seed": args.seed,
        "games": len(MATCHUPS),
        "market_blind_football": True,
        "dfs_inputs_used": False,
        "sportsbook_inputs_used": False,
        "offense_event_derived": True,
        "defense_event_derived": True,
        "role_aware_defensive_attribution": True,
        "zero_inclusive_worlds": True,
        "distribution_view": True,
        "representative_world_view": True,
        "materialized_from_primary_simulation_worlds": True,
        "resimulation_required": False,
    }
    (args.box_out / "manifest.json").write_text(json.dumps(box_manifest, indent=2))
    print(json.dumps({"first_sim": first_manifest, "box_scores": box_manifest}, indent=2))


if __name__ == "__main__":
    main()
