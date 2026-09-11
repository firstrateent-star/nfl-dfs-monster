from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import polars as pl

from run_week1_v13_box_scores import (
    _name_map,
    _position_map,
    _quantiles,
    _read,
    _representative_world,
    _situational_context,
    _team_identity,
    _team_map,
    _with_event_rush_plan,
)
from run_week1_v13_first_sim import GAME_DATE, MATCHUPS, _defensive_unit

from monster.feature_compile.health_pools import apply_health_to_skill_pools
from monster.feature_compile.league_units import compile_league_unit_player_map
from monster.feature_compile.reality_inputs import compile_player_reality_inputs
from monster.feature_compile.skill_pools import compile_current_skill_pools
from monster.sim.defensive_attribution import attribute_defensive_box_score
from monster.sim.event_ledger import assert_event_conservation
from monster.sim.game_loop_v13 import PlayerBoxScore, simulate_game
from monster.sim.play_kernel import PlayEvent, PlayType
from monster.sim.rushing_roles import sample_event_rush_share_plan
from monster.snapshot.league import compile_team_state_map


def _event_offense_team(event: PlayEvent, player_teams: dict[str, str]) -> str | None:
    for player_id in (event.passer_id, event.rusher_id, event.target_id, event.fumbler_id):
        if player_id is not None and player_id in player_teams:
            return player_teams[player_id]
    return None


def _offensive_scrimmage_plays(
    plays: tuple[PlayEvent, ...], team: str, player_teams: dict[str, str]
) -> int:
    return sum(
        event.play_type in {PlayType.RUN, PlayType.PASS}
        and _event_offense_team(event, player_teams) == team
        for event in plays
    )


def _defensive_plays(
    plays: tuple[PlayEvent, ...], opponent: str, player_teams: dict[str, str]
) -> tuple[PlayEvent, ...]:
    return tuple(
        event
        for event in plays
        if event.play_type in {PlayType.RUN, PlayType.PASS}
        and _event_offense_team(event, player_teams) == opponent
    )


def _participant_row(
    *,
    game: str,
    team: str,
    world: int,
    player_id: str,
    player: str,
    position: str,
    offense_box: PlayerBoxScore,
    defense_box,
    offense_snaps: int,
    defense_snaps: int,
    special_teams_snaps_estimated: int,
) -> dict:
    row = {
        "game": game,
        "team": team,
        "world": world,
        "player_id": player_id,
        "player": player,
        "position": position,
        "offense_snaps": offense_snaps,
        "defense_snaps": defense_snaps,
        "special_teams_snaps_estimated": special_teams_snaps_estimated,
    }
    offense_data = asdict(offense_box)
    row["passing_interceptions"] = offense_data.pop("interceptions")
    row.update(offense_data)
    if defense_box is not None:
        defense_data = asdict(defense_box)
        defense_data.pop("defensive_snaps", None)
        row["defensive_interceptions"] = defense_data.pop("interceptions")
        row.update(defense_data)
    else:
        row.update(
            {
                "pressures": 0,
                "sacks": 0,
                "defensive_interceptions": 0,
                "tackles": 0,
                "stuffs": 0,
                "forced_fumbles": 0,
            }
        )
    total_snaps = offense_snaps + defense_snaps + special_teams_snaps_estimated
    row["simulated_participant"] = total_snaps > 0 or any(
        float(row.get(stat, 0)) > 0
        for stat in (
            "pass_attempts",
            "targets",
            "rush_attempts",
            "tackles",
            "pressures",
            "sacks",
            "defensive_interceptions",
        )
    )
    return row


def _aggregate_complete(rows: list[dict]) -> list[dict]:
    identity = ("game", "team", "player_id", "player", "position")
    metrics = [
        "offense_snaps",
        "defense_snaps",
        "special_teams_snaps_estimated",
        "pass_attempts",
        "completions",
        "passing_yards",
        "passing_tds",
        "passing_interceptions",
        "targets",
        "receptions",
        "receiving_yards",
        "receiving_tds",
        "rush_attempts",
        "rushing_yards",
        "rushing_tds",
        "fumbles_lost",
        "pressures",
        "sacks",
        "defensive_interceptions",
        "tackles",
        "stuffs",
        "forced_fumbles",
    ]
    acc: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    participant: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in identity)
        for metric in metrics:
            acc[key][metric].append(float(row.get(metric, 0.0)))
        participant[key].append(float(bool(row["simulated_participant"])))

    out: list[dict] = []
    for key, values in acc.items():
        row = dict(zip(identity, key, strict=True))
        row["participation_probability"] = float(np.mean(participant[key]))
        for metric, series in values.items():
            for label, value in _quantiles(series).items():
                row[f"{metric}_{label}"] = value
            arr = np.asarray(series, dtype=float)
            row[f"{metric}_1_plus_probability"] = float(np.mean(arr >= 1.0))
        out.append(row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--personnel", type=Path, required=True)
    parser.add_argument("--player-usage", type=Path, required=True)
    parser.add_argument("--situation-context", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=250)
    parser.add_argument("--seed", type=int, default=2026110912)
    parser.add_argument(
        "--out", type=Path, default=Path("artifacts/week1-v13-complete-box-scores")
    )
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
    defenses = {team: _defensive_unit(units[team]) for pair in MATCHUPS for team in pair}
    names = _name_map(personnel, pools)
    positions = _position_map(personnel, pools)
    player_teams = _team_map(pools, units)

    world_rows: list[dict] = []
    game_world_rows: dict[str, list[dict]] = defaultdict(list)

    for game_idx, (away, home) in enumerate(MATCHUPS):
        game = f"{away}@{home}"
        for world in range(args.worlds):
            seed = args.seed + game_idx * 1_000_003 + world
            away_plan = sample_event_rush_share_plan(
                pools[away], rng=np.random.default_rng(seed + 101_003)
            )
            home_plan = sample_event_rush_share_plan(
                pools[home], rng=np.random.default_rng(seed + 202_007)
            )
            result = simulate_game(
                _with_event_rush_plan(teams[away], away_plan),
                _with_event_rush_plan(teams[home], home_plan),
                away_defense=defenses[away],
                home_defense=defenses[home],
                seed=seed,
            )
            assert_event_conservation(result)

            game_world_rows[game].append(
                {
                    "world": world,
                    "away_points": float(result.final_state.away_score),
                    "home_points": float(result.final_state.home_score),
                    "total_yards": float(
                        sum(
                            box.passing_yards + box.rushing_yards
                            for box in result.player_stats.values()
                        )
                    ),
                }
            )

            for team, opponent in ((away, home), (home, away)):
                team_offense_plays = _offensive_scrimmage_plays(result.plays, team, player_teams)
                opponent_plays = _defensive_plays(result.plays, opponent, player_teams)
                attributed_defense = attribute_defensive_box_score(
                    opponent_plays,
                    defenses[team],
                    seed=seed + (31_337 if team == away else 47_771),
                )
                special_events_estimate = max(len(result.special_teams_events) // 2, 0)

                for unit_player in units[team]:
                    player_id = unit_player.player_id
                    offense_box = result.player_stats.get(player_id, PlayerBoxScore())
                    defense_box = attributed_defense.get(player_id)
                    offense_snaps = int(
                        np.clip(
                            round(team_offense_plays * max(unit_player.offense_snap_share, 0.0)),
                            0,
                            team_offense_plays,
                        )
                    )
                    defense_snaps = 0 if defense_box is None else defense_box.defensive_snaps
                    special_snaps = int(
                        max(
                            round(
                                special_events_estimate
                                * max(unit_player.special_teams_snap_share, 0.0)
                            ),
                            0,
                        )
                    )
                    world_rows.append(
                        _participant_row(
                            game=game,
                            team=team,
                            world=world,
                            player_id=player_id,
                            player=names.get(player_id, player_id),
                            position=positions.get(player_id, unit_player.position),
                            offense_box=offense_box,
                            defense_box=defense_box,
                            offense_snaps=offense_snaps,
                            defense_snaps=defense_snaps,
                            special_teams_snaps_estimated=special_snaps,
                        )
                    )

    complete_rows = _aggregate_complete(world_rows)
    representative_rows: list[dict] = []
    for game, worlds in game_world_rows.items():
        representative_world = _representative_world(worlds)
        representative_rows.extend(
            row
            for row in world_rows
            if row["game"] == game
            and row["world"] == representative_world
            and row["simulated_participant"]
        )

    args.out.mkdir(parents=True, exist_ok=True)
    complete_df = pl.DataFrame(complete_rows).sort(
        ["game", "team", "position", "player"]
    )
    representative_df = pl.DataFrame(representative_rows).sort(
        ["game", "team", "position", "player"]
    )
    complete_df.write_csv(args.out / "complete_player_box_score_distributions.csv")
    representative_df.write_csv(args.out / "representative_world_complete_box_scores.csv")

    defensive_sanity = {
        "max_mean_tackles": float(complete_df.get_column("tackles_mean").max()),
        "max_mean_pressures": float(complete_df.get_column("pressures_mean").max()),
        "max_mean_sacks": float(complete_df.get_column("sacks_mean").max()),
        "max_mean_defensive_interceptions": float(
            complete_df.get_column("defensive_interceptions_mean").max()
        ),
    }
    manifest = {
        "model": "Monster v1.3 complete participant box-score observability layer",
        "season": 2026,
        "week": 1,
        "games": len(MATCHUPS),
        "worlds_per_game": args.worlds,
        "seed": args.seed,
        "market_blind_football": True,
        "dfs_inputs_used": False,
        "sportsbook_inputs_used": False,
        "event_conservation_required": True,
        "role_aware_defensive_attribution": True,
        "all_projected_unit_players_emitted": True,
        "representative_world_only_includes_simulated_participants": True,
        "offensive_skill_stats_event_derived": True,
        "defensive_stats_event_derived_and_role_attributed": True,
        "offensive_line_and_non_box_positions_receive_snap_participation": True,
        "special_teams_participation_is_estimated": True,
        "special_teams_player_stat_attribution_active": False,
        "special_teams_player_stat_attribution_blocker": (
            "game loop does not yet pass kicker_id, punter_id, or returner_id into special-teams events"
        ),
        "defensive_sanity": defensive_sanity,
        "files": {
            "complete_distributions": "complete_player_box_score_distributions.csv",
            "representative_world": "representative_world_complete_box_scores.csv",
        },
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
