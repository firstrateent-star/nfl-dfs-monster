from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from monster.feature_compile.league_units import compile_league_unit_player_map

OFFENSE_STATS = (
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
)
DEFENSE_STATS = (
    "pressures",
    "sacks",
    "defensive_interceptions",
    "tackles",
    "stuffs",
    "forced_fumbles",
)
QUANTILES = ("mean", "p10", "p50", "p75", "p90", "p95")


def _read(path: Path) -> pl.DataFrame:
    if path.suffix == ".parquet":
        return pl.read_parquet(path)
    return pl.read_csv(path)


def _copy_distributions(row: dict, source: dict | None, stats: tuple[str, ...]) -> None:
    source = source or {}
    for stat in stats:
        source_stat = "interceptions" if stat in {"passing_interceptions", "defensive_interceptions"} else stat
        for label in QUANTILES:
            row[f"{stat}_{label}"] = float(source.get(f"{source_stat}_{label}", 0.0) or 0.0)
        row[f"{stat}_1_plus_probability"] = float(
            source.get(f"{source_stat}_1_plus_probability", 0.0) or 0.0
        )


def _snap_quantiles(game_row: dict, share: float) -> dict[str, float]:
    share = max(float(share), 0.0)
    mean = 0.5 * float(game_row.get("scrimmage_plays_mean", 0.0)) * share
    p10 = 0.5 * float(game_row.get("scrimmage_plays_p10", 0.0)) * share
    p50 = 0.5 * float(game_row.get("scrimmage_plays_p50", 0.0)) * share
    p90 = 0.5 * float(game_row.get("scrimmage_plays_p90", 0.0)) * share
    return {
        "mean": mean,
        "p10": p10,
        "p50": p50,
        "p75": p50 + 0.625 * (p90 - p50),
        "p90": p90,
        "p95": p90 + 0.35 * max(p90 - p50, 0.0),
    }


def _special_team_mean(game_row: dict, share: float) -> float:
    events = (
        float(game_row.get("punts_mean", 0.0))
        + float(game_row.get("field_goal_attempts_mean", 0.0))
        + float(game_row.get("touchdowns_mean", 0.0))
    )
    return max(events * 0.5 * max(float(share), 0.0), 0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--personnel", type=Path, required=True)
    parser.add_argument("--box-scores", type=Path, required=True)
    parser.add_argument("--first-sim", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026110911)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    personnel = _read(args.personnel)
    units = compile_league_unit_player_map(personnel)
    offense = pl.read_csv(args.box_scores / "offensive_player_box_score_distributions.csv")
    defense = pl.read_csv(args.box_scores / "defensive_player_box_score_distributions.csv")
    representative = pl.read_csv(args.box_scores / "representative_world_box_scores.csv")
    game_dist = pl.read_csv(args.first_sim / "game_distributions.csv")

    offense_map = {
        (str(r["game"]), str(r["team"]), str(r["player_id"])): r for r in offense.to_dicts()
    }
    defense_map = {
        (str(r["game"]), str(r["team"]), str(r["player_id"])): r for r in defense.to_dicts()
    }
    game_map = {str(r["game"]): r for r in game_dist.to_dicts()}

    source_identity: dict[tuple[str, str, str], tuple[str, str]] = {}
    for source in (offense_map, defense_map):
        for key, value in source.items():
            source_identity[key] = (str(value.get("player", key[2])), str(value.get("position", "")))

    unit_map = {(team, p.player_id): p for team, players in units.items() for p in players}
    complete_rows: list[dict] = []

    for game in sorted(game_map):
        away, home = game.split("@")
        game_row = game_map[game]
        for team in (away, home):
            ids = {pid for g, t, pid in source_identity if g == game and t == team}
            ids.update(pid for t, pid in unit_map if t == team)
            for player_id in sorted(ids):
                source_key = (game, team, player_id)
                player, position = source_identity.get(source_key, (player_id, ""))
                unit_player = unit_map.get((team, player_id))
                if unit_player is not None and not position:
                    position = unit_player.position
                row = {
                    "game": game,
                    "team": team,
                    "player_id": player_id,
                    "player": player,
                    "position": position,
                }
                _copy_distributions(row, offense_map.get(source_key), OFFENSE_STATS)
                _copy_distributions(row, defense_map.get(source_key), DEFENSE_STATS)

                offense_share = 0.0 if unit_player is None else unit_player.offense_snap_share
                defense_share = 0.0 if unit_player is None else unit_player.defense_snap_share
                special_share = 0.0 if unit_player is None else unit_player.special_teams_snap_share
                for label, value in _snap_quantiles(game_row, offense_share).items():
                    row[f"offense_snaps_{label}"] = value
                for label, value in _snap_quantiles(game_row, defense_share).items():
                    row[f"defense_snaps_{label}"] = value
                special_mean = _special_team_mean(game_row, special_share)
                for label in QUANTILES:
                    row[f"special_teams_snaps_estimated_{label}"] = special_mean

                row["offense_snaps_1_plus_probability"] = float(offense_share > 0.0)
                row["defense_snaps_1_plus_probability"] = float(defense_share > 0.0)
                row["special_teams_snaps_estimated_1_plus_probability"] = float(special_share > 0.0)
                stat_probability = max(
                    [row[f"{stat}_1_plus_probability"] for stat in OFFENSE_STATS + DEFENSE_STATS]
                    + [0.0]
                )
                row["participation_probability"] = max(
                    stat_probability,
                    float(offense_share > 0.0 or defense_share > 0.0 or special_share > 0.0),
                )
                complete_rows.append(row)

    rep_rows: list[dict] = []
    rep_source = [r for r in representative.to_dicts() if str(r.get("record_type", "")) != "game"]
    grouped: dict[tuple[str, int, str, str], dict] = {}
    for source in rep_source:
        key = (
            str(source["game"]),
            int(source["world"]),
            str(source["team"]),
            str(source["player_id"]),
        )
        row = grouped.setdefault(
            key,
            {
                "game": key[0],
                "world": key[1],
                "team": key[2],
                "player_id": key[3],
                "player": str(source.get("player", key[3])),
                "position": str(source.get("position", "")),
            },
        )
        record_type = str(source.get("record_type", ""))
        if record_type == "offense":
            for stat in OFFENSE_STATS:
                src = "interceptions" if stat == "passing_interceptions" else stat
                row[stat] = float(source.get(src, 0.0) or 0.0)
        elif record_type == "defense":
            for stat in DEFENSE_STATS:
                src = "interceptions" if stat == "defensive_interceptions" else stat
                row[stat] = float(source.get(src, 0.0) or 0.0)

    for key, row in grouped.items():
        game, _, team, player_id = key
        unit_player = unit_map.get((team, player_id))
        game_row = game_map[game]
        offense_share = 0.0 if unit_player is None else unit_player.offense_snap_share
        defense_share = 0.0 if unit_player is None else unit_player.defense_snap_share
        special_share = 0.0 if unit_player is None else unit_player.special_teams_snap_share
        row["offense_snaps"] = int(round(_snap_quantiles(game_row, offense_share)["mean"]))
        row["defense_snaps"] = int(round(_snap_quantiles(game_row, defense_share)["mean"]))
        row["special_teams_snaps_estimated"] = int(round(_special_team_mean(game_row, special_share)))
        for stat in OFFENSE_STATS + DEFENSE_STATS:
            row.setdefault(stat, 0.0)
        row["simulated_participant"] = bool(
            row["offense_snaps"] + row["defense_snaps"] + row["special_teams_snaps_estimated"] > 0
            or any(float(row[stat]) > 0 for stat in OFFENSE_STATS + DEFENSE_STATS)
        )
        if row["simulated_participant"]:
            rep_rows.append(row)

    args.out.mkdir(parents=True, exist_ok=True)
    complete_df = pl.DataFrame(complete_rows).sort(["game", "team", "position", "player"])
    rep_df = pl.DataFrame(rep_rows).sort(["game", "team", "position", "player"])
    complete_df.write_csv(args.out / "complete_player_box_score_distributions.csv")
    rep_df.write_csv(args.out / "representative_world_complete_box_scores.csv")

    defensive_sanity = {
        "max_mean_tackles": float(complete_df.get_column("tackles_mean").max()),
        "max_mean_pressures": float(complete_df.get_column("pressures_mean").max()),
        "max_mean_sacks": float(complete_df.get_column("sacks_mean").max()),
        "max_mean_defensive_interceptions": float(
            complete_df.get_column("defensive_interceptions_mean").max()
        ),
    }
    manifest = {
        "model": "Monster v1.3 complete participant box-score materialization layer",
        "season": 2026,
        "week": 1,
        "games": len(game_map),
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
        "materialized_from_existing_box_score_worlds": True,
        "resimulation_required": False,
        "snap_distribution_note": (
            "snap distributions are derived from frozen unit snap shares and first-simulation game play distributions; "
            "box-score statistics are reused from the already simulated box-score worlds"
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
