from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

PLAYER_METRICS = (
    "offense_snaps",
    "defense_snaps",
    "special_teams_snaps_estimated",
    "pass_attempts",
    "completions",
    "passing_yards",
    "passing_tds",
    "passing_interceptions",
    "rush_attempts",
    "rushing_yards",
    "rushing_tds",
    "targets",
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "fumbles_lost",
    "pressures",
    "sacks",
    "defensive_interceptions",
    "tackles",
    "stuffs",
    "forced_fumbles",
)


def _player_box(distributions: pl.DataFrame) -> pl.DataFrame:
    identity = ["game", "team", "player_id", "player", "position", "participation_probability"]
    expressions = [pl.col(column) for column in identity]
    for metric in PLAYER_METRICS:
        mean_column = f"{metric}_mean"
        if mean_column in distributions.columns:
            expressions.append(pl.col(mean_column).alias(metric))
    return distributions.select(expressions).sort(["game", "team", "position", "player"])


def _team_box(team_distributions: pl.DataFrame) -> pl.DataFrame:
    identity = [column for column in ("game", "team") if column in team_distributions.columns]
    mean_columns = [column for column in team_distributions.columns if column.endswith("_mean")]
    expressions = [pl.col(column) for column in identity]
    expressions.extend(pl.col(column).alias(column.removesuffix("_mean")) for column in mean_columns)
    return team_distributions.select(expressions).sort(["game", "team"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--players", type=Path, required=True)
    parser.add_argument("--teams", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    source_manifest = json.loads(args.source_manifest.read_text())
    player_box = _player_box(pl.read_csv(args.players))
    team_box = _team_box(pl.read_csv(args.teams))

    args.out.mkdir(parents=True, exist_ok=True)
    player_box.write_csv(args.out / "projected_player_box_score.csv")
    team_box.write_csv(args.out / "projected_team_box_score.csv")

    manifest = {
        "model": "Monster v1.3 market-blind consensus football box score",
        "season": source_manifest["season"],
        "week": source_manifest["week"],
        "games": source_manifest["games"],
        "worlds_per_game": source_manifest["worlds_per_game"],
        "seed": source_manifest["seed"],
        "market_blind_football": source_manifest["market_blind_football"],
        "dfs_inputs_used": source_manifest["dfs_inputs_used"],
        "sportsbook_inputs_used": source_manifest["sportsbook_inputs_used"],
        "definition": (
            "Each numeric player/team statistic is the arithmetic mean of that statistic "
            "across all simulated worlds for the game. This is the singular projected box score "
            "consumed downstream by DFS translation; simulation distributions remain preserved "
            "in the source corpus."
        ),
        "player_rows": player_box.height,
        "team_rows": team_box.height,
        "files": {
            "players": "projected_player_box_score.csv",
            "teams": "projected_team_box_score.csv",
        },
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
