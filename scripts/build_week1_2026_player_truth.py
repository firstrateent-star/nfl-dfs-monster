from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable

FD = {
    "passing_yards": 0.04,
    "passing_tds": 4.0,
    "interceptions": -1.0,
    "rushing_yards": 0.10,
    "rushing_tds": 6.0,
    "receptions": 0.50,
    "receiving_yards": 0.10,
    "receiving_tds": 6.0,
    "fumbles_lost": -2.0,
}

TEAM_ALIASES = {"JAX": "JAC", "WSH": "WAS", "LA": "LAR"}


def _norm_team(value: str | None) -> str:
    team = (value or "").strip().upper()
    return TEAM_ALIASES.get(team, team)


def _first(row: dict[str, str], names: Iterable[str], default: str = "") -> str:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return str(value)
    return default


def _float(row: dict[str, str], names: Iterable[str]) -> float:
    value = _first(row, names)
    if not value:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _intish(row: dict[str, str], names: Iterable[str]) -> int:
    return int(round(_float(row, names)))


def _fumbles_lost(row: dict[str, str]) -> int:
    direct = _first(row, ("fumbles_lost", "fumble_lost"))
    if direct:
        try:
            return int(round(float(direct)))
        except ValueError:
            pass
    component_names = (
        "sack_fumbles_lost",
        "rushing_fumbles_lost",
        "receiving_fumbles_lost",
        "special_teams_fumbles_lost",
    )
    return int(round(sum(_float(row, (name,)) for name in component_names)))


def fanduel_points(stats: dict[str, float]) -> float:
    return sum(float(stats[key]) * weight for key, weight in FD.items())


def _load_game_map(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            game = row["game"]
            result[_norm_team(row["away"])] = game
            result[_norm_team(row["home"])] = game
    if len(result) != 24:
        raise ValueError(f"expected 24 benchmark teams, found {len(result)}")
    return result


def build_truth(stats_path: Path, games_path: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    team_to_game = _load_game_map(games_path)
    rows: list[dict[str, object]] = []
    source_rows = 0
    with stats_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("player stats CSV has no header")
        for raw in reader:
            season = _intish(raw, ("season",))
            week = _intish(raw, ("week",))
            season_type = _first(raw, ("season_type", "game_type"), "REG").upper()
            if season != 2026 or week != 1 or season_type not in {"REG", "REGULAR"}:
                continue
            team = _norm_team(_first(raw, ("recent_team", "team", "posteam")))
            if team not in team_to_game:
                continue
            source_rows += 1
            stats = {
                "passing_attempts": _intish(raw, ("attempts", "passing_attempts", "pass_attempts")),
                "completions": _intish(raw, ("completions",)),
                "passing_yards": _float(raw, ("passing_yards",)),
                "passing_tds": _intish(raw, ("passing_tds", "passing_touchdowns")),
                "interceptions": _intish(raw, ("interceptions", "passing_interceptions")),
                "carries": _intish(raw, ("carries", "rushing_attempts", "rush_attempts")),
                "rushing_yards": _float(raw, ("rushing_yards",)),
                "rushing_tds": _intish(raw, ("rushing_tds", "rushing_touchdowns")),
                "targets": _intish(raw, ("targets",)),
                "receptions": _intish(raw, ("receptions",)),
                "receiving_yards": _float(raw, ("receiving_yards",)),
                "receiving_tds": _intish(raw, ("receiving_tds", "receiving_touchdowns")),
                "fumbles_lost": _fumbles_lost(raw),
            }
            player_id = _first(raw, ("player_id", "gsis_id", "player_gsis_id"))
            player_name = _first(raw, ("player_display_name", "player_name", "name"))
            if not player_id and not player_name:
                continue
            rows.append(
                {
                    "season": 2026,
                    "week": 1,
                    "game": team_to_game[team],
                    "team": team,
                    "player_id": player_id,
                    "player_name": player_name,
                    "position": _first(raw, ("position", "position_group")),
                    **stats,
                    "fanduel_points": round(fanduel_points(stats), 6),
                }
            )
    rows.sort(key=lambda r: (str(r["game"]), str(r["team"]), str(r["player_id"]), str(r["player_name"])))
    source_sha = hashlib.sha256(stats_path.read_bytes()).hexdigest()
    manifest = {
        "season": 2026,
        "week": 1,
        "source_file": stats_path.name,
        "source_sha256": source_sha,
        "benchmark_games": 12,
        "benchmark_teams": 24,
        "source_rows_retained": source_rows,
        "truth_rows_written": len(rows),
        "fanduel_scoring": FD,
        "team_aliases": TEAM_ALIASES,
    }
    return rows, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze normalized 2026 Week 1 player truth for Monster reality audit.")
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--games", type=Path, default=Path("benchmarks/week1_2026_sunday_games.csv"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows, manifest = build_truth(args.stats, args.games)
    args.out.mkdir(parents=True, exist_ok=True)
    out_csv = args.out / "week1_2026_player_truth.csv"
    fields = [
        "season", "week", "game", "team", "player_id", "player_name", "position",
        "passing_attempts", "completions", "passing_yards", "passing_tds", "interceptions",
        "carries", "rushing_yards", "rushing_tds", "targets", "receptions",
        "receiving_yards", "receiving_tds", "fumbles_lost", "fanduel_points",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (args.out / "week1_2026_player_truth_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
