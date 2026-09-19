from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import polars as pl


def _num(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    return 0.0 if value is None else float(value)


def _int(row: dict[str, Any], key: str) -> int:
    return round(_num(row, key))


def _fd(row: dict[str, Any]) -> float:
    return round(
        0.04 * _num(row, "passing_yards")
        + 4.0 * _num(row, "passing_tds")
        - 1.0 * _num(row, "interceptions")
        + 0.10 * _num(row, "rushing_yards")
        + 6.0 * _num(row, "rushing_tds")
        + 0.50 * _num(row, "receptions")
        + 0.10 * _num(row, "receiving_yards")
        + 6.0 * _num(row, "receiving_tds")
        - 2.0 * _num(row, "fumbles_lost"),
        2,
    )


def _offense_line(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "team": row.get("team", ""),
        "player": row.get("player", ""),
        "position": row.get("position", ""),
        "passing": {
            "comp": _int(row, "completions"),
            "att": _int(row, "pass_attempts"),
            "yds": round(_num(row, "passing_yards"), 1),
            "td": _int(row, "passing_tds"),
            "int": _int(row, "interceptions"),
        },
        "rushing": {
            "att": _int(row, "rush_attempts"),
            "yds": round(_num(row, "rushing_yards"), 1),
            "td": _int(row, "rushing_tds"),
        },
        "receiving": {
            "tgt": _int(row, "targets"),
            "rec": _int(row, "receptions"),
            "yds": round(_num(row, "receiving_yards"), 1),
            "td": _int(row, "receiving_tds"),
        },
        "fumbles_lost": _int(row, "fumbles_lost"),
        "fanduel_points": _fd(row),
    }


def _defense_line(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "team": row.get("team", ""),
        "player": row.get("player", ""),
        "position": row.get("position", ""),
        "tackles": _int(row, "tackles"),
        "pressures": _int(row, "pressures"),
        "sacks": _int(row, "sacks"),
        "interceptions": _int(row, "interceptions"),
        "stuffs": _int(row, "stuffs"),
        "forced_fumbles": _int(row, "forced_fumbles"),
        "return_yards": round(_num(row, "return_yards"), 1),
        "defensive_tds": _int(row, "defensive_tds"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulation", type=Path, required=True)
    parser.add_argument("--box", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    games = pl.read_csv(args.simulation / "game_distributions.csv").to_dicts()
    reps = pl.read_csv(args.box / "representative_world_box_scores.csv").to_dicts()
    teams = pl.read_csv(args.box / "team_box_score_distributions.csv").to_dicts()

    rep_by_game: dict[str, list[dict[str, Any]]] = {}
    for row in reps:
        rep_by_game.setdefault(str(row["game"]), []).append(row)
    team_by_game: dict[str, list[dict[str, Any]]] = {}
    for row in teams:
        team_by_game.setdefault(str(row["game"]), []).append(row)

    reports: list[dict[str, Any]] = []
    for game_row in games:
        game = str(game_row["game"])
        away, home = game.split("@")
        rows = rep_by_game[game]
        offense = [row for row in rows if row["record_type"] == "offense"]
        defense = [row for row in rows if row["record_type"] == "defense"]

        away_score = round(float(game_row["away_points_mean"]))
        home_score = round(float(game_row["home_points_mean"]))
        team_stats = {}
        for row in team_by_game[game]:
            team = str(row["team"])
            team_stats[team] = {
                "points": _int(row, "points_mean"),
                "pass_yards": round(_num(row, "pass_yards_mean"), 1),
                "rush_yards": round(_num(row, "rush_yards_mean"), 1),
                "total_yards": round(_num(row, "total_yards_mean"), 1),
                "turnovers": _int(row, "turnovers_mean"),
            }

        qbs = [
            _offense_line(row)
            for row in offense
            if str(row.get("position", "")).upper() == "QB"
            and (
                _num(row, "pass_attempts") > 0
                or _num(row, "rush_attempts") > 0
            )
        ]
        rushers = [
            _offense_line(row)
            for row in offense
            if _num(row, "rush_attempts") > 0
        ]
        receivers = [
            _offense_line(row)
            for row in offense
            if _num(row, "targets") > 0 or _num(row, "receptions") > 0
        ]

        notable_defense = [
            _defense_line(row)
            for row in defense
            if (
                _num(row, "sacks") > 0
                or _num(row, "interceptions") > 0
                or _num(row, "forced_fumbles") > 0
                or _num(row, "defensive_tds") > 0
            )
        ]
        for team in (away, home):
            team_def = [
                row for row in defense if str(row.get("team", "")) == team
            ]
            top_tacklers = sorted(
                team_def,
                key=lambda row: _num(row, "tackles"),
                reverse=True,
            )[:3]
            existing = {
                (item["team"], item["player"])
                for item in notable_defense
            }
            for row in top_tacklers:
                key = (str(row.get("team", "")), str(row.get("player", "")))
                if key not in existing and _num(row, "tackles") > 0:
                    notable_defense.append(_defense_line(row))
                    existing.add(key)

        reports.append(
            {
                "game": game,
                "score": {
                    "away": away,
                    "away_points": away_score,
                    "home": home,
                    "home_points": home_score,
                    "winner": (
                        away
                        if away_score > home_score
                        else home
                        if home_score > away_score
                        else "TIE"
                    ),
                    "total": away_score + home_score,
                    "margin": abs(away_score - home_score),
                },
                "team_stats": team_stats,
                "game_anatomy": {
                    "scrimmage_plays": _int(game_row, "scrimmage_plays_mean"),
                    "pass_attempts": _int(game_row, "pass_attempts_mean"),
                    "rush_attempts": _int(game_row, "rush_attempts_mean"),
                    "completions": _int(game_row, "completions_mean"),
                    "sacks": _int(game_row, "sacks_mean"),
                    "scrambles": _int(game_row, "scrambles_mean"),
                    "interceptions": _int(game_row, "interceptions_mean"),
                    "fumbles_lost": _int(game_row, "fumbles_lost_mean"),
                    "punts": _int(game_row, "punts_mean"),
                    "field_goal_attempts": _int(
                        game_row, "field_goal_attempts_mean"
                    ),
                    "field_goals_made": _int(
                        game_row, "field_goals_made_mean"
                    ),
                    "touchdowns": _int(game_row, "touchdowns_mean"),
                    "drives": _int(game_row, "drives_mean"),
                    "overtime": bool(round(_num(game_row, "went_to_overtime_mean"))),
                },
                "quarterbacks": sorted(
                    qbs,
                    key=lambda row: (
                        row["team"],
                        -row["passing"]["att"],
                    ),
                ),
                "rushers": sorted(
                    rushers,
                    key=lambda row: (
                        row["team"],
                        -row["rushing"]["att"],
                        -row["rushing"]["yds"],
                    ),
                ),
                "receivers": sorted(
                    receivers,
                    key=lambda row: (
                        row["team"],
                        -row["receiving"]["tgt"],
                        -row["receiving"]["yds"],
                    ),
                ),
                "notable_defense": sorted(
                    notable_defense,
                    key=lambda row: (
                        row["team"],
                        -row["sacks"],
                        -row["interceptions"],
                        -row["tackles"],
                    ),
                ),
                "top_fanduel": sorted(
                    [_offense_line(row) for row in offense],
                    key=lambda row: -row["fanduel_points"],
                )[:8],
            }
        )

    payload = {
        "experiment": "MONSTER v7.1 Week 2 2026 one-world slate",
        "worlds_per_game": 1,
        "games": len(reports),
        "week2_truth_used": False,
        "market_blind": True,
        "interpretation": (
            "Each game is one realized Monte Carlo football universe. "
            "These are not means, forecasts, or calibrated probabilities."
        ),
        "games_report": reports,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "week2_v701_one_world_report.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("=== WEEK2_V701_ONE_WORLD_REPORT_JSON ===")
    print(json.dumps(payload, separators=(",", ":")))
    print("=== END_WEEK2_V701_ONE_WORLD_REPORT_JSON ===")


if __name__ == "__main__":
    main()
