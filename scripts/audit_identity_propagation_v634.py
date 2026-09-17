from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import polars as pl


def _corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(ys) < 2:
        return None
    if statistics.pstdev(xs) == 0.0 or statistics.pstdev(ys) == 0.0:
        return None
    return float(statistics.correlation(xs, ys))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    trace = pl.read_csv(args.simulation / "matchup_identity_trace_v634.csv")
    games = pl.read_csv(args.simulation / "game_distributions.csv")
    players = pl.read_csv(args.simulation / "player_distributions.csv")

    game_map: dict[str, dict[str, object]] = {}
    for row in games.to_dicts():
        away = str(row["away"])
        home = str(row["home"])
        game = str(row["game"])
        game_map[away] = {
            "game": game,
            "opponent": home,
            "team_points_mean": float(row["away_points_mean"]),
            "opponent_points_mean": float(row["home_points_mean"]),
            "game_total_mean": float(row["total_mean"]),
            "margin_mean": float(row["margin_mean"]),
        }
        game_map[home] = {
            "game": game,
            "opponent": away,
            "team_points_mean": float(row["home_points_mean"]),
            "opponent_points_mean": float(row["away_points_mean"]),
            "game_total_mean": float(row["total_mean"]),
            "margin_mean": -float(row["margin_mean"]),
        }

    player_lookup = {
        (str(row["game"]), str(row["player_id"])): row for row in players.to_dicts()
    }

    rows: list[dict[str, object]] = []
    for item in trace.to_dicts():
        team = str(item["team_id"])
        if team not in game_map:
            continue
        game = game_map[team]
        qb = player_lookup.get((str(game["game"]), str(item["quarterback_id"])))
        rows.append(
            {
                **item,
                **game,
                "qb_passing_yards_mean": (
                    None if qb is None else float(qb["passing_yards_mean"])
                ),
                "qb_fanduel_mean": None if qb is None else float(qb["fanduel_mean"]),
            }
        )

    frame = pl.DataFrame(rows).sort("team_id")
    frame.write_csv(args.out / "v634_identity_propagation_by_team.csv")

    def vals(column: str) -> list[float]:
        return [float(v) for v in frame.get_column(column).drop_nulls().to_list()]

    before_pass = vals("pass_efficiency_before")
    after_pass = vals("pass_efficiency_after")
    before_rush = vals("rush_efficiency_before")
    after_rush = vals("rush_efficiency_after")
    points = vals("team_points_mean")
    qb_eff = vals("quarterback_efficiency")

    qb_rows = frame.filter(pl.col("qb_passing_yards_mean").is_not_null())
    qb_eff_for_yards = [
        float(v) for v in qb_rows.get_column("quarterback_efficiency").to_list()
    ]
    qb_yards = [
        float(v) for v in qb_rows.get_column("qb_passing_yards_mean").to_list()
    ]

    report = {
        "experiment": "v6.3.4-identity-world-propagation",
        "teams": frame.height,
        "dispersion": {
            "pass_efficiency_before_sd": statistics.pstdev(before_pass)
            if len(before_pass) > 1
            else 0.0,
            "pass_efficiency_after_sd": statistics.pstdev(after_pass)
            if len(after_pass) > 1
            else 0.0,
            "rush_efficiency_before_sd": statistics.pstdev(before_rush)
            if len(before_rush) > 1
            else 0.0,
            "rush_efficiency_after_sd": statistics.pstdev(after_rush)
            if len(after_rush) > 1
            else 0.0,
            "team_points_mean_sd": statistics.pstdev(points) if len(points) > 1 else 0.0,
            "quarterback_efficiency_sd": statistics.pstdev(qb_eff)
            if len(qb_eff) > 1
            else 0.0,
            "neutral_pass_rate_sd": float(frame.get_column("neutral_pass_rate").std(ddof=0)),
            "situational_pass_rate_sd_mean": float(
                frame.get_column("situational_pass_rate_sd").mean()
            ),
        },
        "propagation_correlations": {
            "pass_efficiency_after_vs_team_points": _corr(after_pass, points),
            "rush_efficiency_after_vs_team_points": _corr(after_rush, points),
            "quarterback_efficiency_vs_qb_passing_yards": _corr(
                qb_eff_for_yards, qb_yards
            ),
        },
        "interpretation": (
            "These correlations are propagation diagnostics, not calibration targets. "
            "The purpose is to detect whether player/scheme identity survives into full-world "
            "production; reality grading remains separate."
        ),
        "direct_points_authority": False,
        "promotion_status": "SHADOW_DIAGNOSTIC_ONLY",
    }
    (args.out / "v634_identity_propagation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
