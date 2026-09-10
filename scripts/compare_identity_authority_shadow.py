from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl


def _game_metrics(frame: pl.DataFrame) -> dict[str, float]:
    team_scores = np.asarray(
        frame.get_column("away_points_mean").to_list()
        + frame.get_column("home_points_mean").to_list(),
        dtype=float,
    )
    totals = frame.get_column("total_mean").to_numpy()
    margins = frame.get_column("margin_mean").to_numpy()
    return {
        "league_mean_team_points": float(team_scores.mean()),
        "between_team_mean_score_sd": float(team_scores.std(ddof=1)),
        "between_team_mean_score_range": float(team_scores.max() - team_scores.min()),
        "between_game_total_mean_sd": float(totals.std(ddof=1)),
        "between_game_margin_mean_sd": float(margins.std(ddof=1)),
        "mean_absolute_margin": float(np.abs(margins).mean()),
        "league_mean_game_total": float(totals.mean()),
    }


def _player_dispersion(frame: pl.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for position in ("QB", "RB", "WR", "TE"):
        sub = frame.filter(pl.col("position") == position)
        if not sub.height:
            continue
        metric = "passing_yards_mean" if position == "QB" else (
            "rushing_yards_mean" if position == "RB" else "receiving_yards_mean"
        )
        values = sub.get_column(metric).to_numpy()
        rows.append(
            {
                "position": position,
                "metric": metric,
                "players": sub.height,
                "mean": float(values.mean()),
                "sd_across_player_means": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "p90_of_player_means": float(np.quantile(values, 0.90)),
                "max": float(values.max()),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--repair", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    control_games = pl.read_csv(args.control / "game_distributions.csv").sort("game")
    repair_games = pl.read_csv(args.repair / "game_distributions.csv").sort("game")
    control_players = pl.read_csv(args.control / "player_distributions.csv")
    repair_players = pl.read_csv(args.repair / "player_distributions.csv")

    control = _game_metrics(control_games)
    repair = _game_metrics(repair_games)
    deltas = {key: repair[key] - control[key] for key in control}

    joined_games = control_games.select(
        "game", "away_points_mean", "home_points_mean", "total_mean", "margin_mean"
    ).join(
        repair_games.select(
            "game", "away_points_mean", "home_points_mean", "total_mean", "margin_mean"
        ),
        on="game",
        suffix="_repair",
    )
    joined_games = joined_games.with_columns(
        (pl.col("away_points_mean_repair") - pl.col("away_points_mean")).alias("away_points_delta"),
        (pl.col("home_points_mean_repair") - pl.col("home_points_mean")).alias("home_points_delta"),
        (pl.col("total_mean_repair") - pl.col("total_mean")).alias("total_delta"),
        (pl.col("margin_mean_repair") - pl.col("margin_mean")).alias("margin_delta"),
    )

    keys = ["game", "player_id"]
    joined_players = control_players.select(
        *keys,
        "player",
        "position",
        "fanduel_mean",
        "passing_yards_mean",
        "rushing_yards_mean",
        "receiving_yards_mean",
    ).join(
        repair_players.select(
            *keys,
            "fanduel_mean",
            "passing_yards_mean",
            "rushing_yards_mean",
            "receiving_yards_mean",
        ),
        on=keys,
        suffix="_repair",
    )
    for metric in (
        "fanduel_mean",
        "passing_yards_mean",
        "rushing_yards_mean",
        "receiving_yards_mean",
    ):
        joined_players = joined_players.with_columns(
            (pl.col(f"{metric}_repair") - pl.col(metric)).alias(f"{metric}_delta")
        )

    trace = pl.read_csv(args.repair / "identity_authority_trace.csv")
    report = {
        "artifact": "Monster v1.3 Stage 3 Identity Authority Shadow Comparison",
        "market_blind": True,
        "paired_common_random_numbers": True,
        "control": control,
        "identity_repair": repair,
        "delta": deltas,
        "mean_absolute_team_score_change": float(
            np.mean(
                np.abs(
                    np.concatenate(
                        [
                            joined_games.get_column("away_points_delta").to_numpy(),
                            joined_games.get_column("home_points_delta").to_numpy(),
                        ]
                    )
                )
            )
        ),
        "max_absolute_team_score_change": float(
            np.max(
                np.abs(
                    np.concatenate(
                        [
                            joined_games.get_column("away_points_delta").to_numpy(),
                            joined_games.get_column("home_points_delta").to_numpy(),
                        ]
                    )
                )
            )
        ),
        "control_player_dispersion": _player_dispersion(control_players),
        "repair_player_dispersion": _player_dispersion(repair_players),
        "identity_trace_ranges": {
            column: {
                "min": float(trace.get_column(column).min()),
                "mean": float(trace.get_column(column).mean()),
                "max": float(trace.get_column(column).max()),
            }
            for column in (
                "quarterback_efficiency",
                "pass_efficiency_after_qb",
                "live_rush_context",
                "rush_efficiency_after_live_prior",
                "mean_receiver_efficiency",
                "mean_rusher_efficiency",
            )
        },
        "interpretation_rule": (
            "Greater between-game/team dispersion is evidence that identity is no longer "
            "collapsed, but is not by itself proof of calibration. Promotion requires "
            "historical/OOS elasticity and anatomy checks rather than maximizing spread."
        ),
        "promotion_status": "SHADOW_IDENTITY_EVIDENCE_ONLY",
    }

    args.out.mkdir(parents=True, exist_ok=True)
    joined_games.write_csv(args.out / "game_deltas.csv")
    joined_players.sort(pl.col("fanduel_mean_delta").abs(), descending=True).write_csv(
        args.out / "player_deltas.csv"
    )
    (args.out / "identity_authority_comparison.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
