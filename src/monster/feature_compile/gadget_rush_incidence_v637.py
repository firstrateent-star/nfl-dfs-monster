from __future__ import annotations

import numpy as np
import polars as pl

POSITIONS = ("WR", "TE")


def _first_column(frame: pl.DataFrame, names: tuple[str, ...]) -> str:
    for name in names:
        if name in frame.columns:
            return name
    raise ValueError(f"Missing required column; tried {names}")


def normalize_weekly_stats(frame: pl.DataFrame) -> pl.DataFrame:
    """Normalize weekly player stats for gadget-rush incidence analysis."""
    season_col = _first_column(frame, ("season",))
    week_col = _first_column(frame, ("week",))
    player_col = _first_column(frame, ("player_id", "gsis_id", "player_gsis_id"))
    position_col = _first_column(frame, ("position", "position_group"))
    team_col = _first_column(frame, ("recent_team", "team", "posteam"))
    carries_col = _first_column(frame, ("carries", "rushing_attempts", "rush_attempts"))

    out = frame
    if "season_type" in out.columns:
        out = out.filter(
            pl.col("season_type").cast(pl.Utf8).str.to_uppercase().is_in(["REG", "REGULAR"])
        )

    out = out.select(
        pl.col(season_col).cast(pl.Int64).alias("season"),
        pl.col(week_col).cast(pl.Int64).alias("week"),
        pl.col(player_col).cast(pl.Utf8).alias("player_id"),
        pl.col(position_col).cast(pl.Utf8).str.to_uppercase().alias("position"),
        pl.col(team_col).cast(pl.Utf8).str.to_uppercase().alias("team"),
        pl.col(carries_col).fill_null(0.0).cast(pl.Float64).alias("carries"),
    ).filter(
        pl.col("player_id").is_not_null()
        & (pl.col("player_id") != "")
        & pl.col("team").is_not_null()
        & (pl.col("team") != "")
    )

    team = out.group_by(["season", "week", "team"]).agg(
        pl.col("carries").sum().alias("team_carries")
    )
    return out.join(team, on=["season", "week", "team"], how="left").with_columns(
        (pl.col("carries") > 0).cast(pl.Int64).alias("rush_entry"),
        (pl.col("carries") / pl.col("team_carries").clip(lower_bound=1.0)).alias(
            "team_rush_share"
        ),
    )


def summarize_incidence(stats: pl.DataFrame) -> dict[str, pl.DataFrame]:
    gadget = stats.filter(pl.col("position").is_in(POSITIONS))

    position = (
        gadget.group_by("position")
        .agg(
            pl.len().alias("player_games"),
            pl.col("rush_entry").sum().alias("rush_entry_games"),
            pl.col("rush_entry").mean().alias("entry_rate"),
            pl.col("carries").mean().alias("carries_per_player_game"),
            pl.when(pl.col("rush_entry") == 1)
            .then(pl.col("carries"))
            .otherwise(None)
            .mean()
            .alias("carries_conditional_on_entry"),
            pl.col("team_rush_share").mean().alias("mean_player_game_team_rush_share"),
        )
        .sort("position")
    )

    team_week = (
        gadget.group_by(["season", "week", "team"])
        .agg(
            pl.col("carries").sum().alias("gadget_carries"),
            pl.col("team_carries").first().alias("team_carries"),
            pl.col("rush_entry").sum().alias("gadget_rushers"),
        )
        .with_columns(
            (
                pl.col("gadget_carries")
                / pl.col("team_carries").clip(lower_bound=1.0)
            ).alias("gadget_team_rush_share")
        )
    )

    team_week_summary = pl.DataFrame(
        {
            "team_weeks": [team_week.height],
            "mean_gadget_team_rush_share": [
                float(team_week["gadget_team_rush_share"].mean())
            ],
            "median_gadget_team_rush_share": [
                float(team_week["gadget_team_rush_share"].median())
            ],
            "p90_gadget_team_rush_share": [
                float(
                    team_week["gadget_team_rush_share"].quantile(
                        0.90, interpolation="linear"
                    )
                )
            ],
            "team_weeks_with_any_gadget_rush": [
                int((team_week["gadget_carries"] > 0).sum())
            ],
            "probability_team_has_any_gadget_rush": [
                float((team_week["gadget_carries"] > 0).mean())
            ],
            "mean_gadget_rushers_per_team_week": [
                float(team_week["gadget_rushers"].mean())
            ],
        }
    )

    player_season = (
        gadget.group_by(["season", "player_id", "position"])
        .agg(
            pl.len().alias("games_observed"),
            pl.col("rush_entry").sum().alias("rush_games"),
            pl.col("carries").sum().alias("season_carries"),
            pl.col("team_rush_share").mean().alias("mean_team_rush_share"),
        )
        .with_columns(
            (
                pl.col("rush_games")
                / pl.col("games_observed").clip(lower_bound=1)
            ).alias("entry_rate")
        )
    )

    return {
        "position_summary": position,
        "team_week": team_week,
        "team_week_summary": team_week_summary,
        "player_season": player_season,
    }


def transition_table(
    player_season: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    prior = (
        player_season.rename(
            {
                "season": "prior_season",
                "games_observed": "prior_games",
                "rush_games": "prior_rush_games",
                "season_carries": "prior_carries",
                "mean_team_rush_share": "prior_mean_team_rush_share",
                "entry_rate": "prior_entry_rate",
            }
        )
        .with_columns((pl.col("prior_season") + 1).alias("season"))
    )

    nxt = player_season.rename(
        {
            "games_observed": "next_games",
            "rush_games": "next_rush_games",
            "season_carries": "next_carries",
            "mean_team_rush_share": "next_mean_team_rush_share",
            "entry_rate": "next_entry_rate",
        }
    )

    joined = (
        prior.join(nxt, on=["season", "player_id", "position"], how="inner")
        .with_columns(
            pl.when(pl.col("prior_carries") <= 0)
            .then(pl.lit("0"))
            .when(pl.col("prior_carries") <= 3)
            .then(pl.lit("1-3"))
            .when(pl.col("prior_carries") <= 8)
            .then(pl.lit("4-8"))
            .otherwise(pl.lit("9+"))
            .alias("prior_carry_bin")
        )
    )

    grouped = (
        joined.group_by(["position", "prior_carry_bin"])
        .agg(
            pl.len().alias("player_seasons"),
            pl.col("prior_games").sum().alias("prior_games"),
            pl.col("prior_rush_games").sum().alias("prior_rush_games"),
            pl.col("next_games").sum().alias("next_games"),
            pl.col("next_rush_games").sum().alias("next_rush_games"),
            pl.col("prior_entry_rate").mean().alias("mean_prior_entry_rate"),
            pl.col("next_entry_rate").mean().alias("mean_next_entry_rate"),
        )
        .with_columns(
            (
                pl.col("next_rush_games")
                / pl.col("next_games").clip(lower_bound=1)
            ).alias("weighted_next_entry_rate")
        )
        .sort(["position", "prior_carry_bin"])
    )
    return joined, grouped


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])
