from __future__ import annotations

from pathlib import Path

import polars as pl

from monster.feature_compile.mechanisms import TeamMechanismInputs
from monster.teams import NFL_TEAMS


def load_team_ratings(path: Path) -> pl.DataFrame:
    """Load a timestamped, provenance-bearing official EA team ratings snapshot."""
    ratings = pl.read_csv(path).with_columns(
        pl.col("team_id").cast(pl.Utf8).str.to_uppercase(),
        pl.col("offense_rating").cast(pl.Float64),
        pl.col("defense_rating").cast(pl.Float64),
        pl.col("overall_rating").cast(pl.Float64),
    )
    if ratings.get_column("team_id").n_unique() != len(NFL_TEAMS):
        raise ValueError("Madden team ratings must contain exactly one row for all 32 NFL teams")
    if set(ratings.get_column("team_id").to_list()) != set(NFL_TEAMS):
        raise ValueError("Madden team ratings team universe does not match canonical NFL teams")
    return ratings.sort("team_id")


def compile_team_madden_inputs(ratings: pl.DataFrame) -> dict[str, TeamMechanismInputs]:
    """Give EA offense ratings small proxy authority; preserve defense only for audit.

    Observed defensive personnel mechanisms outrank team-level EA defense ratings, so
    defense is deliberately not injected into production scoring here.
    """
    required = {"team_id", "offense_rating", "overall_rating"}
    missing = required.difference(ratings.columns)
    if missing:
        raise ValueError(f"Madden ratings missing columns: {sorted(missing)}")
    return {
        str(row["team_id"]): TeamMechanismInputs(
            team_madden_ovr=float(row["overall_rating"]),
            team_madden_offense=float(row["offense_rating"]),
        )
        for row in ratings.to_dicts()
    }
