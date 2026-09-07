from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

import polars as pl

from monster.feature_compile.participation import compile_snap_priors
from monster.teams import NFL_TEAMS, TEAM_ALIASES

_TEAM_MAP = {**{team: team for team in NFL_TEAMS}, **TEAM_ALIASES}
_SNAP_OUTPUTS = [
    "offense_snap_share",
    "defense_snap_share",
    "special_teams_snap_share",
    "offense_snap_uncertainty",
    "defense_snap_uncertainty",
    "special_teams_snap_uncertainty",
    "primary_snap_share",
    "snap_share_uncertainty",
    "snap_games_observed",
]


def canonicalize_team_column(df: pl.DataFrame, column: str = "team") -> pl.DataFrame:
    """Normalize provider team abbreviations before any cross-source joins."""
    if column not in df.columns:
        raise ValueError(f"Missing team column: {column}")
    return df.with_columns(
        pl.col(column)
        .cast(pl.Utf8)
        .str.to_uppercase()
        .replace(_TEAM_MAP)
        .alias("team_id")
    ).filter(pl.col("team_id").is_in(list(NFL_TEAMS)))


def latest_roster_state(rosters: pl.DataFrame) -> pl.DataFrame:
    """Return the most recent roster state for every team, not cumulative season rosters."""
    roster = canonicalize_team_column(rosters)
    if "week" not in roster.columns:
        return roster
    latest = roster.group_by("team_id").agg(pl.col("week").max().alias("_latest_week"))
    return (
        roster.join(latest, on="team_id", how="inner")
        .filter(pl.col("week") == pl.col("_latest_week"))
        .drop("_latest_week")
    )


def _player_static_columns(players: pl.DataFrame) -> list[str]:
    candidates = [
        "gsis_id",
        "display_name",
        "common_first_name",
        "first_name",
        "last_name",
        "short_name",
        "position",
        "position_group",
        "birth_date",
        "height",
        "weight",
        "years_of_experience",
        "rookie_year",
        "draft_club",
        "draft_number",
        "pfr_id",
        "espn_id",
        "pff_id",
        "otc_id",
    ]
    return [c for c in candidates if c in players.columns]


def enrich_roster_identity(roster: pl.DataFrame, players: pl.DataFrame) -> pl.DataFrame:
    """Join stable nflverse player identity/static facts onto current team membership."""
    if "gsis_id" not in roster.columns or "gsis_id" not in players.columns:
        raise ValueError("League identity join requires gsis_id in roster and players datasets")
    missing_static = [
        c for c in _player_static_columns(players) if c == "gsis_id" or c not in roster.columns
    ]
    static = players.select(missing_static).unique(subset=["gsis_id"], keep="last")
    return roster.join(static, on="gsis_id", how="left")


def _combine_snap_history(
    historical_snap_counts: pl.DataFrame,
    current_snap_counts: pl.DataFrame | None,
) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    if historical_snap_counts.height:
        frames.append(historical_snap_counts)
    if current_snap_counts is not None and current_snap_counts.height:
        frames.append(current_snap_counts)
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed")


def _normalize_player_name(value: str | None) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", text)
    return re.sub(r"[^a-z0-9]", "", text)


def _fill_pfr_ids_from_unique_snap_names(roster: pl.DataFrame, snaps: pl.DataFrame) -> pl.DataFrame:
    """Recover missing PFR IDs from exact normalized names when the mapping is unique.

    This is deterministic identity recovery, not fuzzy matching. It is valuable for OL,
    whose current nflverse player rows can lack PFR IDs even though PFR snap counts contain
    the player's full name and PFR ID. Ambiguous names remain unresolved and neutral.
    """
    if (
        "player" not in snaps.columns
        or "pfr_player_id" not in snaps.columns
        or "pfr_id" not in roster.columns
    ):
        return roster

    roster_name = "display_name" if "display_name" in roster.columns else None
    if roster_name is None:
        return roster

    snap_map = (
        snaps.select("player", "pfr_player_id")
        .drop_nulls(["player", "pfr_player_id"])
        .with_columns(
            pl.col("player")
            .map_elements(_normalize_player_name, return_dtype=pl.Utf8)
            .alias("_snap_name_key")
        )
        .filter(pl.col("_snap_name_key") != "")
        .group_by("_snap_name_key")
        .agg(
            pl.col("pfr_player_id").n_unique().alias("_pfr_count"),
            pl.col("pfr_player_id").first().alias("_pfr_from_snap_name"),
        )
        .filter(pl.col("_pfr_count") == 1)
        .drop("_pfr_count")
    )

    return (
        roster.with_columns(
            pl.col(roster_name)
            .map_elements(_normalize_player_name, return_dtype=pl.Utf8)
            .alias("_snap_name_key")
        )
        .join(snap_map, on="_snap_name_key", how="left")
        .with_columns(
            pl.coalesce([pl.col("pfr_id"), pl.col("_pfr_from_snap_name")]).alias("pfr_id"),
            (
                pl.col("pfr_id").is_null() & pl.col("_pfr_from_snap_name").is_not_null()
            ).alias("pfr_id_recovered_from_snap_name"),
        )
        .drop("_snap_name_key", "_pfr_from_snap_name")
    )


def _attach_snap_priors(
    roster: pl.DataFrame,
    historical_snap_counts: pl.DataFrame,
    current_snap_counts: pl.DataFrame | None,
    recent_games: int,
) -> pl.DataFrame:
    snaps = _combine_snap_history(historical_snap_counts, current_snap_counts)
    if not snaps.height:
        return roster.with_columns(
            *[pl.lit(None, dtype=pl.Float64).alias(c) for c in _SNAP_OUTPUTS[:-1]],
            pl.lit(0, dtype=pl.Int64).alias("snap_games_observed"),
            pl.lit(None, dtype=pl.Utf8).alias("prior_team_id"),
            pl.lit(0, dtype=pl.Int64).alias("snap_team_count"),
            pl.lit(False).alias("changed_team_since_snap_history"),
            pl.lit(False).alias("pfr_id_recovered_from_snap_name"),
        )

    snaps = canonicalize_team_column(snaps).with_columns(pl.col("team_id").alias("team"))
    roster = _fill_pfr_ids_from_unique_snap_names(roster, snaps)
    if "pfr_id_recovered_from_snap_name" not in roster.columns:
        roster = roster.with_columns(pl.lit(False).alias("pfr_id_recovered_from_snap_name"))
    priors = compile_snap_priors(snaps, recent_games=recent_games)

    roster_pfr = "pfr_id" if "pfr_id" in roster.columns else None
    if roster_pfr is None and "pfr_player_id" in roster.columns:
        roster_pfr = "pfr_player_id"
    if roster_pfr is None:
        return roster.with_columns(
            *[pl.lit(None, dtype=pl.Float64).alias(c) for c in _SNAP_OUTPUTS[:-1]],
            pl.lit(0, dtype=pl.Int64).alias("snap_games_observed"),
            pl.lit(None, dtype=pl.Utf8).alias("prior_team_id"),
            pl.lit(0, dtype=pl.Int64).alias("snap_team_count"),
            pl.lit(False).alias("changed_team_since_snap_history"),
        )

    joined = roster.join(
        priors,
        left_on=roster_pfr,
        right_on="pfr_player_id",
        how="left",
        suffix="_snap",
    ).with_columns(
        pl.col("snap_games_observed").fill_null(0),
        pl.col("snap_team_count").fill_null(0),
    )
    return joined.with_columns(
        (
            (pl.col("snap_games_observed") > 0)
            & pl.col("prior_team_id").is_not_null()
            & (pl.col("team_id") != pl.col("prior_team_id"))
        ).alias("changed_team_since_snap_history")
    )


def build_league_personnel_snapshot(
    current_rosters: pl.DataFrame,
    players: pl.DataFrame,
    historical_snap_counts: pl.DataFrame,
    current_snap_counts: pl.DataFrame | None = None,
    *,
    recent_games: int = 6,
) -> pl.DataFrame:
    """Compile the persistent all-NFL personnel baseline used by weekly slate snapshots."""
    roster = latest_roster_state(current_rosters)
    roster = enrich_roster_identity(roster, players)
    roster = _attach_snap_priors(
        roster,
        historical_snap_counts,
        current_snap_counts,
        recent_games,
    )
    validate_league_coverage(roster)
    return roster.sort(["team_id", "position", "gsis_id"])


def validate_league_coverage(snapshot: pl.DataFrame) -> None:
    actual = set(snapshot.get_column("team_id").drop_nulls().unique().to_list())
    expected = set(NFL_TEAMS)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(f"League personnel universe mismatch: missing={missing} extra={extra}")


def league_coverage_report(snapshot: pl.DataFrame) -> pl.DataFrame:
    """Make missing-data coverage visible instead of silently converting missing to bad."""
    expressions: list[pl.Expr] = [
        pl.len().alias("roster_players"),
        (pl.col("snap_games_observed") > 0).sum().alias("players_with_snap_prior"),
        pl.col("changed_team_since_snap_history").sum().alias("players_changed_team"),
    ]
    if "pfr_id_recovered_from_snap_name" in snapshot.columns:
        expressions.append(
            pl.col("pfr_id_recovered_from_snap_name").sum().alias("pfr_ids_recovered_from_snap_name")
        )
    for column, alias in [
        ("height", "players_with_height"),
        ("weight", "players_with_weight"),
        ("birth_date", "players_with_birth_date"),
        ("pfr_id", "players_with_pfr_id"),
    ]:
        if column in snapshot.columns:
            expressions.append(pl.col(column).is_not_null().sum().alias(alias))
    return snapshot.group_by("team_id").agg(*expressions).sort("team_id")


def assert_all_teams_present(team_ids: Iterable[str]) -> None:
    """Small reusable quality gate for artifacts and database promotion jobs."""
    actual = set(team_ids)
    if actual != set(NFL_TEAMS):
        raise ValueError(f"Expected 32-team league universe, received {len(actual)} teams")
