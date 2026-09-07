from __future__ import annotations

from typing import Any

import polars as pl


_REPORT_AVAILABILITY = {
    "out": 0.01,
    "doubtful": 0.20,
    "questionable": 0.68,
}
_REPORT_EFFECTIVENESS = {
    "out": 0.88,
    "doubtful": 0.91,
    "questionable": 0.96,
}
_REPORT_UNCERTAINTY = {
    "out": 0.03,
    "doubtful": 0.18,
    "questionable": 0.22,
}
_PRACTICE_AVAILABILITY = {
    "did not participate": 0.72,
    "dnp": 0.72,
    "limited participation": 0.90,
    "limited": 0.90,
    "full participation": 0.985,
    "full": 0.985,
}
_PRACTICE_EFFECTIVENESS = {
    "did not participate": 0.94,
    "dnp": 0.94,
    "limited participation": 0.97,
    "limited": 0.97,
    "full participation": 0.995,
    "full": 0.995,
}
_PRACTICE_UNCERTAINTY = {
    "did not participate": 0.20,
    "dnp": 0.20,
    "limited participation": 0.12,
    "limited": 0.12,
    "full participation": 0.04,
    "full": 0.04,
}


def _text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().lower().replace("_", " ").split())


def _first_existing(frame: pl.DataFrame, candidates: tuple[str, ...]) -> str | None:
    return next((column for column in candidates if column in frame.columns), None)


def _latest_injury_rows(injuries: pl.DataFrame) -> pl.DataFrame:
    if not injuries.height or "gsis_id" not in injuries.columns:
        return pl.DataFrame()
    frame = injuries.filter(pl.col("gsis_id").is_not_null())
    sort_columns: list[str] = []
    for column in ("date_modified", "report_date", "date", "week"):
        if column in frame.columns:
            sort_columns.append(column)
    if sort_columns:
        frame = frame.sort(sort_columns)
    return frame.unique(subset=["gsis_id"], keep="last")


def _provider_health_rows(injuries: pl.DataFrame) -> pl.DataFrame:
    latest = _latest_injury_rows(injuries)
    if not latest.height:
        return pl.DataFrame()

    report_col = _first_existing(latest, ("report_status", "game_status", "status"))
    practice_col = _first_existing(latest, ("practice_status", "practice_participation"))
    injury_col = _first_existing(
        latest,
        ("report_primary_injury", "practice_primary_injury", "primary_injury", "injury"),
    )
    week_col = _first_existing(latest, ("week",))
    date_col = _first_existing(latest, ("date_modified", "report_date", "date"))

    selected = [pl.col("gsis_id")]
    selected.append(
        pl.col(report_col).cast(pl.Utf8).alias("health_report_status")
        if report_col
        else pl.lit(None, dtype=pl.Utf8).alias("health_report_status")
    )
    selected.append(
        pl.col(practice_col).cast(pl.Utf8).alias("health_practice_status")
        if practice_col
        else pl.lit(None, dtype=pl.Utf8).alias("health_practice_status")
    )
    selected.append(
        pl.col(injury_col).cast(pl.Utf8).alias("health_injury")
        if injury_col
        else pl.lit(None, dtype=pl.Utf8).alias("health_injury")
    )
    selected.append(
        pl.col(week_col).cast(pl.Int64, strict=False).alias("health_report_week")
        if week_col
        else pl.lit(None, dtype=pl.Int64).alias("health_report_week")
    )
    selected.append(
        pl.col(date_col).cast(pl.Utf8).alias("health_report_date")
        if date_col
        else pl.lit(None, dtype=pl.Utf8).alias("health_report_date")
    )
    return latest.select(*selected)


def _health_tuple(report_status: Any, practice_status: Any) -> tuple[float, float, float, str, str]:
    report = _text(report_status)
    practice = _text(practice_status)

    if report in _REPORT_AVAILABILITY:
        return (
            _REPORT_AVAILABILITY[report],
            _REPORT_EFFECTIVENESS[report],
            _REPORT_UNCERTAINTY[report],
            report,
            "official_injury_report",
        )
    if practice in _PRACTICE_AVAILABILITY:
        state = "practice_dnp" if practice in {"did not participate", "dnp"} else (
            "practice_limited" if practice in {"limited participation", "limited"} else "practice_full"
        )
        return (
            _PRACTICE_AVAILABILITY[practice],
            _PRACTICE_EFFECTIVENESS[practice],
            _PRACTICE_UNCERTAINTY[practice],
            state,
            "official_practice_report",
        )
    return (1.0, 1.0, 0.08, "no_current_designation", "roster_only")


def _attach_provider_state(snapshot: pl.DataFrame, injuries: pl.DataFrame) -> pl.DataFrame:
    provider = _provider_health_rows(injuries)
    if not provider.height or "gsis_id" not in snapshot.columns:
        return snapshot.with_columns(
            pl.lit(None, dtype=pl.Utf8).alias("health_report_status"),
            pl.lit(None, dtype=pl.Utf8).alias("health_practice_status"),
            pl.lit(None, dtype=pl.Utf8).alias("health_injury"),
            pl.lit(None, dtype=pl.Int64).alias("health_report_week"),
            pl.lit(None, dtype=pl.Utf8).alias("health_report_date"),
        )
    return snapshot.join(provider, on="gsis_id", how="left")


def _apply_override_rows(frame: pl.DataFrame, overrides: pl.DataFrame) -> pl.DataFrame:
    if not overrides.height:
        return frame
    required = {"team_id", "display_name"}
    if not required.issubset(overrides.columns) or not required.issubset(frame.columns):
        return frame

    allowed = [
        "health_availability_probability",
        "health_effectiveness_if_active",
        "health_uncertainty",
        "health_state",
        "health_evidence",
        "health_injury",
    ]
    present = [column for column in allowed if column in overrides.columns]
    if not present:
        return frame

    patch = overrides.select("team_id", "display_name", *present).rename(
        {column: f"{column}_override" for column in present}
    )
    out = frame.join(patch, on=["team_id", "display_name"], how="left")
    expressions: list[pl.Expr] = []
    drops: list[str] = []
    for column in present:
        override = f"{column}_override"
        expressions.append(pl.coalesce([pl.col(override), pl.col(column)]).alias(column))
        drops.append(override)
    return out.with_columns(*expressions).drop(drops)


def attach_health_state(
    snapshot: pl.DataFrame,
    injuries: pl.DataFrame | None = None,
    *,
    overrides: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Attach availability, conditional effectiveness and health uncertainty.

    Health is intentionally separate from role. The availability probability decides
    whether the player can participate in a sampled reality; effectiveness_if_active
    changes only worlds in which the player is active. Missing health evidence is neutral,
    not negative. Optional manual/news overrides are explicit and auditable.
    """
    frame = _attach_provider_state(snapshot, injuries if injuries is not None else pl.DataFrame())

    health = [
        _health_tuple(row.get("health_report_status"), row.get("health_practice_status"))
        for row in frame.select("health_report_status", "health_practice_status").to_dicts()
    ]
    frame = frame.with_columns(
        pl.Series("health_availability_probability", [row[0] for row in health], dtype=pl.Float64),
        pl.Series("health_effectiveness_if_active", [row[1] for row in health], dtype=pl.Float64),
        pl.Series("health_uncertainty", [row[2] for row in health], dtype=pl.Float64),
        pl.Series("health_state", [row[3] for row in health], dtype=pl.Utf8),
        pl.Series("health_evidence", [row[4] for row in health], dtype=pl.Utf8),
    )

    if overrides is not None:
        frame = _apply_override_rows(frame, overrides)

    return frame.with_columns(
        pl.col("health_availability_probability").cast(pl.Float64).clip(0.0, 1.0),
        pl.col("health_effectiveness_if_active").cast(pl.Float64).clip(0.50, 1.0),
        pl.col("health_uncertainty").cast(pl.Float64).clip(0.01, 0.40),
    )


def health_coverage_report(snapshot: pl.DataFrame) -> pl.DataFrame:
    if "team_id" not in snapshot.columns or "health_state" not in snapshot.columns:
        return pl.DataFrame()
    return (
        snapshot.group_by("team_id")
        .agg(
            pl.len().alias("roster_players"),
            (pl.col("health_evidence") != "roster_only").sum().alias("players_with_health_evidence"),
            (pl.col("health_availability_probability") < 0.95).sum().alias("availability_concerns"),
            (pl.col("health_effectiveness_if_active") < 0.98).sum().alias("effectiveness_concerns"),
            pl.col("health_uncertainty").mean().alias("mean_health_uncertainty"),
        )
        .sort("team_id")
    )
