from __future__ import annotations

import math
import re
import time
import unicodedata

import polars as pl
import requests

EA_RATINGS_URL = "https://www.ea.com/games/madden-nfl/ratings"
EA_BASE_URL = "https://www.ea.com"
EA_USER_AGENT = "Mozilla/5.0 (compatible; MonsterFootballReality/1.0)"
_BUILD_ID_PATTERN = re.compile(r"/_next/static/([^/]+)/_buildManifest\.js")


EA_STAT_ALIASES = {
    "speedRating": "madden_speed",
    "accelerationRating": "madden_acceleration",
    "agilityRating": "madden_agility",
    "awarenessRating": "madden_awareness",
    "strengthRating": "madden_strength",
    "catchingRating": "madden_catching",
    "carryRating": "madden_carrying",
    "throwPowerRating": "madden_throw_power",
    "kickPowerRating": "madden_kick_power",
    "kickAccuracyRating": "madden_kick_accuracy",
    "runBlockRating": "madden_run_block",
    "passBlockRating": "madden_pass_block",
    "tackleRating": "madden_tackle",
    "jumpingRating": "madden_jumping",
    "kickReturnRating": "madden_kick_return",
    "truckingRating": "madden_trucking",
    "changeOfDirectionRating": "madden_change_of_direction",
    "stiffArmRating": "madden_stiff_arm",
    "spinMoveRating": "madden_spin_move",
    "jukeMoveRating": "madden_juke_move",
    "impactBlockingRating": "madden_impact_blocking",
    "runBlockPowerRating": "madden_run_block_power",
    "runBlockFinesseRating": "madden_run_block_finesse",
    "passBlockPowerRating": "madden_pass_block_power",
    "passBlockFinesseRating": "madden_pass_block_finesse",
    "leadBlockRating": "madden_lead_block",
    "throwAccuracyShortRating": "madden_throw_accuracy_short",
    "throwAccuracyMidRating": "madden_throw_accuracy_mid",
    "throwAccuracyDeepRating": "madden_throw_accuracy_deep",
    "throwOnTheRunRating": "madden_throw_on_run",
    "playActionRating": "madden_play_action",
    "throwUnderPressureRating": "madden_throw_under_pressure",
    "breakSackRating": "madden_break_sack",
    "breakTackleRating": "madden_break_tackle",
    "spectacularCatchRating": "madden_spectacular_catch",
    "catchInTrafficRating": "madden_catch_in_traffic",
    "shortRouteRunningRating": "madden_short_route_running",
    "mediumRouteRunningRating": "madden_medium_route_running",
    "deepRouteRunningRating": "madden_deep_route_running",
    "releaseRating": "madden_release",
    "powerMovesRating": "madden_power_moves",
    "finesseMovesRating": "madden_finesse_moves",
    "blockSheddingRating": "madden_block_shedding",
    "pursuitRating": "madden_pursuit",
    "playRecognitionRating": "madden_play_recognition",
    "manCoverageRating": "madden_man_coverage",
    "zoneCoverageRating": "madden_zone_coverage",
    "pressRating": "madden_press",
    "hitPowerRating": "madden_hit_power",
    "staminaRating": "madden_stamina",
    "toughnessRating": "madden_toughness",
    "injuryRating": "madden_injury",
}


def _snake(value: str) -> str:
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    value = re.sub(r"[^A-Za-z0-9]+", "_", value)
    return value.strip("_").lower()


def _normalize_name(value: str) -> str:
    folded = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    folded = re.sub(r"\b(jr|sr|ii|iii|iv|v)\.?\b", "", folded, flags=re.IGNORECASE)
    return re.sub(r"[^a-z0-9]", "", folded.lower())


def _extract_team(record: dict) -> str | None:
    for key in ("team", "teamId", "teamName", "teamAbbr", "teamShortName"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict):
            for child in ("abbr", "shortName", "displayName", "name"):
                nested = value.get(child)
                if isinstance(nested, str) and nested.strip():
                    return nested.strip()
    return None


def _extract_name(record: dict) -> str:
    for key in ("fullName", "displayName", "name"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    first = str(record.get("firstName") or "").strip()
    last = str(record.get("lastName") or "").strip()
    return " ".join(part for part in (first, last) if part)


def _flatten_stats(record: dict) -> dict[str, object]:
    row: dict[str, object] = {}
    for key, value in record.items():
        if key in {"stats", "statGroups", "attributes", "ratings"}:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            row[f"madden_raw_{_snake(key)}"] = value

    def add_stat(name: object, value: object) -> None:
        if not isinstance(name, str) or not isinstance(value, (int, float)):
            return
        column = EA_STAT_ALIASES.get(name, f"madden_{_snake(name.removesuffix('Rating'))}")
        row[column] = float(value)

    def visit(value: object) -> None:
        if isinstance(value, dict):
            name = value.get("name") or value.get("statName") or value.get("key") or value.get("slug")
            rating = value.get("value")
            if rating is None:
                rating = value.get("rating")
            add_stat(name, rating)
            for key, child in value.items():
                if isinstance(child, (int, float)) and (key.endswith("Rating") or key in EA_STAT_ALIASES):
                    add_stat(key, child)
                elif isinstance(child, (dict, list)):
                    visit(child)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(record)
    return row


def _flatten_player(record: dict) -> dict[str, object]:
    row = _flatten_stats(record)
    row["madden_player_id"] = record.get("id") or record.get("playerId") or record.get("uuid")
    row["madden_player_name"] = _extract_name(record)
    row["madden_team"] = _extract_team(record)
    row["madden_position"] = record.get("position") or record.get("positionLabel")
    row["madden_archetype"] = record.get("archetype")
    row["madden_iteration"] = record.get("iteration")
    row["madden_handedness"] = record.get("handedness") or record.get("throws")
    row["madden_x_factor"] = record.get("xFactor") or record.get("xfactor")
    row["madden_running_style"] = record.get("runningStyle")
    for index in range(1, 7):
        row[f"madden_ability_{index}"] = record.get(f"ability{index}")
    row["madden_match_name_key"] = _normalize_name(str(row["madden_player_name"] or ""))
    return row


def _build_id(session: requests.Session, timeout: float) -> str:
    response = session.get(EA_RATINGS_URL, timeout=timeout)
    response.raise_for_status()
    match = _BUILD_ID_PATTERN.search(response.text)
    if match is None:
        raise ValueError("EA ratings page did not expose a Next.js build id")
    return match.group(1)


def load_official_madden27_player_ratings(
    *,
    timeout: float = 30.0,
    page_delay_seconds: float = 0.05,
) -> pl.DataFrame:
    session = requests.Session()
    session.headers.update({"User-Agent": EA_USER_AGENT, "Accept": "application/json,text/html"})
    build_id = _build_id(session, timeout=timeout)
    url = f"{EA_BASE_URL}/_next/data/{build_id}/games/madden-nfl/ratings.json?page={{page}}"

    first = session.get(url.format(page=1), timeout=timeout)
    first.raise_for_status()
    rating_details = first.json()["pageProps"]["ratingDetails"]
    total = int(rating_details["totalItems"])
    pages = math.ceil(total / 100.0)
    players = list(rating_details["items"])

    for page in range(2, pages + 1):
        if page_delay_seconds > 0:
            time.sleep(page_delay_seconds)
        response = session.get(url.format(page=page), timeout=timeout)
        response.raise_for_status()
        players.extend(response.json()["pageProps"]["ratingDetails"]["items"])

    rows = [_flatten_player(record) for record in players]
    if not rows:
        raise ValueError("EA Madden ratings endpoint returned no player rows")
    frame = pl.DataFrame(rows, infer_schema_length=None)
    return frame.unique(subset=["madden_player_id"], keep="first")


def _candidate_team_columns(frame: pl.DataFrame) -> tuple[str, ...]:
    return tuple(
        column
        for column in ("team_id", "team", "club", "recent_team")
        if column in frame.columns
    )


def _candidate_name_columns(frame: pl.DataFrame) -> tuple[str, ...]:
    return tuple(
        column
        for column in ("display_name", "full_name", "player_name", "football_name", "name")
        if column in frame.columns
    )


def attach_all_madden_attributes(
    personnel: pl.DataFrame,
    ratings: pl.DataFrame,
    *,
    team_aliases: dict[str, str] | None = None,
) -> pl.DataFrame:
    if personnel.is_empty() or ratings.is_empty():
        return personnel
    team_aliases = team_aliases or {}
    team_columns = _candidate_team_columns(personnel)
    name_columns = _candidate_name_columns(personnel)
    if not name_columns:
        return personnel

    name_column = name_columns[0]
    team_column = team_columns[0] if team_columns else None
    personnel_work = personnel.with_columns(
        pl.col(name_column)
        .cast(pl.String)
        .fill_null("")
        .map_elements(_normalize_name, return_dtype=pl.String)
        .alias("_madden_name_key")
    )
    ratings_work = ratings
    if team_column is not None and "madden_team" in ratings_work.columns:
        ratings_work = ratings_work.with_columns(
            pl.col("madden_team")
            .cast(pl.String)
            .replace(team_aliases)
            .alias("_madden_team_key")
        )
        personnel_work = personnel_work.with_columns(
            pl.col(team_column).cast(pl.String).replace(team_aliases).alias("_madden_team_key")
        )
        joined = personnel_work.join(
            ratings_work,
            left_on=["_madden_name_key", "_madden_team_key"],
            right_on=["madden_match_name_key", "_madden_team_key"],
            how="left",
            suffix="_madden_dup",
        )
    else:
        joined = personnel_work.join(
            ratings_work,
            left_on="_madden_name_key",
            right_on="madden_match_name_key",
            how="left",
            suffix="_madden_dup",
        )

    madden_columns = [column for column in ratings.columns if column.startswith("madden_")]
    duplicate_columns = [column for column in joined.columns if column.endswith("_madden_dup")]
    joined = joined.drop([column for column in ("_madden_name_key", "_madden_team_key") if column in joined.columns])
    if duplicate_columns:
        joined = joined.drop(duplicate_columns)
    return joined.select(
        list(personnel.columns)
        + [column for column in madden_columns if column in joined.columns and column not in personnel.columns]
    )


def official_madden_numeric_columns(frame: pl.DataFrame) -> tuple[str, ...]:
    return tuple(
        column
        for column, dtype in frame.schema.items()
        if column.startswith("madden_") and dtype.is_numeric()
    )