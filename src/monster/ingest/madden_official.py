from __future__ import annotations

import math
import re
import time
import unicodedata
from typing import Any

import polars as pl
import requests

EA_BASE_URL = "https://www.ea.com"
EA_RATINGS_HOME = f"{EA_BASE_URL}/games/madden-nfl/ratings"
EA_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)

TEAM_NAME_TO_ID = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAC",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LAR",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
}

# EA JSON stat key -> stable raw-column name used by the existing Monster Madden adapters.
STAT_MAP = {
    "acceleration": "accel_rating",
    "agility": "agility_rating",
    "awareness": "awareness_rating",
    "bCVision": "bcv_rating",
    "blockShedding": "block_shed_rating",
    "breakSack": "break_sack_rating",
    "breakTackle": "break_tackle_rating",
    "carrying": "carry_rating",
    "catching": "catch_rating",
    "changeOfDirection": "change_of_direction_rating",
    "catchInTraffic": "cit_rating",
    "finesseMoves": "finesse_moves_rating",
    "hitPower": "hit_power_rating",
    "impactBlocking": "impact_block_rating",
    "injury": "injury_rating",
    "jukeMove": "juke_move_rating",
    "jumping": "jump_rating",
    "kickAccuracy": "kick_acc_rating",
    "kickPower": "kick_power_rating",
    "kickReturn": "kick_ret_rating",
    "leadBlock": "lead_block_rating",
    "manCoverage": "man_cover_rating",
    "passBlockFinesse": "pass_block_finesse_rating",
    "passBlockPower": "pass_block_power_rating",
    "passBlock": "pass_block_rating",
    "playAction": "play_action_rating",
    "playRecognition": "play_rec_rating",
    "powerMoves": "power_moves_rating",
    "press": "press_rating",
    "pursuit": "pursuit_rating",
    "release": "release_rating",
    "deepRouteRunning": "route_run_deep_rating",
    "mediumRouteRunning": "route_run_med_rating",
    "shortRouteRunning": "route_run_short_rating",
    "runBlockFinesse": "run_block_finesse_rating",
    "runBlockPower": "run_block_power_rating",
    "runBlock": "run_block_rating",
    "runningStyle": "running_style",
    "spectacularCatch": "spec_catch_rating",
    "speed": "speed_rating",
    "spinMove": "spin_move_rating",
    "stamina": "stamina_rating",
    "stiffArm": "stiff_arm_rating",
    "strength": "strength_rating",
    "tackle": "tackle_rating",
    "throwAccuracyDeep": "throw_acc_deep_rating",
    "throwAccuracyMid": "throw_acc_mid_rating",
    "throwAccuracyShort": "throw_acc_short_rating",
    "throwOnTheRun": "throw_on_run_rating",
    "throwPower": "throw_power_rating",
    "throwUnderPressure": "throw_under_pressure_rating",
    "toughness": "tough_rating",
    "trucking": "truck_rating",
    "zoneCoverage": "zone_cover_rating",
}

# Stable Monster names.  Every numeric Madden rating is retained in the personnel snapshot.
MONSTER_ATTRIBUTE_MAP = {
    "overall": "madden_overall",
    "accel_rating": "madden_acceleration",
    "agility_rating": "madden_agility",
    "awareness_rating": "madden_awareness",
    "bcv_rating": "madden_bc_vision",
    "block_shed_rating": "madden_block_shedding",
    "break_sack_rating": "madden_break_sack",
    "break_tackle_rating": "madden_break_tackle",
    "carry_rating": "madden_carrying",
    "catch_rating": "madden_catching",
    "change_of_direction_rating": "madden_change_of_direction",
    "cit_rating": "madden_catch_in_traffic",
    "finesse_moves_rating": "madden_finesse_moves",
    "hit_power_rating": "madden_hit_power",
    "impact_block_rating": "madden_impact_blocking",
    "injury_rating": "madden_injury",
    "juke_move_rating": "madden_juke_move",
    "jump_rating": "madden_jumping",
    "kick_acc_rating": "madden_kick_accuracy",
    "kick_power_rating": "madden_kick_power",
    "kick_ret_rating": "madden_kick_return",
    "lead_block_rating": "madden_lead_block",
    "man_cover_rating": "madden_man_coverage",
    "pass_block_finesse_rating": "madden_pass_block_finesse",
    "pass_block_power_rating": "madden_pass_block_power",
    "pass_block_rating": "madden_pass_block_raw",
    "play_action_rating": "madden_play_action",
    "play_rec_rating": "madden_play_recognition",
    "power_moves_rating": "madden_power_moves",
    "press_rating": "madden_press",
    "pursuit_rating": "madden_pursuit",
    "release_rating": "madden_release",
    "route_run_deep_rating": "madden_deep_route_running",
    "route_run_med_rating": "madden_medium_route_running",
    "route_run_short_rating": "madden_short_route_running",
    "run_block_finesse_rating": "madden_run_block_finesse",
    "run_block_power_rating": "madden_run_block_power",
    "run_block_rating": "madden_run_block_raw",
    "spec_catch_rating": "madden_spectacular_catch",
    "speed_rating": "madden_speed",
    "spin_move_rating": "madden_spin_move",
    "stamina_rating": "madden_stamina",
    "stiff_arm_rating": "madden_stiff_arm",
    "strength_rating": "madden_strength",
    "tackle_rating": "madden_tackle",
    "throw_acc_deep_rating": "madden_throw_accuracy_deep",
    "throw_acc_mid_rating": "madden_throw_accuracy_mid",
    "throw_acc_short_rating": "madden_throw_accuracy_short",
    "throw_on_run_rating": "madden_throw_on_run",
    "throw_power_rating": "madden_throw_power",
    "throw_under_pressure_rating": "madden_throw_under_pressure",
    "tough_rating": "madden_toughness",
    "truck_rating": "madden_trucking",
    "zone_cover_rating": "madden_zone_coverage",
}


def _normalize_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii")
    text = text.lower().replace("’", "'")
    text = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", text)
    return re.sub(r"[^a-z0-9]", "", text)


def _build_id(session: requests.Session, *, timeout: int) -> str:
    response = session.get(EA_RATINGS_HOME, timeout=timeout)
    response.raise_for_status()
    match = re.search(r"/_next/static/([^/]+)/_buildManifest\.js", response.text)
    if not match:
        raise RuntimeError("Could not find EA Madden ratings Next.js build ID")
    return match.group(1)


def _flatten_player(player: dict[str, Any]) -> dict[str, Any]:
    team_label = str((player.get("team") or {}).get("label") or "")
    abilities = player.get("playerAbilities") or []
    x_factor = ""
    superstar: list[str] = []
    for ability in abilities:
        ability_type = str((ability.get("type") or {}).get("id") or "")
        label = str(ability.get("label") or "")
        if ability_type == "xFactor":
            x_factor = label
        elif ability_type == "superstarAbility" and label:
            superstar.append(label)

    stats = player.get("stats") or {}
    row: dict[str, Any] = {
        "season": "m27-1",
        "player_id": player.get("id"),
        "first_name": player.get("firstName"),
        "last_name": player.get("lastName"),
        "full_name": f"{player.get('firstName', '')} {player.get('lastName', '')}".strip(),
        "position": (player.get("position") or {}).get("id"),
        "team_name": team_label,
        "team_short": team_label.split()[-1] if team_label else "",
        "age": player.get("age"),
        "height_inches": player.get("height"),
        "weight_lbs": player.get("weight"),
        "college": player.get("college"),
        "years_pro": player.get("yearsPro"),
        "jersey_num": player.get("jerseyNum"),
        "overall": player.get("overallRating"),
        "archetype": (player.get("archetype") or {}).get("label"),
        "iteration": (player.get("iteration") or {}).get("label"),
        "handedness": player.get("handedness"),
        "x_factor": x_factor,
        "running_style": (stats.get("runningStyle") or {}).get("value"),
    }
    for index in range(6):
        row[f"ability_{index + 1}"] = superstar[index] if index < len(superstar) else ""
    for ea_name, column in STAT_MAP.items():
        entry = stats.get(ea_name)
        row[column] = entry.get("value") if isinstance(entry, dict) else None
    return row


def load_official_madden27_player_ratings(
    *, timeout: int = 30, page_delay_seconds: float = 0.15
) -> pl.DataFrame:
    """Fetch the current Madden NFL 27 player database directly from EA.

    The ratings page is a Next.js application.  We resolve its current build ID and then
    page the public JSON data route that backs EA's own ratings table.  No DFS or market
    information is introduced here.
    """
    session = requests.Session()
    session.headers.update({"User-Agent": EA_USER_AGENT, "Accept": "application/json,text/html"})
    build_id = _build_id(session, timeout=timeout)
    url = f"{EA_BASE_URL}/_next/data/{build_id}/games/madden-nfl/ratings.json?page={{page}}"

    first = session.get(url.format(page=1), timeout=timeout)
    first.raise_for_status()
    rating_details = first.json()["pageProps"]["ratingDetails"]
    total = int(rating_details["totalItems"])
    pages = int(math.ceil(total / 100.0))
    players = list(rating_details["items"])

    for page in range(2, pages + 1):
        if page_delay_seconds > 0:
            time.sleep(page_delay_seconds)
        response = session.get(url.format(page=page), timeout=timeout)
        response.raise_for_status()
        players.extend(response.json()["pageProps"]["ratingDetails"]["items"])

    unique: dict[str, dict[str, Any]] = {}
    for player in players:
        unique[str(player.get("id"))] = player
    rows = [_flatten_player(player) for player in unique.values()]
    frame = pl.DataFrame(rows, infer_schema_length=None)
    if frame.height < 2000:
        raise RuntimeError(f"EA Madden 27 scrape unexpectedly returned only {frame.height} players")
    return frame


def attach_all_madden_attributes(personnel: pl.DataFrame, ratings: pl.DataFrame) -> pl.DataFrame:
    """Attach every numeric EA rating plus abilities/classification to current personnel.

    Matching is current-team + normalized name first, with a unique-name fallback for
    transactions/name-source disagreement.  Raw pass/run-block ratings use *_raw names so
    the existing bounded OL composites can continue to own madden_pass_block/run_block.
    """
    if not ratings.height:
        return personnel
    name_column = "display_name" if "display_name" in personnel.columns else "full_name"
    current = personnel.with_columns(
        pl.col(name_column)
        .map_elements(_normalize_name, return_dtype=pl.Utf8)
        .alias("_madden_name_key")
    )

    source = ratings.with_columns(
        pl.col("full_name").map_elements(_normalize_name, return_dtype=pl.Utf8).alias("_madden_name_key"),
        pl.col("team_name").replace_strict(TEAM_NAME_TO_ID, default=None).alias("_madden_team_id"),
        pl.col("player_id").cast(pl.Utf8).alias("madden_player_id"),
    )
    expressions = []
    for source_column, destination in MONSTER_ATTRIBUTE_MAP.items():
        if source_column in source.columns:
            expressions.append(pl.col(source_column).cast(pl.Float64, strict=False).alias(destination))
    source = source.with_columns(*expressions)

    keep = [
        "_madden_name_key",
        "_madden_team_id",
        "madden_player_id",
        "archetype",
        "iteration",
        "handedness",
        "x_factor",
        "running_style",
        *[f"ability_{i}" for i in range(1, 7)],
        *[name for name in MONSTER_ATTRIBUTE_MAP.values() if name in source.columns],
    ]
    source = source.select(keep).rename(
        {
            "archetype": "madden_archetype",
            "iteration": "madden_iteration",
            "handedness": "madden_handedness",
            "x_factor": "madden_x_factor",
            "running_style": "madden_running_style",
            **{f"ability_{i}": f"madden_ability_{i}" for i in range(1, 7)},
        }
    )

    value_columns = [c for c in source.columns if c not in {"_madden_name_key", "_madden_team_id"}]
    exact = source.unique(subset=["_madden_name_key", "_madden_team_id"], keep="last")
    out = current.join(
        exact,
        left_on=["_madden_name_key", "team_id"],
        right_on=["_madden_name_key", "_madden_team_id"],
        how="left",
    )

    unique = (
        source.group_by("_madden_name_key")
        .agg(pl.len().alias("_madden_count"), *[pl.col(c).first().alias(f"_fallback_{c}") for c in value_columns])
        .filter(pl.col("_madden_count") == 1)
    )
    out = out.join(unique, on="_madden_name_key", how="left")
    out = out.with_columns(
        *[
            pl.coalesce([pl.col(c), pl.col(f"_fallback_{c}")]).alias(c)
            for c in value_columns
        ],
        pl.when(pl.col("madden_player_id").is_not_null())
        .then(pl.lit("team_name"))
        .when(pl.col("_fallback_madden_player_id").is_not_null())
        .then(pl.lit("unique_name"))
        .otherwise(None)
        .alias("madden_full_match_type"),
    )
    drop = ["_madden_name_key", "_madden_count", *[f"_fallback_{c}" for c in value_columns]]
    return out.drop([c for c in drop if c in out.columns])
