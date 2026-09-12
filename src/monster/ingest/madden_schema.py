from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping

import polars as pl

TEAM_NAME_TO_ID = {
    "ARIZONA CARDINALS": "ARI", "ATLANTA FALCONS": "ATL", "BALTIMORE RAVENS": "BAL",
    "BUFFALO BILLS": "BUF", "CAROLINA PANTHERS": "CAR", "CHICAGO BEARS": "CHI",
    "CINCINNATI BENGALS": "CIN", "CLEVELAND BROWNS": "CLE", "DALLAS COWBOYS": "DAL",
    "DENVER BRONCOS": "DEN", "DETROIT LIONS": "DET", "GREEN BAY PACKERS": "GB",
    "HOUSTON TEXANS": "HOU", "INDIANAPOLIS COLTS": "IND", "JACKSONVILLE JAGUARS": "JAC",
    "KANSAS CITY CHIEFS": "KC", "LOS ANGELES CHARGERS": "LAC", "LOS ANGELES RAMS": "LAR",
    "LAS VEGAS RAIDERS": "LV", "MIAMI DOLPHINS": "MIA", "MINNESOTA VIKINGS": "MIN",
    "NEW ENGLAND PATRIOTS": "NE", "NEW ORLEANS SAINTS": "NO", "NEW YORK GIANTS": "NYG",
    "NEW YORK JETS": "NYJ", "PHILADELPHIA EAGLES": "PHI", "PITTSBURGH STEELERS": "PIT",
    "SAN FRANCISCO 49ERS": "SF", "SEATTLE SEAHAWKS": "SEA", "TAMPA BAY BUCCANEERS": "TB",
    "TENNESSEE TITANS": "TEN", "WASHINGTON COMMANDERS": "WAS",
}
TEAM_NICKNAME_TO_ID = {
    "CARDINALS": "ARI", "FALCONS": "ATL", "RAVENS": "BAL", "BILLS": "BUF",
    "PANTHERS": "CAR", "BEARS": "CHI", "BENGALS": "CIN", "BROWNS": "CLE",
    "COWBOYS": "DAL", "BRONCOS": "DEN", "LIONS": "DET", "PACKERS": "GB",
    "TEXANS": "HOU", "COLTS": "IND", "JAGUARS": "JAC", "CHIEFS": "KC",
    "CHARGERS": "LAC", "RAMS": "LAR", "RAIDERS": "LV", "DOLPHINS": "MIA",
    "VIKINGS": "MIN", "PATRIOTS": "NE", "SAINTS": "NO", "GIANTS": "NYG",
    "JETS": "NYJ", "EAGLES": "PHI", "STEELERS": "PIT", "49ERS": "SF",
    "SEAHAWKS": "SEA", "BUCCANEERS": "TB", "TITANS": "TEN", "COMMANDERS": "WAS",
}
TEAM_IDS = set(TEAM_NAME_TO_ID.values())
TEAM_ALIASES = {
    **{team_id: team_id for team_id in TEAM_IDS},
    **TEAM_NAME_TO_ID,
    **TEAM_NICKNAME_TO_ID,
    "JAX": "JAC", "WSH": "WAS", "OAK": "LV", "SD": "LAC", "SDG": "LAC",
    "STL": "LAR", "LA RAMS": "LAR", "LA CHARGERS": "LAC",
}

POSITION_ALIASES = {
    "HB": "RB", "FB": "RB",
    "LT": "OT", "RT": "OT", "T": "OT", "OT": "OT",
    "LG": "OG", "RG": "OG", "G": "OG", "OG": "OG",
    "LE": "DE", "RE": "DE", "DE": "DE",
    "LOLB": "LB", "ROLB": "LB", "OLB": "LB", "MLB": "LB", "ILB": "LB", "LB": "LB",
    "FS": "S", "SS": "S", "S": "S",
}

LEGACY_MADDEN_ALIASES = {
    "full_name": "madden_player_name", "position": "madden_position", "team_name": "madden_team",
    "speed_rating": "madden_speed", "accel_rating": "madden_acceleration", "acceleration_rating": "madden_acceleration",
    "agility_rating": "madden_agility", "awareness_rating": "madden_awareness", "strength_rating": "madden_strength",
    "catch_rating": "madden_catching", "catching_rating": "madden_catching", "carry_rating": "madden_carrying",
    "carrying_rating": "madden_carrying", "throw_power_rating": "madden_throw_power", "kick_power_rating": "madden_kick_power",
    "kick_acc_rating": "madden_kick_accuracy", "kick_accuracy_rating": "madden_kick_accuracy", "run_block_rating": "madden_run_block",
    "pass_block_rating": "madden_pass_block", "tackle_rating": "madden_tackle", "jump_rating": "madden_jumping",
    "kick_ret_rating": "madden_kick_return", "kick_return_rating": "madden_kick_return", "truck_rating": "madden_trucking",
    "change_of_direction_rating": "madden_change_of_direction", "stiff_arm_rating": "madden_stiff_arm", "spin_move_rating": "madden_spin_move",
    "juke_move_rating": "madden_juke_move", "impact_block_rating": "madden_impact_blocking", "run_block_power_rating": "madden_run_block_power",
    "run_block_finesse_rating": "madden_run_block_finesse", "pass_block_power_rating": "madden_pass_block_power",
    "pass_block_finesse_rating": "madden_pass_block_finesse", "throw_acc_short_rating": "madden_throw_accuracy_short",
    "throw_accuracy_short_rating": "madden_throw_accuracy_short", "throw_acc_mid_rating": "madden_throw_accuracy_mid",
    "throw_accuracy_mid_rating": "madden_throw_accuracy_mid", "throw_acc_deep_rating": "madden_throw_accuracy_deep",
    "throw_accuracy_deep_rating": "madden_throw_accuracy_deep", "throw_on_run_rating": "madden_throw_on_run",
    "play_action_rating": "madden_play_action", "throw_under_pressure_rating": "madden_throw_under_pressure",
    "break_sack_rating": "madden_break_sack", "break_tackle_rating": "madden_break_tackle", "spec_catch_rating": "madden_spectacular_catch",
    "spectacular_catch_rating": "madden_spectacular_catch", "cit_rating": "madden_catch_in_traffic",
    "catch_in_traffic_rating": "madden_catch_in_traffic", "route_run_short_rating": "madden_short_route_running",
    "short_route_running_rating": "madden_short_route_running", "route_run_med_rating": "madden_medium_route_running",
    "medium_route_running_rating": "madden_medium_route_running", "route_run_deep_rating": "madden_deep_route_running",
    "deep_route_running_rating": "madden_deep_route_running", "release_rating": "madden_release", "power_moves_rating": "madden_power_moves",
    "finesse_moves_rating": "madden_finesse_moves", "block_shed_rating": "madden_block_shedding",
    "block_shedding_rating": "madden_block_shedding", "pursuit_rating": "madden_pursuit", "play_rec_rating": "madden_play_recognition",
    "play_recognition_rating": "madden_play_recognition", "man_cover_rating": "madden_man_coverage", "man_coverage_rating": "madden_man_coverage",
    "zone_cover_rating": "madden_zone_coverage", "zone_coverage_rating": "madden_zone_coverage", "press_rating": "madden_press",
    "hit_power_rating": "madden_hit_power", "stamina_rating": "madden_stamina", "injury_rating": "madden_injury",
}


def normalize_madden_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"\b(jr|sr|ii|iii|iv|v)\.?\b", "", text, flags=re.IGNORECASE)
    return re.sub(r"[^a-z0-9]", "", text.lower())


def normalize_madden_team(value: object, aliases: Mapping[str, str] | None = None) -> str | None:
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value).strip())
    if not text:
        return None
    if aliases:
        direct = aliases.get(text)
        if direct is None:
            direct = aliases.get(text.upper())
        if direct is not None:
            text = str(direct).strip()
    key = text.upper()
    return TEAM_ALIASES.get(key, key if key in TEAM_IDS else None)


def normalize_madden_position(value: object) -> str | None:
    text = str(value or "").strip().upper()
    if not text:
        return None
    return POSITION_ALIASES.get(text, text)


def legacy_madden_adapter_view(ratings: pl.DataFrame) -> pl.DataFrame:
    """Expose old adapter names from canonical EA fields without altering raw evidence."""
    if not ratings.height:
        return ratings
    expressions = [
        pl.col(source).alias(target)
        for target, source in LEGACY_MADDEN_ALIASES.items()
        if target not in ratings.columns and source in ratings.columns
    ]
    return ratings.with_columns(expressions) if expressions else ratings
