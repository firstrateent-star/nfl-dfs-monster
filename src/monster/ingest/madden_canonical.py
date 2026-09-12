from __future__ import annotations

import polars as pl

from monster.ingest.madden_schema import normalize_madden_name


# The public extraction mirrors EA's player-detail attributes but uses the historical
# column names consumed by the older adapters.  Convert those columns once at the source
# boundary so every downstream mechanism sees one canonical madden_* vocabulary.
_MIRROR_TO_CANONICAL = {
    "overall": "madden_overall",
    "speed_rating": "madden_speed",
    "accel_rating": "madden_acceleration",
    "agility_rating": "madden_agility",
    "awareness_rating": "madden_awareness",
    "strength_rating": "madden_strength",
    "catch_rating": "madden_catching",
    "carry_rating": "madden_carrying",
    "throw_power_rating": "madden_throw_power",
    "kick_power_rating": "madden_kick_power",
    "kick_acc_rating": "madden_kick_accuracy",
    "run_block_rating": "madden_run_block",
    "pass_block_rating": "madden_pass_block",
    "tackle_rating": "madden_tackle",
    "jump_rating": "madden_jumping",
    "kick_ret_rating": "madden_kick_return",
    "truck_rating": "madden_trucking",
    "change_of_direction_rating": "madden_change_of_direction",
    "stiff_arm_rating": "madden_stiff_arm",
    "spin_move_rating": "madden_spin_move",
    "juke_move_rating": "madden_juke_move",
    "impact_block_rating": "madden_impact_blocking",
    "run_block_power_rating": "madden_run_block_power",
    "run_block_finesse_rating": "madden_run_block_finesse",
    "pass_block_power_rating": "madden_pass_block_power",
    "pass_block_finesse_rating": "madden_pass_block_finesse",
    "lead_block_rating": "madden_lead_block",
    "throw_acc_short_rating": "madden_throw_accuracy_short",
    "throw_acc_mid_rating": "madden_throw_accuracy_mid",
    "throw_acc_deep_rating": "madden_throw_accuracy_deep",
    "throw_on_run_rating": "madden_throw_on_run",
    "play_action_rating": "madden_play_action",
    "throw_under_pressure_rating": "madden_throw_under_pressure",
    "break_sack_rating": "madden_break_sack",
    "break_tackle_rating": "madden_break_tackle",
    "spec_catch_rating": "madden_spectacular_catch",
    "cit_rating": "madden_catch_in_traffic",
    "route_run_short_rating": "madden_short_route_running",
    "route_run_med_rating": "madden_medium_route_running",
    "route_run_deep_rating": "madden_deep_route_running",
    "release_rating": "madden_release",
    "power_moves_rating": "madden_power_moves",
    "finesse_moves_rating": "madden_finesse_moves",
    "block_shed_rating": "madden_block_shedding",
    "pursuit_rating": "madden_pursuit",
    "play_rec_rating": "madden_play_recognition",
    "man_cover_rating": "madden_man_coverage",
    "zone_cover_rating": "madden_zone_coverage",
    "press_rating": "madden_press",
    "hit_power_rating": "madden_hit_power",
    "stamina_rating": "madden_stamina",
    "injury_rating": "madden_injury",
    "tough_rating": "madden_toughness",
    "bcv_rating": "madden_ball_carrier_vision",
}

_IDENTITY_ALIASES = {
    "player_id": "madden_player_id",
    "full_name": "madden_player_name",
    "team_name": "madden_team",
    "position": "madden_position",
    "archetype": "madden_archetype",
    "iteration": "madden_iteration",
    "handedness": "madden_handedness",
    "x_factor": "madden_x_factor",
    "running_style": "madden_running_style",
    "ability_1": "madden_ability_1",
    "ability_2": "madden_ability_2",
    "ability_3": "madden_ability_3",
    "ability_4": "madden_ability_4",
    "ability_5": "madden_ability_5",
    "ability_6": "madden_ability_6",
}


def _mean_present(frame: pl.DataFrame, columns: tuple[str, ...], alias: str) -> pl.Expr | None:
    present = [column for column in columns if column in frame.columns]
    if not present:
        return None
    return pl.mean_horizontal(*[pl.col(column).cast(pl.Float64, strict=False) for column in present]).alias(alias)


def canonicalize_madden_attribute_mirror(ratings: pl.DataFrame) -> pl.DataFrame:
    """Preserve the rich source while exposing canonical football-mechanism attributes."""
    if ratings.is_empty():
        return ratings

    expressions: list[pl.Expr] = []
    for source, target in _IDENTITY_ALIASES.items():
        if target not in ratings.columns and source in ratings.columns:
            expressions.append(pl.col(source).cast(pl.Utf8, strict=False).alias(target))
    for source, target in _MIRROR_TO_CANONICAL.items():
        if target not in ratings.columns and source in ratings.columns:
            expressions.append(pl.col(source).cast(pl.Float64, strict=False).alias(target))
    frame = ratings.with_columns(expressions) if expressions else ratings

    derived: list[pl.Expr] = []
    route = _mean_present(
        frame,
        ("madden_short_route_running", "madden_medium_route_running", "madden_deep_route_running"),
        "madden_route_running",
    )
    accuracy = _mean_present(
        frame,
        ("madden_throw_accuracy_short", "madden_throw_accuracy_mid", "madden_throw_accuracy_deep"),
        "madden_throw_accuracy",
    )
    pass_rush = _mean_present(
        frame,
        ("madden_power_moves", "madden_finesse_moves", "madden_block_shedding", "madden_pursuit", "madden_play_recognition"),
        "madden_pass_rush",
    )
    coverage = _mean_present(
        frame,
        ("madden_man_coverage", "madden_zone_coverage", "madden_press", "madden_play_recognition"),
        "madden_coverage",
    )
    for expr, name in (
        (route, "madden_route_running"),
        (accuracy, "madden_throw_accuracy"),
        (pass_rush, "madden_pass_rush"),
        (coverage, "madden_coverage"),
    ):
        if expr is not None and name not in frame.columns:
            derived.append(expr)

    simple_aliases = {
        "madden_juke": "madden_juke_move",
        "madden_spin": "madden_spin_move",
        "madden_jump": "madden_jumping",
        "madden_return": "madden_kick_return",
    }
    for target, source in simple_aliases.items():
        if target not in frame.columns and source in frame.columns:
            derived.append(pl.col(source).cast(pl.Float64, strict=False).alias(target))

    if "madden_match_name_key" not in frame.columns and "madden_player_name" in frame.columns:
        derived.append(
            pl.col("madden_player_name")
            .map_elements(normalize_madden_name, return_dtype=pl.Utf8)
            .alias("madden_match_name_key")
        )
    if "madden_raw_source_transport" not in frame.columns:
        derived.append(pl.lit("public_attribute_mirror_of_ea_ratings").alias("madden_raw_source_transport"))

    return frame.with_columns(derived) if derived else frame
