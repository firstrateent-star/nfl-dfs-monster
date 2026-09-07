from __future__ import annotations

import polars as pl

# Roster-status meanings follow nflverse's roster dictionary. These are starting
# participation priors, not talent ratings. ACT is on the active roster; DEV is
# practice squad; RES/EXE/CUT/RET are not ordinary game-day participants.
_STATUS_ACTIVE_PROB = {
    "ACT": 0.91,
    "DEV": 0.04,
    "RES": 0.01,
    "EXE": 0.0,
    "CUT": 0.0,
    "RET": 0.0,
}

# Conditional field-time priors used only when recent snap history is absent.
# They are intentionally conservative and are later refined by depth/injury/news.
_OFFENSE_DEFAULT = {
    "QB": 0.32,
    "RB": 0.24,
    "WR": 0.36,
    "TE": 0.30,
    "OL": 0.58,
}
_DEFENSE_DEFAULT = {
    "DL": 0.30,
    "LB": 0.34,
    "DB": 0.38,
}
_SPECIAL_DEFAULT = {
    "SPEC": 0.72,
    "DB": 0.16,
    "LB": 0.20,
    "WR": 0.08,
    "RB": 0.10,
    "TE": 0.10,
    "DL": 0.06,
    "OL": 0.05,
}


def _map_float(column: str, mapping: dict[str, float], default: float = 0.0) -> pl.Expr:
    return pl.col(column).cast(pl.Utf8).replace_strict(mapping, default=default).cast(pl.Float64)


def _conditional_share(prior: str, group: str, defaults: dict[str, float]) -> pl.Expr:
    return (
        pl.when(pl.col(prior).is_not_null())
        .then(pl.col(prior).cast(pl.Float64).clip(0.0, 1.0))
        .otherwise(_map_float(group, defaults))
    )


def _conserve_unit(frame: pl.DataFrame, raw_column: str, output_column: str) -> pl.DataFrame:
    """Rescale independent player priors toward 11 player-equivalents per team.

    Football constrains every ordinary unit to 11 players on the field. Independent
    player priors can sum above/below that because of transfers and missing history.
    Conservation redistributes role mass within the team instead of mistaking source
    coverage for extra/missing players. Individual expected shares remain capped <1.
    """
    if "team_id" not in frame.columns:
        return frame.with_columns(pl.col(raw_column).alias(output_column))
    total = pl.col(raw_column).sum().over("team_id").clip(lower_bound=0.01)
    return frame.with_columns(
        (pl.col(raw_column) * (11.0 / total)).clip(0.0, 0.995).alias(output_column)
    )


def infer_game_day_participation(snapshot: pl.DataFrame, season: int) -> pl.DataFrame:
    """Infer expected game-day participation for every league roster row.

    The output separates three ideas that should not be compressed too early:
    1. probability the player is available/active for the game;
    2. conditional snap share if active;
    3. uncertainty about that role.

    Raw expected influence is P(active) × conditional snap share. When team identity
    is present, a conservation pass then rescales each unit toward the structural
    11-player constraint. Raw shares are retained for audit.
    """
    required = {
        "status",
        "position_group",
        "snap_games_observed",
        "offense_snap_share",
        "defense_snap_share",
        "special_teams_snap_share",
        "changed_team_since_snap_history",
    }
    missing = required.difference(snapshot.columns)
    if missing:
        raise ValueError(f"Participation inference missing columns: {sorted(missing)}")

    frame = snapshot.with_columns(
        _map_float("status", _STATUS_ACTIVE_PROB).alias("game_day_active_probability"),
        _conditional_share("offense_snap_share", "position_group", _OFFENSE_DEFAULT).alias(
            "conditional_offense_snap_share"
        ),
        _conditional_share("defense_snap_share", "position_group", _DEFENSE_DEFAULT).alias(
            "conditional_defense_snap_share"
        ),
        _conditional_share(
            "special_teams_snap_share", "position_group", _SPECIAL_DEFAULT
        ).alias("conditional_special_teams_snap_share"),
    )

    rookie_expr = (
        pl.when(pl.col("rookie_year").is_not_null())
        .then(pl.col("rookie_year").cast(pl.Int64) >= season)
        .otherwise(False)
    )
    no_history = pl.col("snap_games_observed") == 0
    changed_team = pl.col("changed_team_since_snap_history").fill_null(False)
    practice_squad = pl.col("status") == "DEV"

    frame = frame.with_columns(
        (
            0.08
            + pl.when(no_history).then(0.20).otherwise(0.0)
            + pl.when(changed_team).then(0.15).otherwise(0.0)
            + pl.when(rookie_expr).then(0.15).otherwise(0.0)
            + pl.when(practice_squad).then(0.15).otherwise(0.0)
            + pl.col("snap_share_uncertainty").fill_null(0.0).clip(0.0, 0.30)
        )
        .clip(0.05, 0.60)
        .alias("participation_uncertainty")
    )

    frame = frame.with_columns(
        (
            pl.col("game_day_active_probability") * pl.col("conditional_offense_snap_share")
        ).alias("raw_projected_offense_snap_share"),
        (
            pl.col("game_day_active_probability") * pl.col("conditional_defense_snap_share")
        ).alias("raw_projected_defense_snap_share"),
        (
            pl.col("game_day_active_probability")
            * pl.col("conditional_special_teams_snap_share")
        ).alias("raw_projected_special_teams_snap_share"),
    )
    frame = _conserve_unit(
        frame, "raw_projected_offense_snap_share", "projected_offense_snap_share"
    )
    frame = _conserve_unit(
        frame, "raw_projected_defense_snap_share", "projected_defense_snap_share"
    )
    frame = _conserve_unit(
        frame, "raw_projected_special_teams_snap_share", "projected_special_teams_snap_share"
    )
    frame = frame.with_columns(
        pl.max_horizontal(
            "projected_offense_snap_share",
            "projected_defense_snap_share",
            "projected_special_teams_snap_share",
        ).alias("projected_primary_snap_share")
    )

    return frame.with_columns(
        pl.when(pl.col("projected_primary_snap_share") >= 0.60)
        .then(pl.lit("core"))
        .when(pl.col("projected_primary_snap_share") >= 0.25)
        .then(pl.lit("rotation"))
        .when(pl.col("projected_primary_snap_share") >= 0.05)
        .then(pl.lit("fringe"))
        .otherwise(pl.lit("background"))
        .alias("participation_tier"),
        pl.when(pl.col("snap_games_observed") > 0)
        .then(pl.lit("recent_snaps"))
        .when(rookie_expr)
        .then(pl.lit("rookie_prior"))
        .otherwise(pl.lit("roster_prior"))
        .alias("participation_evidence"),
    )


def participation_coverage_report(snapshot: pl.DataFrame) -> pl.DataFrame:
    """Audit how many players actually influence each team after participation inference."""
    return (
        snapshot.group_by("team_id")
        .agg(
            (pl.col("status") == "ACT").sum().alias("active_roster_rows"),
            (pl.col("participation_tier") == "core").sum().alias("core_players"),
            (pl.col("participation_tier") == "rotation").sum().alias("rotation_players"),
            (pl.col("participation_tier") == "fringe").sum().alias("fringe_players"),
            (pl.col("participation_tier") == "background").sum().alias("background_players"),
            pl.col("projected_offense_snap_share").sum().alias("offense_snap_equivalents"),
            pl.col("projected_defense_snap_share").sum().alias("defense_snap_equivalents"),
            pl.col("projected_special_teams_snap_share")
            .sum()
            .alias("special_teams_snap_equivalents"),
            pl.col("participation_uncertainty").mean().alias("mean_participation_uncertainty"),
        )
        .sort("team_id")
    )
