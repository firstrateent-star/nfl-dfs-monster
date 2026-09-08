from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from monster.feature_compile.league_units import compile_league_unit_effects


def _read(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path) if path.suffix == ".parquet" else pl.read_csv(path)


def _team_health_summary(personnel: pl.DataFrame) -> pl.DataFrame:
    required = {
        "team_id",
        "position",
        "projected_offense_snap_share",
        "projected_defense_snap_share",
        "projected_special_teams_snap_share",
    }
    missing = required.difference(personnel.columns)
    if missing:
        raise ValueError(f"Health propagation audit missing personnel columns: {sorted(missing)}")

    avail = (
        pl.col("health_availability_probability").fill_null(1.0)
        if "health_availability_probability" in personnel.columns
        else pl.lit(1.0)
    )
    eff = (
        pl.col("health_effectiveness_if_active").fill_null(1.0)
        if "health_effectiveness_if_active" in personnel.columns
        else pl.lit(1.0)
    )
    uncertainty = (
        pl.col("health_uncertainty").fill_null(0.0)
        if "health_uncertainty" in personnel.columns
        else pl.lit(0.0)
    )
    evidence = (
        (pl.col("health_evidence").fill_null("roster_only") != "roster_only")
        if "health_evidence" in personnel.columns
        else pl.lit(False)
    )
    concern = (avail < 0.95) | (eff < 0.98) | (uncertainty > 0.12)
    health_factor = (avail * eff).clip(0.0, 1.0)

    frame = personnel.with_columns(
        health_factor.alias("_health_factor"),
        (1.0 - health_factor).alias("_health_loss"),
        evidence.alias("_has_health_evidence"),
        concern.alias("_health_concern"),
    )
    return (
        frame.group_by("team_id")
        .agg(
            pl.len().alias("roster_players"),
            pl.col("_has_health_evidence").sum().alias("players_with_health_evidence"),
            pl.col("_health_concern").sum().alias("players_with_health_concern"),
            (
                pl.col("projected_offense_snap_share") * pl.col("_health_loss")
            ).sum().alias("expected_offense_health_loss_mass"),
            (
                pl.col("projected_defense_snap_share") * pl.col("_health_loss")
            ).sum().alias("expected_defense_health_loss_mass"),
            (
                pl.col("projected_special_teams_snap_share") * pl.col("_health_loss")
            ).sum().alias("expected_special_teams_health_loss_mass"),
            pl.col("position")
            .filter(pl.col("_health_concern"))
            .cast(pl.Utf8)
            .unique()
            .sort()
            .implode()
            .alias("concern_positions"),
            uncertainty.mean().alias("mean_health_uncertainty"),
        )
        .sort("team_id")
    )


def _unit_delta(control: pl.DataFrame, health: pl.DataFrame) -> pl.DataFrame:
    c = compile_league_unit_effects(control).select(
        "team_id",
        "pass_protection_effect",
        "run_block_effect",
        "pass_rush_effect",
        "coverage_effect",
        "run_defense_effect",
        "special_teams_effect",
        "personnel_uncertainty",
    )
    h = compile_league_unit_effects(health).select(
        "team_id",
        "pass_protection_effect",
        "run_block_effect",
        "pass_rush_effect",
        "coverage_effect",
        "run_defense_effect",
        "special_teams_effect",
        "personnel_uncertainty",
    )
    joined = c.join(h, on="team_id", how="inner", suffix="_health")
    metrics = [
        "pass_protection_effect",
        "run_block_effect",
        "pass_rush_effect",
        "coverage_effect",
        "run_defense_effect",
        "special_teams_effect",
        "personnel_uncertainty",
    ]
    return joined.select(
        "team_id",
        *[
            (pl.col(f"{m}_health") - pl.col(m)).alias(f"delta_{m}")
            for m in metrics
        ],
    )


def _score_reach(game_ab: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict] = []
    for row in game_ab.to_dicts():
        game = str(row["game"])
        away, home = game.split("@", 1)
        rows.append(
            {
                "team_id": away,
                "game": game,
                "delta_team_points_mean": float(row["delta_away_mean"]),
                "delta_game_total_mean": float(row["delta_total_mean"]),
            }
        )
        rows.append(
            {
                "team_id": home,
                "game": game,
                "delta_team_points_mean": float(row["delta_home_mean"]),
                "delta_game_total_mean": float(row["delta_total_mean"]),
            }
        )
    return pl.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-personnel", type=Path, required=True)
    parser.add_argument("--health-personnel", type=Path, required=True)
    parser.add_argument("--game-health-ab", type=Path, required=True)
    parser.add_argument("--player-health-ab", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/health-propagation-audit"))
    args = parser.parse_args()

    control = _read(args.control_personnel)
    health = _read(args.health_personnel)
    game_ab = _read(args.game_health_ab)
    player_ab = _read(args.player_health_ab)

    summary = _team_health_summary(health)
    unit_delta = _unit_delta(control, health)
    score_reach = _score_reach(game_ab)
    player_reach = (
        player_ab.group_by("team_id")
        .agg(
            pl.col("combined_absolute_yardage_shift").max().alias("max_player_yardage_shift"),
            pl.col("combined_absolute_yardage_shift").sum().alias("sum_player_yardage_shift"),
        )
    )

    audit = (
        summary.join(unit_delta, on="team_id", how="left")
        .join(score_reach, on="team_id", how="left")
        .join(player_reach, on="team_id", how="left")
        .with_columns(
            pl.max_horizontal(
                pl.col("delta_pass_protection_effect").abs(),
                pl.col("delta_run_block_effect").abs(),
                pl.col("delta_pass_rush_effect").abs(),
                pl.col("delta_coverage_effect").abs(),
                pl.col("delta_run_defense_effect").abs(),
                pl.col("delta_special_teams_effect").abs(),
            ).alias("max_abs_unit_effect_shift"),
            pl.col("delta_team_points_mean").abs().alias("abs_team_points_shift"),
        )
        .sort("expected_offense_health_loss_mass", descending=True)
    )

    concerned = audit.filter(pl.col("players_with_health_concern") > 0)
    max_score = float(audit.select(pl.col("abs_team_points_shift").max()).item() or 0.0)
    max_unit = float(audit.select(pl.col("max_abs_unit_effect_shift").max()).item() or 0.0)
    max_player = float(audit.select(pl.col("max_player_yardage_shift").max()).item() or 0.0)
    concern_teams = int(concerned.height)
    low_score_reach = int(
        concerned.filter(pl.col("abs_team_points_shift") < 0.10).height
    ) if concern_teams else 0

    manifest = {
        "artifact": "Monster Health Propagation Audit",
        "purpose": "Measure whether current health evidence reaches game scoring, unit matchup capability, and player allocation before redesigning the health state.",
        "teams_with_health_concerns": concern_teams,
        "concern_teams_with_under_0_10_point_team_mean_shift": low_score_reach,
        "max_abs_team_points_mean_shift": max_score,
        "max_abs_unit_effect_shift": max_unit,
        "max_player_combined_yardage_shift": max_player,
        "diagnostic": (
            "health currently reaches player allocation more strongly than team scoring"
            if max_player > 5.0 and max_score < 0.25
            else "health propagation requires team-level review"
        ),
        "next_candidate": "sampled health reality with replacement-chain unit state and explicit team-capability propagation",
        "market_blind": True,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    audit.write_csv(args.out / "team_health_propagation.csv")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(audit)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
