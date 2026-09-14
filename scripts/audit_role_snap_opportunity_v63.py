from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl


def _safe_ratio(numerator: str, denominator: str, alias: str) -> pl.Expr:
    return (
        pl.when(pl.col(denominator) > 0)
        .then(pl.col(numerator).cast(pl.Float64) / pl.col(denominator).cast(pl.Float64))
        .otherwise(0.0)
        .alias(alias)
    )


def build_audit(simulation: Path) -> tuple[pl.DataFrame, dict[str, float | int]]:
    rush_path = simulation / "rushing_role_plan_audit.csv"
    snap_path = simulation / "same_world_snap_participants_v5.csv"
    if not rush_path.exists() or not snap_path.exists():
        raise FileNotFoundError(
            "role/snap audit requires rushing_role_plan_audit.csv and "
            "same_world_snap_participants_v5.csv"
        )

    rush = pl.read_csv(rush_path)
    player_map = rush.select(
        ["game", "team", "player_id", "player", "position"]
    ).unique()
    actor_map = player_map.select(["game", "team", "player_id"]).rename(
        {"player_id": "_offense_actor"}
    )

    snaps = pl.read_csv(
        snap_path,
        columns=[
            "game",
            "world",
            "play_type",
            "passer_id",
            "target_id",
            "rusher_id",
            "offense_participants",
        ],
    ).with_columns(
        pl.coalesce([pl.col("rusher_id"), pl.col("passer_id")]).alias("_offense_actor")
    )
    snaps = snaps.join(actor_map, on=["game", "_offense_actor"], how="left")
    snaps = snaps.filter(pl.col("team").is_not_null())

    team_counts = snaps.group_by(["game", "team"]).agg(
        pl.len().alias("offensive_snaps"),
        (pl.col("play_type") == "run").sum().alias("run_snaps"),
        (pl.col("play_type") == "pass").sum().alias("pass_snaps"),
        ((pl.col("play_type") == "pass") & pl.col("target_id").is_not_null())
        .sum()
        .alias("targeted_passes"),
    )

    participants = (
        snaps.select(["game", "team", "play_type", "offense_participants"])
        .with_columns(
            pl.col("offense_participants")
            .fill_null("")
            .str.split("|")
            .alias("player_id")
        )
        .explode("player_id")
        .filter(pl.col("player_id") != "")
        .join(player_map, on=["game", "team", "player_id"], how="inner")
    )
    on_field = participants.group_by(
        ["game", "team", "player_id", "player", "position"]
    ).agg(
        pl.len().alias("snap_onfield"),
        (pl.col("play_type") == "run").sum().alias("run_snap_onfield"),
        (pl.col("play_type") == "pass").sum().alias("pass_snap_onfield"),
    )

    rush_selected = (
        snaps.filter((pl.col("play_type") == "run") & pl.col("rusher_id").is_not_null())
        .group_by(["game", "team", "rusher_id"])
        .agg(pl.len().alias("designed_rushes_selected"))
        .rename({"rusher_id": "player_id"})
    )
    target_selected = (
        snaps.filter(pl.col("target_id").is_not_null())
        .group_by(["game", "team", "target_id"])
        .agg(pl.len().alias("targets_selected"))
        .rename({"target_id": "player_id"})
    )

    audit = (
        rush.rename({"plan_share_mean": "rush_plan_share_mean"})
        .join(team_counts, on=["game", "team"], how="left")
        .join(
            on_field.select(
                [
                    "game",
                    "team",
                    "player_id",
                    "snap_onfield",
                    "run_snap_onfield",
                    "pass_snap_onfield",
                ]
            ),
            on=["game", "team", "player_id"],
            how="left",
        )
        .join(rush_selected, on=["game", "team", "player_id"], how="left")
        .join(target_selected, on=["game", "team", "player_id"], how="left")
        .with_columns(
            pl.col("snap_onfield").fill_null(0),
            pl.col("run_snap_onfield").fill_null(0),
            pl.col("pass_snap_onfield").fill_null(0),
            pl.col("designed_rushes_selected").fill_null(0),
            pl.col("targets_selected").fill_null(0),
        )
        .with_columns(
            _safe_ratio("snap_onfield", "offensive_snaps", "snap_participation_rate"),
            _safe_ratio("run_snap_onfield", "run_snaps", "run_snap_participation_rate"),
            _safe_ratio("pass_snap_onfield", "pass_snaps", "pass_snap_participation_rate"),
            _safe_ratio(
                "designed_rushes_selected", "run_snaps", "actual_designed_rush_share"
            ),
            _safe_ratio(
                "designed_rushes_selected",
                "run_snap_onfield",
                "carry_given_run_snap_presence",
            ),
            _safe_ratio("targets_selected", "targeted_passes", "actual_target_share"),
            _safe_ratio(
                "targets_selected",
                "pass_snap_onfield",
                "target_given_pass_snap_presence",
            ),
        )
        .with_columns(
            (pl.col("rush_plan_share_mean") - pl.col("actual_designed_rush_share"))
            .abs()
            .alias("rush_plan_actual_gap_abs")
        )
    )

    target_path = simulation / "target_role_plan_audit_v63.csv"
    if target_path.exists():
        target = pl.read_csv(target_path).select(
            [
                "team",
                "player_id",
                "target_plan_share_mean",
                "target_plan_share_p50",
                "target_plan_share_p90",
                "target_plan_participation_probability",
            ]
        )
        audit = audit.join(target, on=["team", "player_id"], how="left").with_columns(
            pl.col("target_plan_share_mean").fill_null(0.0),
            (
                pl.col("target_plan_share_mean").fill_null(0.0)
                - pl.col("actual_target_share")
            )
            .abs()
            .alias("target_plan_actual_gap_abs"),
        )

    core_rb = audit.filter(
        pl.col("position").is_in(["RB", "FB"])
        & (pl.col("rush_plan_share_mean") >= 0.08)
        & (pl.col("plan_participation_probability") >= 0.50)
    )
    summary: dict[str, float | int] = {
        "rows": audit.height,
        "core_rb_rows": core_rb.height,
        "core_rb_rush_plan_actual_mae": float(
            core_rb.get_column("rush_plan_actual_gap_abs").mean() or 0.0
        ),
        "core_rb_run_snap_participation_mean": float(
            core_rb.get_column("run_snap_participation_rate").mean() or 0.0
        ),
        "core_rb_carry_given_present_mean": float(
            core_rb.get_column("carry_given_run_snap_presence").mean() or 0.0
        ),
    }
    if "target_plan_actual_gap_abs" in audit.columns:
        target_core = audit.filter(
            pl.col("position").is_in(["RB", "WR", "TE"])
            & (pl.col("target_plan_share_mean") >= 0.03)
        )
        summary["target_role_rows"] = target_core.height
        summary["target_plan_actual_mae"] = float(
            target_core.get_column("target_plan_actual_gap_abs").mean() or 0.0
        )

    return audit, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulation", type=Path, required=True)
    args = parser.parse_args()

    audit, summary = build_audit(args.simulation)
    audit.sort(
        ["game", "team", "rush_plan_share_mean"], descending=[False, False, True]
    ).write_csv(args.simulation / "role_snap_opportunity_audit_v63.csv")
    (args.simulation / "role_snap_opportunity_summary_v63.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
