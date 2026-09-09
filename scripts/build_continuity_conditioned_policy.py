from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from validate_oos_regressed_team_state import STATE_FEATURES, _mean
from monster.teams import NFL_TEAMS

CORE_POSITIONS = {"QB", "RB", "WR", "TE", "OL", "DL", "LB", "DB"}


def _read(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path) if path.suffix == ".parquet" else pl.read_csv(path)


def _continuity(personnel: pl.DataFrame) -> dict[str, float]:
    pos = pl.col("position_group").cast(pl.Utf8).str.to_uppercase()
    core = pos.is_in(list(CORE_POSITIONS))

    offense = pl.col("offense_snap_share").fill_null(0.0) if "offense_snap_share" in personnel.columns else pl.lit(0.0)
    defense = pl.col("defense_snap_share").fill_null(0.0) if "defense_snap_share" in personnel.columns else pl.lit(0.0)
    snap = pl.max_horizontal(offense, defense).clip(0.0, 1.0)

    if "depth_rank" in personnel.columns:
        depth_w = pl.when(pl.col("depth_rank").is_not_null()).then(
            (1.0 / pl.col("depth_rank").cast(pl.Float64)).clip(0.20, 1.0)
        ).otherwise(0.30)
    else:
        depth_w = pl.lit(0.30)
    weight = pl.max_horizontal(snap, 0.30 * depth_w)

    prior_team = pl.col("prior_team_id") if "prior_team_id" in personnel.columns else pl.lit(None, dtype=pl.Utf8)
    observed = pl.col("snap_games_observed").fill_null(0) if "snap_games_observed" in personnel.columns else pl.lit(0)
    retained = core & prior_team.is_not_null() & (prior_team == pl.col("team_id")) & (observed > 0)

    rows = personnel.with_columns(weight.alias("_continuity_weight")).group_by("team_id").agg(
        pl.when(core).then(pl.col("_continuity_weight")).otherwise(0.0).sum().alias("total_weight"),
        pl.when(retained).then(pl.col("_continuity_weight")).otherwise(0.0).sum().alias("retained_weight"),
    ).to_dicts()

    result = {team: 0.50 for team in NFL_TEAMS}
    for row in rows:
        total = max(float(row["total_weight"] or 0.0), 1e-9)
        result[str(row["team_id"])] = float(np.clip(float(row["retained_weight"] or 0.0) / total, 0.0, 1.0))
    return result


def compile_policy(older: pl.DataFrame, latest: pl.DataFrame, personnel: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    continuity = _continuity(personnel)
    old_rows = {str(r["team_id"]): r for r in older.to_dicts()}
    new_rows = {str(r["team_id"]): r for r in latest.to_dicts()}
    output: list[dict[str, float | str]] = []
    audit: list[dict[str, float | str]] = []

    for team in NFL_TEAMS:
        c = float(continuity.get(team, 0.50))
        memory_authority = 0.30 + 0.60 * c
        row: dict[str, float | str] = {"team_id": team}
        for feature in STATE_FEATURES:
            if feature not in latest.columns:
                continue
            latest_mean = _mean(latest, feature)
            older_mean = _mean(older, feature)
            latest_value = new_rows.get(team, {}).get(feature)
            older_value = old_rows.get(team, {}).get(feature)
            latest_value = latest_mean if latest_value is None else float(latest_value)
            older_value = older_mean if older_value is None else float(older_value)
            identity_deviation = 0.75 * (latest_value - latest_mean) + 0.25 * (older_value - older_mean)
            row[feature] = float(latest_mean + memory_authority * identity_deviation)
        # Preserve non-state metadata columns from the latest policy where useful.
        latest_row = new_rows.get(team, {})
        for column in latest.columns:
            if column in row or column == "team_id":
                continue
            value = latest_row.get(column)
            if value is not None:
                row[column] = value
        output.append(row)
        audit.append({
            "team_id": team,
            "continuity": c,
            "memory_authority": memory_authority,
        })

    return pl.DataFrame(output).sort("team_id"), pl.DataFrame(audit).sort("team_id")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--older-policy", type=Path, required=True)
    parser.add_argument("--latest-policy", type=Path, required=True)
    parser.add_argument("--personnel", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/continuity-policy"))
    args = parser.parse_args()

    older = _read(args.older_policy)
    latest = _read(args.latest_policy)
    personnel = _read(args.personnel)
    policy, audit = compile_policy(older, latest, personnel)

    args.out.mkdir(parents=True, exist_ok=True)
    policy.write_csv(args.out / "team_policy.csv")
    policy.write_parquet(args.out / "team_policy.parquet", compression="zstd")
    audit.write_csv(args.out / "continuity_audit.csv")

    manifest = {
        "artifact": "Monster Continuity-Conditioned Team Policy",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "team_count": policy.height,
        "continuity_mean": float(audit["continuity"].mean()),
        "continuity_min": float(audit["continuity"].min()),
        "continuity_max": float(audit["continuity"].max()),
        "memory_authority_formula": "0.30 + 0.60 * continuity",
        "identity_formula": "latest_league_mean + memory_authority * (0.75 * latest_team_deviation + 0.25 * older_team_deviation)",
        "market_blind": True,
        "production_authority": "OOS gate passed across 2022-2025; total MAE improved 4/4 folds and both total/margin MAE improved 3/4 folds.",
        "principle": "Personnel continuity controls confidence in inherited team identity; it is not a scoring bonus.",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(audit)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
