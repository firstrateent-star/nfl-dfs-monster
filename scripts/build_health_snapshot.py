from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from monster.feature_compile.health import attach_health_state, health_coverage_report
from monster.feature_compile.participation_inference import infer_game_day_participation, participation_coverage_report
from monster.tabular import write_csv_safe


def _read(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path) if path.suffix == ".parquet" else pl.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--personnel", type=Path, required=True)
    parser.add_argument("--injuries", type=Path, default=None)
    parser.add_argument("--overrides", type=Path, default=None)
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--out", type=Path, default=Path("artifacts/week1-health"))
    args = parser.parse_args()

    personnel = _read(args.personnel)
    injuries = _read(args.injuries) if args.injuries and args.injuries.exists() else pl.DataFrame()
    overrides = _read(args.overrides) if args.overrides and args.overrides.exists() else pl.DataFrame()
    health = attach_health_state(personnel, injuries, overrides=overrides)
    health = infer_game_day_participation(health, season=args.season)

    args.out.mkdir(parents=True, exist_ok=True)
    health.write_parquet(args.out / "health_personnel.parquet", compression="zstd")
    write_csv_safe(health, args.out / "health_personnel.csv")
    health_coverage_report(health).write_csv(args.out / "health_by_team.csv")
    participation_coverage_report(health).write_csv(args.out / "participation_by_team.csv")

    concerns = health.filter(
        (pl.col("health_availability_probability") < 0.95)
        | (pl.col("health_effectiveness_if_active") < 0.98)
        | (pl.col("status") != "ACT")
    ).sort(["team_id", "game_day_active_probability"])
    write_csv_safe(concerns, args.out / "health_concerns.csv")

    manifest = {
        "artifact": "Monster Week 1 Health + Availability State",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "season": args.season,
        "roster_players": health.height,
        "provider_injury_rows": injuries.height,
        "explicit_override_rows": overrides.height,
        "players_with_non_roster_health_evidence": int(health.select((pl.col("health_evidence") != "roster_only").sum()).item()),
        "players_below_95pct_health_availability": int(health.select((pl.col("health_availability_probability") < 0.95).sum()).item()),
        "players_below_98pct_effectiveness_if_active": int(health.select((pl.col("health_effectiveness_if_active") < 0.98).sum()).item()),
        "rich_personnel_storage": "canonical parquet; nested evidence serialized only for CSV views",
        "principle": "Health availability, conditional effectiveness, and role uncertainty remain separate state variables.",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(concerns.select([c for c in [
        "team_id", "display_name", "position", "status", "health_state",
        "health_availability_probability", "health_effectiveness_if_active",
        "health_uncertainty", "game_day_active_probability", "health_injury",
    ] if c in concerns.columns]))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
