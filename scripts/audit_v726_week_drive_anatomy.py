from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

import audit_week2_drive_anatomy_v726 as base

WEEK1_MATCHUPS = (
    ("NE", "SEA"), ("SF", "LAR"), ("CHI", "CAR"), ("TB", "CIN"),
    ("NO", "DET"), ("BUF", "HOU"), ("BAL", "IND"), ("CLE", "JAC"),
    ("ATL", "PIT"), ("NYJ", "TEN"), ("ARI", "LAC"), ("MIA", "LV"),
    ("GB", "MIN"), ("WAS", "PHI"), ("DAL", "NYG"), ("DEN", "KC"),
)
WEEK2_MATCHUPS = (
    ("DET", "BUF"), ("CAR", "ATL"), ("NO", "BAL"), ("MIN", "CHI"),
    ("CIN", "HOU"), ("PIT", "NE"), ("GB", "NYJ"), ("CLE", "TB"),
    ("PHI", "TEN"), ("JAC", "DEN"), ("LV", "LAC"), ("SEA", "ARI"),
    ("WAS", "DAL"), ("MIA", "SF"), ("IND", "KC"), ("NYG", "LAR"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--week", type=int, choices=(1, 2), required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/monster"))
    parser.add_argument("--simulated", type=Path, required=True)
    parser.add_argument("--sim-manifest", type=Path, required=True)
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--historical-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    matchups = WEEK1_MATCHUPS if args.week == 1 else WEEK2_MATCHUPS
    base.MAIN_SLATE_MATCHUPS = matchups
    base.TEAM_ALIASES["LA"] = "LAR"

    sim_manifest = json.loads(args.sim_manifest.read_text(encoding="utf-8"))
    historical_manifest = json.loads(
        args.historical_manifest.read_text(encoding="utf-8")
    )
    worlds = int(sim_manifest["worlds_per_game"])
    games = len(matchups)

    sim_rows = pl.read_csv(args.simulated).to_dicts()
    actual_rows, actual_manifest = base._actual_drive_rows(
        season=args.season,
        week=args.week,
        cache_dir=args.cache_dir,
    )
    historical_rows = pl.read_parquet(args.historical).to_dicts()

    sim_summary = base._summary(
        sim_rows,
        team_games=games * 2 * worlds,
        scope=f"v726_week{args.week}_simulation",
    )
    actual_summary = base._summary(
        actual_rows,
        team_games=games * 2,
        scope=f"actual_week{args.week}",
    )
    historical_summary = base._summary(
        historical_rows,
        team_games=int(historical_manifest["games"]) * 2,
        scope="historical_2025_regular_season",
    )
    comparison = base._comparison_rows(
        sim_summary, actual_summary, historical_summary
    )
    segment_rows = base._segment_rows(
        sim_rows,
        team_games=games * 2 * worlds,
        source="v726",
    ) + base._segment_rows(
        actual_rows,
        team_games=games * 2,
        source=f"actual_week{args.week}",
    )
    team_rows = base._team_comparison(sim_rows, actual_rows, worlds)

    structural = [
        row
        for row in comparison
        if row["triage"] == "structural_model_mismatch_signal"
    ]
    structural.sort(
        key=lambda row: abs(float(row["v726_vs_historical_relative"] or 0.0)),
        reverse=True,
    )
    week_specific = [
        row
        for row in comparison
        if row["triage"] == "week2_specific_outlier_signal"
    ]
    week_specific.sort(
        key=lambda row: abs(float(row["actual_vs_historical_relative"] or 0.0)),
        reverse=True,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(actual_rows).write_parquet(
        args.out / f"actual_week{args.week}_drive_traces.parquet"
    )
    pl.DataFrame(
        [sim_summary, actual_summary, historical_summary]
    ).write_csv(args.out / "drive_anatomy_overall.csv")
    pl.DataFrame(comparison).write_csv(
        args.out / "drive_anatomy_comparison.csv"
    )
    pl.DataFrame(segment_rows).write_csv(
        args.out / "drive_anatomy_by_state.csv"
    )
    pl.DataFrame(team_rows).sort(
        "offensive_points_delta", descending=True
    ).write_csv(args.out / "drive_anatomy_by_team.csv")

    diagnosis = {
        "week": args.week,
        "strongest_structural_model_mismatch_signals": structural[:15],
        "strongest_week_specific_outlier_signals": week_specific[:15],
        "interpretation_rule": (
            "A one-week miss is not a calibration target by itself. Structural "
            "changes require repeatable directional mismatch against both weeks, "
            "the broader 2025 NFL baseline, or a definition-safe mechanism failure."
        ),
    }
    (args.out / "diagnosis.json").write_text(
        json.dumps(diagnosis, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "artifact": f"MONSTER V7.2.6 Week {args.week} Drive Reality Audit",
        "season": args.season,
        "week": args.week,
        "games": games,
        "simulation_worlds_per_game": worlds,
        "simulation_source_runtime": sim_manifest.get("experiment"),
        "historical_baseline_season": historical_manifest.get("season", 2025),
        "actual": actual_manifest,
        "postgame_truth_used_only_for_audit": True,
        "simulation_behavior_changed_by_audit": False,
        "market_or_sportsbook_inputs_added": False,
        "direct_score_adjustment": False,
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"manifest": manifest, "diagnosis": diagnosis}, indent=2))


if __name__ == "__main__":
    main()
