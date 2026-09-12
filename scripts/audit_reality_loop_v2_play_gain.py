from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

import audit_play_gain_reality as audit
from monster.sim import matchup_kernel, play_kernel, resolution_ecology
from monster.sim.dispersion_bridge import enhanced_defensive_unit, enhanced_team_identity
from monster.sim.play_kernel import PassResult, PlayEvent, PlayType
from monster.sim.resolution_bands_v2 import resolve_run_contact_v2, resolve_run_ecology_v2
from monster.sim.snap_ecology_v2 import resolve_pass_snap_v2, resolve_run_snap_v2

_SIM_RUNS: list[dict[str, object]] = []
_HIST_RUNS: list[dict[str, object]] = []
_SIM_THROWS: list[dict[str, object]] = []
_HIST_THROWS: list[dict[str, object]] = []


def _historical_geometry(row: dict[str, object]) -> str:
    if float(row.get("qb_sneak") or 0.0) == 1.0:
        return "qb_sneak"
    location = str(row.get("run_location") or "unknown").lower()
    gap = str(row.get("run_gap") or "unknown").lower()
    if location == "middle":
        return "interior"
    if location in {"left", "right"} and gap in {"guard", "unknown"}:
        return "interior"
    if location == "left" and gap == "tackle":
        return "left_offtackle"
    if location == "right" and gap == "tackle":
        return "right_offtackle"
    if location == "left" and gap == "end":
        return "left_edge"
    if location == "right" and gap == "end":
        return "right_edge"
    return "other"


def _depth_category(air_yards: float) -> str:
    if air_yards < 0.0:
        return "behind_los"
    if air_yards <= 5.0:
        return "short_0_5"
    if air_yards <= 9.0:
        return "short_6_9"
    if air_yards <= 19.0:
        return "intermediate_10_19"
    if air_yards <= 39.0:
        return "deep_20_39"
    return "bomb_40_plus"


def _capture_sim(original):
    def wrapped(store, event: PlayEvent):
        if event.play_type == PlayType.RUN:
            _SIM_RUNS.append(
                {
                    "category": event.run_geometry_category or "other",
                    "yards": float(event.yards),
                }
            )
        elif event.play_type == PlayType.PASS and event.pass_result in {
            PassResult.COMPLETE,
            PassResult.INCOMPLETE,
            PassResult.INTERCEPTION,
        }:
            _SIM_THROWS.append(
                {
                    "category": event.pass_depth_category or _depth_category(float(event.air_yards)),
                    "complete": event.pass_result == PassResult.COMPLETE,
                    "interception": event.pass_result == PassResult.INTERCEPTION,
                    "pressured": bool(event.pressured),
                    "yards": float(event.yards),
                    "air_yards": float(event.air_yards),
                }
            )
        return original(store, event)

    return wrapped


def _capture_history(original):
    def wrapped(store, row: dict[str, object]):
        dropback = float(row.get("qb_dropback") or 0.0) == 1.0
        rush = float(row.get("rush_attempt") or 0.0) == 1.0
        if rush and not dropback:
            _HIST_RUNS.append(
                {
                    "category": _historical_geometry(row),
                    "yards": float(row.get("yards_gained") or 0.0),
                }
            )
        if (
            dropback
            and float(row.get("sack") or 0.0) != 1.0
            and float(row.get("qb_scramble") or 0.0) != 1.0
            and row.get("air_yards") is not None
        ):
            air_yards = float(row.get("air_yards") or 0.0)
            _HIST_THROWS.append(
                {
                    "category": _depth_category(air_yards),
                    "complete": float(row.get("complete_pass") or 0.0) == 1.0,
                    "interception": float(row.get("interception") or 0.0) == 1.0,
                    "yards": float(row.get("yards_gained") or 0.0),
                    "air_yards": air_yards,
                }
            )
        return original(store, row)

    return wrapped


def _run_summary(rows: list[dict[str, object]], source: str) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame()
    frame = pl.DataFrame(rows)
    total = frame.height
    return (
        frame.group_by("category")
        .agg(
            pl.len().alias("events"),
            pl.col("yards").mean().alias("yards_mean"),
            (pl.col("yards") < 0.0).mean().alias("negative_rate"),
            (pl.col("yards") >= 3.0).mean().alias("gain_3plus_rate"),
            (pl.col("yards") >= 5.0).mean().alias("gain_5plus_rate"),
            (pl.col("yards") >= 10.0).mean().alias("gain_10plus_rate"),
            (pl.col("yards") >= 15.0).mean().alias("gain_15plus_rate"),
        )
        .with_columns(
            pl.lit(source).alias("source"),
            (pl.col("events") / total).alias("share"),
        )
        .sort("category")
    )


def _throw_summary(rows: list[dict[str, object]], source: str) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame()
    frame = pl.DataFrame(rows)
    total = frame.height
    aggregations = [
        pl.len().alias("attempts"),
        pl.col("complete").cast(pl.Float64).mean().alias("completion_rate"),
        pl.col("interception").cast(pl.Float64).mean().alias("interception_rate"),
        (pl.col("yards") < 0.0).mean().alias("negative_gain_rate"),
        (pl.col("yards") >= 5.0).mean().alias("gain_5plus_rate"),
        (pl.col("yards") >= 10.0).mean().alias("gain_10plus_rate"),
        pl.col("air_yards").mean().alias("air_yards_mean"),
    ]
    if "pressured" in frame.columns:
        aggregations.append(pl.col("pressured").cast(pl.Float64).mean().alias("pressure_rate"))
    return (
        frame.group_by("category")
        .agg(*aggregations)
        .with_columns(
            pl.lit(source).alias("source"),
            (pl.col("attempts") / total).alias("share"),
        )
        .sort("category")
    )


def _write_geometry(out: Path) -> None:
    sim = _run_summary(_SIM_RUNS, "monster")
    hist = _run_summary(_HIST_RUNS, "nfl_2025")
    if sim.is_empty() or hist.is_empty():
        return
    pl.concat([sim, hist], how="diagonal_relaxed").write_csv(out / "run_geometry_gain_summary.csv")
    metrics = [
        "share",
        "yards_mean",
        "negative_rate",
        "gain_3plus_rate",
        "gain_5plus_rate",
        "gain_10plus_rate",
        "gain_15plus_rate",
    ]
    joined = sim.join(hist, on="category", suffix="_nfl")
    rows: list[dict[str, object]] = []
    for row in joined.to_dicts():
        for metric in metrics:
            monster = float(row[metric])
            nfl = float(row[f"{metric}_nfl"])
            rows.append(
                {
                    "category": row["category"],
                    "metric": metric,
                    "monster": monster,
                    "nfl": nfl,
                    "delta": monster - nfl,
                    "relative_delta": (monster - nfl) / nfl if abs(nfl) > 1e-12 else None,
                }
            )
    comparison = pl.DataFrame(rows).sort(["category", "metric"])
    comparison.write_csv(out / "run_geometry_gain_comparison.csv")
    print("RUN-GEOMETRY LOCALIZATION")
    print(
        comparison.filter(
            pl.col("metric").is_in(["share", "negative_rate", "gain_3plus_rate", "gain_5plus_rate", "gain_10plus_rate", "gain_15plus_rate"])
        )
    )


def _write_pass_depth(out: Path) -> None:
    sim = _throw_summary(_SIM_THROWS, "monster")
    hist = _throw_summary(_HIST_THROWS, "nfl_2025")
    if sim.is_empty() or hist.is_empty():
        return
    pl.concat([sim, hist], how="diagonal_relaxed").write_csv(out / "pass_depth_throw_summary.csv")
    joined = sim.join(hist, on="category", suffix="_nfl")
    metrics = [
        "share",
        "completion_rate",
        "interception_rate",
        "negative_gain_rate",
        "gain_5plus_rate",
        "gain_10plus_rate",
        "air_yards_mean",
    ]
    rows: list[dict[str, object]] = []
    for row in joined.to_dicts():
        for metric in metrics:
            monster = float(row[metric])
            nfl = float(row[f"{metric}_nfl"])
            rows.append(
                {
                    "category": row["category"],
                    "metric": metric,
                    "monster": monster,
                    "nfl": nfl,
                    "delta": monster - nfl,
                    "relative_delta": (monster - nfl) / nfl if abs(nfl) > 1e-12 else None,
                }
            )
    comparison = pl.DataFrame(rows).sort(["category", "metric"])
    comparison.write_csv(out / "pass_depth_throw_comparison.csv")
    print("PASS-DEPTH THROW LOCALIZATION")
    print(
        comparison.filter(
            pl.col("metric").is_in(["share", "completion_rate", "interception_rate", "gain_5plus_rate"])
        )
    )
    if "pressure_rate" in sim.columns:
        weighted_pressure = float(
            sim.select((pl.col("pressure_rate") * pl.col("attempts")).sum() / pl.col("attempts").sum()).item()
        )
        print({"monster_throw_pressure_rate": weighted_pressure, "league_dropback_pressure_reference": 0.297832})


def _out_path() -> Path:
    if "--out" not in sys.argv:
        return Path("artifacts/reality-loop-v2-play-gain")
    return Path(sys.argv[sys.argv.index("--out") + 1])


def main() -> None:
    """Run NFL-vs-Monster play-family, run-geometry and pass-depth audits on Reality Loop v2."""

    resolution_ecology._COARSE_MATCHUP_AUTHORITY = 0.35
    resolution_ecology._COARSE_RUN_MATCHUP_AUTHORITY = 0.25
    matchup_kernel.resolve_pass_snap = resolve_pass_snap_v2
    matchup_kernel.resolve_run_snap = resolve_run_snap_v2
    resolution_ecology.resolve_run_ecology = resolve_run_ecology_v2
    play_kernel.resolve_run_contact = resolve_run_contact_v2
    audit._team_identity = enhanced_team_identity
    audit._defensive_unit = enhanced_defensive_unit

    original_sim = audit._append_sim_event
    original_history = audit._append_historical_row
    audit._append_sim_event = _capture_sim(original_sim)
    audit._append_historical_row = _capture_history(original_history)
    try:
        audit.main()
    finally:
        audit._append_sim_event = original_sim
        audit._append_historical_row = original_history
    out = _out_path()
    _write_geometry(out)
    _write_pass_depth(out)


if __name__ == "__main__":
    main()
