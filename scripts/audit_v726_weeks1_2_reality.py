from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import polars as pl

import compare_week1_v701_v720_distributional as compare


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return pl.read_csv(path, infer_schema_length=10000).to_dicts()


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return pl.read_parquet(path).to_dicts()


def _f(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _rate(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _metric(simulated: float | None, actual: float | None) -> dict[str, float | None]:
    if simulated is None or actual is None:
        return {
            "simulated": simulated,
            "actual": actual,
            "error": None,
            "relative_error": None,
        }
    error = simulated - actual
    return {
        "simulated": simulated,
        "actual": actual,
        "error": error,
        "relative_error": error / actual if abs(actual) > 1e-12 else None,
    }


def _worlds(sim: Path) -> int:
    manifest = json.loads((sim / "manifest.json").read_text(encoding="utf-8"))
    return int(manifest["worlds_per_game"])


def _actual_pass_rows(pbp: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in pbp
        if _f(row.get("pass_attempt")) == 1.0
        and _f(row.get("qb_spike")) != 1.0
    ]


def _actual_dropbacks(pbp: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in pbp if _f(row.get("qb_dropback")) == 1.0]


def _actual_run_rows(pbp: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in pbp
        if _f(row.get("rush_attempt")) == 1.0
        and _f(row.get("qb_dropback")) != 1.0
        and _f(row.get("qb_kneel")) != 1.0
    ]


def _pass_anatomy(sim: Path, truth: Path) -> dict[str, object]:
    sim_rows = _read_csv(sim / "same_world_pass_throws.csv")
    pbp = _read_parquet(truth / "pbp.parquet")
    actual = _actual_pass_rows(pbp)
    worlds = _worlds(sim)

    def aggregate(rows: list[dict[str, Any]], *, simulated: bool) -> dict[str, float | None]:
        if not rows:
            return {}
        if simulated:
            completions = sum(bool(row.get("complete")) for row in rows)
            interceptions = sum(bool(row.get("interception")) for row in rows)
            air = [_f(row.get("air_yards")) for row in rows if row.get("air_yards") is not None]
            yac = [
                _f(row.get("yac"))
                for row in rows
                if bool(row.get("complete")) and row.get("yac") is not None
            ]
        else:
            completions = sum(_f(row.get("complete_pass")) == 1.0 for row in rows)
            interceptions = sum(_f(row.get("interception")) == 1.0 for row in rows)
            air = [_f(row.get("air_yards")) for row in rows if row.get("air_yards") is not None]
            yac = [
                _f(row.get("yards_after_catch"))
                for row in rows
                if _f(row.get("complete_pass")) == 1.0
                and row.get("yards_after_catch") is not None
            ]
        yards = [_f(row.get("yards" if simulated else "yards_gained")) for row in rows]
        return {
            "attempts": len(rows) / worlds if simulated else float(len(rows)),
            "completion_rate": completions / len(rows),
            "interception_rate": interceptions / len(rows),
            "yards_per_attempt": _mean(yards),
            "air_yards_per_attempt": _mean(air),
            "yac_per_completion": _mean(yac),
            "explosive_20_rate": sum(value >= 20.0 for value in yards) / len(rows),
            "explosive_40_rate": sum(value >= 40.0 for value in yards) / len(rows),
        }

    sim_values = aggregate(sim_rows, simulated=True)
    actual_values = aggregate(actual, simulated=False)
    return {
        key: _metric(sim_values.get(key), actual_values.get(key))
        for key in sorted(set(sim_values) | set(actual_values))
    }


def _run_anatomy(sim: Path, truth: Path) -> dict[str, object]:
    sim_rows = _read_csv(sim / "same_world_designed_runs.csv")
    pbp = _read_parquet(truth / "pbp.parquet")
    actual = _actual_run_rows(pbp)
    worlds = _worlds(sim)

    def aggregate(rows: list[dict[str, Any]], *, simulated: bool) -> dict[str, float | None]:
        if not rows:
            return {}
        field = "yards" if simulated else "yards_gained"
        yards = [_f(row.get(field)) for row in rows]
        return {
            "attempts": len(rows) / worlds if simulated else float(len(rows)),
            "yards_per_attempt": _mean(yards),
            "negative_rate": sum(value < 0.0 for value in yards) / len(rows),
            "gain_3plus_rate": sum(value >= 3.0 for value in yards) / len(rows),
            "gain_5plus_rate": sum(value >= 5.0 for value in yards) / len(rows),
            "gain_10plus_rate": sum(value >= 10.0 for value in yards) / len(rows),
            "gain_20plus_rate": sum(value >= 20.0 for value in yards) / len(rows),
        }

    sim_values = aggregate(sim_rows, simulated=True)
    actual_values = aggregate(actual, simulated=False)
    return {
        key: _metric(sim_values.get(key), actual_values.get(key))
        for key in sorted(set(sim_values) | set(actual_values))
    }


def _pfr_pressure_summary(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    usable = [
        row for row in rows
        if row.get("times_pressured") is not None
        and row.get("team") is not None
    ]
    teams = {str(row.get("team")) for row in usable}
    pressures = sum(_f(row.get("times_pressured")) for row in usable)
    sacks = sum(_f(row.get("times_sacked")) for row in usable)
    blitzed = sum(_f(row.get("times_blitzed")) for row in usable)
    inferred_opportunities = 0.0
    for row in usable:
        pct = None
        for name in ("times_pressured_pct", "pressure_pct"):
            if row.get(name) is not None:
                pct = _f(row.get(name))
                break
        pressured = _f(row.get("times_pressured"))
        if pct and pct > 0.0:
            fraction = pct if pct <= 1.0 else pct / 100.0
            inferred_opportunities += pressured / fraction
    complete_enough = len(teams) >= 24
    pressure_rate = (
        _rate(pressures, inferred_opportunities)
        if complete_enough and inferred_opportunities > 0.0
        else None
    )
    blitz_rate = (
        _rate(blitzed, inferred_opportunities)
        if complete_enough and inferred_opportunities > 0.0
        else None
    )
    return {
        "pressures": pressures if complete_enough else None,
        "sacks": sacks if complete_enough else None,
        "pressure_opportunities": (
            inferred_opportunities if complete_enough else None
        ),
        "pressure_rate": pressure_rate,
        "blitz_exposures": blitzed if complete_enough else None,
        "blitz_exposure_rate": blitz_rate,
        "rows": len(usable),
        "teams": len(teams),
        "coverage_complete_enough": complete_enough,
    }


def _ngs_timing(rows: list[dict[str, Any]]) -> dict[str, object]:
    if not rows:
        return {"available": False}
    candidates = [
        column
        for column in rows[0]
        if "time_to_throw" in column.lower()
    ]
    if not candidates:
        return {"available": False, "columns_seen": list(rows[0])}
    column = candidates[0]
    values = [_f(row.get(column)) for row in rows if row.get(column) is not None]
    return {
        "available": bool(values),
        "column": column,
        "mean": _mean(values),
        "players": len(values),
    }


def _protection_and_rush_plan(sim: Path, truth: Path) -> dict[str, object]:
    snaps = _read_csv(sim / "same_world_snap_participants_v5.csv")
    sim_pass = [
        row
        for row in snaps
        if str(row.get("play_type", "")).lower().endswith("pass")
    ]
    dropbacks = _read_csv(sim / "same_world_dropbacks.csv")
    pbp = _read_parquet(truth / "pbp.parquet")
    actual_dropbacks = _actual_dropbacks(pbp)
    pfr = _read_csv(truth / "pfr_pass.csv")
    ftn = _read_parquet(truth / "ftn_charting.parquet")
    ngs = _read_csv(truth / "ngs_passing.csv")

    planned_counts = []
    for row in sim_pass:
        value = str(row.get("planned_rushers") or "")
        planned_counts.append(len([piece for piece in value.split("|") if piece]))
    def _play_id(value: object) -> str:
        try:
            return str(int(float(value)))
        except (TypeError, ValueError):
            return str(value or "")

    dropback_keys = {
        (str(row.get("game_id") or ""), _play_id(row.get("play_id")))
        for row in actual_dropbacks
    }
    ftn_dropbacks = [
        row
        for row in ftn
        if (
            str(row.get("nflverse_game_id") or ""),
            _play_id(row.get("nflverse_play_id")),
        )
        in dropback_keys
    ]
    actual_rusher_counts = [
        _f(row.get("n_pass_rushers"))
        for row in ftn_dropbacks
        if row.get("n_pass_rushers") is not None
    ]
    actual_blitzers = [
        _f(row.get("n_blitzers"))
        for row in ftn_dropbacks
        if row.get("n_blitzers") is not None
    ]

    sim_pressure_rate = (
        sum(bool(row.get("pressured")) for row in sim_pass) / len(sim_pass)
        if sim_pass
        else None
    )
    sim_sack_rate = (
        sum(str(row.get("outcome", "")).lower().endswith("sack") for row in dropbacks)
        / len(dropbacks)
        if dropbacks
        else None
    )
    actual_sack_rate = (
        sum(_f(row.get("sack")) == 1.0 for row in actual_dropbacks)
        / len(actual_dropbacks)
        if actual_dropbacks
        else None
    )
    pfr_summary = _pfr_pressure_summary(pfr)

    rush_plan_counts = Counter(str(row.get("rush_plan")) for row in sim_pass)
    rush_plan_shares = {
        key: value / len(sim_pass)
        for key, value in sorted(rush_plan_counts.items())
    } if sim_pass else {}

    return {
        "pressure_rate": _metric(
            sim_pressure_rate,
            pfr_summary.get("pressure_rate"),
        ),
        "sack_rate_per_dropback": _metric(sim_sack_rate, actual_sack_rate),
        "mean_planned_or_charted_pass_rushers": _metric(
            _mean(planned_counts),
            _mean(actual_rusher_counts),
        ),
        "five_plus_rusher_rate": _metric(
            (
                sum(value >= 5 for value in planned_counts) / len(planned_counts)
                if planned_counts else None
            ),
            (
                sum(value >= 5 for value in actual_rusher_counts)
                / len(actual_rusher_counts)
                if actual_rusher_counts else None
            ),
        ),
        "actual_mean_charted_blitzers": _mean(actual_blitzers),
        "actual_ftn_dropback_rows": len(ftn_dropbacks),
        "actual_pfr_blitz_exposure_rate": pfr_summary.get("blitz_exposure_rate"),
        "actual_pfr_pressure_coverage": {
            "rows": pfr_summary.get("rows"),
            "teams": pfr_summary.get("teams"),
            "complete_enough": pfr_summary.get("coverage_complete_enough"),
        },
        "simulated_rush_plan_shares": rush_plan_shares,
        "actual_ngs_timing": _ngs_timing(ngs),
        "evidence_sources": {
            "simulated": "same-world snap/dropback telemetry",
            "actual_pressure": "PFR advanced passing weekly",
            "actual_rusher_structure": "FTN charting weekly",
            "actual_timing": "NFL Next Gen Stats weekly",
        },
    }


def _coverage_effectiveness(sim: Path, truth: Path) -> dict[str, object]:
    snaps = _read_csv(sim / "same_world_snap_participants_v5.csv")
    pass_snaps = [
        row
        for row in snaps
        if str(row.get("play_type", "")).lower().endswith("pass")
    ]
    shell_counts = Counter(str(row.get("coverage_shell")) for row in pass_snaps)
    shell_shares = {
        key: value / len(pass_snaps)
        for key, value in sorted(shell_counts.items())
    } if pass_snaps else {}
    pass_anatomy = _pass_anatomy(sim, truth)

    return {
        "simulated_shell_shares": shell_shares,
        "actual_exact_shell_call_comparison_available": False,
        "actual_exact_shell_call_limitation": (
            "Current-season nflverse participation coverage fields are not published "
            "in-season. Exact man/zone/Cover-N call matching is therefore intentionally "
            "not inferred. Coverage is audited through realized pass efficiency, "
            "explosive allowance, interceptions, defensive attribution, and snap usage."
        ),
        "realized_pass_coverage_outcomes": pass_anatomy,
    }


def _special_teams_outcomes(sim: Path, truth: Path) -> dict[str, object]:
    sim_rows = _read_csv(sim / "same_world_special_teams_players_v72.csv")
    pbp = _read_parquet(truth / "pbp.parquet")
    worlds = _worlds(sim)

    def sim_type(name: str) -> list[dict[str, Any]]:
        return [
            row for row in sim_rows
            if str(row.get("event_type", "")).lower().endswith(name)
        ]

    sim_fg = sim_type("field_goal")
    sim_punts = sim_type("punt")
    sim_kickoffs = sim_type("kickoff")
    sim_pat = sim_type("pat")

    actual_fg = [row for row in pbp if _f(row.get("field_goal_attempt")) == 1.0]
    actual_punts = [
        row for row in pbp if str(row.get("play_type", "")).lower() == "punt"
    ]
    actual_kickoffs = [
        row for row in pbp if _f(row.get("kickoff_attempt")) == 1.0
    ]
    actual_pat = [
        row for row in pbp if _f(row.get("extra_point_attempt")) == 1.0
    ]

    def count_metric(simulated_rows, actual_rows):
        return _metric(
            len(simulated_rows) / worlds,
            float(len(actual_rows)),
        )

    def sim_make(rows):
        vals = [bool(row.get("made")) for row in rows if row.get("made") is not None]
        return sum(vals) / len(vals) if vals else None

    def actual_fg_make(rows):
        vals = [
            str(row.get("field_goal_result", "")).lower() == "made"
            for row in rows
        ]
        return sum(vals) / len(vals) if vals else None

    def actual_pat_make(rows):
        vals = [
            str(row.get("extra_point_result", "")).lower() == "good"
            for row in rows
        ]
        return sum(vals) / len(vals) if vals else None

    def mean_field(rows, field):
        vals = [_f(row.get(field)) for row in rows if row.get(field) is not None]
        return _mean(vals)

    return {
        "field_goal_attempts": count_metric(sim_fg, actual_fg),
        "field_goal_make_rate": _metric(sim_make(sim_fg), actual_fg_make(actual_fg)),
        "field_goal_distance": _metric(
            mean_field(sim_fg, "kick_distance"),
            mean_field(actual_fg, "kick_distance"),
        ),
        "punts": count_metric(sim_punts, actual_punts),
        "punt_return_yards_per_event": _metric(
            mean_field(sim_punts, "return_yards"),
            mean_field(actual_punts, "return_yards"),
        ),
        "kickoffs": count_metric(sim_kickoffs, actual_kickoffs),
        "kickoff_return_yards_per_event": _metric(
            mean_field(sim_kickoffs, "return_yards"),
            mean_field(actual_kickoffs, "return_yards"),
        ),
        "kickoff_touchback_rate": _metric(
            (
                sum(bool(row.get("touchback")) for row in sim_kickoffs)
                / len(sim_kickoffs)
                if sim_kickoffs else None
            ),
            (
                sum(_f(row.get("touchback")) == 1.0 for row in actual_kickoffs)
                / len(actual_kickoffs)
                if actual_kickoffs else None
            ),
        ),
        "pat_attempts": count_metric(sim_pat, actual_pat),
        "pat_make_rate": _metric(sim_make(sim_pat), actual_pat_make(actual_pat)),
    }


def _week_report(
    *,
    week: int,
    sim: Path,
    box: Path,
    truth: Path,
    personnel: Path,
    drive_audit: Path,
) -> dict[str, object]:
    actual_games, _, players = compare._actual_context(
        truth / "schedules.csv",
        truth / "player_stats.csv",
    )
    scoreboard, scoreboard_detail = compare._scoreboard_arm(sim, actual_games)
    actual_ecology = compare._actual_ecology(players)
    ecology, _ = compare._ecology_arm(sim, actual_ecology)

    report = {
        "week": week,
        "scoreboard": scoreboard,
        "team_offense": compare._team_offense_arm(box, players),
        "offensive_player_box_fit": compare._offense_player_arm(box, players),
        "carry_share": compare._role_share_arm(
            box, players, actual_col="carries", sim_col="rush_attempts_mean"
        ),
        "target_share": compare._role_share_arm(
            box, players, actual_col="targets", sim_col="targets_mean"
        ),
        "participation": compare._participation_arm(
            sim,
            box,
            truth / "snap_counts.csv",
            personnel,
        ),
        "defensive_player_box_fit": compare._defense_arm(
            box,
            players,
            truth / "pfr_def.csv",
        ),
        "football_event_ecology": ecology,
        "pass_anatomy": _pass_anatomy(sim, truth),
        "run_anatomy": _run_anatomy(sim, truth),
        "protection_and_pass_rush": _protection_and_rush_plan(sim, truth),
        "coverage": _coverage_effectiveness(sim, truth),
        "special_teams_identity": compare._special_teams_arm(
            sim,
            players,
            personnel,
        ),
        "special_teams_outcomes": _special_teams_outcomes(sim, truth),
        "drive_anatomy": json.loads(
            (drive_audit / "diagnosis.json").read_text(encoding="utf-8")
        ),
        "source_limits": {
            "exact_current_man_zone_calls": (
                "not available from in-season nflverse participation"
            ),
            "coverage_evaluation": (
                "realized pass outcomes + defensive player attribution + snap "
                "participation + model shell distribution"
            ),
            "postgame_truth_is_observational_only": True,
        },
    }
    pl.DataFrame(scoreboard_detail).write_csv(
        drive_audit.parent / f"week{week}_scoreboard_detail.csv"
    )
    return report


def _flatten(prefix: str, value: object, rows: list[dict[str, object]]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), item, rows)
        return
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number):
            rows.append({"metric": prefix, "value": number})



def _clean_json(value: object) -> object:
    if isinstance(value, dict):
        return {key: _clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _paired_structural_candidates(
    week1: dict[str, object],
    week2: dict[str, object],
) -> list[dict[str, object]]:
    candidates = []
    for domain in ("pass_anatomy", "run_anatomy", "special_teams_outcomes"):
        first = week1.get(domain, {})
        second = week2.get(domain, {})
        if not isinstance(first, dict) or not isinstance(second, dict):
            continue
        for metric in sorted(set(first) & set(second)):
            a = first.get(metric)
            b = second.get(metric)
            if not isinstance(a, dict) or not isinstance(b, dict):
                continue
            r1 = a.get("relative_error")
            r2 = b.get("relative_error")
            if not isinstance(r1, (int, float)) or not isinstance(r2, (int, float)):
                continue
            same_direction = r1 * r2 > 0.0
            candidates.append(
                {
                    "domain": domain,
                    "metric": metric,
                    "week1_relative_error": r1,
                    "week2_relative_error": r2,
                    "same_direction": same_direction,
                    "mean_absolute_relative_error": (abs(r1) + abs(r2)) / 2.0,
                }
            )
    candidates.sort(
        key=lambda row: (
            bool(row["same_direction"]),
            float(row["mean_absolute_relative_error"]),
        ),
        reverse=True,
    )
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    for week in (1, 2):
        parser.add_argument(f"--week{week}-sim", type=Path, required=True)
        parser.add_argument(f"--week{week}-box", type=Path, required=True)
        parser.add_argument(f"--week{week}-truth", type=Path, required=True)
        parser.add_argument(f"--week{week}-personnel", type=Path, required=True)
        parser.add_argument(f"--week{week}-drive-audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    week1 = _week_report(
        week=1,
        sim=args.week1_sim,
        box=args.week1_box,
        truth=args.week1_truth,
        personnel=args.week1_personnel,
        drive_audit=args.week1_drive_audit,
    )
    week2 = _week_report(
        week=2,
        sim=args.week2_sim,
        box=args.week2_box,
        truth=args.week2_truth,
        personnel=args.week2_personnel,
        drive_audit=args.week2_drive_audit,
    )

    (args.out / "week1_report.json").write_text(
        json.dumps(_clean_json(week1), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (args.out / "week2_report.json").write_text(
        json.dumps(_clean_json(week2), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    structural = _paired_structural_candidates(week1, week2)
    (args.out / "cross_week_structural_candidates.json").write_text(
        json.dumps(_clean_json(structural), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    flat_rows: list[dict[str, object]] = []
    _flatten("week1", week1, flat_rows)
    _flatten("week2", week2, flat_rows)
    if flat_rows:
        with (args.out / "metric_index.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=("metric", "value"))
            writer.writeheader()
            writer.writerows(flat_rows)

    manifest = {
        "artifact": "MONSTER V7.2.6 2026 Weeks 1-2 Whole-Football Reality Audit",
        "control_release": "release/v726-verified-baseline",
        "control_commit": "506fea89fc914e56ead1c17ccbc7212d950084c7",
        "weeks": [1, 2],
        "fantasy_scores_are_downstream_observations_not_calibration_targets": True,
        "postgame_truth_used_only_after_simulation": True,
        "market_blind_simulation": True,
        "simulation_behavior_changed_by_audit": False,
        "domains": [
            "scoreboard and distributions",
            "team offense",
            "player offensive box scores",
            "target and carry ownership",
            "offensive and defensive participation",
            "defensive player attribution",
            "drive and series ecology",
            "passing and rushing anatomy",
            "protection, pressure, sacks, rush counts and blitz structure",
            "coverage effectiveness and simulated shell behavior",
            "special-teams identity and outcomes",
        ],
        "coverage_caveat": (
            "Exact 2026 weekly man/zone/Cover-N calls are not available from the "
            "in-season nflverse participation feed. The audit does not fabricate "
            "them; it evaluates realized coverage effectiveness and records the "
            "model's shell distribution separately."
        ),
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
