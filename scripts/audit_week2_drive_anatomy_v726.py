from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import nflreadpy as nfl
import numpy as np
import polars as pl
from audit_historical_drive_reality import _drive_row, _is_scrimmage, _number

from monster.ingest.nflverse import configure_cache

MAIN_SLATE_MATCHUPS = (
    ("CAR", "ATL"),
    ("NO", "BAL"),
    ("MIN", "CHI"),
    ("CIN", "HOU"),
    ("PIT", "NE"),
    ("GB", "NYJ"),
    ("CLE", "TB"),
    ("PHI", "TEN"),
    ("JAC", "DEN"),
    ("LV", "LAC"),
    ("SEA", "ARI"),
    ("WAS", "DAL"),
    ("MIA", "SF"),
)

TEAM_ALIASES = {"JAX": "JAC"}


def _canonical_team(team: str) -> str:
    return TEAM_ALIASES.get(str(team), str(team))


TERMINALS = (
    "touchdown",
    "field_goal",
    "missed_field_goal",
    "punt",
    "turnover",
    "turnover_on_downs",
    "safety",
    "halftime",
    "end_game",
)

COMPARE_METRICS = (
    "drives_per_team_game",
    "points_per_drive",
    "offensive_points_per_team_game",
    "touchdown_rate",
    "field_goal_rate",
    "missed_field_goal_rate",
    "punt_rate",
    "turnover_rate",
    "turnover_on_downs_rate",
    "zero_point_drive_rate",
    "red_zone_reach_rate",
    "red_zone_snap_rate",
    "td_per_red_zone_reach",
    "td_per_red_zone_snap",
    "td_without_red_zone_snap_rate",
    "explosive_drive_rate",
    "td_given_explosive_drive",
    "td_given_no_explosive_drive",
    "long_40_zero_rate",
    "zero_given_long_40",
    "long_50_zero_rate",
    "zero_given_long_50",
    "scrimmage_plays_per_drive",
    "net_yards_per_drive",
    "first_downs_per_drive",
    "series_conversion_rate",
    "third_down_conversion_rate",
    "third_and_long_conversion_rate",
    "fourth_down_snaps_per_drive",
    "turnover_on_downs_per_fourth_down_snap",
    "early_down_run_yards_per_snap",
    "early_down_pass_yards_per_snap",
    "early_down_run_5plus_rate",
    "early_down_pass_5plus_rate",
    "early_down_run_negative_rate",
    "early_down_pass_negative_rate",
    "sacks_per_drive",
    "turnovers_per_drive",
    "start_yardline_100_mean",
    "short_field_start_rate",
    "td_given_short_field",
    "points_per_100_net_yards",
)


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if abs(denominator) > 1e-12 else 0.0


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(ys) < 2:
        return 0.0
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _start_zone(yardline: float) -> str:
    if yardline < 20.0:
        return "own_1_19"
    if yardline < 40.0:
        return "own_20_39"
    if yardline < 60.0:
        return "own_40_to_plus_41"
    if yardline < 80.0:
        return "plus_40_to_21"
    return "red_zone"


def _first_scrimmage(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    return next((row for row in rows if _is_scrimmage(row)), None)


def _last_event(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    return rows[-1] if rows else None


def _actual_drive_rows(
    *, season: int, week: int, cache_dir: Path
) -> tuple[list[dict[str, object]], dict[str, object]]:
    configure_cache(cache_dir)
    schedules = nfl.load_schedules([season])
    slate = schedules.filter(pl.col("week") == week)
    if "game_type" in slate.columns:
        slate = slate.filter(pl.col("game_type") == "REG")

    wanted = {f"{away}@{home}" for away, home in MAIN_SLATE_MATCHUPS}
    schedule_rows = []
    for row in slate.to_dicts():
        away = _canonical_team(str(row.get("away_team") or ""))
        home = _canonical_team(str(row.get("home_team") or ""))
        game = f"{away}@{home}"
        if game in wanted:
            schedule_rows.append(row)
    if len(schedule_rows) != len(MAIN_SLATE_MATCHUPS):
        found = sorted(f"{row.get('away_team')}@{row.get('home_team')}" for row in schedule_rows)
        missing = sorted(wanted.difference(found))
        raise ValueError(f"Week 2 schedule matched {len(schedule_rows)} games; missing={missing}")

    game_map = {
        str(row["game_id"]): {
            "game": f"{_canonical_team(str(row['away_team']))}@{_canonical_team(str(row['home_team']))}",
            "away": _canonical_team(str(row["away_team"])),
            "home": _canonical_team(str(row["home_team"])),
        }
        for row in schedule_rows
    }
    pbp = nfl.load_pbp([season])
    if "season_type" in pbp.columns:
        pbp = pbp.filter(pl.col("season_type") == "REG")
    if "qb_kneel" in pbp.columns:
        pbp = pbp.filter(pl.col("qb_kneel").fill_null(0) == 0)
    if "qb_spike" in pbp.columns:
        pbp = pbp.filter(pl.col("qb_spike").fill_null(0) == 0)
    pbp = pbp.filter(pl.col("game_id").cast(pl.String).is_in(list(game_map)))

    required = {"game_id", "fixed_drive", "posteam", "play_id", "game_seconds_remaining"}
    missing_fields = sorted(required.difference(pbp.columns))
    if missing_fields:
        raise ValueError(f"Week 2 drive audit missing required nflverse fields: {missing_fields}")

    usable = pbp.filter(
        pl.col("fixed_drive").is_not_null()
        & pl.col("posteam").is_not_null()
        & pl.col("game_seconds_remaining").is_not_null()
    ).sort(["game_id", "fixed_drive", "play_id"])

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in usable.to_dicts():
        key = (str(row["game_id"]), str(row["fixed_drive"]), str(row["posteam"]))
        grouped[key].append(row)

    drive_rows: list[dict[str, object]] = []
    skipped = 0
    for (game_id, fixed_drive, posteam), rows in grouped.items():
        canonical_posteam = _canonical_team(posteam)
        base = _drive_row(game_id, fixed_drive, canonical_posteam, rows)
        if base is None:
            skipped += 1
            continue
        first = _first_scrimmage(rows)
        last = _last_event(rows)
        if first is None or last is None:
            skipped += 1
            continue
        meta = game_map[game_id]
        away = str(meta["away"])
        home = str(meta["home"])
        defense = home if canonical_posteam == away else away
        start_margin = first.get("score_differential")
        if start_margin is None:
            start_margin = _number(first, "posteam_score") - _number(first, "defteam_score")
        start_yardline = float(base["start_yardline_100"])
        base.update(
            {
                "game": str(meta["game"]),
                "world": -1,
                "drive_index": int(float(fixed_drive))
                if str(fixed_drive).replace(".", "", 1).isdigit()
                else -1,
                "defense_team_id": defense,
                "start_quarter": int(_number(first, "qtr", 1.0)),
                "end_quarter": int(_number(last, "qtr", _number(first, "qtr", 1.0))),
                "start_score_margin": float(start_margin),
                "start_zone": _start_zone(start_yardline),
                "had_explosive_play": int(base["explosive_plays"]) > 0,
            }
        )
        drive_rows.append(base)

    manifest = {
        "season": season,
        "week": week,
        "games": len(schedule_rows),
        "definition_safe_drives": len(drive_rows),
        "groups_without_definition_safe_drive": skipped,
        "matchups": sorted(wanted),
        "postgame_truth_used_only_for_audit": True,
        "simulation_inputs_modified": False,
    }
    return drive_rows, manifest


def _summary(
    rows: list[dict[str, object]], *, team_games: int, scope: str
) -> dict[str, object]:
    if not rows:
        raise ValueError(f"no drive rows for {scope}")
    drives = len(rows)
    terminals = [str(row.get("terminal") or "") for row in rows]
    points = [float(row.get("points") or 0.0) for row in rows]
    yards = [float(row.get("net_scrimmage_yards") or 0.0) for row in rows]
    plays = [float(row.get("scrimmage_plays") or 0.0) for row in rows]
    first_downs = [float(row.get("first_downs") or 0.0) for row in rows]
    starts = [float(row.get("start_yardline_100") or 0.0) for row in rows]

    def count_terminal(name: str) -> int:
        return sum(terminal == name for terminal in terminals)

    def subset(predicate) -> list[dict[str, object]]:
        return [row for row in rows if predicate(row)]

    touchdown_rows = subset(lambda row: str(row.get("terminal")) == "touchdown")
    rz_reach = subset(lambda row: bool(row.get("red_zone_entered")))
    rz_snap = subset(lambda row: bool(row.get("red_zone_snap_seen")))
    explosive = subset(lambda row: int(row.get("explosive_plays") or 0) > 0)
    no_explosive = subset(lambda row: int(row.get("explosive_plays") or 0) == 0)
    short_field = subset(
        lambda row: float(row.get("start_yardline_100") or 0.0) >= 50.0
    )
    long40 = subset(
        lambda row: float(row.get("net_scrimmage_yards") or 0.0) >= 40.0
    )
    long50 = subset(
        lambda row: float(row.get("net_scrimmage_yards") or 0.0) >= 50.0
    )

    def td_rate(group: list[dict[str, object]]) -> float:
        return _safe_div(
            sum(str(row.get("terminal")) == "touchdown" for row in group), len(group)
        )

    def zero_rate(group: list[dict[str, object]]) -> float:
        return _safe_div(
            sum(float(row.get("points") or 0.0) == 0.0 for row in group), len(group)
        )

    series_started = sum(int(row.get("series_started") or 0) for row in rows)
    series_converted = sum(int(row.get("series_converted") or 0) for row in rows)
    third_snaps = sum(int(row.get("third_down_snaps") or 0) for row in rows)
    third_conversions = sum(
        int(row.get("third_down_conversions") or 0) for row in rows
    )
    third_long_snaps = sum(
        int(row.get("third_and_long_snaps") or 0) for row in rows
    )
    third_long_conversions = sum(
        int(row.get("third_and_long_conversions") or 0) for row in rows
    )
    fourth_snaps = sum(int(row.get("fourth_down_snaps") or 0) for row in rows)
    early_run_snaps = sum(int(row.get("early_down_run_snaps") or 0) for row in rows)
    early_pass_snaps = sum(
        int(row.get("early_down_pass_snaps") or 0) for row in rows
    )
    early_run_yards = sum(
        float(row.get("early_down_run_yards_total") or 0.0) for row in rows
    )
    early_pass_yards = sum(
        float(row.get("early_down_pass_yards_total") or 0.0) for row in rows
    )
    early_run_5plus = sum(
        int(row.get("early_down_run_5plus_gains") or 0) for row in rows
    )
    early_pass_5plus = sum(
        int(row.get("early_down_pass_5plus_gains") or 0) for row in rows
    )
    early_run_negative = sum(
        int(row.get("early_down_run_negative_gains") or 0) for row in rows
    )
    early_pass_negative = sum(
        int(row.get("early_down_pass_negative_gains") or 0) for row in rows
    )
    sacks = sum(int(row.get("sacks") or 0) for row in rows)
    turnovers = sum(int(row.get("turnovers") or 0) for row in rows)
    net_yards_total = sum(yards)
    points_total = sum(points)

    result: dict[str, object] = {
        "scope": scope,
        "drives": drives,
        "team_games": team_games,
        "drives_per_team_game": _safe_div(drives, team_games),
        "points_per_drive": _mean(points),
        "offensive_points_per_team_game": _safe_div(points_total, team_games),
        "mean_touchdown_drive_points": _mean(
            [float(row.get("points") or 0.0) for row in touchdown_rows]
        ),
        "zero_point_drive_rate": _safe_div(
            sum(point == 0.0 for point in points), drives
        ),
        "red_zone_reach_rate": _safe_div(len(rz_reach), drives),
        "red_zone_snap_rate": _safe_div(len(rz_snap), drives),
        "td_per_red_zone_reach": td_rate(rz_reach),
        "td_per_red_zone_snap": td_rate(rz_snap),
        "td_without_red_zone_snap_rate": _safe_div(
            sum(
                str(row.get("terminal")) == "touchdown"
                and not bool(row.get("red_zone_snap_seen"))
                for row in rows
            ),
            drives,
        ),
        "explosive_drive_rate": _safe_div(len(explosive), drives),
        "td_given_explosive_drive": td_rate(explosive),
        "td_given_no_explosive_drive": td_rate(no_explosive),
        "long_40_zero_rate": _safe_div(
            sum(float(row.get("points") or 0.0) == 0.0 for row in long40), drives
        ),
        "zero_given_long_40": zero_rate(long40),
        "long_50_zero_rate": _safe_div(
            sum(float(row.get("points") or 0.0) == 0.0 for row in long50), drives
        ),
        "zero_given_long_50": zero_rate(long50),
        "scrimmage_plays_per_drive": _mean(plays),
        "net_yards_per_drive": _mean(yards),
        "first_downs_per_drive": _mean(first_downs),
        "series_conversion_rate": _safe_div(series_converted, series_started),
        "third_down_conversion_rate": _safe_div(third_conversions, third_snaps),
        "third_and_long_conversion_rate": _safe_div(
            third_long_conversions, third_long_snaps
        ),
        "fourth_down_snaps_per_drive": _safe_div(fourth_snaps, drives),
        "turnover_on_downs_per_fourth_down_snap": _safe_div(
            count_terminal("turnover_on_downs"), fourth_snaps
        ),
        "early_down_run_yards_per_snap": _safe_div(
            early_run_yards, early_run_snaps
        ),
        "early_down_pass_yards_per_snap": _safe_div(
            early_pass_yards, early_pass_snaps
        ),
        "early_down_run_5plus_rate": _safe_div(early_run_5plus, early_run_snaps),
        "early_down_pass_5plus_rate": _safe_div(
            early_pass_5plus, early_pass_snaps
        ),
        "early_down_run_negative_rate": _safe_div(
            early_run_negative, early_run_snaps
        ),
        "early_down_pass_negative_rate": _safe_div(
            early_pass_negative, early_pass_snaps
        ),
        "sacks_per_drive": _safe_div(sacks, drives),
        "turnovers_per_drive": _safe_div(turnovers, drives),
        "start_yardline_100_mean": _mean(starts),
        "short_field_start_rate": _safe_div(len(short_field), drives),
        "td_given_short_field": td_rate(short_field),
        "points_per_100_net_yards": 100.0
        * _safe_div(points_total, net_yards_total),
        "drive_yards_points_correlation": _corr(yards, points),
    }
    for terminal in TERMINALS:
        result[f"{terminal}_rate"] = _safe_div(count_terminal(terminal), drives)
    return result


def _relative_delta(value: float, reference: float) -> float | None:
    if abs(reference) <= 1e-12:
        return None
    return value / reference - 1.0


def _triage_signal(sim: float, actual: float, historical: float) -> str:
    sim_actual = _relative_delta(sim, actual)
    sim_hist = _relative_delta(sim, historical)
    actual_hist = _relative_delta(actual, historical)
    if sim_actual is None or sim_hist is None or actual_hist is None:
        return "insufficient_denominator"
    if abs(sim_hist) <= 0.08 and abs(actual_hist) >= 0.12:
        return "week2_specific_outlier_signal"
    if abs(sim_hist) >= 0.08 and np.sign(sim_actual) == np.sign(sim_hist):
        return "structural_model_mismatch_signal"
    if abs(sim_actual) <= 0.08:
        return "near_week2"
    return "mixed_signal"


def _comparison_rows(
    sim: dict[str, object],
    actual: dict[str, object],
    historical: dict[str, object],
) -> list[dict[str, object]]:
    rows = []
    for metric in COMPARE_METRICS:
        s = float(sim[metric])
        a = float(actual[metric])
        h = float(historical[metric])
        rows.append(
            {
                "metric": metric,
                "v726": s,
                "actual_week2": a,
                "historical_2025": h,
                "v726_minus_actual": s - a,
                "v726_vs_actual_relative": _relative_delta(s, a),
                "v726_minus_historical": s - h,
                "v726_vs_historical_relative": _relative_delta(s, h),
                "actual_minus_historical": a - h,
                "actual_vs_historical_relative": _relative_delta(a, h),
                "triage": _triage_signal(s, a, h),
            }
        )
    return rows


def _product_decomposition(
    sim_a: float, sim_b: float, actual_a: float, actual_b: float
) -> dict[str, float]:
    a_contribution = (sim_a - actual_a) * (sim_b + actual_b) / 2.0
    b_contribution = (sim_b - actual_b) * (sim_a + actual_a) / 2.0
    return {
        "simulated_product": sim_a * sim_b,
        "actual_product": actual_a * actual_b,
        "difference": sim_a * sim_b - actual_a * actual_b,
        "factor_a_contribution": a_contribution,
        "factor_b_contribution": b_contribution,
    }


def _state_bucket(margin: float) -> str:
    if margin <= -9:
        return "trailing_9plus"
    if margin <= -1:
        return "trailing_1_8"
    if margin == 0:
        return "tied"
    if margin <= 8:
        return "leading_1_8"
    return "leading_9plus"


def _quarter_seconds_remaining(quarter: int, game_seconds: float) -> float:
    if quarter <= 4:
        return max(game_seconds - (4 - quarter) * 900.0, 0.0)
    return max(game_seconds, 0.0)


def _segment_rows(
    rows: list[dict[str, object]], *, team_games: int, source: str
) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        quarter = int(row.get("start_quarter") or 1)
        margin = float(row.get("start_score_margin") or 0.0)
        groups[("quarter", f"Q{quarter}")].append(row)
        groups[("score_state", _state_bucket(margin))].append(row)
        q_seconds = _quarter_seconds_remaining(
            quarter, float(row.get("start_seconds_remaining") or 0.0)
        )
        half_segment = (
            "late_half"
            if quarter in {2, 4} and q_seconds <= 120.0
            else "not_late_half"
        )
        groups[("half_clock", half_segment)].append(row)

    output = []
    for (dimension, segment), group in sorted(groups.items()):
        summary = _summary(
            group, team_games=team_games, scope=f"{source}:{dimension}:{segment}"
        )
        output.append(
            {
                "source": source,
                "dimension": dimension,
                "segment": segment,
                "drives": summary["drives"],
                "points_per_drive": summary["points_per_drive"],
                "touchdown_rate": summary["touchdown_rate"],
                "field_goal_rate": summary["field_goal_rate"],
                "punt_rate": summary["punt_rate"],
                "turnover_rate": summary["turnover_rate"],
                "net_yards_per_drive": summary["net_yards_per_drive"],
                "third_down_conversion_rate": summary["third_down_conversion_rate"],
                "red_zone_snap_rate": summary["red_zone_snap_rate"],
                "td_per_red_zone_snap": summary["td_per_red_zone_snap"],
            }
        )
    return output


def _team_comparison(
    sim_rows: list[dict[str, object]],
    actual_rows: list[dict[str, object]],
    worlds: int,
) -> list[dict[str, object]]:
    actual_groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    sim_groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in actual_rows:
        actual_groups[(str(row["game"]), str(row["offense_team_id"]))].append(row)
    for row in sim_rows:
        sim_groups[(str(row["game"]), str(row["offense_team_id"]))].append(row)

    output = []
    for key in sorted(actual_groups):
        game, team = key
        actual = _summary(
            actual_groups[key], team_games=1, scope=f"actual:{game}:{team}"
        )
        sim = _summary(sim_groups[key], team_games=worlds, scope=f"v726:{game}:{team}")
        output.append(
            {
                "game": game,
                "team": team,
                "actual_drives": actual["drives"],
                "v726_drives_mean": sim["drives_per_team_game"],
                "actual_points_per_drive": actual["points_per_drive"],
                "v726_points_per_drive": sim["points_per_drive"],
                "points_per_drive_delta": float(sim["points_per_drive"])
                - float(actual["points_per_drive"]),
                "actual_offensive_points": actual["offensive_points_per_team_game"],
                "v726_offensive_points_mean": sim["offensive_points_per_team_game"],
                "offensive_points_delta": float(
                    sim["offensive_points_per_team_game"]
                )
                - float(actual["offensive_points_per_team_game"]),
                "actual_td_rate": actual["touchdown_rate"],
                "v726_td_rate": sim["touchdown_rate"],
                "actual_punt_rate": actual["punt_rate"],
                "v726_punt_rate": sim["punt_rate"],
                "actual_turnover_rate": actual["turnover_rate"],
                "v726_turnover_rate": sim["turnover_rate"],
                "actual_yards_per_drive": actual["net_yards_per_drive"],
                "v726_yards_per_drive": sim["net_yards_per_drive"],
                "actual_rz_snap_rate": actual["red_zone_snap_rate"],
                "v726_rz_snap_rate": sim["red_zone_snap_rate"],
                "actual_rz_td_conversion": actual["td_per_red_zone_snap"],
                "v726_rz_td_conversion": sim["td_per_red_zone_snap"],
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--week", type=int, default=2)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/monster"))
    parser.add_argument("--simulated", type=Path, required=True)
    parser.add_argument("--sim-manifest", type=Path, required=True)
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--historical-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    sim_manifest = json.loads(args.sim_manifest.read_text())
    historical_manifest = json.loads(args.historical_manifest.read_text())
    worlds = int(sim_manifest["worlds_per_game"])
    games = len(MAIN_SLATE_MATCHUPS)

    sim_rows = pl.read_csv(args.simulated).to_dicts()
    actual_rows, actual_manifest = _actual_drive_rows(
        season=args.season, week=args.week, cache_dir=args.cache_dir
    )
    historical_rows = pl.read_parquet(args.historical).to_dicts()

    sim_summary = _summary(
        sim_rows, team_games=games * 2 * worlds, scope="v726_week2_simulation"
    )
    actual_summary = _summary(
        actual_rows, team_games=games * 2, scope="actual_week2_main_slate"
    )
    historical_summary = _summary(
        historical_rows,
        team_games=int(historical_manifest["games"]) * 2,
        scope="historical_2025_regular_season",
    )

    comparison = _comparison_rows(sim_summary, actual_summary, historical_summary)
    scoring = _product_decomposition(
        float(sim_summary["drives_per_team_game"]),
        float(sim_summary["points_per_drive"]),
        float(actual_summary["drives_per_team_game"]),
        float(actual_summary["points_per_drive"]),
    )
    scoring["factor_a"] = "drives_per_team_game"
    scoring["factor_b"] = "points_per_drive"

    rz = _product_decomposition(
        float(sim_summary["red_zone_snap_rate"]),
        float(sim_summary["td_per_red_zone_snap"]),
        float(actual_summary["red_zone_snap_rate"]),
        float(actual_summary["td_per_red_zone_snap"]),
    )
    rz["factor_a"] = "red_zone_snap_rate"
    rz["factor_b"] = "td_per_red_zone_snap"

    td_contribution_sim = float(sim_summary["touchdown_rate"]) * float(
        sim_summary["mean_touchdown_drive_points"]
    )
    td_contribution_actual = float(actual_summary["touchdown_rate"]) * float(
        actual_summary["mean_touchdown_drive_points"]
    )
    fg_contribution_sim = 3.0 * float(sim_summary["field_goal_rate"])
    fg_contribution_actual = 3.0 * float(actual_summary["field_goal_rate"])

    structural = [
        row
        for row in comparison
        if row["triage"] == "structural_model_mismatch_signal"
    ]
    structural.sort(
        key=lambda row: abs(float(row["v726_vs_historical_relative"] or 0.0)),
        reverse=True,
    )
    week2_specific = [
        row for row in comparison if row["triage"] == "week2_specific_outlier_signal"
    ]
    week2_specific.sort(
        key=lambda row: abs(float(row["actual_vs_historical_relative"] or 0.0)),
        reverse=True,
    )

    diagnosis = {
        "offensive_scoring_gap_points_per_team_game": float(
            sim_summary["offensive_points_per_team_game"]
        )
        - float(actual_summary["offensive_points_per_team_game"]),
        "scoring_gap_decomposition": scoring,
        "touchdown_points_per_drive_contribution": {
            "v726": td_contribution_sim,
            "actual_week2": td_contribution_actual,
            "difference": td_contribution_sim - td_contribution_actual,
        },
        "field_goal_points_per_drive_contribution": {
            "v726": fg_contribution_sim,
            "actual_week2": fg_contribution_actual,
            "difference": fg_contribution_sim - fg_contribution_actual,
        },
        "red_zone_td_path_decomposition": rz,
        "outside_red_zone_td_rate_difference": float(
            sim_summary["td_without_red_zone_snap_rate"]
        )
        - float(actual_summary["td_without_red_zone_snap_rate"]),
        "v726_vs_2025_points_per_drive_relative": _relative_delta(
            float(sim_summary["points_per_drive"]),
            float(historical_summary["points_per_drive"]),
        ),
        "actual_week2_vs_2025_points_per_drive_relative": _relative_delta(
            float(actual_summary["points_per_drive"]),
            float(historical_summary["points_per_drive"]),
        ),
        "strongest_structural_model_mismatch_signals": structural[:12],
        "strongest_week2_specific_outlier_signals": week2_specific[:12],
        "interpretation_rule": (
            "A Week-2 miss is not a calibration target by itself. Structural changes "
            "require the same directional mismatch against the broader 2025 NFL baseline "
            "or a clearly definition-safe mechanism failure in the raw drive anatomy."
        ),
    }

    segment_rows = _segment_rows(
        sim_rows, team_games=games * 2 * worlds, source="v726"
    ) + _segment_rows(
        actual_rows, team_games=games * 2, source="actual_week2"
    )
    team_rows = _team_comparison(sim_rows, actual_rows, worlds)

    args.out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(actual_rows).write_parquet(
        args.out / "actual_week2_drive_traces.parquet"
    )
    pl.DataFrame([sim_summary, actual_summary, historical_summary]).write_csv(
        args.out / "drive_anatomy_overall.csv"
    )
    pl.DataFrame(comparison).write_csv(args.out / "drive_anatomy_comparison.csv")
    pl.DataFrame(segment_rows).write_csv(args.out / "drive_anatomy_by_state.csv")
    pl.DataFrame(team_rows).sort(
        "offensive_points_delta", descending=True
    ).write_csv(args.out / "drive_anatomy_by_team.csv")
    (args.out / "diagnosis.json").write_text(
        json.dumps(diagnosis, indent=2) + "\n"
    )

    manifest = {
        "artifact": "MONSTER V7.2.6 Week 2 Drive Anatomy Reality Audit",
        "season": args.season,
        "week": args.week,
        "games": games,
        "simulation_worlds_per_game": worlds,
        "simulation_source_runtime": sim_manifest.get("experiment", "v7.2.4"),
        "historical_baseline_season": historical_manifest.get("season", 2025),
        "actual": actual_manifest,
        "postgame_truth_used_only_for_audit": True,
        "simulation_behavior_changed_by_audit": False,
        "market_or_sportsbook_inputs_added": False,
        "direct_score_adjustment": False,
        "files": {
            "actual_raw_drives": "actual_week2_drive_traces.parquet",
            "overall": "drive_anatomy_overall.csv",
            "comparison": "drive_anatomy_comparison.csv",
            "state": "drive_anatomy_by_state.csv",
            "team": "drive_anatomy_by_team.csv",
            "diagnosis": "diagnosis.json",
        },
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps({"manifest": manifest, "diagnosis": diagnosis}, indent=2))


if __name__ == "__main__":
    main()
