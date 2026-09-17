from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def _f(value: object, default: float = 0.0) -> float:
    try:
        return float(value) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _mean(values: Iterable[float]) -> float:
    xs = list(values)
    return statistics.fmean(xs) if xs else float("nan")


def _rmse(values: Iterable[float]) -> float:
    xs = list(values)
    return math.sqrt(statistics.fmean(x * x for x in xs)) if xs else float("nan")


def _sd(values: Iterable[float]) -> float:
    xs = list(values)
    return statistics.stdev(xs) if len(xs) >= 2 else 0.0


def empirical_percentile(samples: Iterable[float], actual: float) -> float:
    xs = list(samples)
    if not xs:
        return float("nan")
    less = sum(x < actual for x in xs)
    equal = sum(x == actual for x in xs)
    return (less + 0.5 * equal) / len(xs)


def empirical_crps(samples: Iterable[float], actual: float) -> float:
    xs = sorted(float(x) for x in samples)
    n = len(xs)
    if n == 0:
        return float("nan")
    term1 = sum(abs(x - actual) for x in xs) / n
    term2 = sum((2 * i - n - 1) * x for i, x in enumerate(xs, start=1)) / (n * n)
    return term1 - term2


def _quantile(samples: Iterable[float], q: float) -> float:
    xs = sorted(float(x) for x in samples)
    if not xs:
        return float("nan")
    if len(xs) == 1:
        return xs[0]
    h = (len(xs) - 1) * q
    lo = math.floor(h)
    hi = math.ceil(h)
    if lo == hi:
        return xs[lo]
    return xs[lo] + (h - lo) * (xs[hi] - xs[lo])


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _truth_games(path: Path) -> dict[str, dict[str, object]]:
    result = {}
    for row in _read(path):
        result[row["game"]] = {
            **row,
            "away_points": int(row["away_points"]),
            "home_points": int(row["home_points"]),
            "overtime": str(row["overtime"]).lower() in {"1", "true", "yes"},
        }
    return result


def _game_audit(simulation: Path, truth_path: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    truth = _truth_games(truth_path)
    world_rows = _read(simulation / "football_weirdness_worlds.csv")
    worlds: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in world_rows:
        if row["game"] in truth:
            worlds[row["game"]].append((_f(row["away_points"]), _f(row["home_points"])))
    missing = sorted(set(truth) - set(worlds))
    if missing:
        raise ValueError(f"simulation is missing benchmark games: {missing}")

    rows: list[dict[str, object]] = []
    for game, actual in truth.items():
        pairs = worlds[game]
        away_samples = [a for a, _ in pairs]
        home_samples = [h for _, h in pairs]
        total_samples = [a + h for a, h in pairs]
        margin_samples = [a - h for a, h in pairs]
        away_actual = float(actual["away_points"])
        home_actual = float(actual["home_points"])
        total_actual = away_actual + home_actual
        margin_actual = away_actual - home_actual
        away_mean = _mean(away_samples)
        home_mean = _mean(home_samples)
        total_mean = _mean(total_samples)
        margin_mean = _mean(margin_samples)
        away_win = _mean(1.0 if a > h else 0.0 for a, h in pairs)
        home_win = _mean(1.0 if h > a else 0.0 for a, h in pairs)
        tie = _mean(1.0 if h == a else 0.0 for a, h in pairs)
        actual_winner = "TIE" if away_actual == home_actual else (actual["away"] if away_actual > home_actual else actual["home"])
        projected_winner = "TIE" if away_win == home_win else (actual["away"] if away_win > home_win else actual["home"])
        p05, p10, p90, p95 = (_quantile(total_samples, q) for q in (0.05, 0.10, 0.90, 0.95))
        rows.append(
            {
                "game": game,
                "worlds": len(pairs),
                "actual_away_points": int(away_actual),
                "actual_home_points": int(home_actual),
                "actual_total": int(total_actual),
                "actual_margin_away_minus_home": int(margin_actual),
                "sim_away_mean": away_mean,
                "sim_home_mean": home_mean,
                "sim_total_mean": total_mean,
                "sim_margin_mean": margin_mean,
                "sim_total_sd": _sd(total_samples),
                "sim_margin_sd": _sd(margin_samples),
                "away_score_error": away_mean - away_actual,
                "home_score_error": home_mean - home_actual,
                "total_error": total_mean - total_actual,
                "margin_error": margin_mean - margin_actual,
                "actual_total_percentile": empirical_percentile(total_samples, total_actual),
                "actual_margin_percentile": empirical_percentile(margin_samples, margin_actual),
                "away_score_crps": empirical_crps(away_samples, away_actual),
                "home_score_crps": empirical_crps(home_samples, home_actual),
                "total_crps": empirical_crps(total_samples, total_actual),
                "margin_crps": empirical_crps(margin_samples, margin_actual),
                "sim_total_p05": p05,
                "sim_total_p10": p10,
                "sim_total_p90": p90,
                "sim_total_p95": p95,
                "actual_total_in_p10_p90": p10 <= total_actual <= p90,
                "actual_total_in_p05_p95": p05 <= total_actual <= p95,
                "sim_away_win_probability": away_win,
                "sim_home_win_probability": home_win,
                "sim_tie_probability": tie,
                "projected_winner": projected_winner,
                "actual_winner": actual_winner,
                "winner_correct": projected_winner == actual_winner,
            }
        )

    total_errors = [_f(r["total_error"]) for r in rows]
    margin_errors = [_f(r["margin_error"]) for r in rows]
    team_errors = [x for r in rows for x in (_f(r["away_score_error"]), _f(r["home_score_error"]))]
    expected_totals = [_f(r["sim_total_mean"]) for r in rows]
    actual_totals = [_f(r["actual_total"]) for r in rows]
    within_sds = [_f(r["sim_total_sd"]) for r in rows]
    between_sim = _sd(expected_totals)
    between_actual = _sd(actual_totals)
    summary = {
        "games": len(rows),
        "worlds_per_game_min": min(int(r["worlds"]) for r in rows),
        "worlds_per_game_max": max(int(r["worlds"]) for r in rows),
        "team_points_mae": _mean(abs(x) for x in team_errors),
        "team_points_rmse": _rmse(team_errors),
        "game_total_mae": _mean(abs(x) for x in total_errors),
        "game_total_rmse": _rmse(total_errors),
        "game_total_bias": _mean(total_errors),
        "margin_mae": _mean(abs(x) for x in margin_errors),
        "margin_rmse": _rmse(margin_errors),
        "winner_accuracy": _mean(1.0 if r["winner_correct"] else 0.0 for r in rows),
        "total_crps_mean": _mean(_f(r["total_crps"]) for r in rows),
        "margin_crps_mean": _mean(_f(r["margin_crps"]) for r in rows),
        "between_matchup_expected_total_sd": between_sim,
        "actual_game_total_sd": between_actual,
        "between_matchup_sd_ratio_sim_to_actual": between_sim / between_actual if between_actual else float("nan"),
        "mean_within_game_total_sd": _mean(within_sds),
        "between_to_within_sd_ratio": between_sim / _mean(within_sds) if _mean(within_sds) else float("nan"),
        "actual_total_p10_p90_coverage": _mean(1.0 if r["actual_total_in_p10_p90"] else 0.0 for r in rows),
        "actual_total_p05_p95_coverage": _mean(1.0 if r["actual_total_in_p05_p95"] else 0.0 for r in rows),
    }
    if len(rows) >= 2 and _sd(expected_totals) and _sd(actual_totals):
        summary["expected_total_vs_actual_correlation"] = statistics.correlation(expected_totals, actual_totals)
    else:
        summary["expected_total_vs_actual_correlation"] = float("nan")
    return rows, summary


PLAYER_STATS = (
    ("passing_attempts", "pass_attempts"),
    ("completions", "completions"),
    ("passing_yards", "passing_yards"),
    ("passing_tds", "passing_tds"),
    ("interceptions", "interceptions"),
    ("targets", "targets"),
    ("receptions", "receptions"),
    ("receiving_yards", "receiving_yards"),
    ("receiving_tds", "receiving_tds"),
    ("carries", "rush_attempts"),
    ("rushing_yards", "rushing_yards"),
    ("rushing_tds", "rushing_tds"),
    ("fumbles_lost", "fumbles_lost"),
)


def _name_key(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def _player_audit(simulation: Path, truth_path: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    truth_rows = _read(truth_path)
    sim_rows = _read(simulation / "player_world_fanduel.csv")
    by_id: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    by_name: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in sim_rows:
        by_id[(row["game"], row.get("player_id", ""))].append(row)
        by_name[(row["game"], _name_key(row.get("player", "")))].append(row)

    rows: list[dict[str, object]] = []
    unmatched: list[str] = []
    for actual in truth_rows:
        game = actual["game"]
        pid = actual.get("player_id", "")
        samples = by_id.get((game, pid), []) if pid else []
        match_method = "player_id" if samples else ""
        if not samples:
            samples = by_name.get((game, _name_key(actual.get("player_name", ""))), [])
            match_method = "name" if samples else ""
        actual_fd = _f(actual.get("fanduel_points"))
        actual_activity = sum(abs(_f(actual.get(k))) for k, _ in PLAYER_STATS)
        if not samples:
            if actual_activity > 0 or abs(actual_fd) > 0:
                unmatched.append(f"{game}:{pid}:{actual.get('player_name','')}")
            continue
        fd_samples = [_f(r["fanduel_points"]) for r in samples]
        row_out: dict[str, object] = {
            "game": game,
            "team": actual.get("team", ""),
            "player_id": pid,
            "player": actual.get("player_name", ""),
            "position": actual.get("position", ""),
            "match_method": match_method,
            "worlds": len(samples),
            "actual_fanduel": actual_fd,
            "sim_fanduel_mean": _mean(fd_samples),
            "fanduel_error": _mean(fd_samples) - actual_fd,
            "actual_fanduel_percentile": empirical_percentile(fd_samples, actual_fd),
            "fanduel_crps": empirical_crps(fd_samples, actual_fd),
            "sim_fanduel_p05": _quantile(fd_samples, 0.05),
            "sim_fanduel_p10": _quantile(fd_samples, 0.10),
            "sim_fanduel_p50": _quantile(fd_samples, 0.50),
            "sim_fanduel_p90": _quantile(fd_samples, 0.90),
            "sim_fanduel_p95": _quantile(fd_samples, 0.95),
        }
        row_out["actual_fanduel_in_p10_p90"] = row_out["sim_fanduel_p10"] <= actual_fd <= row_out["sim_fanduel_p90"]
        row_out["actual_fanduel_in_p05_p95"] = row_out["sim_fanduel_p05"] <= actual_fd <= row_out["sim_fanduel_p95"]
        for actual_col, sim_col in PLAYER_STATS:
            actual_value = _f(actual.get(actual_col))
            sim_mean = _mean(_f(s.get(sim_col)) for s in samples)
            row_out[f"actual_{actual_col}"] = actual_value
            row_out[f"sim_{actual_col}_mean"] = sim_mean
            row_out[f"{actual_col}_error"] = sim_mean - actual_value
        rows.append(row_out)

    fd_errors = [_f(r["fanduel_error"]) for r in rows]
    summary: dict[str, object] = {
        "matched_players": len(rows),
        "unmatched_active_truth_players": len(unmatched),
        "unmatched_active_truth_player_examples": unmatched[:25],
        "fanduel_mae": _mean(abs(x) for x in fd_errors),
        "fanduel_rmse": _rmse(fd_errors),
        "fanduel_bias": _mean(fd_errors),
        "fanduel_crps_mean": _mean(_f(r["fanduel_crps"]) for r in rows),
        "fanduel_p10_p90_coverage": _mean(1.0 if r["actual_fanduel_in_p10_p90"] else 0.0 for r in rows),
        "fanduel_p05_p95_coverage": _mean(1.0 if r["actual_fanduel_in_p05_p95"] else 0.0 for r in rows),
    }
    for actual_col, _ in PLAYER_STATS:
        errs = [_f(r[f"{actual_col}_error"]) for r in rows]
        summary[f"{actual_col}_mae"] = _mean(abs(x) for x in errs)
        summary[f"{actual_col}_bias"] = _mean(errs)
    return rows, summary


def _role_audit(simulation: Path, player_truth_path: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    truth = _read(player_truth_path)
    actual_by_id = {(r["game"], r.get("player_id", "")): r for r in truth if r.get("player_id")}
    actual_by_name = {(r["game"], _name_key(r.get("player_name", ""))): r for r in truth}

    # Reality owns the opportunity denominator. This is intentionally computed
    # from every realized Week-1 player, not only players present in Monster's
    # role plan. An omitted real contributor therefore remains a measurable miss.
    truth_totals: dict[tuple[str, str, str], float] = defaultdict(float)
    for actual in truth:
        truth_totals[(actual["game"], actual["team"], "rush")] += _f(actual.get("carries"))
        truth_totals[(actual["game"], actual["team"], "target")] += _f(actual.get("targets"))

    rows: list[dict[str, object]] = []
    modeled_keys: set[tuple[str, str, str, str]] = set()

    def append_plan_row(
        *,
        game: str,
        team: str,
        player_id: str,
        player: str,
        position: str,
        role_type: str,
        plan_share: float,
        base_share: float,
        active_probability: float,
    ) -> None:
        actual = (
            actual_by_id.get((game, player_id))
            or actual_by_name.get((game, _name_key(player)))
        )
        actual_opportunities = 0.0
        if actual:
            actual_opportunities = _f(
                actual.get("carries" if role_type == "rush" else "targets")
            )
        rows.append({
            "game": game,
            "team": team,
            "player_id": player_id,
            "player": player,
            "position": position,
            "role_type": role_type,
            "plan_share_mean": plan_share,
            "base_share": base_share,
            "active_probability": active_probability,
            "actual_opportunities": actual_opportunities,
            "modeled": True,
        })
        modeled_keys.add((game, team, role_type, player_id))

    rush_path = simulation / "rushing_role_plan_audit.csv"
    if rush_path.exists():
        for plan in _read(rush_path):
            append_plan_row(
                game=plan["game"],
                team=plan["team"],
                player_id=plan.get("player_id", ""),
                player=plan.get("player", ""),
                position=plan.get("position", ""),
                role_type="rush",
                plan_share=_f(plan.get("plan_share_mean")),
                base_share=_f(plan.get("base_rush_share")),
                active_probability=_f(plan.get("active_probability")),
            )

    target_path = simulation / "target_role_plan_audit_v63.csv"
    if target_path.exists():
        team_to_game = {r["team"]: r["game"] for r in truth}
        for plan in _read(target_path):
            game = team_to_game.get(plan["team"], "")
            append_plan_row(
                game=game,
                team=plan["team"],
                player_id=plan.get("player_id", ""),
                player=plan.get("player", ""),
                position=plan.get("position", ""),
                role_type="target",
                plan_share=_f(plan.get("target_plan_share_mean")),
                base_share=_f(plan.get("base_target_share")),
                active_probability=_f(plan.get("active_probability")),
            )

    # Add realized opportunity earners absent from Monster's role plan as
    # zero-share predictions rather than silently dropping them.
    for actual in truth:
        for role_type, col in (("rush", "carries"), ("target", "targets")):
            opportunities = _f(actual.get(col))
            if opportunities <= 0:
                continue
            key = (
                actual["game"],
                actual["team"],
                role_type,
                actual.get("player_id", ""),
            )
            if key in modeled_keys:
                continue
            rows.append({
                "game": actual["game"],
                "team": actual["team"],
                "player_id": actual.get("player_id", ""),
                "player": actual.get("player_name", ""),
                "position": actual.get("position", ""),
                "role_type": role_type,
                "plan_share_mean": 0.0,
                "base_share": 0.0,
                "active_probability": 0.0,
                "actual_opportunities": opportunities,
                "modeled": False,
            })

    for row in rows:
        total = truth_totals[
            (str(row["game"]), str(row["team"]), str(row["role_type"]))
        ]
        actual_share = _f(row["actual_opportunities"]) / total if total else 0.0
        row["actual_team_opportunities"] = total
        row["actual_share"] = actual_share
        row["share_error"] = _f(row["plan_share_mean"]) - actual_share
        row["share_abs_error"] = abs(_f(row["share_error"]))

    summary: dict[str, object] = {"rows": len(rows)}
    for role in ("rush", "target"):
        selected = [
            r for r in rows
            if r["role_type"] == role and _f(r["actual_opportunities"]) > 0
        ]
        missing = [r for r in selected if not bool(r["modeled"])]
        summary[f"{role}_active_player_share_mae"] = _mean(
            _f(r["share_abs_error"]) for r in selected
        )
        summary[f"{role}_active_players"] = len(selected)
        summary[f"{role}_unmodeled_actual_players"] = len(missing)
        summary[f"{role}_unmodeled_actual_opportunities"] = sum(
            _f(r["actual_opportunities"]) for r in missing
        )

        by_team_role: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
        for row in rows:
            if row["role_type"] == role:
                by_team_role[(str(row["game"]), str(row["team"]))].append(row)
        tvds = [
            0.5 * sum(_f(r["share_abs_error"]) for r in team_rows)
            for team_rows in by_team_role.values()
            if _f(team_rows[0].get("actual_team_opportunities")) > 0
        ]
        summary[f"{role}_team_share_tvd_mean"] = _mean(tvds)

    return rows, summary

def main() -> None:
    parser = argparse.ArgumentParser(description="Grade Monster v6.3 Week 1 worlds against frozen reality.")
    parser.add_argument("--simulation", type=Path, required=True)
    parser.add_argument("--games", type=Path, default=Path("benchmarks/week1_2026_sunday_games.csv"))
    parser.add_argument("--player-truth", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label", default="v63-baseline")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    game_rows, game_summary = _game_audit(args.simulation, args.games)
    player_rows, player_summary = _player_audit(args.simulation, args.player_truth)
    role_rows, role_summary = _role_audit(args.simulation, args.player_truth)

    _write(args.out / "week1_game_reality.csv", game_rows, list(game_rows[0].keys()))
    if player_rows:
        _write(args.out / "week1_player_reality.csv", player_rows, list(player_rows[0].keys()))
    if role_rows:
        _write(args.out / "week1_role_reality.csv", role_rows, list(role_rows[0].keys()))

    manifest = {}
    manifest_path = args.simulation / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = {
        "label": args.label,
        "audit_version": "week1-reality-benchmark-v1",
        "anti_leakage": {
            "final_truth_consumed_only_by_audit": True,
            "simulation_directory": str(args.simulation),
            "simulation_market_blind_football": manifest.get("market_blind_football"),
            "simulation_scoreboard_event_derived": manifest.get("scoreboard_event_derived"),
            "simulation_fanduel_scoring_downstream_only": manifest.get("fanduel_scoring_downstream_only"),
        },
        "simulation_manifest": {
            "model": manifest.get("model"), "season": manifest.get("season"), "week": manifest.get("week"),
            "worlds_per_game": manifest.get("worlds_per_game"), "seed": manifest.get("seed"),
        },
        "game": game_summary,
        "player": player_summary,
        "role": role_summary,
        "interpretation_guards": [
            "A single realized week is evidence, not a target to fit.",
            "Promotion decisions require paired seeds and causal mechanism review, not lower point error alone.",
            "Final Week 1 outcomes must never feed simulation inputs or parameter selection for this benchmark replay.",
        ],
    }
    (args.out / "week1_reality_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=True))


if __name__ == "__main__":
    main()
