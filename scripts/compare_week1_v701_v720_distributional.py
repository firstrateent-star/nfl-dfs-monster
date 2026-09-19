from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections.abc import Iterable
from collections import defaultdict
from pathlib import Path


TEAM_ALIASES = {"JAX": "JAC", "LA": "LAR"}


def _team(value: object) -> str:
    text = str(value or "")
    return TEAM_ALIASES.get(text, text)


def _name(value: object) -> str:
    return "".join(ch for ch in str(value or "").lower() if ch.isalnum())


def _f(value: object, default: float = 0.0) -> float:
    try:
        if value in (None, "", "nan", "NaN"):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean(values: Iterable[float]) -> float:
    xs = [float(value) for value in values]
    return statistics.fmean(xs) if xs else float("nan")


def _sd(values: Iterable[float]) -> float:
    xs = [float(value) for value in values]
    return statistics.stdev(xs) if len(xs) >= 2 else 0.0


def _quantile(values: Iterable[float], q: float) -> float:
    xs = sorted(float(value) for value in values)
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


def _crps(values: Iterable[float], actual: float) -> float:
    xs = sorted(float(value) for value in values)
    n = len(xs)
    if n == 0:
        return float("nan")
    term1 = sum(abs(value - actual) for value in xs) / n
    term2 = sum(
        (2 * index - n - 1) * value
        for index, value in enumerate(xs, start=1)
    ) / (n * n)
    return term1 - term2


def _correlation(x: Iterable[float], y: Iterable[float]) -> float:
    xs = [float(value) for value in x]
    ys = [float(value) for value in y]
    if len(xs) < 2 or len(xs) != len(ys):
        return float("nan")
    if _sd(xs) == 0.0 or _sd(ys) == 0.0:
        return float("nan")
    return statistics.correlation(xs, ys)


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while (
            end < len(order)
            and values[order[end]] == values[order[cursor]]
        ):
            end += 1
        average_rank = (cursor + 1 + end) / 2.0
        for offset in range(cursor, end):
            ranks[order[offset]] = average_rank
        cursor = end
    return ranks


def _spearman(x: Iterable[float], y: Iterable[float]) -> float:
    xs = [float(value) for value in x]
    ys = [float(value) for value in y]
    if len(xs) < 2 or len(xs) != len(ys):
        return float("nan")
    return _correlation(_rank(xs), _rank(ys))


def _read(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _clean_json(value: object) -> object:
    if isinstance(value, dict):
        return {key: _clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _actual_context(
    schedules_path: Path,
    player_path: Path,
) -> tuple[
    dict[str, dict[str, object]],
    dict[str, str],
    list[dict[str, str]],
]:
    games: dict[str, dict[str, object]] = {}
    team_game: dict[str, str] = {}
    for row in _read(schedules_path):
        away = _team(row.get("away_team"))
        home = _team(row.get("home_team"))
        game = f"{away}@{home}"
        games[game] = {
            "game": game,
            "away": away,
            "home": home,
            "away_points": _f(row.get("away_score")),
            "home_points": _f(row.get("home_score")),
        }
        team_game[away] = game
        team_game[home] = game

    players = _read(player_path)
    for row in players:
        row["team"] = _team(row.get("team"))
        row["game"] = team_game.get(str(row["team"]), "")
    return games, team_game, players


def _scoreboard_arm(
    simulation: Path,
    actual_games: dict[str, dict[str, object]],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    rows = _read(simulation / "football_weirdness_worlds.csv")
    by_game: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_game[row["game"]].append(row)

    detail: list[dict[str, object]] = []
    team_errors: list[float] = []
    total_errors: list[float] = []
    margin_errors: list[float] = []
    total_crps: list[float] = []
    margin_crps: list[float] = []
    expected_totals: list[float] = []
    actual_totals: list[float] = []
    within_total_sd: list[float] = []
    winners: list[float] = []
    p10_90: list[float] = []
    p05_95: list[float] = []

    for game, truth in actual_games.items():
        samples = by_game.get(game, [])
        if not samples:
            continue
        away = [_f(row["away_points"]) for row in samples]
        home = [_f(row["home_points"]) for row in samples]
        totals = [a + h for a, h in zip(away, home)]
        margins = [a - h for a, h in zip(away, home)]
        actual_away = _f(truth["away_points"])
        actual_home = _f(truth["home_points"])
        actual_total = actual_away + actual_home
        actual_margin = actual_away - actual_home
        away_mean = _mean(away)
        home_mean = _mean(home)
        total_mean = _mean(totals)
        margin_mean = _mean(margins)
        away_win = _mean(1.0 if a > h else 0.0 for a, h in zip(away, home))
        home_win = _mean(1.0 if h > a else 0.0 for a, h in zip(away, home))
        actual_winner = (
            "TIE"
            if actual_away == actual_home
            else truth["away"]
            if actual_away > actual_home
            else truth["home"]
        )
        projected_winner = (
            "TIE"
            if away_win == home_win
            else truth["away"]
            if away_win > home_win
            else truth["home"]
        )
        p10 = _quantile(totals, 0.10)
        p90 = _quantile(totals, 0.90)
        p05 = _quantile(totals, 0.05)
        p95 = _quantile(totals, 0.95)
        row_out = {
            "game": game,
            "worlds": len(samples),
            "actual_total": actual_total,
            "actual_margin": actual_margin,
            "sim_away_mean": away_mean,
            "sim_home_mean": home_mean,
            "sim_total_mean": total_mean,
            "sim_margin_mean": margin_mean,
            "sim_total_sd": _sd(totals),
            "sim_margin_sd": _sd(margins),
            "team_points_abs_error": (
                abs(away_mean - actual_away)
                + abs(home_mean - actual_home)
            )
            / 2.0,
            "total_abs_error": abs(total_mean - actual_total),
            "margin_abs_error": abs(margin_mean - actual_margin),
            "total_crps": _crps(totals, actual_total),
            "margin_crps": _crps(margins, actual_margin),
            "actual_total_in_p10_p90": p10 <= actual_total <= p90,
            "actual_total_in_p05_p95": p05 <= actual_total <= p95,
            "projected_winner": projected_winner,
            "actual_winner": actual_winner,
            "winner_correct": projected_winner == actual_winner,
        }
        detail.append(row_out)
        team_errors.extend(
            [away_mean - actual_away, home_mean - actual_home]
        )
        total_errors.append(total_mean - actual_total)
        margin_errors.append(margin_mean - actual_margin)
        total_crps.append(_f(row_out["total_crps"]))
        margin_crps.append(_f(row_out["margin_crps"]))
        expected_totals.append(total_mean)
        actual_totals.append(actual_total)
        within_total_sd.append(_sd(totals))
        winners.append(1.0 if row_out["winner_correct"] else 0.0)
        p10_90.append(1.0 if row_out["actual_total_in_p10_p90"] else 0.0)
        p05_95.append(1.0 if row_out["actual_total_in_p05_p95"] else 0.0)

    return {
        "games": len(detail),
        "team_points_mae": _mean(abs(value) for value in team_errors),
        "game_total_mae": _mean(abs(value) for value in total_errors),
        "margin_mae": _mean(abs(value) for value in margin_errors),
        "game_total_bias": _mean(total_errors),
        "total_crps_mean": _mean(total_crps),
        "margin_crps_mean": _mean(margin_crps),
        "winner_accuracy": _mean(winners),
        "expected_total_vs_actual_correlation": _correlation(
            expected_totals,
            actual_totals,
        ),
        "between_matchup_expected_total_sd": _sd(expected_totals),
        "actual_game_total_sd": _sd(actual_totals),
        "mean_within_game_total_sd": _mean(within_total_sd),
        "total_p10_p90_coverage": _mean(p10_90),
        "total_p05_p95_coverage": _mean(p05_95),
    }, detail


def _actual_ecology(players: list[dict[str, str]]) -> dict[str, float]:
    return {
        "pass_attempts": sum(_f(row.get("attempts")) for row in players),
        "completions": sum(_f(row.get("completions")) for row in players),
        "rush_attempts": sum(_f(row.get("carries")) for row in players),
        "sacks": sum(_f(row.get("def_sacks")) for row in players),
        "interceptions": sum(
            _f(row.get("passing_interceptions")) for row in players
        ),
        "fumbles_lost": sum(
            _f(row.get("fumbles_lost_total")) for row in players
        ),
        "punts": sum(_f(row.get("pt_att")) for row in players),
        "field_goal_attempts": sum(_f(row.get("fg_att")) for row in players),
        "field_goals_made": sum(_f(row.get("fg_made")) for row in players),
        "touchdowns": (
            sum(_f(row.get("passing_tds")) for row in players)
            + sum(_f(row.get("rushing_tds")) for row in players)
            + sum(_f(row.get("def_tds")) for row in players)
            + sum(_f(row.get("special_teams_tds")) for row in players)
        ),
        "explosive_20": (
            sum(_f(row.get("passing_20")) for row in players)
            + sum(_f(row.get("rushing_20")) for row in players)
        ),
        "explosive_40": (
            sum(_f(row.get("passing_40")) for row in players)
            + sum(_f(row.get("rushing_40")) for row in players)
        ),
        "punt_return_yards": sum(
            _f(row.get("punt_return_yards")) for row in players
        ),
        "kickoff_return_yards": sum(
            _f(row.get("kickoff_return_yards")) for row in players
        ),
    }


def _ecology_arm(
    simulation: Path,
    actual: dict[str, float],
) -> tuple[dict[str, object], dict[int, dict[str, float]]]:
    rows = _read(simulation / "football_weirdness_worlds.csv")
    fields = tuple(actual)
    by_world: dict[int, dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for row in rows:
        world = int(_f(row.get("world")))
        for field in fields:
            by_world[world][field] += _f(row.get(field))

    summary: dict[str, object] = {}
    for field in fields:
        values = [metrics[field] for metrics in by_world.values()]
        mean_value = _mean(values)
        actual_value = actual[field]
        summary[field] = {
            "actual": actual_value,
            "sim_mean": mean_value,
            "sim_sd": _sd(values),
            "error": mean_value - actual_value,
            "absolute_error": abs(mean_value - actual_value),
            "relative_error": (
                (mean_value - actual_value) / actual_value
                if actual_value
                else float("nan")
            ),
        }
    summary["mean_absolute_relative_error"] = _mean(
        abs(_f(value["relative_error"]))
        for value in summary.values()
        if isinstance(value, dict)
        and math.isfinite(_f(value.get("relative_error"), float("nan")))
    )
    return summary, by_world


OFFENSE_METRICS = {
    "pass_attempts": ("attempts", "pass_attempts_mean"),
    "completions": ("completions", "completions_mean"),
    "passing_yards": ("passing_yards", "passing_yards_mean"),
    "passing_tds": ("passing_tds", "passing_tds_mean"),
    "interceptions": (
        "passing_interceptions",
        "interceptions_mean",
    ),
    "targets": ("targets", "targets_mean"),
    "receptions": ("receptions", "receptions_mean"),
    "receiving_yards": ("receiving_yards", "receiving_yards_mean"),
    "receiving_tds": ("receiving_tds", "receiving_tds_mean"),
    "carries": ("carries", "rush_attempts_mean"),
    "rushing_yards": ("rushing_yards", "rushing_yards_mean"),
    "rushing_tds": ("rushing_tds", "rushing_tds_mean"),
}


def _metric_fit(
    actual: dict[tuple[str, str, str], float],
    simulated: dict[tuple[str, str, str], float],
) -> dict[str, object]:
    keys = sorted(set(actual) | set(simulated))
    keys = [
        key
        for key in keys
        if abs(actual.get(key, 0.0)) > 0.0
        or abs(simulated.get(key, 0.0)) > 0.01
    ]
    actual_values = [actual.get(key, 0.0) for key in keys]
    sim_values = [simulated.get(key, 0.0) for key in keys]
    errors = [
        simulated_value - actual_value
        for simulated_value, actual_value in zip(sim_values, actual_values)
    ]
    denominator = sum(abs(value) for value in actual_values)
    return {
        "rows": len(keys),
        "mae": _mean(abs(value) for value in errors),
        "bias": _mean(errors),
        "wape": (
            sum(abs(value) for value in errors) / denominator
            if denominator
            else float("nan")
        ),
        "pearson": _correlation(sim_values, actual_values),
        "spearman": _spearman(sim_values, actual_values),
    }


def _offense_player_arm(
    box: Path,
    players: list[dict[str, str]],
) -> dict[str, object]:
    sim_rows = _read(box / "offensive_player_box_score_distributions.csv")
    actual_maps: dict[str, dict[tuple[str, str, str], float]] = {
        metric: {} for metric in OFFENSE_METRICS
    }
    for row in players:
        key = (row["game"], row["team"], row.get("player_id", ""))
        if not key[0] or not key[2]:
            continue
        for metric, (actual_col, _) in OFFENSE_METRICS.items():
            actual_maps[metric][key] = _f(row.get(actual_col))

    sim_maps: dict[str, dict[tuple[str, str, str], float]] = {
        metric: {} for metric in OFFENSE_METRICS
    }
    for row in sim_rows:
        key = (
            row.get("game", ""),
            _team(row.get("team")),
            row.get("player_id", ""),
        )
        for metric, (_, sim_col) in OFFENSE_METRICS.items():
            sim_maps[metric][key] = _f(row.get(sim_col))

    return {
        metric: _metric_fit(actual_maps[metric], sim_maps[metric])
        for metric in OFFENSE_METRICS
    }


def _role_share_arm(
    box: Path,
    players: list[dict[str, str]],
    *,
    actual_col: str,
    sim_col: str,
) -> dict[str, object]:
    sim_rows = _read(box / "offensive_player_box_score_distributions.csv")
    actual_by_team: dict[
        tuple[str, str],
        dict[str, float],
    ] = defaultdict(dict)
    sim_by_team: dict[
        tuple[str, str],
        dict[str, float],
    ] = defaultdict(dict)

    for row in players:
        value = _f(row.get(actual_col))
        if value <= 0.0:
            continue
        team_key = (row["game"], row["team"])
        actual_by_team[team_key][row.get("player_id", "")] = value

    for row in sim_rows:
        value = _f(row.get(sim_col))
        if value <= 0.0:
            continue
        team_key = (row.get("game", ""), _team(row.get("team")))
        sim_by_team[team_key][row.get("player_id", "")] = value

    tvds: list[float] = []
    top_hits: list[float] = []
    top_share_errors: list[float] = []
    for team_key in sorted(set(actual_by_team) | set(sim_by_team)):
        actual = actual_by_team.get(team_key, {})
        simulated = sim_by_team.get(team_key, {})
        actual_total = sum(actual.values())
        sim_total = sum(simulated.values())
        if actual_total <= 0.0 or sim_total <= 0.0:
            continue
        keys = set(actual) | set(simulated)
        actual_share = {
            key: actual.get(key, 0.0) / actual_total for key in keys
        }
        sim_share = {
            key: simulated.get(key, 0.0) / sim_total for key in keys
        }
        tvds.append(
            0.5
            * sum(
                abs(sim_share[key] - actual_share[key])
                for key in keys
            )
        )
        actual_top = max(actual, key=actual.get)
        sim_top = max(simulated, key=simulated.get)
        top_hits.append(1.0 if actual_top == sim_top else 0.0)
        top_share_errors.append(
            abs(
                max(actual_share.values())
                - max(sim_share.values())
            )
        )

    return {
        "team_share_tvd_mean": _mean(tvds),
        "top_identity_hit_rate": _mean(top_hits),
        "top_share_mae": _mean(top_share_errors),
        "team_roles": len(tvds),
    }


def _team_offense_arm(
    box: Path,
    players: list[dict[str, str]],
) -> dict[str, object]:
    actual: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for row in players:
        key = (row["game"], row["team"])
        actual[key]["pass_yards"] += _f(row.get("passing_yards"))
        actual[key]["rush_yards"] += _f(row.get("rushing_yards"))
    for values in actual.values():
        values["total_yards"] = values["pass_yards"] + values["rush_yards"]

    sim_rows = _read(box / "team_box_score_distributions.csv")
    simulated: dict[tuple[str, str], dict[str, float]] = {}
    for row in sim_rows:
        simulated[(row["game"], _team(row["team"]))] = {
            "pass_yards": _f(row.get("pass_yards_mean")),
            "rush_yards": _f(row.get("rush_yards_mean")),
            "total_yards": _f(row.get("total_yards_mean")),
        }

    result: dict[str, object] = {}
    for metric in ("pass_yards", "rush_yards", "total_yards"):
        keys = sorted(set(actual) | set(simulated))
        a = [actual.get(key, {}).get(metric, 0.0) for key in keys]
        s = [simulated.get(key, {}).get(metric, 0.0) for key in keys]
        errors = [sv - av for sv, av in zip(s, a)]
        result[metric] = {
            "teams": len(keys),
            "mae": _mean(abs(value) for value in errors),
            "bias": _mean(errors),
            "pearson": _correlation(s, a),
            "spearman": _spearman(s, a),
        }
    return result


def _player_name_map(box: Path) -> dict[str, tuple[str, str, str]]:
    result: dict[str, tuple[str, str, str]] = {}
    for filename in (
        "offensive_player_box_score_distributions.csv",
        "defensive_player_box_score_distributions.csv",
    ):
        for row in _read(box / filename):
            player_id = row.get("player_id", "")
            if not player_id:
                continue
            result[player_id] = (
                str(row.get("player", "")),
                _team(row.get("team")),
                str(row.get("position", "")),
            )
    return result


def _participation_arm(
    simulation: Path,
    box: Path,
    snaps_truth: Path,
) -> dict[str, object]:
    name_map = _player_name_map(box)
    player_counts: dict[
        tuple[str, int, str, str, str],
        int,
    ] = defaultdict(int)
    team_counts: dict[tuple[str, int, str, str], int] = defaultdict(int)
    game_worlds: dict[str, set[int]] = defaultdict(set)

    for row in _read(simulation / "same_world_snap_participants_v5.csv"):
        game = row.get("game", "")
        world = int(_f(row.get("world")))
        snap_key = row.get("snap_key", "")
        if ">" not in snap_key:
            continue
        offense_team = _team(snap_key.split(">", 1)[0])
        defense_team = _team(
            snap_key.split(">", 1)[1].split(":", 1)[0]
        )
        game_worlds[game].add(world)
        team_counts[(game, world, offense_team, "offense")] += 1
        team_counts[(game, world, defense_team, "defense")] += 1
        for player_id in str(row.get("offense_participants", "")).split("|"):
            if player_id:
                player_counts[
                    (game, world, offense_team, "offense", player_id)
                ] += 1
        for player_id in str(row.get("defense_participants", "")).split("|"):
            if player_id:
                player_counts[
                    (game, world, defense_team, "defense", player_id)
                ] += 1

    share_sum: dict[tuple[str, str, str, str], float] = defaultdict(float)
    for (game, world, team, side, player_id), count in player_counts.items():
        denominator = team_counts[(game, world, team, side)]
        if denominator:
            share_sum[(game, team, side, player_id)] += count / denominator

    sim_by_name: dict[tuple[str, str, str], float] = {}
    for (game, team, side, player_id), total_share in share_sum.items():
        worlds = max(len(game_worlds.get(game, ())), 1)
        player_name = name_map.get(player_id, (player_id, team, ""))[0]
        sim_by_name[(team, side, _name(player_name))] = total_share / worlds

    actual_by_name: dict[tuple[str, str, str], tuple[float, str]] = {}
    for row in _read(snaps_truth):
        team = _team(row.get("team"))
        player_name = _name(row.get("player"))
        position = str(row.get("position", ""))
        actual_by_name[(team, "offense", player_name)] = (
            _f(row.get("offense_pct")),
            position,
        )
        actual_by_name[(team, "defense", player_name)] = (
            _f(row.get("defense_pct")),
            position,
        )

    result: dict[str, object] = {}
    for side in ("offense", "defense"):
        actual_values: list[float] = []
        sim_values: list[float] = []
        high_hits: list[float] = []
        high_errors: list[float] = []
        all_keys = {
            key for key in actual_by_name if key[1] == side
        } | {
            key for key in sim_by_name if key[1] == side
        }
        for key in all_keys:
            actual_share = actual_by_name.get(key, (0.0, ""))[0]
            sim_share = sim_by_name.get(key, 0.0)
            if actual_share >= 0.10 or sim_share >= 0.01:
                actual_values.append(actual_share)
                sim_values.append(sim_share)
            if actual_share >= 0.70:
                high_hits.append(1.0 if sim_share >= 0.50 else 0.0)
                high_errors.append(abs(sim_share - actual_share))
        result[side] = {
            "active_10pct_mae": _mean(
                abs(sv - av)
                for sv, av in zip(sim_values, actual_values)
            ),
            "active_10pct_pearson": _correlation(
                sim_values,
                actual_values,
            ),
            "high_snap_70pct_recognition": _mean(high_hits),
            "high_snap_70pct_mae": _mean(high_errors),
            "high_snap_players": len(high_hits),
        }

    ol_hits: list[float] = []
    ol_errors: list[float] = []
    for key, (actual_share, position) in actual_by_name.items():
        if key[1] != "offense":
            continue
        if position not in {"T", "G", "C", "OL"} or actual_share < 0.80:
            continue
        sim_share = sim_by_name.get(key, 0.0)
        ol_hits.append(1.0 if sim_share >= 0.50 else 0.0)
        ol_errors.append(abs(sim_share - actual_share))
    result["offensive_line"] = {
        "actual_80pct_starters": len(ol_hits),
        "starter_recognition": _mean(ol_hits),
        "snap_share_mae": _mean(ol_errors),
    }
    return result


def _defense_arm(
    box: Path,
    players: list[dict[str, str]],
    pfr_path: Path,
) -> dict[str, object]:
    sim_rows = _read(box / "defensive_player_box_score_distributions.csv")
    actual_by_id: dict[tuple[str, str, str], dict[str, float]] = {}
    for row in players:
        key = (row["game"], row["team"], row.get("player_id", ""))
        if not key[0] or not key[2]:
            continue
        actual_by_id[key] = {
            "sacks": _f(row.get("def_sacks")),
            "tackles": (
                _f(row.get("def_tackles_solo"))
                + _f(row.get("def_tackle_assists"))
            ),
            "interceptions": _f(row.get("def_interceptions")),
            "forced_fumbles": _f(row.get("def_fumbles_forced")),
        }

    pressure_by_name: dict[tuple[str, str], float] = {}
    for row in _read(pfr_path):
        pressure_by_name[
            (_team(row.get("team")), _name(row.get("pfr_player_name")))
        ] = _f(row.get("def_pressures"))

    metrics = {
        "pressures": "pressures_mean",
        "sacks": "sacks_mean",
        "tackles": "tackles_mean",
        "interceptions": "interceptions_mean",
        "forced_fumbles": "forced_fumbles_mean",
    }
    actual_maps: dict[str, dict[tuple[str, str, str], float]] = {
        metric: {} for metric in metrics
    }
    sim_maps: dict[str, dict[tuple[str, str, str], float]] = {
        metric: {} for metric in metrics
    }

    for key, values in actual_by_id.items():
        for metric in ("sacks", "tackles", "interceptions", "forced_fumbles"):
            actual_maps[metric][key] = values[metric]

    for row in sim_rows:
        key = (
            row.get("game", ""),
            _team(row.get("team")),
            row.get("player_id", ""),
        )
        actual_maps["pressures"][key] = pressure_by_name.get(
            (key[1], _name(row.get("player"))),
            0.0,
        )
        for metric, sim_col in metrics.items():
            sim_maps[metric][key] = _f(row.get(sim_col))

    return {
        metric: _metric_fit(actual_maps[metric], sim_maps[metric])
        for metric in metrics
    }


def _qb_family_arm(
    simulation: Path,
    box: Path,
    historical: dict[str, object],
) -> dict[str, object]:
    name_map = _player_name_map(box)
    qb_ids = {
        player_id
        for player_id, (_, _, position) in name_map.items()
        if position.upper() == "QB"
    }
    team_by_id = {
        player_id: team
        for player_id, (_, team, _) in name_map.items()
    }
    weird_rows = _read(simulation / "football_weirdness_worlds.csv")
    worlds = max(
        len({int(_f(row.get("world"))) for row in weird_rows}),
        1,
    )
    denominator = 32 * worlds

    designed_rows = _read(simulation / "same_world_designed_runs.csv")
    designed_non_sneak = [
        row
        for row in designed_rows
        if row.get("rusher_id") in qb_ids
        and row.get("category") != "qb_sneak"
    ]
    scramble_rows = [
        row
        for row in _read(simulation / "same_world_scrambles.csv")
        if row.get("player_id") in qb_ids
    ]

    designed_counts: dict[tuple[str, int, str], int] = defaultdict(int)
    for row in designed_non_sneak:
        key = (
            row.get("game", ""),
            int(_f(row.get("world"))),
            team_by_id.get(row.get("rusher_id", ""), ""),
        )
        designed_counts[key] += 1
    scramble_counts: dict[tuple[str, int, str], int] = defaultdict(int)
    for row in scramble_rows:
        key = (
            row.get("game", ""),
            int(_f(row.get("world"))),
            team_by_id.get(row.get("player_id", ""), ""),
        )
        scramble_counts[key] += 1

    designed_mean = len(designed_non_sneak) / denominator
    scramble_mean = len(scramble_rows) / denominator
    family = historical["family_summary"]
    historical_designed = _f(
        family["designed_non_sneak"]["mean_attempts_per_start_game_population"]
    )
    historical_scramble = _f(
        family["scramble"]["mean_attempts_per_start_game_population"]
    )
    competitive_hist = _f(
        historical["competitive_qb_rushing"]["mean_attempts_per_start"]
    )
    competitive_mean = designed_mean + scramble_mean
    return {
        "worlds_per_game": worlds,
        "designed_non_sneak_attempts_per_team_world": designed_mean,
        "historical_designed_non_sneak": historical_designed,
        "designed_absolute_error": abs(designed_mean - historical_designed),
        "scrambles_per_team_world": scramble_mean,
        "historical_scrambles": historical_scramble,
        "scramble_absolute_error": abs(scramble_mean - historical_scramble),
        "competitive_attempts_per_team_world": competitive_mean,
        "historical_competitive_attempts": competitive_hist,
        "competitive_absolute_error": abs(
            competitive_mean - competitive_hist
        ),
        "designed_team_world_p90": _quantile(
            [
                designed_counts.get((game, world, team), 0)
                for game in {
                    row.get("game", "") for row in weird_rows
                }
                for world in range(worlds)
                for team in game.split("@")
            ],
            0.90,
        ),
        "scramble_team_world_p90": _quantile(
            [
                scramble_counts.get((game, world, team), 0)
                for game in {
                    row.get("game", "") for row in weird_rows
                }
                for world in range(worlds)
                for team in game.split("@")
            ],
            0.90,
        ),
    }


def _substitution_arm(
    simulation: Path,
    box: Path,
    players: list[dict[str, str]],
) -> dict[str, object]:
    name_map = _player_name_map(box)
    team_by_id = {
        player_id: team
        for player_id, (_, team, _) in name_map.items()
    }
    position_by_id = {
        player_id: position
        for player_id, (_, _, position) in name_map.items()
    }
    qbs_by_team_world: dict[
        tuple[str, int, str],
        set[str],
    ] = defaultdict(set)
    for row in _read(simulation / "player_world_fanduel.csv"):
        player_id = row.get("player_id", "")
        if position_by_id.get(player_id, "").upper() != "QB":
            continue
        if _f(row.get("pass_attempts")) <= 0.0:
            continue
        key = (
            row.get("game", ""),
            int(_f(row.get("world"))),
            team_by_id.get(player_id, ""),
        )
        qbs_by_team_world[key].add(player_id)

    weird_rows = _read(simulation / "football_weirdness_worlds.csv")
    worlds = max(
        len({int(_f(row.get("world"))) for row in weird_rows}),
        1,
    )
    game_teams = [
        (game, team)
        for game in {row.get("game", "") for row in weird_rows}
        for team in game.split("@")
    ]
    indicators = [
        1.0
        if len(qbs_by_team_world.get((game, world, team), set())) > 1
        else 0.0
        for game, team in game_teams
        for world in range(worlds)
    ]

    actual_qbs: dict[str, set[str]] = defaultdict(set)
    for row in players:
        if str(row.get("position", "")).upper() != "QB":
            continue
        if _f(row.get("attempts")) > 0.0:
            actual_qbs[row["team"]].add(row.get("player_id", ""))
    actual_multi = _mean(
        1.0 if len(actual_qbs.get(team, set())) > 1 else 0.0
        for _, team in game_teams
    )

    mutation_rows = _read(simulation / "live_mutations_v72.csv")
    return {
        "multi_qb_team_world_rate": _mean(indicators),
        "actual_week1_multi_qb_team_rate": actual_multi,
        "multi_qb_rate_absolute_error": abs(_mean(indicators) - actual_multi),
        "live_mutation_telemetry_available": bool(mutation_rows),
        "mutations_per_team_world": (
            len(mutation_rows) / (32 * worlds) if mutation_rows else 0.0
        ),
        "out_mutations_per_team_world": (
            sum(1 for row in mutation_rows if row.get("mutation") == "out")
            / (32 * worlds)
            if mutation_rows
            else 0.0
        ),
        "qb_out_mutations_per_team_world": (
            sum(
                1
                for row in mutation_rows
                if row.get("mutation") == "out"
                and str(row.get("position", "")).upper() == "QB"
            )
            / (32 * worlds)
            if mutation_rows
            else 0.0
        ),
    }


def _dominant_actual_special(
    players: list[dict[str, str]],
    metric: str,
) -> dict[str, str]:
    values: dict[tuple[str, str], float] = defaultdict(float)
    for row in players:
        player_id = row.get("player_id", "")
        if not player_id:
            continue
        values[(row["team"], player_id)] += _f(row.get(metric))
    result: dict[str, str] = {}
    for (team, player_id), value in values.items():
        if value <= 0.0:
            continue
        current = result.get(team)
        if current is None:
            result[team] = player_id
            continue
        if values[(team, player_id)] > values[(team, current)]:
            result[team] = player_id
    return result


def _special_teams_arm(
    simulation: Path,
    players: list[dict[str, str]],
    personnel_path: Path,
) -> dict[str, object]:
    telemetry = _read(
        simulation / "same_world_special_teams_players_v72.csv"
    )
    if not telemetry:
        return {
            "player_identity_available": False,
            "kicker_hit_rate": None,
            "punter_hit_rate": None,
            "punt_returner_hit_rate": None,
            "kick_returner_hit_rate": None,
        }

    team_by_id: dict[str, str] = {}
    for row in _read(personnel_path):
        player_id = row.get("gsis_id", "")
        if player_id:
            team_by_id[player_id] = _team(
                row.get("team_id") or row.get("team")
            )

    counts: dict[str, dict[tuple[str, str], int]] = {
        role: defaultdict(int)
        for role in ("kicker", "punter", "punt_returner", "kick_returner")
    }
    for row in telemetry:
        event = row.get("event_type", "")
        if row.get("kicker_id"):
            pid = row["kicker_id"]
            team = team_by_id.get(pid, "")
            if team:
                counts["kicker"][(team, pid)] += 1
        if event == "punt" and row.get("punter_id"):
            pid = row["punter_id"]
            team = team_by_id.get(pid, "")
            if team:
                counts["punter"][(team, pid)] += 1
        if event == "punt" and row.get("returner_id"):
            pid = row["returner_id"]
            team = team_by_id.get(pid, "")
            if team:
                counts["punt_returner"][(team, pid)] += 1
        if event == "kickoff" and row.get("returner_id"):
            pid = row["returner_id"]
            team = team_by_id.get(pid, "")
            if team:
                counts["kick_returner"][(team, pid)] += 1

    actual = {
        "kicker": _dominant_actual_special(players, "fg_att"),
        "punter": _dominant_actual_special(players, "pt_att"),
        "punt_returner": _dominant_actual_special(players, "punt_returns"),
        "kick_returner": _dominant_actual_special(players, "kickoff_returns"),
    }
    result: dict[str, object] = {"player_identity_available": True}
    for role, actual_map in actual.items():
        sim_map: dict[str, str] = {}
        for (team, player_id), count in counts[role].items():
            current = sim_map.get(team)
            if (
                current is None
                or count > counts[role][(team, current)]
            ):
                sim_map[team] = player_id
        teams = sorted(actual_map)
        hits = [
            1.0 if sim_map.get(team) == actual_map[team] else 0.0
            for team in teams
        ]
        result[f"{role}_hit_rate"] = _mean(hits)
        result[f"{role}_teams_with_actual_event"] = len(teams)
    return result


def _paired_ecology(
    control: dict[int, dict[str, float]],
    challenger: dict[int, dict[str, float]],
) -> dict[str, object]:
    worlds = sorted(set(control) & set(challenger))
    if not worlds:
        return {}
    fields = sorted(
        set().union(
            *(metrics.keys() for metrics in control.values()),
            *(metrics.keys() for metrics in challenger.values()),
        )
    )
    result: dict[str, object] = {}
    for field in fields:
        deltas = [
            challenger[world].get(field, 0.0)
            - control[world].get(field, 0.0)
            for world in worlds
        ]
        result[field] = {
            "paired_mean_change_v72_minus_v71": _mean(deltas),
            "paired_sd": _sd(deltas),
            "paired_p10": _quantile(deltas, 0.10),
            "paired_p50": _quantile(deltas, 0.50),
            "paired_p90": _quantile(deltas, 0.90),
        }
    return result


def _improvement(
    control: float,
    challenger: float,
    *,
    higher_is_better: bool = False,
) -> float:
    if higher_is_better:
        return challenger - control
    return control - challenger


def _comparison_summary(
    control: dict[str, object],
    challenger: dict[str, object],
) -> dict[str, object]:
    return {
        "scoreboard": {
            "team_points_mae_improvement": _improvement(
                _f(control["scoreboard"]["team_points_mae"]),
                _f(challenger["scoreboard"]["team_points_mae"]),
            ),
            "game_total_crps_improvement": _improvement(
                _f(control["scoreboard"]["total_crps_mean"]),
                _f(challenger["scoreboard"]["total_crps_mean"]),
            ),
            "margin_crps_improvement": _improvement(
                _f(control["scoreboard"]["margin_crps_mean"]),
                _f(challenger["scoreboard"]["margin_crps_mean"]),
            ),
            "expected_total_correlation_change": _improvement(
                _f(control["scoreboard"]["expected_total_vs_actual_correlation"]),
                _f(challenger["scoreboard"]["expected_total_vs_actual_correlation"]),
                higher_is_better=True,
            ),
        },
        "participation": {
            "ol_snap_mae_improvement": _improvement(
                _f(control["participation"]["offensive_line"]["snap_share_mae"]),
                _f(challenger["participation"]["offensive_line"]["snap_share_mae"]),
            ),
            "ol_starter_recognition_change": _improvement(
                _f(control["participation"]["offensive_line"]["starter_recognition"]),
                _f(challenger["participation"]["offensive_line"]["starter_recognition"]),
                higher_is_better=True,
            ),
            "defensive_snap_mae_improvement": _improvement(
                _f(control["participation"]["defense"]["active_10pct_mae"]),
                _f(challenger["participation"]["defense"]["active_10pct_mae"]),
            ),
        },
        "opportunity": {
            "carry_tvd_improvement": _improvement(
                _f(control["carry_share"]["team_share_tvd_mean"]),
                _f(challenger["carry_share"]["team_share_tvd_mean"]),
            ),
            "target_tvd_improvement": _improvement(
                _f(control["target_share"]["team_share_tvd_mean"]),
                _f(challenger["target_share"]["team_share_tvd_mean"]),
            ),
            "carry_top_identity_change": _improvement(
                _f(control["carry_share"]["top_identity_hit_rate"]),
                _f(challenger["carry_share"]["top_identity_hit_rate"]),
                higher_is_better=True,
            ),
            "target_top_identity_change": _improvement(
                _f(control["target_share"]["top_identity_hit_rate"]),
                _f(challenger["target_share"]["top_identity_hit_rate"]),
                higher_is_better=True,
            ),
        },
        "qb_rushing": {
            "designed_family_error_improvement": _improvement(
                _f(control["qb_family"]["designed_absolute_error"]),
                _f(challenger["qb_family"]["designed_absolute_error"]),
            ),
            "scramble_family_error_improvement": _improvement(
                _f(control["qb_family"]["scramble_absolute_error"]),
                _f(challenger["qb_family"]["scramble_absolute_error"]),
            ),
            "competitive_family_error_improvement": _improvement(
                _f(control["qb_family"]["competitive_absolute_error"]),
                _f(challenger["qb_family"]["competitive_absolute_error"]),
            ),
        },
        "team_identity": {
            "total_yards_mae_improvement": _improvement(
                _f(control["team_offense"]["total_yards"]["mae"]),
                _f(challenger["team_offense"]["total_yards"]["mae"]),
            ),
            "total_yards_correlation_change": _improvement(
                _f(control["team_offense"]["total_yards"]["pearson"]),
                _f(challenger["team_offense"]["total_yards"]["pearson"]),
                higher_is_better=True,
            ),
        },
        "receiver_identity": {
            "targets_mae_improvement": _improvement(
                _f(control["offense_player"]["targets"]["mae"]),
                _f(challenger["offense_player"]["targets"]["mae"]),
            ),
            "target_correlation_change": _improvement(
                _f(control["offense_player"]["targets"]["pearson"]),
                _f(challenger["offense_player"]["targets"]["pearson"]),
                higher_is_better=True,
            ),
            "receiving_yards_correlation_change": _improvement(
                _f(control["offense_player"]["receiving_yards"]["pearson"]),
                _f(challenger["offense_player"]["receiving_yards"]["pearson"]),
                higher_is_better=True,
            ),
        },
        "defensive_identity": {
            "pressure_mae_improvement": _improvement(
                _f(control["defense"]["pressures"]["mae"]),
                _f(challenger["defense"]["pressures"]["mae"]),
            ),
            "pressure_correlation_change": _improvement(
                _f(control["defense"]["pressures"]["pearson"]),
                _f(challenger["defense"]["pressures"]["pearson"]),
                higher_is_better=True,
            ),
            "tackle_mae_improvement": _improvement(
                _f(control["defense"]["tackles"]["mae"]),
                _f(challenger["defense"]["tackles"]["mae"]),
            ),
        },
        "live_state": {
            "multi_qb_rate_error_improvement": _improvement(
                _f(control["substitution"]["multi_qb_rate_absolute_error"]),
                _f(challenger["substitution"]["multi_qb_rate_absolute_error"]),
            ),
            "v72_mutations_per_team_world": _f(
                challenger["substitution"]["mutations_per_team_world"]
            ),
        },
        "special_teams": {
            "v71_player_identity_available": bool(
                control["special_teams"]["player_identity_available"]
            ),
            "v72_player_identity_available": bool(
                challenger["special_teams"]["player_identity_available"]
            ),
            "v72_kicker_hit_rate": challenger["special_teams"].get(
                "kicker_hit_rate"
            ),
            "v72_punter_hit_rate": challenger["special_teams"].get(
                "punter_hit_rate"
            ),
            "v72_punt_returner_hit_rate": challenger["special_teams"].get(
                "punt_returner_hit_rate"
            ),
            "v72_kick_returner_hit_rate": challenger["special_teams"].get(
                "kick_returner_hit_rate"
            ),
        },
    }


def _arm_summary(
    *,
    simulation: Path,
    box: Path,
    actual_games: dict[str, dict[str, object]],
    players: list[dict[str, str]],
    snap_truth: Path,
    pfr_truth: Path,
    personnel: Path,
    historical: dict[str, object],
) -> tuple[dict[str, object], dict[int, dict[str, float]], list[dict[str, object]]]:
    scoreboard, score_detail = _scoreboard_arm(simulation, actual_games)
    ecology, ecology_worlds = _ecology_arm(
        simulation,
        _actual_ecology(players),
    )
    return {
        "scoreboard": scoreboard,
        "ecology": ecology,
        "team_offense": _team_offense_arm(box, players),
        "offense_player": _offense_player_arm(box, players),
        "carry_share": _role_share_arm(
            box,
            players,
            actual_col="carries",
            sim_col="rush_attempts_mean",
        ),
        "target_share": _role_share_arm(
            box,
            players,
            actual_col="targets",
            sim_col="targets_mean",
        ),
        "participation": _participation_arm(
            simulation,
            box,
            snap_truth,
        ),
        "defense": _defense_arm(box, players, pfr_truth),
        "qb_family": _qb_family_arm(simulation, box, historical),
        "substitution": _substitution_arm(simulation, box, players),
        "special_teams": _special_teams_arm(
            simulation,
            players,
            personnel,
        ),
    }, ecology_worlds, score_detail


def _markdown(
    control: dict[str, object],
    challenger: dict[str, object],
    comparison: dict[str, object],
    worlds: int,
) -> str:
    def val(path: tuple[str, ...], source: dict[str, object]) -> object:
        current: object = source
        for key in path:
            current = current[key]  # type: ignore[index]
        return current

    lines = [
        "# MONSTER V7.1 vs V7.2 paired Week 1 validation",
        "",
        f"Common-random-number worlds per game per arm: **{worlds}**.",
        "",
        "## Headline",
        "",
        (
            "This report separates mechanism realism from scoreboard luck. "
            "Lower error is better unless the metric is explicitly a correlation "
            "or recognition/hit rate."
        ),
        "",
        "## Core paired metrics",
        "",
        "| Metric | V7.1 | V7.2 | V7.2 change |",
        "|---|---:|---:|---:|",
    ]
    table = [
        (
            "OL starter recognition",
            ("participation", "offensive_line", "starter_recognition"),
            comparison["participation"]["ol_starter_recognition_change"],
        ),
        (
            "OL snap-share MAE",
            ("participation", "offensive_line", "snap_share_mae"),
            comparison["participation"]["ol_snap_mae_improvement"],
        ),
        (
            "Designed QB run abs error",
            ("qb_family", "designed_absolute_error"),
            comparison["qb_rushing"]["designed_family_error_improvement"],
        ),
        (
            "Target share TVD",
            ("target_share", "team_share_tvd_mean"),
            comparison["opportunity"]["target_tvd_improvement"],
        ),
        (
            "Carry share TVD",
            ("carry_share", "team_share_tvd_mean"),
            comparison["opportunity"]["carry_tvd_improvement"],
        ),
        (
            "Team total-yards MAE",
            ("team_offense", "total_yards", "mae"),
            comparison["team_identity"]["total_yards_mae_improvement"],
        ),
        (
            "Team total-yards correlation",
            ("team_offense", "total_yards", "pearson"),
            comparison["team_identity"]["total_yards_correlation_change"],
        ),
        (
            "Pressure MAE",
            ("defense", "pressures", "mae"),
            comparison["defensive_identity"]["pressure_mae_improvement"],
        ),
        (
            "Team points MAE",
            ("scoreboard", "team_points_mae"),
            comparison["scoreboard"]["team_points_mae_improvement"],
        ),
        (
            "Game-total CRPS",
            ("scoreboard", "total_crps_mean"),
            comparison["scoreboard"]["game_total_crps_improvement"],
        ),
        (
            "Margin CRPS",
            ("scoreboard", "margin_crps_mean"),
            comparison["scoreboard"]["margin_crps_improvement"],
        ),
    ]
    for label, path, change in table:
        c = _f(val(path, control))
        h = _f(val(path, challenger))
        lines.append(f"| {label} | {c:.4f} | {h:.4f} | {_f(change):+.4f} |")

    lines.extend(
        [
            "",
            "## Special teams identity",
            "",
            (
                f"V7.1 player identity telemetry available: "
                f"**{control['special_teams']['player_identity_available']}**."
            ),
            (
                f"V7.2 kicker hit rate: "
                f"**{_f(challenger['special_teams'].get('kicker_hit_rate')):.3f}**; "
                f"punter: **{_f(challenger['special_teams'].get('punter_hit_rate')):.3f}**; "
                f"PR: **{_f(challenger['special_teams'].get('punt_returner_hit_rate')):.3f}**; "
                f"KR: **{_f(challenger['special_teams'].get('kick_returner_hit_rate')):.3f}**."
            ),
            "",
            "## Interpretation guard",
            "",
            (
                "Week 1 reality is an external benchmark, not a tuning target. "
                "A V7.2 mechanism can be retained when its causal layer improves "
                "even if a downstream scoreboard metric is flat or worse. "
                "Scoreboard changes should be investigated through the event chain, "
                "not patched directly."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired distributional Week 1 audit for MONSTER V7.1 vs V7.2."
    )
    parser.add_argument("--control-sim", type=Path, required=True)
    parser.add_argument("--control-box", type=Path, required=True)
    parser.add_argument("--challenger-sim", type=Path, required=True)
    parser.add_argument("--challenger-box", type=Path, required=True)
    parser.add_argument("--schedules", type=Path, required=True)
    parser.add_argument("--player-truth", type=Path, required=True)
    parser.add_argument("--snap-truth", type=Path, required=True)
    parser.add_argument("--pfr-defense-truth", type=Path, required=True)
    parser.add_argument("--personnel", type=Path, required=True)
    parser.add_argument("--historical-qb", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    actual_games, _, players = _actual_context(
        args.schedules,
        args.player_truth,
    )
    historical = json.loads(
        args.historical_qb.read_text(encoding="utf-8")
    )

    control, control_worlds, control_scores = _arm_summary(
        simulation=args.control_sim,
        box=args.control_box,
        actual_games=actual_games,
        players=players,
        snap_truth=args.snap_truth,
        pfr_truth=args.pfr_defense_truth,
        personnel=args.personnel,
        historical=historical,
    )
    challenger, challenger_worlds, challenger_scores = _arm_summary(
        simulation=args.challenger_sim,
        box=args.challenger_box,
        actual_games=actual_games,
        players=players,
        snap_truth=args.snap_truth,
        pfr_truth=args.pfr_defense_truth,
        personnel=args.personnel,
        historical=historical,
    )
    comparison = _comparison_summary(control, challenger)
    paired_ecology = _paired_ecology(control_worlds, challenger_worlds)

    worlds = max(len(control_worlds), len(challenger_worlds))
    payload = {
        "experiment": "v7.1-vs-v7.2-week1-common-random-distributional",
        "worlds_per_game_each_arm": worlds,
        "games": len(actual_games),
        "common_random_numbers": True,
        "truth_consumed_only_after_both_simulations": True,
        "market_blind": True,
        "week1_truth_used_for_priors": False,
        "control_architecture": "v7.1-qb-rush-family-shadow",
        "challenger_architecture": "v7.2-reality-allocation-shadow",
        "control": control,
        "challenger": challenger,
        "v72_directional_changes": comparison,
        "paired_league_ecology": paired_ecology,
        "interpretation": {
            "primary": (
                "Mechanism fidelity is evaluated separately from downstream "
                "scoreboard fit. Week 1 is evidence, not a target."
            ),
            "promotion_rule": (
                "Do not promote or reject V7.2 from one score metric. "
                "Require persistent upstream realism gains without a repeated "
                "league-ecology regression."
            ),
        },
    }

    _write(args.out / "v71_scoreboard_games.csv", control_scores)
    _write(args.out / "v72_scoreboard_games.csv", challenger_scores)
    cleaned = _clean_json(payload)
    (args.out / "v71_v72_distributional_summary.json").write_text(
        json.dumps(cleaned, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.out / "V7_1_V7_2_WEEK1_REPORT.md").write_text(
        _markdown(control, challenger, comparison, worlds),
        encoding="utf-8",
    )
    print(json.dumps(cleaned, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
