from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import nflreadpy as nfl
import numpy as np
import polars as pl

from monster.feature_compile.depth import attach_depth_chart
from monster.feature_compile.team import compile_team_policy
from monster.ingest.league import build_league_personnel_snapshot
from monster.ingest.nflverse import PBP_COLUMNS, load_league_personnel_inputs
from monster.sim.game import simulate_game
from monster.snapshot.league import compile_team_state_map
from monster.snapshot.model import GameState, TeamState
from monster.teams import NFL_TEAMS, normalize_team_id

VARIANTS = ("policy", "offense", "continuity_offense")


def _regular_pbp(season: int) -> pl.DataFrame:
    pbp = nfl.load_pbp([season]).select([c for c in PBP_COLUMNS if c in nfl.load_pbp([season]).columns])
    if "season_type" in pbp.columns:
        pbp = pbp.filter(pl.col("season_type") == "REG")
    return pbp


def _regular_schedule(season: int) -> pl.DataFrame:
    frame = nfl.load_schedules([season]).filter(
        pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null()
    )
    if "game_type" in frame.columns:
        frame = frame.filter(pl.col("game_type") == "REG")
    return frame


def _week1(frame: pl.DataFrame) -> pl.DataFrame:
    if not frame.height or "week" not in frame.columns:
        return frame
    weeks = frame.get_column("week").drop_nulls()
    if not len(weeks):
        return frame
    return frame.filter(pl.col("week") == int(weeks.min()))


def _col(df: pl.DataFrame, name: str) -> pl.Expr:
    return pl.col(name).fill_null(0.0).cast(pl.Float64) if name in df.columns else pl.lit(0.0)


def _rank_signal(expr: pl.Expr, eligible: pl.Expr, group: pl.Expr) -> pl.Expr:
    ranked = pl.when(eligible).then(expr).otherwise(None).rank(method="average").over(group)
    count = eligible.cast(pl.Int64).sum().over(group).clip(lower_bound=1)
    return pl.when(eligible).then((2.0 * ranked / count - 1.0).clip(-1.0, 1.0)).otherwise(None)


def _attach_offensive_history(snapshot: pl.DataFrame, stats: pl.DataFrame) -> pl.DataFrame:
    if not stats.height or "player_id" not in stats.columns:
        return snapshot.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("offense_capability_signal"),
            pl.lit(0.0).alias("offense_history_weight"),
        )

    attempts = _col(stats, "attempts")
    pass_yards = _col(stats, "passing_yards")
    pass_tds = _col(stats, "passing_tds")
    ints = _col(stats, "interceptions")
    carries = _col(stats, "carries")
    rush_yards = _col(stats, "rushing_yards")
    rush_tds = _col(stats, "rushing_tds")
    targets = _col(stats, "targets")
    rec_yards = _col(stats, "receiving_yards")
    rec_tds = _col(stats, "receiving_tds")

    hist = (
        stats.with_columns(pl.col("player_id").cast(pl.Utf8))
        .group_by("player_id")
        .agg(
            attempts.sum().alias("hist_attempts"),
            pass_yards.sum().alias("hist_pass_yards"),
            pass_tds.sum().alias("hist_pass_tds"),
            ints.sum().alias("hist_interceptions"),
            carries.sum().alias("hist_carries"),
            rush_yards.sum().alias("hist_rush_yards"),
            rush_tds.sum().alias("hist_rush_tds"),
            targets.sum().alias("hist_targets"),
            rec_yards.sum().alias("hist_rec_yards"),
            rec_tds.sum().alias("hist_rec_tds"),
        )
    )
    out = snapshot.join(hist, left_on="gsis_id", right_on="player_id", how="left")
    pos = pl.col("position_group").cast(pl.Utf8).str.to_uppercase()
    qb = pos == "QB"
    skill = pos.is_in(["RB", "WR", "TE"])

    att = pl.col("hist_attempts").fill_null(0.0)
    py = pl.col("hist_pass_yards").fill_null(0.0)
    ptd = pl.col("hist_pass_tds").fill_null(0.0)
    interceptions = pl.col("hist_interceptions").fill_null(0.0)
    car = pl.col("hist_carries").fill_null(0.0)
    ry = pl.col("hist_rush_yards").fill_null(0.0)
    rtd = pl.col("hist_rush_tds").fill_null(0.0)
    tar = pl.col("hist_targets").fill_null(0.0)
    recy = pl.col("hist_rec_yards").fill_null(0.0)
    rectd = pl.col("hist_rec_tds").fill_null(0.0)

    out = out.with_columns(
        (
            (py / att.clip(lower_bound=1.0))
            + 18.0 * (ptd / att.clip(lower_bound=1.0))
            - 22.0 * (interceptions / att.clip(lower_bound=1.0))
            + 0.12 * (ry / car.clip(lower_bound=1.0))
            + 5.0 * (rtd / car.clip(lower_bound=1.0))
        ).alias("_qb_value"),
        (
            ((ry + recy) / (car + tar).clip(lower_bound=1.0))
            + 8.0 * ((rtd + rectd) / (car + tar).clip(lower_bound=1.0))
        ).alias("_skill_value"),
        (att / 300.0).clip(0.0, 1.0).alias("_qb_reliability"),
        ((car + tar) / 100.0).clip(0.0, 1.0).alias("_skill_reliability"),
    )
    out = out.with_columns(
        _rank_signal(pl.col("_qb_value"), qb & (att >= 50), pos).alias("_qb_rank"),
        _rank_signal(pl.col("_skill_value"), skill & ((car + tar) >= 20), pos).alias("_skill_rank"),
    )
    return out.with_columns(
        pl.when(qb)
        .then(pl.col("_qb_rank") * pl.col("_qb_reliability"))
        .when(skill)
        .then(pl.col("_skill_rank") * pl.col("_skill_reliability"))
        .otherwise(None)
        .alias("offense_capability_signal"),
        pl.when(qb)
        .then(pl.col("_qb_reliability"))
        .when(skill)
        .then(pl.col("_skill_reliability"))
        .otherwise(0.0)
        .alias("offense_history_weight"),
    )


def _team_offense_map(snapshot: pl.DataFrame) -> dict[str, dict[str, float]]:
    pos = pl.col("position_group").cast(pl.Utf8).str.to_uppercase()
    depth_weight = (
        pl.when(pl.col("depth_rank").is_not_null())
        .then((1.0 / pl.col("depth_rank").cast(pl.Float64)).clip(0.20, 1.0))
        .otherwise(0.35)
    )
    snap = pl.col("offense_snap_share").fill_null(0.0).clip(0.0, 1.0)
    participation = pl.max_horizontal(snap, 0.35 * depth_weight)
    signal = pl.col("offense_capability_signal").fill_null(0.0)

    rows = (
        snapshot.with_columns((participation * depth_weight).alias("_w"))
        .group_by("team_id")
        .agg(
            pl.when(pos == "QB").then(signal * pl.col("_w")).otherwise(None).sum().alias("qb_num"),
            pl.when(pos == "QB").then(pl.col("_w")).otherwise(None).sum().alias("qb_den"),
            pl.when(pos.is_in(["RB", "WR", "TE"])).then(signal * pl.col("_w")).otherwise(None).sum().alias("skill_num"),
            pl.when(pos.is_in(["RB", "WR", "TE"])).then(pl.col("_w")).otherwise(None).sum().alias("skill_den"),
            pl.when(
                (pl.col("snap_games_observed") > 0)
                & pl.col("prior_team_id").is_not_null()
                & (pl.col("prior_team_id") == pl.col("team_id"))
                & pos.is_in(["QB", "RB", "WR", "TE"])
            ).then(pl.col("_w")).otherwise(0.0).sum().alias("retained_w"),
            pl.when(pos.is_in(["QB", "RB", "WR", "TE"])).then(pl.col("_w")).otherwise(0.0).sum().alias("all_w"),
        )
        .to_dicts()
    )
    result: dict[str, dict[str, float]] = {}
    for r in rows:
        qb = float(r["qb_num"] or 0.0) / max(float(r["qb_den"] or 0.0), 1e-9)
        skill = float(r["skill_num"] or 0.0) / max(float(r["skill_den"] or 0.0), 1e-9)
        continuity = float(r["retained_w"] or 0.0) / max(float(r["all_w"] or 0.0), 1e-9)
        result[str(r["team_id"])] = {
            "qb": float(np.clip(qb, -1.0, 1.0)),
            "skill": float(np.clip(skill, -1.0, 1.0)),
            "continuity": float(np.clip(continuity, 0.0, 1.0)),
        }
    return result


def _apply_offense(state: TeamState, signals: dict[str, float], continuity_mode: bool) -> TeamState:
    qb = signals["qb"]
    skill = signals["skill"]
    authority = 0.55 + 0.45 * signals["continuity"] if continuity_mode else 1.0
    score_signal = authority * (0.68 * qb + 0.32 * skill)
    return replace(
        state,
        td_drive_rate=float(np.clip(state.td_drive_rate * (1.0 + 0.075 * score_signal), 0.03, 0.55)),
        turnover_drive_rate=float(np.clip(state.turnover_drive_rate * (1.0 - 0.060 * qb * authority), 0.02, 0.35)),
        offensive_epa_per_play=float(np.clip(state.offensive_epa_per_play + 0.035 * score_signal, -0.45, 0.45)),
        offensive_success_rate=float(np.clip(state.offensive_success_rate + 0.012 * score_signal, 0.25, 0.70)),
        offensive_explosive_rate=float(np.clip(state.offensive_explosive_rate + 0.010 * score_signal, 0.02, 0.35)),
    )


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def run_fold(train: int, test: int, worlds: int, seed: int) -> tuple[list[dict], pl.DataFrame]:
    policy = pl.DataFrame({"team_id": list(NFL_TEAMS)}).join(
        compile_team_policy(_regular_pbp(train)), on="team_id", how="left"
    ).sort("team_id")
    inputs = load_league_personnel_inputs([train], test, Path(".cache/monster"))
    roster = _week1(inputs["current_rosters"])
    snapshot = build_league_personnel_snapshot(
        roster, inputs["players"], inputs["historical_snap_counts"], None, recent_games=6
    )
    snapshot = attach_depth_chart(snapshot, _week1(inputs["depth_charts"]))
    snapshot = _attach_offensive_history(snapshot, inputs["player_stats_history"])
    team_offense = _team_offense_map(snapshot)
    schedule = _regular_schedule(test)

    records: list[dict] = []
    for idx, row in enumerate(schedule.to_dicts()):
        away = normalize_team_id(str(row["away_team"]))
        home = normalize_team_id(str(row["home_team"]))
        base = compile_team_state_map(policy, {away: home, home: away}, prior_uncertainty=0.16)
        for v_idx, variant in enumerate(VARIANTS):
            away_state = base[away]
            home_state = base[home]
            if variant != "policy":
                away_state = _apply_offense(away_state, team_offense[away], variant == "continuity_offense")
                home_state = _apply_offense(home_state, team_offense[home], variant == "continuity_offense")
            result = simulate_game(
                GameState(f"{away}@{home}-{variant}", away_state, home_state),
                worlds=worlds,
                seed=seed + idx * 101 + v_idx * 1_000_003,
            )
            records.append({
                "train_season": train,
                "test_season": test,
                "variant": variant,
                "game_id": str(row.get("game_id") or f"{away}@{home}"),
                "actual_total": float(row["away_score"] + row["home_score"]),
                "actual_margin": float(row["away_score"] - row["home_score"]),
                "model_total": float((result.away_points + result.home_points).mean()),
                "model_margin": float((result.away_points - result.home_points).mean()),
                "away_qb_signal": team_offense[away]["qb"],
                "home_qb_signal": team_offense[home]["qb"],
                "away_skill_signal": team_offense[away]["skill"],
                "home_skill_signal": team_offense[home]["skill"],
            })

    frame = pl.DataFrame(records)
    metrics: list[dict] = []
    for variant in VARIANTS:
        sub = frame.filter(pl.col("variant") == variant)
        at = sub.get_column("actual_total").to_numpy()
        mt = sub.get_column("model_total").to_numpy()
        am = sub.get_column("actual_margin").to_numpy()
        mm = sub.get_column("model_margin").to_numpy()
        metrics.append({
            "train_season": train,
            "test_season": test,
            "variant": variant,
            "games": sub.height,
            "total_bias": float((mt-at).mean()),
            "total_mae": float(np.abs(mt-at).mean()),
            "total_corr": _corr(mt, at),
            "margin_mae": float(np.abs(mm-am).mean()),
            "margin_corr": _corr(mm, am),
        })
    return metrics, frame


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--worlds", type=int, default=2500)
    p.add_argument("--seed", type=int, default=2026090723)
    p.add_argument("--out", type=Path, default=Path("artifacts/oos-offensive-personnel"))
    args = p.parse_args()

    all_metrics: list[dict] = []
    frames: list[pl.DataFrame] = []
    for fold_idx, (train, test) in enumerate(((2023, 2024), (2024, 2025))):
        m, f = run_fold(train, test, args.worlds, args.seed + fold_idx * 10_000_019)
        all_metrics.extend(m)
        frames.append(f)

    mf = pl.DataFrame(all_metrics)
    summary = mf.group_by("variant").agg(
        pl.col("total_mae").mean().alias("mean_total_mae"),
        pl.col("total_corr").mean().alias("mean_total_corr"),
        pl.col("margin_mae").mean().alias("mean_margin_mae"),
        pl.col("margin_corr").mean().alias("mean_margin_corr"),
        pl.col("total_bias").mean().alias("mean_total_bias"),
    ).sort("mean_total_mae")
    policy = summary.filter(pl.col("variant") == "policy").row(0, named=True)
    candidates = summary.filter(pl.col("variant") != "policy").with_columns(
        (pl.lit(policy["mean_total_mae"]) - pl.col("mean_total_mae")).alias("total_mae_gain_vs_policy"),
        (pl.lit(policy["mean_margin_mae"]) - pl.col("mean_margin_mae")).alias("margin_mae_gain_vs_policy"),
    )
    best = candidates.sort(
        ["total_mae_gain_vs_policy", "margin_mae_gain_vs_policy"], descending=True
    ).row(0, named=True)
    bridge_pass = bool(
        best["total_mae_gain_vs_policy"] > 0
        and best["margin_mae_gain_vs_policy"] > 0
        and best["mean_total_corr"] >= policy["mean_total_corr"]
        and best["mean_margin_corr"] >= policy["mean_margin_corr"]
    )
    manifest = {
        "artifact": "Monster Pre-Week-1 Offensive Personnel OOS Ablation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "worlds_per_game_variant": args.worlds,
        "fold_metrics": all_metrics,
        "summary": summary.to_dicts(),
        "best_candidate": best,
        "bridge_pass": bridge_pass,
        "allowed_test_season_information": ["Week 1 roster", "Week 1 depth chart", "static identity/physical facts"],
        "historical_information": ["prior-season player production", "prior-season PBP/team policy"],
        "forbidden": ["test-season PBP", "test-season snap outcomes", "sportsbook market", "DFS salary/ownership/projections"],
        "principle": "Current offensive personnel must earn causal authority out of sample before entering the live team scoring engine.",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    pl.concat(frames).write_csv(args.out / "games.csv")
    mf.write_csv(args.out / "fold_metrics.csv")
    summary.write_csv(args.out / "summary.csv")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
