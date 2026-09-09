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
MODERN_DEPTH = {"dt", "team", "pos_grp", "pos_abb", "pos_slot", "pos_rank"}


def _regular_pbp(season: int) -> pl.DataFrame:
    raw = nfl.load_pbp([season])
    frame = raw.select([c for c in PBP_COLUMNS if c in raw.columns])
    if "season_type" in frame.columns:
        frame = frame.filter(pl.col("season_type") == "REG")
    return frame


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
    return frame if not len(weeks) else frame.filter(pl.col("week") == int(weeks.min()))


def _optional_depth(snapshot: pl.DataFrame, depth: pl.DataFrame) -> tuple[pl.DataFrame, bool]:
    depth = _week1(depth)
    if depth.height and MODERN_DEPTH.issubset(depth.columns):
        return attach_depth_chart(snapshot, depth), True
    if "depth_rank" not in snapshot.columns:
        snapshot = snapshot.with_columns(pl.lit(None, dtype=pl.Int64).alias("depth_rank"))
    return snapshot, False


def _num(df: pl.DataFrame, name: str) -> pl.Expr:
    return pl.col(name).fill_null(0.0).cast(pl.Float64) if name in df.columns else pl.lit(0.0)


def _rank_signal(expr: pl.Expr, eligible: pl.Expr, group: pl.Expr) -> pl.Expr:
    ranked = pl.when(eligible).then(expr).otherwise(None).rank(method="average").over(group)
    count = eligible.cast(pl.Int64).sum().over(group).clip(lower_bound=1)
    return pl.when(eligible).then((2.0 * ranked / count - 1.0).clip(-1.0, 1.0)).otherwise(None)


def _attach_offensive_history(snapshot: pl.DataFrame, stats: pl.DataFrame) -> pl.DataFrame:
    names = [
        "attempts", "passing_yards", "passing_tds", "interceptions",
        "carries", "rushing_yards", "rushing_tds", "targets",
        "receiving_yards", "receiving_tds",
    ]
    if not stats.height or "player_id" not in stats.columns:
        return snapshot.with_columns(pl.lit(None).alias("offense_capability_signal"))

    hist = stats.with_columns(pl.col("player_id").cast(pl.Utf8)).group_by("player_id").agg(
        *[_num(stats, name).sum().alias(f"hist_{name}") for name in names]
    )
    out = snapshot.join(hist, left_on="gsis_id", right_on="player_id", how="left")
    pos = pl.col("position_group").cast(pl.Utf8).str.to_uppercase()
    qb = pos == "QB"
    skill = pos.is_in(["RB", "WR", "TE"])
    c = lambda name: pl.col(f"hist_{name}").fill_null(0.0)
    att, py, ptd, ints = c("attempts"), c("passing_yards"), c("passing_tds"), c("interceptions")
    car, ry, rtd = c("carries"), c("rushing_yards"), c("rushing_tds")
    tar, recy, rectd = c("targets"), c("receiving_yards"), c("receiving_tds")

    out = out.with_columns(
        (
            py / att.clip(lower_bound=1.0)
            + 18.0 * ptd / att.clip(lower_bound=1.0)
            - 22.0 * ints / att.clip(lower_bound=1.0)
            + 0.12 * ry / car.clip(lower_bound=1.0)
            + 5.0 * rtd / car.clip(lower_bound=1.0)
        ).alias("_qb_value"),
        (
            (ry + recy) / (car + tar).clip(lower_bound=1.0)
            + 8.0 * (rtd + rectd) / (car + tar).clip(lower_bound=1.0)
        ).alias("_skill_value"),
        (att / 300.0).clip(0.0, 1.0).alias("_qb_rel"),
        ((car + tar) / 100.0).clip(0.0, 1.0).alias("_skill_rel"),
    ).with_columns(
        _rank_signal(pl.col("_qb_value"), qb & (att >= 50), pos).alias("_qb_rank"),
        _rank_signal(pl.col("_skill_value"), skill & ((car + tar) >= 20), pos).alias("_skill_rank"),
    )
    return out.with_columns(
        pl.when(qb).then(pl.col("_qb_rank") * pl.col("_qb_rel"))
        .when(skill).then(pl.col("_skill_rank") * pl.col("_skill_rel"))
        .otherwise(None).alias("offense_capability_signal")
    )


def _team_offense_map(snapshot: pl.DataFrame) -> dict[str, dict[str, float]]:
    pos = pl.col("position_group").cast(pl.Utf8).str.to_uppercase()
    depth_w = pl.when(pl.col("depth_rank").is_not_null()).then(
        (1.0 / pl.col("depth_rank").cast(pl.Float64)).clip(0.20, 1.0)
    ).otherwise(0.35)
    snap = pl.col("offense_snap_share").fill_null(0.0).clip(0.0, 1.0)
    weight = pl.max_horizontal(snap, 0.35 * depth_w) * depth_w
    signal = pl.col("offense_capability_signal").fill_null(0.0)

    rows = snapshot.with_columns(weight.alias("_w")).group_by("team_id").agg(
        pl.when(pos == "QB").then(signal * pl.col("_w")).otherwise(None).sum().alias("qb_num"),
        pl.when(pos == "QB").then(pl.col("_w")).otherwise(None).sum().alias("qb_den"),
        pl.when(pos.is_in(["RB", "WR", "TE"])).then(signal * pl.col("_w")).otherwise(None).sum().alias("skill_num"),
        pl.when(pos.is_in(["RB", "WR", "TE"])).then(pl.col("_w")).otherwise(None).sum().alias("skill_den"),
        pl.when(
            (pl.col("snap_games_observed") > 0)
            & pl.col("prior_team_id").is_not_null()
            & (pl.col("prior_team_id") == pl.col("team_id"))
            & pos.is_in(["QB", "RB", "WR", "TE"])
        ).then(pl.col("_w")).otherwise(0.0).sum().alias("retained"),
        pl.when(pos.is_in(["QB", "RB", "WR", "TE"])).then(pl.col("_w")).otherwise(0.0).sum().alias("all_w"),
    ).to_dicts()

    out: dict[str, dict[str, float]] = {}
    for r in rows:
        qd = max(float(r["qb_den"] or 0.0), 1e-9)
        sd = max(float(r["skill_den"] or 0.0), 1e-9)
        ad = max(float(r["all_w"] or 0.0), 1e-9)
        out[str(r["team_id"])] = {
            "qb": float(np.clip(float(r["qb_num"] or 0.0) / qd, -1, 1)),
            "skill": float(np.clip(float(r["skill_num"] or 0.0) / sd, -1, 1)),
            "continuity": float(np.clip(float(r["retained"] or 0.0) / ad, 0, 1)),
        }
    return out


def _apply_offense(state: TeamState, s: dict[str, float], continuity_mode: bool) -> TeamState:
    authority = (0.55 + 0.45 * s["continuity"]) if continuity_mode else 1.0
    score = authority * (0.68 * s["qb"] + 0.32 * s["skill"])
    return replace(
        state,
        td_drive_rate=float(np.clip(state.td_drive_rate * (1.0 + 0.075 * score), 0.03, 0.55)),
        turnover_drive_rate=float(np.clip(state.turnover_drive_rate * (1.0 - 0.060 * s["qb"] * authority), 0.02, 0.35)),
        offensive_epa_per_play=float(np.clip(state.offensive_epa_per_play + 0.035 * score, -0.45, 0.45)),
        offensive_success_rate=float(np.clip(state.offensive_success_rate + 0.012 * score, 0.25, 0.70)),
        offensive_explosive_rate=float(np.clip(state.offensive_explosive_rate + 0.010 * score, 0.02, 0.35)),
    )


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    return float("nan") if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0 else float(np.corrcoef(a, b)[0, 1])


def run_fold(train: int, test: int, worlds: int, seed: int) -> tuple[list[dict], pl.DataFrame, bool]:
    policy = pl.DataFrame({"team_id": list(NFL_TEAMS)}).join(
        compile_team_policy(_regular_pbp(train)), on="team_id", how="left"
    ).sort("team_id")
    inputs = load_league_personnel_inputs([train], test, Path(".cache/monster"))
    snapshot = build_league_personnel_snapshot(
        _week1(inputs["current_rosters"]), inputs["players"], inputs["historical_snap_counts"], None, recent_games=6
    )
    snapshot, depth_used = _optional_depth(snapshot, inputs["depth_charts"])
    snapshot = _attach_offensive_history(snapshot, inputs["player_stats_history"])
    team_offense = _team_offense_map(snapshot)
    schedule = _regular_schedule(test)

    records: list[dict] = []
    for idx, row in enumerate(schedule.to_dicts()):
        away, home = normalize_team_id(str(row["away_team"])), normalize_team_id(str(row["home_team"]))
        base = compile_team_state_map(policy, {away: home, home: away}, prior_uncertainty=0.16)
        paired_seed = seed + idx * 101
        for variant in VARIANTS:
            a, h = base[away], base[home]
            if variant != "policy":
                a = _apply_offense(a, team_offense[away], variant == "continuity_offense")
                h = _apply_offense(h, team_offense[home], variant == "continuity_offense")
            result = simulate_game(
                GameState(f"{away}@{home}-{variant}", a, h), worlds=worlds,
                seed=paired_seed,
            )
            records.append({
                "train_season": train, "test_season": test, "variant": variant,
                "game_id": str(row.get("game_id") or f"{away}@{home}"),
                "actual_total": float(row["away_score"] + row["home_score"]),
                "actual_margin": float(row["away_score"] - row["home_score"]),
                "model_total": float((result.away_points + result.home_points).mean()),
                "model_margin": float((result.away_points - result.home_points).mean()),
            })

    frame = pl.DataFrame(records)
    metrics: list[dict] = []
    for variant in VARIANTS:
        sub = frame.filter(pl.col("variant") == variant)
        at, mt = sub["actual_total"].to_numpy(), sub["model_total"].to_numpy()
        am, mm = sub["actual_margin"].to_numpy(), sub["model_margin"].to_numpy()
        metrics.append({
            "train_season": train, "test_season": test, "variant": variant, "games": sub.height,
            "total_bias": float((mt-at).mean()), "total_mae": float(np.abs(mt-at).mean()),
            "total_corr": _corr(mt, at), "margin_mae": float(np.abs(mm-am).mean()),
            "margin_corr": _corr(mm, am), "modern_depth_used": depth_used,
        })
    return metrics, frame, depth_used


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--worlds", type=int, default=2500)
    p.add_argument("--seed", type=int, default=2026090723)
    p.add_argument("--out", type=Path, default=Path("artifacts/oos-offensive-personnel"))
    args = p.parse_args()
    all_metrics, frames, depth_flags = [], [], []
    for i, (train, test) in enumerate(((2023, 2024), (2024, 2025))):
        m, f, d = run_fold(train, test, args.worlds, args.seed + i * 10_000_019)
        all_metrics += m; frames.append(f); depth_flags.append(d)

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
    best = candidates.sort(["total_mae_gain_vs_policy", "margin_mae_gain_vs_policy"], descending=True).row(0, named=True)
    passed = bool(
        best["total_mae_gain_vs_policy"] > 0 and best["margin_mae_gain_vs_policy"] > 0
        and best["mean_total_corr"] >= policy["mean_total_corr"]
        and best["mean_margin_corr"] >= policy["mean_margin_corr"]
    )
    manifest = {
        "artifact": "Monster Pre-Week-1 Offensive Personnel OOS Ablation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "worlds_per_game_variant": args.worlds,
        "paired_common_random_numbers": True,
        "fold_metrics": all_metrics, "summary": summary.to_dicts(), "best_candidate": best,
        "bridge_pass": passed, "modern_depth_available_by_fold": depth_flags,
        "allowed_test_season_information": ["Week 1 roster", "Week 1 depth chart when schema-supported", "static identity facts"],
        "historical_information": ["prior-season player production", "prior-season PBP/team policy"],
        "forbidden": ["test-season PBP", "test-season snap outcomes", "sportsbook market", "DFS salary/ownership/projections"],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    pl.concat(frames).write_csv(args.out / "games.csv")
    mf.write_csv(args.out / "fold_metrics.csv"); summary.write_csv(args.out / "summary.csv")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
