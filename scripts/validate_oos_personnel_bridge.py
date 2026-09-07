from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import nflreadpy as nfl
import numpy as np
import polars as pl

from monster.feature_compile.capability import attach_capability_evidence
from monster.feature_compile.league_units import compile_league_unit_player_map
from monster.feature_compile.participation_inference import infer_game_day_participation
from monster.feature_compile.team import compile_team_policy
from monster.feature_compile.units import apply_team_unit_effects, compile_team_unit_effects
from monster.ingest.league import build_league_personnel_snapshot
from monster.ingest.nflverse import PBP_COLUMNS, load_league_personnel_inputs
from monster.sim.game import simulate_game
from monster.snapshot.league import compile_team_state_map
from monster.snapshot.model import GameState, TeamState
from monster.teams import NFL_TEAMS, normalize_team_id

VARIANTS = ("policy", "continuity", "units", "continuity_units")


def _regular_pbp(season: int) -> pl.DataFrame:
    pbp = nfl.load_pbp([season])
    pbp = pbp.select([c for c in PBP_COLUMNS if c in pbp.columns])
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


def _week1_rosters(frame: pl.DataFrame) -> pl.DataFrame:
    if "week" not in frame.columns:
        return frame
    weeks = frame.get_column("week").drop_nulls()
    if not len(weeks):
        return frame
    first_week = int(weeks.min())
    return frame.filter(pl.col("week") == first_week)


def _continuity_map(snapshot: pl.DataFrame) -> dict[str, dict[str, float]]:
    same_team = (
        (pl.col("snap_games_observed") > 0)
        & pl.col("prior_team_id").is_not_null()
        & (pl.col("prior_team_id") == pl.col("team_id"))
    )
    return {
        str(row["team_id"]): {
            "offense": float(np.clip(row["retained_offense"] / 11.0, 0.0, 1.0)),
            "defense": float(np.clip(row["retained_defense"] / 11.0, 0.0, 1.0)),
        }
        for row in snapshot.group_by("team_id")
        .agg(
            pl.when(same_team)
            .then(pl.col("offense_snap_share").fill_null(0.0))
            .otherwise(0.0)
            .sum()
            .alias("retained_offense"),
            pl.when(same_team)
            .then(pl.col("defense_snap_share").fill_null(0.0))
            .otherwise(0.0)
            .sum()
            .alias("retained_defense"),
        )
        .to_dicts()
    }


def _policy_means(policy: pl.DataFrame) -> dict[str, float]:
    cols = [
        "td_drive_rate", "fg_drive_rate", "turnover_drive_rate", "red_zone_td_rate",
        "offensive_epa_per_play", "offensive_success_rate", "offensive_explosive_rate",
        "defensive_td_drive_rate_allowed", "defensive_fg_drive_rate_allowed",
        "defensive_takeaway_drive_rate", "defensive_epa_allowed_per_play",
        "defensive_explosive_rate_allowed", "defensive_sack_rate", "defensive_qb_hit_rate",
    ]
    means: dict[str, float] = {}
    for col in cols:
        if col in policy.columns:
            value = policy.get_column(col).mean()
            means[col] = float(value) if value is not None else 0.0
    return means


def _shrink(value: float, anchor: float, retention: float) -> float:
    # Roster-retained snap share is the evidence weight. No fitted coefficient is used.
    return float(retention * value + (1.0 - retention) * anchor)


def _apply_continuity(
    state: TeamState,
    continuity: dict[str, float],
    means: dict[str, float],
) -> TeamState:
    offense = continuity["offense"]
    defense = continuity["defense"]
    return replace(
        state,
        td_drive_rate=_shrink(state.td_drive_rate, means["td_drive_rate"], offense),
        fg_drive_rate=_shrink(state.fg_drive_rate, means["fg_drive_rate"], offense),
        turnover_drive_rate=_shrink(
            state.turnover_drive_rate, means["turnover_drive_rate"], offense
        ),
        red_zone_td_rate=_shrink(state.red_zone_td_rate, means["red_zone_td_rate"], offense),
        offensive_epa_per_play=_shrink(
            state.offensive_epa_per_play, means["offensive_epa_per_play"], offense
        ),
        offensive_success_rate=_shrink(
            state.offensive_success_rate, means["offensive_success_rate"], offense
        ),
        offensive_explosive_rate=_shrink(
            state.offensive_explosive_rate, means["offensive_explosive_rate"], offense
        ),
        defensive_td_drive_rate_allowed=_shrink(
            state.defensive_td_drive_rate_allowed,
            means["defensive_td_drive_rate_allowed"],
            defense,
        ),
        defensive_fg_drive_rate_allowed=_shrink(
            state.defensive_fg_drive_rate_allowed,
            means["defensive_fg_drive_rate_allowed"],
            defense,
        ),
        defensive_takeaway_drive_rate=_shrink(
            state.defensive_takeaway_drive_rate,
            means["defensive_takeaway_drive_rate"],
            defense,
        ),
        defensive_epa_allowed_per_play=_shrink(
            state.defensive_epa_allowed_per_play,
            means["defensive_epa_allowed_per_play"],
            defense,
        ),
        defensive_explosive_rate_allowed=_shrink(
            state.defensive_explosive_rate_allowed,
            means["defensive_explosive_rate_allowed"],
            defense,
        ),
        defensive_sack_rate=_shrink(
            state.defensive_sack_rate, means["defensive_sack_rate"], defense
        ),
        defensive_qb_hit_rate=_shrink(
            state.defensive_qb_hit_rate, means["defensive_qb_hit_rate"], defense
        ),
        continuity=float(np.clip((offense + defense) / 2.0, 0.0, 1.0)),
    )


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _build_preweek1_personnel(train: int, test: int) -> tuple[pl.DataFrame, dict]:
    inputs = load_league_personnel_inputs([train], test, Path(".cache/monster"))
    rosters = _week1_rosters(inputs["current_rosters"])
    snapshot = build_league_personnel_snapshot(
        rosters,
        inputs["players"],
        inputs["historical_snap_counts"],
        None,
        recent_games=6,
    )
    snapshot = infer_game_day_participation(snapshot, season=test)
    snapshot = attach_capability_evidence(
        snapshot,
        inputs["combine"],
        inputs["pfr_defense_weekly"],
        inputs["player_stats_history"],
    )
    return snapshot, inputs


def run_fold(train: int, test: int, worlds: int, seed: int) -> tuple[list[dict], pl.DataFrame]:
    pbp = _regular_pbp(train)
    raw_policy = compile_team_policy(pbp)
    policy = pl.DataFrame({"team_id": list(NFL_TEAMS)}).join(
        raw_policy, on="team_id", how="left"
    ).sort("team_id")
    means = _policy_means(policy)
    snapshot, _ = _build_preweek1_personnel(train, test)
    continuity = _continuity_map(snapshot)
    unit_map = compile_league_unit_player_map(snapshot)
    effects = {team: compile_team_unit_effects(players)[0] for team, players in unit_map.items()}
    schedule = _regular_schedule(test)

    records: list[dict] = []
    for idx, row in enumerate(schedule.to_dicts()):
        away = normalize_team_id(str(row["away_team"]))
        home = normalize_team_id(str(row["home_team"]))
        base = compile_team_state_map(policy, {away: home, home: away}, prior_uncertainty=0.16)
        actual_total = float(row["away_score"] + row["home_score"])
        actual_margin = float(row["away_score"] - row["home_score"])
        for variant_idx, variant in enumerate(VARIANTS):
            away_state = base[away]
            home_state = base[home]
            if variant in {"continuity", "continuity_units"}:
                away_state = _apply_continuity(away_state, continuity[away], means)
                home_state = _apply_continuity(home_state, continuity[home], means)
            if variant in {"units", "continuity_units"}:
                away_state = apply_team_unit_effects(away_state, effects[away])
                home_state = apply_team_unit_effects(home_state, effects[home])
            result = simulate_game(
                GameState(f"{away}@{home}-{variant}", away_state, home_state),
                worlds=worlds,
                seed=seed + idx * 101 + variant_idx * 1_000_003,
            )
            model_total = float((result.away_points + result.home_points).mean())
            model_margin = float((result.away_points - result.home_points).mean())
            records.append(
                {
                    "train_season": train,
                    "test_season": test,
                    "variant": variant,
                    "game_id": str(row.get("game_id") or f"{away}@{home}"),
                    "week": row.get("week"),
                    "away": away,
                    "home": home,
                    "actual_total": actual_total,
                    "actual_margin": actual_margin,
                    "model_total": model_total,
                    "model_margin": model_margin,
                    "away_offense_continuity": continuity[away]["offense"],
                    "away_defense_continuity": continuity[away]["defense"],
                    "home_offense_continuity": continuity[home]["offense"],
                    "home_defense_continuity": continuity[home]["defense"],
                }
            )

    frame = pl.DataFrame(records)
    metrics: list[dict] = []
    for variant in VARIANTS:
        subset = frame.filter(pl.col("variant") == variant)
        actual_total = subset.get_column("actual_total").to_numpy()
        model_total = subset.get_column("model_total").to_numpy()
        actual_margin = subset.get_column("actual_margin").to_numpy()
        model_margin = subset.get_column("model_margin").to_numpy()
        metrics.append(
            {
                "train_season": train,
                "test_season": test,
                "variant": variant,
                "games": subset.height,
                "total_bias": float((model_total - actual_total).mean()),
                "total_mae": float(np.abs(model_total - actual_total).mean()),
                "total_corr": _corr(model_total, actual_total),
                "margin_mae": float(np.abs(model_margin - actual_margin).mean()),
                "margin_corr": _corr(model_margin, actual_margin),
            }
        )
    return metrics, frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worlds", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026090721)
    parser.add_argument("--out", type=Path, default=Path("artifacts/oos-personnel-bridge"))
    args = parser.parse_args()

    all_metrics: list[dict] = []
    frames: list[pl.DataFrame] = []
    for fold_idx, (train, test) in enumerate(((2023, 2024), (2024, 2025))):
        metrics, frame = run_fold(
            train, test, args.worlds, args.seed + fold_idx * 10_000_019
        )
        all_metrics.extend(metrics)
        frames.append(frame)

    metrics_frame = pl.DataFrame(all_metrics)
    summary = (
        metrics_frame.group_by("variant")
        .agg(
            pl.col("total_mae").mean().alias("mean_total_mae"),
            pl.col("total_corr").mean().alias("mean_total_corr"),
            pl.col("margin_mae").mean().alias("mean_margin_mae"),
            pl.col("margin_corr").mean().alias("mean_margin_corr"),
            pl.col("total_bias").mean().alias("mean_total_bias"),
        )
        .sort("mean_total_mae")
    )
    policy_row = summary.filter(pl.col("variant") == "policy").row(0, named=True)
    candidates = summary.filter(pl.col("variant") != "policy").with_columns(
        (pl.lit(policy_row["mean_total_mae"]) - pl.col("mean_total_mae")).alias(
            "total_mae_gain_vs_policy"
        ),
        (pl.lit(policy_row["mean_margin_mae"]) - pl.col("mean_margin_mae")).alias(
            "margin_mae_gain_vs_policy"
        ),
    )
    best = candidates.sort(
        ["total_mae_gain_vs_policy", "margin_mae_gain_vs_policy"],
        descending=True,
    ).row(0, named=True)
    bridge_pass = bool(
        best["total_mae_gain_vs_policy"] > 0
        and best["margin_mae_gain_vs_policy"] > 0
        and best["mean_total_corr"] >= policy_row["mean_total_corr"]
        and best["mean_margin_corr"] >= policy_row["mean_margin_corr"]
    )

    manifest = {
        "artifact": "Monster Pre-Week-1 Personnel Bridge OOS Ablation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "worlds_per_game_variant": args.worlds,
        "fold_metrics": all_metrics,
        "summary": summary.to_dicts(),
        "best_bridge_candidate": best,
        "bridge_pass": bridge_pass,
        "test_season_information_allowed": [
            "Week 1 roster membership/status",
            "static player identity/physical facts",
        ],
        "test_season_information_forbidden": [
            "test-season PBP",
            "test-season snap outcomes",
            "sportsbook market",
            "DFS salary/ownership/projections",
        ],
        "principle": "Offseason team identity must be reconstructed from current personnel rather than assumed to persist unchanged.",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    pl.concat(frames).write_csv(args.out / "ablation_games.csv")
    metrics_frame.write_csv(args.out / "fold_metrics.csv")
    summary.write_csv(args.out / "summary.csv")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
