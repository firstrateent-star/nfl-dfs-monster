from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import nflreadpy as nfl
import numpy as np
import polars as pl

from monster.feature_compile.game_flow_policy import (
    compile_game_flow_policy,
    prepare_game_flow_plays,
)
from monster.feature_compile.situation import compile_situational_pass_context
from monster.sim.decision_policy import situation_policy
from monster.sim.football_state import FootballState
from monster.sim.game_flow import FlowTag, derive_game_flow_state
from monster.sim.game_flow_brain import decide_game_flow
from monster.sim.game_flow_lookup import build_team_game_flow_policy
from monster.teams import TEAM_ALIASES

_DISTANCE_BUCKETS = ("short", "medium", "long")
_EPS = 1e-6


def _legacy_context_map(context: pl.DataFrame) -> tuple[float, dict[tuple[int, str], float]]:
    rows = {
        (int(row["down"]), str(row["distance_bucket"])): float(row["pass_rate"])
        for row in context.to_dicts()
    }
    league = float(context.get_column("league_neutral_pass_rate").drop_nulls()[0])
    return league, rows


def _legacy_bucket(distance: float) -> str:
    if distance <= 3.0:
        return "short"
    if distance <= 7.0:
        return "medium"
    return "long"


def _state_from_row(row: dict[str, object]) -> FootballState:
    diff = round(float(row["score_differential"]))
    return FootballState(
        possession="away",
        defense="home",
        quarter=min(max(int(row["qtr"]), 1), 5),
        seconds_remaining=max(int(float(row["game_seconds_remaining"])), 0),
        yardline_100=float(np.clip(100.0 - float(row["yardline_100"]), 1.0, 99.0)),
        down=int(row["down"]),
        distance=max(float(row["ydstogo"]), 0.1),
        away_score=max(diff, 0),
        home_score=max(-diff, 0),
    )


def _log_loss(actual: np.ndarray, probability: np.ndarray) -> float:
    p = np.clip(probability, _EPS, 1.0 - _EPS)
    return float(np.mean(-(actual * np.log(p) + (1.0 - actual) * np.log(1.0 - p))))


def _brier(actual: np.ndarray, probability: np.ndarray) -> float:
    return float(np.mean((probability - actual) ** 2))


def _primary_state(tags: frozenset[FlowTag]) -> str:
    ordered = (
        (FlowTag.THIRD_AND_EXTREME, "third_and_18_plus"),
        (FlowTag.THIRD_AND_LONG, "third_and_7_plus"),
        (FlowTag.THIRD_AND_MEDIUM, "third_and_medium"),
        (FlowTag.THIRD_AND_SHORT, "third_and_short"),
        (FlowTag.SECOND_AND_SHORT, "second_and_2_or_less"),
        (FlowTag.END_FIRST_HALF, "end_first_half"),
        (FlowTag.LATE_TRAILING, "late_trailing"),
        (FlowTag.FOUR_MINUTE_LEAD, "four_minute_lead"),
        (FlowTag.LOW_RED_ZONE, "low_red_zone"),
        (FlowTag.BACKED_UP, "backed_up"),
    )
    for tag, label in ordered:
        if tag in tags:
            return label
    return "ordinary"


def _team_neutral_rates(frame: pl.DataFrame) -> tuple[float, dict[str, float]]:
    league = float(frame.get_column("is_dropback").mean())
    team = {
        str(row["posteam"]): float(row["neutral_rate"])
        for row in frame.group_by("posteam")
        .agg(pl.col("is_dropback").mean().alias("neutral_rate"))
        .to_dicts()
    }
    return league, team


def _evaluate_fold(train_season: int, test_season: int) -> tuple[dict, pl.DataFrame]:
    train = nfl.load_pbp([train_season])
    test = nfl.load_pbp([test_season])
    if "season_type" in train.columns:
        train = train.filter(pl.col("season_type") == "REG")
    if "season_type" in test.columns:
        test = test.filter(pl.col("season_type") == "REG")

    train_flow = prepare_game_flow_plays(train)
    test_flow = prepare_game_flow_plays(test)
    league_rows, team_rows = compile_game_flow_policy(train)
    league_neutral, team_neutral = _team_neutral_rates(train_flow)
    legacy_league, legacy_context = _legacy_context_map(compile_situational_pass_context(train))

    league_dicts = league_rows.to_dicts()
    team_dicts = team_rows.to_dicts()
    policies = {
        team_id: build_team_game_flow_policy(
            team_id=team_id,
            league_rows=league_dicts,
            team_rows=team_dicts,
            team_neutral_rate=team_neutral.get(team_id),
            league_neutral_rate=league_neutral,
        )
        for team_id in team_neutral
    }

    rows: list[dict] = []
    for raw in test_flow.with_columns(pl.col("posteam").replace(TEAM_ALIASES)).to_dicts():
        team_id = str(raw["posteam"])
        policy = policies.get(team_id)
        if policy is None:
            continue
        state = _state_from_row(raw)
        flow = derive_game_flow_state(state)
        new_probability = decide_game_flow(flow, policy.evidence_for(flow)).dropback_probability

        bucket = _legacy_bucket(state.distance)
        league_context_rate = legacy_context[(state.down, bucket)]
        team_delta = team_neutral.get(team_id, legacy_league) - legacy_league
        contextual = float(np.clip(league_context_rate + team_delta, 0.18, 0.90))
        old_probability = situation_policy(
            state,
            team_neutral.get(team_id, legacy_league),
            contextual_pass_rate=contextual,
        ).pass_probability
        rows.append(
            {
                "team_id": team_id,
                "actual": float(raw["is_dropback"]),
                "legacy_probability": old_probability,
                "game_flow_probability": new_probability,
                "special_state": _primary_state(flow.tags),
            }
        )

    scored = pl.DataFrame(rows)
    actual = scored.get_column("actual").to_numpy()
    legacy = scored.get_column("legacy_probability").to_numpy()
    new = scored.get_column("game_flow_probability").to_numpy()
    fold = {
        "train_season": train_season,
        "test_season": test_season,
        "samples": scored.height,
        "legacy_brier": _brier(actual, legacy),
        "game_flow_brier": _brier(actual, new),
        "legacy_log_loss": _log_loss(actual, legacy),
        "game_flow_log_loss": _log_loss(actual, new),
    }
    fold["brier_gain"] = fold["legacy_brier"] - fold["game_flow_brier"]
    fold["log_loss_gain"] = fold["legacy_log_loss"] - fold["game_flow_log_loss"]

    special = (
        scored.group_by("special_state")
        .agg(
            pl.len().alias("samples"),
            pl.col("actual").mean().alias("actual_dropback_rate"),
            pl.col("legacy_probability").mean().alias("legacy_mean_probability"),
            pl.col("game_flow_probability").mean().alias("game_flow_mean_probability"),
        )
        .with_columns(
            pl.lit(train_season).alias("train_season"),
            pl.lit(test_season).alias("test_season"),
            (pl.col("legacy_mean_probability") - pl.col("actual_dropback_rate"))
            .abs()
            .alias("legacy_abs_calibration_error"),
            (pl.col("game_flow_mean_probability") - pl.col("actual_dropback_rate"))
            .abs()
            .alias("game_flow_abs_calibration_error"),
        )
    )
    return fold, special


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("artifacts/game-flow-brain-oos"))
    args = parser.parse_args()

    folds = []
    specials = []
    for train_season, test_season in ((2023, 2024), (2024, 2025)):
        fold, special = _evaluate_fold(train_season, test_season)
        folds.append(fold)
        specials.append(special)

    fold_df = pl.DataFrame(folds)
    special_df = pl.concat(specials, how="vertical")
    args.out.mkdir(parents=True, exist_ok=True)
    fold_df.write_csv(args.out / "fold_metrics.csv")
    special_df.write_csv(args.out / "special_state_calibration.csv")

    manifest = {
        "artifact": "Monster Hierarchical Game Flow Brain OOS Audit",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "folds": [[2023, 2024], [2024, 2025]],
        "market_blind": True,
        "behavior_changed_by_audit": False,
        "mean_brier_gain": float(fold_df.get_column("brier_gain").mean()),
        "mean_log_loss_gain": float(fold_df.get_column("log_loss_gain").mean()),
        "positive_brier_folds": int((fold_df.get_column("brier_gain") > 0).sum()),
        "positive_log_loss_folds": int((fold_df.get_column("log_loss_gain") > 0).sum()),
        "principle": (
            "Game Flow earns live decision authority only if fine context plus shrunk team style "
            "generalizes beyond the season used to compile it."
        ),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(fold_df)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
