from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import nflreadpy as nfl
import numpy as np
import polars as pl

from monster.feature_compile.play_intent import (
    PASS_DEPTH_CATEGORIES,
    RUN_GEOMETRY_CATEGORIES,
    compile_pass_intent_policy,
    compile_run_intent_policy,
    prepare_pass_intent_plays,
    prepare_run_intent_plays,
)
from monster.ingest.nflverse import configure_cache
from monster.sim.categorical_policy import build_contextual_categorical_policy
from monster.sim.game_flow import GameFlowState


def _period_seconds(qtr: int, game_seconds: float) -> int:
    offsets = {1: 2700, 2: 1800, 3: 900, 4: 0}
    if qtr in offsets:
        return max(round(game_seconds - offsets[qtr]), 0)
    return max(round(game_seconds), 0)


def _flow(row: dict) -> GameFlowState:
    qtr = int(row["qtr"])
    game_seconds = float(row["game_seconds_remaining"])
    yardline = 100.0 - float(row["yardline_100"])
    return GameFlowState(
        down=int(row["down"]),
        distance=float(row["ydstogo"]),
        yardline=yardline,
        quarter=qtr,
        seconds_remaining=max(round(game_seconds), 0),
        seconds_remaining_in_period=_period_seconds(qtr, game_seconds),
        score_margin=round(float(row["score_differential"])),
        yards_to_goal=float(row["yardline_100"]),
        tags=frozenset(),
    )


def _metrics(
    actual: list[str],
    probabilities: list[np.ndarray],
    categories: tuple[str, ...],
) -> tuple[float, float]:
    index = {category: i for i, category in enumerate(categories)}
    brier = 0.0
    log_loss = 0.0
    for label, probs in zip(actual, probabilities, strict=True):
        one_hot = np.zeros(len(categories), dtype=float)
        one_hot[index[label]] = 1.0
        brier += float(np.square(probs - one_hot).sum())
        log_loss += -float(np.log(max(float(probs[index[label]]), 1e-12)))
    n = max(len(actual), 1)
    return brier / n, log_loss / n


def _evaluate_family(
    *,
    train: pl.DataFrame,
    test: pl.DataFrame,
    categories: tuple[str, ...],
    compiler,
    actor_column: str | None,
) -> dict:
    compiled = compiler(train)
    league = compiled[0]
    team = compiled[1]
    actor = compiled[2]
    league_rows = league.to_dicts()
    team_rows = team.to_dicts()
    actor_rows = actor.to_dicts()

    teams = sorted(
        {str(value) for value in test.get_column("posteam").drop_nulls().to_list()}
    )
    policies = {
        team_id: build_contextual_categorical_policy(
            categories=categories,
            league_rows=league_rows,
            team_rows=team_rows,
            actor_rows=actor_rows if actor_column is not None else (),
            team_id=team_id,
            league_shrinkage_samples=100.0,
            team_shrinkage_samples=100.0,
            actor_shrinkage_samples=120.0,
        )
        for team_id in teams
    }

    actual: list[str] = []
    predicted: list[np.ndarray] = []
    baseline: list[np.ndarray] = []
    for row in test.to_dicts():
        team_id = str(row["posteam"])
        policy = policies[team_id]
        flow = _flow(row)
        actor_id = None
        if actor_column is not None and row.get(actor_column) is not None:
            actor_id = str(row[actor_column])
        predicted.append(policy.probabilities_for(flow, actor_id=actor_id))
        baseline.append(policy.league_overall.probabilities(categories))
        actual.append(str(row["category"]))

    pred_brier, pred_log = _metrics(actual, predicted, categories)
    base_brier, base_log = _metrics(actual, baseline, categories)
    return {
        "samples": len(actual),
        "hierarchical_brier": pred_brier,
        "baseline_brier": base_brier,
        "brier_improvement": base_brier - pred_brier,
        "hierarchical_log_loss": pred_log,
        "baseline_log_loss": base_log,
        "log_loss_improvement": base_log - pred_log,
    }


def _load(season: int) -> pl.DataFrame:
    pbp = nfl.load_pbp([season])
    if "season_type" in pbp.columns:
        pbp = pbp.filter(pl.col("season_type") == "REG")
    return pbp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/monster"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/play-intent-oos"))
    args = parser.parse_args()
    configure_cache(args.cache_dir)

    seasons = {season: _load(season) for season in (2023, 2024, 2025)}
    folds = []
    for train_season, test_season in ((2023, 2024), (2024, 2025)):
        train = seasons[train_season]
        test = seasons[test_season]
        pass_result = _evaluate_family(
            train=train,
            test=prepare_pass_intent_plays(test),
            categories=PASS_DEPTH_CATEGORIES,
            compiler=compile_pass_intent_policy,
            actor_column="passer_player_id",
        )
        run_result = _evaluate_family(
            train=train,
            test=prepare_run_intent_plays(test),
            categories=RUN_GEOMETRY_CATEGORIES,
            compiler=compile_run_intent_policy,
            actor_column=None,
        )
        folds.append(
            {
                "train_season": train_season,
                "test_season": test_season,
                "pass": pass_result,
                "run": run_result,
            }
        )

    passed = all(
        fold[family][metric] > 0.0
        for fold in folds
        for family in ("pass", "run")
        for metric in ("brier_improvement", "log_loss_improvement")
    )
    manifest = {
        "artifact": "Monster Pass + Run Intent OOS Gate",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "market_blind": True,
        "folds": folds,
        "gate_requires": (
            "positive Brier and log-loss improvement for pass and run in both folds"
        ),
        "gate_passed": passed,
        "principle": (
            "Intent may gain runtime authority only when richer pre-snap context generalizes "
            "out of sample. Player identity is used only when known before the choice."
        ),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    if not passed:
        raise SystemExit("play intent OOS gate failed")


if __name__ == "__main__":
    main()
