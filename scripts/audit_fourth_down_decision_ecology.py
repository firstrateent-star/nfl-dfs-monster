from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import nflreadpy as nfl
import polars as pl

from monster.ingest.nflverse import configure_cache
from monster.sim.clock import seconds_remaining_in_quarter
from monster.sim.decision_policy import fourth_down_probabilities
from monster.sim.football_state import FootballState


def _ensure(frame: pl.DataFrame, defaults: dict[str, object]) -> pl.DataFrame:
    missing = [pl.lit(value).alias(name) for name, value in defaults.items() if name not in frame.columns]
    return frame.with_columns(*missing) if missing else frame


def _actual_action_expr() -> pl.Expr:
    punt = pl.col("punt_attempt").fill_null(0).cast(pl.Int64) == 1
    fg = pl.col("field_goal_attempt").fill_null(0).cast(pl.Int64) == 1
    go = (
        (pl.col("qb_dropback").fill_null(0).cast(pl.Int64) == 1)
        | (pl.col("rush_attempt").fill_null(0).cast(pl.Int64) == 1)
    )
    return (
        pl.when(punt)
        .then(pl.lit("punt"))
        .when(fg)
        .then(pl.lit("field_goal"))
        .when(go)
        .then(pl.lit("go"))
        .otherwise(pl.lit(None, dtype=pl.Utf8))
    )


def _context(*, quarter: int, game_seconds: int, margin: float) -> str:
    q_clock = seconds_remaining_in_quarter(game_seconds)
    if quarter == 5:
        if margin < 0:
            return "overtime_trailing"
        if margin == 0:
            return "overtime_tied"
        return "overtime_leading"
    if quarter == 2 and q_clock <= 30:
        return "end_half_30s"
    if quarter == 2 and q_clock <= 120:
        return "end_half_2m"
    if quarter == 4 and q_clock <= 30:
        window = "30s"
    elif quarter == 4 and q_clock <= 120:
        window = "2m"
    elif quarter == 4 and q_clock <= 360:
        window = "6m"
    else:
        return "ordinary"
    if margin < -8:
        score = "trailing_9plus"
    elif margin < -3:
        score = "trailing_4_8"
    elif margin < 0:
        score = "trailing_1_3"
    elif margin == 0:
        score = "tied"
    else:
        score = "leading"
    return f"late_game_{window}_{score}"


def _distance_bucket(distance: float) -> str:
    if distance <= 1.0:
        return "1"
    if distance <= 3.0:
        return "2_3"
    if distance <= 6.0:
        return "4_6"
    return "7_plus"


def _field_zone(yardline: float) -> str:
    if yardline < 40.0:
        return "own_1_39"
    if yardline < 55.0:
        return "own_40_to_midfield"
    if yardline < 70.0:
        return "plus_45_to_31"
    if yardline < 80.0:
        return "plus_30_to_21"
    return "red_zone"


def _load_history(season: int, cache: Path) -> pl.DataFrame:
    configure_cache(cache)
    pbp = nfl.load_pbp([season])
    if "season_type" in pbp.columns:
        pbp = pbp.filter(pl.col("season_type") == "REG")
    pbp = _ensure(
        pbp,
        {
            "punt_attempt": 0,
            "field_goal_attempt": 0,
            "qb_dropback": 0,
            "rush_attempt": 0,
            "score_differential": 0.0,
            "game_seconds_remaining": None,
            "yardline_100": None,
            "ydstogo": None,
            "qtr": None,
            "down": None,
            "posteam": None,
        },
    )
    return (
        pbp.with_columns(_actual_action_expr().alias("actual_action"))
        .filter(
            (pl.col("down") == 4)
            & pl.col("actual_action").is_not_null()
            & pl.col("yardline_100").is_not_null()
            & pl.col("ydstogo").is_not_null()
            & pl.col("qtr").is_not_null()
            & pl.col("game_seconds_remaining").is_not_null()
            & pl.col("posteam").is_not_null()
        )
        .with_columns(
            (100.0 - pl.col("yardline_100").cast(pl.Float64)).alias("monster_yardline_100"),
            pl.col("ydstogo").cast(pl.Float64).alias("distance"),
            pl.col("score_differential").cast(pl.Float64).fill_null(0.0).alias("margin"),
            pl.col("game_seconds_remaining").cast(pl.Int64).alias("game_seconds"),
            pl.col("qtr").cast(pl.Int64).alias("quarter"),
        )
    )


def _model_rows(history: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for row in history.select(
        "game_id",
        "posteam",
        "quarter",
        "game_seconds",
        "monster_yardline_100",
        "distance",
        "margin",
        "actual_action",
    ).iter_rows(named=True):
        margin = float(row["margin"])
        away_score = max(int(round(margin)), 0)
        home_score = max(int(round(-margin)), 0)
        state = FootballState(
            possession="off",
            defense="def",
            quarter=int(row["quarter"]),
            seconds_remaining=int(row["game_seconds"]),
            yardline_100=float(row["monster_yardline_100"]),
            down=4,
            distance=float(row["distance"]),
            away_score=away_score,
            home_score=home_score,
            away_team_id="off",
            home_team_id="def",
        )
        probs = fourth_down_probabilities(state)
        context = _context(
            quarter=state.quarter,
            game_seconds=state.seconds_remaining,
            margin=state.score_margin_for_offense,
        )
        actual = str(row["actual_action"])
        actual_probability = {
            "go": probs.go,
            "field_goal": probs.field_goal,
            "punt": probs.punt,
        }[actual]
        rows.append(
            {
                **row,
                "context": context,
                "distance_bucket": _distance_bucket(state.distance),
                "field_zone": _field_zone(state.yardline_100),
                "q_clock": seconds_remaining_in_quarter(state.seconds_remaining),
                "model_go": probs.go,
                "model_field_goal": probs.field_goal,
                "model_punt": probs.punt,
                "actual_probability": actual_probability,
                "brier": (
                    (probs.go - float(actual == "go")) ** 2
                    + (probs.field_goal - float(actual == "field_goal")) ** 2
                    + (probs.punt - float(actual == "punt")) ** 2
                ),
            }
        )
    return pl.DataFrame(rows)


def _summary(frame: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    return (
        frame.group_by(keys)
        .agg(
            pl.len().alias("states"),
            (pl.col("actual_action") == "go").mean().alias("actual_go_rate"),
            (pl.col("actual_action") == "field_goal").mean().alias("actual_field_goal_rate"),
            (pl.col("actual_action") == "punt").mean().alias("actual_punt_rate"),
            pl.col("model_go").mean().alias("model_go_rate"),
            pl.col("model_field_goal").mean().alias("model_field_goal_rate"),
            pl.col("model_punt").mean().alias("model_punt_rate"),
            pl.col("actual_probability").mean().alias("mean_probability_on_actual_choice"),
            pl.col("brier").mean().alias("multiclass_brier"),
        )
        .with_columns(
            (pl.col("model_go_rate") - pl.col("actual_go_rate")).alias("go_rate_delta"),
            (pl.col("model_field_goal_rate") - pl.col("actual_field_goal_rate")).alias("field_goal_rate_delta"),
            (pl.col("model_punt_rate") - pl.col("actual_punt_rate")).alias("punt_rate_delta"),
        )
        .sort(keys)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--out", type=Path, default=Path("artifacts/fourth-down-decision-audit"))
    parser.add_argument("--cache", type=Path, default=Path(".cache/nflverse"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    history = _load_history(args.season, args.cache)
    evaluated = _model_rows(history)
    evaluated.write_csv(args.out / "fourth_down_states.csv")
    _summary(evaluated, ["context"]).write_csv(args.out / "context_summary.csv")
    _summary(evaluated, ["context", "distance_bucket"]).write_csv(
        args.out / "context_distance_summary.csv"
    )
    _summary(evaluated, ["field_zone", "distance_bucket"]).write_csv(
        args.out / "zone_distance_summary.csv"
    )

    key_contexts = evaluated.filter(
        pl.col("context").str.starts_with("end_half")
        | pl.col("context").str.starts_with("late_game")
        | pl.col("context").str.starts_with("overtime")
    )
    _summary(key_contexts, ["context", "field_zone", "distance_bucket"]).write_csv(
        args.out / "terminal_context_detail.csv"
    )

    manifest = {
        "experiment": "MON-LEDGER-002C-DECISION",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "season": args.season,
        "states": evaluated.height,
        "center": "Fourth-down choices are state-conditioned football decisions, not automatic drive terminals.",
        "questions": [
            "Does Monster preserve observed probabilistic go/field-goal/punt behavior on ordinary fourth downs?",
            "Does Monster react correctly when the half is ending and punting may have little value?",
            "Does Monster preserve possession when trailing late and the game otherwise ends?",
            "Does Monster choose a tying/winning field goal when score, clock, and field position make that rational?",
            "Does overtime use a distinct decision contract?",
        ],
        "governance": {
            "stage": "LAB",
            "market_inputs_used": False,
            "football_coefficients_changed_for_experiment": False,
            "production_promoted": False,
        },
        "authority_note": "Historical 2025 fourth-down actions are evaluation evidence. The audit has no authority to change scoring or play outcomes.",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(_summary(evaluated, ["context"]))


if __name__ == "__main__":
    main()
