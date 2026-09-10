from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import nflreadpy as nfl
import polars as pl

from monster.ingest.nflverse import configure_cache
from monster.sim.decision_policy import FourthDownDecision, fourth_down_decision
from monster.sim.football_state import FootballState


def _num(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _historical_choice(row: dict[str, Any]) -> str | None:
    if str(row.get("play_type", "")).lower() == "punt":
        return FourthDownDecision.PUNT.value
    if _num(row, "field_goal_attempt") == 1.0:
        return FourthDownDecision.FIELD_GOAL.value
    if _num(row, "qb_dropback") == 1.0 or _num(row, "rush_attempt") == 1.0:
        return FourthDownDecision.GO.value
    return None


def _bucket_distance(distance: float) -> str:
    if distance <= 1.0:
        return "1"
    if distance <= 3.0:
        return "2_3"
    if distance <= 6.0:
        return "4_6"
    return "7_plus"


def _bucket_field(yardline_100: float) -> str:
    if yardline_100 < 40.0:
        return "own_1_39"
    if yardline_100 < 55.0:
        return "own_40_to_midfield"
    if yardline_100 < 70.0:
        return "plus_45_to_31"
    if yardline_100 < 80.0:
        return "plus_30_to_21"
    return "red_zone"


def _rate(counter: Counter[str]) -> dict[str, float]:
    total = sum(counter.values())
    return {
        decision: counter[decision] / total if total else 0.0
        for decision in ("go", "field_goal", "punt")
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/monster"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    configure_cache(args.cache_dir)
    pbp = nfl.load_pbp([args.season])
    if "season_type" in pbp.columns:
        pbp = pbp.filter(pl.col("season_type") == "REG")
    usable = pbp.filter(
        (pl.col("down") == 4)
        & pl.col("posteam").is_not_null()
        & pl.col("yardline_100").is_not_null()
        & pl.col("ydstogo").is_not_null()
    )

    overall_hist: Counter[str] = Counter()
    overall_policy: Counter[str] = Counter()
    cells: dict[tuple[str, str], dict[str, Counter[str]]] = defaultdict(
        lambda: {"historical": Counter(), "policy": Counter()}
    )
    compared = 0
    agreements = 0

    for row in usable.to_dicts():
        historical = _historical_choice(row)
        if historical is None:
            continue
        distance = _num(row, "ydstogo", 10.0)
        yardline_100 = 100.0 - _num(row, "yardline_100", 75.0)
        quarter = int(_num(row, "qtr", 1.0))
        game_seconds = int(_num(row, "game_seconds_remaining", 3600.0))
        posteam_score = _num(row, "posteam_score", 0.0)
        defteam_score = _num(row, "defteam_score", 0.0)

        state = FootballState(
            possession="offense",
            defense="defense",
            quarter=max(min(quarter, 5), 1),
            seconds_remaining=max(game_seconds, 0),
            yardline_100=yardline_100,
            down=4,
            distance=max(distance, 0.1),
            away_score=round(posteam_score),
            home_score=round(defteam_score),
            away_team_id="offense",
            home_team_id="defense",
        )
        policy = fourth_down_decision(state).value
        key = (_bucket_field(yardline_100), _bucket_distance(distance))
        overall_hist[historical] += 1
        overall_policy[policy] += 1
        cells[key]["historical"][historical] += 1
        cells[key]["policy"][policy] += 1
        compared += 1
        agreements += int(historical == policy)

    rows = []
    for (field_bucket, distance_bucket), stores in sorted(cells.items()):
        historical = _rate(stores["historical"])
        policy = _rate(stores["policy"])
        rows.append(
            {
                "field_bucket": field_bucket,
                "distance_bucket": distance_bucket,
                "samples": sum(stores["historical"].values()),
                "historical_go_rate": historical["go"],
                "policy_go_rate": policy["go"],
                "go_rate_delta": policy["go"] - historical["go"],
                "historical_fg_rate": historical["field_goal"],
                "policy_fg_rate": policy["field_goal"],
                "historical_punt_rate": historical["punt"],
                "policy_punt_rate": policy["punt"],
            }
        )

    report = {
        "artifact": "Monster Fourth-Down Decision Ecology Audit",
        "season": args.season,
        "market_blind": True,
        "behavior_changed_by_audit": False,
        "fourth_downs_compared": compared,
        "exact_decision_agreement_rate": agreements / compared if compared else 0.0,
        "historical_overall": _rate(overall_hist),
        "monster_policy_overall_on_historical_states": _rate(overall_policy),
        "interpretation_rule": "Use historical game states to identify decision-policy jurisdiction. Do not tune scores, punts or totals directly.",
        "promotion_status": "SHADOW_DIAGNOSTIC_ONLY",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(args.out / "fourth_down_decision_cells.csv")
    (args.out / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
