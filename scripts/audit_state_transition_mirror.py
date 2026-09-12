from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import nflreadpy as nfl
import numpy as np
import polars as pl
from run_week1_v13_first_sim import (
    GAME_DATE,
    MATCHUPS,
    _defensive_unit,
    _read,
    _situational_context,
    _team_identity,
    _with_event_rush_plan,
)

from monster.feature_compile.health_pools import apply_health_to_skill_pools
from monster.feature_compile.league_units import compile_league_unit_player_map
from monster.feature_compile.reality_inputs import compile_player_reality_inputs
from monster.feature_compile.skill_pools import compile_current_skill_pools
from monster.ingest.nflverse import configure_cache
from monster.sim import play_kernel as play_kernel_module
from monster.sim.football_state import FootballState, apply_scrimmage_yards
from monster.sim.game_flow_lookup import build_team_game_flow_policy
from monster.sim.intent_ecology import build_intent_ecology
from monster.sim.play_kernel import PassResult, PlayType, simulate_scrimmage_play
from monster.sim.rules_v13 import PenaltySide, enforce_penalty, simulate_penalty
from monster.sim.rushing_roles import sample_event_rush_share_plan
from monster.sim.state_transition_eval import (
    ABSORB_CONVERTED,
    ABSORB_FAILED,
    ABSORB_TOUCHDOWN,
    coarse_state_key,
    fine_state_key,
    normalized,
    replace_transition_family,
    solve_series_survival,
    total_variation,
    transition_probabilities,
)
from monster.snapshot.league import compile_team_state_map


def _rows(path: Path) -> list[dict[str, object]]:
    return _read(path).to_dicts()


def _number(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _flag(row: dict[str, Any], key: str) -> bool:
    return _number(row, key) == 1.0


def _historical_first_down(row: dict[str, Any]) -> bool:
    fields = ("first_down", "first_down_pass", "first_down_rush", "first_down_penalty")
    return any(_flag(row, field) for field in fields if field in row)


def _historical_candidate(row: dict[str, Any]) -> bool:
    down = _number(row, "down", 0.0)
    if down < 1.0 or down > 4.0 or row.get("posteam") is None:
        return False
    return any(
        (
            _flag(row, "qb_dropback"),
            _flag(row, "rush_attempt"),
            _flag(row, "penalty"),
            _flag(row, "field_goal_attempt"),
            str(row.get("play_type", "")).lower() == "punt",
        )
    )


def _historical_terminal(row: dict[str, Any]) -> str | None:
    if any(
        _flag(row, field)
        for field in ("touchdown", "pass_touchdown", "rush_touchdown")
        if field in row
    ):
        return ABSORB_TOUCHDOWN
    if _flag(row, "interception") or _flag(row, "fumble_lost"):
        return ABSORB_FAILED
    if _flag(row, "fourth_down_failed"):
        return ABSORB_FAILED
    if str(row.get("play_type", "")).lower() == "punt":
        return ABSORB_FAILED
    if _flag(row, "field_goal_attempt"):
        return ABSORB_FAILED
    if _flag(row, "safety"):
        return ABSORB_FAILED
    return None


def _historical_cause(row: dict[str, Any], converted: bool) -> tuple[str, str]:
    penalty = _flag(row, "penalty") or row.get("penalty_team") is not None
    if penalty:
        penalty_team = str(row.get("penalty_team", ""))
        posteam = str(row.get("posteam", ""))
        if _flag(row, "first_down_penalty"):
            return "penalty", "defensive_penalty_first_down"
        if penalty_team and penalty_team == posteam:
            return "penalty", "offensive_penalty"
        return "penalty", "defensive_penalty"
    if _flag(row, "sack"):
        return "dropback", "sack"
    if _flag(row, "interception"):
        return "dropback", "interception"
    if _flag(row, "qb_scramble"):
        return "dropback", "scramble_conversion" if converted else "scramble_short"
    if _flag(row, "complete_pass"):
        return "dropback", "completion_conversion" if converted else "completion_short"
    if _flag(row, "incomplete_pass") or _flag(row, "qb_dropback"):
        return "dropback", "incomplete"
    if _flag(row, "rush_attempt"):
        return "run", "run_conversion" if converted else "run_short"
    if str(row.get("play_type", "")).lower() == "punt":
        return "punt", "punt"
    if _flag(row, "field_goal_attempt"):
        return "field_goal", "field_goal"
    return "other", "other"


def _next_historical_state(rows: list[dict[str, Any]], index: int, posteam: str) -> dict[str, Any] | None:
    for candidate in rows[index + 1 :]:
        if str(candidate.get("posteam", "")) != posteam:
            continue
        down = _number(candidate, "down", 0.0)
        if 1.0 <= down <= 4.0:
            return candidate
    return None


def _historical_transition_rows(season: int, cache_dir: Path) -> list[dict[str, object]]:
    configure_cache(cache_dir)
    pbp = nfl.load_pbp([season])
    if "season_type" in pbp.columns:
        pbp = pbp.filter(pl.col("season_type") == "REG")
    if "qb_kneel" in pbp.columns:
        pbp = pbp.filter(pl.col("qb_kneel").fill_null(0) == 0)
    if "qb_spike" in pbp.columns:
        pbp = pbp.filter(pl.col("qb_spike").fill_null(0) == 0)

    required = {
        "game_id",
        "fixed_drive",
        "posteam",
        "play_id",
        "down",
        "ydstogo",
        "yardline_100",
        "game_seconds_remaining",
    }
    missing = sorted(required.difference(pbp.columns))
    if missing:
        raise ValueError(f"state-transition mirror missing nflverse fields: {missing}")

    usable = pbp.filter(
        pl.col("game_id").is_not_null()
        & pl.col("fixed_drive").is_not_null()
        & pl.col("posteam").is_not_null()
    ).sort(["game_id", "fixed_drive", "play_id"])

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in usable.to_dicts():
        key = (str(row["game_id"]), str(row["fixed_drive"]), str(row["posteam"]))
        groups[key].append(row)

    output: list[dict[str, object]] = []
    for (game_id, fixed_drive, posteam), rows in groups.items():
        for index, row in enumerate(rows):
            if not _historical_candidate(row):
                continue
            quarter = int(_number(row, "qtr", 1.0))
            if quarter < 1 or quarter > 4:
                continue
            down = int(_number(row, "down", 0.0))
            distance = max(_number(row, "ydstogo", 10.0), 0.1)
            yardline_from_own = float(np.clip(100.0 - _number(row, "yardline_100", 75.0), 1.0, 99.0))
            converted = _historical_first_down(row)
            terminal = _historical_terminal(row)
            if terminal is not None:
                next_state = terminal
            elif converted:
                next_state = ABSORB_CONVERTED
            else:
                following = _next_historical_state(rows, index, posteam)
                if following is None:
                    next_state = ABSORB_FAILED
                else:
                    next_down = int(_number(following, "down", min(down + 1, 4)))
                    next_distance = max(_number(following, "ydstogo", distance), 0.1)
                    if next_down == 1 and down != 1:
                        next_state = ABSORB_CONVERTED
                    else:
                        next_state = coarse_state_key(next_down, next_distance)
            family, cause = _historical_cause(row, converted)
            output.append(
                {
                    "source": "nfl_2025",
                    "game_id": game_id,
                    "drive_id": fixed_drive,
                    "offense_team_id": posteam,
                    "down": down,
                    "distance": distance,
                    "yardline_from_own": yardline_from_own,
                    "quarter": quarter,
                    "seconds_remaining": int(_number(row, "game_seconds_remaining", 1800.0)),
                    "score_margin": _number(row, "posteam_score", 0.0) - _number(row, "defteam_score", 0.0),
                    "coarse_state": coarse_state_key(down, distance),
                    "fine_state": fine_state_key(down, distance, yardline_from_own),
                    "play_family": family,
                    "cause": cause,
                    "converted": converted,
                    "touchdown": next_state == ABSORB_TOUCHDOWN,
                    "yards": _number(row, "yards_gained", 0.0),
                    "next_state": next_state,
                }
            )
    if not output:
        raise RuntimeError("historical transition mirror produced no rows")
    return output


def _stratified_sample(
    rows: list[dict[str, object]],
    *,
    maximum: int,
    seed: int,
) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row["coarse_state"])].append(row)
    rng = np.random.default_rng(seed)
    per_group = max(maximum // max(len(groups), 1), 1)
    chosen: list[dict[str, object]] = []
    leftovers: list[dict[str, object]] = []
    for key in sorted(groups):
        group = groups[key]
        order = rng.permutation(len(group)).tolist()
        cut = min(per_group, len(group))
        chosen.extend(group[index] for index in order[:cut])
        leftovers.extend(group[index] for index in order[cut:])
    if len(chosen) < maximum and leftovers:
        order = rng.permutation(len(leftovers)).tolist()
        chosen.extend(leftovers[index] for index in order[: maximum - len(chosen)])
    return chosen[:maximum]


def _monster_cause(event, converted: bool) -> tuple[str, str]:
    if event.play_type == PlayType.PUNT:
        return "punt", "punt"
    if event.play_type == PlayType.FIELD_GOAL:
        return "field_goal", "field_goal"
    if event.play_type == PlayType.RUN:
        if event.turnover:
            return "run", "run_fumble_turnover"
        if event.touchdown:
            return "run", "run_touchdown"
        return "run", "run_conversion" if converted else "run_short"
    if event.pass_result == PassResult.SACK:
        return "dropback", "sack"
    if event.pass_result == PassResult.INTERCEPTION:
        return "dropback", "interception"
    if event.pass_result == PassResult.SCRAMBLE:
        return "dropback", "scramble_conversion" if converted else "scramble_short"
    if event.pass_result == PassResult.COMPLETE:
        if event.touchdown:
            return "dropback", "completion_touchdown"
        return "dropback", "completion_conversion" if converted else "completion_short"
    return "dropback", "incomplete"


def _build_current_teams(args) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    policy = _read(args.policy)
    personnel = _read(args.personnel)
    usage = _read(args.player_usage)
    situation_context = _read(args.situation_context)
    league_neutral_pass_rate, situational_pass_rates = _situational_context(situation_context)

    pools = apply_health_to_skill_pools(
        compile_current_skill_pools(personnel, usage, policy=policy),
        personnel,
    )
    reality = compile_player_reality_inputs(personnel, game_date=GAME_DATE)
    units = compile_league_unit_player_map(personnel)
    states = {}
    opponents: dict[str, str] = {}
    for away, home in MATCHUPS:
        states.update(compile_team_state_map(policy, {away: home, home: away}))
        opponents[away] = home
        opponents[home] = away

    game_flow_league = _rows(args.game_flow_league)
    game_flow_team = _rows(args.game_flow_team)
    pass_league = _rows(args.pass_depth_league)
    pass_team = _rows(args.pass_depth_team)
    pass_qb = _rows(args.pass_depth_qb)
    pass_outcomes = _rows(args.pass_depth_outcomes)
    target_depth = _rows(args.target_depth)
    run_league = _rows(args.run_geometry_league)
    run_team = _rows(args.run_geometry_team)
    run_rusher = _rows(args.run_geometry_rusher)
    run_outcomes = _rows(args.run_geometry_outcomes)

    teams = {}
    for team in sorted(opponents):
        identity = _team_identity(
            team,
            pools[team],
            reality,
            units[team],
            states[team],
            league_neutral_pass_rate=league_neutral_pass_rate,
            situational_pass_rates=situational_pass_rates,
        )
        flow_policy = build_team_game_flow_policy(
            team_id=team,
            league_rows=game_flow_league,
            team_rows=game_flow_team,
            team_neutral_rate=pools[team].neutral_pass_rate,
            league_neutral_rate=league_neutral_pass_rate,
        )
        intent = build_intent_ecology(
            team_id=team,
            pass_league_rows=pass_league,
            pass_team_rows=pass_team,
            pass_qb_rows=pass_qb,
            pass_outcome_rows=pass_outcomes,
            target_depth_rows=target_depth,
            run_league_rows=run_league,
            run_team_rows=run_team,
            run_rusher_rows=run_rusher,
            run_outcome_rows=run_outcomes,
        )
        teams[team] = identity.__class__(**{**identity.__dict__, "game_flow_policy": flow_policy, "intent_ecology": intent})
    defenses = {team: _defensive_unit(units[team]) for team in opponents}
    return teams, defenses, {"pools": pools, "opponents": opponents}


def _monster_transition_rows(
    sampled: list[dict[str, object]],
    *,
    teams: dict[str, object],
    defenses: dict[str, object],
    pools: dict[str, object],
    opponents: dict[str, str],
    replicates: int,
    seed: int,
) -> list[dict[str, object]]:
    team_ids = sorted(teams)
    output: list[dict[str, object]] = []
    for sample_index, historical in enumerate(sampled):
        for replicate in range(replicates):
            offense_id = team_ids[(sample_index * replicates + replicate) % len(team_ids)]
            defense_id = opponents[offense_id]
            draw_seed = seed + sample_index * 1_000_003 + replicate * 97_003
            plan = sample_event_rush_share_plan(
                pools[offense_id], rng=np.random.default_rng(draw_seed + 17)
            )
            offense = _with_event_rush_plan(teams[offense_id], plan)
            margin = int(round(float(historical["score_margin"])))
            state = FootballState(
                possession=offense_id,
                defense=defense_id,
                away_team_id=offense_id,
                home_team_id=defense_id,
                quarter=int(historical["quarter"]),
                seconds_remaining=max(int(historical["seconds_remaining"]), 1),
                yardline_100=float(historical["yardline_from_own"]),
                down=int(historical["down"]),
                distance=float(historical["distance"]),
                away_score=max(margin, 0),
                home_score=max(-margin, 0),
            )
            rng = np.random.default_rng(draw_seed)
            play_kernel_module._FATIGUE_BY_RNG.pop(id(rng), None)
            event = simulate_scrimmage_play(
                state,
                offense,
                1.0,
                rng,
                defense=defenses[defense_id],
            )
            penalty = simulate_penalty(rng, base_rate=0.055)
            converted = False
            touchdown = False
            yards = float(event.yards)
            if penalty is not None and event.play_type in {PlayType.RUN, PlayType.PASS}:
                after = enforce_penalty(state, penalty, elapsed_seconds=event.elapsed_seconds)
                converted = (
                    penalty.side == PenaltySide.DEFENSE
                    and (penalty.automatic_first_down or float(penalty.yards) >= state.distance)
                )
                next_state = ABSORB_CONVERTED if converted else coarse_state_key(after.down, after.distance)
                family = "penalty"
                if penalty.side == PenaltySide.OFFENSE:
                    cause = "offensive_penalty"
                elif converted:
                    cause = "defensive_penalty_first_down"
                else:
                    cause = "defensive_penalty"
            elif event.play_type == PlayType.PUNT or event.play_type == PlayType.FIELD_GOAL:
                next_state = ABSORB_FAILED
                family, cause = _monster_cause(event, False)
            elif event.turnover:
                next_state = ABSORB_FAILED
                family, cause = _monster_cause(event, False)
            elif event.touchdown:
                touchdown = True
                next_state = ABSORB_TOUCHDOWN
                family, cause = _monster_cause(event, False)
            else:
                after = apply_scrimmage_yards(state, event.yards, event.elapsed_seconds)
                converted = event.yards >= state.distance
                if state.down == 4 and not converted:
                    next_state = ABSORB_FAILED
                elif converted:
                    next_state = ABSORB_CONVERTED
                else:
                    next_state = coarse_state_key(after.down, after.distance)
                family, cause = _monster_cause(event, converted)
            play_kernel_module._FATIGUE_BY_RNG.pop(id(rng), None)
            output.append(
                {
                    "source": "monster_micro",
                    "historical_sample_index": sample_index,
                    "replicate": replicate,
                    "offense_team_id": offense_id,
                    "defense_team_id": defense_id,
                    "down": state.down,
                    "distance": state.distance,
                    "yardline_from_own": state.yardline_100,
                    "quarter": state.quarter,
                    "seconds_remaining": state.seconds_remaining,
                    "score_margin": margin,
                    "coarse_state": coarse_state_key(state.down, state.distance),
                    "fine_state": fine_state_key(state.down, state.distance, state.yardline_100),
                    "play_family": family,
                    "cause": cause,
                    "converted": converted,
                    "touchdown": touchdown,
                    "yards": yards,
                    "next_state": next_state,
                }
            )
    return output


def _distribution(rows: list[dict[str, object]], field: str) -> dict[str, float]:
    return normalized(Counter(str(row[field]) for row in rows))


def _cell_summary(
    historical: list[dict[str, object]],
    monster: list[dict[str, object]],
    *,
    key_field: str,
) -> list[dict[str, object]]:
    hist_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    mon_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in historical:
        hist_groups[str(row[key_field])].append(row)
    for row in monster:
        mon_groups[str(row[key_field])].append(row)
    total_hist = max(len(historical), 1)
    output = []
    for key in sorted(set(hist_groups) | set(mon_groups)):
        h = hist_groups.get(key, [])
        m = mon_groups.get(key, [])
        if not h or not m:
            continue
        hist_conversion = sum(bool(row["converted"]) for row in h) / len(h)
        mon_conversion = sum(bool(row["converted"]) for row in m) / len(m)
        hist_success = sum(
            bool(row["converted"]) or bool(row["touchdown"]) for row in h
        ) / len(h)
        mon_success = sum(
            bool(row["converted"]) or bool(row["touchdown"]) for row in m
        ) / len(m)
        next_tvd = total_variation(_distribution(h, "next_state"), _distribution(m, "next_state"))
        cause_tvd = total_variation(_distribution(h, "cause"), _distribution(m, "cause"))
        exposure = len(h) / total_hist
        deficit = hist_success - mon_success
        output.append(
            {
                key_field: key,
                "historical_samples": len(h),
                "monster_samples": len(m),
                "historical_exposure_weight": exposure,
                "historical_conversion_rate": hist_conversion,
                "monster_conversion_rate": mon_conversion,
                "conversion_gap": hist_conversion - mon_conversion,
                "historical_series_success_rate": hist_success,
                "monster_series_success_rate": mon_success,
                "series_success_gap": deficit,
                "next_state_total_variation": next_tvd,
                "cause_total_variation": cause_tvd,
                "priority_score": exposure * max(deficit, 0.0) * (1.0 + next_tvd + cause_tvd),
            }
        )
    return sorted(output, key=lambda row: float(row["priority_score"]), reverse=True)


def _cause_decomposition(
    historical: list[dict[str, object]], monster: list[dict[str, object]]
) -> list[dict[str, object]]:
    hist_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    mon_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in historical:
        hist_groups[str(row["coarse_state"])].append(row)
    for row in monster:
        mon_groups[str(row["coarse_state"])].append(row)
    output: list[dict[str, object]] = []
    for state in sorted(set(hist_groups) & set(mon_groups)):
        h = hist_groups[state]
        m = mon_groups[state]
        hd = _distribution(h, "cause")
        md = _distribution(m, "cause")
        for cause in sorted(set(hd) | set(md)):
            output.append(
                {
                    "coarse_state": state,
                    "cause": cause,
                    "historical_rate": hd.get(cause, 0.0),
                    "monster_rate": md.get(cause, 0.0),
                    "monster_minus_historical": md.get(cause, 0.0) - hd.get(cause, 0.0),
                    "historical_samples": len(h),
                    "monster_samples": len(m),
                }
            )
    return output


def _counterfactuals(
    historical: list[dict[str, object]], monster: list[dict[str, object]]
) -> tuple[list[dict[str, object]], dict[str, float]]:
    hist_transitions = transition_probabilities(historical)
    mon_transitions = transition_probabilities(monster)
    start_counts = Counter(
        str(row["coarse_state"]) for row in historical if int(row["down"]) == 1
    )
    start_distribution = normalized(start_counts)
    hist_survival, _ = solve_series_survival(hist_transitions, start_distribution)
    mon_survival, _ = solve_series_survival(mon_transitions, start_distribution)
    gap = hist_survival - mon_survival

    candidates: list[tuple[str, list[str]]] = []
    for down in range(1, 5):
        states = sorted(state for state in set(hist_transitions) | set(mon_transitions) if state.startswith(f"d{down}:"))
        candidates.append((f"all_down_{down}", states))
    hist_counts = Counter(str(row["coarse_state"]) for row in historical)
    mon_counts = Counter(str(row["coarse_state"]) for row in monster)
    for state in sorted(set(hist_transitions) & set(mon_transitions)):
        if hist_counts[state] >= 50 and mon_counts[state] >= 50:
            candidates.append((state, [state]))

    output = []
    for label, states in candidates:
        hybrid = replace_transition_family(mon_transitions, hist_transitions, states)
        survival, _ = solve_series_survival(hybrid, start_distribution)
        delta = survival - mon_survival
        output.append(
            {
                "replacement_family": label,
                "states_replaced": len(states),
                "monster_baseline_series_survival": mon_survival,
                "hybrid_series_survival": survival,
                "historical_series_survival": hist_survival,
                "absolute_recovery": delta,
                "recovery_percentage_points": delta * 100.0,
                "fraction_of_gap_recovered": delta / gap if gap > 1e-12 else 0.0,
            }
        )
    output.sort(key=lambda row: float(row["absolute_recovery"]), reverse=True)
    return output, {
        "historical_series_survival": hist_survival,
        "monster_series_survival": mon_survival,
        "series_survival_gap": gap,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/monster"))
    parser.add_argument("--game-flow-league", type=Path, required=True)
    parser.add_argument("--game-flow-team", type=Path, required=True)
    parser.add_argument("--pass-depth-league", type=Path, required=True)
    parser.add_argument("--pass-depth-team", type=Path, required=True)
    parser.add_argument("--pass-depth-qb", type=Path, required=True)
    parser.add_argument("--pass-depth-outcomes", type=Path, required=True)
    parser.add_argument("--target-depth", type=Path, required=True)
    parser.add_argument("--run-geometry-league", type=Path, required=True)
    parser.add_argument("--run-geometry-team", type=Path, required=True)
    parser.add_argument("--run-geometry-rusher", type=Path, required=True)
    parser.add_argument("--run-geometry-outcomes", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--personnel", type=Path, required=True)
    parser.add_argument("--player-usage", type=Path, required=True)
    parser.add_argument("--situation-context", type=Path, required=True)
    parser.add_argument("--max-historical-states", type=int, default=12000)
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026091202)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    historical = _historical_transition_rows(args.season, args.cache_dir)
    sampled = _stratified_sample(
        historical,
        maximum=args.max_historical_states,
        seed=args.seed,
    )
    teams, defenses, context = _build_current_teams(args)
    monster = _monster_transition_rows(
        sampled,
        teams=teams,
        defenses=defenses,
        pools=context["pools"],
        opponents=context["opponents"],
        replicates=args.replicates,
        seed=args.seed + 10_000_019,
    )

    coarse = _cell_summary(historical, monster, key_field="coarse_state")
    fine = _cell_summary(historical, monster, key_field="fine_state")
    causes = _cause_decomposition(historical, monster)
    counterfactuals, survival = _counterfactuals(historical, monster)

    args.out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(historical).write_parquet(args.out / "historical_transition_rows.parquet")
    pl.DataFrame(monster).write_parquet(args.out / "monster_transition_rows.parquet")
    pl.DataFrame(coarse).write_csv(args.out / "coarse_state_transition_cells.csv")
    pl.DataFrame(fine).write_csv(args.out / "fine_state_transition_cells.csv")
    pl.DataFrame(causes).write_csv(args.out / "cause_decomposition.csv")
    pl.DataFrame(counterfactuals).write_csv(args.out / "counterfactual_replacement.csv")

    top_cells = coarse[:10]
    top_counterfactuals = counterfactuals[:10]
    ledger = {
        "experiment": "MON-LEDGER-002C.1-STATE-TRANSITION-MIRROR",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "vlourish_runtime": {
            "center": "Locate the first causal state transition where Monster stops preserving NFL series-continuation reality.",
            "boundary": "One-snap and one-series transition mechanics only. No score tuning, DFS inputs, full-game calibration or production promotion authority.",
            "context": {
                "historical_truth": f"{args.season} regular-season nflverse play-by-play",
                "monster_truth": "current Week 1 Stage 3 identities and mechanisms evaluated on matched historical football states",
                "current_problem": "individual pass/run anatomy is closer to NFL reality, while series conversion and red-zone arrival remain too low",
            },
            "unknown": [
                "Which down-distance transitions create the largest series-survival loss?",
                "Which result classes explain those transition gaps?",
                "How much of the survival gap can each transition family recover if replaced by NFL-observed behavior?",
                "Are penalties, scrambles, sticks-aware completions or ordinary run/pass gains the highest-value mechanism to repair next?",
            ],
            "direction": "Contract the experiment boundary from full games to matched state transitions, then rank repairs by counterfactual recovery rather than intuition.",
            "practice": [
                "Build an NFL state-transition mirror.",
                "Evaluate Monster on the same state distribution without changing football coefficients.",
                "Decompose divergence by outcome cause.",
                "Counterfactually transplant one NFL transition family at a time.",
                "Promote only the highest-evidence mechanism to a repair experiment.",
            ],
            "evidence": {
                "historical_transition_rows": len(historical),
                "monster_micro_transition_rows": len(monster),
                "coarse_cells": len(coarse),
                "fine_cells": len(fine),
                **survival,
                "top_priority_cells": top_cells,
                "top_counterfactual_replacements": top_counterfactuals,
            },
            "reflection": "PENDING_EVIDENCE_REVIEW",
            "expansion": "After evidence review, open MON-LEDGER-002C.2 only for the mechanism families that explain material survival recovery.",
        },
        "authority_map": {
            "nflverse_2025": "authoritative for observed historical state-transition frequencies",
            "monster_micro_sim": "authoritative for what current Monster mechanisms do on matched football states",
            "audit": "diagnostic only; may rank hypotheses but may not change football behavior",
            "dfs_market": "prohibited upstream of football-world generation",
        },
        "governance": {
            "stage": "LAB",
            "market_blind": True,
            "football_coefficients_changed": False,
            "full_game_simulation_required": False,
            "promotion_allowed": False,
        },
    }
    (args.out / "methodology_ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")
    manifest = {
        "artifact": "Monster Vlourish State Transition Mirror",
        "experiment": ledger["experiment"],
        "season": args.season,
        "current_slate_teams": len(teams),
        "historical_rows": len(historical),
        "historical_states_sampled_for_monster": len(sampled),
        "monster_rows": len(monster),
        "replicates": args.replicates,
        "seed": args.seed,
        "market_blind": True,
        "behavior_changed_by_audit": False,
        "production_authority": False,
        "series_survival": survival,
        "files": {
            "historical": "historical_transition_rows.parquet",
            "monster": "monster_transition_rows.parquet",
            "coarse_cells": "coarse_state_transition_cells.csv",
            "fine_cells": "fine_state_transition_cells.csv",
            "causes": "cause_decomposition.csv",
            "counterfactuals": "counterfactual_replacement.csv",
            "ledger": "methodology_ledger.json",
        },
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
