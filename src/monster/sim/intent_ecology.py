from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from monster.feature_compile.play_intent import (
    PASS_DEPTH_CATEGORIES,
    RUN_GEOMETRY_CATEGORIES,
)
from monster.sim.categorical_policy import (
    ContextualCategoricalPolicy,
    build_contextual_categorical_policy,
    sample_category,
)
from monster.sim.game_flow import GameFlowState


@dataclass(frozen=True)
class PassDepthOutcome:
    category: str
    attempts: int
    completion_rate: float
    interception_rate: float
    touchdown_rate: float
    air_yards_mean: float
    air_yards_sd: float
    yac_mean_completed: float
    yac_sd_completed: float
    negative_completion_rate: float
    zero_completion_rate: float


@dataclass(frozen=True)
class RunGeometryOutcome:
    category: str
    attempts: int
    yards_mean: float
    yards_sd: float
    negative_rate: float
    zero_rate: float
    loss_2_plus_rate: float
    loss_5_plus_rate: float
    explosive_10_rate: float
    explosive_15_rate: float
    explosive_20_rate: float
    touchdown_rate: float
    fumble_lost_rate: float
    yards_p10: float
    yards_p50: float
    yards_p90: float
    yards_p99: float


@dataclass(frozen=True)
class IntentEcology:
    pass_depth: ContextualCategoricalPolicy
    run_geometry: ContextualCategoricalPolicy
    pass_outcomes: Mapping[str, PassDepthOutcome]
    run_outcomes: Mapping[str, RunGeometryOutcome]
    target_depth_attempts: Mapping[tuple[str, str], int]
    rusher_geometry_attempts: Mapping[tuple[str, str], int]


def _float(row: Mapping[str, object], name: str, default: float = 0.0) -> float:
    value = row.get(name)
    return default if value is None else float(value)


def build_pass_outcomes(rows: Iterable[Mapping[str, object]]) -> dict[str, PassDepthOutcome]:
    out = {}
    for row in rows:
        category = str(row["category"])
        out[category] = PassDepthOutcome(
            category=category,
            attempts=int(row["attempts"]),
            completion_rate=_float(row, "completion_rate"),
            interception_rate=_float(row, "interception_rate"),
            touchdown_rate=_float(row, "touchdown_rate"),
            air_yards_mean=_float(row, "air_yards_mean"),
            air_yards_sd=max(_float(row, "air_yards_sd", 1.0), 0.35),
            yac_mean_completed=max(_float(row, "yac_mean_completed"), 0.0),
            yac_sd_completed=max(_float(row, "yac_sd_completed", 1.0), 0.35),
            negative_completion_rate=_float(row, "negative_completion_rate"),
            zero_completion_rate=_float(row, "zero_completion_rate"),
        )
    return out


def build_run_outcomes(rows: Iterable[Mapping[str, object]]) -> dict[str, RunGeometryOutcome]:
    out = {}
    for row in rows:
        category = str(row["category"])
        out[category] = RunGeometryOutcome(
            category=category,
            attempts=int(row["attempts"]),
            yards_mean=_float(row, "yards_mean"),
            yards_sd=max(_float(row, "yards_sd", 1.0), 0.35),
            negative_rate=_float(row, "negative_rate"),
            zero_rate=_float(row, "zero_rate"),
            loss_2_plus_rate=_float(row, "loss_2_plus_rate"),
            loss_5_plus_rate=_float(row, "loss_5_plus_rate"),
            explosive_10_rate=_float(row, "explosive_10_rate"),
            explosive_15_rate=_float(row, "explosive_15_rate"),
            explosive_20_rate=_float(row, "explosive_20_rate"),
            touchdown_rate=_float(row, "touchdown_rate"),
            fumble_lost_rate=_float(row, "fumble_lost_rate"),
            yards_p10=_float(row, "yards_p10"),
            yards_p50=_float(row, "yards_p50"),
            yards_p90=_float(row, "yards_p90"),
            yards_p99=_float(row, "yards_p99"),
        )
    return out


def build_intent_ecology(
    *,
    team_id: str,
    pass_league_rows: Iterable[Mapping[str, object]],
    pass_team_rows: Iterable[Mapping[str, object]],
    pass_qb_rows: Iterable[Mapping[str, object]],
    pass_outcome_rows: Iterable[Mapping[str, object]],
    target_depth_rows: Iterable[Mapping[str, object]],
    run_league_rows: Iterable[Mapping[str, object]],
    run_team_rows: Iterable[Mapping[str, object]],
    run_rusher_rows: Iterable[Mapping[str, object]],
    run_outcome_rows: Iterable[Mapping[str, object]],
    league_shrinkage_samples: float = 100.0,
    team_shrinkage_samples: float = 100.0,
    actor_shrinkage_samples: float = 120.0,
) -> IntentEcology:
    pass_qb_list = list(pass_qb_rows)
    run_rusher_list = list(run_rusher_rows)
    pass_policy = build_contextual_categorical_policy(
        categories=PASS_DEPTH_CATEGORIES,
        league_rows=pass_league_rows,
        team_rows=pass_team_rows,
        actor_rows=pass_qb_list,
        team_id=team_id,
        league_shrinkage_samples=league_shrinkage_samples,
        team_shrinkage_samples=team_shrinkage_samples,
        actor_shrinkage_samples=actor_shrinkage_samples,
    )
    run_policy = build_contextual_categorical_policy(
        categories=RUN_GEOMETRY_CATEGORIES,
        league_rows=run_league_rows,
        team_rows=run_team_rows,
        team_id=team_id,
        league_shrinkage_samples=league_shrinkage_samples,
        team_shrinkage_samples=team_shrinkage_samples,
        actor_shrinkage_samples=actor_shrinkage_samples,
    )
    target_attempts = {
        (str(row["player_id"]), str(row["category"])): int(row["attempts"])
        for row in target_depth_rows
        if str(row.get("team_id")) == team_id
    }
    rusher_attempts = {
        (str(row["actor_id"]), str(row["category"])): int(row["attempts"])
        for row in run_rusher_list
    }
    return IntentEcology(
        pass_depth=pass_policy,
        run_geometry=run_policy,
        pass_outcomes=build_pass_outcomes(pass_outcome_rows),
        run_outcomes=build_run_outcomes(run_outcome_rows),
        target_depth_attempts=target_attempts,
        rusher_geometry_attempts=rusher_attempts,
    )


def feasible_pass_probabilities(
    ecology: IntentEcology,
    flow: GameFlowState,
    *,
    quarterback_id: str,
) -> np.ndarray:
    """Return depth intent probabilities after physical field geometry masks impossibilities."""

    probs = ecology.pass_depth.probabilities_for(flow, actor_id=quarterback_id)
    max_target_depth = flow.yards_to_goal + 9.0
    lower_bounds = {
        "behind_los": -10.0,
        "short_0_5": 0.0,
        "short_6_9": 6.0,
        "intermediate_10_19": 10.0,
        "deep_20_39": 20.0,
        "bomb_40_plus": 40.0,
    }
    mask = np.asarray(
        [
            1.0 if lower_bounds[category] <= max_target_depth else 0.0
            for category in ecology.pass_depth.categories
        ],
        dtype=float,
    )
    probs = probs * mask
    total = float(probs.sum())
    if total <= 0.0:
        fallback = np.zeros_like(probs)
        fallback[ecology.pass_depth.categories.index("short_0_5")] = 1.0
        return fallback
    return probs / total


def sample_pass_depth_intent(
    ecology: IntentEcology,
    flow: GameFlowState,
    *,
    quarterback_id: str,
    rng: np.random.Generator,
) -> str:
    probs = feasible_pass_probabilities(ecology, flow, quarterback_id=quarterback_id)
    return sample_category(ecology.pass_depth.categories, probs, rng)


def sample_air_yards(
    category: str,
    outcome: PassDepthOutcome,
    *,
    yards_to_goal: float,
    rng: np.random.Generator,
) -> float:
    bounds = {
        "behind_los": (-12.0, -0.01),
        "short_0_5": (0.0, 5.999),
        "short_6_9": (6.0, 9.999),
        "intermediate_10_19": (10.0, 19.999),
        "deep_20_39": (20.0, 39.999),
        "bomb_40_plus": (40.0, 70.0),
    }
    low, high = bounds[category]
    high = min(high, yards_to_goal + 9.0)
    high = max(high, low)
    draw = float(rng.normal(outcome.air_yards_mean, outcome.air_yards_sd))
    return float(np.clip(draw, low, high))


def _compatibility_weights(
    players: Sequence[object],
    *,
    category: str,
    attempts: Mapping[tuple[str, str], int],
    shrinkage_samples: float,
) -> np.ndarray:
    base = np.asarray(
        [max(float(player.usage_weight), 0.001) for player in players],
        dtype=float,
    )
    base /= base.sum()
    observed = np.asarray(
        [
            max(int(attempts.get((str(player.player_id), category), 0)), 0)
            for player in players
        ],
        dtype=float,
    )
    total = float(observed.sum())
    if total <= 0.0:
        return base
    observed /= total
    authority = total / (total + shrinkage_samples)
    weights = (1.0 - authority) * base + authority * observed
    return weights / weights.sum()


def choose_target_for_depth(
    players: Sequence[object],
    ecology: IntentEcology,
    category: str,
    rng: np.random.Generator,
    *,
    shrinkage_samples: float = 45.0,
):
    if not players:
        raise ValueError("receiver pool cannot be empty")
    weights = _compatibility_weights(
        players,
        category=category,
        attempts=ecology.target_depth_attempts,
        shrinkage_samples=shrinkage_samples,
    )
    return players[int(rng.choice(len(players), p=weights))]


def sample_run_geometry_intent(
    ecology: IntentEcology,
    flow: GameFlowState,
    *,
    rng: np.random.Generator,
) -> str:
    return ecology.run_geometry.sample(flow, rng)


def choose_rusher_for_geometry(
    players: Sequence[object],
    ecology: IntentEcology,
    category: str,
    rng: np.random.Generator,
    *,
    shrinkage_samples: float = 60.0,
):
    if not players:
        raise ValueError("rusher pool cannot be empty")
    weights = _compatibility_weights(
        players,
        category=category,
        attempts=ecology.rusher_geometry_attempts,
        shrinkage_samples=shrinkage_samples,
    )
    return players[int(rng.choice(len(players), p=weights))]
