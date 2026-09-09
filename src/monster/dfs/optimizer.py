from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import polars as pl

from monster.dfs.lineup import SALARY_CAP, audit_fanduel_lineup


@dataclass(frozen=True)
class OptimalLineup:
    indices: tuple[int, ...]
    score: float
    salary: int


def _position_indices(pool: pl.DataFrame, position: str) -> np.ndarray:
    return np.flatnonzero(pool["position"].cast(pl.String).str.to_uppercase().to_numpy() == position)


def _validate(pool: pl.DataFrame, scores: np.ndarray):
    required = {"position", "salary", "fanduel_id"}
    missing = required - set(pool.columns)
    if missing:
        raise ValueError(f"Optimizer pool missing columns: {sorted(missing)}")
    values = np.asarray(scores, dtype=np.float32)
    if values.shape != (pool.height,):
        raise ValueError("World score vector must align exactly with optimizer pool")
    salary = pool["salary"].cast(pl.Int64).to_numpy()
    qb = _position_indices(pool, "QB")
    rb = _position_indices(pool, "RB")
    wr = _position_indices(pool, "WR")
    te = _position_indices(pool, "TE")
    defense = np.flatnonzero(
        np.isin(pool["position"].cast(pl.String).str.to_uppercase().to_numpy(), ["D", "DEF", "DST"])
    )
    if min(len(qb), len(rb), len(wr), len(te), len(defense)) == 0:
        raise ValueError("Optimizer pool lacks a required FanDuel position")
    return values, salary, qb, rb, wr, te, defense


def _finish(pool: pl.DataFrame, best: OptimalLineup | None, salary_cap: int) -> OptimalLineup:
    if best is None:
        raise RuntimeError("No legal FanDuel lineup exists under the salary cap")
    audit = audit_fanduel_lineup(pool[list(best.indices)], salary_cap=salary_cap)
    if not audit.legal:
        raise RuntimeError(f"Optimizer produced illegal lineup: {audit.reason}")
    return best


def solve_world_optimal(
    pool: pl.DataFrame,
    scores: np.ndarray,
    *,
    salary_cap: int = SALARY_CAP,
) -> OptimalLineup:
    """Reference exact solver. Slow by design; retained as the correctness oracle."""
    values, salary, qb, rb, wr, te, defense = _validate(pool, scores)
    best: OptimalLineup | None = None
    shapes = ((3, 3, 1), (2, 4, 1), (2, 3, 2))
    for q in qb:
        for d in defense:
            base_salary = int(salary[q] + salary[d])
            if base_salary >= salary_cap:
                continue
            for rb_count, wr_count, te_count in shapes:
                for rbs in combinations(rb.tolist(), rb_count):
                    rb_salary = sum(int(salary[i]) for i in rbs)
                    if base_salary + rb_salary >= salary_cap:
                        continue
                    for tes in combinations(te.tolist(), te_count):
                        partial_salary = base_salary + rb_salary + sum(int(salary[i]) for i in tes)
                        if partial_salary >= salary_cap:
                            continue
                        for wrs in combinations(wr.tolist(), wr_count):
                            total_salary = partial_salary + sum(int(salary[i]) for i in wrs)
                            if total_salary > salary_cap:
                                continue
                            indices = (int(q), *map(int, rbs), *map(int, wrs), *map(int, tes), int(d))
                            score = float(values[list(indices)].sum())
                            if best is None or score > best.score or (
                                score == best.score and total_salary < best.salary
                            ):
                                best = OptimalLineup(tuple(indices), score, total_salary)
    return _finish(pool, best, salary_cap)


def _combination_frontier(
    indices: np.ndarray,
    count: int,
    salary: np.ndarray,
    values: np.ndarray,
    salary_cap: int,
) -> list[tuple[int, float, tuple[int, ...]]]:
    """Return nondominated exact-k combinations ordered by increasing salary.

    A combination is discarded only when another combination costs no more and scores at least as
    much in this same simulated world. This is an exact dominance rule, not projection pruning.
    """
    best_at_salary: dict[int, tuple[float, tuple[int, ...]]] = {}
    for combo in combinations(indices.tolist(), count):
        cost = sum(int(salary[i]) for i in combo)
        if cost > salary_cap:
            continue
        score = float(values[list(combo)].sum())
        prior = best_at_salary.get(cost)
        if prior is None or score > prior[0] or (score == prior[0] and combo < prior[1]):
            best_at_salary[cost] = (score, tuple(map(int, combo)))
    frontier: list[tuple[int, float, tuple[int, ...]]] = []
    best_score = -np.inf
    for cost, (score, combo) in sorted(best_at_salary.items()):
        if score > best_score:
            frontier.append((cost, score, combo))
            best_score = score
    return frontier


def _merge_frontiers(
    left: list[tuple[int, float, tuple[int, ...]]],
    right: list[tuple[int, float, tuple[int, ...]]],
    salary_cap: int,
) -> list[tuple[int, float, tuple[int, ...]]]:
    best_at_salary: dict[int, tuple[float, tuple[int, ...]]] = {}
    for left_cost, left_score, left_indices in left:
        for right_cost, right_score, right_indices in right:
            cost = left_cost + right_cost
            if cost > salary_cap:
                continue
            score = left_score + right_score
            indices = left_indices + right_indices
            prior = best_at_salary.get(cost)
            if prior is None or score > prior[0] or (score == prior[0] and indices < prior[1]):
                best_at_salary[cost] = (score, indices)
    frontier: list[tuple[int, float, tuple[int, ...]]] = []
    best_score = -np.inf
    for cost, (score, indices) in sorted(best_at_salary.items()):
        if score > best_score:
            frontier.append((cost, score, indices))
            best_score = score
    return frontier


def solve_world_optimal_frontier(
    pool: pl.DataFrame,
    scores: np.ndarray,
    *,
    salary_cap: int = SALARY_CAP,
) -> OptimalLineup:
    """Exact solver using salary/score Pareto frontiers instead of the full lineup cross-product."""
    values, salary, qb, rb, wr, te, defense = _validate(pool, scores)
    position_cache: dict[tuple[str, int], list[tuple[int, float, tuple[int, ...]]]] = {}
    position_indices = {"RB": rb, "WR": wr, "TE": te}

    def frontier(position: str, count: int):
        key = (position, count)
        if key not in position_cache:
            position_cache[key] = _combination_frontier(
                position_indices[position], count, salary, values, salary_cap
            )
        return position_cache[key]

    best: OptimalLineup | None = None
    for rb_count, wr_count, te_count in ((3, 3, 1), (2, 4, 1), (2, 3, 2)):
        skill = _merge_frontiers(frontier("RB", rb_count), frontier("WR", wr_count), salary_cap)
        skill = _merge_frontiers(skill, frontier("TE", te_count), salary_cap)
        for q in qb:
            for d in defense:
                base_cost = int(salary[q] + salary[d])
                if base_cost > salary_cap:
                    continue
                base_score = float(values[q] + values[d])
                remaining = salary_cap - base_cost
                feasible = [item for item in skill if item[0] <= remaining]
                if not feasible:
                    continue
                skill_cost, skill_score, skill_indices = feasible[-1]
                total_cost = base_cost + skill_cost
                total_score = base_score + skill_score
                indices = (int(q), *skill_indices, int(d))
                if best is None or total_score > best.score or (
                    total_score == best.score and total_cost < best.salary
                ):
                    best = OptimalLineup(indices, total_score, total_cost)
    return _finish(pool, best, salary_cap)
