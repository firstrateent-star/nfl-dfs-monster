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


def solve_world_optimal(
    pool: pl.DataFrame,
    scores: np.ndarray,
    *,
    salary_cap: int = SALARY_CAP,
) -> OptimalLineup:
    """Solve one FanDuel Classic world exactly from that world's correlated scores.

    Enumeration is position-structured rather than projection-pruned: every legal QB/RB/WR/TE/DEF
    construction under the salary cap remains eligible. This is deliberately independent of mean
    projection so rare simulated Sundays can select players that only matter in those worlds.
    """
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
    if best is None:
        raise RuntimeError("No legal FanDuel lineup exists under the salary cap")
    audit = audit_fanduel_lineup(pool[list(best.indices)], salary_cap=salary_cap)
    if not audit.legal:
        raise RuntimeError(f"Optimizer produced illegal lineup: {audit.reason}")
    return best
