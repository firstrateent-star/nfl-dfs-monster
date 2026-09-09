from __future__ import annotations

import numpy as np
import polars as pl
from scipy.optimize import Bounds, LinearConstraint, milp

from monster.dfs.lineup import SALARY_CAP, audit_fanduel_lineup
from monster.dfs.optimizer import OptimalLineup


def solve_world_optimal_milp(
    pool: pl.DataFrame,
    scores: np.ndarray,
    *,
    salary_cap: int = SALARY_CAP,
) -> OptimalLineup:
    """Solve one FanDuel world exactly as a binary integer program.

    The objective is only the supplied same-world Monster FanDuel score. Salary is a downstream
    legality constraint; no projection, ownership, market, or mean-score pruning enters the solve.
    """
    required = {"position", "salary", "fanduel_id"}
    missing = required - set(pool.columns)
    if missing:
        raise ValueError(f"Optimizer pool missing columns: {sorted(missing)}")
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (pool.height,):
        raise ValueError("World score vector must align exactly with optimizer pool")
    if not np.isfinite(values).all():
        raise ValueError("World score vector contains non-finite values")

    positions = pool["position"].cast(pl.String).str.to_uppercase().to_numpy()
    salary = pool["salary"].cast(pl.Int64).to_numpy().astype(np.float64)
    is_qb = positions == "QB"
    is_rb = positions == "RB"
    is_wr = positions == "WR"
    is_te = positions == "TE"
    is_d = np.isin(positions, ["D", "DEF", "DST"])
    if min(is_qb.sum(), is_rb.sum(), is_wr.sum(), is_te.sum(), is_d.sum()) == 0:
        raise ValueError("Optimizer pool lacks a required FanDuel position")

    matrix = np.vstack(
        [
            np.ones(pool.height),
            is_qb.astype(float),
            is_d.astype(float),
            is_rb.astype(float),
            is_wr.astype(float),
            is_te.astype(float),
            salary,
        ]
    )
    lower = np.array([9, 1, 1, 2, 3, 1, -np.inf], dtype=float)
    upper = np.array([9, 1, 1, np.inf, np.inf, np.inf, salary_cap], dtype=float)
    result = milp(
        c=-values,
        integrality=np.ones(pool.height, dtype=np.int8),
        bounds=Bounds(np.zeros(pool.height), np.ones(pool.height)),
        constraints=LinearConstraint(matrix, lower, upper),
        options={"presolve": True},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"No exact FanDuel optimum found: {result.message}")
    indices = tuple(map(int, np.flatnonzero(result.x > 0.5)))
    lineup = pool[list(indices)]
    audit = audit_fanduel_lineup(lineup, salary_cap=salary_cap)
    if not audit.legal:
        raise RuntimeError(f"MILP optimizer produced illegal lineup: {audit.reason}")
    total_salary = int(salary[list(indices)].sum())
    score = float(values[list(indices)].sum())
    return OptimalLineup(indices=indices, score=score, salary=total_salary)
