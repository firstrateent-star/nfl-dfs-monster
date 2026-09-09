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
    """Solve one FanDuel Classic world exactly as a binary mixed-integer program.

    Constraints encode the contest itself rather than projection heuristics: nine players, one QB,
    one D/ST, at least two RB, three WR and one TE, with the seventh skill slot becoming the FLEX.
    The football-world score vector is the only objective signal.
    """
    required = {"position", "salary", "fanduel_id"}
    missing = required - set(pool.columns)
    if missing:
        raise ValueError(f"Optimizer pool missing columns: {sorted(missing)}")
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (pool.height,):
        raise ValueError("World score vector must align exactly with optimizer pool")

    salary = pool["salary"].cast(pl.Int64).to_numpy().astype(np.float64)
    positions = pool["position"].cast(pl.String).str.to_uppercase().to_numpy()
    qb = (positions == "QB").astype(np.float64)
    rb = (positions == "RB").astype(np.float64)
    wr = (positions == "WR").astype(np.float64)
    te = (positions == "TE").astype(np.float64)
    defense = np.isin(positions, ["D", "DEF", "DST"]).astype(np.float64)
    if min(qb.sum(), rb.sum(), wr.sum(), te.sum(), defense.sum()) == 0:
        raise ValueError("Optimizer pool lacks a required FanDuel position")

    matrix = np.vstack([np.ones(pool.height), salary, qb, defense, rb, wr, te])
    lower = np.array([9, -np.inf, 1, 1, 2, 3, 1], dtype=np.float64)
    upper = np.array([9, salary_cap, 1, 1, np.inf, np.inf, np.inf], dtype=np.float64)
    constraints = LinearConstraint(matrix, lower, upper)
    result = milp(
        c=-values,
        integrality=np.ones(pool.height, dtype=np.int8),
        bounds=Bounds(np.zeros(pool.height), np.ones(pool.height)),
        constraints=constraints,
        options={"presolve": True},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"No legal FanDuel optimum found: {result.message}")
    indices = tuple(int(i) for i in np.flatnonzero(result.x > 0.5))
    total_salary = int(salary[list(indices)].sum())
    score = float(values[list(indices)].sum())
    lineup = OptimalLineup(indices=indices, score=score, salary=total_salary)
    audit = audit_fanduel_lineup(pool[list(indices)], salary_cap=salary_cap)
    if not audit.legal:
        raise RuntimeError(f"MILP optimizer produced illegal lineup: {audit.reason}")
    return lineup
