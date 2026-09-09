import numpy as np
import polars as pl

from monster.dfs.lineup import audit_fanduel_lineup
from monster.dfs.optimizer import solve_world_optimal
from monster.dfs.optimizer_milp import solve_world_optimal_milp


def _pool() -> pl.DataFrame:
    rows = []
    counter = 0
    salaries = {
        "QB": [6200, 7600],
        "RB": [5400, 6100, 6800, 7200],
        "WR": [5000, 5600, 5900, 6500, 7100],
        "TE": [4900, 5700, 6300],
        "D": [4000, 4600],
    }
    for position, values in salaries.items():
        for idx, salary in enumerate(values):
            counter += 1
            rows.append(
                {
                    "position": position,
                    "player": f"{position}{idx}",
                    "salary": salary,
                    "fanduel_id": f"fd-{counter}",
                }
            )
    return pl.DataFrame(rows)


def test_milp_matches_bruteforce_optimal_score_on_random_worlds() -> None:
    pool = _pool()
    rng = np.random.default_rng(2026090902)
    for _ in range(40):
        scores = rng.normal(12.0, 9.0, pool.height).astype(np.float32)
        reference = solve_world_optimal(pool, scores, salary_cap=54000)
        scalable = solve_world_optimal_milp(pool, scores, salary_cap=54000)
        assert np.isclose(scalable.score, reference.score, atol=1e-5)
        assert scalable.salary <= 54000
        assert audit_fanduel_lineup(pool[list(scalable.indices)], salary_cap=54000).legal


def test_milp_preserves_rare_breakout_and_all_flex_shapes() -> None:
    pool = _pool()
    for position in ("RB", "WR", "TE"):
        scores = np.ones(pool.height, dtype=np.float32)
        for idx, value in enumerate(pool["position"].to_list()):
            if value == position:
                scores[idx] = 25.0
        result = solve_world_optimal_milp(pool, scores)
        lineup = pool[list(result.indices)]
        expected = {"RB": 3, "WR": 4, "TE": 2}[position]
        assert lineup.filter(pl.col("position") == position).height == expected

    scores = np.ones(pool.height, dtype=np.float32)
    rare = pool["player"].to_list().index("WR4")
    scores[rare] = 100.0
    assert rare in solve_world_optimal_milp(pool, scores).indices
