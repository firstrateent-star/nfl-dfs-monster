import numpy as np
import polars as pl

from monster.dfs.lineup import audit_fanduel_lineup
from monster.dfs.optimizer import solve_world_optimal, solve_world_optimal_frontier


def _pool() -> pl.DataFrame:
    rows = []
    counter = 0
    for position, count in (("QB", 2), ("RB", 4), ("WR", 5), ("TE", 3), ("D", 2)):
        for idx in range(count):
            counter += 1
            rows.append(
                {
                    "position": position,
                    "player": f"{position}{idx}",
                    "salary": 5000 + 100 * idx,
                    "fanduel_id": f"fd-{counter}",
                }
            )
    return pl.DataFrame(rows)


def test_optimizer_returns_legal_nine_player_lineup() -> None:
    pool = _pool()
    scores = np.arange(pool.height, dtype=np.float32)
    result = solve_world_optimal(pool, scores)
    lineup = pool[list(result.indices)]
    audit = audit_fanduel_lineup(lineup)
    assert audit.legal
    assert len(result.indices) == 9
    assert result.salary <= 60000
    assert result.score == float(scores[list(result.indices)].sum())


def test_optimizer_respects_salary_cap_over_raw_score() -> None:
    pool = _pool().with_columns(pl.lit(6500).alias("salary"))
    pool = pool.with_columns(
        pl.when(pl.col("player") == "QB1")
        .then(15000)
        .otherwise(4000)
        .alias("salary")
    )
    scores = np.ones(pool.height, dtype=np.float32)
    scores[pool["player"].to_list().index("QB1")] = 100.0
    result = solve_world_optimal(pool, scores, salary_cap=40000)
    chosen = pool[list(result.indices)]["player"].to_list()
    assert "QB1" not in chosen
    assert result.salary <= 40000
    assert audit_fanduel_lineup(pool[list(result.indices)], salary_cap=40000).legal


def test_optimizer_custom_cap_audit_cannot_fall_back_to_default() -> None:
    pool = _pool().with_columns(pl.lit(6000).alias("salary"))
    scores = np.arange(pool.height, dtype=np.float32)
    result = solve_world_optimal(pool, scores, salary_cap=54000)
    assert result.salary <= 54000
    assert audit_fanduel_lineup(pool[list(result.indices)], salary_cap=54000).legal


def test_optimizer_can_select_each_flex_shape() -> None:
    pool = _pool()
    for boosted_position, expected_count in (("RB", 3), ("WR", 4), ("TE", 2)):
        scores = np.ones(pool.height, dtype=np.float32)
        for idx, position in enumerate(pool["position"].to_list()):
            if position == boosted_position:
                scores[idx] = 20.0
        result = solve_world_optimal(pool, scores)
        lineup = pool[list(result.indices)]
        assert lineup.filter(pl.col("position") == boosted_position).height == expected_count
        assert audit_fanduel_lineup(lineup).legal


def test_frontier_solver_matches_reference_oracle_on_random_worlds() -> None:
    pool = _pool().with_columns(
        pl.Series(
            "salary",
            [6200, 7600, 5400, 6100, 6800, 7200, 5000, 5600, 5900, 6500, 7100, 4900, 5700, 6300, 4000, 4600],
        )
    )
    rng = np.random.default_rng(20260909)
    for _ in range(30):
        scores = rng.normal(12.0, 8.0, pool.height).astype(np.float32)
        reference = solve_world_optimal(pool, scores, salary_cap=54000)
        frontier = solve_world_optimal_frontier(pool, scores, salary_cap=54000)
        assert np.isclose(frontier.score, reference.score, atol=1e-5)
        assert frontier.salary == reference.salary
        assert audit_fanduel_lineup(pool[list(frontier.indices)], salary_cap=54000).legal


def test_frontier_solver_preserves_rare_world_breakout() -> None:
    pool = _pool()
    scores = np.ones(pool.height, dtype=np.float32)
    rare = pool["player"].to_list().index("WR4")
    scores[rare] = 80.0
    result = solve_world_optimal_frontier(pool, scores)
    assert rare in result.indices
