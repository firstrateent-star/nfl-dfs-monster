import polars as pl
import pytest

from monster.dfs.value import attach_salary_value, normalize_fanduel_pool, salary_join_audit


def test_salary_value_is_downstream_and_exact() -> None:
    distributions = pl.DataFrame(
        {
            "position": ["RB"],
            "player": ["Example Runner"],
            "team_id": ["DET"],
            "fd_mean": [18.0],
            "fd_p90": [27.0],
            "fd_p95": [30.0],
            "fd_p99": [36.0],
        }
    )
    pool = pl.DataFrame(
        {
            "Position": ["RB"],
            "Player": ["Example Runner"],
            "Team": ["DET"],
            "Salary": [9000],
            "FanDuel_ID": ["133104-1"],
        }
    )
    out = attach_salary_value(distributions, pool)
    row = out.row(0, named=True)
    assert row["salary"] == 9000
    assert row["fd_mean_per_1k"] == pytest.approx(2.0)
    assert row["fd_p90_per_1k"] == pytest.approx(3.0)
    assert salary_join_audit(out)["salary_unmatched_offensive_players"] == 0


def test_missing_salary_columns_fail_loudly() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        normalize_fanduel_pool(pl.DataFrame({"Player": ["A"]}))


def test_unmatched_players_remain_auditable() -> None:
    distributions = pl.DataFrame(
        {
            "position": ["WR"],
            "player": ["Missing Player"],
            "team_id": ["BUF"],
            "fd_mean": [8.0],
            "fd_p90": [16.0],
            "fd_p95": [18.0],
            "fd_p99": [24.0],
        }
    )
    pool = pl.DataFrame(
        schema={
            "Position": pl.String,
            "Player": pl.String,
            "Team": pl.String,
            "Salary": pl.Int64,
            "FanDuel_ID": pl.String,
        }
    )
    out = attach_salary_value(distributions, pool)
    assert out["salary"].null_count() == 1
    assert salary_join_audit(out)["salary_unmatched_offensive_players"] == 1
