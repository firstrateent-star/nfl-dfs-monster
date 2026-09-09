import polars as pl
import pytest

from monster.dfs.value import attach_salary_value, normalize_fanduel_pool, salary_join_audit


def _dist(player: str, team: str = "DET", position: str = "RB") -> pl.DataFrame:
    return pl.DataFrame(
        {
            "position": [position],
            "player": [player],
            "team_id": [team],
            "fd_mean": [18.0],
            "fd_p90": [27.0],
            "fd_p95": [30.0],
            "fd_p99": [36.0],
        }
    )


def _pool(player: str, team: str = "DET", position: str = "RB") -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Position": [position],
            "Player": [player],
            "Team": [team],
            "Salary": [9000],
            "FanDuel_ID": ["133104-1"],
        }
    )


def test_salary_value_is_downstream_and_exact() -> None:
    out = attach_salary_value(_dist("Example Runner"), _pool("Example Runner"))
    row = out.row(0, named=True)
    assert row["salary"] == 9000
    assert row["fd_mean_per_1k"] == pytest.approx(2.0)
    assert row["fd_p90_per_1k"] == pytest.approx(3.0)
    assert salary_join_audit(out)["salary_unmatched_offensive_players"] == 0


def test_suffix_identity_matches_without_guessing() -> None:
    out = attach_salary_value(_dist("James Cook", "BUF"), _pool("James Cook III", "BUF"))
    assert out["salary"].item() == 9000


def test_known_display_alias_matches() -> None:
    out = attach_salary_value(
        _dist("Marquise Brown", "KC", "WR"),
        _pool("Hollywood Brown", "KC", "WR"),
    )
    assert out["fanduel_id"].item() == "133104-1"


def test_missing_salary_columns_fail_loudly() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        normalize_fanduel_pool(pl.DataFrame({"Player": ["A"]}))


def test_unmatched_players_remain_auditable() -> None:
    pool = pl.DataFrame(
        schema={
            "Position": pl.String,
            "Player": pl.String,
            "Team": pl.String,
            "Salary": pl.Int64,
            "FanDuel_ID": pl.String,
        }
    )
    out = attach_salary_value(_dist("Missing Player", "BUF", "WR"), pool)
    assert out["salary"].null_count() == 1
    assert salary_join_audit(out)["salary_unmatched_offensive_players"] == 1
