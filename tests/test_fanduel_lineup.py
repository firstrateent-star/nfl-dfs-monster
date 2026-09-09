import polars as pl

from monster.dfs.lineup import audit_fanduel_lineup


def _lineup(positions: list[str], salaries: list[int] | None = None) -> pl.DataFrame:
    salaries = salaries or [6500] * len(positions)
    return pl.DataFrame(
        {
            "position": positions,
            "fanduel_id": [f"id-{idx}" for idx in range(len(positions))],
            "salary": salaries,
        }
    )


def test_standard_rb_flex_is_legal() -> None:
    result = audit_fanduel_lineup(_lineup(["QB", "RB", "RB", "RB", "WR", "WR", "WR", "TE", "D"]))
    assert result.legal


def test_wr_and_te_flex_are_legal() -> None:
    assert audit_fanduel_lineup(_lineup(["QB", "RB", "RB", "WR", "WR", "WR", "WR", "TE", "DEF"])).legal
    assert audit_fanduel_lineup(_lineup(["QB", "RB", "RB", "WR", "WR", "WR", "TE", "TE", "DST"])).legal


def test_salary_cap_and_position_failures_are_explicit() -> None:
    expensive = _lineup(["QB", "RB", "RB", "RB", "WR", "WR", "WR", "TE", "D"], [7000] * 9)
    assert not audit_fanduel_lineup(expensive).legal
    malformed = _lineup(["QB", "RB", "RB", "WR", "WR", "WR", "TE", "QB", "D"])
    assert not audit_fanduel_lineup(malformed).legal
