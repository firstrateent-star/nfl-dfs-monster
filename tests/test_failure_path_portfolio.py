from __future__ import annotations

from monster.dfs.portfolio import (
    PortfolioCandidate,
    select_failure_path_portfolio,
)


def _candidate(
    lineup_id,
    score,
    players,
    paths,
    salary=59800,
):
    return PortfolioCandidate(
        lineup_id=lineup_id,
        objective_score=score,
        salary=salary,
        player_ids=tuple(players),
        failure_paths=frozenset(paths),
    )


def test_portfolio_prefers_new_failure_path_over_cosmetic_player_diversity() -> None:
    core = [
        "p1",
        "p2",
        "p3",
        "p4",
        "p5",
        "p6",
        "p7",
        "p8",
        "p9",
    ]
    candidates = [
        _candidate("A", 200.0, core, {"ATL:COLLAPSE"}),
        _candidate(
            "B",
            197.0,
            [
                "p1",
                "p2",
                "p3",
                "p4",
                "p5",
                "p6",
                "p7",
                "p8",
                "x9",
            ],
            {"HOU:COLLAPSE"},
        ),
        _candidate(
            "C",
            198.0,
            [
                "c1",
                "c2",
                "c3",
                "c4",
                "c5",
                "c6",
                "c7",
                "c8",
                "c9",
            ],
            {"ATL:COLLAPSE"},
        ),
    ]
    selected = select_failure_path_portfolio(
        candidates,
        count=2,
        salary_cap=60000,
        min_salary=59000,
    )
    assert [item.lineup_id for item in selected] == ["A", "B"]


def test_portfolio_keeps_salary_as_downstream_tiebreaker() -> None:
    candidates = [
        _candidate(
            "A",
            200.0,
            [f"a{i}" for i in range(9)],
            {"PATH:A"},
            salary=59500,
        ),
        _candidate(
            "B",
            199.0,
            [f"b{i}" for i in range(9)],
            {"PATH:B"},
            salary=60000,
        ),
        _candidate(
            "C",
            199.0,
            [f"c{i}" for i in range(9)],
            {"PATH:B"},
            salary=59300,
        ),
    ]
    selected = select_failure_path_portfolio(
        candidates,
        count=2,
        salary_cap=60000,
        min_salary=59300,
    )
    assert selected[1].lineup_id == "B"
