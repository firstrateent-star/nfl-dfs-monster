from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioCandidate:
    lineup_id: str
    objective_score: float
    salary: int
    player_ids: tuple[str, ...]
    failure_paths: frozenset[str]
    source_world: int | None = None


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def _player_overlap(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    a = set(left)
    b = set(right)
    denom = max(min(len(a), len(b)), 1)
    return len(a & b) / denom


def select_failure_path_portfolio(
    candidates: list[PortfolioCandidate],
    *,
    count: int,
    salary_cap: int,
    min_salary: int = 0,
    score_floor_ratio: float = 0.88,
    scenario_weight: float = 0.34,
    player_weight: float = 0.07,
    salary_weight: float = 0.02,
) -> list[PortfolioCandidate]:
    """Select ceiling lineups that fail differently, not merely look different.

    The football-world objective remains primary. Scenario novelty receives materially
    more weight than raw player novelty, so a repeated conviction can survive when it
    reaches first place through a different football path. Salary is only a small
    downstream tiebreak signal.
    """

    if count <= 0:
        return []
    feasible = [
        candidate
        for candidate in candidates
        if min_salary <= candidate.salary <= salary_cap
    ]
    if not feasible:
        return []

    best_score = max(candidate.objective_score for candidate in feasible)
    floor = best_score * float(score_floor_ratio)
    elite = [
        candidate
        for candidate in feasible
        if candidate.objective_score >= floor
    ]
    elite.sort(key=lambda c: (-c.objective_score, -c.salary, c.lineup_id))

    selected: list[PortfolioCandidate] = [elite.pop(0)]
    while elite and len(selected) < count:
        best_candidate: PortfolioCandidate | None = None
        best_key: tuple[float, float, int, str] | None = None
        for candidate in elite:
            score_ratio = candidate.objective_score / max(best_score, 1e-9)
            scenario_novelty = min(
                1.0
                - _jaccard(
                    candidate.failure_paths,
                    chosen.failure_paths,
                )
                for chosen in selected
            )
            player_novelty = min(
                1.0
                - _player_overlap(
                    candidate.player_ids,
                    chosen.player_ids,
                )
                for chosen in selected
            )
            salary_ratio = candidate.salary / max(salary_cap, 1)
            utility = (
                score_ratio
                + scenario_weight * scenario_novelty
                + player_weight * player_novelty
                + salary_weight * salary_ratio
            )
            key = (
                utility,
                candidate.objective_score,
                candidate.salary,
                candidate.lineup_id,
            )
            if best_key is None or key > best_key:
                best_key = key
                best_candidate = candidate

        if best_candidate is None:
            break
        selected.append(best_candidate)
        elite.remove(best_candidate)

    return selected
