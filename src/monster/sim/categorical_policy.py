from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from monster.sim.game_flow import GameFlowState
from monster.sim.game_flow_lookup import FlowContextKey, context_key


@dataclass(frozen=True)
class CategoryEvidence:
    """Observed categorical counts with sample size preserved for shrinkage."""

    counts: Mapping[str, int]
    samples: int

    def probabilities(self, categories: Sequence[str]) -> np.ndarray:
        values = np.asarray([max(int(self.counts.get(category, 0)), 0) for category in categories], dtype=float)
        total = float(values.sum())
        if total <= 0.0:
            return np.full(len(categories), 1.0 / len(categories), dtype=float)
        return values / total


def shrink_distribution(
    parent: np.ndarray,
    child: CategoryEvidence | None,
    *,
    categories: Sequence[str],
    shrinkage_samples: float,
) -> np.ndarray:
    """Shrink a finer categorical distribution toward its broader parent."""

    if shrinkage_samples <= 0:
        raise ValueError("shrinkage_samples must be positive")
    if child is None or child.samples <= 0:
        return parent.copy()
    authority = float(child.samples) / (float(child.samples) + shrinkage_samples)
    child_probs = child.probabilities(categories)
    out = (1.0 - authority) * parent + authority * child_probs
    total = float(out.sum())
    return out / total if total > 0 else parent.copy()


def sample_category(
    categories: Sequence[str], probabilities: Sequence[float], rng: np.random.Generator
) -> str:
    if len(categories) != len(probabilities) or not categories:
        raise ValueError("categories and probabilities must be non-empty and aligned")
    probs = np.asarray(probabilities, dtype=float)
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if total <= 0:
        probs = np.full(len(categories), 1.0 / len(categories), dtype=float)
    else:
        probs /= total
    return str(categories[int(rng.choice(len(categories), p=probs))])


def _evidence_from_counts(counts: Mapping[str, int], categories: Sequence[str]) -> CategoryEvidence | None:
    clean = {category: max(int(counts.get(category, 0)), 0) for category in categories}
    samples = sum(clean.values())
    return CategoryEvidence(clean, samples) if samples > 0 else None


def _row_context(row: Mapping[str, object]) -> FlowContextKey:
    return FlowContextKey(
        down=int(row["down"]),
        distance_bucket=str(row["distance_bucket"]),
        field_zone=str(row["field_zone"]),
        time_mode=str(row["time_mode"]),
        score_state=str(row["score_state"]),
    )


def _group_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    categories: Sequence[str],
    key_fn,
    category_field: str,
    count_field: str,
) -> dict[object, CategoryEvidence]:
    grouped: dict[object, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        category = str(row[category_field])
        if category not in categories:
            continue
        grouped[key_fn(row)][category] += int(row[count_field])
    out: dict[object, CategoryEvidence] = {}
    for key, counts in grouped.items():
        evidence = _evidence_from_counts(counts, categories)
        if evidence is not None:
            out[key] = evidence
    return out


@dataclass(frozen=True)
class ContextualCategoricalPolicy:
    """Hierarchical categorical policy shared by pass depth and run geometry.

    Resolution success is intentionally absent. This object answers only which intent
    category is attempted given pre-snap state and optional team/player identity.
    """

    categories: tuple[str, ...]
    league_exact: Mapping[FlowContextKey, CategoryEvidence]
    league_no_score: Mapping[tuple[int, str, str, str], CategoryEvidence]
    league_down_distance_field: Mapping[tuple[int, str, str], CategoryEvidence]
    league_down_distance: Mapping[tuple[int, str], CategoryEvidence]
    league_overall: CategoryEvidence
    team_exact: Mapping[FlowContextKey, CategoryEvidence]
    team_down_distance: Mapping[tuple[int, str], CategoryEvidence]
    team_overall: CategoryEvidence | None
    actor_overall: Mapping[str, CategoryEvidence]
    league_shrinkage_samples: float = 100.0
    team_shrinkage_samples: float = 100.0
    actor_shrinkage_samples: float = 120.0

    def probabilities_for(self, flow: GameFlowState, *, actor_id: str | None = None) -> np.ndarray:
        key = context_key(flow)
        probs = self.league_overall.probabilities(self.categories)
        probs = shrink_distribution(
            probs,
            self.league_down_distance.get((key.down, key.distance_bucket)),
            categories=self.categories,
            shrinkage_samples=self.league_shrinkage_samples,
        )
        probs = shrink_distribution(
            probs,
            self.league_down_distance_field.get((key.down, key.distance_bucket, key.field_zone)),
            categories=self.categories,
            shrinkage_samples=self.league_shrinkage_samples,
        )
        probs = shrink_distribution(
            probs,
            self.league_no_score.get((key.down, key.distance_bucket, key.field_zone, key.time_mode)),
            categories=self.categories,
            shrinkage_samples=self.league_shrinkage_samples,
        )
        probs = shrink_distribution(
            probs,
            self.league_exact.get(key),
            categories=self.categories,
            shrinkage_samples=self.league_shrinkage_samples,
        )

        team = self.team_exact.get(key)
        if team is None:
            team = self.team_down_distance.get((key.down, key.distance_bucket))
        if team is None:
            team = self.team_overall
        probs = shrink_distribution(
            probs,
            team,
            categories=self.categories,
            shrinkage_samples=self.team_shrinkage_samples,
        )
        if actor_id is not None:
            probs = shrink_distribution(
                probs,
                self.actor_overall.get(actor_id),
                categories=self.categories,
                shrinkage_samples=self.actor_shrinkage_samples,
            )
        return probs

    def sample(
        self,
        flow: GameFlowState,
        rng: np.random.Generator,
        *,
        actor_id: str | None = None,
    ) -> str:
        return sample_category(
            self.categories,
            self.probabilities_for(flow, actor_id=actor_id),
            rng,
        )


def build_contextual_categorical_policy(
    *,
    categories: Sequence[str],
    league_rows: Iterable[Mapping[str, object]],
    team_rows: Iterable[Mapping[str, object]],
    actor_rows: Iterable[Mapping[str, object]] = (),
    team_id: str,
    category_field: str = "category",
    count_field: str = "attempts",
    actor_field: str = "actor_id",
    league_shrinkage_samples: float = 100.0,
    team_shrinkage_samples: float = 100.0,
    actor_shrinkage_samples: float = 120.0,
) -> ContextualCategoricalPolicy:
    categories_tuple = tuple(str(category) for category in categories)
    if not categories_tuple:
        raise ValueError("categories cannot be empty")
    league = list(league_rows)
    team = [row for row in team_rows if str(row["team_id"]) == team_id]
    actors = list(actor_rows)
    if not league:
        raise ValueError("league categorical evidence cannot be empty")

    league_exact = _group_rows(
        league,
        categories=categories_tuple,
        key_fn=_row_context,
        category_field=category_field,
        count_field=count_field,
    )
    league_no_score = _group_rows(
        league,
        categories=categories_tuple,
        key_fn=lambda row: (
            int(row["down"]),
            str(row["distance_bucket"]),
            str(row["field_zone"]),
            str(row["time_mode"]),
        ),
        category_field=category_field,
        count_field=count_field,
    )
    league_ddf = _group_rows(
        league,
        categories=categories_tuple,
        key_fn=lambda row: (
            int(row["down"]),
            str(row["distance_bucket"]),
            str(row["field_zone"]),
        ),
        category_field=category_field,
        count_field=count_field,
    )
    league_dd = _group_rows(
        league,
        categories=categories_tuple,
        key_fn=lambda row: (int(row["down"]), str(row["distance_bucket"])),
        category_field=category_field,
        count_field=count_field,
    )
    league_overall_map = _group_rows(
        league,
        categories=categories_tuple,
        key_fn=lambda _row: "ALL",
        category_field=category_field,
        count_field=count_field,
    )
    league_overall = league_overall_map.get("ALL")
    if league_overall is None:
        raise ValueError("league categorical evidence has zero usable samples")

    team_exact = _group_rows(
        team,
        categories=categories_tuple,
        key_fn=_row_context,
        category_field=category_field,
        count_field=count_field,
    )
    team_dd = _group_rows(
        team,
        categories=categories_tuple,
        key_fn=lambda row: (int(row["down"]), str(row["distance_bucket"])),
        category_field=category_field,
        count_field=count_field,
    )
    team_overall = _group_rows(
        team,
        categories=categories_tuple,
        key_fn=lambda _row: "ALL",
        category_field=category_field,
        count_field=count_field,
    ).get("ALL")
    actor_overall = _group_rows(
        actors,
        categories=categories_tuple,
        key_fn=lambda row: str(row[actor_field]),
        category_field=category_field,
        count_field=count_field,
    )

    return ContextualCategoricalPolicy(
        categories=categories_tuple,
        league_exact=league_exact,
        league_no_score=league_no_score,
        league_down_distance_field=league_ddf,
        league_down_distance=league_dd,
        league_overall=league_overall,
        team_exact=team_exact,
        team_down_distance=team_dd,
        team_overall=team_overall,
        actor_overall=actor_overall,
        league_shrinkage_samples=league_shrinkage_samples,
        team_shrinkage_samples=team_shrinkage_samples,
        actor_shrinkage_samples=actor_shrinkage_samples,
    )
