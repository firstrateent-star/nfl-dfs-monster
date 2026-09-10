from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from monster.sim.game_flow import GameFlowState
from monster.sim.game_flow_brain import (
    DecisionAdjustment,
    HierarchicalFlowEvidence,
    RateEvidence,
)


@dataclass(frozen=True)
class FlowContextKey:
    down: int
    distance_bucket: str
    field_zone: str
    time_mode: str
    score_state: str


def distance_bucket(distance: float) -> str:
    if distance <= 1.0:
        return "1"
    if distance <= 2.0:
        return "2"
    if distance <= 4.0:
        return "3_4"
    if distance <= 7.0:
        return "5_7"
    if distance <= 10.0:
        return "8_10"
    if distance <= 15.0:
        return "11_15"
    return "16_plus"


def field_zone(yardline: float) -> str:
    # Monster yardline grows from the possessing offense's own goal line.
    if yardline <= 20.0:
        return "backed_up"
    if yardline <= 40.0:
        return "own_field"
    if yardline <= 60.0:
        return "midfield"
    if yardline < 80.0:
        return "opp_territory"
    if yardline < 90.0:
        return "high_red_zone"
    return "low_red_zone"


def score_state(margin: int) -> str:
    if margin <= -9:
        return "trail_9_plus"
    if margin < 0:
        return "trail_1_8"
    if margin == 0:
        return "tied"
    if margin <= 8:
        return "lead_1_8"
    return "lead_9_plus"


def time_mode(flow: GameFlowState) -> str:
    if flow.quarter == 2 and flow.seconds_remaining_in_period <= 120:
        return "two_minute_first_half"
    if flow.quarter == 4 and flow.seconds_remaining_in_period <= 120:
        return "two_minute_game"
    if flow.quarter == 4 and flow.seconds_remaining_in_period <= 240:
        return "four_minute_game"
    if flow.quarter in (1, 2, 3, 4):
        return f"q{flow.quarter}_normal"
    return "overtime"


def context_key(flow: GameFlowState) -> FlowContextKey:
    return FlowContextKey(
        down=flow.down,
        distance_bucket=distance_bucket(flow.distance),
        field_zone=field_zone(flow.yardline),
        time_mode=time_mode(flow),
        score_state=score_state(flow.score_margin),
    )


def _rate(rows: Iterable[tuple[float, int]]) -> RateEvidence | None:
    total_samples = 0
    weighted_success = 0.0
    for value, samples in rows:
        count = max(int(samples), 0)
        total_samples += count
        weighted_success += float(value) * count
    if total_samples <= 0:
        return None
    return RateEvidence(weighted_success / total_samples, total_samples)


def _shrink_child_to_parent(
    child: RateEvidence | None,
    parent: RateEvidence,
    *,
    shrinkage_samples: float,
) -> RateEvidence:
    """Empirically shrink a finer league cell toward its broader parent context."""

    if child is None or child.samples <= 0:
        return parent
    if shrinkage_samples <= 0:
        raise ValueError("league shrinkage_samples must be positive")
    authority = child.samples / (child.samples + shrinkage_samples)
    rate = parent.rate + authority * (child.rate - parent.rate)
    return RateEvidence(rate=float(rate), samples=child.samples)


@dataclass(frozen=True)
class TeamGameFlowPolicy:
    team_id: str
    league_exact: Mapping[FlowContextKey, RateEvidence]
    league_no_score: Mapping[tuple[int, str, str, str], RateEvidence]
    league_down_distance_field: Mapping[tuple[int, str, str], RateEvidence]
    league_down_distance: Mapping[tuple[int, str], RateEvidence]
    league_overall: RateEvidence
    team_exact: Mapping[FlowContextKey, RateEvidence]
    team_down_distance: Mapping[tuple[int, str], RateEvidence]
    team_overall: RateEvidence | None
    team_neutral_rate: float | None
    league_neutral_rate: float | None
    shrinkage_samples: float = 80.0
    league_shrinkage_samples: float = 80.0

    def evidence_for(
        self,
        flow: GameFlowState,
        *,
        personnel: DecisionAdjustment | None = None,
        opponent: DecisionAdjustment | None = None,
        environment: DecisionAdjustment | None = None,
        venue: DecisionAdjustment | None = None,
        adaptation: DecisionAdjustment | None = None,
    ) -> HierarchicalFlowEvidence:
        key = context_key(flow)

        # Every finer league context earns authority from its own sample size instead of
        # replacing a broader prior merely because an exact cell exists. This prevents
        # tiny five-dimensional situation cells from becoming false certainty.
        league = self.league_overall
        league = _shrink_child_to_parent(
            self.league_down_distance.get((key.down, key.distance_bucket)),
            league,
            shrinkage_samples=self.league_shrinkage_samples,
        )
        league = _shrink_child_to_parent(
            self.league_down_distance_field.get(
                (key.down, key.distance_bucket, key.field_zone)
            ),
            league,
            shrinkage_samples=self.league_shrinkage_samples,
        )
        league = _shrink_child_to_parent(
            self.league_no_score.get(
                (key.down, key.distance_bucket, key.field_zone, key.time_mode)
            ),
            league,
            shrinkage_samples=self.league_shrinkage_samples,
        )
        league = _shrink_child_to_parent(
            self.league_exact.get(key),
            league,
            shrinkage_samples=self.league_shrinkage_samples,
        )

        team = self.team_exact.get(key)
        if team is None:
            team = self.team_down_distance.get((key.down, key.distance_bucket))
        if team is None:
            team = self.team_overall

        return HierarchicalFlowEvidence(
            league_context=league,
            team_context=team,
            team_neutral_rate=self.team_neutral_rate,
            league_neutral_rate=self.league_neutral_rate,
            shrinkage_samples=self.shrinkage_samples,
            personnel=personnel or DecisionAdjustment("personnel"),
            opponent=opponent or DecisionAdjustment("opponent"),
            environment=environment or DecisionAdjustment("environment"),
            venue=venue or DecisionAdjustment("venue"),
            adaptation=adaptation or DecisionAdjustment("adaptation"),
        )


def _row_key(row: Mapping[str, object]) -> FlowContextKey:
    return FlowContextKey(
        down=int(row["down"]),
        distance_bucket=str(row["distance_bucket"]),
        field_zone=str(row["field_zone"]),
        time_mode=str(row["time_mode"]),
        score_state=str(row["score_state"]),
    )


def build_team_game_flow_policy(
    *,
    team_id: str,
    league_rows: Iterable[Mapping[str, object]],
    team_rows: Iterable[Mapping[str, object]],
    team_neutral_rate: float | None,
    league_neutral_rate: float | None,
    shrinkage_samples: float = 80.0,
    league_shrinkage_samples: float = 80.0,
) -> TeamGameFlowPolicy:
    """Build one team's pure-Python lookup from compact compiled tables."""

    league_rows_list = list(league_rows)
    team_rows_list = [row for row in team_rows if str(row["team_id"]) == team_id]
    if not league_rows_list:
        raise ValueError("league game-flow policy rows cannot be empty")
    if league_shrinkage_samples <= 0:
        raise ValueError("league_shrinkage_samples must be positive")

    league_exact = {
        _row_key(row): RateEvidence(float(row["dropback_rate"]), int(row["samples"]))
        for row in league_rows_list
    }
    team_exact = {
        _row_key(row): RateEvidence(float(row["dropback_rate"]), int(row["samples"]))
        for row in team_rows_list
    }

    no_score_parts: dict[tuple[int, str, str, str], list[tuple[float, int]]] = defaultdict(list)
    ddf_parts: dict[tuple[int, str, str], list[tuple[float, int]]] = defaultdict(list)
    dd_parts: dict[tuple[int, str], list[tuple[float, int]]] = defaultdict(list)
    league_all: list[tuple[float, int]] = []
    for row in league_rows_list:
        key = _row_key(row)
        pair = (float(row["dropback_rate"]), int(row["samples"]))
        no_score_parts[(key.down, key.distance_bucket, key.field_zone, key.time_mode)].append(pair)
        ddf_parts[(key.down, key.distance_bucket, key.field_zone)].append(pair)
        dd_parts[(key.down, key.distance_bucket)].append(pair)
        league_all.append(pair)

    team_dd_parts: dict[tuple[int, str], list[tuple[float, int]]] = defaultdict(list)
    team_all: list[tuple[float, int]] = []
    for row in team_rows_list:
        key = _row_key(row)
        pair = (float(row["dropback_rate"]), int(row["samples"]))
        team_dd_parts[(key.down, key.distance_bucket)].append(pair)
        team_all.append(pair)

    league_overall = _rate(league_all)
    if league_overall is None:
        raise ValueError("league game-flow policy has zero samples")

    return TeamGameFlowPolicy(
        team_id=team_id,
        league_exact=league_exact,
        league_no_score={
            key: value
            for key, parts in no_score_parts.items()
            if (value := _rate(parts)) is not None
        },
        league_down_distance_field={
            key: value
            for key, parts in ddf_parts.items()
            if (value := _rate(parts)) is not None
        },
        league_down_distance={
            key: value
            for key, parts in dd_parts.items()
            if (value := _rate(parts)) is not None
        },
        league_overall=league_overall,
        team_exact=team_exact,
        team_down_distance={
            key: value
            for key, parts in team_dd_parts.items()
            if (value := _rate(parts)) is not None
        },
        team_overall=_rate(team_all),
        team_neutral_rate=team_neutral_rate,
        league_neutral_rate=league_neutral_rate,
        shrinkage_samples=shrinkage_samples,
        league_shrinkage_samples=league_shrinkage_samples,
    )
