from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from monster.sim.game_flow import FlowTag, GameFlowState

_STATE_PRIORITY = (
    (FlowTag.THIRD_AND_EXTREME, "third_and_18_plus"),
    (FlowTag.THIRD_AND_LONG, "third_and_7_plus"),
    (FlowTag.THIRD_AND_MEDIUM, "third_and_medium"),
    (FlowTag.THIRD_AND_SHORT, "third_and_short"),
    (FlowTag.SECOND_AND_SHORT, "second_and_2_or_less"),
    (FlowTag.END_FIRST_HALF, "end_first_half"),
    (FlowTag.LATE_TRAILING, "late_trailing"),
    (FlowTag.FOUR_MINUTE_LEAD, "four_minute_lead"),
    (FlowTag.LOW_RED_ZONE, "low_red_zone"),
    (FlowTag.BACKED_UP, "backed_up"),
)


def primary_flow_state_label(flow: GameFlowState) -> str:
    """Return one stable audit label without affecting decision authority."""

    for tag, label in _STATE_PRIORITY:
        if tag in flow.tags:
            return label
    return "ordinary"


@dataclass
class _Bucket:
    samples: int = 0
    dropbacks: int = 0
    predicted_sum: float = 0.0
    league_prior_sum: float = 0.0

    def add(self, *, is_dropback: bool, predicted: float, league_prior: float) -> None:
        self.samples += 1
        self.dropbacks += int(is_dropback)
        self.predicted_sum += float(predicted)
        self.league_prior_sum += float(league_prior)


class GameFlowTraceRecorder:
    """Aggregate passive pre-snap call traces without changing simulation behavior."""

    def __init__(self) -> None:
        self._buckets: dict[tuple[str, str, str], _Bucket] = defaultdict(_Bucket)

    def observe(
        self,
        *,
        game: str,
        offense: str,
        flow: GameFlowState,
        is_dropback: bool,
        predicted_probability: float,
        league_prior: float,
    ) -> None:
        label = primary_flow_state_label(flow)
        for game_key, offense_key in ((game, offense), ("ALL", "ALL")):
            self._buckets[(game_key, offense_key, label)].add(
                is_dropback=is_dropback,
                predicted=predicted_probability,
                league_prior=league_prior,
            )

    def rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for (game, offense, label), bucket in sorted(self._buckets.items()):
            if bucket.samples <= 0:
                continue
            simulated_rate = bucket.dropbacks / bucket.samples
            predicted_rate = bucket.predicted_sum / bucket.samples
            league_rate = bucket.league_prior_sum / bucket.samples
            rows.append(
                {
                    "game": game,
                    "offense": offense,
                    "special_state": label,
                    "samples": bucket.samples,
                    "dropbacks": bucket.dropbacks,
                    "simulated_dropback_rate": simulated_rate,
                    "mean_brain_probability": predicted_rate,
                    "mean_stabilized_league_prior": league_rate,
                    "sampler_calibration_error": abs(simulated_rate - predicted_rate),
                }
            )
        return rows
