from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class MaddenAttributeVector:
    """Lossless numeric Madden player evidence with mechanism-specific views.

    Raw attributes are preserved instead of collapsing the source into OVR or a few
    hand-picked ratings.  Consumers request only the attributes that have football
    jurisdiction for the mechanism they resolve, preventing unrelated ratings from
    becoming generic fantasy-point multipliers.
    """

    values: tuple[tuple[str, float], ...] = ()

    @classmethod
    def from_row(cls, row: Mapping[str, object]) -> "MaddenAttributeVector":
        values: list[tuple[str, float]] = []
        for key, value in row.items():
            name = str(key).lower()
            if not name.startswith("madden_") or name in {"madden_ovr", "madden_overall"}:
                continue
            try:
                number = float(value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
            if np.isfinite(number):
                values.append((name, number))
        return cls(tuple(sorted(values)))

    def as_dict(self) -> dict[str, float]:
        return dict(self.values)

    def get(self, *names: str) -> float | None:
        data = self.as_dict()
        for name in names:
            key = name if name.startswith("madden_") else f"madden_{name}"
            if key in data:
                return data[key]
        return None

    def mean(self, *names: str) -> float | None:
        found = [self.get(name) for name in names]
        values = [float(value) for value in found if value is not None]
        return float(np.mean(values)) if values else None


# Madden field names vary slightly by source/version.  These aliases let the raw
# vector remain lossless while mechanisms consume semantically equivalent fields.
MADDEN_CHANNELS: dict[str, tuple[str, ...]] = {
    "movement": ("speed", "acceleration", "agility", "change_of_direction", "jumping"),
    "receiver_release": ("release", "short_route_running", "medium_route_running", "deep_route_running"),
    "catching": ("catching", "catch_in_traffic", "spectacular_catch"),
    "ball_security": ("carrying",),
    "open_field_rushing": ("break_tackle", "trucking", "stiff_arm", "juke_move", "spin_move", "change_of_direction"),
    "qb_accuracy": ("throw_accuracy_short", "throw_accuracy_mid", "throw_accuracy_medium", "throw_accuracy_deep", "throw_on_run", "throw_under_pressure", "play_action"),
    "qb_arm": ("throw_power",),
    "pass_block": ("pass_block", "pass_block_power", "pass_block_finesse", "awareness"),
    "run_block": ("run_block", "run_block_power", "run_block_finesse", "impact_blocking", "lead_block"),
    "pass_rush": ("power_moves", "finesse_moves", "block_shedding", "pursuit", "play_recognition"),
    "run_defense": ("block_shedding", "pursuit", "tackle", "play_recognition", "strength"),
    "man_coverage": ("man_coverage", "press", "play_recognition", "speed", "acceleration"),
    "zone_coverage": ("zone_coverage", "play_recognition", "awareness", "speed"),
    "tackling": ("tackle", "hit_power", "pursuit", "strength"),
    "kicking": ("kick_power", "kick_accuracy"),
    "returning": ("kick_return", "speed", "acceleration", "change_of_direction", "carrying"),
}


def channel_mean(vector: MaddenAttributeVector, channel: str) -> float | None:
    return vector.mean(*MADDEN_CHANNELS.get(channel, ()))
