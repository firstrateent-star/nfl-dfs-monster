from __future__ import annotations

from dataclasses import dataclass, field
from math import exp
from pathlib import Path
from typing import Sequence

import numpy as np
import polars as pl


def _channel(player: object, name: str) -> float:
    return float(np.clip(getattr(player, f"{name}_skill", 0.0) or 0.0, -1.0, 1.0))


def _skill_composite(player: object, family: str) -> float:
    if family == "receiving":
        value = (
            0.30 * _channel(player, "speed")
            + 0.30 * _channel(player, "route_separation")
            + 0.25 * _channel(player, "open_field")
            + 0.15 * _channel(player, "catchpoint")
        )
    elif family == "designed_run":
        value = (
            0.22 * _channel(player, "speed")
            + 0.28 * _channel(player, "open_field")
            + 0.30 * _channel(player, "rush_creation")
            + 0.20 * _channel(player, "runner_power")
        )
    else:
        value = 0.0
    return float(np.clip(value, -1.0, 1.0))


def relative_opportunity(player: object, pool: Sequence[object]) -> float:
    """Role concentration relative to the mean eligible teammate opportunity."""
    if not pool:
        return 1.0
    weights = np.asarray(
        [max(float(getattr(item, "usage_weight", 0.0) or 0.0), 0.001) for item in pool],
        dtype=float,
    )
    mean = float(weights.mean()) if len(weights) else 1.0
    selected = max(float(getattr(player, "usage_weight", 0.0) or 0.0), 0.001)
    return float(np.clip(selected / max(mean, 1e-6), 0.25, 4.0))


def opportunity_skill_multiplier(
    player: object,
    *,
    family: str,
    relative_opportunity_value: float,
) -> float:
    """Convert skill × role concentration into a bounded neutral-centered tail multiplier."""
    skill = _skill_composite(player, family)
    if abs(skill) <= 1e-12:
        return 1.0
    opportunity_authority = float(
        np.clip(max(relative_opportunity_value, 0.01) ** 0.25, 0.72, 1.38)
    )
    coefficient = 0.68 if family == "receiving" else 0.72
    multiplier = exp(coefficient * skill * opportunity_authority)
    return float(np.clip(multiplier, 0.68, 1.55))


@dataclass
class _ActorState:
    player_id: str
    family: str
    relative_opportunity: float
    skill_composite: float
    multiplier: float


@dataclass
class OpportunitySkillTailV3:
    """Runtime context for the opportunity × skill × matchup tail experiment."""

    current_pass: _ActorState | None = None
    current_run: _ActorState | None = None
    records: dict[tuple[str, str], dict[str, object]] = field(default_factory=dict)

    def _register(self, player: object, pool: Sequence[object], family: str) -> _ActorState:
        rel = relative_opportunity(player, pool)
        skill = _skill_composite(player, family)
        multiplier = opportunity_skill_multiplier(
            player,
            family=family,
            relative_opportunity_value=rel,
        )
        player_id = str(getattr(player, "player_id", ""))
        state = _ActorState(
            player_id=player_id,
            family=family,
            relative_opportunity=rel,
            skill_composite=skill,
            multiplier=multiplier,
        )
        key = (player_id, family)
        existing = self.records.get(key)
        selections = int(existing.get("selections", 0)) + 1 if existing else 1
        self.records[key] = {
            "player_id": player_id,
            "player_name": str(getattr(player, "name", player_id)),
            "position": str(getattr(player, "position", "")),
            "play_family": family,
            "usage_weight": float(getattr(player, "usage_weight", 0.0) or 0.0),
            "relative_opportunity": rel,
            "skill_composite": skill,
            "tail_multiplier": multiplier,
            "selections": selections,
        }
        return state

    def register_pass(self, player: object, pool: Sequence[object]) -> None:
        self.current_pass = self._register(player, pool, "receiving")

    def register_run(self, player: object, pool: Sequence[object]) -> None:
        self.current_run = self._register(player, pool, "designed_run")

    def pass_multiplier(self) -> float:
        return 1.0 if self.current_pass is None else self.current_pass.multiplier

    def run_multiplier(self) -> float:
        return 1.0 if self.current_run is None else self.current_run.multiplier

    def write(self, out: Path) -> None:
        if not self.records:
            return
        out.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(list(self.records.values())).sort(
            ["play_family", "selections", "player_name"],
            descending=[False, True, False],
        ).write_csv(out / "opportunity_skill_tail_snapshot.csv")
