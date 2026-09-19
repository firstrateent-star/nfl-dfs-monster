from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Protocol

from monster.reality.ledger import RealityLedger, ledger_from_v6_result
from monster.reality.world_state import (
    GameDayLatents,
    PregameWorld,
    TeamPregameState,
    WorldKey,
)
from monster.sim import game_loop_v13
from monster.sim.chaos_ecology import DEFAULT_CHAOS_ECOLOGY, ChaosEcology
from monster.sim.event_ledger import assert_event_conservation
from monster.sim.game_loop_v13 import GameResultV13
from monster.sim.matchup_kernel import DefensiveUnit
from monster.sim.play_kernel import TeamIdentity


@dataclass(frozen=True)
class RealityRuntimeFingerprint:
    architecture: str
    schema_version: str
    game_runtime: str
    scrimmage_runtime: str
    football_first: bool
    market_inputs_to_football: bool
    direct_fantasy_inputs_to_football: bool
    direct_score_adjustment: bool
    runtime_hash: str


@dataclass(frozen=True)
class RealityGameRequest:
    world: PregameWorld
    away: TeamIdentity
    home: TeamIdentity
    away_defense_strength: float = 1.0
    home_defense_strength: float = 1.0
    away_defense: DefensiveUnit | None = None
    home_defense: DefensiveUnit | None = None
    max_plays: int = 260
    max_overtime_plays: int = 80
    penalty_rate: float = 0.055
    chaos_ecology: ChaosEcology = DEFAULT_CHAOS_ECOLOGY

    def __post_init__(self) -> None:
        if self.away.team_id != self.world.away.team_id:
            raise ValueError("away identity must match the pregame world")
        if self.home.team_id != self.world.home.team_id:
            raise ValueError("home identity must match the pregame world")

    @classmethod
    def compatibility(
        cls,
        *,
        away: TeamIdentity,
        home: TeamIdentity,
        seed: int,
        world_index: int = 0,
        away_defense_strength: float = 1.0,
        home_defense_strength: float = 1.0,
        away_defense: DefensiveUnit | None = None,
        home_defense: DefensiveUnit | None = None,
        latents: GameDayLatents | None = None,
    ) -> RealityGameRequest:
        game_id = f"{away.team_id}@{home.team_id}"

        def roster(team: TeamIdentity) -> tuple[str, ...]:
            values = {
                team.quarterback.player_id,
                *(player.player_id for player in team.rushers),
                *(player.player_id for player in team.receivers),
            }
            return tuple(sorted(values))

        world = PregameWorld(
            key=WorldKey(game_id=game_id, world_index=world_index, seed=seed),
            away=TeamPregameState(
                team_id=away.team_id,
                starting_qb_id=away.quarterback.player_id,
                active_player_ids=roster(away),
            ),
            home=TeamPregameState(
                team_id=home.team_id,
                starting_qb_id=home.quarterback.player_id,
                active_player_ids=roster(home),
            ),
            latents=latents or GameDayLatents.neutral(),
            provenance=(
                ("architecture", "v7.0-shell"),
                ("football_runtime", "frozen-v6-compatible"),
            ),
        )
        return cls(
            world=world,
            away=away,
            home=home,
            away_defense_strength=away_defense_strength,
            home_defense_strength=home_defense_strength,
            away_defense=away_defense,
            home_defense=home_defense,
        )


@dataclass(frozen=True)
class RealityGameResult:
    world: PregameWorld
    football: GameResultV13
    ledger: RealityLedger
    runtime: RealityRuntimeFingerprint


class RealityEngine(Protocol):
    def run(self, request: RealityGameRequest) -> RealityGameResult: ...


def _callable_id(fn: object) -> str:
    return (
        f"{getattr(fn, '__module__', '<unknown>')}."
        f"{getattr(fn, '__qualname__', getattr(fn, '__name__', '<unknown>'))}"
    )


def compatibility_runtime_fingerprint() -> RealityRuntimeFingerprint:
    payload = {
        "architecture": "v7.0-shell-v6-compatibility",
        "schema_version": "v7.0.0",
        "game_runtime": _callable_id(game_loop_v13.simulate_game),
        "scrimmage_runtime": _callable_id(game_loop_v13.simulate_scrimmage_play),
        "football_first": True,
        "market_inputs_to_football": False,
        "direct_fantasy_inputs_to_football": False,
        "direct_score_adjustment": False,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return RealityRuntimeFingerprint(**payload, runtime_hash=digest)


class V6CompatibilityEngine:
    """New v7 contracts around unchanged, already-composed v6 football behavior."""

    def run(self, request: RealityGameRequest) -> RealityGameResult:
        runtime = compatibility_runtime_fingerprint()
        result = game_loop_v13.simulate_game(
            request.away,
            request.home,
            away_defense_strength=request.away_defense_strength,
            home_defense_strength=request.home_defense_strength,
            away_defense=request.away_defense,
            home_defense=request.home_defense,
            seed=request.world.key.seed,
            max_plays=request.max_plays,
            max_overtime_plays=request.max_overtime_plays,
            penalty_rate=request.penalty_rate,
            chaos_ecology=request.chaos_ecology,
        )
        assert_event_conservation(result)
        ledger = ledger_from_v6_result(
            world=request.world.key,
            result=result,
            away=request.away,
            home=request.home,
            source_runtime=runtime.runtime_hash,
        )
        return RealityGameResult(
            world=request.world,
            football=result,
            ledger=ledger,
            runtime=runtime,
        )
