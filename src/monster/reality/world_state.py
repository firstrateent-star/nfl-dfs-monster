from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class WorldKey:
    """Stable identity for one Monte Carlo football universe."""

    game_id: str
    world_index: int
    seed: int

    def __post_init__(self) -> None:
        if not self.game_id.strip():
            raise ValueError("game_id cannot be empty")
        if self.world_index < 0:
            raise ValueError("world_index cannot be negative")
        if self.seed < 0:
            raise ValueError("seed cannot be negative")

    @property
    def world_id(self) -> str:
        return f"{self.game_id}:w{self.world_index}:s{self.seed}"


@dataclass(frozen=True)
class LatentFactor:
    """One coherent game-day condition, never a direct score adjustment."""

    name: str
    value: float
    source: str = "neutral"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("latent name cannot be empty")
        if not isfinite(self.value):
            raise ValueError("latent values must be finite")
        if not self.source.strip():
            raise ValueError("latent source cannot be empty")


@dataclass(frozen=True)
class GameDayLatents:
    factors: tuple[LatentFactor, ...] = ()

    def __post_init__(self) -> None:
        names = [factor.name for factor in self.factors]
        if len(names) != len(set(names)):
            raise ValueError("game-day latent names must be unique")

    @classmethod
    def neutral(cls) -> GameDayLatents:
        return cls()

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, float],
        *,
        source: str = "external",
    ) -> GameDayLatents:
        return cls(
            tuple(
                LatentFactor(name=name, value=float(value), source=source)
                for name, value in sorted(values.items())
            )
        )

    def value(self, name: str, default: float = 0.0) -> float:
        for factor in self.factors:
            if factor.name == name:
                return factor.value
        return default


@dataclass(frozen=True)
class TeamPregameState:
    """Pregame roster state before the live game can mutate it."""

    team_id: str
    starting_qb_id: str | None = None
    active_player_ids: tuple[str, ...] = ()
    unavailable_player_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.team_id.strip():
            raise ValueError("team_id cannot be empty")
        active = tuple(str(player_id) for player_id in self.active_player_ids)
        unavailable = tuple(str(player_id) for player_id in self.unavailable_player_ids)
        if len(active) != len(set(active)):
            raise ValueError("active player ids must be unique")
        if len(unavailable) != len(set(unavailable)):
            raise ValueError("unavailable player ids must be unique")
        if set(active) & set(unavailable):
            raise ValueError("a player cannot be active and unavailable in the same pregame world")
        if self.starting_qb_id is not None and active and self.starting_qb_id not in active:
            raise ValueError("starting QB must be active when an active roster is supplied")


@dataclass(frozen=True)
class PregameWorld:
    """Immutable upstream world sampled before live game transitions."""

    key: WorldKey
    away: TeamPregameState
    home: TeamPregameState
    latents: GameDayLatents = GameDayLatents()
    provenance: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.away.team_id == self.home.team_id:
            raise ValueError("away and home teams must differ")
        if self.key.game_id != f"{self.away.team_id}@{self.home.team_id}":
            raise ValueError("world game_id must equal '<away>@<home>'")
        keys = [key for key, _ in self.provenance]
        if len(keys) != len(set(keys)):
            raise ValueError("provenance keys must be unique")
