from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from monster.snapshot.model import GameState

@dataclass
class GameWorlds:
    away_points: np.ndarray
    home_points: np.ndarray

    @property
    def total(self) -> np.ndarray:
        return self.away_points + self.home_points


def _mean_one_lognormal(rng: np.random.Generator, sigma: float, n: int) -> np.ndarray:
    return np.exp(rng.normal(-0.5 * sigma * sigma, sigma, n))


def simulate_game(game: GameState, worlds: int, seed: int) -> GameWorlds:
    """Vectorized market-blind score skeleton.

    This is infrastructure, not the final football model. The final Monster v1 replaces the compact
    strength formula with drive/opportunity mechanics compiled from the rich slate snapshot.
    """
    rng = np.random.default_rng(seed)
    shared = _mean_one_lognormal(rng, 0.12, worlds)

    def team_points(team, home: bool):
        home_shift = 0.7 if home else -0.7
        mu = 23.0 + 5.0 * team.offense_strength - 4.0 * team.defense_strength + home_shift
        mu += team.injury_effect + team.weather_effect
        mu = float(np.clip(mu, 10.0, 38.0))
        state = _mean_one_lognormal(rng, team.uncertainty, worlds)
        lam_td = np.maximum(0.4, mu * 0.78 / 7.0) * shared * state
        lam_fg = np.maximum(0.25, mu * 0.22 / 3.0) * (shared ** 0.6) * (state ** 0.5)
        td = rng.poisson(lam_td)
        fg = rng.poisson(lam_fg)
        return (7 * td + 3 * fg).astype(np.int16)

    return GameWorlds(
        away_points=team_points(game.away, False),
        home_points=team_points(game.home, True),
    )
