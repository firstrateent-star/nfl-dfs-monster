from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from monster.dfs.defense import require_complete_defense_authority, score_defense_worlds
from monster.dfs.defense_events import simulate_defense_events


@dataclass(frozen=True)
class DefenseWorldRow:
    team_id: str
    opponent: str
    position: str
    player: str
    scores: np.ndarray


def build_defense_world_rows(
    *,
    game_index: list[dict],
    game_worlds: dict[str, np.ndarray],
    team_pass_attempts: dict[str, np.ndarray],
    seed: int,
) -> list[DefenseWorldRow]:
    """Translate frozen correlated game worlds into 24 downstream FanDuel D/ST rows."""
    require_complete_defense_authority(
        sacks_modeled=True,
        defensive_scores_modeled=True,
        special_teams_scores_modeled=True,
    )
    rows: list[DefenseWorldRow] = []
    for game in game_index:
        idx = int(game["game_index"])
        away = str(game["away_team"])
        home = str(game["home_team"])
        away_points = np.asarray(game_worlds[f"g{idx}_away_points"])
        home_points = np.asarray(game_worlds[f"g{idx}_home_points"])
        away_turnovers = np.asarray(game_worlds[f"g{idx}_away_turnovers"])
        home_turnovers = np.asarray(game_worlds[f"g{idx}_home_turnovers"])
        away_disruption = np.asarray(game_worlds[f"g{idx}_away_pass_disruption"])
        home_disruption = np.asarray(game_worlds[f"g{idx}_home_pass_disruption"])

        # Home defense faces away offense. Away defense faces home offense. Pass disruption is
        # already stored from the offense-facing game state, so preserve that same-world pairing.
        pairings = (
            (away, home, away_points, home_turnovers, home_disruption, team_pass_attempts[home]),
            (home, away, home_points, away_turnovers, away_disruption, team_pass_attempts[away]),
        )
        for side, (team, opponent, opponent_points, opponent_turnovers, disruption, attempts) in enumerate(pairings):
            events = simulate_defense_events(
                opponent_pass_attempts=np.asarray(attempts),
                opponent_turnovers=np.asarray(opponent_turnovers),
                pass_disruption=np.asarray(disruption),
                seed=seed + idx * 10007 + side * 101 + 7000001,
            )
            scores = score_defense_worlds(
                opponent_points=np.asarray(opponent_points),
                opponent_turnovers=np.asarray(opponent_turnovers),
                sacks=events.sacks,
                defensive_touchdowns=events.defensive_touchdowns,
                special_teams_touchdowns=events.special_teams_touchdowns,
                safeties=events.safeties,
            ).astype(np.float32)
            rows.append(
                DefenseWorldRow(
                    team_id=team,
                    opponent=opponent,
                    position="D",
                    player=f"{team} D/ST",
                    scores=scores,
                )
            )
    if len(rows) != 24:
        raise RuntimeError(f"Expected 24 Week 1 D/ST rows, found {len(rows)}")
    shapes = {row.scores.shape for row in rows}
    if len(shapes) != 1:
        raise RuntimeError("D/ST rows do not share one correlated world dimension")
    return rows
