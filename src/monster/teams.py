from __future__ import annotations

NFL_TEAMS: dict[str, str] = {
    "ARI": "Arizona Cardinals",
    "ATL": "Atlanta Falcons",
    "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers",
    "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals",
    "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos",
    "DET": "Detroit Lions",
    "GB": "Green Bay Packers",
    "HOU": "Houston Texans",
    "IND": "Indianapolis Colts",
    "JAC": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs",
    "LAC": "Los Angeles Chargers",
    "LAR": "Los Angeles Rams",
    "LV": "Las Vegas Raiders",
    "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings",
    "NE": "New England Patriots",
    "NO": "New Orleans Saints",
    "NYG": "New York Giants",
    "NYJ": "New York Jets",
    "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks",
    "SF": "San Francisco 49ers",
    "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans",
    "WAS": "Washington Commanders",
}

# Source-specific or historical aliases. Canonical IDs above are what the Monster stores.
TEAM_ALIASES: dict[str, str] = {
    "JAX": "JAC",
    "LA": "LAR",
    "ARZ": "ARI",
    "CLV": "CLE",
    "GNB": "GB",
    "KAN": "KC",
    "NWE": "NE",
    "NOR": "NO",
    "SFO": "SF",
    "TAM": "TB",
    "LVR": "LV",
    "OAK": "LV",
    "SD": "LAC",
    "STL": "LAR",
    "WSH": "WAS",
}


def normalize_team_id(team_id: str | None) -> str | None:
    """Return one canonical Monster team ID across nflverse/PFR/DFS/history sources."""
    if team_id is None:
        return None
    cleaned = team_id.strip().upper()
    canonical = TEAM_ALIASES.get(cleaned, cleaned)
    if canonical not in NFL_TEAMS:
        raise ValueError(f"Unknown NFL team identifier: {team_id!r}")
    return canonical


def is_current_nfl_team(team_id: str | None) -> bool:
    try:
        return normalize_team_id(team_id) in NFL_TEAMS
    except ValueError:
        return False
