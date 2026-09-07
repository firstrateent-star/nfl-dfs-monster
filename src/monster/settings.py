from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os

@dataclass(frozen=True)
class Settings:
    database_url: str | None
    data_dir: Path
    season: int
    week: int
    nws_user_agent: str

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            database_url=os.getenv("DATABASE_URL"),
            data_dir=Path(os.getenv("MONSTER_DATA_DIR", ".monster-data")),
            season=int(os.getenv("MONSTER_SEASON", "2026")),
            week=int(os.getenv("MONSTER_WEEK", "1")),
            nws_user_agent=os.getenv("NWS_USER_AGENT", "nfl-dfs-monster/0.1 (contact: unset)"),
        )
