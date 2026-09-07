from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    database_url: str | None
    cache_dir: Path
    artifact_dir: Path
    nws_user_agent: str

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            database_url=os.getenv("DATABASE_URL"),
            cache_dir=Path(os.getenv("MONSTER_CACHE_DIR", ".cache/monster")),
            artifact_dir=Path(os.getenv("MONSTER_ARTIFACT_DIR", "artifacts")),
            nws_user_agent=os.getenv(
                "NWS_USER_AGENT", "nfl-dfs-monster/0.1 (contact: local-user)"
            ),
        )
