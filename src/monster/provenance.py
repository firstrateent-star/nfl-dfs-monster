from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(path: str | Path, payload: dict) -> None:
    payload = dict(payload)
    payload.setdefault("created_at", datetime.now(UTC).isoformat())
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True))
