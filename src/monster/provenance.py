from __future__ import annotations
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(path: str | Path, payload: dict) -> None:
    payload = dict(payload)
    payload.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True))
