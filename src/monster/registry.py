from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    family: str
    cadence: str
    jurisdiction: tuple[str, ...]
    authority: float
    production_weight: float
    market_derived: bool
    promotion_gate: str | None = None


class FeatureRegistry:
    def __init__(self, specs: dict[str, FeatureSpec], raw_text: str):
        self.specs = specs
        self.raw_text = raw_text

    @classmethod
    def load(cls, path: str | Path = "config/feature_registry.yaml") -> FeatureRegistry:
        path = Path(path)
        raw_text = path.read_text()
        raw = yaml.safe_load(raw_text)
        specs = {}
        for name, cfg in raw["features"].items():
            specs[name] = FeatureSpec(
                name=name,
                family=cfg["family"],
                cadence=cfg["cadence"],
                jurisdiction=tuple(cfg.get("jurisdiction", [])),
                authority=float(cfg.get("authority", 0.0)),
                production_weight=float(cfg.get("production_weight", 1.0)),
                market_derived=bool(cfg.get("market_derived", False)),
                promotion_gate=cfg.get("promotion_gate"),
            )
        return cls(specs, raw_text)

    def hash(self) -> str:
        return hashlib.sha256(self.raw_text.encode()).hexdigest()

    def blind_allowed(self) -> set[str]:
        return {n for n, s in self.specs.items() if not s.market_derived}

    def forbidden_upstream(self) -> set[str]:
        return {n for n, s in self.specs.items() if s.market_derived}
