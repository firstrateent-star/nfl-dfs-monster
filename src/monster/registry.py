from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import yaml

@dataclass(frozen=True)
class FeatureSpec:
    name: str
    family: str
    cadence: str
    jurisdiction: tuple[str, ...]
    authority: float
    market_derived: bool
    production_weight: float | None = None
    promotion_gate: str | None = None

class FeatureRegistry:
    def __init__(self, specs: dict[str, FeatureSpec], source_path: Path):
        self.specs = specs
        self.source_path = source_path

    @classmethod
    def load(cls, path: str | Path = "config/feature_registry.yaml") -> "FeatureRegistry":
        path = Path(path)
        raw = yaml.safe_load(path.read_text())
        specs: dict[str, FeatureSpec] = {}
        for name, cfg in raw["features"].items():
            specs[name] = FeatureSpec(
                name=name,
                family=cfg["family"],
                cadence=cfg.get("cadence", "unknown"),
                jurisdiction=tuple(cfg.get("jurisdiction", [])),
                authority=float(cfg.get("authority", 0.0)),
                market_derived=bool(cfg.get("market_derived", False)),
                production_weight=(None if cfg.get("production_weight") is None else float(cfg["production_weight"])),
                promotion_gate=cfg.get("promotion_gate"),
            )
        return cls(specs, path)

    def blind_allowed(self) -> set[str]:
        return {name for name, spec in self.specs.items() if not spec.market_derived}

    def forbidden_upstream(self) -> set[str]:
        return {name for name, spec in self.specs.items() if spec.market_derived}

    def hash(self) -> str:
        return hashlib.sha256(self.source_path.read_bytes()).hexdigest()
