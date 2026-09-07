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
        registry_paths = [path]
        fragment_dir = path.parent / "feature_registry.d"
        if fragment_dir.exists():
            registry_paths.extend(sorted(fragment_dir.glob("*.yaml")))

        specs: dict[str, FeatureSpec] = {}
        raw_parts: list[str] = []
        for registry_path in registry_paths:
            raw_text = registry_path.read_text()
            raw_parts.append(f"# {registry_path}\n{raw_text}")
            raw = yaml.safe_load(raw_text) or {}
            for name, cfg in raw.get("features", {}).items():
                if name in specs:
                    raise ValueError(f"Duplicate feature registry entry: {name}")
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
        return cls(specs, "\n".join(raw_parts))

    def hash(self) -> str:
        return hashlib.sha256(self.raw_text.encode()).hexdigest()

    def blind_allowed(self) -> set[str]:
        return {n for n, s in self.specs.items() if not s.market_derived}

    def forbidden_upstream(self) -> set[str]:
        return {n for n, s in self.specs.items() if s.market_derived}
