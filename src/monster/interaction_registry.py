from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml

from monster.registry import FeatureRegistry


@dataclass(frozen=True)
class InteractionSpec:
    feature_name: str
    entity_scopes: tuple[str, ...]
    interaction_scales: tuple[str, ...]
    phases: tuple[str, ...]
    mechanisms: tuple[str, ...]
    audit_only: bool = False


class InteractionRegistry:
    """Declare where governed evidence is allowed to participate in football reality.

    The feature registry answers "what is this evidence and how much authority may it
    have?". This registry answers "where in football is it allowed to matter?". Keeping
    those questions separate lets Monster add new metrics without turning the simulator
    into an unrestricted weighted feature soup.
    """

    def __init__(self, specs: dict[str, InteractionSpec], raw_text: str):
        self.specs = specs
        self.raw_text = raw_text

    @classmethod
    def load(
        cls,
        path: str | Path = "config/interaction_registry.yaml",
        *,
        feature_registry: FeatureRegistry | None = None,
    ) -> InteractionRegistry:
        path = Path(path)
        raw_text = path.read_text()
        raw = yaml.safe_load(raw_text) or {}
        specs: dict[str, InteractionSpec] = {}
        for feature_name, cfg in raw.get("interactions", {}).items():
            specs[feature_name] = InteractionSpec(
                feature_name=feature_name,
                entity_scopes=tuple(cfg.get("entity_scopes", [])),
                interaction_scales=tuple(cfg.get("interaction_scales", [])),
                phases=tuple(cfg.get("phases", [])),
                mechanisms=tuple(cfg.get("mechanisms", [])),
                audit_only=bool(cfg.get("audit_only", False)),
            )

        if feature_registry is not None:
            unknown = sorted(set(specs).difference(feature_registry.specs))
            if unknown:
                raise ValueError(
                    "Interaction registry references unknown governed features: "
                    f"{unknown}"
                )

        return cls(specs, raw_text)

    def hash(self) -> str:
        return hashlib.sha256(self.raw_text.encode()).hexdigest()

    def for_mechanism(self, mechanism: str) -> tuple[InteractionSpec, ...]:
        return tuple(spec for spec in self.specs.values() if mechanism in spec.mechanisms)
