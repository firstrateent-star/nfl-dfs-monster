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
        registry_paths = [path]
        fragment_dir = path.parent / "interaction_registry.d"
        if fragment_dir.exists():
            registry_paths.extend(sorted(fragment_dir.glob("*.yaml")))

        specs: dict[str, InteractionSpec] = {}
        raw_parts: list[str] = []
        for registry_path in registry_paths:
            raw_text = registry_path.read_text()
            raw_parts.append(f"# {registry_path}\n{raw_text}")
            raw = yaml.safe_load(raw_text) or {}
            for feature_name, cfg in raw.get("interactions", {}).items():
                if feature_name in specs:
                    raise ValueError(
                        f"Duplicate interaction registry entry: {feature_name}"
                    )
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

        return cls(specs, "\n".join(raw_parts))

    def hash(self) -> str:
        return hashlib.sha256(self.raw_text.encode()).hexdigest()

    def for_mechanism(self, mechanism: str) -> tuple[InteractionSpec, ...]:
        return tuple(spec for spec in self.specs.values() if mechanism in spec.mechanisms)
