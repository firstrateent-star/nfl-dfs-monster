from __future__ import annotations

import json
import sys
from pathlib import Path

import build_league_baseline as baseline

from monster.ingest.madden_canonical import canonicalize_madden_attribute_mirror
from monster.ingest.madden_players import MADDEN27_RATINGS_URL, load_madden27_player_ratings


def _load_rich_ea_attributes():
    return canonicalize_madden_attribute_mirror(load_madden27_player_ratings())


def _output_path() -> Path:
    if "--out" in sys.argv:
        index = sys.argv.index("--out")
        if index + 1 < len(sys.argv):
            return Path(sys.argv[index + 1])
    return Path("artifacts/league-baseline")


def _verify_and_record_provenance(out: Path) -> None:
    path = out / "manifest.json"
    manifest = json.loads(path.read_text())
    traits = manifest.get("trait_coverage", {})
    defense = manifest.get("madden_defense_special_coverage", {})

    required = {
        "skill_speed": int(traits.get("madden_speed", 0)),
        "skill_route": int(traits.get("madden_route_running", 0)),
        "skill_catching": int(traits.get("madden_catching", 0)),
        "ol_blocking": int(manifest.get("ol_players_with_madden_blocking", 0)),
        "defense_pass_rush": int(defense.get("pass_rush", 0)),
        "defense_coverage": int(defense.get("coverage", 0)),
        "defense_tackle": int(defense.get("tackle", 0)),
    }
    missing = {name: value for name, value in required.items() if value <= 0}
    if missing:
        raise RuntimeError(
            "Rich Madden identity matched but numeric mechanism coverage remained disconnected: "
            f"{missing}"
        )

    manifest["madden_source"] = "ea_ratings_public_attribute_mirror"
    manifest["madden_source_transport"] = MADDEN27_RATINGS_URL
    manifest["madden_attribute_boundary_verified"] = True
    manifest["madden_numeric_mechanism_coverage"] = required
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"rich_madden_boundary": required}, indent=2))


def main() -> None:
    # build_league_baseline owns the canonical personnel/history pipeline. Replace only the
    # ratings transport for this experiment: the mirror is an extraction of EA player-detail
    # attributes and is canonicalized before the existing identity/adapter pipeline sees it.
    baseline.load_official_madden27_player_ratings = _load_rich_ea_attributes
    baseline.main()
    _verify_and_record_provenance(_output_path())


if __name__ == "__main__":
    main()
