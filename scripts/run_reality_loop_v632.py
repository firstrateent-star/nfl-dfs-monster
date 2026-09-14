from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import polars as pl
import run_reality_loop_v2_smoke as v61
import run_reality_loop_v63 as v63

from monster.sim import game_loop_v13
from monster.sim.chaos_ecology import ReturnKind
from monster.sim.return_channels_v632 import (
    DEFAULT_CHANNEL_PRIORS_V632,
    ChannelReturnPriorsV632,
    channel_priors_from_policy_row,
    resolve_turnover_return_v632,
    simulate_kickoff_v632,
    simulate_punt_v632,
)

_NATIVE_LOAD_CHAOS = v61.runner._load_chaos_ecology
_ACTIVE_CHANNEL_PRIORS: ChannelReturnPriorsV632 = DEFAULT_CHANNEL_PRIORS_V632


class ReturnChannelTelemetryV632:
    def __init__(self) -> None:
        self.games = 0
        self.return_events: Counter[str] = Counter()
        self.return_tds: Counter[str] = Counter()
        self.special_events: Counter[str] = Counter()

    def capture(self, result) -> None:
        self.games += 1
        for event in result.return_events:
            kind = str(event.kind)
            self.return_events[kind] += 1
            if event.touchdown:
                self.return_tds[kind] += 1
        for event in result.special_teams_events:
            event_type = str(event.event_type)
            self.special_events[f"{event_type}:events"] += 1
            self.special_events[f"{event_type}:touchbacks"] += int(bool(event.touchback))
            self.special_events[f"{event_type}:fair_catches"] += int(bool(event.fair_catch))
            self.special_events[f"{event_type}:muffs"] += int(bool(event.muffed))
            self.special_events[f"{event_type}:blocked"] += int(bool(event.blocked))
            self.special_events[f"{event_type}:return_tds"] += int(bool(event.return_touchdown))

    def write(self, out: Path) -> None:
        games = max(self.games, 1)
        payload = {
            "games": self.games,
            "return_events": dict(self.return_events),
            "return_tds": dict(self.return_tds),
            "return_events_per_game": {
                key: value / games for key, value in sorted(self.return_events.items())
            },
            "return_tds_per_game": {
                key: value / games for key, value in sorted(self.return_tds.items())
            },
            "special_teams": dict(self.special_events),
            "special_teams_per_game": {
                key: value / games for key, value in sorted(self.special_events.items())
            },
            "principle": (
                "Return selection and conditional live-return distance are audited separately; "
                "touchdowns remain a consequence of sampled distance crossing live field geometry."
            ),
        }
        out.mkdir(parents=True, exist_ok=True)
        (out / "return_channel_telemetry_v632.json").write_text(
            json.dumps(payload, indent=2) + "\n"
        )


_RETURN_CHANNEL_TELEMETRY = ReturnChannelTelemetryV632()


def _load_chaos_ecology_v632():
    global _ACTIVE_CHANNEL_PRIORS
    ecology = _NATIVE_LOAD_CHAOS()
    player_usage = v61.runner._argument_path(
        "--player-usage", "artifacts/league-policy/player_usage.parquet"
    )
    path = player_usage.parent / "chaos_ecology.parquet"
    if path.exists():
        rows = pl.read_parquet(path).to_dicts()
        _ACTIVE_CHANNEL_PRIORS = channel_priors_from_policy_row(rows[0] if rows else None)
    else:
        _ACTIVE_CHANNEL_PRIORS = DEFAULT_CHANNEL_PRIORS_V632
    return ecology


def configure_reality_loop_v632() -> None:
    """Layer channel-correct return semantics over the protected v6.3 football world."""

    v63.configure_reality_loop_v63()
    v61.runner._load_chaos_ecology = _load_chaos_ecology_v632

    def simulate_punt_channel(*args, **kwargs):
        kwargs["priors"] = _ACTIVE_CHANNEL_PRIORS
        return simulate_punt_v632(*args, **kwargs)

    def simulate_kickoff_channel(*args, **kwargs):
        kwargs["priors"] = _ACTIVE_CHANNEL_PRIORS
        return simulate_kickoff_v632(*args, **kwargs)

    def resolve_turnover_channel(*args, **kwargs):
        kwargs["priors"] = _ACTIVE_CHANNEL_PRIORS
        return resolve_turnover_return_v632(*args, **kwargs)

    # game_loop_v13 imported these callables by name, so production authority must be patched
    # at that module seam rather than only changing special_teams_v13 exports.
    game_loop_v13.simulate_punt = simulate_punt_channel
    game_loop_v13.simulate_kickoff = simulate_kickoff_channel
    game_loop_v13.resolve_turnover_return = resolve_turnover_channel

    native_simulate_game = v61.runner.integrated.simulate_game

    def simulate_game_with_channel_telemetry(*args, **kwargs):
        result = native_simulate_game(*args, **kwargs)
        _RETURN_CHANNEL_TELEMETRY.capture(result)
        return result

    v61.runner.integrated.simulate_game = simulate_game_with_channel_telemetry


def _record_v632_manifest(out: Path) -> None:
    manifest_path = out / "manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text())
    manifest.update(
        {
            "reality_loop_v632_active": True,
            "v63_protected_control_inherited": True,
            "fumble_recovery_specific_yards_active": True,
            "punt_return_selection_separated_from_distance": True,
            "kickoff_return_selection_separated_from_distance": True,
            "explicit_kickoff_touchback_prior_active": True,
            "conditional_live_return_distance_priors_active": True,
            "direct_return_touchdown_probability": False,
            "v632_principle": (
                "First decide whether the ball is actually returnable and returned. Only then "
                "sample distance from that live-return population. Fumble returns use recovery "
                "yards. A touchdown exists only when the resulting return traverses the live field."
            ),
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    original = v61.configure_reality_loop_v2
    original_loader = v61.runner._load_chaos_ecology
    try:
        v61.configure_reality_loop_v2 = configure_reality_loop_v632
        v61.main()
    finally:
        v61.configure_reality_loop_v2 = original
        v61.runner._load_chaos_ecology = original_loader
    out = v61._first_out_path()
    _RETURN_CHANNEL_TELEMETRY.write(out)
    _record_v632_manifest(out)


if __name__ == "__main__":
    main()
