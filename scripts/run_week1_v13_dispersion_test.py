from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import polars as pl

import run_week1_v13_integrated as integrated
from monster.sim import resolution_ecology
from monster.sim.chaos_ecology import DEFAULT_CHAOS_ECOLOGY, from_policy_row
from monster.sim.dispersion_bridge import enhanced_defensive_unit, enhanced_team_identity

_WORLD_WEIRDNESS: list[dict] = []
# Shadow architectures may supply an explicitly gated authority without changing the stable
# v1.3 default. None preserves the historical dispersion-test behavior exactly.
_PASS_MATCHUP_AUTHORITY_OVERRIDE: float | None = None


def _argument_path(flag: str, default: str) -> Path:
    if flag in sys.argv:
        index = sys.argv.index(flag)
        if index + 1 < len(sys.argv):
            return Path(sys.argv[index + 1])
    return Path(default)


def _load_chaos_ecology():
    player_usage = _argument_path("--player-usage", "artifacts/league-policy/player_usage.parquet")
    path = player_usage.parent / "chaos_ecology.parquet"
    if not path.exists():
        return DEFAULT_CHAOS_ECOLOGY
    rows = pl.read_parquet(path).to_dicts()
    return from_policy_row(rows[0] if rows else None)


def _record_experiment() -> None:
    first_out = _argument_path("--first-out", "artifacts/week1-v13-first-sim")
    first_out.mkdir(parents=True, exist_ok=True)
    manifest_path = first_out / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest.update(
            {
                "dispersion_architecture_test": True,
                "rich_player_capability_bridge_active": True,
                "team_offense_multidimensional_identity_active": True,
                "team_defense_context_active": True,
                "primary_receiver_rotation_active": True,
                "primary_receiver_rotation_max_players": 6,
                "pass_matchup_relative_authority": resolution_ecology._COARSE_MATCHUP_AUTHORITY,
                "field_position_chaos_ecology_active": True,
                "turnover_return_geometry_active": True,
                "defensive_touchdowns_active": True,
                "special_teams_return_touchdowns_active": True,
                "muffs_and_blocked_kick_returns_active": True,
                "opening_and_halftime_kickoffs_simulated": True,
                "2026_dynamic_kickoff_touchback_yardline": 35,
                "football_weirdness_audit_active": True,
                "scoreboard_conservation_includes_return_touchdowns": True,
                "score_dispersion_not_directly_calibrated": True,
                "principle_dispersion": (
                    "Totals, margins and rare tails must emerge from offense, defense, personnel, "
                    "field position, turnover geometry, returns and matchup differences; no target "
                    "game scores or sportsbook inputs are used."
                ),
            }
        )
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    if _WORLD_WEIRDNESS:
        worlds = pl.DataFrame(_WORLD_WEIRDNESS)
        worlds.write_csv(first_out / "football_weirdness_worlds.csv")
        metric_columns = [column for column in worlds.columns if column != "game"]
        aggregate_expressions = []
        for column in metric_columns:
            aggregate_expressions.extend(
                [
                    pl.col(column).mean().alias(f"{column}_mean"),
                    pl.col(column).std(ddof=1).alias(f"{column}_sd"),
                    pl.col(column).quantile(0.10).alias(f"{column}_p10"),
                    pl.col(column).quantile(0.50).alias(f"{column}_p50"),
                    pl.col(column).quantile(0.90).alias(f"{column}_p90"),
                ]
            )
        worlds.group_by("game").agg(*aggregate_expressions).sort("game").write_csv(
            first_out / "football_weirdness.csv"
        )


def main() -> None:
    # Stable v1.3 uses 0.50. Shadow architectures can supply an explicit, separately tested
    # authority so a wrapper cannot be silently overwritten here after configuration.
    resolution_ecology._COARSE_MATCHUP_AUTHORITY = (
        0.50
        if _PASS_MATCHUP_AUTHORITY_OVERRIDE is None
        else float(_PASS_MATCHUP_AUTHORITY_OVERRIDE)
    )
    integrated._team_identity = enhanced_team_identity
    integrated._defensive_unit = enhanced_defensive_unit

    chaos = _load_chaos_ecology()
    native_simulate_game = integrated.simulate_game
    native_summarize_game = integrated.summarize_game

    def simulate_game_with_chaos(*args, **kwargs):
        kwargs.setdefault("chaos_ecology", chaos)
        return native_simulate_game(*args, **kwargs)

    def summarize_and_capture(result):
        summary = native_summarize_game(result)
        game = f"{result.final_state.away_team_id}@{result.final_state.home_team_id}"
        _WORLD_WEIRDNESS.append({"game": game, **asdict(summary)})
        return summary

    integrated.simulate_game = simulate_game_with_chaos
    integrated.summarize_game = summarize_and_capture
    integrated.main()
    _record_experiment()


if __name__ == "__main__":
    main()
