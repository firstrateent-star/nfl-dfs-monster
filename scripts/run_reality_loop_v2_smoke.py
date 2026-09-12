from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import polars as pl
import run_week1_v13_dispersion_test as runner

from monster.sim import matchup_kernel, play_kernel, resolution_ecology
from monster.sim.clock_ecology_v2 import sample_snap_cadence_v2
from monster.sim.full_world_telemetry_v2 import FullWorldTelemetryV2
from monster.sim.game_flow_lookup import build_team_game_flow_policy
from monster.sim.pass_resolution_bands_v2 import sample_yac_v2
from monster.sim.resolution_bands_v2 import resolve_run_contact_v2, resolve_run_ecology_v2
from monster.sim.snap_ecology_v2 import resolve_pass_snap_v2, resolve_run_snap_v2

_NATIVE_ATTACH_INTENT = runner.integrated._attach_historical_intent_ecology
_NATIVE_SIMULATE_GAME = runner.integrated.simulate_game
_TELEMETRY = FullWorldTelemetryV2()


def _attach_reality_loop_policy(teams: dict, policy_dir: Path) -> dict:
    """Attach intent/resolution priors and the same contextual game-flow policy used by audits.

    The old full-game path attached depth/run intent ecology but left game_flow_policy unset,
    while the causal audit attached both. That meant two nominally identical Reality Loop
    simulations could visit different score/down/distance state distributions. V2 makes the
    full-game path canonical: one historical policy stack for pass-vs-run choice *and* for
    within-play intent, with current team/player identity still providing bounded deviations.
    """
    enriched = _NATIVE_ATTACH_INTENT(teams, policy_dir)
    league_path = policy_dir / "game_flow_league.parquet"
    team_path = policy_dir / "game_flow_team.parquet"
    if not league_path.exists() or not team_path.exists():
        raise FileNotFoundError(
            "Reality Loop v2 requires game-flow policy inputs alongside intent ecology: "
            f"{league_path}, {team_path}"
        )
    league_rows = pl.read_parquet(league_path).to_dicts()
    team_rows = pl.read_parquet(team_path).to_dicts()
    return {
        team_id: replace(
            team,
            game_flow_policy=build_team_game_flow_policy(
                team_id=team_id,
                league_rows=league_rows,
                team_rows=team_rows,
                team_neutral_rate=team.neutral_pass_rate,
                league_neutral_rate=team.league_neutral_pass_rate,
            ),
        )
        for team_id, team in enriched.items()
    }


def _simulate_game_with_telemetry(*args, **kwargs):
    result = _NATIVE_SIMULATE_GAME(*args, **kwargs)
    _TELEMETRY.capture(result)
    return result


def _first_out_path() -> Path:
    if "--first-out" in sys.argv:
        index = sys.argv.index("--first-out")
        if index + 1 < len(sys.argv):
            return Path(sys.argv[index + 1])
    return Path("artifacts/reality-loop-v2-smoke")


def configure_reality_loop_v2() -> None:
    """Activate the Reality Loop v2 causal seams without mutating the stable v1.3 defaults."""

    # Historical outcome families provide the NFL center. Team/player/Madden matchup evidence
    # should explain deviations around that center, not re-apply average league difficulty.
    resolution_ecology._COARSE_MATCHUP_AUTHORITY = 0.35
    resolution_ecology._COARSE_RUN_MATCHUP_AUTHORITY = 0.25
    runner._PASS_MATCHUP_AUTHORITY_OVERRIDE = 0.35

    # Full-game and audit worlds must traverse the same contextual play-calling policy. This
    # eliminates a hidden experiment-path difference before any further completion tuning.
    runner.integrated._attach_historical_intent_ecology = _attach_reality_loop_policy

    # Capture topology from these exact worlds before the dispersion runner adds its chaos
    # wrapper. runner.main() will treat this function as its native game simulator and then add
    # chaos around it, so the recorded plays are the same objects used for score/stat/DFS output.
    runner.integrated.simulate_game = _simulate_game_with_telemetry

    # Role/participation-aware snap ecology preserves individual skill while removing the old
    # selection bias where the strongest defensive players effectively participated in every
    # relevant interaction.
    matchup_kernel.resolve_pass_snap = resolve_pass_snap_v2
    matchup_kernel.resolve_run_snap = resolve_run_snap_v2

    # Preserve designed-run failure/explosive branches while restoring empirical routine bands,
    # give QB scrambles their own escape topology, and use the already-shadow-gated shallow-pass
    # gain-band sampler. Completion probability remains owned by the throw resolver; this changes
    # only the yards topology after a shallow catch.
    resolution_ecology.resolve_run_ecology = resolve_run_ecology_v2
    resolution_ecology.sample_yac = sample_yac_v2
    play_kernel.resolve_run_contact = resolve_run_contact_v2

    # Drive conversion/survival is already close to NFL reality; low play volume was therefore
    # a clock ecology problem rather than an invitation to inflate offensive success.
    play_kernel._sample_snap_cadence = sample_snap_cadence_v2


def main() -> None:
    configure_reality_loop_v2()
    runner.main()
    _TELEMETRY.write(_first_out_path())


if __name__ == "__main__":
    main()
