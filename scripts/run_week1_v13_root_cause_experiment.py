from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import polars as pl

import run_week1_v13_integrated as integrated
from monster.feature_compile.health_pools import apply_health_to_skill_pools
from monster.feature_compile.league_units import compile_league_unit_player_map
from monster.feature_compile.reality_inputs import compile_player_reality_inputs
from monster.feature_compile.skill_pools import compile_current_skill_pools
from monster.sim import game_loop_v13 as game_loop
from monster.sim.chaos_ecology import DEFAULT_CHAOS_ECOLOGY, ChaosEcology, ReturnEvent, from_policy_row
from monster.sim.dispersion_bridge import enhanced_defensive_unit, enhanced_team_identity
from monster.sim.event_ledger import assert_event_conservation, summarize_game
from monster.sim.rushing_roles import sample_event_rush_share_plan
from monster.sim.special_teams_v13 import SpecialTeamsEvent
from monster.snapshot.league import compile_team_state_map


BRANCHES = (
    "B_shared_suppressed",
    "C_shared_applied",
    "D_isolated_applied",
    "E_isolated_suppressed",
)
PAIR_COMPARISONS = (
    ("B_shared_suppressed", "C_shared_applied"),
    ("D_isolated_applied", "E_isolated_suppressed"),
)


@dataclass
class RngStreams:
    """Deterministic experiment-only RNG namespaces.

    This is intentionally a Lab implementation, not yet a production contract. The experiment
    asks whether hidden shared mutable randomness is part of the observed football regression.
    Promotion into the runtime requires evidence from this experiment.
    """

    snap: np.random.Generator
    penalty: np.random.Generator
    special: np.random.Generator
    turnover_return: np.random.Generator
    try_event: np.random.Generator
    counterfactual: np.random.Generator

    @classmethod
    def from_seed(cls, seed: int) -> "RngStreams":
        children = np.random.SeedSequence(seed).spawn(6)
        return cls(*(np.random.default_rng(child) for child in children))


@dataclass(frozen=True)
class BranchConfig:
    isolated_rng: bool
    apply_chaos_consequences: bool


CONFIGS = {
    "B_shared_suppressed": BranchConfig(False, False),
    "C_shared_applied": BranchConfig(False, True),
    "D_isolated_applied": BranchConfig(True, True),
    "E_isolated_suppressed": BranchConfig(True, False),
}


_NATIVE = {
    "simulate_scrimmage_play": game_loop.simulate_scrimmage_play,
    "simulate_penalty": game_loop.simulate_penalty,
    "simulate_try": game_loop.simulate_try,
    "simulate_kickoff": game_loop.simulate_kickoff,
    "simulate_punt": game_loop.simulate_punt,
    "simulate_field_goal": game_loop.simulate_field_goal,
    "resolve_turnover_return": game_loop.resolve_turnover_return,
    "sample_return_yards": game_loop.sample_return_yards,
}


def _load_chaos_ecology(player_usage: Path) -> ChaosEcology:
    path = player_usage.parent / "chaos_ecology.parquet"
    if not path.exists():
        return DEFAULT_CHAOS_ECOLOGY
    rows = pl.read_parquet(path).to_dicts()
    return from_policy_row(rows[0] if rows else None)


def _rng(config: BranchConfig, streams: RngStreams, name: str, passed: np.random.Generator) -> np.random.Generator:
    return getattr(streams, name) if config.isolated_rng else passed


def _neutral_turnover_return(event: ReturnEvent) -> ReturnEvent:
    if event.touchback:
        return replace(event, return_yards=0.0, touchdown=False)
    receiving = float(np.clip(100.0 - event.change_spot_yardline_100, 1.0, 99.0))
    return replace(
        event,
        return_yards=0.0,
        receiving_yardline_100=receiving,
        touchdown=False,
    )


def _counterfactual_unblocked_field_goal(
    *,
    distance: float,
    kicking_skill: float,
    kicker_id: str | None,
    ecology: ChaosEcology,
    rng: np.random.Generator,
) -> SpecialTeamsEvent:
    no_block = replace(ecology, blocked_field_goal_rate=0.0)
    return _NATIVE["simulate_field_goal"](
        rng,
        distance=distance,
        kicking_skill=kicking_skill,
        kicker_id=kicker_id,
        ecology=no_block,
    )


@contextmanager
def _experiment_runtime(
    config: BranchConfig,
    *,
    streams: RngStreams,
    ecology: ChaosEcology,
    audit: defaultdict[str, float],
) -> Iterator[None]:
    """Route stochastic jurisdictions and optionally suppress sampled chaos consequences.

    Both applied/suppressed siblings still *sample* the same chaos events. Suppressed branches
    neutralize their football consequences after sampling, so shared-RNG B/C and isolated-RNG
    D/E are paired experiments rather than unrelated reruns.
    """

    def scrimmage(state, offense, defense_strength, rng, *, defense=None):
        chosen = _rng(config, streams, "snap", rng)
        return _NATIVE["simulate_scrimmage_play"](
            state, offense, defense_strength, chosen, defense=defense
        )

    def penalty(rng, *, base_rate=0.055):
        chosen = _rng(config, streams, "penalty", rng)
        return _NATIVE["simulate_penalty"](chosen, base_rate=base_rate)

    def try_event(
        rng,
        *,
        go_for_two,
        kicking_skill=1.0,
        offense_skill=1.0,
        defense_skill=1.0,
    ):
        chosen = _rng(config, streams, "try_event", rng)
        return _NATIVE["simulate_try"](
            chosen,
            go_for_two=go_for_two,
            kicking_skill=kicking_skill,
            offense_skill=offense_skill,
            defense_skill=defense_skill,
        )

    def kickoff(
        rng,
        *,
        returner_id=None,
        kicker_id=None,
        return_skill=1.0,
        ecology=ecology,
    ):
        chosen = _rng(config, streams, "special", rng)
        sampled = _NATIVE["simulate_kickoff"](
            chosen,
            returner_id=returner_id,
            kicker_id=kicker_id,
            return_skill=return_skill,
            ecology=ecology,
        )
        audit["sampled_kickoffs"] += 1
        audit["latent_kickoff_muffs"] += int(sampled.muffed)
        audit["latent_kickoff_40_plus"] += int(sampled.return_yards >= 40.0)
        if config.apply_chaos_consequences:
            return sampled
        # Preserve the sampling draw but neutralize return/muff consequences to a 2026 touchback.
        return replace(
            sampled,
            return_yards=0.0,
            touchback=True,
            muffed=False,
            kicking_team_recovery=False,
            return_touchdown=False,
            return_start_yardline_100=None,
        )

    def punt(
        rng,
        *,
        punter_skill=1.0,
        returner_id=None,
        punter_id=None,
        return_skill=1.0,
        ecology=ecology,
    ):
        chosen = _rng(config, streams, "special", rng)
        sampled = _NATIVE["simulate_punt"](
            chosen,
            punter_skill=punter_skill,
            returner_id=returner_id,
            punter_id=punter_id,
            return_skill=return_skill,
            ecology=ecology,
        )
        audit["sampled_punts"] += 1
        audit["latent_blocked_punts"] += int(sampled.blocked)
        audit["latent_punt_muffs"] += int(sampled.muffed)
        audit["latent_punt_40_plus"] += int(sampled.return_yards >= 40.0)
        if config.apply_chaos_consequences:
            return sampled
        gross = sampled.kick_distance
        if sampled.blocked:
            gross = float(np.clip(streams.counterfactual.normal(45.0 * punter_skill, 6.0), 20.0, 70.0))
        return replace(
            sampled,
            kick_distance=gross,
            return_yards=0.0,
            blocked=False,
            muffed=False,
            kicking_team_recovery=False,
            fair_catch=not sampled.touchback,
            return_touchdown=False,
        )

    def field_goal(
        rng,
        *,
        distance,
        kicking_skill=1.0,
        kicker_id=None,
        ecology=ecology,
    ):
        chosen = _rng(config, streams, "special", rng)
        sampled = _NATIVE["simulate_field_goal"](
            chosen,
            distance=distance,
            kicking_skill=kicking_skill,
            kicker_id=kicker_id,
            ecology=ecology,
        )
        audit["sampled_field_goals"] += 1
        audit["latent_blocked_field_goals"] += int(sampled.blocked)
        if config.apply_chaos_consequences or not sampled.blocked:
            return sampled
        return _counterfactual_unblocked_field_goal(
            distance=float(distance),
            kicking_skill=float(kicking_skill),
            kicker_id=kicker_id,
            ecology=ecology,
            rng=streams.counterfactual,
        )

    def turnover_return(before, event, *, defense, rng, ecology=ecology):
        chosen = _rng(config, streams, "turnover_return", rng)
        sampled = _NATIVE["resolve_turnover_return"](
            before,
            event,
            defense=defense,
            rng=chosen,
            ecology=ecology,
        )
        audit["sampled_turnover_returns"] += 1
        audit["latent_turnover_return_yards"] += float(sampled.return_yards)
        audit["latent_turnover_return_tds"] += int(sampled.touchdown)
        audit["latent_turnover_return_40_plus"] += int(sampled.return_yards >= 40.0)
        return sampled if config.apply_chaos_consequences else _neutral_turnover_return(sampled)

    def return_yards(*, mean, sd, zero_rate, forty_plus_rate, return_skill, rng, maximum=100.0):
        chosen = _rng(config, streams, "turnover_return", rng)
        sampled = _NATIVE["sample_return_yards"](
            mean=mean,
            sd=sd,
            zero_rate=zero_rate,
            forty_plus_rate=forty_plus_rate,
            return_skill=return_skill,
            rng=chosen,
            maximum=maximum,
        )
        audit["sampled_special_loose_ball_returns"] += 1
        audit["latent_special_loose_ball_return_yards"] += float(sampled)
        return sampled if config.apply_chaos_consequences else 0.0

    patches = {
        "simulate_scrimmage_play": scrimmage,
        "simulate_penalty": penalty,
        "simulate_try": try_event,
        "simulate_kickoff": kickoff,
        "simulate_punt": punt,
        "simulate_field_goal": field_goal,
        "resolve_turnover_return": turnover_return,
        "sample_return_yards": return_yards,
    }
    for name, value in patches.items():
        setattr(game_loop, name, value)
    try:
        yield
    finally:
        for name, value in _NATIVE.items():
            setattr(game_loop, name, value)


def _common_play_prefix(left, right) -> int:
    limit = min(len(left), len(right))
    for idx in range(limit):
        if left[idx] != right[idx]:
            return idx
    return limit


def _branch_summary(worlds: pl.DataFrame, games: pl.DataFrame) -> pl.DataFrame:
    base = worlds.group_by("branch").agg(
        pl.len().alias("worlds"),
        (pl.col("away_points") + pl.col("home_points")).mean().alias("game_total_mean"),
        (pl.col("away_points") + pl.col("home_points")).std(ddof=1).alias("game_total_sd"),
        (pl.col("away_points") - pl.col("home_points")).abs().mean().alias("absolute_margin_mean"),
        pl.col("scrimmage_plays").mean().alias("scrimmage_plays_mean"),
        pl.col("drives").mean().alias("drives_mean"),
        pl.col("punts").mean().alias("punts_mean"),
        pl.col("sacks").mean().alias("sacks_mean"),
        pl.col("interceptions").mean().alias("interceptions_mean"),
        pl.col("fumbles_lost").mean().alias("fumbles_lost_mean"),
        pl.col("touchdowns").mean().alias("touchdowns_mean"),
        pl.col("offensive_touchdowns").mean().alias("offensive_touchdowns_mean"),
        pl.col("defensive_touchdowns").mean().alias("defensive_touchdowns_mean"),
        pl.col("special_teams_touchdowns").mean().alias("special_teams_touchdowns_mean"),
        pl.col("explosive_40").mean().alias("explosive_40_mean"),
        pl.col("explosive_60").mean().alias("explosive_60_mean"),
        pl.col("short_field_drives").mean().alias("short_field_drives_mean"),
        pl.col("drive_start_yardline_mean").mean().alias("drive_start_yardline_mean"),
        pl.col("turnover_drive_start_yardline_mean")
        .drop_nulls()
        .mean()
        .alias("turnover_drive_start_yardline_mean"),
        pl.col("completions").sum().alias("completions_sum"),
        pl.col("pass_attempts").sum().alias("pass_attempts_sum"),
    ).with_columns(
        (pl.col("completions_sum") / pl.col("pass_attempts_sum")).alias("completion_percentage")
    )
    dispersion = games.group_by("branch").agg(
        pl.col("total_mean").std(ddof=1).alias("between_game_total_mean_sd"),
        pl.col("total_mean").min().alias("lowest_game_total_mean"),
        pl.col("total_mean").max().alias("highest_game_total_mean"),
        pl.col("margin_mean").abs().mean().alias("between_game_absolute_margin_mean"),
    )
    return base.join(dispersion, on="branch", how="left").sort("branch")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--personnel", type=Path, required=True)
    parser.add_argument("--player-usage", type=Path, required=True)
    parser.add_argument("--situation-context", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026190921)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    policy = integrated._read(args.policy)
    personnel = integrated._read(args.personnel)
    usage = integrated._read(args.player_usage)
    situation_context = integrated._read(args.situation_context)
    league_neutral_pass_rate, situational_pass_rates = integrated._situational_context(
        situation_context
    )
    pools = apply_health_to_skill_pools(
        compile_current_skill_pools(personnel, usage, policy=policy), personnel
    )
    reality = compile_player_reality_inputs(personnel, game_date=integrated.GAME_DATE)
    units = compile_league_unit_player_map(personnel)
    states: dict[str, Any] = {}
    for away, home in integrated.MATCHUPS:
        states.update(compile_team_state_map(policy, {away: home, home: away}))

    teams = {
        team: enhanced_team_identity(
            team,
            pools[team],
            reality,
            units[team],
            states[team],
            league_neutral_pass_rate=league_neutral_pass_rate,
            situational_pass_rates=situational_pass_rates,
        )
        for pair in integrated.MATCHUPS
        for team in pair
    }
    teams = integrated._attach_historical_intent_ecology(teams, args.player_usage.parent)
    defenses = {
        team: enhanced_defensive_unit(units[team])
        for pair in integrated.MATCHUPS
        for team in pair
    }
    ecology = _load_chaos_ecology(args.player_usage)

    rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    total_runs = len(integrated.MATCHUPS) * args.worlds * len(BRANCHES)
    completed = 0
    started = time.monotonic()

    for game_idx, (away, home) in enumerate(integrated.MATCHUPS):
        game = f"{away}@{home}"
        print(f"ROOT-CAUSE starting {game}: {args.worlds} paired worlds x {len(BRANCHES)} branches", flush=True)
        for world in range(args.worlds):
            seed = args.seed + game_idx * 1_000_003 + world
            away_plan = sample_event_rush_share_plan(
                pools[away], rng=np.random.default_rng(seed + 101_003)
            )
            home_plan = sample_event_rush_share_plan(
                pools[home], rng=np.random.default_rng(seed + 202_007)
            )
            away_team = integrated._with_event_rush_plan(teams[away], away_plan)
            home_team = integrated._with_event_rush_plan(teams[home], home_plan)
            branch_results = {}
            branch_summaries = {}

            for branch in BRANCHES:
                config = CONFIGS[branch]
                streams = RngStreams.from_seed(seed + 7_000_019)
                audit: defaultdict[str, float] = defaultdict(float)
                with _experiment_runtime(config, streams=streams, ecology=ecology, audit=audit):
                    result = game_loop.simulate_game(
                        away_team,
                        home_team,
                        away_defense=defenses[away],
                        home_defense=defenses[home],
                        seed=seed,
                        chaos_ecology=ecology,
                    )
                assert_event_conservation(result)
                summary = summarize_game(result)
                branch_results[branch] = result
                branch_summaries[branch] = summary
                rows.append(
                    {
                        "branch": branch,
                        "game": game,
                        "world": world,
                        "seed": seed,
                        **asdict(summary),
                        **dict(audit),
                    }
                )
                completed += 1
                if completed % max(args.progress_every, 1) == 0 or completed == total_runs:
                    elapsed = max(time.monotonic() - started, 1e-9)
                    rate = completed / elapsed
                    eta = (total_runs - completed) / rate if rate > 0 else 0.0
                    print(
                        f"ROOT-CAUSE {completed:,}/{total_runs:,} branch-worlds | "
                        f"{game} world {world + 1}/{args.worlds} {branch} | "
                        f"{rate:.2f}/s | ETA {eta / 60:.1f}m",
                        flush=True,
                    )

            for left_name, right_name in PAIR_COMPARISONS:
                left = branch_results[left_name]
                right = branch_results[right_name]
                left_summary = branch_summaries[left_name]
                right_summary = branch_summaries[right_name]
                prefix = _common_play_prefix(left.plays, right.plays)
                paired_rows.append(
                    {
                        "pair": f"{left_name}__vs__{right_name}",
                        "game": game,
                        "world": world,
                        "seed": seed,
                        "common_play_prefix": prefix,
                        "left_play_count": len(left.plays),
                        "right_play_count": len(right.plays),
                        "full_play_sequence_equal": left.plays == right.plays,
                        "game_total_delta_right_minus_left": (
                            right_summary.away_points
                            + right_summary.home_points
                            - left_summary.away_points
                            - left_summary.home_points
                        ),
                        "completion_delta_right_minus_left": (
                            right_summary.completions - left_summary.completions
                        ),
                        "sack_delta_right_minus_left": right_summary.sacks - left_summary.sacks,
                        "punt_delta_right_minus_left": right_summary.punts - left_summary.punts,
                        "drive_delta_right_minus_left": right_summary.drives - left_summary.drives,
                    }
                )

    worlds = pl.DataFrame(rows).fill_null(0)
    paired = pl.DataFrame(paired_rows)
    worlds.write_csv(args.out / "root_cause_worlds.csv")
    paired.write_csv(args.out / "paired_world_deltas.csv")

    game_summary = worlds.group_by(["branch", "game"]).agg(
        pl.len().alias("worlds"),
        (pl.col("away_points") + pl.col("home_points")).mean().alias("total_mean"),
        (pl.col("away_points") + pl.col("home_points")).std(ddof=1).alias("total_sd"),
        (pl.col("away_points") - pl.col("home_points")).mean().alias("margin_mean"),
        pl.col("scrimmage_plays").mean().alias("scrimmage_plays_mean"),
        pl.col("drives").mean().alias("drives_mean"),
        pl.col("punts").mean().alias("punts_mean"),
        pl.col("sacks").mean().alias("sacks_mean"),
        pl.col("touchdowns").mean().alias("touchdowns_mean"),
        pl.col("defensive_touchdowns").mean().alias("defensive_touchdowns_mean"),
        pl.col("special_teams_touchdowns").mean().alias("special_teams_touchdowns_mean"),
        pl.col("explosive_40").mean().alias("explosive_40_mean"),
        pl.col("explosive_60").mean().alias("explosive_60_mean"),
        pl.col("completions").sum().alias("completions_sum"),
        pl.col("pass_attempts").sum().alias("pass_attempts_sum"),
    ).with_columns(
        (pl.col("completions_sum") / pl.col("pass_attempts_sum")).alias("completion_percentage")
    ).sort(["branch", "game"])
    game_summary.write_csv(args.out / "branch_game_summary.csv")
    branch_summary = _branch_summary(worlds, game_summary)
    branch_summary.write_csv(args.out / "branch_summary.csv")

    pair_summary = paired.group_by("pair").agg(
        pl.len().alias("paired_worlds"),
        pl.col("full_play_sequence_equal").mean().alias("full_play_sequence_equal_rate"),
        pl.col("common_play_prefix").mean().alias("common_play_prefix_mean"),
        pl.col("game_total_delta_right_minus_left").mean().alias("game_total_delta_mean"),
        pl.col("game_total_delta_right_minus_left").std(ddof=1).alias("game_total_delta_sd"),
        pl.col("completion_delta_right_minus_left").mean().alias("completion_delta_mean"),
        pl.col("sack_delta_right_minus_left").mean().alias("sack_delta_mean"),
        pl.col("punt_delta_right_minus_left").mean().alias("punt_delta_mean"),
        pl.col("drive_delta_right_minus_left").mean().alias("drive_delta_mean"),
    ).sort("pair")
    pair_summary.write_csv(args.out / "paired_summary.csv")

    manifest = {
        "experiment": "MON-LEDGER-001",
        "center": "maximum causal fidelity to observable NFL reality",
        "question": "Why did ordinary offensive efficiency regress after field-position/chaos ecology while matchup mean environments remained compressed?",
        "worlds_per_game_per_branch": args.worlds,
        "games": len(integrated.MATCHUPS),
        "executed_branches": list(BRANCHES),
        "external_reference_A": {
            "description": "pre-chaos 100-world artifact; intentionally not copied into this run",
            "artifact_id": 10291902121,
            "run_id": 34674818426,
            "commit": "595f2e8b8c725b8cb9135018a7865c97ad9b58b9",
        },
        "branches": {
            "B_shared_suppressed": "current game/state architecture; shared RNG; chaos sampled then consequences neutralized",
            "C_shared_applied": "current candidate behavior; shared RNG; chaos sampled and applied",
            "D_isolated_applied": "current candidate behavior; deterministic isolated RNG jurisdictions; chaos applied",
            "E_isolated_suppressed": "deterministic isolated RNG jurisdictions; same chaos sampled then consequences neutralized",
        },
        "pair_logic": {
            "B_vs_C": "isolates applied chaos consequences under the shared-RNG architecture while preserving sampling draws until state diverges",
            "D_vs_E": "isolates applied chaos consequences under isolated deterministic RNG jurisdictions",
            "C_vs_D": "aggregate comparison isolates the effect of RNG architecture while chaos remains active",
            "B_vs_E": "aggregate comparison observes RNG architecture when chaos consequences are suppressed",
        },
        "rng_jurisdictions": [
            "snap",
            "penalty",
            "special",
            "turnover_return",
            "try_event",
            "counterfactual_lab_only",
        ],
        "source_of_truth": {
            "historical_football": "compiled NFL play-by-play/policy artifacts",
            "current_personnel": str(args.personnel),
            "current_policy": str(args.policy),
            "simulation_behavior": "GitHub code at workflow commit",
            "experiment_outcome": "this artifact",
        },
        "unknowns": [
            "shared RNG contamination contribution",
            "game/state architecture contribution",
            "chaos consequence contribution",
            "remaining between-matchup identity compression after execution regression is isolated",
        ],
        "governance": {
            "stage": "LAB",
            "production_promoted": False,
            "football_coefficients_changed_for_experiment": False,
            "market_inputs_used": False,
            "next_promotion_requires_evidence": True,
        },
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(branch_summary)
    print(pair_summary)


if __name__ == "__main__":
    main()
