from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import polars as pl

import audit_week1_v13_drive_survival as drive
import audit_week1_v13_early_down_anatomy as early
import run_week1_v13_integrated as integrated
import run_week1_v13_root_cause_experiment as root
from monster.sim import play_kernel as pk
from monster.sim import resolution_ecology as resolution
from monster.sim.matchup_kernel import DefensiveIdentity, DefensiveUnit, LEAGUE_PRESSURE_RATE
from monster.sim.play_kernel import PassResult, PlayerIdentity, PlayType, TeamIdentity


MECHANISM_WORLDS_DEFAULT = 100
SENSITIVITY_WORLDS_DEFAULT = 30
SENSITIVITY_BRANCHES = (
    "full",
    "qb_flat",
    "players_flat",
    "team_context_flat",
    "defense_flat",
    "all_flat",
    "players_amplified",
)


def _neutral_player(player: PlayerIdentity) -> PlayerIdentity:
    return replace(player, efficiency=1.0, explosive=1.0, turnover_security=1.0)


def _amplify_player(player: PlayerIdentity, factor: float = 2.0) -> PlayerIdentity:
    def amp(value: float) -> float:
        return float(np.clip(1.0 + factor * (float(value) - 1.0), 0.58, 1.48))

    return replace(
        player,
        efficiency=amp(player.efficiency),
        explosive=amp(player.explosive),
        turnover_security=amp(player.turnover_security),
    )


def _map_team_players(team: TeamIdentity, fn, *, qb_only: bool = False) -> TeamIdentity:
    qb_id = team.quarterback.player_id

    def mapped(player: PlayerIdentity) -> PlayerIdentity:
        if qb_only and player.player_id != qb_id:
            return player
        return fn(player)

    return replace(
        team,
        quarterback=mapped(team.quarterback),
        rushers=tuple(mapped(player) for player in team.rushers),
        receivers=tuple(mapped(player) for player in team.receivers),
    )


def _team_for_branch(team: TeamIdentity, branch: str) -> TeamIdentity:
    out = team
    if branch in {"qb_flat", "all_flat"}:
        out = _map_team_players(out, _neutral_player, qb_only=True)
    if branch in {"players_flat", "all_flat"}:
        out = _map_team_players(out, _neutral_player)
    if branch == "players_amplified":
        out = _map_team_players(out, _amplify_player)
    if branch in {"team_context_flat", "all_flat"}:
        out = replace(
            out,
            pass_efficiency=1.0,
            rush_efficiency=1.0,
            pass_protection=1.0,
            run_blocking=1.0,
        )
    return out


def _neutral_defender(player: DefensiveIdentity) -> DefensiveIdentity:
    return replace(
        player,
        coverage=1.0,
        pass_rush=1.0,
        run_defense=1.0,
        tackling=1.0,
        ball_hawk=1.0,
        speed=1.0,
    )


def _defense_for_branch(defense: DefensiveUnit, branch: str) -> DefensiveUnit:
    if branch not in {"defense_flat", "all_flat"}:
        return defense
    return replace(
        defense,
        front=tuple(_neutral_defender(player) for player in defense.front),
        coverage=tuple(_neutral_defender(player) for player in defense.coverage),
        pressure_rate=LEAGUE_PRESSURE_RATE,
        run_stuff_rate=0.18,
    )


def _score_for_team(result: Any, team_id: str, away: str) -> int:
    return int(result.final_state.away_score if team_id == away else result.final_state.home_score)


def _aggregate_player_stats(result: Any, team: TeamIdentity) -> tuple[float, float, int, int]:
    player_ids = {team.quarterback.player_id, *(p.player_id for p in team.rushers), *(p.player_id for p in team.receivers)}
    passing = 0.0
    rushing = 0.0
    attempts = 0
    completions = 0
    for player_id in player_ids:
        box = result.player_stats.get(player_id)
        if box is None:
            continue
        passing += float(box.passing_yards)
        rushing += float(box.rushing_yards)
        attempts += int(box.pass_attempts)
        completions += int(box.completions)
    return passing, rushing, attempts, completions


@contextmanager
def _mechanism_capture(
    *,
    event_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    qb_rows: list[dict[str, Any]],
    throw_rows: list[dict[str, Any]],
    run_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> Iterator[None]:
    native_scrimmage = root._NATIVE["simulate_scrimmage_play"]
    native_target = pk._field_read_target
    native_qb_response = pk.resolve_qb_response
    native_depth = resolution.depth_throw_probabilities
    native_run = resolution.resolve_run_ecology
    current: dict[str, Any] = {}
    counter = 0

    def row_context() -> dict[str, Any]:
        state = current.get("state")
        offense = current.get("offense")
        return {
            "snap_id": current.get("snap_id", -1),
            "game": metadata.get("game"),
            "world": metadata.get("world"),
            "seed": metadata.get("seed"),
            "offense_team": None if state is None else state.possession,
            "defense_team": None if state is None else state.defense,
            "down": None if state is None else int(state.down),
            "distance": None if state is None else float(state.distance),
            "yardline_100": None if state is None else float(state.yardline_100),
            "early_down": bool(state is not None and state.down in {1, 2}),
            "qb_player_efficiency": None if offense is None else float(offense.quarterback.efficiency),
            "qb_player_explosive": None if offense is None else float(offense.quarterback.explosive),
            "team_pass_efficiency": None if offense is None else float(offense.pass_efficiency),
            "team_rush_efficiency": None if offense is None else float(offense.rush_efficiency),
            "team_pass_protection": None if offense is None else float(offense.pass_protection),
            "team_run_blocking": None if offense is None else float(offense.run_blocking),
        }

    def scrimmage(state, offense, defense_strength, rng, defense=None):
        nonlocal counter
        counter += 1
        current.clear()
        current.update({"snap_id": counter, "state": state, "offense": offense, "defense": defense})
        event = native_scrimmage(state, offense, defense_strength, rng, defense=defense)
        context = row_context()
        context.update(
            {
                "play_type": event.play_type.value,
                "pass_result": None if event.pass_result is None else event.pass_result.value,
                "yards": float(event.yards),
                "pressured": bool(event.pressured),
                "turnover": bool(event.turnover),
                "touchdown": bool(event.touchdown),
                "pass_depth_category": event.pass_depth_category,
                "run_geometry_category": event.run_geometry_category,
                "air_yards": float(event.air_yards),
                "yards_after_catch": float(event.yards_after_catch),
                "yards_before_contact": float(event.yards_before_contact),
                "yards_after_contact": float(event.yards_after_contact),
                "qb_read_quality": float(event.qb_read_quality),
                "fatigue_factor": float(event.fatigue_factor),
            }
        )
        event_rows.append(context)
        current.clear()
        return event

    def target(offense, defense, rng, *, preferred=None, fatigue):
        receiver, matchup = native_target(offense, defense, rng, preferred=preferred, fatigue=fatigue)
        context = row_context()
        context.update(
            {
                "target_id": receiver.player_id,
                "target_efficiency": float(receiver.efficiency),
                "target_explosive": float(receiver.explosive),
                "matchup_pressure_probability": float(matchup.pressure_probability),
                "matchup_completion_probability": float(matchup.completion_probability),
                "matchup_interception_probability": float(matchup.interception_probability),
                "matchup_yards_multiplier": float(matchup.yards_multiplier),
                "matchup_qb_read_quality": float(matchup.qb_read_quality),
                "matchup_coverage_strength": float(matchup.coverage_strength),
                "matchup_local_coverage_strength": float(matchup.local_coverage_strength),
                "matchup_local_rush_strength": float(matchup.local_rush_strength),
                "matchup_pocket_integrity": float(matchup.pocket_integrity),
            }
        )
        target_rows.append(context)
        return receiver, matchup

    def qb_response(*, pressured, mobility, pocket_skill, rng):
        response = native_qb_response(
            pressured=pressured,
            mobility=mobility,
            pocket_skill=pocket_skill,
            rng=rng,
        )
        context = row_context()
        context.update(
            {
                "pressured": bool(pressured),
                "mobility_input": float(mobility),
                "pocket_skill_input": float(pocket_skill),
                "qb_response": response.value,
            }
        )
        qb_rows.append(context)
        return response

    def depth(profile, *, matchup_completion_probability, matchup_interception_probability, pressured):
        outcome = native_depth(
            profile,
            matchup_completion_probability=matchup_completion_probability,
            matchup_interception_probability=matchup_interception_probability,
            pressured=pressured,
        )
        context = row_context()
        context.update(
            {
                "depth_category": profile.category,
                "profile_completion_rate": float(profile.completion_rate),
                "profile_interception_rate": float(profile.interception_rate),
                "matchup_completion_probability": None if matchup_completion_probability is None else float(matchup_completion_probability),
                "matchup_interception_probability": None if matchup_interception_probability is None else float(matchup_interception_probability),
                "pressured": bool(pressured),
                "resolved_completion_probability": float(outcome.completion),
                "resolved_interception_probability": float(outcome.interception),
            }
        )
        throw_rows.append(context)
        return outcome

    def run(profile, *, matchup_stuff_probability, matchup_yards_multiplier, runner_power, tackling, explosiveness, rng):
        anatomy = native_run(
            profile,
            matchup_stuff_probability=matchup_stuff_probability,
            matchup_yards_multiplier=matchup_yards_multiplier,
            runner_power=runner_power,
            tackling=tackling,
            explosiveness=explosiveness,
            rng=rng,
        )
        context = row_context()
        context.update(
            {
                "geometry": profile.category,
                "profile_yards_mean": float(profile.yards_mean),
                "profile_negative_rate": float(profile.negative_rate),
                "profile_zero_rate": float(profile.zero_rate),
                "profile_explosive_10_rate": float(profile.explosive_10_rate),
                "profile_explosive_15_rate": float(profile.explosive_15_rate),
                "profile_explosive_20_rate": float(profile.explosive_20_rate),
                "profile_p99": float(profile.yards_p99),
                "neutral_routine_mean": float(resolution._neutral_routine_mean(profile)) if profile.category != "other" else None,
                "matchup_stuff_probability": float(matchup_stuff_probability),
                "matchup_yards_multiplier": float(matchup_yards_multiplier),
                "runner_power": float(runner_power),
                "tackling": float(tackling),
                "explosiveness": float(explosiveness),
                "resolved_yards": float(anatomy.total_yards),
                "resolved_contact": anatomy.contact.value,
                "resolved_ybc": float(anatomy.yards_before_contact),
                "resolved_yac": float(anatomy.yards_after_contact),
                "at_lower_clip": abs(float(anatomy.total_yards) - 0.1) < 1e-9,
                "at_upper_routine_clip": abs(float(anatomy.total_yards) - 9.999) < 1e-9,
            }
        )
        run_rows.append(context)
        return anatomy

    root._NATIVE["simulate_scrimmage_play"] = scrimmage
    pk._field_read_target = target
    pk.resolve_qb_response = qb_response
    resolution.depth_throw_probabilities = depth
    resolution.resolve_run_ecology = run
    try:
        yield
    finally:
        root._NATIVE["simulate_scrimmage_play"] = native_scrimmage
        pk._field_read_target = native_target
        pk.resolve_qb_response = native_qb_response
        resolution.depth_throw_probabilities = native_depth
        resolution.resolve_run_ecology = native_run


def _run_mechanism_audit(*, pools, teams, defenses, ecology, worlds: int, seed: int) -> dict[str, pl.DataFrame]:
    event_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    qb_rows: list[dict[str, Any]] = []
    throw_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}
    config = root.CONFIGS["E_isolated_suppressed"]
    total = len(integrated.MATCHUPS) * worlds
    completed = 0
    started = time.monotonic()

    with _mechanism_capture(
        event_rows=event_rows,
        target_rows=target_rows,
        qb_rows=qb_rows,
        throw_rows=throw_rows,
        run_rows=run_rows,
        metadata=metadata,
    ):
        for game_idx, (away, home) in enumerate(integrated.MATCHUPS):
            game = f"{away}@{home}"
            for world in range(worlds):
                world_seed = seed + game_idx * 1_000_003 + world
                metadata.clear()
                metadata.update({"game": game, "world": world, "seed": world_seed})
                away_plan = root.sample_event_rush_share_plan(pools[away], rng=np.random.default_rng(world_seed + 101_003))
                home_plan = root.sample_event_rush_share_plan(pools[home], rng=np.random.default_rng(world_seed + 202_007))
                away_team = integrated._with_event_rush_plan(teams[away], away_plan)
                home_team = integrated._with_event_rush_plan(teams[home], home_plan)
                streams = root.RngStreams.from_seed(world_seed + 7_000_019)
                audit: defaultdict[str, float] = defaultdict(float)
                with root._experiment_runtime(config, streams=streams, ecology=ecology, audit=audit):
                    root.game_loop.simulate_game(
                        away_team,
                        home_team,
                        away_defense=defenses[away],
                        home_defense=defenses[home],
                        seed=world_seed,
                        chaos_ecology=ecology,
                    )
                completed += 1
                if completed % 100 == 0 or completed == total:
                    elapsed = max(time.monotonic() - started, 1e-9)
                    rate = completed / elapsed
                    eta = (total - completed) / rate if rate > 0 else 0.0
                    print(f"ROOT-MECH {completed:,}/{total:,} games | {game} | {rate:.2f}/s | ETA {eta/60:.1f}m", flush=True)

    return {
        "events": pl.DataFrame(event_rows),
        "targets": pl.DataFrame(target_rows),
        "qb": pl.DataFrame(qb_rows),
        "throws": pl.DataFrame(throw_rows),
        "runs": pl.DataFrame(run_rows),
    }


def _identity_snapshot(teams: dict[str, TeamIdentity], defenses: dict[str, DefensiveUnit]) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for team_id, team in teams.items():
        defense = defenses[team_id]
        receiver_eff = [p.efficiency for p in team.receivers]
        receiver_exp = [p.explosive for p in team.receivers]
        rusher_eff = [p.efficiency for p in team.rushers]
        coverage = [p.coverage for p in defense.coverage]
        rush = [p.pass_rush for p in defense.front]
        rows.append(
            {
                "team_id": team_id,
                "qb_id": team.quarterback.player_id,
                "qb_efficiency": float(team.quarterback.efficiency),
                "qb_explosive": float(team.quarterback.explosive),
                "team_pass_efficiency": float(team.pass_efficiency),
                "team_rush_efficiency": float(team.rush_efficiency),
                "pass_protection": float(team.pass_protection),
                "run_blocking": float(team.run_blocking),
                "receiver_efficiency_mean": float(np.mean(receiver_eff)) if receiver_eff else 1.0,
                "receiver_efficiency_sd": float(np.std(receiver_eff)) if receiver_eff else 0.0,
                "receiver_explosive_mean": float(np.mean(receiver_exp)) if receiver_exp else 1.0,
                "rusher_efficiency_mean": float(np.mean(rusher_eff)) if rusher_eff else 1.0,
                "defense_coverage_mean": float(np.mean(coverage)) if coverage else 1.0,
                "defense_pass_rush_mean": float(np.mean(rush)) if rush else 1.0,
                "defense_pressure_rate": float(defense.pressure_rate),
                "defense_run_stuff_rate": float(defense.run_stuff_rate),
            }
        )
    return pl.DataFrame(rows)


def _run_sensitivity(*, pools, teams, defenses, ecology, worlds: int, seed: int) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    config = root.CONFIGS["E_isolated_suppressed"]
    total = len(SENSITIVITY_BRANCHES) * len(integrated.MATCHUPS) * worlds
    completed = 0
    started = time.monotonic()

    for branch in SENSITIVITY_BRANCHES:
        branch_teams = {team_id: _team_for_branch(team, branch) for team_id, team in teams.items()}
        branch_defenses = {team_id: _defense_for_branch(defense, branch) for team_id, defense in defenses.items()}
        for game_idx, (away, home) in enumerate(integrated.MATCHUPS):
            game = f"{away}@{home}"
            for world in range(worlds):
                world_seed = seed + game_idx * 1_000_003 + world
                away_plan = root.sample_event_rush_share_plan(pools[away], rng=np.random.default_rng(world_seed + 101_003))
                home_plan = root.sample_event_rush_share_plan(pools[home], rng=np.random.default_rng(world_seed + 202_007))
                away_team = integrated._with_event_rush_plan(branch_teams[away], away_plan)
                home_team = integrated._with_event_rush_plan(branch_teams[home], home_plan)
                streams = root.RngStreams.from_seed(world_seed + 7_000_019)
                audit: defaultdict[str, float] = defaultdict(float)
                with root._experiment_runtime(config, streams=streams, ecology=ecology, audit=audit):
                    result = root.game_loop.simulate_game(
                        away_team,
                        home_team,
                        away_defense=branch_defenses[away],
                        home_defense=branch_defenses[home],
                        seed=world_seed,
                        chaos_ecology=ecology,
                    )
                away_score = _score_for_team(result, away, away)
                home_score = _score_for_team(result, home, away)
                away_pass, away_rush, away_att, away_comp = _aggregate_player_stats(result, away_team)
                home_pass, home_rush, home_att, home_comp = _aggregate_player_stats(result, home_team)
                rows.append(
                    {
                        "branch": branch,
                        "game": game,
                        "world": world,
                        "seed": world_seed,
                        "away_team": away,
                        "home_team": home,
                        "away_score": away_score,
                        "home_score": home_score,
                        "game_total": away_score + home_score,
                        "margin": home_score - away_score,
                        "abs_margin": abs(home_score - away_score),
                        "away_pass_yards": away_pass,
                        "home_pass_yards": home_pass,
                        "away_rush_yards": away_rush,
                        "home_rush_yards": home_rush,
                        "away_completion_rate": float(away_comp / away_att) if away_att else 0.0,
                        "home_completion_rate": float(home_comp / home_att) if home_att else 0.0,
                        "drives": int(result.drives),
                        "scrimmage_plays": int(sum(trace.scrimmage_plays for trace in result.drive_traces)),
                    }
                )
                completed += 1
                if completed % 180 == 0 or completed == total:
                    elapsed = max(time.monotonic() - started, 1e-9)
                    rate = completed / elapsed
                    eta = (total - completed) / rate if rate > 0 else 0.0
                    print(f"ROOT-SKILL {completed:,}/{total:,} games | {branch} {game} | {rate:.2f}/s | ETA {eta/60:.1f}m", flush=True)
    return pl.DataFrame(rows)


def _sensitivity_summaries(worlds: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    game_means = (
        worlds.group_by(["branch", "game"])
        .agg(
            pl.col("game_total").mean().alias("mean_total"),
            pl.col("margin").mean().alias("mean_margin"),
            pl.col("away_score").mean().alias("mean_away_score"),
            pl.col("home_score").mean().alias("mean_home_score"),
            pl.col("scrimmage_plays").mean().alias("mean_scrimmage_plays"),
            pl.col("drives").mean().alias("mean_drives"),
        )
        .sort(["branch", "game"])
    )
    branch_summary = (
        game_means.group_by("branch")
        .agg(
            pl.col("mean_total").mean().alias("slate_mean_total"),
            pl.col("mean_total").std().alias("between_game_total_sd"),
            (pl.col("mean_total").max() - pl.col("mean_total").min()).alias("between_game_total_range"),
            pl.col("mean_margin").abs().mean().alias("mean_abs_expected_margin"),
            pl.col("mean_margin").std().alias("between_game_margin_sd"),
            pl.col("mean_scrimmage_plays").mean().alias("slate_scrimmage_plays"),
            pl.col("mean_drives").mean().alias("slate_drives"),
        )
        .sort("branch")
    )

    full = worlds.filter(pl.col("branch") == "full").select(
        "game", "world", "game_total", "margin", "away_score", "home_score"
    ).rename(
        {
            "game_total": "full_total",
            "margin": "full_margin",
            "away_score": "full_away_score",
            "home_score": "full_home_score",
        }
    )
    pairs: list[pl.DataFrame] = []
    for branch in SENSITIVITY_BRANCHES:
        if branch == "full":
            continue
        candidate = worlds.filter(pl.col("branch") == branch).select(
            "game", "world", "game_total", "margin", "away_score", "home_score"
        )
        pair = candidate.join(full, on=["game", "world"], how="inner").with_columns(
            pl.lit(branch).alias("branch"),
            (pl.col("game_total") - pl.col("full_total")).alias("delta_total"),
            (pl.col("margin") - pl.col("full_margin")).alias("delta_margin"),
            ((pl.col("away_score") - pl.col("full_away_score")).abs() + (pl.col("home_score") - pl.col("full_home_score")).abs()).alias("team_score_l1_delta"),
        )
        pairs.append(pair)
    paired = pl.concat(pairs) if pairs else pl.DataFrame()
    paired_summary = (
        paired.group_by("branch")
        .agg(
            pl.col("delta_total").mean().alias("mean_delta_total_vs_full"),
            pl.col("delta_total").abs().mean().alias("mean_abs_total_delta_vs_full"),
            pl.col("delta_margin").abs().mean().alias("mean_abs_margin_delta_vs_full"),
            pl.col("team_score_l1_delta").mean().alias("mean_team_score_l1_delta_vs_full"),
            (pl.col("team_score_l1_delta") > 0).mean().alias("world_fraction_any_score_change"),
        )
        .sort("branch")
    )
    return game_means, branch_summary, paired_summary


def _mechanism_summary(frames: dict[str, pl.DataFrame]) -> pl.DataFrame:
    events = frames["events"].filter(pl.col("early_down") & (pl.col("play_type") == "pass"))
    qb = frames["qb"].filter(pl.col("early_down"))
    targets = frames["targets"].filter(pl.col("early_down"))
    throws = frames["throws"].filter(pl.col("early_down"))
    runs = frames["runs"].filter(pl.col("early_down"))

    rows: list[dict[str, Any]] = []

    def add(metric: str, value: float, note: str) -> None:
        rows.append({"metric": metric, "value": float(value), "note": note})

    if targets.height:
        add("mean_matchup_pressure_probability", targets.get_column("matchup_pressure_probability").mean(), "pre-snap matchup pressure hazard")
        add("mean_matchup_completion_probability", targets.get_column("matchup_completion_probability").mean(), "coarse matchup completion before depth and pressure conditioning")
        add("mean_matchup_qb_read_quality", targets.get_column("matchup_qb_read_quality").mean(), "currently receives team pass efficiency at runtime")
        add("mean_qb_player_efficiency_on_pass_snaps", targets.get_column("qb_player_efficiency").mean(), "compiled QB player trait")
        add("mean_team_pass_efficiency_on_pass_snaps", targets.get_column("team_pass_efficiency").mean(), "team historical context")
    if events.height:
        add("realized_pressure_rate", events.get_column("pressured").mean(), "actual early-down pass pressure")
        add("realized_sack_rate", (events.get_column("pass_result") == PassResult.SACK.value).mean(), "all early-down dropbacks")
        pressured = events.filter(pl.col("pressured"))
        if pressured.height:
            add("sack_rate_given_pressure", (pressured.get_column("pass_result") == PassResult.SACK.value).mean(), "conversion of pressure into sack")
        add("realized_completion_rate_per_dropback", (events.get_column("pass_result") == PassResult.COMPLETE.value).mean(), "includes sacks/scrambles in denominator")
    if qb.height:
        pressured_qb = qb.filter(pl.col("pressured"))
        if pressured_qb.height:
            add("qb_response_sack_rate_given_pressure", (pressured_qb.get_column("qb_response") == "sack").mean(), "QB response branch")
        add("mean_pocket_skill_input", qb.get_column("pocket_skill_input").mean(), "runtime pocket skill argument")
        add("mean_qb_player_efficiency_qb_response", qb.get_column("qb_player_efficiency").mean(), "compiled QB efficiency beside runtime pocket input")
    if throws.height:
        add("mean_profile_completion_prior", throws.get_column("profile_completion_rate").mean(), "historical depth prior")
        add("mean_resolved_completion_probability", throws.get_column("resolved_completion_probability").mean(), "after matchup and pressure conditioning")
        add("mean_profile_interception_prior", throws.get_column("profile_interception_rate").mean(), "historical depth prior")
        add("mean_resolved_interception_probability", throws.get_column("resolved_interception_probability").mean(), "after matchup and pressure conditioning")
    if runs.height:
        add("run_profile_yards_mean", runs.get_column("profile_yards_mean").mean(), "historical geometry mean entering resolver")
        add("run_resolved_yards_mean", runs.get_column("resolved_yards").mean(), "resolved run mean before field clipping")
        add("run_lower_clip_rate_0_1", runs.get_column("at_lower_clip").mean(), "exact 0.1-yard routine floor")
        add("run_upper_routine_clip_rate_9_999", runs.get_column("at_upper_routine_clip").mean(), "exact 9.999-yard routine ceiling")
        add("run_mean_matchup_yards_multiplier", runs.get_column("matchup_yards_multiplier").mean(), "player/blocking/front efficiency input")
        add("run_mean_neutral_routine_mean", runs.get_column("neutral_routine_mean").drop_nulls().mean(), "mean routine branch target before matchup")
    return pl.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--personnel", type=Path, required=True)
    parser.add_argument("--player-usage", type=Path, required=True)
    parser.add_argument("--situation-context", type=Path, required=True)
    parser.add_argument("--history", type=int, default=2025)
    parser.add_argument("--mechanism-worlds", type=int, default=MECHANISM_WORLDS_DEFAULT)
    parser.add_argument("--sensitivity-worlds", type=int, default=SENSITIVITY_WORLDS_DEFAULT)
    parser.add_argument("--seed", type=int, default=2026190921)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/monster"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    historical, corrected_starts = early._historical_inputs(season=args.history, cache_dir=args.cache_dir)
    pools, teams, defenses, _states, ecology = drive._build_current_world_inputs(
        policy_path=args.policy,
        personnel_path=args.personnel,
        player_usage_path=args.player_usage,
        situation_context_path=args.situation_context,
    )

    identity = _identity_snapshot(teams, defenses)
    identity.write_csv(args.out / "identity_snapshot.csv")
    corrected_starts.write_csv(args.out / "historical_first_scrimmage_drive_starts.csv")

    frames = _run_mechanism_audit(
        pools=pools,
        teams=teams,
        defenses=defenses,
        ecology=ecology,
        worlds=args.mechanism_worlds,
        seed=args.seed,
    )
    for name, frame in frames.items():
        frame.write_parquet(args.out / f"mechanism_{name}.parquet")
    mechanism_summary = _mechanism_summary(frames)
    mechanism_summary.write_csv(args.out / "mechanism_summary.csv")

    # Retain the exact historical early-down comparison source used by MON-LEDGER-002A.
    historical.write_parquet(args.out / "historical_early_down_reference.parquet")

    sensitivity = _run_sensitivity(
        pools=pools,
        teams=teams,
        defenses=defenses,
        ecology=ecology,
        worlds=args.sensitivity_worlds,
        seed=args.seed + 91_991,
    )
    sensitivity.write_parquet(args.out / "skill_sensitivity_worlds.parquet")
    game_means, branch_summary, paired_summary = _sensitivity_summaries(sensitivity)
    game_means.write_csv(args.out / "skill_sensitivity_game_means.csv")
    branch_summary.write_csv(args.out / "skill_sensitivity_branch_summary.csv")
    paired_summary.write_csv(args.out / "skill_sensitivity_paired_summary.csv")

    # Static authority evidence: the current pass runtime feeds team pass efficiency into
    # quarterback read/pocket mechanisms while the QB's compiled efficiency is not the input.
    authority = {
        "pass_snap_qb_read_runtime_input": "TeamIdentity.pass_efficiency",
        "qb_response_pocket_skill_runtime_input": "TeamIdentity.pass_efficiency",
        "compiled_qb_player_efficiency_present": True,
        "compiled_qb_player_efficiency_direct_throw_quality_input": False,
        "qb_player_efficiency_known_direct_runtime_roles": ["scramble runner power"],
        "receiver_efficiency_direct_runtime_roles": ["coverage separation", "completion relative", "target read value"],
        "receiver_explosive_direct_runtime_roles": ["yards multiplier", "target read value", "YAC interaction"],
        "diagnostic_interpretation": "Treat the counterfactual qb_flat branch as causal evidence before changing authority.",
    }
    (args.out / "player_skill_authority_map.json").write_text(json.dumps(authority, indent=2) + "\n")

    manifest = {
        "experiment": "MON-ROOT-DIAGNOSTIC-BATTERY-001",
        "parent_ledgers": ["MON-LEDGER-002", "MON-LEDGER-002A"],
        "center": "maximum causal fidelity to observable NFL reality with meaningful player and matchup identity",
        "questions": [
            "Which play-resolution transformation suppresses ordinary offensive gains?",
            "Is excess sack pressure generated pre-snap or during QB response?",
            "Does the run resolver preserve empirical geometry distributions without boundary pileups?",
            "How much do QB skill, offensive player skill, team context, and defensive player skill causally move scores?",
            "Which identity layer preserves or destroys between-game score differentiation?",
        ],
        "mechanism_worlds_per_game": args.mechanism_worlds,
        "sensitivity_worlds_per_game_per_branch": args.sensitivity_worlds,
        "sensitivity_branches": list(SENSITIVITY_BRANCHES),
        "games": len(integrated.MATCHUPS),
        "governance": {
            "stage": "LAB",
            "production_promoted": False,
            "football_coefficients_changed_for_experiment": False,
            "market_inputs_used": False,
            "counterfactual_identity_changes_are_audit_only": True,
            "chaos_consequences_suppressed": True,
            "isolated_rng": True,
        },
        "known_prior_evidence": {
            "drive_survival_first_major_break": "between scrimmage play 3 and play 4",
            "run_0_2_overproduction": True,
            "pass_completed_yardage_shape_close_to_history": True,
            "field_position_mean_corrected_to_first_scrimmage_snap": True,
        },
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(mechanism_summary)
    print(branch_summary)
    print(paired_summary)


if __name__ == "__main__":
    main()
