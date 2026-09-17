from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from monster.feature_compile.mechanisms import PlayerMechanismInputs
from monster.feature_compile.units import UnitPlayerInputs
from monster.feature_compile.v13_identity_bridge import compile_v13_player_identity
from monster.feature_compile.v634_matchup_identity_authority import (
    apply_matchup_identity_authority_v634,
)
from monster.sim.matchup_kernel import (
    DefensiveIdentity,
    DefensiveUnit,
    resolve_pass_matchup,
    resolve_run_matchup,
)
from monster.sim.play_kernel import PlayerIdentity, TeamIdentity
from monster.sim.snap_ecology import register_team_units
from monster.snapshot.model import TeamState
from monster.snapshot.player import PlayerState, TeamPlayerPool


RATINGS = (68.0, 78.0, 88.0, 96.0)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _capability(player_id: str, position: str, rating: float) -> UnitPlayerInputs:
    kwargs: dict[str, object] = {
        "player_id": player_id,
        "position": position,
        "madden_speed": rating,
        "madden_acceleration": rating,
        "madden_awareness": rating,
    }
    if position == "QB":
        kwargs.update(
            madden_throw_accuracy=rating,
            madden_throw_power=rating,
            madden_throw_under_pressure=rating,
            madden_throw_on_run=rating,
            madden_play_action=rating,
            madden_break_sack=rating,
        )
    elif position == "RB":
        kwargs.update(
            madden_carrying=rating,
            madden_break_tackle=rating,
            madden_ball_carrier_vision=rating,
            madden_juke=rating,
            madden_change_of_direction=rating,
            madden_trucking=rating,
        )
    elif position in {"WR", "TE"}:
        kwargs.update(
            madden_route_running=rating,
            madden_catching=rating,
            madden_release=rating,
            madden_catch_in_traffic=rating,
            madden_spectacular_catch=rating,
            madden_change_of_direction=rating,
        )
    return UnitPlayerInputs(**kwargs)


def _base_world() -> tuple[
    TeamIdentity,
    TeamPlayerPool,
    dict[str, PlayerMechanismInputs],
    TeamState,
]:
    pool = TeamPlayerPool(
        "LAB",
        (
            PlayerState("qb", "QB", "QB", "LAB", qb_pass_share=1.0),
            PlayerState("rb", "RB", "RB", "LAB", rush_share=0.75),
            PlayerState("wr", "WR", "WR", "LAB", target_share=0.32),
        ),
    )
    identity = TeamIdentity(
        team_id="LAB",
        quarterback=PlayerIdentity("qb", "QB", "QB", usage_weight=1.0),
        rushers=(PlayerIdentity("rb", "RB", "RB", usage_weight=0.75),),
        receivers=(PlayerIdentity("wr", "WR", "WR", usage_weight=0.32),),
        pass_efficiency=1.0,
        rush_efficiency=1.0,
        pass_protection=1.0,
        run_blocking=1.0,
        neutral_pass_rate=0.57,
    )
    reality = {
        pid: PlayerMechanismInputs(active_probability=1.0, effectiveness_if_active=1.0)
        for pid in ("qb", "rb", "wr")
    }
    return identity, pool, reality, TeamState(team_id="LAB", opponent_id="DEF")


def _team_curve(position: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for rating in RATINGS:
        base, pool, reality, state = _base_world()
        units = (
            _capability("qb", "QB", rating if position == "QB" else 82.0),
            _capability("rb", "RB", rating if position == "RB" else 82.0),
            _capability("wr", "WR", rating if position == "WR" else 82.0),
        )
        _, trace = apply_matchup_identity_authority_v634(
            base,
            pool=pool,
            reality=reality,
            unit_players=units,
            state=state,
        )
        rows.append(
            {
                "position": position,
                "rating": rating,
                "pass_efficiency": trace.pass_efficiency_after,
                "rush_efficiency": trace.rush_efficiency_after,
                "qb_efficiency": trace.quarterback_efficiency,
                "runner_efficiency": trace.weighted_runner_efficiency,
                "receiver_efficiency": trace.mean_receiver_efficiency,
                "receiver_route_skill": trace.mean_receiver_route_skill,
            }
        )
    return rows


def _receiver(rating: float) -> object:
    identity, _ = compile_v13_player_identity(
        player_id="wr",
        name="WR",
        position="WR",
        usage_weight=0.30,
        inputs=PlayerMechanismInputs(active_probability=1.0, effectiveness_if_active=1.0),
        capability_inputs=_capability("wr", "WR", rating),
    )
    return identity


def _coverage_strength(rating: float) -> float:
    # Same neutral-centered relative scale used by the defensive identity layer.
    return float(np.clip(np.exp((rating - 78.0) / 55.0), 0.72, 1.34))


def _wr_cb_matrix() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for wr_rating in RATINGS:
        for cb_rating in RATINGS:
            defense = DefensiveUnit(
                front=(
                    DefensiveIdentity(
                        "edge", "EDGE", "EDGE", pass_rush=1.0, run_defense=1.0, tackling=1.0
                    ),
                ),
                coverage=(
                    DefensiveIdentity(
                        "cb",
                        "CB",
                        "CB",
                        coverage=_coverage_strength(cb_rating),
                        ball_hawk=_coverage_strength(cb_rating),
                        speed=_coverage_strength(cb_rating),
                    ),
                ),
            )
            matchup = resolve_pass_matchup(
                _receiver(wr_rating),
                defense,
                pass_protection=1.0,
                quarterback_efficiency=1.0,
                responsibility_key=f"v634-wr{wr_rating}-cb{cb_rating}",
            )
            rows.append(
                {
                    "wr_rating": wr_rating,
                    "cb_rating": cb_rating,
                    "completion_probability": matchup.completion_probability,
                    "interception_probability": matchup.interception_probability,
                    "yards_multiplier": matchup.yards_multiplier,
                    "separation_edge": matchup.local_separation_edge,
                }
            )
    return rows


def _register_line(pass_block_rating: float, run_block_rating: float) -> None:
    rows = [
        SimpleNamespace(
            player_id="wr",
            position="WR",
            offense_snap_share=0.90,
        )
    ]
    for idx, position in enumerate(("LT", "LG", "C", "RG", "RT")):
        rows.append(
            SimpleNamespace(
                player_id=f"ol-{idx}",
                position=position,
                offense_snap_share=0.96,
                pass_block_signal=None,
                run_block_signal=None,
                madden_pass_block=pass_block_rating,
                madden_run_block=run_block_rating,
                madden_awareness=82.0,
                madden_stamina=88.0,
            )
        )
    register_team_units("LAB", rows)


def _line_edge_matrix() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    receiver = _receiver(82.0)
    for ol_rating in RATINGS:
        _register_line(ol_rating, ol_rating)
        for edge_rating in RATINGS:
            strength = _coverage_strength(edge_rating)
            defense = DefensiveUnit(
                front=(
                    DefensiveIdentity(
                        "edge",
                        "EDGE",
                        "EDGE",
                        pass_rush=strength,
                        run_defense=strength,
                        tackling=strength,
                    ),
                ),
                coverage=(
                    DefensiveIdentity("cb", "CB", "CB", coverage=1.0, ball_hawk=1.0),
                ),
            )
            pass_matchup = resolve_pass_matchup(
                receiver,
                defense,
                pass_protection=1.0,
                quarterback_efficiency=1.0,
                responsibility_key=f"v634-ol{ol_rating}-edge{edge_rating}",
            )
            rows.append(
                {
                    "ol_rating": ol_rating,
                    "edge_rating": edge_rating,
                    "pressure_probability": pass_matchup.pressure_probability,
                    "time_to_pressure": pass_matchup.time_to_pressure,
                    "pocket_integrity": pass_matchup.pocket_integrity,
                }
            )
    return rows


def _rb_front_matrix() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for rb_rating in RATINGS:
        runner, _ = compile_v13_player_identity(
            player_id="rb",
            name="RB",
            position="RB",
            usage_weight=0.70,
            inputs=PlayerMechanismInputs(active_probability=1.0, effectiveness_if_active=1.0),
            capability_inputs=_capability("rb", "RB", rb_rating),
        )
        for front_rating in RATINGS:
            strength = _coverage_strength(front_rating)
            defense = DefensiveUnit(
                front=(
                    DefensiveIdentity(
                        "box",
                        "BOX",
                        "LB",
                        pass_rush=1.0,
                        run_defense=strength,
                        tackling=strength,
                    ),
                ),
                coverage=(
                    DefensiveIdentity("s", "S", "S", coverage=1.0, tackling=strength),
                ),
            )
            matchup = resolve_run_matchup(
                runner,
                defense,
                run_blocking=1.0,
                responsibility_key=f"v634-rb{rb_rating}-front{front_rating}",
            )
            rows.append(
                {
                    "rb_rating": rb_rating,
                    "front_rating": front_rating,
                    "stuff_probability": matchup.stuff_probability,
                    "yards_multiplier": matchup.yards_multiplier,
                    "runner_edge": matchup.runner_edge,
                }
            )
    return rows


def _monotone(values: list[float], increasing: bool = True) -> bool:
    pairs = zip(values, values[1:])
    return all(a <= b + 1e-12 for a, b in pairs) if increasing else all(
        a >= b - 1e-12 for a, b in pairs
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    team_rows = _team_curve("QB") + _team_curve("RB") + _team_curve("WR")
    wr_cb = _wr_cb_matrix()
    line_edge = _line_edge_matrix()
    rb_front = _rb_front_matrix()

    _write_csv(args.out / "team_identity_response_curves.csv", team_rows)
    _write_csv(args.out / "wr_cb_interaction_matrix.csv", wr_cb)
    _write_csv(args.out / "ol_edge_interaction_matrix.csv", line_edge)
    _write_csv(args.out / "rb_front_interaction_matrix.csv", rb_front)

    qb_curve = [r["pass_efficiency"] for r in team_rows if r["position"] == "QB"]
    rb_curve = [r["rush_efficiency"] for r in team_rows if r["position"] == "RB"]

    weak_cb = [r for r in wr_cb if r["cb_rating"] == min(RATINGS)]
    elite_cb = [r for r in wr_cb if r["cb_rating"] == max(RATINGS)]
    wr_weak_cb = [float(r["completion_probability"]) for r in weak_cb]
    wr_elite_cb = [float(r["completion_probability"]) for r in elite_cb]

    weak_edge = [r for r in line_edge if r["edge_rating"] == min(RATINGS)]
    elite_edge = [r for r in line_edge if r["edge_rating"] == max(RATINGS)]
    pressure_weak_edge = [float(r["pressure_probability"]) for r in weak_edge]
    pressure_elite_edge = [float(r["pressure_probability"]) for r in elite_edge]

    report = {
        "experiment": "v6.3.4-matchup-identity-sensitivity",
        "ratings": list(RATINGS),
        "monotonicity": {
            "qb_rating_to_team_pass_efficiency": _monotone(qb_curve, True),
            "rb_rating_to_team_rush_efficiency": _monotone(rb_curve, True),
            "wr_rating_vs_weak_cb_to_completion": _monotone(wr_weak_cb, True),
            "wr_rating_vs_elite_cb_to_completion": _monotone(wr_elite_cb, True),
            "ol_rating_vs_weak_edge_to_pressure": _monotone(pressure_weak_edge, False),
            "ol_rating_vs_elite_edge_to_pressure": _monotone(pressure_elite_edge, False),
        },
        "spans": {
            "qb_pass_efficiency_relative_span": float(max(qb_curve) / min(qb_curve) - 1.0),
            "rb_rush_efficiency_relative_span": float(max(rb_curve) / min(rb_curve) - 1.0),
            "wr_cb_completion_absolute_span": float(
                max(float(r["completion_probability"]) for r in wr_cb)
                - min(float(r["completion_probability"]) for r in wr_cb)
            ),
            "wr_cb_yards_multiplier_relative_span": float(
                max(float(r["yards_multiplier"]) for r in wr_cb)
                / min(float(r["yards_multiplier"]) for r in wr_cb)
                - 1.0
            ),
            "ol_edge_pressure_absolute_span": float(
                max(float(r["pressure_probability"]) for r in line_edge)
                - min(float(r["pressure_probability"]) for r in line_edge)
            ),
            "rb_front_yards_multiplier_relative_span": float(
                max(float(r["yards_multiplier"]) for r in rb_front)
                / min(float(r["yards_multiplier"]) for r in rb_front)
                - 1.0
            ),
        },
        "interaction": {
            "wr_cb_best_completion": max(float(r["completion_probability"]) for r in wr_cb),
            "wr_cb_worst_completion": min(float(r["completion_probability"]) for r in wr_cb),
            "ol_edge_max_pressure": max(float(r["pressure_probability"]) for r in line_edge),
            "ol_edge_min_pressure": min(float(r["pressure_probability"]) for r in line_edge),
            "rb_front_best_yards_multiplier": max(float(r["yards_multiplier"]) for r in rb_front),
            "rb_front_worst_yards_multiplier": min(float(r["yards_multiplier"]) for r in rb_front),
        },
        "governance": {
            "direct_points_used": False,
            "fantasy_targets_used": False,
            "market_inputs_used": False,
            "same_mechanisms_as_runtime": True,
            "promotion_status": "SHADOW_DIAGNOSTIC_ONLY",
        },
    }
    report["all_monotonicity_checks_pass"] = all(report["monotonicity"].values())
    (args.out / "matchup_identity_sensitivity_v634.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
