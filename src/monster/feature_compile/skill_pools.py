from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime
from typing import Any

import numpy as np
import polars as pl

from monster.feature_compile.mechanisms import PlayerMechanismInputs
from monster.snapshot.player import PlayerState, TeamPlayerPool

_SKILL_POSITIONS = {"QB", "RB", "WR", "TE"}
_TARGET_DEFAULT = {"RB": 0.08, "WR": 0.18, "TE": 0.12}
_RUSH_DEFAULT = {"QB": 0.08, "RB": 0.45, "WR": 0.03, "TE": 0.005}
_CATCH_DEFAULT = {"RB": 0.76, "WR": 0.64, "TE": 0.68, "QB": 0.50}
_YPR_DEFAULT = {"RB": 8.0, "WR": 12.0, "TE": 10.5, "QB": 5.0}
_YPC_DEFAULT = {"QB": 5.0, "RB": 4.2, "WR": 6.5, "TE": 3.5}
# Quarterback depth is a contingent state, not an ordinary rotation. QB2/QB3 keep tiny
# latent mass so they can inherit the offense when QB1 is unavailable, but they do not
# receive routine 15-20% passing shares merely because they are dressed.
_QB_DEPTH_WEIGHT = {1: 1.0, 2: 0.006, 3: 0.001, 4: 0.0005}


def _finite(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return default if not np.isfinite(number) else number


def _normalize(values: list[float]) -> list[float]:
    arr = np.clip(np.asarray(values, dtype=float), 0.0, None)
    total = float(arr.sum())
    if total <= 0.0:
        return [0.0 for _ in values]
    return (arr / total).tolist()


def _role_prior(history_share: float, volume: float, default: float, scale: float) -> float:
    confidence = float(np.clip(volume / scale, 0.0, 0.85))
    return (1.0 - confidence) * default + confidence * history_share


def compile_player_role_priors(historical_usage: pl.DataFrame) -> dict[str, dict[str, float]]:
    """Reduce old-team usage into transfer-safe player role tendencies."""
    if not historical_usage.height or "player_id" not in historical_usage.columns:
        return {}
    sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in historical_usage.to_dicts():
        player_id = str(row.get("player_id") or "")
        if not player_id:
            continue
        item = sums[player_id]
        targets = _finite(row.get("targets"))
        rushes = _finite(row.get("rushes"))
        pass_attempts = _finite(row.get("pass_attempts"))
        item["targets"] += targets
        item["rushes"] += rushes
        item["pass_attempts"] += pass_attempts
        item["receptions"] += _finite(row.get("receptions"))
        item["receiving_yards"] += _finite(row.get("receiving_yards"))
        item["receiving_tds"] += _finite(row.get("receiving_tds"))
        item["red_zone_targets"] += _finite(row.get("red_zone_targets"))
        item["rushing_yards"] += _finite(row.get("rushing_yards"))
        item["rushing_tds"] += _finite(row.get("rushing_tds"))
        item["red_zone_rushes"] += _finite(row.get("red_zone_rushes"))
        item["target_share_num"] += _finite(row.get("target_share")) * max(targets, 1.0)
        item["target_share_den"] += max(targets, 1.0) if targets > 0 else 0.0
        item["rush_share_num"] += _finite(row.get("rush_share")) * max(rushes, 1.0)
        item["rush_share_den"] += max(rushes, 1.0) if rushes > 0 else 0.0
        item["rz_target_num"] += _finite(row.get("red_zone_target_share")) * max(
            _finite(row.get("red_zone_targets")), 1.0
        )
        item["rz_target_den"] += max(_finite(row.get("red_zone_targets")), 1.0) if _finite(row.get("red_zone_targets")) > 0 else 0.0
        item["rz_rush_num"] += _finite(row.get("red_zone_rush_share")) * max(
            _finite(row.get("red_zone_rushes")), 1.0
        )
        item["rz_rush_den"] += max(_finite(row.get("red_zone_rushes")), 1.0) if _finite(row.get("red_zone_rushes")) > 0 else 0.0
        item["rec_td_num"] += _finite(row.get("receiving_td_share")) * max(
            _finite(row.get("receiving_tds")), 1.0
        )
        item["rec_td_den"] += max(_finite(row.get("receiving_tds")), 1.0) if _finite(row.get("receiving_tds")) > 0 else 0.0
        item["rush_td_num"] += _finite(row.get("rushing_td_share")) * max(
            _finite(row.get("rushing_tds")), 1.0
        )
        item["rush_td_den"] += max(_finite(row.get("rushing_tds")), 1.0) if _finite(row.get("rushing_tds")) > 0 else 0.0
        item["qb_share_num"] += _finite(row.get("qb_pass_share")) * max(pass_attempts, 1.0)
        item["qb_share_den"] += max(pass_attempts, 1.0) if pass_attempts > 0 else 0.0

    result: dict[str, dict[str, float]] = {}
    for player_id, item in sums.items():
        targets = item["targets"]
        rushes = item["rushes"]
        receptions = item["receptions"]
        result[player_id] = {
            "targets": targets,
            "rushes": rushes,
            "pass_attempts": item["pass_attempts"],
            "target_share": item["target_share_num"] / max(item["target_share_den"], 1.0),
            "rush_share": item["rush_share_num"] / max(item["rush_share_den"], 1.0),
            "red_zone_target_share": item["rz_target_num"] / max(item["rz_target_den"], 1.0),
            "red_zone_rush_share": item["rz_rush_num"] / max(item["rz_rush_den"], 1.0),
            "receiving_td_share": item["rec_td_num"] / max(item["rec_td_den"], 1.0),
            "rushing_td_share": item["rush_td_num"] / max(item["rush_td_den"], 1.0),
            "qb_pass_share": item["qb_share_num"] / max(item["qb_share_den"], 1.0),
            "catch_rate": receptions / max(targets, 1.0),
            "yards_per_reception": item["receiving_yards"] / max(receptions, 1.0),
            "yards_per_carry": item["rushing_yards"] / max(rushes, 1.0),
            "receiving_tds": item["receiving_tds"],
            "rushing_tds": item["rushing_tds"],
            "red_zone_targets": item["red_zone_targets"],
            "red_zone_rushes": item["red_zone_rushes"],
        }
    return result


def _policy_rows(policy: pl.DataFrame | None) -> dict[str, dict[str, Any]]:
    if policy is None or not policy.height or "team_id" not in policy.columns:
        return {}
    return {str(row["team_id"]): row for row in policy.to_dicts()}


def compile_current_skill_pools(
    personnel: pl.DataFrame,
    historical_usage: pl.DataFrame,
    *,
    policy: pl.DataFrame | None = None,
    minimum_active_probability: float = 0.03,
) -> dict[str, TeamPlayerPool]:
    """Build all current QB/RB/WR/TE pools from current role state + historical tendency."""
    priors = compile_player_role_priors(historical_usage)
    policies = _policy_rows(policy)
    candidates = personnel.filter(
        pl.col("position").cast(pl.Utf8).str.to_uppercase().is_in(sorted(_SKILL_POSITIONS))
        & (pl.col("game_day_active_probability") >= minimum_active_probability)
    )
    result: dict[str, TeamPlayerPool] = {}
    for team_id in sorted(candidates.get_column("team_id").unique().to_list()):
        rows = candidates.filter(pl.col("team_id") == team_id).to_dicts()
        drafts: list[dict[str, Any]] = []
        for row in rows:
            player_id = str(row.get("gsis_id") or row.get("pfr_id") or row.get("display_name"))
            position = str(row.get("position") or "").upper()
            hist = priors.get(player_id, {})
            conditional_snap = _finite(row.get("conditional_offense_snap_share"), 0.0)
            active_probability = float(np.clip(_finite(row.get("game_day_active_probability"), 0.0), 0.0, 1.0))
            uncertainty = float(np.clip(_finite(row.get("participation_uncertainty"), 0.16), 0.025, 0.35))

            target_default = _TARGET_DEFAULT.get(position, 0.0)
            rush_default = _RUSH_DEFAULT.get(position, 0.0)
            target_prior = _role_prior(
                _finite(hist.get("target_share")), _finite(hist.get("targets")), target_default, 60.0
            ) if target_default > 0 else 0.0
            rush_prior = _role_prior(
                _finite(hist.get("rush_share")), _finite(hist.get("rushes")), rush_default, 100.0
            ) if rush_default > 0 else 0.0

            depth_rank = int(_finite(row.get("depth_rank"), 0.0))
            if position == "QB":
                # Current depth is authoritative for ordinary Week 1 passing role; old-team
                # starter volume must not grant a transferred QB routine share behind QB1.
                qb_weight = _QB_DEPTH_WEIGHT.get(depth_rank, 0.002)
                if depth_rank == 1 and str(row.get("status") or "") == "ACT":
                    active_probability = max(active_probability, 0.985)
                    uncertainty = min(uncertainty, 0.10)
                elif depth_rank >= 2 and str(row.get("status") or "") == "ACT":
                    active_probability = max(active_probability, 0.97)
            else:
                qb_weight = 0.0

            rz_target_hist = _finite(hist.get("red_zone_target_share"))
            rz_target_conf = float(np.clip(_finite(hist.get("red_zone_targets")) / 15.0, 0.0, 0.75))
            rz_target_prior = (1.0 - rz_target_conf) * target_prior + rz_target_conf * rz_target_hist
            rz_rush_hist = _finite(hist.get("red_zone_rush_share"))
            rz_rush_conf = float(np.clip(_finite(hist.get("red_zone_rushes")) / 25.0, 0.0, 0.75))
            rz_rush_prior = (1.0 - rz_rush_conf) * rush_prior + rz_rush_conf * rz_rush_hist

            rec_td_hist = _finite(hist.get("receiving_td_share"))
            rec_td_conf = float(np.clip(_finite(hist.get("receiving_tds")) / 6.0, 0.0, 0.70))
            rec_td_prior = (1.0 - rec_td_conf) * target_prior + rec_td_conf * rec_td_hist
            rush_td_hist = _finite(hist.get("rushing_td_share"))
            rush_td_conf = float(np.clip(_finite(hist.get("rushing_tds")) / 6.0, 0.0, 0.70))
            rush_td_prior = (1.0 - rush_td_conf) * rush_prior + rush_td_conf * rush_td_hist

            catch_rate = _CATCH_DEFAULT.get(position, 0.65)
            if _finite(hist.get("targets")) >= 10:
                catch_rate = float(np.clip(_finite(hist.get("catch_rate"), catch_rate), 0.35, 0.92))
            ypr = _YPR_DEFAULT.get(position, 10.5)
            if _finite(hist.get("targets")) >= 10:
                ypr = float(np.clip(_finite(hist.get("yards_per_reception"), ypr), 4.0, 22.0))
            ypc = _YPC_DEFAULT.get(position, 4.2)
            if _finite(hist.get("rushes")) >= 15:
                ypc = float(np.clip(_finite(hist.get("yards_per_carry"), ypc), 1.5, 9.0))

            drafts.append(
                {
                    "row": row,
                    "player_id": player_id,
                    "position": position,
                    "target_weight": conditional_snap * target_prior,
                    "rush_weight": conditional_snap * rush_prior,
                    "rz_target_weight": conditional_snap * rz_target_prior,
                    "rz_rush_weight": conditional_snap * rz_rush_prior,
                    "rec_td_weight": conditional_snap * rec_td_prior,
                    "rush_td_weight": conditional_snap * rush_td_prior,
                    "qb_weight": qb_weight,
                    "active_probability": active_probability,
                    "uncertainty": uncertainty,
                    "catch_rate": catch_rate,
                    "ypr": ypr,
                    "ypc": ypc,
                }
            )

        target = _normalize([d["target_weight"] for d in drafts])
        rush = _normalize([d["rush_weight"] for d in drafts])
        rz_target = _normalize([d["rz_target_weight"] for d in drafts])
        rz_rush = _normalize([d["rz_rush_weight"] for d in drafts])
        rec_td = _normalize([d["rec_td_weight"] for d in drafts])
        rush_td = _normalize([d["rush_td_weight"] for d in drafts])
        qb = _normalize([d["qb_weight"] for d in drafts])

        players: list[PlayerState] = []
        for idx, draft in enumerate(drafts):
            row = draft["row"]
            players.append(
                PlayerState(
                    player_id=draft["player_id"],
                    display_name=str(row.get("display_name") or row.get("full_name") or draft["player_id"]),
                    position=draft["position"],
                    team_id=str(team_id),
                    target_share=target[idx],
                    rush_share=rush[idx],
                    red_zone_target_share=rz_target[idx],
                    red_zone_rush_share=rz_rush[idx],
                    receiving_td_share=rec_td[idx],
                    rushing_td_share=rush_td[idx],
                    catch_rate=draft["catch_rate"],
                    yards_per_reception=draft["ypr"],
                    yards_per_carry=draft["ypc"],
                    active_probability=draft["active_probability"],
                    effectiveness_if_active=1.0,
                    role_uncertainty=draft["uncertainty"],
                    qb_pass_share=qb[idx],
                )
            )

        p = policies.get(str(team_id), {})
        result[str(team_id)] = TeamPlayerPool(
            team_id=str(team_id),
            players=tuple(players),
            neutral_pass_rate=float(np.clip(_finite(p.get("neutral_pass_rate"), 0.56), 0.34, 0.72)),
            plays_per_drive=float(np.clip(_finite(p.get("plays_per_drive"), 6.1), 4.5, 8.0)),
            pass_td_share=float(np.clip(_finite(p.get("pass_td_share"), 0.64), 0.20, 0.90)),
            sack_rate=float(np.clip(_finite(p.get("sack_rate_allowed"), 0.065), 0.01, 0.18)),
            play_volume_uncertainty=0.08,
        )
    return result


def compile_player_physical_inputs(
    personnel: pl.DataFrame,
    *,
    game_date: date | None = None,
) -> dict[str, PlayerMechanismInputs]:
    """Compile physical/biology inputs for current skill players without fantasy data."""
    target_date = game_date or date.today()
    result: dict[str, PlayerMechanismInputs] = {}
    for row in personnel.to_dicts():
        position = str(row.get("position") or "").upper()
        if position not in _SKILL_POSITIONS:
            continue
        player_id = str(row.get("gsis_id") or row.get("pfr_id") or "")
        if not player_id:
            continue
        birth = row.get("birth_date")
        birth_date: date | None = None
        if isinstance(birth, datetime):
            birth_date = birth.date()
        elif isinstance(birth, date):
            birth_date = birth
        elif isinstance(birth, str) and birth:
            try:
                birth_date = date.fromisoformat(birth[:10])
            except ValueError:
                birth_date = None
        age_years = None
        if birth_date is not None:
            age_years = (target_date - birth_date).days / 365.2425
        result[player_id] = PlayerMechanismInputs(
            height_in=_finite(row.get("height"), 0.0) or None,
            weight_lbs=_finite(row.get("weight"), 0.0) or None,
            forty_time=_finite(row.get("forty"), 0.0) or None,
            age_years=age_years,
        )
    return result
