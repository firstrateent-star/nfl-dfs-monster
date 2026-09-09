from __future__ import annotations

from pathlib import Path

PATH = Path("src/monster/feature_compile/skill_pools.py")


def main() -> None:
    text = PATH.read_text()

    old_helper_anchor = '''def _role_prior(history_share: float, volume: float, default: float, scale: float) -> float:\n    confidence = float(np.clip(volume / scale, 0.0, 0.85))\n    return (1.0 - confidence) * default + confidence * history_share\n\n\n'''
    new_helper_anchor = '''def _role_prior(history_share: float, volume: float, default: float, scale: float) -> float:\n    confidence = float(np.clip(volume / scale, 0.0, 0.85))\n    return (1.0 - confidence) * default + confidence * history_share\n\n\ndef _qb_rush_reservoir_prior(\n    historical_rushes: float,\n    historical_pass_attempts: float,\n    neutral_pass_rate: float,\n) -> float:\n    \"\"\"Estimate the QB slice of team rushes from transferable player tendency.\n\n    Historical 2022-25 starting-QB games have a 12.5% median team-rush share, 29.2% p90,\n    and 34.1% p95.  A QB's own rush/pass ratio moves him around that distribution; the\n    current offense's pass/run mix translates the player tendency into current-team supply.\n    \"\"\"\n    rushes = max(float(historical_rushes), 0.0)\n    pass_attempts = max(float(historical_pass_attempts), 0.0)\n    pass_rate = float(np.clip(neutral_pass_rate, 0.34, 0.72))\n    if pass_attempts <= 0.0:\n        return 0.125\n\n    rushes_per_pass = rushes / pass_attempts\n    implied_share = pass_rate * rushes_per_pass / max(1.0 - pass_rate, 0.15)\n    implied_share = float(np.clip(implied_share, 0.02, 0.3415))\n    confidence = float(np.clip(pass_attempts / 120.0, 0.0, 0.90))\n    return float(np.clip((1.0 - confidence) * 0.125 + confidence * implied_share, 0.03, 0.3415))\n\n\ndef _reserve_share(\n    drafts: list[dict[str, Any]],\n    base_key: str,\n    qb_distribution: list[float],\n    qb_reservoir: float,\n) -> list[float]:\n    \"\"\"Reserve a finite QB opportunity slice, then normalize non-QBs into the remainder.\"\"\"\n    non_qb = _normalize([d[base_key] if d[\"position\"] != \"QB\" else 0.0 for d in drafts])\n    reservoir = float(np.clip(qb_reservoir, 0.0, 0.60))\n    return [\n        reservoir * qb_distribution[idx]\n        if draft[\"position\"] == \"QB\"\n        else (1.0 - reservoir) * non_qb[idx]\n        for idx, draft in enumerate(drafts)\n    ]\n\n\n'''
    if old_helper_anchor not in text:
        raise RuntimeError("role-prior anchor not found")
    text = text.replace(old_helper_anchor, new_helper_anchor, 1)

    old_team_loop = '''    for team_id in sorted(candidates.get_column("team_id").unique().to_list()):\n        rows = candidates.filter(pl.col("team_id") == team_id).to_dicts()\n        drafts: list[dict[str, Any]] = []\n'''
    new_team_loop = '''    for team_id in sorted(candidates.get_column("team_id").unique().to_list()):\n        rows = candidates.filter(pl.col("team_id") == team_id).to_dicts()\n        p = policies.get(str(team_id), {})\n        neutral_pass_rate = float(np.clip(_finite(p.get("neutral_pass_rate"), 0.56), 0.34, 0.72))\n        drafts: list[dict[str, Any]] = []\n'''
    if old_team_loop not in text:
        raise RuntimeError("team-loop anchor not found")
    text = text.replace(old_team_loop, new_team_loop, 1)

    old_rush_prior = '''            rush_prior = _role_prior(\n                _finite(hist.get("rush_share")), _finite(hist.get("rushes")), rush_default, 100.0\n            ) if rush_default > 0 else 0.0\n\n            depth_rank = int(_finite(row.get("depth_rank"), 0.0))\n'''
    new_rush_prior = '''            if position == "QB":\n                rush_prior = _qb_rush_reservoir_prior(\n                    _finite(hist.get("rushes")),\n                    _finite(hist.get("pass_attempts")),\n                    neutral_pass_rate,\n                )\n            else:\n                rush_prior = _role_prior(\n                    _finite(hist.get("rush_share")), _finite(hist.get("rushes")), rush_default, 100.0\n                ) if rush_default > 0 else 0.0\n\n            depth_rank = int(_finite(row.get("depth_rank"), 0.0))\n'''
    if old_rush_prior not in text:
        raise RuntimeError("rush-prior anchor not found")
    text = text.replace(old_rush_prior, new_rush_prior, 1)

    old_draft_fields = '''                    "rush_weight": conditional_snap * rush_prior,\n                    "rz_target_weight": conditional_snap * rz_target_prior,\n                    "rz_rush_weight": conditional_snap * rz_rush_prior,\n                    "rec_td_weight": conditional_snap * rec_td_prior,\n                    "rush_td_weight": conditional_snap * rush_td_prior,\n'''
    new_draft_fields = '''                    "rush_weight": conditional_snap * rush_prior if position != "QB" else rush_prior,\n                    "qb_rush_reservoir": rush_prior if position == "QB" else 0.0,\n                    "historical_rushes": _finite(hist.get("rushes")),\n                    "historical_rushing_tds": _finite(hist.get("rushing_tds")),\n                    "historical_red_zone_rushes": _finite(hist.get("red_zone_rushes")),\n                    "rz_target_weight": conditional_snap * rz_target_prior,\n                    "rz_rush_weight": conditional_snap * rz_rush_prior,\n                    "rec_td_weight": conditional_snap * rec_td_prior,\n                    "rush_td_weight": conditional_snap * rush_td_prior,\n'''
    if old_draft_fields not in text:
        raise RuntimeError("draft-field anchor not found")
    text = text.replace(old_draft_fields, new_draft_fields, 1)

    old_normalization = '''        target = _normalize([d["target_weight"] for d in drafts])\n        rush = _normalize([d["rush_weight"] for d in drafts])\n        rz_target = _normalize([d["rz_target_weight"] for d in drafts])\n        rz_rush = _normalize([d["rz_rush_weight"] for d in drafts])\n        rec_td = _normalize([d["rec_td_weight"] for d in drafts])\n        rush_td = _normalize([d["rush_td_weight"] for d in drafts])\n        qb = _normalize([d["qb_weight"] for d in drafts])\n'''
    new_normalization = '''        target = _normalize([d["target_weight"] for d in drafts])\n        rz_target = _normalize([d["rz_target_weight"] for d in drafts])\n        rec_td = _normalize([d["rec_td_weight"] for d in drafts])\n        qb = _normalize([d["qb_weight"] for d in drafts])\n\n        qb_indices = [idx for idx, d in enumerate(drafts) if d["position"] == "QB"]\n        starter_idx = max(qb_indices, key=lambda idx: drafts[idx]["qb_weight"]) if qb_indices else None\n        qb_rush_reservoir = (\n            float(drafts[starter_idx]["qb_rush_reservoir"]) if starter_idx is not None else 0.0\n        )\n        rush = _reserve_share(drafts, "rush_weight", qb, qb_rush_reservoir)\n\n        if starter_idx is not None:\n            starter = drafts[starter_idx]\n            hist_rushes = float(starter["historical_rushes"])\n            hist_rz = float(starter["historical_red_zone_rushes"])\n            hist_tds = float(starter["historical_rushing_tds"])\n            rz_rate = hist_rz / max(hist_rushes, 1.0)\n            td_rate = hist_tds / max(hist_rushes, 1.0)\n            rz_factor = float(np.clip(rz_rate / 0.22, 0.65, 1.65)) if hist_rushes >= 10 else 1.0\n            td_factor = float(np.clip(td_rate / 0.045, 0.45, 2.00)) if hist_rushes >= 10 else 1.0\n            qb_rz_reservoir = float(np.clip(qb_rush_reservoir * rz_factor, 0.02, 0.45))\n            qb_td_reservoir = float(np.clip(qb_rush_reservoir * td_factor, 0.01, 0.55))\n        else:\n            qb_rz_reservoir = 0.0\n            qb_td_reservoir = 0.0\n\n        rz_rush = _reserve_share(drafts, "rz_rush_weight", qb, qb_rz_reservoir)\n        rush_td = _reserve_share(drafts, "rush_td_weight", qb, qb_td_reservoir)\n'''
    if old_normalization not in text:
        raise RuntimeError("normalization anchor not found")
    text = text.replace(old_normalization, new_normalization, 1)

    old_final_policy = '''        p = policies.get(str(team_id), {})\n        result[str(team_id)] = TeamPlayerPool(\n            team_id=str(team_id),\n            players=tuple(players),\n            neutral_pass_rate=float(np.clip(_finite(p.get("neutral_pass_rate"), 0.56), 0.34, 0.72)),\n'''
    new_final_policy = '''        result[str(team_id)] = TeamPlayerPool(\n            team_id=str(team_id),\n            players=tuple(players),\n            neutral_pass_rate=neutral_pass_rate,\n'''
    if old_final_policy not in text:
        raise RuntimeError("final-policy anchor not found")
    text = text.replace(old_final_policy, new_final_policy, 1)

    PATH.write_text(text)
    print(f"patched {PATH}")


if __name__ == "__main__":
    main()
