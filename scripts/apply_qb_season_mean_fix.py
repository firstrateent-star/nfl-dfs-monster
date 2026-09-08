from __future__ import annotations

from pathlib import Path

PATH = Path("src/monster/feature_compile/skill_pools.py")


def main() -> None:
    text = PATH.read_text()

    anchor = '''def _role_prior(history_share: float, volume: float, default: float, scale: float) -> float:\n    confidence = float(np.clip(volume / scale, 0.0, 0.85))\n    return (1.0 - confidence) * default + confidence * history_share\n\n\n'''
    helper = '''def _role_prior(history_share: float, volume: float, default: float, scale: float) -> float:\n    confidence = float(np.clip(volume / scale, 0.0, 0.85))\n    return (1.0 - confidence) * default + confidence * history_share\n\n\ndef _qb_season_mean_share(\n    historical_rushes: float,\n    historical_pass_attempts: float,\n    neutral_pass_rate: float,\n) -> float:\n    \"\"\"Translate a QB's season rushing tendency into a current-team mean carry share.\n\n    2022-25 QB-season starter evidence (180 QB-seasons) centers at 3.35 carries/start,\n    12.72% team-rush share, with p90 7.0 carries and 24.82% share. Player rush/pass\n    tendency is the identity signal; the game-level tail belongs downstream in simulation.\n    \"\"\"\n    rushes = max(float(historical_rushes), 0.0)\n    passes = max(float(historical_pass_attempts), 0.0)\n    pass_rate = float(np.clip(neutral_pass_rate, 0.34, 0.72))\n    population = 0.12717320312257022\n    if passes <= 0.0:\n        return population\n    rush_per_pass = rushes / passes\n    implied = pass_rate * rush_per_pass / max(1.0 - pass_rate, 0.15)\n    confidence = float(np.clip(passes / 250.0, 0.0, 0.92))\n    share = (1.0 - confidence) * population + confidence * implied\n    # p95 season share is 29.43%; means above that require stronger evidence than one old season.\n    return float(np.clip(share, 0.025, 0.2943107546048722))\n\n\ndef _reserve_qb_share(\n    drafts: list[dict[str, Any]], base_key: str, qb_distribution: list[float], reservoir: float\n) -> list[float]:\n    non_qb = _normalize([d[base_key] if d[\"position\"] != \"QB\" else 0.0 for d in drafts])\n    reservoir = float(np.clip(reservoir, 0.0, 0.45))\n    return [\n        reservoir * qb_distribution[idx]\n        if draft[\"position\"] == \"QB\"\n        else (1.0 - reservoir) * non_qb[idx]\n        for idx, draft in enumerate(drafts)\n    ]\n\n\n'''
    if anchor not in text:
        raise RuntimeError("helper anchor not found")
    text = text.replace(anchor, helper, 1)

    old_loop = '''    for team_id in sorted(candidates.get_column("team_id").unique().to_list()):\n        rows = candidates.filter(pl.col("team_id") == team_id).to_dicts()\n        drafts: list[dict[str, Any]] = []\n'''
    new_loop = '''    for team_id in sorted(candidates.get_column("team_id").unique().to_list()):\n        rows = candidates.filter(pl.col("team_id") == team_id).to_dicts()\n        p = policies.get(str(team_id), {})\n        neutral_pass_rate = float(np.clip(_finite(p.get("neutral_pass_rate"), 0.56), 0.34, 0.72))\n        drafts: list[dict[str, Any]] = []\n'''
    if old_loop not in text:
        raise RuntimeError("team loop anchor not found")
    text = text.replace(old_loop, new_loop, 1)

    old_prior = '''            rush_prior = _role_prior(\n                _finite(hist.get("rush_share")), _finite(hist.get("rushes")), rush_default, 100.0\n            ) if rush_default > 0 else 0.0\n\n            depth_rank = int(_finite(row.get("depth_rank"), 0.0))\n'''
    new_prior = '''            if position == "QB":\n                rush_prior = _qb_season_mean_share(\n                    _finite(hist.get("rushes")), _finite(hist.get("pass_attempts")), neutral_pass_rate\n                )\n            else:\n                rush_prior = _role_prior(\n                    _finite(hist.get("rush_share")), _finite(hist.get("rushes")), rush_default, 100.0\n                ) if rush_default > 0 else 0.0\n\n            depth_rank = int(_finite(row.get("depth_rank"), 0.0))\n'''
    if old_prior not in text:
        raise RuntimeError("rush prior anchor not found")
    text = text.replace(old_prior, new_prior, 1)

    old_fields = '''                    "rush_weight": conditional_snap * rush_prior,\n                    "rz_target_weight": conditional_snap * rz_target_prior,\n                    "rz_rush_weight": conditional_snap * rz_rush_prior,\n                    "rec_td_weight": conditional_snap * rec_td_prior,\n                    "rush_td_weight": conditional_snap * rush_td_prior,\n'''
    new_fields = '''                    "rush_weight": conditional_snap * rush_prior if position != "QB" else rush_prior,\n                    "qb_rush_mean_share": rush_prior if position == "QB" else 0.0,\n                    "historical_rushes": _finite(hist.get("rushes")),\n                    "historical_rushing_tds": _finite(hist.get("rushing_tds")),\n                    "rz_target_weight": conditional_snap * rz_target_prior,\n                    "rz_rush_weight": conditional_snap * rz_rush_prior,\n                    "rec_td_weight": conditional_snap * rec_td_prior,\n                    "rush_td_weight": conditional_snap * rush_td_prior,\n'''
    if old_fields not in text:
        raise RuntimeError("draft fields anchor not found")
    text = text.replace(old_fields, new_fields, 1)

    old_norm = '''        target = _normalize([d["target_weight"] for d in drafts])\n        rush = _normalize([d["rush_weight"] for d in drafts])\n        rz_target = _normalize([d["rz_target_weight"] for d in drafts])\n        rz_rush = _normalize([d["rz_rush_weight"] for d in drafts])\n        rec_td = _normalize([d["rec_td_weight"] for d in drafts])\n        rush_td = _normalize([d["rush_td_weight"] for d in drafts])\n        qb = _normalize([d["qb_weight"] for d in drafts])\n'''
    new_norm = '''        target = _normalize([d["target_weight"] for d in drafts])\n        rz_target = _normalize([d["rz_target_weight"] for d in drafts])\n        rec_td = _normalize([d["rec_td_weight"] for d in drafts])\n        qb = _normalize([d["qb_weight"] for d in drafts])\n\n        qb_indices = [idx for idx, d in enumerate(drafts) if d["position"] == "QB"]\n        starter_idx = max(qb_indices, key=lambda idx: drafts[idx]["qb_weight"]) if qb_indices else None\n        qb_mean_share = float(drafts[starter_idx]["qb_rush_mean_share"]) if starter_idx is not None else 0.0\n        rush = _reserve_qb_share(drafts, "rush_weight", qb, qb_mean_share)\n\n        if starter_idx is not None:\n            starter = drafts[starter_idx]\n            rushes = float(starter["historical_rushes"])\n            tds = float(starter["historical_rushing_tds"])\n            # Population QB-season rushing TD rate is 0.183/start. Preserve QB identity,\n            # but do not let noisy TD share normalization manufacture an extreme mean.\n            td_per_rush = tds / max(rushes, 1.0)\n            identity = float(np.clip(td_per_rush / 0.0468, 0.45, 1.75)) if rushes >= 20 else 1.0\n            qb_td_share = float(np.clip(qb_mean_share * identity, 0.01, 0.34))\n            qb_rz_share = float(np.clip(qb_mean_share * np.sqrt(identity), 0.02, 0.36))\n        else:\n            qb_td_share = 0.0\n            qb_rz_share = 0.0\n        rz_rush = _reserve_qb_share(drafts, "rz_rush_weight", qb, qb_rz_share)\n        rush_td = _reserve_qb_share(drafts, "rush_td_weight", qb, qb_td_share)\n'''
    if old_norm not in text:
        raise RuntimeError("normalization anchor not found")
    text = text.replace(old_norm, new_norm, 1)

    old_final = '''        p = policies.get(str(team_id), {})\n        result[str(team_id)] = TeamPlayerPool(\n            team_id=str(team_id),\n            players=tuple(players),\n            neutral_pass_rate=float(np.clip(_finite(p.get("neutral_pass_rate"), 0.56), 0.34, 0.72)),\n'''
    new_final = '''        result[str(team_id)] = TeamPlayerPool(\n            team_id=str(team_id),\n            players=tuple(players),\n            neutral_pass_rate=neutral_pass_rate,\n'''
    if old_final not in text:
        raise RuntimeError("final policy anchor not found")
    text = text.replace(old_final, new_final, 1)

    PATH.write_text(text)
    print(f"patched {PATH}")


if __name__ == "__main__":
    main()
