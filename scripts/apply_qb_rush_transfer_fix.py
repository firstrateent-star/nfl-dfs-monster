from __future__ import annotations

from pathlib import Path

PATH = Path("src/monster/feature_compile/skill_pools.py")


def main() -> None:
    text = PATH.read_text()

    old_helper_anchor = '''def _role_prior(history_share: float, volume: float, default: float, scale: float) -> float:\n    confidence = float(np.clip(volume / scale, 0.0, 0.85))\n    return (1.0 - confidence) * default + confidence * history_share\n\n\n'''
    new_helper_anchor = '''def _role_prior(history_share: float, volume: float, default: float, scale: float) -> float:\n    confidence = float(np.clip(volume / scale, 0.0, 0.85))\n    return (1.0 - confidence) * default + confidence * history_share\n\n\ndef _qb_rush_prior(\n    historical_rushes: float,\n    historical_pass_attempts: float,\n    neutral_pass_rate: float,\n) -> float:\n    \"\"\"Translate a quarterback's own rushing tendency into the current offense.\n\n    QB rushing is a player tendency, not a transferable old-team rush share.  We use the\n    player's rushes per pass attempt, then translate that rate through the current team's\n    pass/run mix.  Sparse historical samples are shrunk toward an 8% team-rush-share prior.\n    \"\"\"\n    rushes = max(float(historical_rushes), 0.0)\n    pass_attempts = max(float(historical_pass_attempts), 0.0)\n    pass_rate = float(np.clip(neutral_pass_rate, 0.34, 0.72))\n    if pass_attempts <= 0.0:\n        return 0.08\n\n    rushes_per_pass = rushes / pass_attempts\n    implied_share = pass_rate * rushes_per_pass / max(1.0 - pass_rate, 0.15)\n    implied_share = float(np.clip(implied_share, 0.01, 0.38))\n    confidence = float(np.clip(pass_attempts / 100.0, 0.0, 0.90))\n    return (1.0 - confidence) * 0.08 + confidence * implied_share\n\n\n'''
    if old_helper_anchor not in text:
        raise RuntimeError("role-prior anchor not found")
    text = text.replace(old_helper_anchor, new_helper_anchor, 1)

    old_team_loop = '''    for team_id in sorted(candidates.get_column("team_id").unique().to_list()):\n        rows = candidates.filter(pl.col("team_id") == team_id).to_dicts()\n        drafts: list[dict[str, Any]] = []\n'''
    new_team_loop = '''    for team_id in sorted(candidates.get_column("team_id").unique().to_list()):\n        rows = candidates.filter(pl.col("team_id") == team_id).to_dicts()\n        p = policies.get(str(team_id), {})\n        neutral_pass_rate = float(np.clip(_finite(p.get("neutral_pass_rate"), 0.56), 0.34, 0.72))\n        drafts: list[dict[str, Any]] = []\n'''
    if old_team_loop not in text:
        raise RuntimeError("team-loop anchor not found")
    text = text.replace(old_team_loop, new_team_loop, 1)

    old_rush_prior = '''            rush_prior = _role_prior(\n                _finite(hist.get("rush_share")), _finite(hist.get("rushes")), rush_default, 100.0\n            ) if rush_default > 0 else 0.0\n\n            depth_rank = int(_finite(row.get("depth_rank"), 0.0))\n'''
    new_rush_prior = '''            if position == "QB":\n                rush_prior = _qb_rush_prior(\n                    _finite(hist.get("rushes")),\n                    _finite(hist.get("pass_attempts")),\n                    neutral_pass_rate,\n                )\n            else:\n                rush_prior = _role_prior(\n                    _finite(hist.get("rush_share")), _finite(hist.get("rushes")), rush_default, 100.0\n                ) if rush_default > 0 else 0.0\n\n            depth_rank = int(_finite(row.get("depth_rank"), 0.0))\n'''
    if old_rush_prior not in text:
        raise RuntimeError("rush-prior anchor not found")
    text = text.replace(old_rush_prior, new_rush_prior, 1)

    old_final_policy = '''        p = policies.get(str(team_id), {})\n        result[str(team_id)] = TeamPlayerPool(\n            team_id=str(team_id),\n            players=tuple(players),\n            neutral_pass_rate=float(np.clip(_finite(p.get("neutral_pass_rate"), 0.56), 0.34, 0.72)),\n'''
    new_final_policy = '''        result[str(team_id)] = TeamPlayerPool(\n            team_id=str(team_id),\n            players=tuple(players),\n            neutral_pass_rate=neutral_pass_rate,\n'''
    if old_final_policy not in text:
        raise RuntimeError("final-policy anchor not found")
    text = text.replace(old_final_policy, new_final_policy, 1)

    PATH.write_text(text)
    print(f"patched {PATH}")


if __name__ == "__main__":
    main()
