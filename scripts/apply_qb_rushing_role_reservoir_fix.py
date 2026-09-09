from __future__ import annotations

from pathlib import Path

PATH = Path("src/monster/sim/rushing_roles.py")


def main() -> None:
    text = PATH.read_text()

    old = '''    attempts_out = np.zeros((worlds, n_players), dtype=np.int16)\n    rush_td_out = np.zeros((worlds, n_players), dtype=np.int16)\n    team_rushes = side.team_rush_attempts.astype(int)\n    team_rush_tds = side.rushing_tds.astype(int)\n'''
    new = '''    # The generic rushing hierarchy is downstream of the calibrated QB reservoir.\n    # Preserve QB attempts/TDs already allocated against finite TEAM rush supply; the\n    # A/B/C role tree may reshape only the remaining non-QB opportunity. Otherwise a\n    # mobile QB can be promoted to generic Role A and inherit ~62% of core carries.\n    qb_indices = np.array([idx for idx, p in enumerate(players) if p.position == "QB"], dtype=int)\n    non_qb_indices = np.array([idx for idx, p in enumerate(players) if p.position != "QB"], dtype=int)\n    reserved_qb_attempts = np.zeros((worlds, n_players), dtype=np.int16)\n    reserved_qb_tds = np.zeros((worlds, n_players), dtype=np.int16)\n    for idx in qb_indices:\n        stats = side.player_stats[players[idx].player_id]\n        reserved_qb_attempts[:, idx] = stats["rush_attempts"].astype(np.int16)\n        reserved_qb_tds[:, idx] = stats["rushing_tds"].astype(np.int16)\n\n    attempts_out = np.zeros((worlds, n_players), dtype=np.int16)\n    rush_td_out = np.zeros((worlds, n_players), dtype=np.int16)\n    team_rushes = side.team_rush_attempts.astype(int)\n    team_rush_tds = side.rushing_tds.astype(int)\n'''
    if old not in text:
        raise RuntimeError("output initialization anchor not found")
    text = text.replace(old, new, 1)

    anchor = '''    for idx, player in enumerate(players):\n        stats = side.player_stats[player.player_id]\n'''
    correction = '''    # Re-impose the upstream QB reservoir, then proportionally compress the generic\n    # non-QB hierarchy into the exact residual team carry/TD supply. This preserves\n    # conservation and the validated RB/WR/TE role ordering without allowing that tree\n    # to redefine QB expected state.\n    for w in range(worlds):\n        qb_carries = int(reserved_qb_attempts[w].sum())\n        qb_tds = int(reserved_qb_tds[w].sum())\n        residual_carries = max(int(team_rushes[w]) - qb_carries, 0)\n        residual_tds = max(int(team_rush_tds[w]) - qb_tds, 0)\n\n        non_qb_weights = attempts_out[w, non_qb_indices].astype(float)\n        if len(non_qb_indices) and residual_carries > 0:\n            if non_qb_weights.sum() <= 0:\n                non_qb_weights = np.clip(base[non_qb_indices], 0.0, None)\n            if non_qb_weights.sum() <= 0:\n                non_qb_weights = np.ones(len(non_qb_indices), dtype=float)\n            non_qb_weights /= non_qb_weights.sum()\n            attempts_out[w, non_qb_indices] = rng.multinomial(\n                residual_carries, non_qb_weights\n            ).astype(np.int16)\n        elif len(non_qb_indices):\n            attempts_out[w, non_qb_indices] = 0\n\n        attempts_out[w, qb_indices] = reserved_qb_attempts[w, qb_indices]\n\n        if len(non_qb_indices) and residual_tds > 0:\n            td_weights = attempts_out[w, non_qb_indices].astype(float) * np.array(\n                [max(players[idx].red_zone_rush_share, players[idx].rushing_td_share, 0.01)\n                 for idx in non_qb_indices],\n                dtype=float,\n            )\n            if td_weights.sum() <= 0:\n                td_weights = np.ones(len(non_qb_indices), dtype=float)\n            td_weights /= td_weights.sum()\n            rush_td_out[w, non_qb_indices] = rng.multinomial(\n                residual_tds, td_weights\n            ).astype(np.int16)\n        elif len(non_qb_indices):\n            rush_td_out[w, non_qb_indices] = 0\n        rush_td_out[w, qb_indices] = reserved_qb_tds[w, qb_indices]\n\n    for idx, player in enumerate(players):\n        stats = side.player_stats[player.player_id]\n'''
    if anchor not in text:
        raise RuntimeError("player writeback anchor not found")
    text = text.replace(anchor, correction, 1)

    PATH.write_text(text)
    print(f"patched {PATH}")


if __name__ == "__main__":
    main()

# Orchestration trigger only: 2026-09-08 Player Reality certification.
