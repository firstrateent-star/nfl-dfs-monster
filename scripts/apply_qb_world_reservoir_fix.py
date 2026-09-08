from __future__ import annotations

from pathlib import Path

PATH = Path("src/monster/sim/allocation.py")


def main() -> None:
    text = PATH.read_text()

    anchor = '''def _gamma_sum(\n    rng: np.random.Generator,\n    counts: np.ndarray,\n    mean_per_event: float,\n    shape_per_event: float = 2.2,\n) -> np.ndarray:\n'''
    helper = '''def _allocate_rush_counts(\n    rng: np.random.Generator,\n    counts: np.ndarray,\n    players: tuple[PlayerState, ...],\n    base_values: np.ndarray,\n    role_probability: np.ndarray,\n) -> np.ndarray:\n    \"\"\"Allocate a finite QB rushing reservoir before distributing non-QB carries.\n\n    The compiled QB share is an expected share of TEAM rush attempts. Sampling all rushers\n    in one Dirichlet-like tree can renormalize that reservoir upward whenever other rushers\n    miss the role tree. Preserve the QB mean at the team-rush denominator, then allow\n    world-level variance around that mean and allocate the remainder among non-QBs.\n    \"\"\"\n    qb_mask = np.array([p.position == \"QB\" for p in players], dtype=bool)\n    if not qb_mask.any() or qb_mask.all():\n        shares = _sample_role_shares(\n            rng, players, base_values, len(counts), role_probability=role_probability\n        )\n        return _allocate_integer_counts(rng, counts, shares)\n\n    qb_mean_share = float(np.clip(base_values[qb_mask].sum(), 0.0, 0.45))\n    # Mean-one multiplicative noise creates mobile-QB tail worlds without redefining the mean.\n    qb_world_share = np.clip(\n        qb_mean_share * _mean_one_lognormal(rng, 0.32, len(counts)), 0.0, 0.45\n    )\n    qb_counts = rng.binomial(counts.astype(np.int64), qb_world_share).astype(np.int16)\n    non_qb_counts = (counts.astype(np.int64) - qb_counts.astype(np.int64)).astype(np.int16)\n\n    qb_base = np.where(qb_mask, base_values, 0.0)\n    non_qb_base = np.where(qb_mask, 0.0, base_values)\n    qb_roles = np.where(qb_mask, role_probability, 0.0)\n    non_qb_roles = np.where(qb_mask, 0.0, role_probability)\n\n    qb_shares = _sample_role_shares(\n        rng, players, qb_base, len(counts), role_probability=qb_roles\n    )\n    non_qb_shares = _sample_role_shares(\n        rng, players, non_qb_base, len(counts), role_probability=non_qb_roles\n    )\n    return (\n        _allocate_integer_counts(rng, qb_counts, qb_shares)\n        + _allocate_integer_counts(rng, non_qb_counts, non_qb_shares)\n    )\n\n\n'''
    if anchor not in text:
        raise RuntimeError("gamma anchor not found")
    text = text.replace(anchor, helper + anchor, 1)

    old = '''    targets = _allocate_integer_counts(rng, team_targets, target_shares)\n    rushes = _allocate_integer_counts(rng, team_rush_attempts, rush_shares)\n'''
    new = '''    targets = _allocate_integer_counts(rng, team_targets, target_shares)\n    # QB expected carry share is a reservoir against TEAM rush attempts. Do not allow\n    # non-QB role dropout/renormalization to inflate the QB's expected state.\n    rushes = _allocate_rush_counts(\n        rng, team_rush_attempts, players, rush_base, rush_role_probability\n    )\n'''
    if old not in text:
        raise RuntimeError("rush allocation anchor not found")
    text = text.replace(old, new, 1)

    PATH.write_text(text)
    print(f"patched {PATH}")


if __name__ == "__main__":
    main()
