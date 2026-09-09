# Monster whole-system Flower code audit — 2026-09-09

## Center
The governing object is a correlated simulated NFL world. Football reality must be compiled once, conserved through game and player allocation, frozen blind, and only then exposed to DFS scoring/value.

## Inhale — what is strong
- Shared finite possession and play budgets are explicit.
- Team scoring is market blind and opponent-interactive.
- Player targets/rushes/TDs are finite allocations rather than independent projections.
- Health separates availability, effectiveness, and uncertainty.
- Current personnel, depth, OL, defensive units, Madden/scouting evidence, physical evidence, continuity, and environment have explicit source/provenance seams.
- Madden/physical evidence is bounded and causal rather than direct fantasy-point authority.
- Turnovers and D/ST are downstream and world-correlated.
- Full-Reality counterfactual, ablation, player-jurisdiction, Player Reality, QB and multi-seed gates provide unusually strong governance.

## Hold — pressure findings
### P0: Full-Reality environment compiler was only half-wired
`compile_team_mechanisms()` defines two environment jurisdictions: team scoring quality and pool-level pass opportunity (`neutral_pass_rate`, `targetable_dropback_rate`, play-volume uncertainty). Production Reality v1 runners were only copying `_weather_effect()` onto `TeamState`. The pool-level mechanism was therefore not reaching simulated Sundays. This is the most important current-path wiring defect.

### P0: Multiple production runners create architecture drift
`run_week1_full_monster.py`, `run_week1_reality_v1.py`, `run_week1_fantasy_complete.py`, and `run_week1_reality_fantasy_complete.py` compose overlapping pieces by monkey-patching module globals. The bridge worked, but it makes it too easy for a certified mechanism to be present in one runner and absent in another. Current production should converge on one canonical Reality-v1 composition seam.

### P1: Player role uncertainty is pooled too aggressively
`_sample_role_shares()` derives one concentration from the mean `role_uncertainty` of every player. Individual uncertainty therefore changes the whole role tree rather than primarily that player's distribution. This weakens the semantic meaning of player-specific uncertainty. A future allocator revision should use player-specific dispersion while preserving finite totals.

### P1: Effectiveness participates in both role weighting and efficiency
The role sampler multiplies weights by `effectiveness_if_active`, while catch/rush efficiency also multiplies by effectiveness. This can double-count conditional effectiveness and partially collapse the intended distinction between availability, role, and performance. It should be separated in the next allocator revision, with role changes driven by role evidence rather than automatically by effectiveness.

### P1: TD recipient allocation is not world-event constrained
Receiving-TD and rushing-TD shares are sampled independently from targets/receptions and rush attempts. Team TD totals conserve, but a player can theoretically receive a receiving TD in a world with no reception/target, or a rushing TD with no carry. TD allocation should eventually be conditioned on world-level eligible events while retaining exact team TD conservation.

### P1: Team mechanism API is richer than the production composition
`TeamMechanismInputs` includes team Madden overall, OL, opponent front, continuity and coaching entropy, but production mostly obtains these through separate unit/team-state compilers. This is not necessarily wrong, but it leaves two overlapping mechanism paths and increases double-counting/drift risk. Canonical composition should make jurisdiction explicit and prevent the same evidence family from entering twice.

### P2: Current health freshness is the dominant data limitation
The architecture can consume provider injuries, but the current Week 1 provider snapshot has zero formal injury rows and relies on explicit overrides. This is a state-freshness limitation, not a simulation-code defect.

### P2: Environment is a dated snapshot
Weather inputs are explicit and market blind but must be refreshed nearer lock. Surface is represented descriptively but has no validated causal effect; keeping it neutral is preferable to inventing one.

### P2: Wingspan/travel/rest/coaching detail remain bounded limitations
These should remain neutral/partial until trustworthy evidence and a causal mechanism exist. Absence is better than fabricated precision.

### P2: D/ST blocked-kick scoring is explicitly unmodeled
This is a small downstream completeness gap and should not contaminate football-reality promotion. It can be added when event generation supports it.

## Exhale — changes selected now
1. Create one canonical environment application helper that updates both TeamState and TeamPlayerPool.
2. Route both Reality-v1 football and Reality-v1 fantasy-complete runners through that helper.
3. Add regression tests proving dome neutrality and open-air weather changes both scoring state and pass-opportunity state.
4. Make the current-value workflow rerun whenever canonical environment composition changes.
5. Preserve all existing conservation, market-blindness, causal, player-jurisdiction and source gates.

## Stillness — changes deliberately not forced now
The role-uncertainty/effectiveness and event-conditioned TD issues are real allocator-design improvements, but they alter calibrated player-distribution anatomy. They should not be silently rewritten in the same patch as a clear wiring repair without a replacement calibration target. They are recorded as next-kernel work, not forgotten.

## Recenter
The best immediate Monster is not the one with the most mechanisms. It is the one where every claimed mechanism reaches the correct jurisdiction exactly once. The current repair therefore prioritizes composition integrity over adding speculative features.