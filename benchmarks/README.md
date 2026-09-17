# 2026 Week 1 Reality Benchmark

This benchmark grades Monster's simulated football worlds against realized Week 1 outcomes without allowing final results to influence the simulation.

## Evidence classes

- **A0 — frozen pre-lock evidence:** the original 10,000-world Sunday main-slate run made before the 12 benchmark games. This is the cleanest true out-of-sample evidence and remains the historical reference.
- **A1 — protected v6.3 replay:** the validated v6.3 code at `5fd1e5b9cc664046efe5f6cf40231a14dc64e5ff`, replayed with a Week-1 input freezer. This is a retrospective architecture test because the v6.3 code itself was finalized after the Sunday games; it must not be described as a pre-game prediction.
- **B — DEN@KC:** separate later v6.3/pre-lock evidence. It is never mixed into the A0/A1 Sunday scorecard because it came from a different model/input state.

## Anti-leakage boundary

The A1 replay uses only 2025/2024 football-history priors, Week-1 roster/depth/injury state, the explicit `config/health/week1_2026_2026-09-07.csv` overrides, and Madden 27 attributes pinned to the pre-season mirror commit `ad0350f3b47b5559f2b2580dab9d9403acd4f4d0` (2026-08-03). 2026 regular-season snap counts are forcibly disabled. Final scores and player results are read only by `audit_week1_reality_benchmark.py` after simulation files already exist.

## Primary diagnostics

The benchmark intentionally scores both **accuracy** and **distribution calibration**:

- team score, total and margin MAE/RMSE/bias;
- empirical percentile and CRPS of the realized score inside simulated worlds;
- P10–P90 and P05–P95 coverage;
- between-matchup expected-total standard deviation versus actual Week-1 total standard deviation;
- mean within-game total standard deviation;
- FanDuel and component-stat player calibration from `player_world_fanduel.csv`;
- rushing and target role-plan share error versus realized opportunities.

The benchmark is evidence, not a fitting target. A mechanism is not promoted merely because it moves Week-1 scores closer to reality; changes must remain causal, survive paired-seed replay, and preserve already-validated football behavior.
