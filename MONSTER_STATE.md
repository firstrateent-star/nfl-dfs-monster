# MONSTER STATE

Last updated: 2026-09-08
Branch: `feature/slate-simulator-v1`

## What we are building
The Monster is an independent NFL slate simulator. Its primary job is to simulate coherent football worlds from football/context evidence before any sportsbook, salary, ownership, or DFS optimization information is introduced.

Canonical path:

NFL information -> personnel reality -> team reality -> game reality -> player reality -> simulated Sundays -> blind freeze -> DFS scoring -> salary -> player value -> ownership -> lineup value -> portfolio -> actual-results calibration.

## Primary objective
Predict distributions of NFL game and player outcomes by simulating finite, correlated worlds. DFS value is downstream of football reality; the optimizer must never decide what football reality looks like.

## Current production game state
Monster Beta22 v0.6.3 — Market-Blind Game Reality.

Upstream game-score inputs include 2025 offense scoring/yards, opponent defense points/yards allowed, TD composition, home-field margin shift, continuity-conditioned team identity, current personnel/health context, and documented role adjustments. Sportsbook totals/spreads are excluded upstream.

## Passed / retained components
- Market-blind game-score architecture.
- Continuity-conditioned team identity.
- Shared finite game play supply.
- Receiving hierarchy and eligibility gates.
- Rushing anatomy / role structure except current QB mean-carry calibration.
- Health as separate availability/effectiveness/uncertainty state.
- Opportunity-conservation architecture.
- QB rushing-TD calibration in the latest 10k test.
- Evidence-gated promotion: failed football candidates are not promoted.

## Current evidence
Latest completed 10k Week 1 validation: GitHub Actions run `34256283755`, seed `2026090812`.

QB starter rushing results:
- historical 2022-25 QB-season median carries/start: 3.35
- Monster median mean carries: 6.79 (FAIL)
- historical p90: 7.0
- Monster p90 mean: 9.99 (FAIL)
- Monster p95 mean: 10.59 (FAIL)
- Monster maximum mean: 12.03 (FAIL)
- rushing-TD median: 0.083 (PASS)
- rushing-TD p90: 0.460 (PASS)
- rushing-TD max: 0.545 (PASS)

The failed repair was not promoted.

## Personnel audit correction
The earlier suspicion that Jacoby Brissett/Arizona and Tua Tagovailoa/Atlanta represented bad team mappings was incorrect. Current 2026 evidence supports Brissett as Arizona's Week 1 starter/presumed starter and Atlanta officially named Tagovailoa its Week 1 starter on 2026-09-07. Do not treat those assignments as roster bugs.

## Current blocker
QB expected rushing attempts remain too high. Diagnosis: the compiled QB reservoir can be inflated downstream because all rushing roles are sampled/renormalized together. When non-QB rushers drop from a world's role tree, the QB share can expand above its intended expected share of TEAM rush attempts.

## Current experiment
Candidate world-level QB reservoir repair:
1. retain season-level QB tendency calibration in `skill_pools.py`;
2. interpret compiled QB rush share explicitly against total team rush attempts;
3. sample a finite QB rushing reservoir at the world level;
4. allocate remaining carries only among non-QBs;
5. preserve world-level QB tail variance without allowing role-tree renormalization to redefine the mean;
6. leave the already-passing QB rushing-TD mechanism unchanged.

Candidate patch script: `scripts/apply_qb_world_reservoir_fix.py`.
Evidence workflow: `.github/workflows/qb-world-reservoir-repair.yml`.

## Evidence gate for current experiment
10,000 Week 1 worlds, new seed `2026090813`.
Promotion only if:
- QB median mean carries: 2.4 to 4.6
- p90 <= 8.25
- p95 <= 10.0
- max <= 11.5
- rushing-TD median <= 0.22
- rushing-TD p90 <= 0.50
- rushing-TD max <= 0.65
- lint/tests pass.

## Next actions
1. Run the QB world-reservoir 10k evidence gate.
2. If it fails, diagnose the mechanism from evidence and do not promote.
3. If it passes, promote only the verified repair.
4. Run broader player-reality/conservation audit.
5. Run 60,000-world final blind Week 1 rehearsal.
6. Freeze football model if gates pass.
7. Join FanDuel salary/player file only after football freeze.
8. Produce Monster Player Value Board: mean/median/tails, boom probabilities, salary efficiency, threshold probability, correlations, and optimal-lineup rate.
9. Build the first Monster 150 as a portfolio across valuable simulated Sunday states rather than 150 near-identical projection-max lineups.

## DFS constitutional boundary
Football reality must be generated without sportsbook lines, DFS salaries, ownership, or optimizer feedback. Those layers are revealed only downstream after the football worlds are frozen.

## Do not confuse with current state
Older v0.1/v0.2/v0.6.1 artifacts and prior Week 1 score maps are historical experiments, not the current production state. The latest validated evidence and this file take precedence.
