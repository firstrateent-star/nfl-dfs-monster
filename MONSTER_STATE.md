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
- Rushing anatomy / role structure.
- Health as separate availability/effectiveness/uncertainty state.
- Opportunity-conservation architecture.
- Evidence-gated promotion: failed football candidates are not promoted.
- QB season-mean rushing calibration.
- World-level QB rushing reservoir against finite TEAM rush attempts.
- Downstream rushing-role hierarchy now preserves the upstream QB carry/TD reservoir.

## Latest completed evidence
GitHub Actions run `34258951529`, workflow `QB season-mean calibration repair`, seed `2026090814`, 10,000 worlds per Week 1 game, completed successfully on 2026-09-08.

The workflow itself rebuilt current-season personnel with `--season 2026 --history 2025`, rebuilt 2025 and 2024 policy inputs, applied the dated Week 1 health override snapshot, compiled continuity-conditioned policy, ran the full Week 1 Monster simulator, enforced the football gate, and promoted only after the gate passed.

QB starter rushing results from the passing run:
- starter QB count: 24
- rush mean: 3.75245
- rush median: 3.52935 (PASS; gate 2.4-4.6)
- rush p90: 6.31247 (PASS; gate <= 8.25)
- rush p95: 7.08962 (PASS; gate <= 10.0)
- rush max: 7.70420 (PASS; gate <= 11.5)
- rushing-TD median: 0.08360 (PASS; gate <= 0.22)
- rushing-TD p90: 0.45716 (PASS; gate <= 0.50)
- rushing-TD max: 0.54670 (PASS; gate <= 0.65)

Promotion commit: `97d55bf719d41e19496e54a4c57a7037b652e768` — `Preserve QB rushing reservoir through role hierarchy`.
Artifact: `monster-qb-role-reservoir-repair`, artifact id `10069106913`.

## What "current" means for this run
- Current roster/depth/personnel layer: 2026 data loaded during the GitHub Actions run through the nflverse/nflreadpy ingestion path.
- Historical behavioral priors: primarily 2025, with 2024 used in continuity-conditioned team identity.
- Week 1 health overrides: explicit snapshot dated 2026-09-07.
- Week 1 slate encoded in the runner for games on 2026-09-13.
- Market data, DFS salary and ownership are excluded from football simulation.

Important limitation: `current` does not mean every real-world input is guaranteed updated to the exact second. Provider data can lag, and the explicit health override file is dated 2026-09-07. Therefore the next certification stage must include a current-state/personnel freshness audit before final football freeze.

## Personnel audit correction
The earlier suspicion that Jacoby Brissett/Arizona and Tua Tagovailoa/Atlanta represented bad team mappings was incorrect. Current 2026 evidence supports Brissett as Arizona's Week 1 starter/presumed starter and Atlanta officially named Tagovailoa its Week 1 starter on 2026-09-07. Do not treat those assignments as roster bugs.

## Resolved blocker: QB rushing
The prior failure was caused by two downstream normalization layers. First, generic allocation could inflate a compiled QB share when non-QB roles disappeared. Second, the later A/B/C rushing hierarchy could overwrite the calibrated QB reservoir and promote a mobile QB into generic Role A. The promoted repair now:
1. calibrates QB expected rushing from QB-season starter behavior;
2. interprets QB rush share against total TEAM rush attempts;
3. samples a finite QB reservoir at the world level;
4. allocates the residual carry supply among non-QBs;
5. preserves the QB reservoir through the final A/B/C rushing-role refinement;
6. conserves exact team rushing attempts and rushing TDs.

## Current blocker / certification target
QB Reality has cleared its evidence gate. The next blocker is broader Player Reality certification, including conservation, passing coherence, receiving hierarchy, RB opportunity structure, participation/health behavior, and distribution/tail sanity. A pass here is required before the 60,000-world final blind rehearsal.

## Next actions
1. Run broader player-reality/conservation audit against the promoted branch.
2. Diagnose and repair only evidence-backed failures.
3. Audit current-state/personnel/health freshness before final freeze.
4. Run 60,000-world final blind Week 1 rehearsal.
5. Freeze football model if gates pass.
6. Reveal market only after football freeze for audit/calibration; do not tune upstream toward market.
7. Join FanDuel salary/player file only after football freeze.
8. Produce Monster Player Value Board: mean/median/tails, boom probabilities, salary efficiency, threshold probability, correlations, and optimal-lineup rate.
9. Build the first Monster 150 as a portfolio across valuable simulated Sunday states rather than 150 near-identical projection-max lineups.

## DFS constitutional boundary
Football reality must be generated without sportsbook lines, DFS salaries, ownership, or optimizer feedback. Those layers are revealed only downstream after the football worlds are frozen.

## Do not confuse with current state
Older v0.1/v0.2/v0.6.1 artifacts and prior Week 1 score maps are historical experiments, not the current production state. Run `34258951529` and promotion commit `97d55bf719d41e19496e54a4c57a7037b652e768` are the latest QB-rushing evidence. The broader player-reality layer is not yet frozen.
