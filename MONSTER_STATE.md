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
Monster Beta22 v0.6.3 — Market-Blind Game Reality, with promoted QB rushing-reservoir repair and passing broad Player Reality structural gate.

## Passed / retained components
- Market-blind game-score architecture.
- Continuity-conditioned team identity.
- Shared finite game play supply.
- Receiving hierarchy and eligibility gates.
- Rushing anatomy / role structure.
- Health as separate availability/effectiveness/uncertainty state.
- Opportunity-conservation architecture.
- Evidence-gated promotion.
- QB season-mean rushing calibration.
- World-level QB rushing reservoir against finite TEAM rush attempts.
- Downstream rushing-role hierarchy preserves upstream QB carry/TD reservoir.
- Broad Player Reality structural gate passed on fresh 10K Week 1 sample.

## Latest completed evidence — Player Reality
GitHub Actions run `34291763174`, seed `2026090815`, 10,000 worlds per Week 1 game.

Run rebuilt current 2026 personnel (`--season 2026 --history 2025`), 2025/2024 football priors, Week 1 health snapshot, continuity-conditioned policy, and then simulated the 12-game Week 1 slate. Market data, salary and ownership were excluded.

### QB reconfirmation — PASS
- starter QB count: 24
- rush mean: 3.82736
- rush median: 3.50250
- rush p90: 6.24007
- rush p95: 7.26819
- rush max: 7.88290
- rushing-TD median: 0.13655
- rushing-TD p90: 0.41941
- rushing-TD max: 0.52300
- gate_pass: true

### Broad Player Reality structural gate — PASS
All 20 checks passed:
- finite player outputs
- team plays sane
- team pass attempts sane
- team rush attempts sane
- team targets <= pass attempts
- yards/play sane
- target ranks 1/2/3 sane
- rush ranks 1/2/3 sane
- target breadth sane
- core-rusher breadth sane
- starter QB count sane
- starter QB attempts/yards/TDs sane
- RB outputs nonnegative
- WR/TE outputs nonnegative

Observed structural metrics:
- player rows: 514
- team rows: 24
- team plays min/median/max: 53.10 / 60.44 / 65.19
- team pass attempts min/median/max: 26.14 / 30.73 / 35.98
- team rush attempts min/median/max: 23.25 / 27.46 / 31.45
- team targets min/median/max: 24.52 / 28.84 / 33.72
- yards/play min/median/max: 4.80 / 5.67 / 6.36
- mean target rank shares: 0.332 / 0.222 / 0.158
- mean rush rank shares: 0.631 / 0.225 / 0.102
- mean target earners/world: 7.21
- mean core rushers/world: 2.53
- starter QB pass-attempt median / p90: 29.95 / 33.26
- starter QB passing-yards median / p90: 219.19 / 237.75
- starter QB passing-TD median / p90: 1.399 / 1.677

Artifact: `monster-player-reality-gate`, artifact id `10081639259`.

## Current-data caveat
The run's personnel layer is current-season 2026 provider data, but provider injury rows were 0 in this run. Health therefore relied on the explicit Week 1 override snapshot dated 2026-09-07 (13 override rows). This is not a football-gate failure, but it is a freshness limitation that must be audited before final freeze.

## Personnel audit correction
Jacoby Brissett/Arizona and Tua Tagovailoa/Atlanta were previously suspected as mapping errors. Current 2026 evidence supported those Week 1 assignments at the time of audit. Do not alter them merely because they look surprising; reverify current personnel before freeze.

## Resolved blocker: QB rushing
The generic allocation and downstream A/B/C rushing hierarchy had been able to inflate/overwrite QB expected state. The promoted repair calibrates QB expected rushing from QB-season behavior, samples a finite QB reservoir against total team rush attempts, allocates residual carries among non-QBs, preserves the reservoir through final rushing-role refinement, and conserves exact team rush attempts/TDs.

## Current certification target
Broad Player Reality structure has passed. Before football freeze, the remaining certification path is:
1. deeper distribution/conservation/tail audit where current outputs permit it;
2. current-state personnel/health freshness audit;
3. 60,000-world final blind Week 1 rehearsal;
4. freeze only if those gates pass.

## Next actions
1. Audit player distributions/tails and role extremes from run `34291763174`; repair only evidence-backed failures.
2. Reverify Week 1 current personnel and health freshness, especially because provider injury feed returned 0 rows.
3. Run 60,000-world final blind Week 1 rehearsal.
4. Freeze as `Monster Football v1 — Week 1 FROZEN` only if final gates pass.
5. Reveal market only after freeze for audit/calibration; never tune upstream toward market.
6. Join FanDuel salary/player file after football freeze.
7. Produce Monster Player Value Board and then Monster 150 portfolio.

## DFS constitutional boundary
Football reality must be generated without sportsbook lines, DFS salaries, ownership, or optimizer feedback. Those layers are revealed only downstream after football worlds are frozen.

## Do not confuse with current state
Older v0.1/v0.2/v0.6.1 artifacts and prior Week 1 score maps are historical experiments. Run `34291763174` is the latest broad Player Reality evidence. Passing a structural gate means the tested invariants survived; it does not prove the forecasts are correct or eliminate the need for final freshness/tail/60K certification.
