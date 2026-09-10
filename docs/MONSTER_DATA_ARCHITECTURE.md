# MONSTER DATA ARCHITECTURE

Status: WORKING GOVERNANCE — live schema reconciled 2026-09-10

## Center

Monster stores evidence so football reality can be reproduced, audited and recalibrated without allowing downstream market/DFS information to contaminate the blind simulator.

Canonical data path:

source evidence -> versioned features/current state -> model run -> raw simulation artifacts -> compact football summaries/audits -> blind freeze -> downstream market/DFS audit

The database is not the simulator. Postgres owns durable identity, provenance, run metadata, compact summaries and audit indices. Large world-level and play/drive-level matrices remain columnar artifacts (Parquet/CSV) referenced by the database.

## Live database reality

The live Supabase project uses a private `monster` schema. `anon`/public access is revoked. The observed live schema contains:

### Identity / slate reality
- `teams`
- `team_aliases`
- `players`
- `slates`
- `games`
- `slate_players`

### Provenance / feature governance
- `source_files`
- `feature_registry`
- `experiment_registry`
- `player_features`
- `team_features`
- `v_latest_player_features`

### Run / recomputation governance
- `model_runs`
- `dependency_rules`
- `dirty_queue`
- `meta`

### Football outputs
- `game_projections`
- `team_projections`
- `player_projections`
- `v_canonical_team_points`
- `simulation_artifacts`

### Audit / downstream boundary
- `audit_events`
- `market_lines`
- `market_audits`

Market tables are downstream only and must never be queried by blind football simulation code.

## Storage law

### Postgres stores
- stable IDs and aliases;
- source provenance and hashes;
- feature definitions/governance;
- current or versioned compact feature values;
- model-run identity, parentage, seed, world count and blind status;
- game/team/player distribution summaries;
- artifact pointers/hashes;
- compact audit results and causal diagnoses;
- dependency/recompute state.

### Object/artifact storage stores
- world-level score matrices;
- player-world matrices;
- raw play traces;
- raw drive traces;
- large historical evidence extracts;
- Stage 3 anatomy tables;
- Drive Survival Ledger raw Parquet;
- other high-volume experiment outputs.

Do not put millions of world/play rows into transactional Postgres unless a specific query workload proves that is necessary.

## Drive Survival Ledger placement

Stage 3 drive-survival work follows the same architecture:

1. A `model_runs` row identifies the simulated candidate and exact seed/world count.
2. Historical and simulated raw drive traces are written as Parquet artifacts.
3. `simulation_artifacts` records their storage paths, hashes and row counts when persisted.
4. `audit_events` records compact diagnosis (series survival, third-down survival, three-and-out rate, third-and-long creation, early-down chunk creation).
5. No new football coefficient receives authority merely because an audit mismatch exists.
6. A bounded causal mutation creates a child run via `parent_run_id`, then paired evidence determines whether the mechanism improved.

## Current live-database usage gap

The live database is structurally useful but under-populated relative to the current repository engine. At reconciliation time it contained only a small set of early Beta22 source/run records and compact game/team projections; current v1/v1.3 GitHub Actions artifacts are not yet comprehensively registered in `model_runs`, `simulation_artifacts` and `audit_events`.

This is an operational integration gap, not a reason to redesign the simulator. The correct fix is to add a persistence step after an experiment passes software integrity, not to make the simulator depend on live database writes while running.

## Migration-history drift

The live Supabase migration history records timestamped migrations (`monster_core_schema`, `monster_fk_indexes`, `add_team_aliases_for_league_universe`), while the repository currently retains older numbered SQL files whose object names do not exactly match the live schema. Therefore:

- live database state is the current operational reality;
- repository migration history must be reconciled before new schema DDL is introduced;
- do not apply the old numbered migration files blindly to the live project;
- Drive Survival Ledger v1 requires no new table and should use existing artifact/audit structures first.

## Causal data layers

### Layer 0 — Source truth
NFL/nflverse, roster/depth, injury/current-state, weather/environment, Madden/scouting proxy, physical/athletic evidence and explicitly governed manual overrides.

Every source should have identity, effective time, ingestion time and content hash.

### Layer 1 — Persistent identity
Player/team ability evidence that should not swing simply because one weekly state changes.

### Layer 2 — Current circumstance
Availability, effectiveness-if-active, uncertainty, role, recent workload, depth changes, weather and other refreshable state.

### Layer 3 — Unit/system state
OL, receiving corps, backfield, front, coverage, special teams, coaching/system/continuity and team strategic priors.

### Layer 4 — Matchup/game state
Opponent interactions, venue, home field and the initial football state used by a world.

### Layer 5 — World realization
World-level active personnel, possessions, play decisions, play outcomes, clock transitions, drives, scoring and player statistics.

### Layer 6 — Blind football summaries
Game/team/player distributions and mechanism audits generated without market/DFS authority.

### Layer 7 — Post-freeze downstream
Market reveal, DFS salaries/scoring, ownership, OLR and lineup portfolio construction.

## Priority build sequence

1. Drive Survival Ledger and excess-punt localization.
2. World-level personnel availability/substitution semantics.
3. Red-zone/scoring-state ecology.
4. Clock/possession calibration and possession opportunity accounting.
5. Run second-level/open-field explosive ecology.
6. Pressure/coverage-conditioned pass failure ecology.
7. Defensive tactical intent.
8. Deeper assignment-level matchup resolution.
9. Coaching/system expansion.
10. Special-teams depth and travel/rest/environment expansion.
11. Longitudinal score-distribution calibration.
12. Final blind freeze -> downstream DFS generation.

## Promotion law

Database completeness is not football completeness. A source row, feature row or green CI run does not make a mechanism ACTIVE. Promotion requires:

- source provenance;
- definition-matched historical evidence;
- demonstrated causal jurisdiction;
- conservation/invariant tests;
- paired same-seed evidence;
- multiple-seed stability where appropriate;
- no market leakage;
- an artifact/audit trail that can reproduce the decision.
