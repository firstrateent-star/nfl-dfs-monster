# NFL DFS Monster — Free-Stack Slate Simulator

The Monster's upstream job is **football only**:

1. simulate how many points each NFL game/team scores;
2. simulate how opportunities and scoring are allocated to players;
3. freeze the football outputs;
4. only then translate those outputs into FanDuel / DraftKings scoring and tournament economics.

## Free architecture

- **Supabase Free / Postgres** — canonical facts, features, snapshots, run metadata, summaries.
- **Supabase Storage Free** — current-slate raw snapshots and compressed current simulation artifacts only.
- **GitHub Free repo** — code, SQL migrations, feature registry, tests, workflows. Keep it private if you prefer the codebase non-public.
- **GitHub Actions** — light scheduled ingestion + manual heavy simulation; budgeted for the free allowance.
- **Python + NumPy** — Monte Carlo.
- **Polars + DuckDB + Parquet** — efficient analytical data processing and world-matrix queries.
- **nflreadpy / nflverse** — primary free football data backbone, including rosters, depth charts, injuries, snap counts and defensive advanced statistics.
- **NOAA/NWS API** — free U.S. weather/conditions.

## Key design law

> Complexity is retained in the evidence graph. Speed comes from caching, vectorization, columnar storage, and recomputing only the dependency cone of changed information.

## Pipeline

```text
FREE RAW SOURCES
    ↓
INGEST / SNAPSHOT / HASH
    ↓
CANONICAL SUPABASE STATE
    ↓
FEATURE COMPILER
    ↓
ALL-PLAYER PARTICIPATION + SNAP WEIGHTS
    ↓
OL / PASS RUSH / COVERAGE / RUN DEFENSE / SPECIAL TEAMS UNIT MECHANISMS
    ↓
SLATE SNAPSHOT (market blind)
    ↓
12 INDEPENDENT GAME SIMULATORS
    ↓
TEAM SCORE DISTRIBUTIONS
    ↓
PLAYER OPPORTUNITY + STAT ALLOCATION
    ↓
SLATE WORLDS
    ↓
FROZEN FOOTBALL OUTPUTS
    ↓
FanDuel / DraftKings decoders (later)
```

## Full player universe + snap counts

The Monster player universe comes from NFL/nflverse rosters, **not DFS player lists**. Offense, offensive line, defense and special teams are all eligible to influence the football simulation.

Snap share is a first-class weighting layer:

- offensive snap share weights offensive personnel / OL influence;
- defensive snap share weights pass rush, coverage and run-defense influence;
- special-teams snap share weights kicking/return/ST influence;
- recent snap volatility contributes to personnel uncertainty.

A player's size/Madden/statistical capability does not matter at full strength simply because he is on the roster. It matters in proportion to how often the model expects him to be on the field and whether he is active/effective.

For Week 1, recent prior-season snap counts provide the baseline and current depth chart/injury/role evidence can update that prior. As the current season progresses, current-season snap counts become increasingly authoritative.

## Mechanism-specific personnel model

All-player evidence is aggregated into bounded team mechanisms before scoring:

- pass protection
- run blocking
- pass rush
- coverage
- run defense
- special teams

These mechanisms affect drive success, touchdown/field-goal/turnover probabilities and uncertainty. They do **not** add fantasy points directly.

For example, an elite edge rusher projected for 25% of defensive snaps has less influence than a slightly weaker edge projected for 90% of snaps. The same logic applies to OL, corners, safeties, linebackers and special-teamers.

## Market anti-leakage

Sportsbook totals, spreads, implied totals, ownership and DFS projections live in a downstream namespace. The football snapshot builder and simulator are not allowed to query them.

## Quick start

```bash
cp .env.example .env
uv sync --dev
uv run monster validate-config
uv run pytest
```

Once Supabase exists:

```bash
supabase link --project-ref YOUR_PROJECT_REF
supabase db push
uv run monster db-check
```

## Current build stage

This repository is the permanent infrastructure spine. The current working milestone is **Monster Slate Simulator v1**: compile a rich market-blind Week 1 snapshot using the full NFL roster universe, snap-weighted personnel mechanisms, and then run team-score + player-allocation worlds.

## Live Supabase project
- Project ID: `nuyvuwqvsgopapjbnndu`
- URL: `https://nuyvuwqvsgopapjbnndu.supabase.co`
- Plan: Free
- Region: `us-east-1`
- Private schema: `monster`
- Public/anon access to `monster` is revoked.

The canonical v0.6.3 market-blind run, 24 teams, 12 games, feature definitions, dependency rules, and snap/unit feature governance are seeded in the live database.
