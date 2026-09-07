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
- **nflreadpy / nflverse** — primary free football data backbone.
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
MECHANISM COMPILER
    ↓
SLATE SNAPSHOT (market blind)
    ↓
12 INDEPENDENT DRIVE-BASED GAME SIMULATORS
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

## Mechanism compiler

The simulator now has a bounded mechanism layer for the extra Monster metrics.

Player inputs include height, weight, wingspan, 40 time, Madden speed/acceleration/route-running/catching, age + career workload, health/effectiveness and continuity. These compile into small football-specific modifiers such as explosive-play ability, catch-point conversion, rushing efficiency and role uncertainty.

Team inputs include derived Madden/team strength, offensive-line and opponent-front indices, continuity, coaching entropy, wind, precipitation, temperature and dome state. These compile into bounded personnel/environment effects before game simulation.

Constitutional rules enforced by tests:

- missing extra data is neutral;
- physical/Madden effects are bounded rather than direct point bonuses;
- age alone cannot reduce expected production;
- age + workload can increase biological uncertainty;
- astrology is retained as a traceable shadow signal with zero production authority;
- player mechanism changes cannot retroactively change an already-generated team scoring world;
- all player opportunities and touchdowns are allocated from the same team worlds.

The canonical single-game call is `simulate_monster_game(...)`:

```text
compile rich football evidence
→ simulate drives / TD / FG / turnovers / team points
→ allocate plays / targets / carries / TDs / yards to players
```

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

The permanent infrastructure spine, drive-based scoring layer, player allocation layer, and bounded mechanism compiler are implemented on the Slate Simulator v1 development branch. The next modeling milestone is to populate a real Week 1 market-blind snapshot from nflverse + Supabase and run the 12-game slate end to end.

## Live Supabase project
- Project ID: `nuyvuwqvsgopapjbnndu`
- URL: `https://nuyvuwqvsgopapjbnndu.supabase.co`
- Plan: Free
- Region: `us-east-1`
- Private schema: `monster`
- Public/anon access to `monster` is revoked.

The canonical v0.6.3 market-blind run, 24 teams, 12 games, feature definitions, and dependency rules are already seeded in the live database.
