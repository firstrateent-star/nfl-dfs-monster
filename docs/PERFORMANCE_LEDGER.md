# Monster Performance Ledger

Purpose: preserve runtime evidence for every meaningful Monster simulation so engineering can improve speed/cost without weakening football fidelity or evidence gates.

## What to record
For each validation/rehearsal run:
- GitHub run ID and commit SHA
- model/workflow version
- worlds per game and game count
- total simulated game-worlds
- seed
- queue time
- environment/setup time
- data/personnel build time
- policy/health/continuity build time
- simulation wall-clock time
- evidence-gate time
- artifact-upload time
- total workflow wall-clock time
- success/failure and failure stage
- runner/OS when available
- artifact size when available
- worlds/second and game-worlds/second
- estimated compute scaling vs prior runs

## Optimization rule
Performance changes may optimize orchestration, caching, vectorization, parallelism, serialization, artifact size, and redundant data preparation. They must not reduce world count, remove evidence gates, introduce market/DFS information upstream, or change football behavior merely to make the simulator faster.

## Known runs

| Run | Stage | Worlds/game | Games | Total game-worlds | Seed | Result | Notes |
|---|---|---:|---:|---:|---:|---|---|
| 34258951529 | QB repair evidence | 10,000 | 12 | 120,000 | 2026090814 | PASS | QB reservoir repair promoted. |
| 34291763174 | Player Reality certification | 10,000 | 12 | 120,000 | 2026090815 | PASS | QB reconfirmed; broad Player Reality 20/20. |
| 34292058208 | Final blind rehearsal | 60,000 | 12 | 720,000 | 2026090816 | RUNNING | Record exact step timings after completion. |

## Baseline questions after 60K completes
1. Does simulation time scale approximately linearly from 10K to 60K?
2. What fraction of total runtime is simulation versus repeated data preparation?
3. Can roster/policy/health artifacts be safely cached by immutable input hash?
4. Can games or world batches run in parallel without changing seeded reproducibility?
5. Are CSV/Parquet serialization or artifact uploads material bottlenecks?
6. Can validation use streaming aggregates where raw worlds are unnecessary while preserving the exact evidence standard?
7. What is the fastest reproducible architecture that produces statistically identical football distributions?

The objective is not simply faster code. It is **more validated simulated Sundays per unit of time and compute without losing truthfulness, reproducibility, or model independence.**
