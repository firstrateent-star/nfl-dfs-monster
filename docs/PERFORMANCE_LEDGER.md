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
| 34292058208 | Final blind rehearsal | 60,000 | 12 | 720,000 | 2026090816 | PASS | Final blind Player Reality + QB gates passed. |

## 60K baseline — run 34292058208
- commit: `f33b6d693730443facce9a9e2f2f709c6c46976a`
- runner: GitHub hosted Ubuntu 24.04, runner 2.337.0
- workflow created/started: 2026-09-08 23:45:31 UTC
- runner job began: 23:45:35.325 UTC
- final cleanup complete: 23:51:45.307 UTC
- workflow API updated/completed: 23:51:48 UTC
- approximate workflow wall time: 6m17s from run start to API completion; runner execution about 6m10s
- simulation command start: 23:46:03.358 UTC
- simulation output complete: 23:51:43.136 UTC
- simulation wall time: ~339.78s = 5m39.8s
- simulation throughput: ~2,119 game-worlds/sec across 720,000 game-worlds
- non-simulation runner time: ~30.2s
- tests: 63 passed in 12.55s
- personnel baseline: ~3.49s
- 2025 + 2024 policy builds: ~2.56s combined
- health snapshot: ~0.24s
- continuity compile: ~0.45s
- broad Player Reality gate: ~0.24s
- QB gate: ~0.22s
- artifact upload/finalization: ~1.19s; 107,906 bytes; artifact id `10081863532`
- dominant bottleneck: simulation (~91.8% of runner execution)

## Baseline questions
1. Does simulation time scale approximately linearly from 10K to 60K?
2. What fraction of total runtime is simulation versus repeated data preparation?
3. Can roster/policy/health artifacts be safely cached by immutable input hash?
4. Can games or world batches run in parallel without changing seeded reproducibility?
5. Are CSV/Parquet serialization or artifact uploads material bottlenecks?
6. Can validation use streaming aggregates where raw worlds are unnecessary while preserving the exact evidence standard?
7. What is the fastest reproducible architecture that produces statistically identical football distributions?

Current evidence says optimization attention should focus first on the simulation kernel or safe parallel execution: setup, gates, and artifact upload are collectively small relative to the 60K simulation.

The objective is not simply faster code. It is **more validated simulated Sundays per unit of time and compute without losing truthfulness, reproducibility, or model independence.**
