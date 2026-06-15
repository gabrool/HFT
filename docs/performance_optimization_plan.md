# MMRT Performance Optimization Plan

This document tracks the staged optimization work for the market-making
pipeline. The first draft PR adds the benchmark and golden-output harness that
later PRs must keep green while changing hot paths.

## Draft PR Sequence

1. Benchmark and safety rails: add deterministic fixtures, heavy-file inventory,
   benchmark JSON output, and golden-output checks.
2. Pipeline IO and feature hot paths: batch writers/readers, ring-buffer window
   reductions, vectorized labels, and reader/split caching.
3. Execution and adverse selection: vectorized tape/grid writes, execution-env
   caches, conservative-fill reuse, and adverse dataset batching.
4. Linear, RL, and analysis: shared-head scans, streaming metrics, signal stats
   pass reduction, PPO preallocation, and audit accumulator reuse.

## Safety Rules

- Keep current artifact schemas, manifest fields, CLI defaults, and validation
  behavior unchanged.
- Prefer internal batch/trusted helpers over public API changes.
- Treat output equivalence as the gate: exact array equality where possible,
  tight tolerances only for floating-point training math.
- Keep draft PRs small enough to review and benchmark independently.

## Running The Baseline

```powershell
python -m mmrt.cli.benchmark_pipeline --iterations 3 --work-root work/perf-baseline --output-json work/perf-baseline.json
```

Use `--no-optional` when optional runtime packages such as Torch are not
available locally.
