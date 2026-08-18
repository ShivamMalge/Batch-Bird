# Phases

Each phase carries a **Status** line — per `agents.md`, a phase is only marked `done`
when its tests pass.

## Phase 0 — Setup
**Status:** ✅ done (2026-08-18) — both toolchain paths build and tests pass.
- `cargo init --lib` as `batchbird`, edition 2024. Deps: `csv` 1.4, `sqlparser` 0.62,
  `hashbrown` 0.17; dev-dep `criterion` 0.8 with `html_reports` + `csv_output`
  (neither is a criterion default; both are needed for the Phase 7 chart deliverable).
- `simd` cargo feature defined, gating `#![feature(portable_simd)]` at the crate root so
  the default build stays on stable.
- `[profile.bench]`: `codegen-units = 1` (deterministic inlining → stable measurements),
  `debug = true` (profile the two group-by phases separately), `lto = thin`.
- **Toolchain confirmed empirically**, not just researched: `src/simd/mod.rs` holds a
  smoke test (vectorized vs. scalar i64 sum) proving `std::simd` compiles and is correct
  on the installed nightly. It also pins the Phase 5 invariant — every SIMD kernel has a
  scalar twin under the same test — and flags that lane arithmetic wraps where debug-mode
  scalar `+` panics. Replace it with the real kernels in Phase 5.

Verified commands:
```
cargo test                            # stable; SIMD module compiled out
cargo +nightly test --features simd   # nightly; smoke test passes
```

## Phase 1 — Storage Layer
**Status:** not started
- `Column` enum (`Int64`, `Float64`, `Utf8Dict`), `Table` (`HashMap<String, Column>` + row count).
- CSV loader: infer or declare column types, build dictionary encoding for string columns.
- Unit tests: load a small CSV, assert column contents and dictionary correctness.

## Phase 2 — SQL Parsing
**Status:** not started
- `sqlparser-rs` -> AST for the scoped query shape only (`WHERE` op ∈ {=, <, >}).
- Reject (clear error) anything outside scope: joins, multi-column GROUP BY, unsupported clauses.

## Phase 3 — Naive Baseline
**Status:** not started
- Plain row-scan loop (no operator abstraction) implementing the same query shape.
- This is the row-at-a-time control for the benchmark — build it before the batch engine.

## Phase 4 — Batch Execution Pipeline
**Status:** not started
- `RecordBatch`, `Operator` trait, `Scan`/`Filter`/`Project`/`Aggregate`.
- Bitset filter with compaction (see systemDesign.md).
- Hash-based two-phase group-by with `GroupKey(u64)`/`Accumulator<T>`
  (both pinned — see agents.md "Resolved Decisions").
- Correctness tests: batch pipeline output matches naive baseline output on same input.

## Phase 5 — SIMD
**Status:** not started
- `std::simd` for filter comparison and sum reduction, scoped to `Int64`/`Float64` only.
- Scalar version must exist and pass tests first; SIMD lives behind the `simd` cargo
  feature (nightly-only: `cargo +nightly test --features simd`) so scalar correctness on
  stable is never at risk.

## Phase 6 — Sort-Based Grouping + Full Benchmark Suite
**Status:** not started
- Implement sort-then-run-length-aggregate as a secondary group-by strategy.
- `criterion` benchmarks: naive vs. batched vs. SIMD; hash-group vs. sort-group, phases
  profiled separately (build-index vs. scatter-accumulate).

> **⚠️ Do not unify the naive baseline with the batch engine here.** This is exactly the
> phase where the temptation shows up — you'll be staring at both implementations side by
> side while writing benchmarks and it'll look like they share filter logic worth extracting.
> Resist it. The naive baseline exists specifically to isolate "row-at-a-time" as a variable;
> sharing code with the batch engine defeats that isolation. See `agents.md` Benchmarking
> Expectations for the same rule.

## Phase 7 — Write-Up
**Status:** not started
- Benchmark chart(s).
- Explicit findings: where SIMD helps and where it doesn't, and why; hash-group vs.
  sort-group tradeoff at benchmark scale; documented scope cuts (no joins, no multi-column
  group-by, no SIMD on strings) stated as decisions, not omissions.

## Explicit Non-Goals (do not implement without revisiting prd.md)
- Joins, multi-column GROUP BY, full SQL surface, SIMD string matching.
