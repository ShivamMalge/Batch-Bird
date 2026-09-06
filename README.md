# Batchbird

A mini columnar query engine in Rust, built to demonstrate *why* analytical databases are
fast — and to prove it with benchmarks rather than assert it.

DuckDB and ClickHouse get their speed from column-wise storage and vectorized execution
rather than row-at-a-time processing. Batchbird implements a minimal version of both and
measures the difference, with the same query executed four ways over the same data:

1. **Row-oriented** — array-of-structs storage, plain row loop
2. **Columnar, naive** — columnar storage, plain row loop
3. **Columnar, batched** — 1024-row `RecordBatch` pipeline, bitset filters, hash group-by
4. **Columnar, batched + SIMD** — `std::simd` filter compare and sum reduction

Splitting it that way is the point: the gap between 1 and 2 is attributable to *storage
layout*, and the gap between 2 and 3–4 to *execution model*. A benchmark that only compared
row-loop-over-columns against batched-over-columns could not tell those apart.

**The write-up is the deliverable, not the code.** The interesting output is an explanation of
where SIMD helps, where it does not, and why.

## Supported query shape

Exactly one, deliberately:

```sql
SELECT col1, SUM(col2) FROM t WHERE col3 <op> x GROUP BY col1
```

`<op>` is `=`, `<`, or `>`. Everything else — joins, multi-column `GROUP BY`, `HAVING`,
`ORDER BY`, other aggregates, subqueries, `AND`/`OR` — is rejected at parse time with a
message naming the construct. That rejection surface is most of the parser, and it is
intentional: an engine that silently ignored a clause it did not implement would return
confidently wrong answers.

## Building and testing

The default build is **stable Rust** (edition 2024). `std::simd` is still nightly-only, so
SIMD lives behind a cargo feature and the scalar path is never at risk:

```bash
cargo test                            # stable; scalar kernels
cargo clippy --all-targets            # stable
cargo +nightly test --features simd   # nightly; runs scalar AND SIMD, asserts they agree

cargo bench                           # criterion suite + CSV export
cargo run --release --example kernels                            # scalar kernel timings
cargo +nightly run --release --example kernels --features simd   # SIMD kernel timings
```

Every SIMD kernel has a scalar twin that is always compiled and always tested. With the
feature on, the tests run both and assert agreement — exact for integers, within a relative
tolerance for floats, because floating-point addition is not associative.

## Findings so far

Measured on this project's own hardware; see `phases.md` for machine and toolchain details.

**SIMD kernels, 16M rows:**

| kernel | scalar | SIMD | speedup |
|---|---|---|---|
| filter compare, Int64 | 18.87 ms | 12.89 ms | 1.46× |
| filter compare, Float64 | 13.68 ms | 8.61 ms | 1.59× |
| sum reduction, Int64 | 6.80 ms | 6.38 ms | **1.07×** |
| sum reduction, Float64 | 24.85 ms | 6.52 ms | **3.81×** |

The two sum rows look contradictory and share one explanation. Integer addition is
associative, so LLVM was already free to auto-vectorize the scalar `i64` loop and did —
leaving nothing for a hand-written version to win. Floating-point addition is *not*, so the
compiler must preserve serial order; writing `Simd<f64, 8>` accumulation is precisely the act
of granting permission to reorder. Both SIMD sums then land at the same wall time for the same
128 MB, consistent with a shared memory-bandwidth ceiling the `i64` loop had already reached.

The same property drives both halves: non-associativity is why the `f64` sum needs a test
tolerance, and why it is the one with a speedup to win.

**A bug the benchmark caught.** `Column::Utf8Dict` originally owned its dictionary, so every
`RecordBatch` cloned it — twice, at `Scan` and at `Filter` compaction. Over 2M rows the batch
pipeline degraded from 0.80× to **84×** *slower* than the row loop as group cardinality went
8 → 10,000, while an `Int64` group column over identical data stayed flat at ~1.1×. Sharing
the dictionary behind an `Arc` fixed it; both now sit at ~0.70×. The failure was invisible at
low cardinality — exactly where a casual benchmark would have looked.

## Layout

```
src/storage/   Column (Int64 / Float64 / Utf8Dict), Table, two-pass CSV loader
src/parser/    SQL text -> LogicalPlan, via sqlparser-rs; rejection-first
src/plan/      logical (the data) + physical (LogicalPlan -> operator tree)
src/exec/      Scan / Filter / Project / Aggregate, packed bitsets, kernels
src/simd/      vector kernels, feature-gated
src/bench/     row-oriented and columnar row-at-a-time baselines, data generation
benches/       criterion harnesses
tests/         cross-engine parity
```

## Design docs

These are the source of truth, and the code cites them constantly:

| doc | contents |
|---|---|
| `prd.md` | goals, non-goals, supported query shape, success criteria |
| `architecture.md` | modules, data flow, core types |
| `systemDesign.md` | the *why* behind each decision |
| `techstack.md` | toolchain, crates, benchmarking methodology |
| `phases.md` | build order, per-phase status, findings as they land |
| `agents.md` | guardrails for AI agents working in this repo |
| `mermaid.md` | reference diagrams |

## Scope

A learning and portfolio project, not a production system. Explicit non-goals, each a decision
rather than an omission: no joins, no multi-column `GROUP BY`, no SQL surface beyond the shape
above, no NULL handling, and no SIMD on string filtering.

That last one costs more than it looks like it does. Dictionary encoding had already reduced
`Utf8` equality to comparing `u32` codes — a dense numeric compare that would vectorize
cleanly — so the guardrail forecloses a win that was already sitting there.
