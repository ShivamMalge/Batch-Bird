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

**The write-up is the deliverable, not the code:** [WRITEUP.md](WRITEUP.md) — where SIMD helps,
where it does not, and why.

Short version: the kernels are 1.46–3.81x faster in isolation and the queries are ~1.00x, and
the Amdahl arithmetic predicts that to within 0.1 percentage points. Vectorizing the two hot
loops this engine has does not make its queries faster.

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

Measured on an AMD Ryzen 7 5800HS, pinned to one core with boost disabled, arms alternated
inside a single process and reported as minimum-of-N. The harness's own resolution is measured,
not assumed: an A/A null (the same arm registered twice) returns **median 1.56%, worst 5.68%**,
and nothing smaller than that is reported as an effect. Full detail in `phases.md`; raw output
in `results/`.

**SIMD kernels are fast in isolation and invisible at query level.**

| kernel, 16M rows | scalar | SIMD | speedup |
|---|---|---|---|
| filter compare, Int64 | 18.87 ms | 12.89 ms | 1.46× |
| filter compare, Float64 | 13.68 ms | 8.61 ms | 1.59× |
| sum reduction, Int64 | 6.80 ms | 6.38 ms | 1.07× |
| sum reduction, Float64 | 24.85 ms | 6.52 ms | **3.81×** |

Yet running whole queries with the same kernels, alternated scalar against SIMD in one process:
every speedup lands in **0.94–1.05×** — inside the noise floor, at every cardinality, for both
group-by strategies. The kernels are a small slice of a query that also hashes, allocates,
compacts and scatters, and Amdahl does the rest. **Vectorizing the two hot loops this engine has
does not make its queries faster.** That is the finding, and it is only visible because the
query-level number was measured rather than extrapolated from the kernel.

The two sum rows explain each other. Integer addition is associative, so LLVM already
auto-vectorized the scalar `i64` loop — nothing left to win. Floating-point addition is not, so
the compiler must preserve serial order; writing `Simd<f64, 8>` is precisely the act of granting
permission to reorder. Both SIMD sums then land at the same wall time for the same 128 MB,
against a shared memory-bandwidth ceiling of ~20 GB/s.

**Row-oriented vs columnar is mostly dictionary encoding, not memory layout.** The row arm is
~1.7× slower at fixed cardinality. A controlled probe — padding the row struct from 24 B to
144 B while reading the same three fields — shows cache-line utilization is real but weak
(6× the width buys 1.65× the cost), and that **below ~64 B per row the row layout is actually
faster** than columnar, because one sequential stream beats three. At the engine's 32 B row the
probe reports 0.93×, against the engine's 1.67×. The difference is the group key: a `Box<str>`
with a pointer chase and string hash per row, versus a `u32` dictionary code.

**Sort-based grouping beats hash at high cardinality**, refuting this project's own design-doc
prediction. Below 32k groups hash wins by 1.3–2.4×; at 250k groups sort wins by 0.68–0.90×
across runs. An LSD radix sort was implemented specifically to check whether "sort loses" was a
statement about the strategy or about pdqsort — it is about the strategy: radix does not change
any verdict's direction, though its advantage grows with cardinality as an O(n) sort should.

**Hash group-by carries a variance that sort does not.** Five fixed hasher seeds span **10.7%**
at 250k cardinality — the same order as the effects under study — purely from collision pattern.
A sort has no collision pattern and no such component. Every hash figure above is read with that
band, and comparisons falling inside it are reported as not separable rather than as results.

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
