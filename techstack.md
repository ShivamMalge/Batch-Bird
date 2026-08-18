# Tech Stack

## Language / Toolchain
- Rust, **edition 2024** — resolved (stable since Rust 1.85; dev machine has stable 1.94.0
  and a nightly toolchain installed).
- `std::simd` (`portable_simd`) — **still nightly-only as of 2026-08**; stabilization remains
  blocked upstream. Resolution: the default build and all tests run on **stable**; SIMD code
  sits behind a `simd` cargo feature compiled with nightly
  (`cargo +nightly test --features simd`, `cargo +nightly bench --features simd`).
  The scalar path is always present and never feature-gated — see phases.md Phase 5.

## Crates
| Crate | Purpose |
|---|---|
| `csv` | CSV loading into `storage::Table` |
| `sqlparser` (sqlparser-rs) | SQL string -> AST; we do not hand-roll a parser |
| `criterion` | Benchmarking (naive vs. vectorized vs. SIMD, hash-group vs. sort-group) |
| `hashbrown` | Group-by index. Chosen over bare `std::collections::HashMap` because std's default SipHash is slow on small integer keys and would dominate the build-group-index phase timing — the benchmark should measure group-by structure, not hasher overhead. (`std` HashMap + `ahash`/`fxhash` is an acceptable equivalent.) |

## Deliberately Not Used
- No async runtime — this is a single-threaded, in-memory batch engine.
- No existing query engine crate (DataFusion, Polars) as a dependency — the point of the
  project is implementing the vectorized execution ourselves.
- No `Box<dyn Accumulator>` — `Aggregate` is generic over `A: Accumulator<T>`, monomorphized,
  since only one concrete accumulator (`SumAccumulator`) exists today. Revisit only if a
  future query needs heterogeneous accumulators chosen at plan time (e.g. `SUM` + `COUNT`
  in one query).

## Benchmarking Methodology
- Dataset: synthetic CSV, a few million rows, mixed Int64/Float64/Utf8 columns.
- Comparisons:
  1. Naive row-scan (no operators, no batching)
  2. Batched/vectorized (RecordBatch pipeline, hash-based group-by)
  3. SIMD-accelerated filter + sum reduction on top of (2)
  4. Secondary: hash-group vs. sort-group aggregation, profiled in two phases
     (build group index vs. scatter-accumulate) to isolate where SIMD does/doesn't help.
- Output: timing table + chart (naive vs vectorized vs SIMD), exported from `bench/`.
