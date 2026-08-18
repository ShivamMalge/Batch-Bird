# Architecture

## Module Breakdown
```
src/
  storage/      Column, Table, CSV loader
  parser/       sqlparser-rs AST -> internal LogicalPlan
  plan/         LogicalPlan -> operator tree
  exec/         Scan, Filter, Project, Aggregate operators; RecordBatch
  simd/         SIMD-accelerated filter compare + sum reduction (feature-gated)
  bench/        naive row-scan baseline + synthetic data generator. Library code, not
                a benchmark harness, so Phase 4+ correctness tests can compare against it.
benches/        criterion harness targets + chart data export. Cargo requires benchmark
                targets to live here rather than under src/ — this split is a layout
                constraint, not a design change.
```

## Data Flow
```
CSV file
  -> storage::Table (columnar, in memory)
  -> parser: SQL string -> sqlparser AST
  -> plan: AST -> operator tree (Scan -> Filter -> Project -> Aggregate)
  -> exec: tree pulls RecordBatches (~1024 rows) via next_batch()
  -> Filter compacts batch by bitset -> Project selects columns -> Aggregate groups + sums
  -> final result: small Table (one row per group)
```

## Core Types (see systemDesign.md for rationale)
```rust
enum Column {
    Int64(Vec<i64>),
    Float64(Vec<f64>),
    Utf8Dict { dict: Vec<String>, codes: Vec<u32> },
}

struct RecordBatch {
    columns: HashMap<String, Column>,
    len: usize,
}

// Single-column only: dict code (u32 widened) or raw i64 (bit-cast via `as u64`).
// Audit fix 2026-08-18: was u32, which could not hold a raw i64 key — see systemDesign.md.
struct GroupKey(u64);

// ✅ CONFIRMED 2026-08-18 — generic over T (avoids i64→f64 precision loss on Int64 sums).
// See systemDesign.md "Accumulator" for rationale.
trait Accumulator<T> {
    fn update(&mut self, val: T);
    fn finalize(&self) -> T;
}
struct SumAccumulator<T> { total: T }
```

## Operator Chain
Each operator implements:
```rust
trait Operator {
    fn next_batch(&mut self) -> Option<RecordBatch>;
}
```
- **Scan**: reads from `storage::Table` in fixed-size slices.
- **Filter**: computes a bitset over the incoming batch, *compacts* (materializes only
  selected rows) rather than passing a selection vector through. Downstream operators
  never see unfiltered rows.
- **Project**: column selection only (no expressions beyond the scoped query shape).
- **Aggregate**: two-phase per batch — build group index (hash map: group key -> slot),
  then scatter-accumulate into per-slot `Accumulator`s.

## Naive Baseline (for benchmarking only)
Lives in `bench/`, deliberately has no operator abstraction — a plain `for` loop over
rows — so the benchmark isolates "row-at-a-time" as the variable, not operator overhead.
