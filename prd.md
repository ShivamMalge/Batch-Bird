# PRD — Mini Columnar Query Engine

## Problem / Motivation
Analytical databases (DuckDB, ClickHouse) get their speed from column-wise storage and
vectorized execution rather than row-at-a-time processing. This project builds a minimal
engine that demonstrates *why*, with a benchmark to prove the effect rather than just claim it.

## Goals
- Implement column-wise storage (`Int64`, `Float64`, `Utf8Dict`) over CSV input.
- Parse a narrow SQL subset via `sqlparser-rs` into an internal plan.
- Execute via a batch-oriented operator pipeline (Scan → Filter → Project → Aggregate).
- Vectorize filter and sum-reduction hot loops with `std::simd`.
- Produce a credible three-way benchmark: naive row-scan vs. batched/vectorized vs. SIMD,
  plus hash-group vs. sort-group as a secondary comparison.

## Non-Goals (explicit scope cuts)
- No joins (hash join, join ordering — separate project).
- No multi-column `GROUP BY` (single group column only; `GroupKey` is a plain newtype).
- No full SQL surface — only the query shape below.
- No SIMD for string filtering (`>`/`<` on strings falls back to scalar comparison via
  the dictionary; only equality benefits from dictionary-code comparison).
- No `Bool` column type — no source data in scope is boolean-shaped; the only boolean
  artifact is the filter mask (bitset), not a stored column.

## Supported Query Shape
```sql
SELECT col1, SUM(col2) FROM t WHERE col3 > x GROUP BY col1
```
The `WHERE` comparison operator may be `=`, `<`, or `>` against a literal, and `col3` may
be any of the three column types. (String equality compares dictionary codes; string
`<`/`>` falls back to scalar comparison — see Non-Goals.)

## Success Criteria
- Correctness: output matches a trusted reference (e.g. pandas/DuckDB) on the same CSV.
- Benchmark chart showing naive vs. vectorized vs. SIMD timings across a few million rows.
- A written explanation of *where* SIMD helps (filter, sum reduction) and where it doesn't
  (hash-group scatter writes) — this finding is the actual deliverable, not just the code.

## Audience
Personal portfolio/learning project — written up as a resume artifact demonstrating
systems/database engineering judgment, not a production system.
