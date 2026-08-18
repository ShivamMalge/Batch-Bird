# System Design

This doc captures the *why* behind the execution model — the decisions, not just the shapes.

## Batching
- Batch size 1024 rows — in the range real vectorized engines use (DuckDB vectors are 2048
  rows; DataFusion defaults to 8192-row batches), chosen at the small end so a batch's working
  set sits comfortably in L1/L2 cache during vectorized ops. Not a magic constant: a small
  batch-size sweep (512 / 1024 / 4096) makes a nice optional secondary benchmark axis.

## String Columns — Dictionary Encoding
`Utf8Dict { dict: Vec<String>, codes: Vec<u32> }`. Group-by hashes `u32` codes, not raw
strings — cheap integer hashing/equality instead of string hashing per row.
- **Equality filters** (`col = 'x'`) resolve the target to a dict code once, then compare
  codes — benefits from dictionary encoding.
- **Ordering filters** (`col > 'x'`, `col < 'x'`) fall back to per-row string comparison via
  the dictionary, because code order does not match lexical order unless the dictionary is
  sorted (not done here). This is a documented scope limitation, not an oversight.

## Filter — Compaction, not Selection-Vector Passthrough
`Filter` computes a bitset over the batch, then **materializes** a new, smaller `RecordBatch`
containing only selected rows. Downstream operators (`Project`, `Aggregate`) always see plain
batches and never need to know filtering happened.
- Tradeoff: compaction costs a copy. The alternative (pass a selection vector through and have
  every downstream operator apply it) avoids the copy but couples every operator's signature to
  always-filtered execution. Compaction is simpler and correct at this project's scale; the
  passthrough approach (used by some real engines, e.g. DuckDB) is noted as the road not taken.

## Bitset Representation
Packed `Vec<u64>` (64 rows/word), not `Vec<bool>`. Rust's `Vec<bool>` is 1 byte/element by
default — no memory-bandwidth advantage over a plain byte mask. Since the benchmark's whole
point is demonstrating a bandwidth effect, only a real bit-packed bitset shows it.

## Group-By — Hash-Based, Two-Phase
`GroupKey(u64)` — single group column only. Holds either a dict code (`u32` widened to `u64`)
or a raw `Int64` key bit-cast via `as u64`. (Audit fix 2026-08-18: the earlier `GroupKey(u32)`
draft could not actually hold a raw i64 key.) The i64→u64 bit-cast is bijective, so hashing
and equality are exact; it does not preserve numeric order (negatives map above positives),
which is fine because grouping only needs *equality* — even sort-based grouping (below) only
needs equal keys adjacent, and SQL guarantees no output order for GROUP BY.

Deliberately not generic over tuples: multi-column keys would require heterogeneous
hashing/equality (trait objects or macro-generated tuple impls) for a feature explicitly out
of scope. Extension seam left cheaply: `Aggregate` takes `group_cols: Vec<ColumnRef>` but
panics if `len() != 1`.

Per batch, two phases:
1. **Build group index** — hash map from `GroupKey` -> accumulator slot. Sequential read over
   the group column's codes; partially vectorizes on the read side even though the hash insert
   itself doesn't.
2. **Scatter-accumulate** — `accumulators[slot].update(col2[row])`. Slot index is data-dependent,
   so this phase is inherently non-vectorizable (SIMD can't help data-dependent scatter writes).

Profile these two phases *separately* in benchmarks — conflating them muddies the "SIMD doesn't
help group-by" finding, since phase 1 benefits a little and phase 2 doesn't at all.

## Sort-Based Grouping (secondary comparison)
Sort rows by group key, then run-length-aggregate contiguous runs. The sort itself doesn't
vectorize trivially (comparison sort, O(n log n)), but the aggregation *after* sorting becomes
a sequential scan, which does. Expected honest result: sort-group's aggregation phase is faster,
but the sort's O(n log n) cost likely makes it net slower than hash-group's O(n) at benchmark
scale (a few million rows) — report whatever the numbers actually show, don't force a ranking.

## Accumulator

> **✅ CONFIRMED 2026-08-18** — generic-over-`T` is the pinned design (explicit human
> sign-off recorded). This supersedes the earlier `f64`-only draft.

```rust
trait Accumulator<T> {
    fn update(&mut self, val: T);
    fn finalize(&self) -> T;
}
struct SumAccumulator<T> { total: T }
```
Generic over `T` (not hardcoded to `f64`) so summing `Int64` columns doesn't silently lose
precision by casting through `f64` past 2^53. Only `SumAccumulator` is implemented; the trait
existing is what answers "why no AVG/COUNT" without implying a rearchitecture would be needed.

`Aggregate` is generic over `A: Accumulator<T>` (monomorphized), not `Box<dyn Accumulator>` —
no runtime dispatch cost, and dynamic dispatch would only pay for itself if a single query
needed heterogeneous accumulators chosen at plan time, which is out of scope.

## SIMD Scope
Applied only to:
- Filter comparison on `Int64`/`Float64` columns (dense numeric compare -> bitmask).
- `SUM` reduction over `Int64`/`Float64` slices.

Not applied to:
- Group-by scatter-accumulate (data-dependent writes defeat vectorization — see above).
- Utf8/string filtering (see dictionary-encoding section above).

This asymmetry is the headline benchmark finding, not a footnote: SIMD helps dense numeric
ops and does not help scatter-heavy aggregation, and being able to explain *why* is the point.
