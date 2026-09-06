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
**Status:** ✅ done (2026-08-18) — 24 tests pass on stable, 25 on nightly + `--features simd`,
clippy clean.
- `Column` enum (`Int64`, `Float64`, `Utf8Dict`) + `DataType`; `Table`
  (`HashMap<String, Column>` + row count) validating equal column lengths on construction,
  so `Scan` can slice batches in Phase 4 without re-checking.
- Crate-wide typed `Error` enum (`src/error.rs`) — a typed error, not `Box<dyn Error>`,
  because Phase 2 must *reject* out-of-scope SQL and that is far easier to test against.
- CSV loader supporting **both** paths phases.md allows: `infer_schema` (whole-file scan)
  and `read_csv` with an explicit `Schema`. Two passes rather than single-pass type
  promotion — promoting a numeric column to `Utf8` mid-file would require re-formatting
  already-parsed values, and `3.10` does not round-trip back from an `f64`.
- Dictionary encoding in **first-seen order, not sorted** — the property systemDesign.md
  leans on to justify the `<`/`>` string-filter fallback. Pinned by test.

Decisions made here that the design docs did not cover:
- **No NULL/missing-value support.** Empty fields get no special meaning: `` fails
  numeric parsing, so a column with blanks infers as `Utf8` and the blank is an ordinary
  dictionary entry. Nothing coerces a missing value to `0`. Real NULL semantics (validity
  bitmap + null-skipping in every operator) stays a deliberate scope decision.
- **Non-finite text is not numeric.** `inf` and `NaN` both parse as `f64` in Rust;
  accepting them would smuggle `NaN` into the engine, where it breaks the equality that
  grouping depends on. Rejected at the boundary instead.
- **Whitespace is trimmed for numeric parsing only**; `Utf8` values are stored verbatim.
- **Duplicate CSV column names are a load-time error** — `Table` keys by name, so they
  would otherwise silently shadow each other and produce a wrong answer.
- **Wholly blank lines are skipped by the `csv` crate**, so `nrows` can be lower than the
  file line count. Pinned by test; matters when reconciling against pandas/DuckDB in Phase 7.

## Phase 2 — SQL Parsing
**Status:** ✅ done (2026-08-18) — 41 unit tests + 2 doctests pass on stable, 42 on nightly +
`--features simd`, clippy and rustfmt clean.
- `src/plan/` — the internal `LogicalPlan` (`table`, `group_by`, `aggregation`, `filter`) plus
  `Aggregation`/`AggFunc`, `Predicate`/`CompareOp`, `Literal`. Deliberately a flat record, not
  a relational tree: with exactly one supported query shape, a general plan tree would be a
  general planner nobody asked for. Every field is non-optional because every clause in the
  shape is mandatory.
- `src/parser/` — `parse(sql) -> LogicalPlan` over `sqlparser-rs` + `GenericDialect`.
- Two error variants, because they mean opposite things to a user: `Error::Sql` ("that is not
  valid SQL") vs `Error::Unsupported` ("valid SQL this engine deliberately does not do").
  Every `Unsupported` message names the offending construct, and `Display` appends the one
  supported shape.
- `Query` is destructured **by name, not with `..`**, so a future sqlparser upgrade that adds
  a clause fails to compile here rather than silently ignoring it. A silently-ignored clause
  is precisely the wrong-answer bug this module exists to prevent.

Rejection surface (the actual deliverable — ~1 line builds a plan, the rest refuse):
joins (all three spellings incl. implicit cross join), multi-column GROUP BY, GROUP BY
ALL/CUBE/ROLLUP/GROUPING SETS, missing GROUP BY, missing WHERE, non-SUM aggregates, `SUM(DISTINCT)`,
`SUM(*)`, `SUM(expr)`, window functions, HAVING, ORDER BY, LIMIT/OFFSET, DISTINCT, CTEs,
UNION/INTERSECT/EXCEPT, FROM-subqueries, aliases (column and table), `SELECT *`, wrong item
count or order, qualified names (`t.col`, `schema.table`), `>=`/`<=`/`<>`, AND/OR/NOT,
BETWEEN, IN, LIKE, IS NULL, column-to-column comparison, literal-on-the-left, boolean
literals, NULL literals, non-SELECT statements, multiple statements.

Decisions made here that the design docs did not cover:
- **Identifiers are case-sensitive.** Standard SQL folds unquoted identifiers, but column
  names must match CSV header text, and folding would mean picking a winning case and
  applying it consistently through storage. Names are taken as written; quotes are stripped,
  so `"region"` and `region` are the same name but `Region` is not.
- **SELECT item order is fixed** (`col1` then `SUM(col2)`). `SELECT SUM(x), region` is
  sensible SQL but accepting it widens the surface past prd.md, which needs sign-off.
- **Boolean and NULL literals are rejected at parse time**, tying to two existing scope cuts:
  no `Bool` column type (prd.md) and no NULL representation (Phase 1).
- **Negative literals** arrive as unary minus over a positive token; the sign is re-attached
  as text before parsing so `i64::MIN` survives (its absolute value does not fit in an i64).

Behaviours found while testing (each cost a test failure first):
- `SELECT FROM WHERE` is **not** a syntax error to sqlparser — it reads as a query with an
  empty projection. Genuinely malformed input is needed to exercise the `Error::Sql` path.
- `CUBE`/`ROLLUP`/`GROUPING SETS` arrive as *expressions in the GROUP BY list*, not as
  `GroupByWithModifier`s, so the modifier check never sees them.
- `c > 1 AND d < 2` parses as `AND` at the top with a comparison on each side, so the
  operator must be inspected before the left operand or the error names the wrong thing.
- `SELECT *` is one projection item, so the wildcard check must precede the item-count check.

## Phase 3 — Naive Baseline
**Status:** ✅ done (2026-08-18) — 56 unit tests + 2 doctests on stable, 57 on nightly +
`--features simd`, clippy and rustfmt clean.
- `src/bench/naive.rs` — `naive_query(&Table, &LogicalPlan) -> Table`, a plain row loop with
  no operator abstraction. Result is a small `Table` (one row per group) with the group
  column under its own name and sums under `SUM(<col>)`, matching architecture.md.
- Serves two jobs at once: the benchmark's row-at-a-time control, and the correctness oracle
  every later strategy must match (agents.md Testing Expectations).
- `Error` gains `UnknownColumn { column, available }` (available names sorted, so the message
  does not depend on the column map's hash order) and `TypeError(String)`. The parser
  validates *shape* and has never seen data, so name and type checks land here, where the
  plan first meets a `Table`.

Fairness decisions, so the Phase 7 write-up can defend the numbers:
- **Not a strawman.** Column types are resolved once into typed slices before the loop; the
  baseline does not re-dispatch on the `Column` enum per row and carries no artificial work.
  What remains is one well-predicted branch per row, which is what row-at-a-time genuinely
  costs. Handicapping the control would be forcing a cleaner story, which agents.md forbids.
- **Same storage, same hasher.** It reads the same columnar `Table` and uses the same
  `hashbrown` map as the batch engine will, so the execution model is the only variable —
  architecture.md's "isolates row-at-a-time as the variable, not operator overhead".

⚠️ **Open question for Phase 6/7 — needs a decision.** Holding storage constant means the
headline benchmark measures the *execution model* half of the columnar thesis, not the
*storage layout* half. prd.md's motivation cites both ("column-wise storage **and**
vectorized execution"). A second baseline over row-oriented storage (array-of-structs) would
demonstrate the storage half, since a column scan there touches every cache line — but it is
outside what phases.md currently scopes. Decide before writing the benchmark narrative.

Decisions made here that the design docs did not cover:
- **Integer sums wrap on overflow** rather than panicking or saturating. SIMD lane arithmetic
  wraps and cannot panic, so wrapping is what lets the naive, batched, and SIMD paths agree
  bit-for-bit — required by the agents.md rule that SIMD and scalar pass the same tests.
- **Float64 columns cannot be group keys.** `GroupKey` holds a dict code or a raw i64
  (systemDesign.md), so there is no float representation; grouping on floats is also
  ill-defined (`0.0` and `-0.0` are equal but differently-encoded). Rejected with a type error.
- **Mixed numeric comparisons are allowed**: `int_col > 2.5` widens the column value to f64
  (lossy past 2^53, noted in code), `float_col < 3` widens the literal. Both are common and
  the alternative is exact int/float comparison logic this query shape does not need.
- **Group order is unspecified but deterministic** (first-seen). The batch engine's hash map
  will produce a different order for the same data, so Phase 4 correctness tests must compare
  results order-insensitively.

Interaction found by a failing test: a header-only CSV infers as all-`Utf8` (Phase 1: no
values, no evidence for a narrower type), and `Utf8` is not summable — so an inferred empty
table cannot be queried. Correct behaviour, not a bug; the explicit-schema loader path is the
fix, and both halves are now pinned by tests.

⚠️ **Noted for Phase 5.** `f64` addition is not associative, so a SIMD sum reduction over 8
lanes accumulates in a different order than the scalar loop and can differ in the last ULP.
Float sum comparisons between implementations will need a tolerance; integer sums stay exact.

## Phase 4 — Batch Execution Pipeline
**Status:** ✅ done (2026-09-04) — 112 unit + 10 parity + 2 doctests on stable, 113 on nightly
+ `--features simd`, clippy and rustfmt clean.
- `src/exec/`: `RecordBatch`, the `Operator` trait, `Bitset`, `materialize`, and
  `Scan`/`Filter`/`Project`/`Aggregate`. `src/plan/` split into `logical` (the data) and
  `physical` (the operator-tree builder), matching architecture.md.
- **Errors are impossible during execution.** `next_batch` returns `Option`, with no error
  channel (pinned by architecture.md) — so every check needing the table's schema happens in
  `plan::build`. The hot loop carries no error handling; `None` means exhausted, never failed.
- **Bitset** is packed `Vec<u64>`, built a *word at a time*: 64 predicate results OR'd into
  one `u64` and stored once. That is the exact shape a SIMD compare produces, so Phase 5
  replaces the inner loop and nothing around it.
- **Filter compacts** (systemDesign.md's choice over selection-vector passthrough) and skips
  wholly-empty batches rather than forwarding them.
- **Aggregate** is blocking: it drains its input on the first `next_batch`, then returns the
  whole result as one batch. `build_group_index` and `scatter_accumulate` are separate public
  methods so Phase 6 can time them independently — the systemDesign.md requirement that makes
  the "SIMD helps phase 1 a little and phase 2 not at all" finding measurable.
- **Monomorphized, no dynamic dispatch anywhere.** Operators are generic over their input, and
  `Aggregate` over `A: Accumulator<T>`. `BatchPipeline` is the single enum switch, resolved
  once per query, that picks the i64 or f64 instantiation.
- `Aggregate::new` takes `group_columns: Vec<String>` and asserts `len() == 1` — the cheap
  extension seam systemDesign.md describes.
- Both engines now label results through one shared definition (`Aggregation::output_name`),
  since a query's output schema is a property of the query, not of the engine running it.

Acceptance criterion met: `tests/parity.rs` runs both engines over deterministic synthetic data
and asserts **exact** agreement — including `f64` sums compared by raw bit pattern — across
every supported query shape, batch boundaries (0/1/1023/1024/1025/2047/2048/2049 rows),
selective and non-matching and match-everything filters, high cardinality, one-row-per-group,
one-group-total, and identical rejection of every invalid plan.

⚠️ **Pinned type amended, with sign-off (2026-09-04):**
`Column::Utf8Dict` now holds `Arc<[String]>` instead of `Vec<String>`. `RecordBatch` owns its
columns, so an owned dictionary was cloned into *every* batch, twice (`Scan`, then `Filter`
compaction). Measured over 2M rows, batched/naive by group cardinality:

| cardinality | Int64 group | Utf8 before | Utf8 after |
|---|---|---|---|
| 8 | 0.79x | 0.80x | 0.70x |
| 100 | 0.95x | 2.63x | 0.69x |
| 1,000 | 1.10x | 12.6x | 0.69x |
| 10,000 | 1.12x | **84x** | 0.70x |

Identical data, cardinality, slot count and scatter pattern in each row — the only variable was
whether the group column carried a dictionary. Left unfixed this would have made the Phase 6
headline chart a measurement of allocator throughput. architecture.md, systemDesign.md and
mermaid.md are updated; `examples/smoke.rs` reproduces it and guards the regression.

Decisions made here that the design docs did not cover:
- **Scan reads only the columns the query mentions.** Not an optimization so much as
  measurement hygiene: copying unmentioned columns would tax the pipeline for work the naive
  baseline never does.
- **Type validation is duplicated** between `plan::physical` and the naive baseline rather than
  shared, keeping agents.md no-shared-code rule a bright line. The parity test asserts both
  engines reject invalid plans with byte-identical messages, so drift cannot go unnoticed.
- **Group order genuinely differs** between the engines (first-seen over all rows vs. over
  surviving rows), so parity comparisons sort. Both are valid: SQL guarantees no GROUP BY order.

Note for Phase 6: batched currently runs at **~0.70x** naive (i.e. ~30% faster) before any SIMD.
The gap is smaller than a naive reading of "vectorization wins" would predict, because the
pipeline pays two copies (`Scan` slice, `Filter` compaction) that the row loop does not. That
is an honest and interesting result, not a problem to tune away — report it.

## Phase 5 — SIMD
**Status:** ✅ done (2026-09-06) — 115 tests on stable, 126 on nightly + `--features simd`
(all 10 parity tests included), clippy clean on both configurations, rustfmt clean.
- `src/exec/kernels.rs` — scalar reference kernels, **always compiled and always tested**, plus
  the four dispatchers that are the single place the `simd` feature changes behaviour.
- `src/simd/mod.rs` — the vector implementations, replacing Phase 0's toolchain smoke test.
  8 lanes (512-bit vectors, deliberately wider than AVX2: LLVM legalizes them into two native
  registers, which gives the pipeline two independent chains). Eight lanes also divide 64
  exactly, so a chunk's bits never straddle a bitset word.
- Wiring cost was **two call sites**. Building the Phase 4 mask a word at a time already
  matched the shape a vector compare produces, so `Filter` did not need restructuring.
- Scope held exactly to systemDesign.md: `Int64`/`Float64` compare and sum only. Int-column
  vs float-literal stays scalar (mixed-type, not the dense same-type compare the docs scope),
  `Utf8` stays scalar by guardrail, scatter-accumulate is untouched.

Measured, 16M rows, best of 5, `examples/kernels.rs`:

| kernel | scalar | SIMD | speedup |
|---|---|---|---|
| filter compare, Int64 | 18.87 ms | 12.89 ms | 1.46x |
| filter compare, Float64 | 13.68 ms | 8.61 ms | 1.59x |
| sum reduction, Int64 | 6.80 ms | 6.38 ms | **1.07x** |
| sum reduction, Float64 | 24.85 ms | 6.52 ms | **3.81x** |

**The two sum rows are the real finding, and they have one explanation.** Integer addition is
associative, so LLVM is already free to auto-vectorize the scalar `i64` loop — it did, which is
why writing the vector version by hand buys almost nothing (1.07x). Floating-point addition is
*not* associative, so the compiler must preserve the serial order and cannot vectorize it; the
scalar `f64` loop is latency-bound on one dependency chain. Writing `Simd<f64, 8>` accumulation
is precisely the act of granting permission to reorder, which is why it wins 3.81x.

Note that both SIMD sums land at essentially the same time (6.38 / 6.52 ms for the same 128 MB),
consistent with both having reached a shared memory-bandwidth ceiling — the `i64` loop was
already there, the `f64` loop was not.

So the same property drives both halves of the story: **non-associativity is why the `f64` sum
needs a test tolerance, and it is also why the `f64` sum is the one with a speedup to win.**
That connection is exactly the kind of "explain *where* SIMD helps and why" that prd.md names
as the actual deliverable.

Correction to an earlier note: Phase 3 predicted the parity test's exact-bit float assertion
would start failing here. It did not. The SIMD filter is exact (a comparison has no rounding),
and the SIMD sum is not on the query path at all — the hash group-by scatters rather than
reducing a dense slice. Parity still holds bit-for-bit with `--features simd`, which is a
stronger check than a tolerance. **Phase 6** is where it changes: sort-based grouping makes each
group contiguous, so its aggregation is a dense reduction and will use the lane-wise sum.

Decisions made here that the design docs did not cover:
- **The `f64` sum tolerance is asymmetric by type**: `i64` twins are asserted bit-identical
  (integer addition stays associative even when wrapping), `f64` twins within a relative
  tolerance. A separate test pins that the two float orders *actually* disagree on a crafted
  input, so the tolerance test cannot pass vacuously.
- **One accumulator, not four.** Unrolling to break the loop-carried dependency is the standard
  next step; Phase 6 can measure whether the reduction is still latency-bound before paying the
  complexity. Left documented in the kernel rather than done speculatively.
- **`Utf8` equality is the guardrail's real cost.** It reduces to comparing `u32` dictionary
  codes — a dense numeric compare that would vectorize perfectly. Dictionary encoding had
  already turned the string problem into an integer one, so "no SIMD on strings" forecloses more
  than it appears to. Worth a line in the Phase 7 write-up.

## Phase 6 — Sort-Based Grouping + Full Benchmark Suite
**Status:** 🟡 in progress — scalar arms measured and recorded; SIMD dispatch restructure
outstanding. 147 tests on stable, 158 on nightly + `--features simd`, 12 parity, clippy and
rustfmt clean, CI green.

### What landed
- **Row-oriented arm** (`src/bench/row_store.rs`): array-of-structs storage, generic over the
  three field types so a row is packed with no per-field tags. Supplies the *layout* half of
  the thesis, which the existing arms could not: they all read the same columnar `Table`.
- **Sort-based grouping** (`src/exec/sort_aggregate.rs`), selectable via `GroupStrategy`
  alongside — never replacing — the hash path.
- **Criterion suite** (`benches/arms.rs`, `benches/group_by.rs`) + deterministic data
  generation (`src/bench/data.rs`) + `scripts/summarize_bench.py` for CSV export.
- **In-process alternating harness** (`src/bench/harness.rs`) and the examples that drive it.

### ⚠️ The measurement had to be rebuilt before any number could be trusted

This phase produced a large body of numbers that had to be **thrown away**. Recorded because
the failure is more instructive than the result would have been.

**What went wrong, in order:**
1. A scalar-vs-SIMD comparison silently also changed compiler (stable 1.94 vs nightly 1.96).
   Caught by the `rustc` line in `environment()`.
2. The control-validation gate then failed outright: benchmarks containing **no vector code**
   (`sort/sort-pairs`, `hash/build-index`) moved 5–12% with p ≤ 0.01 when the SIMD feature was
   toggled. A comparison sort cannot get 12% faster from a SIMD feature it never calls.
3. The decisive test: two runs of the **same binary on the same data** differed by a median of
   **7.5%** and a maximum of **66.8%**. Every effect under study was below that floor.

**Three independent causes, separated by experiment:**

| cause | evidence | fix |
|---|---|---|
| One-sided interference | `sort-pairs/8` moved 19.7% on *provably identical work*; the minimum of the same samples moved 0.3% | minimum-of-N, not mean |
| Per-process hasher seeding | `std` and `hashbrown` both seed randomly; three runs gave three iteration orders | fixed-seed FxHash as the default (`src/hash.rs`) |
| Between-process drift | levels shift ~8% run to run regardless of pinning | alternate arms ABAB **inside one process** |

The dataset was cleared as a cause first: byte-identical across three processes
(`examples/determinism.rs`).

**Machine, and a hypothesis ruled out.** AMD Ryzen 7 5800HS — 8 homogeneous Zen 3 cores, **no
P/E hybrid**, so core heterogeneity was not the explanation. L3 is **16 MB** and L2 512 KB per
core, which corrects an earlier assumption of 32 MB. Runs are pinned to one core with the
processor state capped at 99% to disable boost; conditions are declared through
`BATCHBIRD_RUN_CONFIG` and recorded by `environment()` as declarations rather than as facts the
program verified.

**Deliberate deviation from phases.md, with reasoning.** phases.md specified criterion for the
comparisons. Criterion is kept for the reproducible artifact and CSV export, but **its
cross-benchmark deltas are indicative only** and no claim rests on them. It runs A to
completion then B, so drift attaches to one arm; and its estimator assumes symmetric noise when
interference is strictly one-sided. What survived every check in Phases 5 and 6 was the
alternate-in-process, best-of-N pattern from `examples/kernels.rs`, so that is what the
comparisons are read from.

### Harness resolution — measured, not assumed

`examples/aa_null.rs` registers **the same arm twice** and alternates them. True difference is
zero, so the spread is the harness's resolution. Six axis points:

**median 1.57%, worst 5.45%** — against a smallest claimed effect of ~40%, that is 25× headroom.

The distribution shape is the finding: per-sample spread runs 24–160%, while the *minimum*
reproduces to 0.27–5.45%. The tail is entirely one-sided, exactly as the estimator choice
predicted.

### Results — three scalar arms, 1M rows unless stated

**Row count. There is no knee, and both cache hypotheses are refuted.**

| rows | working set | row-oriented | naive | batched | layout | execution |
|---|---|---|---|---|---|---|
| 250k | 3 MB | 22.19 ns/row | 12.89 | 9.22 | 1.72x | 1.40x |
| 1M | 12 MB | 22.27 | 12.99 | 9.30 | 1.71x | 1.40x |
| 2M | 24 MB | 22.92 | 13.41 | 9.44 | 1.71x | 1.42x |
| 4M | 48 MB | 23.44 | 13.51 | 9.68 | 1.74x | 1.40x |
| 8M | 96 MB | 23.60 | 13.34 | 9.74 | 1.77x | 1.37x |

Per-row cost rises 3.5% (naive) and 5.6% (batched) across a **32x range** of rows and working
set — at or barely above resolution. The set crosses the 16 MB L3 between 1M and 2M and the L2
TLB's ~8 MB of 4K coverage between 250k and 1M, and **nothing happens at either point**. The
L3 and TLB hypotheses are both dead, and the THP discriminator run is no longer needed because
there is nothing to discriminate.

This corroborates the bandwidth arithmetic independently: the pipeline moves 12 B/row and runs
at **0.75–1.4 GB/s against a ~20 GB/s ceiling** demonstrated by the Phase 5 sum kernels. At 4–9%
of achievable bandwidth, working-set size simply is not a variable. An earlier note calling
this "bandwidth-bound" was wrong by an order of magnitude and is struck.

**The gaps, stable across 32x scale:** layout **~1.7x**, execution model **~1.4x**. Both far
above resolution, and both *higher and tighter* than the retracted cross-run numbers (1.21–1.44x
and 1.19–1.52x), which understated the layout effect and scattered the execution effect.

**Selectivity (1M rows, cardinality 128).**

| selectivity | layout | execution |
|---|---|---|
| 1% | 0.96x | **0.91x** |
| 10% | 2.44x | **0.87x** |
| 50% | 1.75x | 1.40x |
| 90% | 2.21x | 1.22x |
| 100% | 2.38x | 1.21x |

Batched is genuinely **slower** than the naive row loop below ~10% selectivity, well outside
resolution. The mechanism is confirmed in code, not inferred: `slice_column` is an
unconditional `.to_vec()`, and `Scan` calls it for every column of every batch *before* the
filter runs. At 1% the pipeline pays a full copy to keep one row in a hundred; at 90–100% it
pays that copy plus compaction's second one.

Stated precisely: this is not "batched execution loses at extreme selectivity" but **"this
pipeline's copy-on-scan design loses at extreme selectivity."** `Scan` holds `&'a Table`, which
outlives the query, so a borrowing `RecordBatch` would delete the first copy outright. Named,
measured, and deliberately not implemented — it needs a pinned-type change.

**Cardinality (1M rows, 50% selectivity).**

| cardinality | layout | execution |
|---|---|---|
| 8 | 1.61x | 1.44x |
| 128 | 1.74x | 1.44x |
| 2,048 | 3.17x | 1.40x |
| 32,768 | 4.35x | 1.58x |
| 250,000 | **8.97x** | 1.19x |

The layout gap grows 5.6x across the sweep while the execution gap stays flat. That is
dictionary encoding measured directly: only the row store hashes and compares actual strings,
and at 250k distinct values it costs 846 ns/row against columnar's 94.

**Hash vs sort group-by (1M rows, 50% selectivity).**

| cardinality | hash | sort | sort/hash | verdict |
|---|---|---|---|---|
| 8 | 9.463 ms | 19.266 | 2.04x | hash wins |
| 2,048 | 10.306 | 25.209 | 2.45x | hash wins |
| 32,768 | 22.189 | 37.354 | 1.68x | hash wins |
| 250,000 | 75.203 | 67.726 | 0.90x | **not separable** |

`systemDesign.md` predicted sort-group loses at this scale. It does, at every cardinality where
the comparison can be made. An earlier reading that sort *won* at 250k (0.81x, from the invalid
cross-run data) **does not survive**: the clean measurement gives 0.90x, which falls inside
hash group-by's own ±10.7% seed-dependent variance.

That band is a finding in its own right, promoted out of methodology. `examples/seed_sweep.rs`
measured a **10.7% spread** across five fixed hasher seeds at 250k cardinality — the same order
as the effects under study — with our chosen seed landing **+0.1% off the mean**, so fixing it
cost essentially no bias. **Hash group-by carries a variance component from collision pattern
that is independent of execution model and data; sort-based grouping has none, because a sort
has no collision pattern.** It shows up within a single run too: at 250k, hash's per-sample
spread is 18.4% against sort's 4.3%. The comparison reports the band and refuses to call a
winner inside it.

**Method validation.** Between two full runs the absolute level drifted ~8% (naive 12.99 →
14.08 ns/row) while the ratios moved 2–3% (layout 1.71 → 1.67x, execution 1.40 → 1.44x). Drift
lands on both arms and cancels in the ratio, which is the whole reason for alternating.

### Caveats the write-up must carry
- **Both gaps are lower bounds.** The two row loops share a per-row enum-dispatch floor of
  ~5 ns/row that is layout-independent, so it inflates numerator and denominator alike and
  pulls every ratio toward 1.0. Measured at 1% selectivity, where the row store moves 4x the
  bytes (32 MB vs 8 MB) in the same wall time — neither arm is memory-bound there. The naive
  baseline is **not** optimized to fix this: it is the control (`agents.md`).
- **The row struct is 32 B** and stores `amount` twice, since this query filters and sums the
  same column. A realistic row store would carry all four columns inline at 40 B, so the arm
  understates rather than inflates the row-oriented penalty.
- **Permanent canaries.** The control gate and `examples/determinism.rs` run on every
  measurement. If either moves, the run is void.

### Outstanding
- SIMD dispatch restructure: runtime selection behind a bench-only feature, resolved **per
  kernel call, never per row**; default build keeps `cfg` dispatch and zero runtime branch.
- Re-measure the SIMD arm through the alternating harness once that lands.
- Restore the power scheme (Balanced, `381b4222-f694-41f0-9685-ff5bb260df2e`, unrestricted
  processor state) when Phase 6 measurement is finished.

## Phase 7 — Write-Up
**Status:** not started
- Benchmark chart(s).
- Explicit findings: where SIMD helps and where it doesn't, and why; hash-group vs.
  sort-group tradeoff at benchmark scale; documented scope cuts (no joins, no multi-column
  group-by, no SIMD on strings) stated as decisions, not omissions.

## Explicit Non-Goals (do not implement without revisiting prd.md)
- Joins, multi-column GROUP BY, full SQL surface, SIMD string matching.
