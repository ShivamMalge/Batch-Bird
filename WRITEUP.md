# Where SIMD helps, where it doesn't, and why

Batchbird is a mini columnar query engine built to answer one question with measurements
instead of assertions: *why* are analytical databases fast, and how much of it is vectorized
execution?

**The answer is that vectorization did essentially nothing, and the arithmetic predicts it.**

| | |
|---|---|
| SIMD kernels, measured in isolation | **1.46× / 1.59× / 1.07× / 3.81×** |
| The same kernels, measured running whole queries | **0.94–1.05×** |
| Amdahl ceiling predicted from the kernels' measured share | **1.038×** |
| Observed at that point | **1.039×** |

The filter kernel occupies **11.7%** of query time at 1M rows, 50% selectivity and 128 groups.
Speeding up 11.7% of anything by 1.46× caps the whole at 1.038×, and the query delivered 1.039×
— agreement to **one tenth of a percentage point**, well inside the harness's 2.25% resolution.
That share is not a constant: at 250,000 groups it falls to 1.6%, because hash group-by comes to
dominate everything else. §2 gives both.

So the null result is not "SIMD mysteriously failed." It is Amdahl's law, quantified, with a
model that says exactly what would have been required instead: to get a 1.3× query from a 1.46×
kernel, that kernel would need to be **~90%** of query time rather than 11.7%.

Everything below carries the conditions in [Reproducing](#reproducing). No number here is
reported without them.

---

## 1. Where SIMD helps: the kernels

16M rows, `examples/kernels.rs`, alternating scalar and vector in one process, minimum of 5.

| kernel | scalar | SIMD | speedup |
|---|---|---|---|
| filter compare, `Int64` | 18.87 ms | 12.89 ms | 1.46× |
| filter compare, `Float64` | 13.68 ms | 8.61 ms | 1.59× |
| sum reduction, `Int64` | 6.80 ms | 6.38 ms | **1.07×** |
| sum reduction, `Float64` | 24.85 ms | 6.52 ms | **3.81×** |

The two sum rows look contradictory and share one cause. **Integer addition is associative**, so
LLVM was already free to auto-vectorize the scalar `i64` loop — and did, leaving nothing for a
hand-written version to win. **Floating-point addition is not**, so the compiler must preserve
serial order and cannot vectorize it at all; writing `Simd<f64, 8>` accumulation is precisely
the act of granting permission to reorder.

Both SIMD sums then land at the same wall time for the same 128 MB — 6.38 and 6.52 ms, about
20 GB/s — which is the memory-bandwidth ceiling the `i64` loop had already reached on its own.

That ceiling is **a property of a dense reduction over a contiguous 128 MB array, and the query
pipeline does not inherit it.** A kernel sweeping one column in a tight loop can saturate memory;
a query that also hashes, probes, allocates and scatters cannot get near it, and does not — it
runs at 4–9% of that figure. An early claim that the pipeline was bandwidth-bound above 4M rows
rested on exactly this conflation and is struck in §5.

The same property runs through the whole project: **non-associativity is why the `f64` sum needs
a test tolerance, and why it is the one with a speedup to win.**

## 2. Where it doesn't: whole queries

1M rows, 50% selectivity, scalar and SIMD alternated in one process.

| cardinality | strategy | scalar | SIMD | speedup |
|---|---|---|---|---|
| 8 | hash | 11.111 ms | 10.870 | 1.022× |
| 8 | sort/radix | 23.907 | 24.036 | 0.995× |
| 2,048 | hash | 12.246 | 11.704 | 1.046× |
| 2,048 | sort/radix | 27.384 | 26.809 | 1.021× |
| 250,000 | hash | 102.991 | 100.220 | 1.028× |
| 250,000 | sort/radix | 59.547 | 63.386 | 0.939× |

Every figure sits inside the resolution. The decomposition says why:

| cardinality | stage | ms | share of query |
|---|---|---|---|
| 128 | query (hash) | 10.087 | 100% |
| | filter kernel | 1.176 | **11.7%** |
| | scan (copy) | 1.264 | 12.5% |
| 250,000 | query (hash) | 80.950 | 100% |
| | filter kernel | 1.266 | **1.6%** |
| | scan (copy) | 1.344 | 1.7% |

| cardinality | p | s | Amdahl ceiling | observed | agreement |
|---|---|---|---|---|---|
| 128 | 0.117 | 1.46× | 1.038× | 1.039× | **+0.1 pp** |
| 250,000 | 0.016 | 1.46× | 1.005× | 0.982× | −2.3 pp |

At 250k groups the filter's share collapses to 1.6% because hash group-by dominates everything;
the observed 0.982× is itself inside resolution of 1.0.

**The sum kernel is worse off still.** It reaches a real query only through sort-based grouping,
where run-length aggregation gives it a dense slice:

| cardinality | run reduction | share of sort query | s | ceiling |
|---|---|---|---|---|
| 128 | 0.575 ms | 2.6% | 1.07× | **1.002×** |
| 250,000 | 6.233 ms | 10.7% | 1.07× | **1.007×** |

Even at 10.7% share, a 1.07× kernel yields a 1.007× query. The `f64` sum's 3.81× would fare
better, but the shape of the trap is structural: **run length is `surviving rows / cardinality`,
so the kernel is fast exactly where there is nothing to sum.** At 250k groups runs average two
rows — all tail, no vector. At low cardinality runs are long, but there hash already beats sort
by 1.9–2.4×, far more than any sum speedup could recover.

That prediction was written down and committed *before* the measurement (`a1c3f79`) and is
confirmed: **no cardinality exists at which sort+SIMD beats hash.**

### The asymmetry the project was built to show

`systemDesign.md` predicted SIMD would help dense numeric work and not scatter-heavy
aggregation. The phase breakdown confirms it directly — 1M rows, ~500k surviving:

| cardinality | hash build-index | hash scatter (incremental) | sort run-reduction |
|---|---|---|---|
| 8 | 4.35 ms | 0.74 ms | 0.52 ms |
| 2,048 | 5.40 | 1.07 | 0.55 |
| 250,000 | 23.43 | **10.50** | 5.54 |

Scatter-accumulate grows **14×** across the sweep as data-dependent writes leave cache. It
cannot be vectorized at all: SIMD can load eight values at once but cannot store them to eight
computed addresses. Sort's run reduction — the same arithmetic over contiguous memory — is an
order of magnitude cheaper at every point.

The prediction was right about the mechanism. What it did not anticipate is that **being right
about the mechanism does not produce a faster query**, because the vectorizable part is too
small a share to matter.

### The guardrail that costs more than it looks

`agents.md` forbids SIMD on string filtering. But dictionary encoding had already reduced `Utf8`
equality to comparing `u32` codes — a dense numeric compare that would vectorize perfectly. The
scope cut therefore forecloses a win that was already sitting there, not merely a hard problem
avoided. Worth stating plainly, because "no SIMD on strings" sounds cheaper than it is.

## 3. What columnar actually buys — encoding, not cache lines

The row-oriented arm is **~1.7×** slower than the columnar one at fixed cardinality. The obvious
explanation is cache-line utilization: a row store reads one field but drags the whole row into
cache. That explanation is **wrong here**, and a controlled probe shows it.

`examples/layout_probe.rs` pads the row struct while holding the field count fixed, so only the
bytes dragged along with each read change. 2M rows, cardinality 128, `u32` group key on both
sides:

| row width | cache lines / 1k rows | row | columnar | gap |
|---|---|---|---|---|
| 24 B | 375 | 8.64 ns | 9.67 | **0.89×** |
| 32 B | 500 | 8.87 | 9.57 | **0.93×** |
| 48 B | 750 | 9.11 | 9.94 | 0.92× |
| 80 B | 1250 | 10.48 | 9.58 | 1.09× |
| 144 B | 2250 | 14.28 | 10.01 | 1.43× |

Two results, both unexpected:

1. **Cache-line utilization is real but weak and sub-linear.** Six times the width buys 1.65×
   the cost, not 6×. Hardware prefetchers hide most of a wider stride.
2. **Below ~64 B per row the row layout is *faster* than columnar.** One sequential stream beats
   three: the columnar loop walks three independent arrays, each consuming its own prefetch
   stream and TLB entries.

The engine's row struct is **32 B**, where the probe reports **0.93×** — yet the engine's row arm
measures **1.67×**. The probe differs in exactly one thing: its group key is a `u32` where the
engine's is a `Box<str>`. The missing factor is **dictionary encoding** — a pointer chase and a
string hash per row, versus an integer.

The cardinality sweep corroborates it independently — 1M rows, 50% selectivity, `Box<str>` group
key as the engine actually stores it. The gap grows 1.61× → **8.97×** as distinct values rise
from 8 to 250,000, which is what string-hashing cost does and what cache-line utilization does
not:

| cardinality | row-vs-columnar | execution model |
|---|---|---|
| 8 | 1.61× | 1.44× |
| 128 | 1.74× | 1.44× |
| 2,048 | 3.17× | 1.40× |
| 32,768 | 4.35× | 1.58× |
| 250,000 | **8.97×** | 1.19× |

**This does not mean columnar failed to help.** Dictionary encoding *is* a columnar technique —
a row store has nowhere cheap to put a dictionary, which is exactly why the row arm carries
`Box<str>` per row. The precise claim is:

> **Columnar's advantage in this engine comes from encoding, not from cache-line utilization.
> Absent encoding, the layout is a liability below ~64 B per row.**

One scope note that sharpens it: the probe held **column count at three**, because the supported
query projects three columns. Three columns means three sequential streams. The sub-64 B
inversion is a property of that ratio — with more projected columns the columnar side pays for
more streams and the crossover moves. The sweep varied width, not column count, so the crossover
point is specific to a three-column projection and should not be quoted as a general figure.

## 4. Hash vs sort group-by

1M rows, 50% selectivity, all three alternated in one run.

| cardinality | hash | sort/pdqsort | sort/radix |
|---|---|---|---|
| 8 | 10.706 ms | 1.90× | 2.25× |
| 2,048 | 12.348 | 2.30× | 2.44× |
| 32,768 | 23.323 | 1.71× | **1.31×** |
| 250,000 | 91.377 | **0.77×** | **0.68×** |

`systemDesign.md` predicted sort-based grouping loses at this scale. It holds below 32k groups —
and is **refuted at 250k**, where sort wins by 0.68–0.90× across runs. The direction is
consistent in every run; the magnitude is not, because hash's 250k-entry table carries
per-process placement variance that in-process alternation cannot remove. The range is reported
rather than a point.

The mechanism is the phase breakdown above: hash's scatter degrades as the group table leaves
cache while a sort stays sequential. This is why production engines switch to sort-based or
partitioned aggregation at high cardinality.

**An LSD radix sort was implemented specifically to scope the claim.** Otherwise "sort loses"
could have meant "pdqsort on 16-byte pairs loses", which is a statement about a library rather
than about the strategy. It settles the question: the conclusion is **about sort-based
grouping**. Radix does not change any verdict's direction — it is even *worse* than pdqsort at
low cardinality, where pdqsort's three-way partitioning handles heavy duplication well — but its
advantage grows with cardinality as an O(n) sort should, improving 1.71× → 1.31× at 32k and
0.77× → 0.68× at 250k.

### Hash group-by carries a variance that sort does not

Five fixed hasher seeds at 250k cardinality, minimum of 7:

| seed | time |
|---|---|
| `0xba7cb18d00000001` (default) | 30.586 ms |
| `0x0000000000000001` | 29.399 |
| `0x9e3779b97f4a7c15` | 29.254 |
| `0x00000000deadbeef` | 32.391 |
| `0xffffffffffffffff` | 31.072 |

**Spread 10.7%; the chosen seed sits +0.1% off the mean.** That spread is the same order as the
effects under study, and it comes purely from collision pattern — a property of hashing, not of
the data or the execution model. **A sort has no collision pattern and no such component.** It
shows up within a single run too: at 250k, hash's per-sample spread is 18.4% against sort's 4.3%.

Every hash figure in this document is read against that band, and any comparison falling inside
it is reported as *not separable* rather than as a result. That rule earned itself immediately:
an earlier reading that sort beat hash by 0.90× fell inside the band and was correctly withheld.

## 5. What was struck

Four findings were reported and then withdrawn. Collected here rather than scattered, because
the controls that caught them are the reason to trust what remains.

| # | What was claimed | What was wrong | The control that caught it |
|---|---|---|---|
| 1 | SIMD is ~22% *slower* at query level | The comparison changed compiler as well as kernel — stable 1.94 vs nightly 1.96 | The `rustc` line in `environment()`, printed next to every result |
| 2 | Sort-group beats hash at 250k (0.81×) | Came from cross-process criterion runs whose drift exceeded the effect | A/A gate: benchmarks containing **no vector code** moved 5–12% (p ≤ 0.01) when the SIMD feature was toggled |
| 3 | The pipeline is bandwidth-bound above 4M rows | Off by an order of magnitude — it runs at 0.75–1.4 GB/s against a demonstrated ~20 GB/s ceiling | Arithmetic: bytes touched ÷ wall time, checked against the Phase 5 sum kernels' saturation figure |
| 4 | The ~1.7× row-vs-columnar gap is cache-line utilization | Almost all of it is dictionary encoding | The padding probe in §3: same fields, only width varies |

Two smaller corrections, for completeness. The shared per-row dispatch floor was quoted as
~5 ns/row from the retracted data; re-derived cleanly at 0% selectivity it is **~2.5 ns/row**, so
the reported gaps were *less* understated than claimed. And a benchmark checksum was
order-sensitive, which made sort-group appear to disagree with hash-group when it had simply
emitted the same groups in key order — caught by the harness asserting arms agree at measurement
time.

The pattern is worth naming: **every one of these was caught by a control that existed for a
different reason.** None was found by inspecting the numbers and finding them implausible.

That has a consequence, and it runs against how benchmark harnesses normally get built. The
control gate was written to detect a third variable inside the kernels, and what it actually
exposed was that the whole measurement floor was above the effect. `environment()` was written so
results would be reproducible later, and it caught a scalar-vs-SIMD comparison that had silently
also changed compiler. The determinism canary was written to rule out a seeding bug in the data
generator, and instead found that both hash maps were seeded per process. In each case the
control was built before there was a known use for it, and in each case it caught something its
author was not looking for — which is an argument for **building the controls first, on
principle, rather than adding them once a specific doubt appears.** By the time a number looks
wrong enough to investigate, the control that would have explained it is the one you did not
build.

## 6. Why the numbers are trustworthy

The first benchmark suite produced a large body of results that all had to be discarded. Two
runs of the *same binary on the same data* differed by a median of **7.5%** and a maximum of
**66.8%** — larger than every effect under study. Rebuilding the measurement took longer than
building the engine.

Three independent causes, separated by experiment:

| cause | evidence | fix |
|---|---|---|
| One-sided interference | A benchmark on provably identical work moved 19.7% in the mean and **0.3% in the minimum** | Report the **minimum** of N, never the mean |
| Per-process hasher seeding | `std` and `hashbrown` both seed randomly; three runs gave three iteration orders and genuinely different collision work | Fixed-seed FxHash as the crate default |
| Between-process drift | Levels shift ~8% run to run whatever the pinning | Alternate arms **ABAB inside one process** |

The dataset was cleared as a cause first — byte-identical across three processes.

Interference is **one-sided**: a sample can be slowed by an interrupt, never sped up. That is
why the minimum is the right estimator and the mean is the misleading one, and the A/A null
shows it starkly — per-sample spread runs 24–160% while the *minimum* reproduces to 0.27–5.79%.

**The harness's resolution is measured, not assumed.** `examples/aa_null.rs` registers the same
arm twice and alternates them; the true difference is zero, so whatever appears is the floor.
Across six axis points: **median 2.25%, worst 5.79%.** Nothing smaller is reported as an effect.

Criterion is kept for the reproducible artifact and CSV export, but **its cross-benchmark deltas
are indicative only** and no claim here rests on them: it runs A to completion then B, so drift
attaches to one arm, and its estimator assumes symmetric noise. Recorded in `phases.md` as a
deliberate deviation from the plan, with reasoning.

Two canaries run before every measurement, and if either moves the run is void: the A/A null
above, and `examples/determinism.rs`, which asserts the generated dataset and the engine's map
iteration order are byte-identical across processes.

## 7. Scope cuts, as decisions

Each of these is a choice with a stated cost, not a gap.

- **No joins, no multi-column `GROUP BY`, no SQL surface beyond one query shape.** The parser is
  mostly rejection: ~30 constructs are matched and refused by name, because an engine that
  silently ignored a clause it did not implement would return confidently wrong answers.
- **No NULL support.** Empty CSV fields get no special meaning rather than inventing semantics;
  real null handling is a validity bitmap plus null-skipping in every operator.
- **No SIMD on string filtering.** Costs more than it appears — see §2.
- **Selection-vector passthrough**, the alternative to `Filter` compaction. Noted in
  `architecture.md` as the road not taken; DuckDB takes it.
- **Copy-on-scan.** `slice_column` is an unconditional `.to_vec()`, and `Scan` calls it for every
  column of every batch *before* the filter runs. Since `Scan` holds `&'a Table`, which outlives
  the query, a borrowing `RecordBatch` would remove that copy entirely.

  **Deliberately not implemented**, and this is the one cut with a measured price. At 1M rows
  and 128 groups, batched execution runs at **0.87–0.91×** the naive row loop below 10%
  selectivity — batching is a net *loss* there. Scan is 12.5% of query time at 50% selectivity, but its cost is independent of
  selectivity while everything downstream shrinks, so at 1% selectivity it is roughly 30% of the
  query and removing the write half would be worth about 1.18× — enough to turn that loss into
  roughly break-even.

  It is cut because it changes a type pinned in `architecture.md`: a borrowing `RecordBatch`
  needs a lifetime that propagates onto the `Operator` trait and every operator, which is far
  larger than the one-field `Arc` change made earlier, and Phase 7 is a write-up phase with the
  engine feature-complete and green. **The honest statement is that this is the known
  optimization most worth doing next, not that it does not matter.**

## 8. What this cost, and what it bought

A bug the benchmark caught that reading the code would not have: `Column::Utf8Dict` originally
owned its dictionary, so every `RecordBatch` cloned it — twice, at `Scan` and at `Filter`
compaction. Over 2M rows the batch pipeline degraded from 0.80× to **84× slower** than the row
loop as cardinality went 8 → 10,000, while an `Int64` group column over identical data stayed
flat at ~1.1×. Sharing the dictionary behind an `Arc` fixed it. **The failure was invisible at
low cardinality** — exactly where a casual benchmark would have looked.

If there is one transferable lesson, it is that the measurement needed as much design as the
thing measured, and that most of the effort went into being able to believe the numbers rather
than into producing them.

## Reproducing

Every figure above was produced under these conditions.

| | |
|---|---|
| machine | AMD Ryzen 7 5800HS — 8 homogeneous Zen 3 cores, no P/E hybrid; L3 16 MB, L2 512 KB/core |
| toolchain | rustc 1.94.0 stable / 1.96.0-nightly, edition 2024 |
| profile | release, `codegen-units = 1`, `lto = "thin"`, debug symbols on |
| power | Balanced scheme with processor min/max pinned to 99% (boost disabled) |
| affinity | pinned to core 0 |
| estimator | **minimum of N**, arms alternated ABAB within one process |
| resolution | A/A null: median 2.25%, worst 5.79% |
| data seed | `0xba7cb12d` |
| hasher seed | `0xba7cb18d00000001` (fixed; ±10.7% collision-pattern band on hash figures) |

```bash
cargo run --release --example determinism    # canary: must not move
cargo run --release --example aa_null        # resolution
cargo run --release --example measure        # the arms
cargo run --release --example layout_probe   # §3
cargo run --release --example seed_sweep     # the hash variance band

cargo +nightly run --release --example kernels --features simd          # §1
cargo +nightly run --release --example measure --features bench-dispatch # §2 observed
cargo +nightly run --release --example amdahl  --features bench-dispatch # §2 predicted
```

Raw output is committed under `results/`. `phases.md` carries the full decision log, including
everything struck and why.
