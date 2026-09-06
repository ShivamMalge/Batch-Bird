# Phase 6 results

Everything here was produced under the conditions recorded at the top of `scalar-arms.txt`:
one core, boost disabled, minimum-of-N, arms alternated inside a single process.

| file | what it is | trust |
|---|---|---|
| `scalar-arms.txt` | The measurement. Three scalar arms alternated ABC per sample across row count, selectivity, cardinality, plus hash vs sort. | **Primary.** Claims rest on this. |
| `aa-null.txt` | A/A null: the same arm registered twice. Establishes the harness's resolution at median 1.57%, worst 5.45%. | **Primary.** Read this before any number in `scalar-arms.txt`. |
| `layout-probe.txt` | Row width swept 24-144 B at fixed field count. Shows the row-vs-columnar gap is *not* mostly cache-line utilization. | **Primary.** |
| `amdahl.txt` | Kernel share of query time, Amdahl ceiling, and observed speedup side by side. Closes the headline argument quantitatively. | **Primary.** |
| `seed-sweep.txt` | Five fixed hasher seeds at 250k cardinality. 10.7% spread; our seed sits +0.1% off the mean. | **Primary.** The band on every hash group-by figure. |
| `criterion-indicative.csv` | Criterion point estimates, exported by `scripts/summarize_bench.py`. | **Indicative only.** See below. |

## Why criterion's numbers are indicative only

Criterion is kept for the reproducible artifact and the CSV export. No claim rests on it, for
two reasons this project measured rather than assumed:

- It runs one benchmark to completion before the next, so drift over a run attaches to one arm
  rather than being shared. Two runs of the *same binary on the same data* differed by a median
  of 7.5% and a maximum of 66.8%.
- Its estimator assumes symmetric noise. Interference is one-sided — a sample can be slowed by
  an interrupt, never sped up — so the mean absorbs every such event. On identical work, a
  19.7% apparent difference in the mean was 0.3% in the minimum.

`phases.md` Phase 6 records this as a deliberate deviation, with the reasoning.

## Reproducing

```bash
cargo run --release --example determinism   # canary: dataset + map order must not move
cargo run --release --example aa_null       # resolution
cargo run --release --example measure       # the measurement
cargo run --release --example seed_sweep    # hasher-seed bias
cargo run --release --example layout_probe  # what causes the row-vs-columnar gap

# SIMD arms need runtime kernel selection:
cargo +nightly run --release --example measure --features bench-dispatch

cargo bench && python scripts/summarize_bench.py results/criterion-indicative.csv
```

Declare the measurement conditions through `BATCHBIRD_RUN_CONFIG`; `environment()` records them
as declarations, never as facts it verified.
