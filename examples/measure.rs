//! Phase 6 measurement: the three scalar arms, alternated in-process.
//!
//! ```text
//! cargo run --release --example measure
//! ```
//!
//! Read `examples/aa_null.rs` first. It establishes what this harness can resolve — median
//! 1.57%, worst 5.45% — which is the number every difference below must be judged against. An
//! effect smaller than that is not an effect.
//!
//! # The arms, and what separates them
//! - `row-oriented` -> `columnar-naive` is **storage layout**. Identical plain row loop,
//!   identical query; only the memory layout differs.
//! - `columnar-naive` -> `batched` is **execution model**. Identical columnar storage; only
//!   batching, bitset filtering and compaction differ.
//!
//! Both gaps are understated, and the write-up must say so. The two row loops share a per-row
//! enum-dispatch floor of roughly 5 ns that is independent of layout, so it inflates both
//! numerator and denominator and pulls every ratio toward 1.0.
//!
//! # Hash vs sort carries a variance the other comparisons do not
//! `examples/seed_sweep.rs` measured a **10.7% spread** in hash group-by purely from the
//! hasher's collision pattern — the same order as the effects under study. Sort-based grouping
//! has no equivalent: a sort has no collision pattern. That asymmetry is reported alongside
//! the hash means below rather than left in a methodology note, because it is a property of
//! the strategy, not of the measurement.

use batchbird::bench::data::{SEED, amount_threshold, environment, generate};
use batchbird::bench::harness::{Arm, alternate, report, result_checksum};

/// Order-insensitive, because the group-by strategies emit groups in different orders.
fn result_checksum_of(table: &Table) -> u64 {
    result_checksum(table, "region", "SUM(amount)")
}
use batchbird::bench::{build_row_store, naive_query};
use batchbird::parser::parse;
use batchbird::plan::{GroupStrategy, LogicalPlan, batch_query, batch_query_with};
use batchbird::storage::Table;

const SAMPLES: usize = 30;
const WARMUP: usize = 3;

/// Resolution from `examples/aa_null.rs`, quoted so every table carries its own error bar.
const RESOLUTION_MEDIAN: f64 = 1.57;
const RESOLUTION_WORST: f64 = 5.45;

/// Seed-dependent variance in hash group-by, from `examples/seed_sweep.rs`.
const HASH_SEED_VARIANCE: f64 = 10.7;

fn query(selectivity: f64) -> LogicalPlan {
    let threshold = amount_threshold(selectivity);
    parse(&format!(
        "SELECT region, SUM(amount) FROM t WHERE amount > {threshold} GROUP BY region"
    ))
    .expect("benchmark query must parse")
}

/// Time the three scalar arms over one dataset, and report the two gaps.
fn scalar_arms(label: &str, table: &Table, plan: &LogicalPlan, rows: usize) {
    // Built outside every timed region: a real row database would have loaded its data in this
    // layout, so the conversion is an artifact of this project's setup.
    let row_store = build_row_store(table, plan).expect("row store builds");

    let arms = vec![
        Arm::new("row-oriented", || {
            result_checksum_of(&row_store.run(plan).expect("row"))
        }),
        Arm::new("columnar-naive", || {
            result_checksum_of(&naive_query(table, plan).expect("naive"))
        }),
        Arm::new("batched", || {
            result_checksum_of(&batch_query(table, plan).expect("batched"))
        }),
    ];

    let m = alternate(arms, SAMPLES, WARMUP);
    println!("\n{label}");
    report(&m);

    // All three must agree, or the comparison is between different questions.
    assert_eq!(
        m[0].checksum, m[1].checksum,
        "row-oriented disagrees with naive"
    );
    assert_eq!(m[1].checksum, m[2].checksum, "batched disagrees with naive");

    let (row, naive, batched) = (m[0].min(), m[1].min(), m[2].min());
    println!(
        "  layout gap (row/naive):     {:.2}x        {:.2} ns/row -> {:.2} ns/row",
        row / naive,
        row * 1e6 / rows as f64,
        naive * 1e6 / rows as f64,
    );
    println!(
        "  execution gap (naive/batched): {:.2}x     {:.2} ns/row -> {:.2} ns/row",
        naive / batched,
        naive * 1e6 / rows as f64,
        batched * 1e6 / rows as f64,
    );
}

fn main() {
    println!("{}\n", environment());
    println!(
        "harness resolution (examples/aa_null.rs): median {RESOLUTION_MEDIAN}%, worst \
         {RESOLUTION_WORST}%\nA difference smaller than that is not a difference. \
         {SAMPLES} samples, {WARMUP} warmup, alternated ABC per sample, minimum reported.\n"
    );

    // ---- row count: where does the curve bend, and why -------------------------------
    // 2M added because 12 MB against a 16 MB L3 shared with the group table is not a clean
    // fit, so the knee is likely between 1M and 4M and nearer the low end. 8M puts the
    // working set far past both L3 and the L2 TLB's ~8 MB of 4K-page coverage.
    println!("=== row count (cardinality 128, selectivity 50%) ===");
    let plan = query(0.5);
    for rows in [250_000usize, 1_000_000, 2_000_000, 4_000_000, 8_000_000] {
        let table = generate(rows, 128, SEED);
        let columnar_mb = rows as f64 * 12.0 / 1e6;
        scalar_arms(
            &format!("{rows} rows  (columnar working set {columnar_mb:.0} MB, L3 is 16 MB)"),
            &table,
            &plan,
            rows,
        );
    }

    // ---- selectivity: the copy-on-scan curve ------------------------------------------
    println!("\n\n=== selectivity (1M rows, cardinality 128) ===");
    println!("Scan copies every batch unconditionally before the filter runs, so the ends of");
    println!("this sweep are where copy-on-scan costs most relative to what it returns.");
    let table = generate(1_000_000, 128, SEED);
    // 0% first: nothing survives, so what remains is the pure scan-plus-dispatch cost per
    // row. That is the floor both row loops share, and the reason both gaps are lower bounds.
    for fraction in [0.00, 0.01, 0.10, 0.50, 0.90, 1.00] {
        let plan = query(fraction);
        scalar_arms(
            &format!("selectivity {:.0}%", fraction * 100.0),
            &table,
            &plan,
            1_000_000,
        );
    }

    // ---- cardinality ------------------------------------------------------------------
    println!("\n\n=== cardinality (1M rows, selectivity 50%) ===");
    let plan = query(0.5);
    for cardinality in [8u64, 128, 2_048, 32_768, 250_000] {
        let table = generate(1_000_000, cardinality, SEED);
        scalar_arms(
            &format!("cardinality {cardinality}"),
            &table,
            &plan,
            1_000_000,
        );
    }

    // ---- hash vs sort -----------------------------------------------------------------
    println!("\n\n=== group-by strategy (1M rows, selectivity 50%) ===");
    println!(
        "hash-group carries a +/-{HASH_SEED_VARIANCE}% seed-dependent variance component\n\
         (examples/seed_sweep.rs); sort-group carries none, since a sort has no collision\n\
         pattern. Read the hash column with that band, not as a point."
    );
    for cardinality in [8u64, 2_048, 32_768, 250_000] {
        let table = generate(1_000_000, cardinality, SEED);
        let arms = vec![
            Arm::new("hash", || {
                result_checksum_of(
                    &batch_query_with(&table, &plan, GroupStrategy::Hash).expect("hash"),
                )
            }),
            Arm::new("sort/pdqsort", || {
                result_checksum_of(
                    &batch_query_with(&table, &plan, GroupStrategy::Sort).expect("sort"),
                )
            }),
            // Separates "sort-based grouping loses" from "pdqsort on 16-byte pairs loses".
            // Only one of those is a claim about the strategy.
            Arm::new("sort/radix", || {
                result_checksum_of(
                    &batch_query_with(&table, &plan, GroupStrategy::SortRadix).expect("radix"),
                )
            }),
        ];

        let m = alternate(arms, SAMPLES, WARMUP);
        println!("\ncardinality {cardinality}");
        report(&m);
        assert_eq!(m[0].checksum, m[1].checksum, "strategies disagree");

        for candidate in &m[1..] {
            let ratio = candidate.min() / m[0].min();
            let verdict = if (ratio - 1.0).abs() * 100.0 < HASH_SEED_VARIANCE {
                "within hash's seed variance -- not separable"
            } else if ratio < 1.0 {
                "beats hash"
            } else {
                "hash wins"
            };
            println!("  {:<14} vs hash: {ratio:.2}x   {verdict}", candidate.name);
        }
    }

    simd_arms();
}

/// Scalar vs SIMD, alternated inside one process.
///
/// The only way to compare them honestly: across `cargo` invocations the two configurations
/// differ by more than the effect, and a control gate caught benchmarks with no vector code
/// moving 5-12% purely from the toolchain and run-to-run drift.
#[cfg(feature = "bench-dispatch")]
fn simd_arms() {
    use batchbird::exec::kernels::{Kernel, dispatch};

    println!(
        "

=== scalar vs SIMD (alternated in-process, 1M rows, 50% selectivity) ==="
    );
    println!("The filter kernel runs on every arm; the sum kernel reaches a query only through");
    println!("sort-group, whose run length is rows/cardinality -- see the recorded prediction.");

    let plan = query(0.5);
    for cardinality in [8u64, 2_048, 250_000] {
        let table = generate(1_000_000, cardinality, SEED);

        for (label, strategy) in [
            ("hash", GroupStrategy::Hash),
            ("sort/radix", GroupStrategy::SortRadix),
        ] {
            let arms = vec![
                Arm::new("scalar", || {
                    dispatch::set(Kernel::Scalar);
                    result_checksum_of(&batch_query_with(&table, &plan, strategy).expect("q"))
                }),
                Arm::new("simd", || {
                    dispatch::set(Kernel::Simd);
                    result_checksum_of(&batch_query_with(&table, &plan, strategy).expect("q"))
                }),
            ];

            let m = alternate(arms, SAMPLES, WARMUP);
            println!(
                "
cardinality {cardinality}, {label}"
            );
            report(&m);
            assert_eq!(
                m[0].checksum, m[1].checksum,
                "kernels disagree on the answer"
            );

            let speedup = m[0].min() / m[1].min();
            let resolved = if (speedup - 1.0).abs() * 100.0 < RESOLUTION_WORST {
                "  <- below resolution, no effect"
            } else {
                ""
            };
            println!("  simd speedup: {speedup:.3}x{resolved}");
        }
    }
}

#[cfg(not(feature = "bench-dispatch"))]
fn simd_arms() {
    println!(
        "

=== scalar vs SIMD ===
Skipped: build with          `cargo +nightly run --release --example measure --features bench-dispatch`."
    );
}
