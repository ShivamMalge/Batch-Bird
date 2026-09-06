//! Execution model and storage layout: the four-arm comparison.
//!
//! ```text
//! cargo bench --bench arms                            # arms 1-3
//! cargo +nightly bench --bench arms --features simd   # arm 4 (batched + SIMD)
//! ```
//!
//! The SIMD arm is the same code compiled differently, so it is a second run rather than a
//! fourth benchmark id. Criterion keeps both under the same ids, so the second run reports
//! change-vs-baseline directly — which is exactly the comparison wanted.
//!
//! # Reading the arms
//! - `row-oriented` → `columnar-naive` is the **storage layout** effect. Same plain row loop,
//!   same query; only the memory layout differs.
//! - `columnar-naive` → `batched` is the **execution model** effect. Same columnar storage;
//!   only batching and bitset filtering differ.
//!
//! Splitting them is the entire reason the row-oriented arm exists. A benchmark with only the
//! last two arms can say nothing about columnar storage, which is half of what `prd.md` claims.
//!
//! # Methodology
//! - Data is generated once per parameter set, **outside** the timed region. Timings measure
//!   query execution, never generation or loading.
//! - The row-oriented store is likewise built outside the timed region: a real row database
//!   would have loaded its data in that layout, so converting from columnar is an artifact of
//!   this project's setup and must not be charged to the arm.
//! - Seeded generation, so a rerun is comparable to an earlier one.
//! - Sample size is reduced from criterion's default 100 because the parameter grid is wide
//!   and each sample runs a multi-million-row query; the configuration is recorded alongside
//!   the results by `report_environment`.

use std::time::Duration;

use batchbird::bench::data::{SEED, amount_threshold, environment, generate};
use batchbird::bench::{build_row_store, naive_query};
use batchbird::parser::parse;
use batchbird::plan::{LogicalPlan, batch_query};
use batchbird::storage::Table;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

/// Written next to the timings, because a speedup without a machine and a toolchain is not a
/// result anyone can check.
fn report_environment() {
    let manifest = format!(
        "{}\ncriterion: sample_size=20, warm_up=1s, measurement=3s\n",
        environment()
    );
    println!("\n=== benchmark environment ===\n{manifest}");

    let dir = std::path::Path::new("target/criterion");
    if std::fs::create_dir_all(dir).is_ok() {
        let _ = std::fs::write(dir.join("environment.txt"), &manifest);
    }
}

fn configured() -> Criterion {
    Criterion::default()
        .sample_size(20)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3))
}

/// The query every arm runs. `region` is the dictionary-encoded group key, so this exercises
/// the path where layout differences are largest.
fn query(selectivity: f64) -> LogicalPlan {
    let threshold = amount_threshold(selectivity);
    parse(&format!(
        "SELECT region, SUM(amount) FROM t WHERE amount > {threshold} GROUP BY region"
    ))
    .expect("benchmark query must parse")
}

/// Time all three arms over one dataset.
fn bench_arms(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    table: &Table,
    plan: &LogicalPlan,
    label: impl std::fmt::Display + Clone,
) {
    // Built once, outside every timed region -- see the module docs on fairness.
    let row_store = build_row_store(table, plan).expect("row store builds");

    group.bench_with_input(
        BenchmarkId::new("row-oriented", label.clone()),
        plan,
        |b, plan| b.iter(|| row_store.run(plan).expect("row query")),
    );

    group.bench_with_input(
        BenchmarkId::new("columnar-naive", label.clone()),
        plan,
        |b, plan| b.iter(|| naive_query(table, plan).expect("naive query")),
    );

    group.bench_with_input(BenchmarkId::new("batched", label), plan, |b, plan| {
        b.iter(|| batch_query(table, plan).expect("batched query"))
    });
}

/// Does anything here scale with bandwidth rather than compute?
///
/// Phase 5 measured both SIMD sum kernels converging on the same wall time for the same bytes,
/// which predicts a ceiling. Throughput is reported in elements so criterion prints a
/// per-row rate: a flat rate across row counts means bandwidth-bound, a falling one means
/// something else dominates.
fn row_count(c: &mut Criterion) {
    report_environment();

    let mut group = c.benchmark_group("row-count");
    let plan = query(0.5);

    for rows in [250_000usize, 1_000_000, 4_000_000] {
        let table = generate(rows, 100, SEED);
        group.throughput(Throughput::Elements(rows as u64));
        bench_arms(&mut group, &table, &plan, rows);
    }

    group.finish();
}

/// Compaction cost is a function of how many rows survive, so the ends of this sweep exercise
/// different halves of the pipeline: a 1% filter barely compacts anything and skips whole
/// batches, a 100% filter copies every row twice.
fn selectivity(c: &mut Criterion) {
    let mut group = c.benchmark_group("selectivity");
    let table = generate(1_000_000, 100, SEED);
    group.throughput(Throughput::Elements(1_000_000));

    for fraction in [0.01, 0.10, 0.50, 0.90, 1.00] {
        let plan = query(fraction);
        bench_arms(
            &mut group,
            &table,
            &plan,
            format!("{:.0}%", fraction * 100.0),
        );
    }

    group.finish();
}

/// The axis this project has already been burned on.
///
/// The `Arc` dictionary bug was invisible at 8 distinct values and 84x at 10,000. Sweeping it
/// is what turns "the fix works" into something continuously verified rather than spot-checked.
fn cardinality(c: &mut Criterion) {
    let mut group = c.benchmark_group("cardinality");
    let plan = query(0.5);
    group.throughput(Throughput::Elements(1_000_000));

    for cardinality in [8u64, 128, 2_048, 32_768] {
        let table = generate(1_000_000, cardinality, SEED);
        bench_arms(&mut group, &table, &plan, cardinality);
    }

    group.finish();
}

criterion_group! {
    name = arms;
    config = configured();
    targets = row_count, selectivity, cardinality
}
criterion_main!(arms);
