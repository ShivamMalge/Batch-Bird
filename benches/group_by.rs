//! Hash group-by vs sort group-by, and the phase breakdown inside each.
//!
//! ```text
//! cargo bench --bench group_by
//! cargo +nightly bench --bench group_by --features simd
//! ```
//!
//! # Why phases are timed separately
//! `systemDesign.md` is emphatic that one combined "group-by" number would muddy the headline
//! finding, and it is right: the two halves scale differently and only one of them can be
//! vectorized.
//!
//! - **Hash phase 1, build-index** — a sequential read of the group column plus a hash probe
//!   per row. Partially vectorizes on the read side.
//! - **Hash phase 2, scatter-accumulate** — `accumulators[slot] += value` with a
//!   data-dependent slot. Cannot vectorize at all: SIMD can load eight values but cannot store
//!   them to eight computed addresses. This is the asymmetry the whole project exists to show.
//! - **Sort phase 1, sort-pairs** — the comparison sort plus the split into dense arrays.
//!   Expected to dominate the strategy.
//! - **Sort phase 2, aggregate-runs** — reduces contiguous runs, and is the only place the
//!   SIMD sum kernel reaches a real query.
//!
//! # The cardinality prediction to check
//! Sort-group's run lengths are `rows / cardinality`. At low cardinality runs are long and the
//! sum kernel has real work per call; as cardinality rises the runs shrink toward a single row
//! and per-call overhead should overtake the vector work. Where that crossover falls is a
//! measurement, not a prediction, and it is the interesting number in this file.
//!
//! Batches are materialized **once**, outside every timed region, so the phase timings measure
//! the phases rather than the scan and filter feeding them.

use std::time::Duration;

use batchbird::bench::data::{SEED, amount_threshold, generate};
use batchbird::exec::{Aggregate, GroupKind, Operator, RecordBatch, SortAggregate, SumAccumulator};
use batchbird::parser::parse;
use batchbird::plan::{GroupStrategy, LogicalPlan, batch_query_with, build};
use batchbird::storage::Table;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

fn configured() -> Criterion {
    Criterion::default()
        .sample_size(20)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3))
}

fn query(selectivity: f64) -> LogicalPlan {
    let threshold = amount_threshold(selectivity);
    parse(&format!(
        "SELECT region, SUM(amount) FROM t WHERE amount > {threshold} GROUP BY region"
    ))
    .expect("benchmark query must parse")
}

/// An operator that yields nothing.
///
/// Lets an `Aggregate` be constructed for phase timing without it pulling its own input — the
/// batches are fed in by hand, already materialized, so the scan and filter cost is not
/// charged to either phase.
struct NoInput;

impl Operator for NoInput {
    fn next_batch(&mut self) -> Option<RecordBatch> {
        None
    }
}

/// Collect the post-filter batches an aggregate would have seen, once.
///
/// `build` is validated against the same table first, so a plan this benchmark cannot run is
/// caught here rather than producing a silently empty measurement.
fn materialize_batches(table: &Table, plan: &LogicalPlan) -> Vec<RecordBatch> {
    build(table, plan).expect("plan must be valid for this table");

    let mut source = build_source(table, plan);
    let mut batches = Vec::new();
    while let Some(batch) = source.next_batch() {
        batches.push(batch);
    }
    batches
}

/// The `Scan -> Filter -> Project` chain below the aggregate.
///
/// Assembled directly rather than pulled out of `build`, because the aggregate at the top of a
/// built pipeline drains its input on the first `next_batch` and there is no way to intercept
/// the batches on the way past. The chain is identical to what `build` assembles.
fn build_source<'a>(table: &'a Table, plan: &LogicalPlan) -> impl Operator + 'a {
    use batchbird::exec::{Filter, FilterKind, Project, Scan};
    use batchbird::plan::{CompareOp, Literal};

    let threshold = match &plan.filter.literal {
        Literal::Int64(v) => *v,
        other => panic!("benchmark filter should be an integer literal, got {other:?}"),
    };
    assert_eq!(plan.filter.op, CompareOp::Gt);

    let mut scan_columns = vec![plan.group_by.clone(), plan.aggregation.input.clone()];
    if !scan_columns.contains(&plan.filter.column) {
        scan_columns.push(plan.filter.column.clone());
    }

    let scan = Scan::new(table, scan_columns.clone());
    let filter = Filter::new(
        scan,
        plan.filter.column.clone(),
        FilterKind::IntVsInt(CompareOp::Gt, threshold),
        scan_columns,
    );
    Project::new(
        filter,
        vec![plan.group_by.clone(), plan.aggregation.input.clone()],
    )
}

/// Whole-strategy comparison across the axis where they should trade places.
fn strategies(c: &mut Criterion) {
    let mut group = c.benchmark_group("strategy");
    let plan = query(0.5);
    group.throughput(Throughput::Elements(1_000_000));

    for cardinality in [8u64, 128, 2_048, 32_768, 250_000] {
        let table = generate(1_000_000, cardinality, SEED);

        group.bench_with_input(BenchmarkId::new("hash", cardinality), &plan, |b, plan| {
            b.iter(|| batch_query_with(&table, plan, GroupStrategy::Hash).expect("hash"))
        });

        group.bench_with_input(BenchmarkId::new("sort", cardinality), &plan, |b, plan| {
            b.iter(|| batch_query_with(&table, plan, GroupStrategy::Sort).expect("sort"))
        });
    }

    group.finish();
}

/// The phase breakdown `systemDesign.md` requires.
fn phases(c: &mut Criterion) {
    let mut group = c.benchmark_group("phases");
    let plan = query(0.5);

    for cardinality in [8u64, 2_048, 250_000] {
        let table = generate(1_000_000, cardinality, SEED);
        let batches = materialize_batches(&table, &plan);
        let rows: usize = batches.iter().map(|b| b.len()).sum();
        group.throughput(Throughput::Elements(rows as u64));

        let group_columns = vec![plan.group_by.clone()];
        let value_column = plan.aggregation.input.clone();
        let output_column = plan.aggregation.output_name();

        // ---- hash: build-index, then scatter-accumulate ----------------------------
        group.bench_with_input(
            BenchmarkId::new("hash/build-index", cardinality),
            &batches,
            |b, batches| {
                b.iter(|| {
                    let mut agg: Aggregate<NoInput, i64, SumAccumulator<i64>> = Aggregate::new(
                        NoInput,
                        group_columns.clone(),
                        GroupKind::Utf8,
                        value_column.clone(),
                        output_column.clone(),
                    );
                    for batch in batches {
                        agg.build_group_index(batch);
                    }
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("hash/scatter-accumulate", cardinality),
            &batches,
            |b, batches| {
                b.iter(|| {
                    let mut agg: Aggregate<NoInput, i64, SumAccumulator<i64>> = Aggregate::new(
                        NoInput,
                        group_columns.clone(),
                        GroupKind::Utf8,
                        value_column.clone(),
                        output_column.clone(),
                    );
                    // Phase 2 needs phase 1's slots, so each batch runs both -- and the
                    // difference against the timing above is phase 2's own cost.
                    for batch in batches {
                        agg.build_group_index(batch);
                        agg.scatter_accumulate(batch);
                    }
                })
            },
        );

        // ---- sort: collect, sort, then reduce runs ---------------------------------
        group.bench_with_input(
            BenchmarkId::new("sort/sort-pairs", cardinality),
            &batches,
            |b, batches| {
                b.iter(|| {
                    let mut agg: SortAggregate<NoInput, i64> = SortAggregate::new(
                        NoInput,
                        group_columns.clone(),
                        GroupKind::Utf8,
                        value_column.clone(),
                        output_column.clone(),
                    );
                    for batch in batches {
                        agg.collect_batch(batch);
                    }
                    agg.sort_pairs();
                })
            },
        );

        // Sorted once, outside the loop: this times only the run reduction, which is the
        // phase the SIMD sum kernel actually accelerates.
        let mut sorted: SortAggregate<NoInput, i64> = SortAggregate::new(
            NoInput,
            group_columns.clone(),
            GroupKind::Utf8,
            value_column.clone(),
            output_column.clone(),
        );
        for batch in &batches {
            sorted.collect_batch(batch);
        }
        sorted.sort_pairs();

        group.bench_with_input(
            BenchmarkId::new("sort/aggregate-runs", cardinality),
            &sorted,
            |b, sorted| b.iter(|| sorted.aggregate_runs()),
        );
    }

    group.finish();
}

criterion_group! {
    name = group_by;
    config = configured();
    targets = strategies, phases
}
criterion_main!(group_by);
