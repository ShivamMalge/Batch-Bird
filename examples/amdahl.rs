//! Does Amdahl actually predict the null result, or is it just a story told afterwards?
//!
//! ```text
//! cargo +nightly run --release --example amdahl --features bench-dispatch
//! ```
//!
//! Phase 5 measured the SIMD kernels at 1.46x, 1.59x, 1.07x and 3.81x. Phase 6 measured whole
//! queries at 0.94-1.05x. The explanation offered was "the kernels are a small slice" — which
//! is prose, not a model. This turns it into arithmetic:
//!
//! 1. **Measure the share.** Time each kernel in isolation over the same batches the query
//!    feeds it, and divide by the query's total time. That is `p`.
//! 2. **Predict.** Amdahl's ceiling for speeding up a fraction `p` by `s` is
//!    `1 / ((1 - p) + p/s)`.
//! 3. **Compare** against the observed query-level speedup, measured by alternating scalar and
//!    SIMD in this same process.
//!
//! If prediction and observation agree, the null result has a model behind it and generalizes:
//! it says what kernel speedup *would* have been needed. If they disagree, the disagreement is
//! the finding and gets reported as-is.
//!
//! The same decomposition answers a second question for free: what share of query time is
//! `Scan`'s unconditional copy, which is the one known unexploited optimization in the engine.

use batchbird::bench::data::{SEED, amount_threshold, environment, generate};
use batchbird::bench::harness::{Arm, alternate, result_checksum};
use batchbird::exec::kernels::{Kernel, dispatch};
use batchbird::exec::{
    Filter, FilterKind, GroupKind, Operator, Project, RecordBatch, Scan, SortAggregate,
};
use batchbird::parser::parse;
use batchbird::plan::{CompareOp, GroupStrategy, Literal, LogicalPlan, batch_query_with};
use batchbird::storage::Table;

const ROWS: usize = 1_000_000;
const SAMPLES: usize = 30;
const WARMUP: usize = 3;

/// Kernel speedups measured in isolation in Phase 5 (`examples/kernels.rs`, 16M rows).
const FILTER_I64_SPEEDUP: f64 = 1.46;
const SUM_I64_SPEEDUP: f64 = 1.07;

fn query(selectivity: f64) -> LogicalPlan {
    let threshold = amount_threshold(selectivity);
    parse(&format!(
        "SELECT region, SUM(amount) FROM t WHERE amount > {threshold} GROUP BY region"
    ))
    .expect("query must parse")
}

fn checksum(table: &Table) -> u64 {
    result_checksum(table, "region", "SUM(amount)")
}

/// Amdahl's ceiling: the best a whole can do when a fraction `p` of it gets `s` times faster.
fn amdahl(p: f64, s: f64) -> f64 {
    1.0 / ((1.0 - p) + p / s)
}

fn scan_columns(plan: &LogicalPlan) -> Vec<String> {
    let mut columns = vec![plan.group_by.clone(), plan.aggregation.input.clone()];
    if !columns.contains(&plan.filter.column) {
        columns.push(plan.filter.column.clone());
    }
    columns
}

fn threshold_of(plan: &LogicalPlan) -> i64 {
    match &plan.filter.literal {
        Literal::Int64(v) => *v,
        other => panic!("expected an integer literal, got {other:?}"),
    }
}

/// Batches as the aggregate sees them: scanned, filtered, projected.
fn projected_batches(table: &Table, plan: &LogicalPlan) -> Vec<RecordBatch> {
    let columns = scan_columns(plan);
    let scan = Scan::new(table, columns.clone());
    let filter = Filter::new(
        scan,
        plan.filter.column.clone(),
        FilterKind::IntVsInt(CompareOp::Gt, threshold_of(plan)),
        columns,
    );
    let mut project = Project::new(
        filter,
        vec![plan.group_by.clone(), plan.aggregation.input.clone()],
    );

    let mut batches = Vec::new();
    while let Some(batch) = project.next_batch() {
        batches.push(batch);
    }
    batches
}

/// Raw batches straight off the scan, before filtering.
fn scanned_batches(table: &Table, plan: &LogicalPlan) -> Vec<RecordBatch> {
    let mut scan = Scan::new(table, scan_columns(plan));
    let mut batches = Vec::new();
    while let Some(batch) = scan.next_batch() {
        batches.push(batch);
    }
    batches
}

fn main() {
    println!("{}\n", environment());
    println!(
        "Amdahl check: {ROWS} rows, 50% selectivity, {SAMPLES} samples, min-of-N, arms\n\
         alternated in-process. Kernel speedups quoted from Phase 5 at 16M rows.\n"
    );

    let plan = query(0.5);

    for cardinality in [128u64, 250_000] {
        let table = generate(ROWS, cardinality, SEED);
        println!("\n================ cardinality {cardinality} ================");

        // ---- 1. Stage decomposition, all alternated against each other ----------------
        let scanned = scanned_batches(&table, &plan);
        let projected = projected_batches(&table, &plan);
        let surviving: usize = projected.iter().map(|b| b.len()).sum();

        let mask_filter = Filter::new(
            Scan::new(&table, scan_columns(&plan)),
            plan.filter.column.clone(),
            FilterKind::IntVsInt(CompareOp::Gt, threshold_of(&plan)),
            scan_columns(&plan),
        );

        let arms = vec![
            // The whole thing, hash strategy.
            Arm::new("query (hash)", || {
                checksum(&batch_query_with(&table, &plan, GroupStrategy::Hash).expect("q"))
            }),
            // Just the filter kernel, over exactly the batches the query feeds it.
            Arm::new("  filter kernel", || {
                let mut ones = 0u64;
                for batch in &scanned {
                    ones += mask_filter.mask(batch).count_ones() as u64;
                }
                ones
            }),
            // Scan on its own: the unconditional copy, with nothing downstream.
            Arm::new("  scan (copy)", || {
                let mut scan = Scan::new(&table, scan_columns(&plan));
                let mut rows = 0u64;
                while let Some(batch) = scan.next_batch() {
                    rows += batch.len() as u64;
                }
                rows
            }),
        ];

        let m = alternate(arms, SAMPLES, WARMUP);
        let (query_ms, filter_ms, scan_ms) = (m[0].min(), m[1].min(), m[2].min());

        println!("\n  {:<18} {:>9}  {:>7}", "stage", "min (ms)", "share");
        for (measurement, ms) in m.iter().zip([query_ms, filter_ms, scan_ms]) {
            println!(
                "  {:<18} {ms:>9.3}  {:>6.1}%",
                measurement.name,
                ms / query_ms * 100.0
            );
        }
        println!("  ({surviving} of {ROWS} rows survive the filter)");

        // ---- 2. Predict from the share -----------------------------------------------
        let p_filter = filter_ms / query_ms;
        let predicted = amdahl(p_filter, FILTER_I64_SPEEDUP);

        // ---- 3. Observe, alternated in this same process ------------------------------
        let observed_arms = vec![
            Arm::new("scalar", || {
                dispatch::set(Kernel::Scalar);
                checksum(&batch_query_with(&table, &plan, GroupStrategy::Hash).expect("q"))
            }),
            Arm::new("simd", || {
                dispatch::set(Kernel::Simd);
                checksum(&batch_query_with(&table, &plan, GroupStrategy::Hash).expect("q"))
            }),
        ];
        let o = alternate(observed_arms, SAMPLES, WARMUP);
        let observed = o[0].min() / o[1].min();

        println!("\n  hash path, filter kernel:");
        println!("    share of query      p = {:.3}", p_filter);
        println!("    kernel speedup      s = {FILTER_I64_SPEEDUP:.2}x  (Phase 5, isolated)");
        println!("    Amdahl ceiling          {predicted:.3}x");
        println!("    observed                {observed:.3}x");
        println!(
            "    agreement               {:+.1} percentage points",
            (observed - predicted) * 100.0
        );

        // ---- 4. The sum kernel, which only sort-group reaches --------------------------
        let mut sorter: SortAggregate<_, i64> = SortAggregate::new(
            Scan::new(&table, vec![plan.group_by.clone()]),
            vec![plan.group_by.clone()],
            GroupKind::Utf8,
            plan.aggregation.input.clone(),
            plan.aggregation.output_name(),
        );
        for batch in &projected {
            sorter.collect_batch(batch);
        }
        sorter.sort_pairs();

        let sort_arms = vec![
            Arm::new("query (sort/radix)", || {
                checksum(&batch_query_with(&table, &plan, GroupStrategy::SortRadix).expect("q"))
            }),
            Arm::new("  sum kernel (runs)", || {
                let (keys, _) = sorter.aggregate_runs();
                keys.len() as u64
            }),
        ];
        let sm = alternate(sort_arms, SAMPLES, WARMUP);
        let p_sum = sm[1].min() / sm[0].min();
        let predicted_sum = amdahl(p_sum, SUM_I64_SPEEDUP);

        println!("\n  sort path, sum kernel:");
        println!("    sort query              {:.3} ms", sm[0].min());
        println!("    run reduction           {:.3} ms", sm[1].min());
        println!("    share of query      p = {p_sum:.3}");
        println!("    kernel speedup      s = {SUM_I64_SPEEDUP:.2}x  (Phase 5, isolated)");
        println!("    Amdahl ceiling          {predicted_sum:.3}x");

        // ---- 5. What copy-on-scan is worth, from the same decomposition ---------------
        println!("\n  copy-on-scan:");
        println!(
            "    Scan is {:.1}% of the query. A borrowing Scan removes the write half of that\n\
             \x20   copy, so the ceiling on removing it entirely is {:.3}x — and less in practice,\n\
             \x20   since the read is compulsory either way.",
            scan_ms / query_ms * 100.0,
            amdahl(scan_ms / query_ms, 2.0),
        );
    }
}
