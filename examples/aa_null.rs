//! A/A null test: what is the smallest difference this harness can honestly resolve?
//!
//! Registers **the same arm twice** under different names and alternates them sample by
//! sample. The true difference is zero by construction, so whatever spread comes back is the
//! harness's resolution — the noise floor of the measurement itself.
//!
//! ```text
//! cargo run --release --example aa_null
//! ```
//!
//! # Why this and not "run the harness twice and compare"
//! Because that would be a between-process comparison again, which is the thing that made
//! every Phase 6 number unusable: two runs of an identical binary on identical data differed
//! by a median of 7.5% and a maximum of 66.8%. The floor that matters is the one *inside* a
//! single run, because that is where the real comparison will be read.
//!
//! # Why the arms cannot collapse
//! `Arm` holds a boxed closure, so each registration is a separate indirect call the optimizer
//! cannot merge. The output confirms it: both arms report substantial, comparable times. If
//! one had been optimized away it would report near zero.
//!
//! # Swept, not spot-checked
//! Resolution can depend on working-set size, so the null runs across cardinality and
//! selectivity as well — the same axes the real comparison uses.

use batchbird::bench::data::{SEED, amount_threshold, environment, generate};
use batchbird::bench::harness::{Arm, alternate, report, result_checksum};

/// Order-insensitive, because the group-by strategies emit groups in different orders.
fn result_checksum_of(table: &Table) -> u64 {
    result_checksum(table, "region", "SUM(amount)")
}
use batchbird::bench::naive_query;
use batchbird::parser::parse;
use batchbird::plan::LogicalPlan;
use batchbird::storage::Table;

const ROWS: usize = 1_000_000;
const SAMPLES: usize = 40;
const WARMUP: usize = 3;

fn query(selectivity: f64) -> LogicalPlan {
    let threshold = amount_threshold(selectivity);
    parse(&format!(
        "SELECT region, SUM(amount) FROM t WHERE amount > {threshold} GROUP BY region"
    ))
    .expect("benchmark query must parse")
}

/// One A/A point: the identical arm, registered twice.
fn null(label: &str, table: &Table, plan: &LogicalPlan) -> f64 {
    let arms = vec![
        Arm::new("A", || {
            result_checksum_of(&naive_query(table, plan).expect("query"))
        }),
        Arm::new("A'", || {
            result_checksum_of(&naive_query(table, plan).expect("query"))
        }),
    ];

    let measurements = alternate(arms, SAMPLES, WARMUP);
    println!("\n{label}");
    report(&measurements);

    let (a, b) = (&measurements[0], &measurements[1]);
    assert_eq!(a.checksum, b.checksum, "the two registrations disagree");
    assert!(
        a.min() > 0.001 && b.min() > 0.001,
        "an arm reported near-zero: the two registrations were merged, and this test is void"
    );

    // The number that matters: how far apart two identical arms landed.
    let delta = (a.min() / b.min() - 1.0).abs() * 100.0;
    println!("  A/A delta on min: {delta:.2}%");
    delta
}

fn main() {
    println!("{}\n", environment());
    println!(
        "A/A null: the same arm registered twice, alternated ABAB, {SAMPLES} samples after \
         {WARMUP} warmup rounds.\nTrue difference is zero; whatever appears is resolution."
    );

    let mut deltas = Vec::new();

    println!("\n=== cardinality sweep (selectivity 50%) ===");
    let plan = query(0.5);
    for cardinality in [8u64, 2_048, 250_000] {
        let table = generate(ROWS, cardinality, SEED);
        deltas.push(null(&format!("cardinality {cardinality}"), &table, &plan));
    }

    println!("\n=== selectivity sweep (cardinality 128) ===");
    let table = generate(ROWS, 128, SEED);
    for fraction in [0.01, 0.5, 1.0] {
        let plan = query(fraction);
        deltas.push(null(
            &format!("selectivity {:.0}%", fraction * 100.0),
            &table,
            &plan,
        ));
    }

    deltas.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
    let median = deltas[deltas.len() / 2];
    let worst = deltas.last().copied().unwrap_or(0.0);

    println!("\n=== resolution ===");
    println!(
        "A/A delta on min:  median {median:.2}%   worst {worst:.2}%   (n={})",
        deltas.len()
    );
    println!(
        "\nDecision rule: the smallest effect to be claimed is the ~19% execution-model gap.\n\
         A resolution well under that means the harness can support the claim; a resolution\n\
         near the 4.2% between-process floor would mean in-process alternation bought nothing."
    );
}
