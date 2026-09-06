//! Phase 4's acceptance criterion: the batch pipeline must agree with the naive baseline.
//!
//! The baseline is the oracle (`agents.md` Testing Expectations). These are integration tests
//! rather than unit tests because the point is precisely that two *whole engines* agree —
//! anything reaching inside either one would be testing something narrower.
//!
//! # Why equality can be exact, even for floats
//! Both engines visit surviving rows in the same global order: the baseline sweeps rows
//! 0..n, and the pipeline sweeps batches in order and rows in order within each batch, with
//! compaction preserving relative order. So every group's accumulator sees the same values in
//! the same sequence, and `f64` sums come out bit-identical despite floating-point addition
//! being non-associative.
//!
//! Phase 5 did **not** change this, contrary to what the phase notes predicted. The SIMD
//! filter is exact (a comparison has no rounding), and the SIMD sum reduction is not on this
//! path at all — the hash group-by scatters into per-group accumulators rather than reducing
//! a dense slice. These assertions therefore still hold bit-for-bit under
//! `cargo +nightly test --features simd`, which is a stronger check than a tolerance would be.
//!
//! **Phase 6 is where it changes.** Sort-based grouping makes each group contiguous, so its
//! aggregation *is* a dense reduction and will use the lane-wise sum. Float comparisons
//! against this engine will need a tolerance then; integer sums stay exact regardless,
//! because every path wraps.

use batchbird::bench::naive_query;
use batchbird::parser::parse;
use batchbird::plan::batch_query;
use batchbird::storage::{Column, DataType, Field, Schema, Table, infer_schema, read_csv};

/// Deterministic pseudo-random numbers, so a failure is always reproducible.
///
/// xorshift64* rather than a dependency: the data only needs to be varied and repeatable,
/// not statistically sound.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// A CSV with a string group column, an int group column, and int/float measures.
///
/// `cardinality` controls how many distinct regions appear, which matters because it drives
/// both the number of accumulator slots and the per-batch dictionary copy.
fn synthetic_csv(rows: usize, cardinality: u64, seed: u64) -> String {
    let mut rng = Rng(seed);
    let mut csv = String::from("region,bucket,amount,price\n");

    for _ in 0..rows {
        let region = rng.below(cardinality);
        let bucket = rng.below(7) as i64 - 3;
        // Spans negatives and zero, so filters actually cut somewhere interesting.
        let amount = rng.below(2000) as i64 - 1000;
        let price = (rng.below(100_000) as f64) / 100.0 - 500.0;
        csv.push_str(&format!("r{region},{bucket},{amount},{price}\n"));
    }
    csv
}

fn table(csv: &str) -> Table {
    let schema = infer_schema(csv.as_bytes()).expect("schema inference");
    read_csv(csv.as_bytes(), &schema).expect("csv load")
}

/// A synthetic table with its schema **declared**, not inferred.
///
/// Inference needs data to look at: a zero-row file types every column `Utf8` (Phase 1), which
/// `SUM` then rightly rejects — so the empty-input case cannot be built by inference at all.
/// Declaring the schema is both the fix and what Phase 6's generator will do anyway, since it
/// knows the shape of the data it just produced.
fn synthetic_table(rows: usize, cardinality: u64, seed: u64) -> Table {
    let csv = synthetic_csv(rows, cardinality, seed);
    let schema = Schema::new(vec![
        Field::new("region", DataType::Utf8),
        Field::new("bucket", DataType::Int64),
        Field::new("amount", DataType::Int64),
        Field::new("price", DataType::Float64),
    ]);
    read_csv(csv.as_bytes(), &schema).expect("csv load")
}

/// Result rows as sortable `(group, sum)` pairs.
///
/// Sorting is required, not cosmetic: group order is unspecified in both engines and they
/// genuinely differ — the baseline emits first-seen order over all rows, the pipeline over
/// surviving rows. Comparing raw `Table`s would fail on ordering alone.
///
/// Sums are compared through their exact bit patterns (`to_bits` for floats), so this cannot
/// paper over a last-ULP difference the way formatting would.
fn rows_of(result: &Table, group: &str, agg: &str) -> Vec<(String, u64)> {
    let group_col = result.column(group).expect("group column");
    let agg_col = result.column(agg).expect("agg column");

    let mut rows: Vec<(String, u64)> = (0..result.nrows())
        .map(|row| {
            let key = match group_col {
                Column::Int64(v) => v[row].to_string(),
                Column::Utf8Dict { .. } => group_col.utf8_value(row).expect("label").to_string(),
                Column::Float64(_) => unreachable!("floats cannot be group keys"),
            };
            let sum = match agg_col {
                Column::Int64(v) => v[row] as u64,
                Column::Float64(v) => v[row].to_bits(),
                Column::Utf8Dict { .. } => unreachable!("sums are numeric"),
            };
            (key, sum)
        })
        .collect();

    rows.sort();
    rows
}

/// Run both engines and assert they agree exactly.
fn assert_parity(table: &Table, sql: &str) -> usize {
    let plan = parse(sql).unwrap_or_else(|e| panic!("{sql:?} did not parse: {e}"));

    let naive = naive_query(table, &plan).unwrap_or_else(|e| panic!("naive failed: {e}"));
    let batched = batch_query(table, &plan).unwrap_or_else(|e| panic!("batched failed: {e}"));

    assert_eq!(
        naive.nrows(),
        batched.nrows(),
        "group count differs for {sql:?}"
    );

    let mut naive_names: Vec<&str> = naive.column_names().collect();
    let mut batched_names: Vec<&str> = batched.column_names().collect();
    naive_names.sort();
    batched_names.sort();
    assert_eq!(
        naive_names, batched_names,
        "result schema differs for {sql:?}"
    );

    let group = &plan.group_by;
    let agg = plan.aggregation.output_name();
    assert_eq!(
        rows_of(&naive, group, &agg),
        rows_of(&batched, group, &agg),
        "results differ for {sql:?}"
    );

    naive.nrows()
}

/// Every query shape the engine supports, over one dataset.
fn all_supported_queries() -> Vec<&'static str> {
    vec![
        // String group column, both sum types, all three operators.
        "SELECT region, SUM(amount) FROM t WHERE amount > 0 GROUP BY region",
        "SELECT region, SUM(amount) FROM t WHERE amount < 0 GROUP BY region",
        "SELECT region, SUM(amount) FROM t WHERE bucket = 1 GROUP BY region",
        "SELECT region, SUM(price) FROM t WHERE amount > 500 GROUP BY region",
        // Integer group column, including the negative buckets.
        "SELECT bucket, SUM(amount) FROM t WHERE amount > -900 GROUP BY bucket",
        "SELECT bucket, SUM(price) FROM t WHERE price < 100.5 GROUP BY bucket",
        // Filtering on the group column itself.
        "SELECT bucket, SUM(amount) FROM t WHERE bucket > 0 GROUP BY bucket",
        // Filtering on the summed column itself.
        "SELECT region, SUM(amount) FROM t WHERE amount > 100 GROUP BY region",
        // Grouping and summing the same column.
        "SELECT bucket, SUM(bucket) FROM t WHERE amount > 0 GROUP BY bucket",
        // Mixed numeric comparisons: int column vs float literal, float column vs int literal.
        "SELECT region, SUM(amount) FROM t WHERE amount > 250.5 GROUP BY region",
        "SELECT region, SUM(price) FROM t WHERE price > 0 GROUP BY region",
        // String filters: equality resolves to a dictionary code, ordering compares strings.
        "SELECT region, SUM(amount) FROM t WHERE region = 'r0' GROUP BY region",
        "SELECT region, SUM(amount) FROM t WHERE region > 'r1' GROUP BY region",
        "SELECT region, SUM(amount) FROM t WHERE region < 'r2' GROUP BY region",
    ]
}

#[test]
fn engines_agree_across_every_supported_query_shape() {
    let table = synthetic_table(5_000, 6, 0x5EED);

    for sql in all_supported_queries() {
        let groups = assert_parity(&table, sql);
        assert!(
            groups > 0,
            "{sql:?} matched nothing; the test would be vacuous"
        );
    }
}

#[test]
fn engines_agree_across_batch_boundaries() {
    // BATCH_SIZE is 1024, so these row counts put the last batch exactly on, just under, and
    // just over a boundary — where an off-by-one in slicing or compaction would show up.
    for rows in [0, 1, 2, 1023, 1024, 1025, 2047, 2048, 2049, 5000] {
        let table = synthetic_table(rows, 4, 0xB0A7);
        for sql in [
            "SELECT region, SUM(amount) FROM t WHERE amount > 0 GROUP BY region",
            "SELECT bucket, SUM(price) FROM t WHERE price > -100 GROUP BY bucket",
        ] {
            assert_parity(&table, sql);
        }
    }
}

#[test]
fn engines_agree_when_the_filter_is_extremely_selective() {
    // Exercises Filter skipping whole batches: most batches produce nothing at all.
    let table = synthetic_table(4_000, 3, 0x11CE);
    assert_parity(
        &table,
        "SELECT region, SUM(amount) FROM t WHERE amount > 995 GROUP BY region",
    );
}

#[test]
fn engines_agree_when_the_filter_matches_nothing() {
    let table = synthetic_table(3_000, 3, 0xDEAD);

    let sql = "SELECT region, SUM(amount) FROM t WHERE amount > 100000 GROUP BY region";
    assert_eq!(assert_parity(&table, sql), 0, "expected an empty result");

    // A string literal absent from the dictionary takes a different path to the same place.
    let missing = "SELECT region, SUM(amount) FROM t WHERE region = 'nowhere' GROUP BY region";
    assert_eq!(assert_parity(&table, missing), 0);
}

#[test]
fn engines_agree_when_the_filter_matches_everything() {
    // The opposite corner: compaction copies every row rather than shrinking anything.
    let table = synthetic_table(3_000, 5, 0xFEED);
    assert_parity(
        &table,
        "SELECT region, SUM(amount) FROM t WHERE amount > -10000 GROUP BY region",
    );
}

#[test]
fn engines_agree_at_high_group_cardinality() {
    // Many accumulator slots, so phase 2's scatter writes are genuinely scattered rather than
    // landing in a handful of cache-resident totals. Also the case where the per-batch
    // dictionary copy is most expensive.
    let table = synthetic_table(6_000, 500, 0xCA5E);
    let groups = assert_parity(
        &table,
        "SELECT region, SUM(amount) FROM t WHERE amount > -500 GROUP BY region",
    );
    assert!(groups > 100, "expected many groups, got {groups}");
}

#[test]
fn engines_agree_when_every_row_is_its_own_group() {
    // Degenerate case: one row per group means the hash map never finds an existing slot.
    let mut csv = String::from("k,v\n");
    for i in 0..2_500 {
        csv.push_str(&format!("{i},{}\n", i * 3));
    }
    let table = table(&csv);

    let groups = assert_parity(&table, "SELECT k, SUM(v) FROM t WHERE v > 0 GROUP BY k");
    assert_eq!(groups, 2_499, "row 0 is filtered out by v > 0");
}

#[test]
fn engines_agree_when_all_rows_share_one_group() {
    // The other degenerate case: one slot, so every scatter write hits the same address.
    let mut csv = String::from("k,v\n");
    for i in 0..2_500 {
        csv.push_str(&format!("same,{i}\n"));
    }
    let table = table(&csv);

    assert_eq!(
        assert_parity(&table, "SELECT k, SUM(v) FROM t WHERE v > 10 GROUP BY k"),
        1
    );
}

#[test]
fn engines_agree_on_float_sums_bit_for_bit() {
    // Documented above: identical accumulation order means identical bits, today. When Phase 5
    // reduces across SIMD lanes this is the assertion that will start failing, and that
    // failure is informative rather than a bug — it is the non-associativity showing up.
    let table = synthetic_table(5_000, 8, 0xF10A);
    let plan =
        parse("SELECT region, SUM(price) FROM t WHERE price > -250 GROUP BY region").unwrap();

    let naive = naive_query(&table, &plan).unwrap();
    let batched = batch_query(&table, &plan).unwrap();

    let naive_sums = naive.column("SUM(price)").unwrap().as_f64().unwrap();
    assert!(
        naive_sums.iter().any(|v| v.fract() != 0.0),
        "the test data should produce genuinely fractional sums"
    );

    assert_eq!(
        rows_of(&naive, "region", "SUM(price)"),
        rows_of(&batched, "region", "SUM(price)")
    );
}

#[test]
fn both_engines_reject_the_same_invalid_plans() {
    // Parity is not just about results: an engine that accepted a query the other rejected
    // would make the benchmark a comparison between different questions.
    let table = synthetic_table(100, 3, 0xBAD5);

    let cases = [
        "SELECT region, SUM(missing) FROM t WHERE amount > 1 GROUP BY region",
        "SELECT missing, SUM(amount) FROM t WHERE amount > 1 GROUP BY missing",
        "SELECT price, SUM(amount) FROM t WHERE amount > 1 GROUP BY price",
        "SELECT region, SUM(region) FROM t WHERE amount > 1 GROUP BY region",
        "SELECT region, SUM(amount) FROM t WHERE region > 1 GROUP BY region",
        "SELECT region, SUM(amount) FROM t WHERE amount > 'x' GROUP BY region",
    ];

    for sql in cases {
        let plan = parse(sql).unwrap();
        let naive = naive_query(&table, &plan);
        let batched = batch_query(&table, &plan);

        assert!(naive.is_err(), "naive accepted {sql:?}");
        assert!(batched.is_err(), "batched accepted {sql:?}");
        assert_eq!(
            naive.unwrap_err().to_string(),
            batched.unwrap_err().to_string(),
            "engines disagree on why {sql:?} is invalid"
        );
    }
}
