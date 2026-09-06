//! Every execution arm must produce the same answer.
//!
//! The columnar naive baseline is the oracle (`agents.md` Testing Expectations). These are
//! integration tests rather than unit tests because the point is precisely that *whole
//! engines* agree — anything reaching inside one would be testing something narrower.
//!
//! Three arms are compared here: row-oriented storage, columnar row-at-a-time, and the
//! columnar batch pipeline (which the `simd` feature turns into the fourth arm without
//! changing its results). A benchmark that compares arms producing different answers is
//! measuring nothing, so this suite is what makes the numbers in `phases.md` mean anything.
//!
//! # Integers exact, floats within a tolerance — and why the line is drawn there
//! Integer sums are asserted **bit-identical** across every arm and both strategies. Integer
//! addition stays associative even when it wraps, and every arm wraps, so nothing about
//! ordering or vectorization can make them disagree. A tolerance there would hide real bugs.
//!
//! Float sums get a **relative tolerance**, because two arms genuinely accumulate in different
//! orders and floating-point addition is not associative:
//! - Row-oriented, columnar naive, and hash group-by all visit surviving rows in the same
//!   global order, so they agree bit-for-bit.
//! - **Sort group-by does not.** It sorts rows by key, so a group's values are summed in key
//!   order rather than file order — and under `--features simd` its run reduction is lane-wise
//!   as well, eight partial sums combined at the end.
//!
//! This is the change Phase 5's notes predicted for Phase 6. Phase 5 itself kept bit-exactness
//! because the SIMD sum never reached the query path: hash group-by scatters into per-group
//! accumulators and has no dense slice to hand a kernel. Sort group-by is what puts it there.
//!
//! Neither order is "wrong", and the lane-wise one is often *more* accurate — eight shorter
//! chains accumulate less rounding error than one long one. `float_sum_orders_actually_differ`
//! pins that the orders really do diverge, so the tolerance cannot pass vacuously.

use batchbird::bench::{naive_query, row_query};
use batchbird::parser::parse;
use batchbird::plan::{GroupStrategy, batch_query, batch_query_with};
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

/// One result row's sum: exact for integers, compared with a tolerance for floats.
#[derive(Debug, Clone, PartialEq)]
enum Sum {
    Int(i64),
    Float(f64),
}

/// Result rows as sortable `(group, sum)` pairs.
///
/// Sorting is required, not cosmetic: group order is unspecified everywhere and the arms
/// genuinely differ — first-seen over all rows, first-seen over surviving rows, and key order
/// for sort-group. Comparing raw `Table`s would fail on ordering alone.
fn rows_of(result: &Table, group: &str, agg: &str) -> Vec<(String, Sum)> {
    let group_col = result.column(group).expect("group column");
    let agg_col = result.column(agg).expect("agg column");

    let mut rows: Vec<(String, Sum)> = (0..result.nrows())
        .map(|row| {
            let key = match group_col {
                Column::Int64(v) => v[row].to_string(),
                Column::Utf8Dict { .. } => group_col.utf8_value(row).expect("label").to_string(),
                Column::Float64(_) => unreachable!("floats cannot be group keys"),
            };
            let sum = match agg_col {
                Column::Int64(v) => Sum::Int(v[row]),
                Column::Float64(v) => Sum::Float(v[row]),
                Column::Utf8Dict { .. } => unreachable!("sums are numeric"),
            };
            (key, sum)
        })
        .collect();

    rows.sort_by(|a, b| a.0.cmp(&b.0));
    rows
}

/// Assert two arms agree: integers exactly, floats within a relative tolerance.
///
/// See the module docs for why the line falls there. The tolerance absorbs a different
/// summation order over millions of values and is far too tight to absorb an arithmetic bug.
fn assert_rows_match(expected: &[(String, Sum)], actual: &[(String, Sum)], context: &str) {
    assert_eq!(
        expected.len(),
        actual.len(),
        "{context}: different number of groups"
    );

    for ((expected_key, expected_sum), (actual_key, actual_sum)) in expected.iter().zip(actual) {
        assert_eq!(expected_key, actual_key, "{context}: group labels differ");

        match (expected_sum, actual_sum) {
            (Sum::Int(a), Sum::Int(b)) => assert_eq!(
                a, b,
                "{context}: integer sums must be bit-identical for group {expected_key:?}"
            ),
            (Sum::Float(a), Sum::Float(b)) => {
                let tolerance = a.abs() * 1e-12 + 1e-9;
                assert!(
                    (a - b).abs() <= tolerance,
                    "{context}: float sums differ beyond rounding for {expected_key:?}: {a} vs {b}"
                );
            }
            _ => panic!("{context}: result types differ for group {expected_key:?}"),
        }
    }
}

/// Run every arm and assert they agree exactly with the oracle.
fn assert_parity(table: &Table, sql: &str) -> usize {
    let plan = parse(sql).unwrap_or_else(|e| panic!("{sql:?} did not parse: {e}"));

    let naive = naive_query(table, &plan).unwrap_or_else(|e| panic!("naive failed: {e}"));
    let group = &plan.group_by;
    let agg = plan.aggregation.output_name();
    let expected = rows_of(&naive, group, &agg);

    let arms = [
        ("batched/hash", batch_query(table, &plan)),
        (
            "batched/sort",
            batch_query_with(table, &plan, GroupStrategy::Sort),
        ),
        (
            "batched/sort-radix",
            batch_query_with(table, &plan, GroupStrategy::SortRadix),
        ),
        ("row-oriented", row_query(table, &plan)),
    ];

    for (name, result) in arms {
        let result = result.unwrap_or_else(|e| panic!("{name} failed on {sql:?}: {e}"));

        assert_eq!(
            naive.nrows(),
            result.nrows(),
            "{name}: group count differs for {sql:?}"
        );

        let mut expected_names: Vec<&str> = naive.column_names().collect();
        let mut actual_names: Vec<&str> = result.column_names().collect();
        expected_names.sort();
        actual_names.sort();
        assert_eq!(
            expected_names, actual_names,
            "{name}: result schema differs for {sql:?}"
        );

        assert_rows_match(
            &expected,
            &rows_of(&result, group, &agg),
            &format!("{name} on {sql:?}"),
        );
    }

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
fn the_row_oriented_arm_holds_an_independent_copy_of_the_data() {
    // Not a correctness property but a fairness one: if the row store were somehow sharing
    // the dictionary, it would not be measuring row-oriented layout at all. Every row owns
    // its own text, which is exactly the memory cost the comparison is meant to expose.
    use batchbird::bench::build_row_store;

    let table = synthetic_table(5_000, 6, 0x0BEEF);
    let plan = parse("SELECT region, SUM(amount) FROM t WHERE amount > 0 GROUP BY region").unwrap();

    let store = build_row_store(&table, &plan).unwrap();
    assert_eq!(store.rows(), 5_000, "one struct per source row");
}

#[test]
fn arms_sharing_an_accumulation_order_stay_bit_for_bit() {
    // The arms that visit rows in the same order must agree exactly -- a tolerance here would
    // be slack the implementation has not earned. Only sort-group is exempt, and only because
    // it genuinely sums in a different order.
    let table = synthetic_table(5_000, 8, 0xF10A);
    let plan =
        parse("SELECT region, SUM(price) FROM t WHERE price > -250 GROUP BY region").unwrap();

    let naive = naive_query(&table, &plan).unwrap();
    let batched = batch_query(&table, &plan).unwrap();
    let rows = row_query(&table, &plan).unwrap();

    let naive_sums = naive.column("SUM(price)").unwrap().as_f64().unwrap();
    assert!(
        naive_sums.iter().any(|v| v.fract() != 0.0),
        "the test data should produce genuinely fractional sums"
    );

    let expected = rows_of(&naive, "region", "SUM(price)");
    assert_eq!(expected, rows_of(&batched, "region", "SUM(price)"));
    assert_eq!(expected, rows_of(&rows, "region", "SUM(price)"));
}

#[test]
fn sort_group_float_sums_diverge_under_simd() {
    // Guards the float tolerance against being slack nobody needs.
    //
    // On the scalar path this test cannot show a divergence, and saying so is the honest
    // version: with a single group, sorting does not reorder anything, so sort-group sums in
    // the same order as everyone else. The divergence the tolerance exists for is the
    // *lane-wise* reduction, which only runs with `--features simd`.
    //
    // One huge value, then many small ones: 1.0 sits below the ULP of 2^53, so a serial chain
    // absorbs every addend into nothing while eight parallel lanes accumulate the small values
    // together and keep them.
    let mut csv = String::from(
        "k,v
",
    );
    for i in 0..4_000 {
        if i % 2_000 == 0 {
            csv.push_str(
                "big,9007199254740992.0
",
            );
        }
        csv.push_str(
            "big,1.0
",
        );
    }

    let table = table(&csv);
    let plan = parse("SELECT k, SUM(v) FROM t WHERE v > 0 GROUP BY k").unwrap();

    let naive = naive_query(&table, &plan).unwrap();
    let sorted = batch_query_with(&table, &plan, GroupStrategy::Sort).unwrap();

    let naive_sum = naive.column("SUM(v)").unwrap().as_f64().unwrap()[0];
    let sorted_sum = sorted.column("SUM(v)").unwrap().as_f64().unwrap()[0];
    assert!(naive_sum.is_finite() && sorted_sum.is_finite());

    // Whatever the order, the answers must still agree to within rounding.
    assert_rows_match(
        &rows_of(&naive, "k", "SUM(v)"),
        &rows_of(&sorted, "k", "SUM(v)"),
        "sort-group vs naive",
    );

    #[cfg(feature = "simd")]
    assert_ne!(
        naive_sum, sorted_sum,
        "lane-wise reduction should reach a different sum than the serial chain here;          if this passes trivially the tolerance is no longer testing anything"
    );

    #[cfg(not(feature = "simd"))]
    assert_eq!(
        naive_sum, sorted_sum,
        "on the scalar path a single group is summed in the same order by both"
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
        let sorted = batch_query_with(&table, &plan, GroupStrategy::Sort);
        let radix = batch_query_with(&table, &plan, GroupStrategy::SortRadix);
        let rows = row_query(&table, &plan);

        assert!(naive.is_err(), "naive accepted {sql:?}");
        assert!(batched.is_err(), "batched accepted {sql:?}");
        assert!(sorted.is_err(), "sort-group accepted {sql:?}");
        assert!(radix.is_err(), "sort-radix accepted {sql:?}");
        assert!(rows.is_err(), "row-oriented accepted {sql:?}");

        // Identical messages, not merely identical failure: the three arms validate
        // independently (agents.md forbids the baselines sharing code with the batch engine),
        // so matching text is what proves the duplication has not drifted.
        let reason = naive.unwrap_err().to_string();
        assert_eq!(
            reason,
            batched.unwrap_err().to_string(),
            "batched disagrees on why {sql:?} is invalid"
        );
        assert_eq!(
            reason,
            sorted.unwrap_err().to_string(),
            "sort-group disagrees on why {sql:?} is invalid"
        );
        assert_eq!(
            reason,
            rows.unwrap_err().to_string(),
            "row-oriented disagrees on why {sql:?} is invalid"
        );
    }
}
