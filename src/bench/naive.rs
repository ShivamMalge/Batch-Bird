//! The row-at-a-time baseline: the control the whole benchmark is measured against.
//!
//! # What this is for
//! Two jobs, and they pull in the same direction:
//! 1. **Benchmark control.** It is the "before" in naive vs. batched vs. SIMD (`prd.md`).
//! 2. **Correctness oracle.** From Phase 4 on, every execution strategy must agree with this
//!    one on the same input (`agents.md` Testing Expectations).
//!
//! # Do not refactor this to share code with the batch engine
//! `agents.md` and `phases.md` both call this out, and Phase 6 is where the temptation
//! arrives: writing the benchmarks means staring at both implementations side by side, and
//! the filter logic will look like it wants extracting. It does not. This file exists to
//! isolate row-at-a-time execution as the benchmark's single variable; sharing code with the
//! batch pipeline destroys exactly that isolation.
//!
//! # What is held constant, and what that means for the result
//! The baseline reads the **same columnar [`Table`]** as the batch engine, and uses the same
//! `hashbrown` map for grouping. Only the execution model differs -- one row at a time
//! against batches of 1024 -- which is what `architecture.md` means by "isolates
//! row-at-a-time as the variable, not operator overhead".
//!
//! The consequence is worth stating plainly in the write-up: this measures the *execution
//! model* half of the columnar thesis, not the *storage layout* half. A baseline over
//! row-oriented storage (array-of-structs) would be a different and harsher comparison,
//! because a column scan there touches every cache line. That is a bigger benchmark than
//! `phases.md` scopes, and adding it is a decision to make deliberately.
//!
//! # This is not a strawman
//! Column types are resolved **once**, before the loop, into typed slices -- the baseline
//! does not re-dispatch on the `Column` enum per row, and it is not handicapped with
//! artificial work. What remains inside the loop is one predictable branch per row on a
//! small enum, which is what row-at-a-time execution genuinely costs. Making the control
//! artificially slow would be adjusting the benchmark to force a cleaner story, which
//! `agents.md` explicitly forbids.

use std::collections::HashMap;

use crate::error::{Error, Result};
use crate::plan::{AggFunc, CompareOp, Literal, LogicalPlan, Predicate};
use crate::storage::{Column, Table};

/// Execute `plan` against `table` with a plain row loop.
///
/// Returns a small [`Table`] with one row per group: the group column under its original
/// name, and the sums under `SUM(<column>)`.
///
/// Group order is **unspecified but deterministic** -- groups come out in first-seen order.
/// Nothing should depend on it: SQL guarantees no ordering for `GROUP BY`, and the batch
/// engine's hash map will produce a different order for the same data.
pub fn naive_query(table: &Table, plan: &LogicalPlan) -> Result<Table> {
    let AggFunc::Sum = plan.aggregation.func;

    let group = GroupReader::new(table, &plan.group_by)?;
    let mut summer = Summer::new(table, &plan.aggregation.input)?;
    let filter = RowFilter::new(table, &plan.filter)?;

    // Group key -> slot. `hashbrown` rather than std for the same reason the batch engine
    // uses it (`techstack.md`): std's SipHash on small integer keys would dominate both
    // sides and blur the difference the benchmark is trying to show.
    let mut slots: hashbrown::HashMap<u64, usize> = hashbrown::HashMap::new();
    let mut keys: Vec<u64> = Vec::new();

    for row in 0..table.nrows() {
        if !filter.passes(row) {
            continue;
        }

        let key = group.key(row);
        let slot = match slots.get(&key) {
            Some(slot) => *slot,
            None => {
                let slot = keys.len();
                slots.insert(key, slot);
                keys.push(key);
                summer.push_group();
                slot
            }
        };

        summer.add(slot, row);
    }

    build_result(plan, &group, &keys, summer)
}

/// The group column, resolved to a typed slice.
enum GroupReader<'a> {
    Int64(&'a [i64]),
    /// Dictionary codes plus the dictionary, kept for materializing the result.
    Utf8 {
        dict: &'a [String],
        codes: &'a [u32],
    },
}

impl<'a> GroupReader<'a> {
    fn new(table: &'a Table, name: &str) -> Result<Self> {
        match column(table, name)? {
            Column::Int64(values) => Ok(GroupReader::Int64(values)),
            Column::Utf8Dict { dict, codes } => Ok(GroupReader::Utf8 { dict, codes }),
            // `GroupKey` holds a dictionary code or a raw i64 (`systemDesign.md`), so there
            // is no representation for a float key. Grouping on floats is a bad idea anyway
            // -- equality on f64 makes 0.0 and -0.0 the same group but distinct bit patterns.
            Column::Float64(_) => Err(Error::TypeError(format!(
                "cannot GROUP BY column {name:?}: it is Float64, and group keys must be \
                 Int64 or Utf8"
            ))),
        }
    }

    /// The row's group key, widened to `u64`.
    ///
    /// The `i64 -> u64` cast is a bit-cast, not a numeric conversion: it is bijective, so
    /// equality and hashing stay exact. It does not preserve numeric order (negatives land
    /// above positives), which does not matter because grouping only needs equality.
    #[inline]
    fn key(&self, row: usize) -> u64 {
        match self {
            GroupReader::Int64(values) => values[row] as u64,
            GroupReader::Utf8 { codes, .. } => codes[row] as u64,
        }
    }
}

/// The aggregated column plus its running per-group totals.
enum Summer<'a> {
    Int64 { values: &'a [i64], totals: Vec<i64> },
    Float64 { values: &'a [f64], totals: Vec<f64> },
}

impl<'a> Summer<'a> {
    fn new(table: &'a Table, name: &str) -> Result<Self> {
        match column(table, name)? {
            Column::Int64(values) => Ok(Summer::Int64 {
                values,
                totals: Vec::new(),
            }),
            Column::Float64(values) => Ok(Summer::Float64 {
                values,
                totals: Vec::new(),
            }),
            Column::Utf8Dict { .. } => Err(Error::TypeError(format!(
                "cannot SUM column {name:?}: it is Utf8"
            ))),
        }
    }

    fn push_group(&mut self) {
        match self {
            Summer::Int64 { totals, .. } => totals.push(0),
            Summer::Float64 { totals, .. } => totals.push(0.0),
        }
    }

    #[inline]
    fn add(&mut self, slot: usize, row: usize) {
        match self {
            // Wrapping, not checked or saturating. `Accumulator<T>` keeps Int64 sums in i64
            // rather than casting through f64 (`systemDesign.md`), and SIMD lane arithmetic
            // wraps -- so wrapping here is what lets the naive, batched, and SIMD paths agree
            // bit-for-bit on overflow instead of one of them panicking in a debug build.
            Summer::Int64 { values, totals } => {
                totals[slot] = totals[slot].wrapping_add(values[row]);
            }
            Summer::Float64 { values, totals } => totals[slot] += values[row],
        }
    }
}

/// The filter column and literal, resolved to a typed comparison.
enum RowFilter<'a> {
    IntVsInt(&'a [i64], CompareOp, i64),
    /// `WHERE int_col > 2.5`. The column value widens to `f64` per row, which is lossy past
    /// 2^53 -- acceptable for a threshold comparison, and the alternative is exact integer/
    /// float comparison logic that this query shape does not need.
    IntVsFloat(&'a [i64], CompareOp, f64),
    FloatVsFloat(&'a [f64], CompareOp, f64),
    /// String equality: the literal resolves to a dictionary code **once**, then the loop
    /// compares `u32`s. `None` means the literal is not in the dictionary at all, so no row
    /// can match -- the whole scan is a constant `false`.
    Utf8Eq(&'a [u32], Option<u32>),
    /// String ordering: no shortcut. Dictionary codes are in first-seen order, not lexical
    /// order, so `<` and `>` must decode and compare the strings themselves. This is the
    /// documented cost of leaving the dictionary unsorted (`systemDesign.md`), and it is why
    /// Phase 5 applies SIMD to numeric filters only.
    Utf8Ord {
        dict: &'a [String],
        codes: &'a [u32],
        op: CompareOp,
        literal: &'a str,
    },
}

impl<'a> RowFilter<'a> {
    fn new(table: &'a Table, predicate: &'a Predicate) -> Result<Self> {
        let name = &predicate.column;
        let op = predicate.op;

        match (column(table, name)?, &predicate.literal) {
            (Column::Int64(values), Literal::Int64(x)) => Ok(RowFilter::IntVsInt(values, op, *x)),
            (Column::Int64(values), Literal::Float64(x)) => {
                Ok(RowFilter::IntVsFloat(values, op, *x))
            }
            (Column::Float64(values), Literal::Float64(x)) => {
                Ok(RowFilter::FloatVsFloat(values, op, *x))
            }
            (Column::Float64(values), Literal::Int64(x)) => {
                Ok(RowFilter::FloatVsFloat(values, op, *x as f64))
            }
            (Column::Utf8Dict { dict, codes }, Literal::Utf8(literal)) => match op {
                CompareOp::Eq => {
                    let code = dict.iter().position(|d| d == literal).map(|i| i as u32);
                    Ok(RowFilter::Utf8Eq(codes, code))
                }
                _ => Ok(RowFilter::Utf8Ord {
                    dict,
                    codes,
                    op,
                    literal,
                }),
            },
            (column, literal) => Err(Error::TypeError(format!(
                "cannot compare column {name:?} ({}) against {}",
                column.data_type(),
                describe(literal)
            ))),
        }
    }

    #[inline]
    fn passes(&self, row: usize) -> bool {
        match self {
            RowFilter::IntVsInt(values, op, x) => compare(&values[row], x, *op),
            RowFilter::IntVsFloat(values, op, x) => compare(&(values[row] as f64), x, *op),
            RowFilter::FloatVsFloat(values, op, x) => compare(&values[row], x, *op),
            RowFilter::Utf8Eq(codes, code) => match code {
                Some(code) => codes[row] == *code,
                None => false,
            },
            RowFilter::Utf8Ord {
                dict,
                codes,
                op,
                literal,
            } => compare(&dict[codes[row] as usize].as_str(), literal, *op),
        }
    }
}

#[inline]
fn compare<T: PartialOrd + ?Sized>(a: &T, b: &T, op: CompareOp) -> bool {
    match op {
        CompareOp::Eq => a == b,
        CompareOp::Lt => a < b,
        CompareOp::Gt => a > b,
    }
}

fn describe(literal: &Literal) -> String {
    match literal {
        Literal::Int64(v) => format!("the integer {v}"),
        Literal::Float64(v) => format!("the float {v}"),
        Literal::Utf8(v) => format!("the string {v:?}"),
    }
}

/// Look up a column, naming the alternatives when it is missing.
fn column<'a>(table: &'a Table, name: &str) -> Result<&'a Column> {
    table.column(name).ok_or_else(|| {
        let mut available: Vec<String> = table.column_names().map(str::to_string).collect();
        available.sort();
        Error::UnknownColumn {
            column: name.to_string(),
            available,
        }
    })
}

/// The output column name for the aggregate, matching how SQL would label it.
pub fn sum_column_name(input: &str) -> String {
    format!("SUM({input})")
}

fn build_result(
    plan: &LogicalPlan,
    group: &GroupReader<'_>,
    keys: &[u64],
    summer: Summer<'_>,
) -> Result<Table> {
    let group_column = match group {
        // Recovering the i64 by casting back is exact -- see `GroupReader::key`.
        GroupReader::Int64(_) => Column::Int64(keys.iter().map(|k| *k as i64).collect()),
        GroupReader::Utf8 { dict, .. } => {
            // One row per group, so the result dictionary is exactly the group values and
            // the codes are simply 0..n.
            let dict = keys.iter().map(|k| dict[*k as usize].clone()).collect();
            let codes = (0..keys.len() as u32).collect();
            Column::Utf8Dict { dict, codes }
        }
    };

    let agg_column = match summer {
        Summer::Int64 { totals, .. } => Column::Int64(totals),
        Summer::Float64 { totals, .. } => Column::Float64(totals),
    };

    let mut columns = HashMap::new();
    columns.insert(plan.group_by.clone(), group_column);
    columns.insert(sum_column_name(&plan.aggregation.input), agg_column);

    Table::new(columns, keys.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;
    use crate::storage::{DataType, Field, Schema, infer_schema, read_csv};

    /// Five rows, three regions, mixed types. Small enough that every expectation below is
    /// hand-computed rather than produced by the code under test.
    ///
    /// ```text
    /// region  amount  price
    /// north   10      1.5
    /// south   -20     2.25
    /// north   30      3.0
    /// east    40      4.75
    /// north   50      5.5
    /// ```
    const SALES: &str = concat!(
        "region,amount,price\n",
        "north,10,1.5\n",
        "south,-20,2.25\n",
        "north,30,3.0\n",
        "east,40,4.75\n",
        "north,50,5.5\n",
    );

    fn table(csv: &str) -> Table {
        let schema = infer_schema(csv.as_bytes()).unwrap();
        read_csv(csv.as_bytes(), &schema).unwrap()
    }

    fn run(csv: &str, sql: &str) -> Table {
        let plan = parse(sql).unwrap_or_else(|e| panic!("{sql:?} did not parse: {e}"));
        naive_query(&table(csv), &plan).unwrap_or_else(|e| panic!("{sql:?} failed: {e}"))
    }

    fn error(csv: &str, sql: &str) -> Error {
        let plan = parse(sql).unwrap();
        naive_query(&table(csv), &plan).expect_err("expected the query to fail")
    }

    /// Result rows as `(group, sum)` pairs, sorted so assertions never depend on group order.
    fn pairs(result: &Table, group: &str, input: &str) -> Vec<(String, String)> {
        let group_col = result.column(group).expect("group column");
        let agg_col = result.column(&sum_column_name(input)).expect("agg column");

        let mut rows: Vec<(String, String)> = (0..result.nrows())
            .map(|row| {
                let key = match group_col {
                    Column::Int64(v) => v[row].to_string(),
                    Column::Utf8Dict { .. } => group_col.utf8_value(row).unwrap().to_string(),
                    Column::Float64(_) => unreachable!("floats cannot be group keys"),
                };
                let sum = match agg_col {
                    Column::Int64(v) => v[row].to_string(),
                    Column::Float64(v) => v[row].to_string(),
                    Column::Utf8Dict { .. } => unreachable!("sums are numeric"),
                };
                (key, sum)
            })
            .collect();
        rows.sort();
        rows
    }

    fn owned(rows: Vec<(&str, &str)>) -> Vec<(String, String)> {
        rows.into_iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn groups_by_string_and_sums_integers() {
        // price > 2 keeps every row except "north,10,1.5".
        // north: 30 + 50 = 80   south: -20   east: 40
        let result = run(
            SALES,
            "SELECT region, SUM(amount) FROM sales WHERE price > 2 GROUP BY region",
        );

        assert_eq!(result.nrows(), 3);
        assert_eq!(result.ncols(), 2);
        assert_eq!(
            pairs(&result, "region", "amount"),
            owned(vec![("east", "40"), ("north", "80"), ("south", "-20")])
        );
    }

    #[test]
    fn sums_floats() {
        // amount > 0 drops the south row. north: 1.5 + 3.0 + 5.5 = 10.0   east: 4.75
        let result = run(
            SALES,
            "SELECT region, SUM(price) FROM sales WHERE amount > 0 GROUP BY region",
        );
        assert_eq!(
            pairs(&result, "region", "price"),
            owned(vec![("east", "4.75"), ("north", "10")])
        );
    }

    #[test]
    fn groups_by_integer_column() {
        // Group keys go through the i64 -> u64 bit-cast, so negatives must survive intact.
        let csv = "k,v\n-1,10\n2,20\n-1,30\n2,40\n";
        let result = run(csv, "SELECT k, SUM(v) FROM t WHERE v > 5 GROUP BY k");
        assert_eq!(
            pairs(&result, "k", "v"),
            owned(vec![("-1", "40"), ("2", "60")])
        );
    }

    #[test]
    fn string_equality_filter_uses_dictionary_codes() {
        let result = run(
            SALES,
            "SELECT region, SUM(amount) FROM sales WHERE region = 'north' GROUP BY region",
        );
        assert_eq!(
            pairs(&result, "region", "amount"),
            owned(vec![("north", "90")])
        );
    }

    #[test]
    fn string_ordering_filter_compares_strings_not_codes() {
        // The dictionary is in first-seen order (north, south, east), so comparing codes
        // would give a different -- wrong -- answer here. Lexically, only "south" is
        // greater than "north".
        let result = run(
            SALES,
            "SELECT region, SUM(amount) FROM sales WHERE region > 'north' GROUP BY region",
        );
        assert_eq!(
            pairs(&result, "region", "amount"),
            owned(vec![("south", "-20")])
        );
    }

    #[test]
    fn literal_missing_from_the_dictionary_matches_nothing() {
        let result = run(
            SALES,
            "SELECT region, SUM(amount) FROM sales WHERE region = 'west' GROUP BY region",
        );
        assert_eq!(result.nrows(), 0);
    }

    #[test]
    fn filter_matching_no_rows_yields_an_empty_result() {
        let result = run(
            SALES,
            "SELECT region, SUM(amount) FROM sales WHERE amount > 1000 GROUP BY region",
        );
        assert_eq!(result.nrows(), 0);
        assert_eq!(result.ncols(), 2, "the columns exist, they are just empty");
    }

    #[test]
    fn numeric_columns_and_literals_mix() {
        // Int column vs float literal.
        let by_float = run(
            SALES,
            "SELECT region, SUM(amount) FROM sales WHERE amount > 25.5 GROUP BY region",
        );
        assert_eq!(
            pairs(&by_float, "region", "amount"),
            owned(vec![("east", "40"), ("north", "80")])
        );

        // Float column vs int literal.
        let by_int = run(
            SALES,
            "SELECT region, SUM(amount) FROM sales WHERE price < 3 GROUP BY region",
        );
        assert_eq!(
            pairs(&by_int, "region", "amount"),
            owned(vec![("north", "10"), ("south", "-20")])
        );
    }

    #[test]
    fn all_three_operators_agree_with_hand_counted_rows() {
        let cases = [
            (">", vec![("east", "40"), ("north", "50")]),
            ("<", vec![("north", "10"), ("south", "-20")]),
            ("=", vec![("north", "30")]),
        ];

        for (op, expected) in cases {
            let sql =
                format!("SELECT region, SUM(amount) FROM s WHERE amount {op} 30 GROUP BY region");
            assert_eq!(
                pairs(&run(SALES, &sql), "region", "amount"),
                owned(expected),
                "operator {op}"
            );
        }
    }

    #[test]
    fn integer_sums_wrap_rather_than_panic() {
        // Documented on `Summer::add`: wrapping is what keeps the naive, batched, and SIMD
        // paths agreeing bit-for-bit, since SIMD lane arithmetic wraps and cannot panic.
        let csv = format!("k,v\n1,{}\n1,1\n", i64::MAX);
        let result = run(&csv, "SELECT k, SUM(v) FROM t WHERE v > 0 GROUP BY k");
        assert_eq!(
            pairs(&result, "k", "v"),
            vec![("1".to_string(), i64::MIN.to_string())]
        );
    }

    #[test]
    fn reports_unknown_columns_with_the_available_names() {
        let err = error(
            SALES,
            "SELECT region, SUM(nope) FROM s WHERE price > 1 GROUP BY region",
        );
        match err {
            Error::UnknownColumn { column, available } => {
                assert_eq!(column, "nope");
                // Sorted, so the message does not depend on the column map's hash order.
                assert_eq!(available, vec!["amount", "price", "region"]);
            }
            other => panic!("expected UnknownColumn, got {other:?}"),
        }
    }

    #[test]
    fn rejects_column_types_that_cannot_fill_their_role() {
        let cases = [
            // GroupKey holds a dict code or a raw i64; there is no float representation.
            (
                "SELECT price, SUM(amount) FROM s WHERE amount > 1 GROUP BY price",
                "GROUP BY",
            ),
            (
                "SELECT region, SUM(region) FROM s WHERE amount > 1 GROUP BY region",
                "SUM",
            ),
            (
                "SELECT region, SUM(amount) FROM s WHERE region > 1 GROUP BY region",
                "compare",
            ),
            (
                "SELECT region, SUM(amount) FROM s WHERE amount > 'x' GROUP BY region",
                "compare",
            ),
        ];

        for (sql, expected) in cases {
            match error(SALES, sql) {
                Error::TypeError(detail) => assert!(
                    detail.contains(expected),
                    "for {sql:?}\n  expected {expected:?} in: {detail}"
                ),
                other => panic!("for {sql:?} expected TypeError, got {other:?}"),
            }
        }
    }

    #[test]
    fn result_column_names_match_sql_conventions() {
        let result = run(
            SALES,
            "SELECT region, SUM(amount) FROM sales WHERE price > 0 GROUP BY region",
        );
        let mut names: Vec<&str> = result.column_names().collect();
        names.sort();
        assert_eq!(names, vec!["SUM(amount)", "region"]);
    }

    #[test]
    fn an_empty_table_produces_an_empty_result() {
        // Needs an explicit schema: inference over a header-only file has no values to go on
        // and types every column `Utf8` (Phase 1), which `SUM` then rightly rejects. This is
        // the case the explicit-schema loader path exists for.
        let csv = "region,amount\n";
        let schema = Schema::new(vec![
            Field::new("region", DataType::Utf8),
            Field::new("amount", DataType::Int64),
        ]);
        let table = read_csv(csv.as_bytes(), &schema).unwrap();
        let plan =
            parse("SELECT region, SUM(amount) FROM s WHERE amount > 0 GROUP BY region").unwrap();

        let result = naive_query(&table, &plan).unwrap();
        assert_eq!(result.nrows(), 0);
        assert_eq!(result.ncols(), 2);
    }

    #[test]
    fn an_inferred_empty_table_cannot_be_summed() {
        // Consequence of two Phase 1 decisions meeting: a header-only file infers as all
        // Utf8, and Utf8 is not summable. Pinned so the interaction is a documented
        // behaviour rather than a surprise, and so the fix (an explicit schema) stays visible.
        let err = error(
            "region,amount\n",
            "SELECT region, SUM(amount) FROM s WHERE amount > 0 GROUP BY region",
        );
        assert!(
            matches!(&err, Error::TypeError(d) if d.contains("SUM")),
            "got {err:?}"
        );
    }
}
