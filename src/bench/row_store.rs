//! The row-oriented arm: array-of-structs storage, to isolate *layout* from *execution model*.
//!
//! # Why this exists
//! Every other arm reads the same columnar [`Table`], so the naive-vs-batched comparison holds
//! storage constant and isolates the execution model. That is deliberate (`architecture.md`),
//! but it means those numbers cannot support any claim about columnar *storage* — and
//! `prd.md`'s motivation cites both halves: "column-wise storage **and** vectorized execution".
//!
//! This arm supplies the missing half. With four arms the write-up can attribute the
//! row-oriented → columnar-naive gap to layout, and the columnar-naive → batched/SIMD gap to
//! execution model, instead of eliding the two into one number.
//!
//! # What makes it row-oriented, and why that costs
//! A [`Row`] holds the query's three fields inline, so consecutive rows are contiguous and
//! consecutive *values of one field* are strided. Scanning the filter column therefore touches
//! every cache line the other fields occupy. That is the entire effect being measured.
//!
//! Two costs come along with it, and both are genuine properties of row storage rather than
//! handicaps added to make a point:
//! - **No dictionary encoding.** A text field is a `Box<str>` per row, duplicated across every
//!   row that repeats a value. Group-by then hashes and compares *strings* rather than `u32`
//!   codes, and string equality filters compare bytes rather than resolving to a code once.
//!   Dictionary encoding is a columnar technique; a row store has nowhere to put the
//!   dictionary.
//! - **Wider rows.** The struct carries all three fields whether or not a given loop reads
//!   them all.
//!
//! # Fairness
//! No per-field type tags: the store is generic over the three field types and monomorphized,
//! so a row is a packed struct exactly as a real row store would lay it out. Tagging every
//! field with a discriminant would have inflated the row and exaggerated the result.
//!
//! Building the store is **not** part of any measurement. A real row-oriented database would
//! have loaded its data in this layout to begin with; converting from columnar is an artifact
//! of this project's setup, so [`RowStore::build`] runs outside the timed region and only
//! [`RowEngine::run`] is timed.
//!
//! # Scope
//! A benchmark arm and nothing else. It does not implement [`Operator`](crate::exec::Operator),
//! nothing else in the crate depends on it, and it is not an execution strategy anyone can
//! select. It must still agree with the naive baseline on every input, which `tests/parity.rs`
//! enforces.

use std::hash::Hash;

use crate::error::{Error, Result};
use crate::plan::{AggFunc, CompareOp, Literal, LogicalPlan};
use crate::storage::{Column, Table};

/// One row: the three fields the supported query touches, stored inline.
///
/// Field order is group, filter, value — chosen so the two most commonly read together sit
/// adjacent. It makes no difference to the finding, since the row is read whole either way.
#[derive(Debug, Clone)]
pub struct Row<G, F, V> {
    pub group: G,
    pub filter: F,
    pub value: V,
}

/// A field usable as a group key.
///
/// `Hash + Eq` on the field itself, not on a dictionary code — that difference is the point.
pub trait GroupField: Clone + Eq + Hash {
    fn read(column: &Column, row: usize) -> Self;
    /// Materialize the distinct group values back into a result column.
    fn into_column(keys: Vec<Self>) -> Column
    where
        Self: Sized;
}

impl GroupField for i64 {
    fn read(column: &Column, row: usize) -> Self {
        column.as_i64().expect("validated at build time")[row]
    }

    fn into_column(keys: Vec<Self>) -> Column {
        Column::Int64(keys)
    }
}

impl GroupField for Box<str> {
    fn read(column: &Column, row: usize) -> Self {
        // The dictionary is decoded away here, once, at build time. From this point the row
        // store owns an independent copy of the text per row -- which is what a row store is.
        column
            .utf8_value(row)
            .expect("validated at build time")
            .into()
    }

    fn into_column(keys: Vec<Self>) -> Column {
        let dict: Vec<String> = keys.iter().map(|k| k.to_string()).collect();
        Column::Utf8Dict {
            dict: dict.into(),
            codes: (0..keys.len() as u32).collect(),
        }
    }
}

/// A field the filter can compare against a literal.
pub trait FilterField {
    /// The literal pre-converted to whatever this field compares against, resolved once when
    /// the store is built. The other arms resolve their literals once too, so comparing
    /// like-for-like keeps the arms honest.
    type Threshold;

    fn read(column: &Column, row: usize) -> Self;
    fn resolve(literal: &Literal, column_name: &str) -> Result<Self::Threshold>;
    fn passes(&self, op: CompareOp, threshold: &Self::Threshold) -> bool;
}

/// An `Int64` field compared against either an integer or a float literal.
///
/// Mirrors the naive baseline: `WHERE int_col > 2.5` is supported, and widening the value to
/// `f64` is lossy past 2^53.
pub enum IntThreshold {
    Int(i64),
    Float(f64),
}

impl FilterField for i64 {
    type Threshold = IntThreshold;

    fn read(column: &Column, row: usize) -> Self {
        column.as_i64().expect("validated at build time")[row]
    }

    fn resolve(literal: &Literal, column_name: &str) -> Result<Self::Threshold> {
        match literal {
            Literal::Int64(v) => Ok(IntThreshold::Int(*v)),
            Literal::Float64(v) => Ok(IntThreshold::Float(*v)),
            other => Err(type_error(column_name, "Int64", other)),
        }
    }

    fn passes(&self, op: CompareOp, threshold: &Self::Threshold) -> bool {
        match threshold {
            IntThreshold::Int(v) => compare(self, v, op),
            IntThreshold::Float(v) => compare(&(*self as f64), v, op),
        }
    }
}

impl FilterField for f64 {
    type Threshold = f64;

    fn read(column: &Column, row: usize) -> Self {
        column.as_f64().expect("validated at build time")[row]
    }

    fn resolve(literal: &Literal, column_name: &str) -> Result<Self::Threshold> {
        match literal {
            Literal::Float64(v) => Ok(*v),
            Literal::Int64(v) => Ok(*v as f64),
            other => Err(type_error(column_name, "Float64", other)),
        }
    }

    fn passes(&self, op: CompareOp, threshold: &Self::Threshold) -> bool {
        compare(self, threshold, op)
    }
}

impl FilterField for Box<str> {
    type Threshold = String;

    fn read(column: &Column, row: usize) -> Self {
        column
            .utf8_value(row)
            .expect("validated at build time")
            .into()
    }

    fn resolve(literal: &Literal, column_name: &str) -> Result<Self::Threshold> {
        match literal {
            Literal::Utf8(v) => Ok(v.clone()),
            other => Err(type_error(column_name, "Utf8", other)),
        }
    }

    /// Byte comparison per row, for equality as well as ordering.
    ///
    /// The columnar arms resolve an equality literal to a dictionary code once and then
    /// compare integers. Without a dictionary there is nothing to resolve to, so this is the
    /// price of the layout rather than a missing optimization.
    fn passes(&self, op: CompareOp, threshold: &Self::Threshold) -> bool {
        compare(&**self, threshold.as_str(), op)
    }
}

/// A field that can be summed.
pub trait ValueField: Copy {
    fn read(column: &Column, row: usize) -> Self;
    fn zero() -> Self;
    /// Wrapping for integers, matching every other arm so overflow cannot make them disagree.
    fn accumulate(&mut self, other: Self);
    fn into_column(totals: Vec<Self>) -> Column
    where
        Self: Sized;
}

impl ValueField for i64 {
    fn read(column: &Column, row: usize) -> Self {
        column.as_i64().expect("validated at build time")[row]
    }

    fn zero() -> Self {
        0
    }

    fn accumulate(&mut self, other: Self) {
        *self = self.wrapping_add(other);
    }

    fn into_column(totals: Vec<Self>) -> Column {
        Column::Int64(totals)
    }
}

impl ValueField for f64 {
    fn read(column: &Column, row: usize) -> Self {
        column.as_f64().expect("validated at build time")[row]
    }

    fn zero() -> Self {
        0.0
    }

    fn accumulate(&mut self, other: Self) {
        *self += other;
    }

    fn into_column(totals: Vec<Self>) -> Column {
        Column::Float64(totals)
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

fn type_error(column: &str, found: &str, literal: &Literal) -> Error {
    let described = match literal {
        Literal::Int64(v) => format!("the integer {v}"),
        Literal::Float64(v) => format!("the float {v}"),
        Literal::Utf8(v) => format!("the string {v:?}"),
    };
    Error::TypeError(format!(
        "cannot compare column {column:?} ({found}) against {described}"
    ))
}

/// Runs one query over a built row store.
///
/// `Box<dyn>` costs exactly one virtual call per *query*, not per row — the loop inside each
/// concrete `RowStore` is fully monomorphized. That is a different cost class from the
/// per-row dispatch `agents.md` rules out for accumulators, and it saves a twelve-variant
/// enum over every (group, filter, value) type combination.
pub trait RowEngine {
    fn run(&self, plan: &LogicalPlan) -> Result<Table>;
    fn rows(&self) -> usize;
}

pub struct RowStore<G, F, V> {
    rows: Vec<Row<G, F, V>>,
}

impl<G: GroupField, F: FilterField, V: ValueField> RowStore<G, F, V> {
    fn from_columns(table: &Table, group: &str, filter: &str, value: &str) -> Self {
        let group_col = table.column(group).expect("validated at build time");
        let filter_col = table.column(filter).expect("validated at build time");
        let value_col = table.column(value).expect("validated at build time");

        let rows = (0..table.nrows())
            .map(|row| Row {
                group: G::read(group_col, row),
                filter: F::read(filter_col, row),
                value: V::read(value_col, row),
            })
            .collect();

        RowStore { rows }
    }
}

impl<G: GroupField, F: FilterField, V: ValueField> RowEngine for RowStore<G, F, V> {
    fn run(&self, plan: &LogicalPlan) -> Result<Table> {
        let AggFunc::Sum = plan.aggregation.func;

        let threshold = F::resolve(&plan.filter.literal, &plan.filter.column)?;
        let op = plan.filter.op;

        // Same map and the same first-seen slot assignment as the other arms, so group-by
        // structure is not a variable. The keys differ, which is the point.
        let mut slots: hashbrown::HashMap<G, usize> = hashbrown::HashMap::new();
        let mut keys: Vec<G> = Vec::new();
        let mut totals: Vec<V> = Vec::new();

        for row in &self.rows {
            if !row.filter.passes(op, &threshold) {
                continue;
            }

            let slot = match slots.get(&row.group) {
                Some(slot) => *slot,
                None => {
                    let slot = keys.len();
                    slots.insert(row.group.clone(), slot);
                    keys.push(row.group.clone());
                    totals.push(V::zero());
                    slot
                }
            };

            totals[slot].accumulate(row.value);
        }

        let nrows = keys.len();
        let mut columns = std::collections::HashMap::with_capacity(2);
        columns.insert(plan.group_by.clone(), G::into_column(keys));
        columns.insert(plan.aggregation.output_name(), V::into_column(totals));

        Table::new(columns, nrows)
    }

    fn rows(&self) -> usize {
        self.rows.len()
    }
}

/// Convert a columnar table into the row-oriented layout for one query.
///
/// Deliberately separate from [`RowEngine::run`] so the conversion never lands inside a timed
/// region — see the module docs on fairness.
pub fn build(table: &Table, plan: &LogicalPlan) -> Result<Box<dyn RowEngine>> {
    let group = &plan.group_by;
    let filter = &plan.filter.column;
    let value = &plan.aggregation.input;

    // Same validation the other arms perform, kept separate for the same reason: `agents.md`
    // forbids the baselines growing shared code with the batch engine, and the parity suite
    // asserts all of them reject invalid plans identically.
    let group_col = column(table, group)?;
    let filter_col = column(table, filter)?;
    let value_col = column(table, value)?;

    if matches!(group_col, Column::Float64(_)) {
        return Err(Error::TypeError(format!(
            "cannot GROUP BY column {group:?}: it is Float64, and group keys must be \
             Int64 or Utf8"
        )));
    }
    if matches!(value_col, Column::Utf8Dict { .. }) {
        return Err(Error::TypeError(format!(
            "cannot SUM column {value:?}: it is Utf8"
        )));
    }

    // The literal must be resolvable against the filter column's type, checked here so
    // `run` cannot fail for a reason the other arms would have caught at build time.
    match filter_col {
        Column::Int64(_) => {
            <i64 as FilterField>::resolve(&plan.filter.literal, filter)?;
        }
        Column::Float64(_) => {
            <f64 as FilterField>::resolve(&plan.filter.literal, filter)?;
        }
        Column::Utf8Dict { .. } => {
            <Box<str> as FilterField>::resolve(&plan.filter.literal, filter)?;
        }
    }

    /// One arm of the (group, filter, value) type product.
    macro_rules! store {
        ($g:ty, $f:ty, $v:ty) => {
            Box::new(RowStore::<$g, $f, $v>::from_columns(
                table, group, filter, value,
            )) as Box<dyn RowEngine>
        };
    }

    Ok(match (group_col, filter_col, value_col) {
        (Column::Int64(_), Column::Int64(_), Column::Int64(_)) => store!(i64, i64, i64),
        (Column::Int64(_), Column::Int64(_), Column::Float64(_)) => store!(i64, i64, f64),
        (Column::Int64(_), Column::Float64(_), Column::Int64(_)) => store!(i64, f64, i64),
        (Column::Int64(_), Column::Float64(_), Column::Float64(_)) => store!(i64, f64, f64),
        (Column::Int64(_), Column::Utf8Dict { .. }, Column::Int64(_)) => store!(i64, Box<str>, i64),
        (Column::Int64(_), Column::Utf8Dict { .. }, Column::Float64(_)) => {
            store!(i64, Box<str>, f64)
        }
        (Column::Utf8Dict { .. }, Column::Int64(_), Column::Int64(_)) => {
            store!(Box<str>, i64, i64)
        }
        (Column::Utf8Dict { .. }, Column::Int64(_), Column::Float64(_)) => {
            store!(Box<str>, i64, f64)
        }
        (Column::Utf8Dict { .. }, Column::Float64(_), Column::Int64(_)) => {
            store!(Box<str>, f64, i64)
        }
        (Column::Utf8Dict { .. }, Column::Float64(_), Column::Float64(_)) => {
            store!(Box<str>, f64, f64)
        }
        (Column::Utf8Dict { .. }, Column::Utf8Dict { .. }, Column::Int64(_)) => {
            store!(Box<str>, Box<str>, i64)
        }
        (Column::Utf8Dict { .. }, Column::Utf8Dict { .. }, Column::Float64(_)) => {
            store!(Box<str>, Box<str>, f64)
        }
        // The group-is-Float64 and value-is-Utf8 cases were rejected above.
        _ => unreachable!("group and value column types are validated above"),
    })
}

/// Build and run in one step. Convenient for parity tests; **never** use it to time a query,
/// since it puts the layout conversion inside the measurement.
pub fn row_query(table: &Table, plan: &LogicalPlan) -> Result<Table> {
    build(table, plan)?.run(plan)
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;
    use crate::storage::{infer_schema, read_csv};

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

    fn pairs(result: &Table, group: &str, agg: &str) -> Vec<(String, String)> {
        let group_col = result.column(group).expect("group column");
        let agg_col = result.column(agg).expect("agg column");

        let mut rows: Vec<(String, String)> = (0..result.nrows())
            .map(|row| {
                let key = match group_col {
                    Column::Int64(v) => v[row].to_string(),
                    Column::Utf8Dict { .. } => group_col.utf8_value(row).unwrap().to_string(),
                    Column::Float64(_) => unreachable!(),
                };
                let sum = match agg_col {
                    Column::Int64(v) => v[row].to_string(),
                    Column::Float64(v) => v[row].to_string(),
                    Column::Utf8Dict { .. } => unreachable!(),
                };
                (key, sum)
            })
            .collect();
        rows.sort();
        rows
    }

    fn run(csv: &str, sql: &str) -> Table {
        let plan = parse(sql).unwrap();
        row_query(&table(csv), &plan).unwrap()
    }

    fn owned(rows: Vec<(&str, &str)>) -> Vec<(String, String)> {
        rows.into_iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn groups_by_text_and_sums_integers() {
        let result = run(
            SALES,
            "SELECT region, SUM(amount) FROM t WHERE price > 2 GROUP BY region",
        );
        assert_eq!(
            pairs(&result, "region", "SUM(amount)"),
            owned(vec![("east", "40"), ("north", "80"), ("south", "-20")])
        );
    }

    #[test]
    fn groups_by_integer_and_sums_floats() {
        let csv = "k,v\n-1,1.5\n2,2.5\n-1,3.0\n";
        let result = run(csv, "SELECT k, SUM(v) FROM t WHERE v > 0 GROUP BY k");
        assert_eq!(
            pairs(&result, "k", "SUM(v)"),
            owned(vec![("-1", "4.5"), ("2", "2.5")])
        );
    }

    #[test]
    fn string_equality_compares_bytes_not_codes() {
        // No dictionary to resolve against -- the cost this arm exists to expose.
        let result = run(
            SALES,
            "SELECT region, SUM(amount) FROM t WHERE region = 'north' GROUP BY region",
        );
        assert_eq!(
            pairs(&result, "region", "SUM(amount)"),
            owned(vec![("north", "90")])
        );
    }

    #[test]
    fn string_ordering_works_without_a_dictionary() {
        let result = run(
            SALES,
            "SELECT region, SUM(amount) FROM t WHERE region > 'north' GROUP BY region",
        );
        assert_eq!(
            pairs(&result, "region", "SUM(amount)"),
            owned(vec![("south", "-20")])
        );
    }

    #[test]
    fn mixed_numeric_comparison_widens_like_the_other_arms() {
        let result = run(
            SALES,
            "SELECT region, SUM(amount) FROM t WHERE amount > 25.5 GROUP BY region",
        );
        assert_eq!(
            pairs(&result, "region", "SUM(amount)"),
            owned(vec![("east", "40"), ("north", "80")])
        );
    }

    #[test]
    fn integer_sums_wrap_like_the_other_arms() {
        let csv = format!("k,v\n1,{}\n1,1\n", i64::MAX);
        let result = run(&csv, "SELECT k, SUM(v) FROM t WHERE v > 0 GROUP BY k");
        assert_eq!(
            pairs(&result, "k", "SUM(v)"),
            vec![("1".to_string(), i64::MIN.to_string())]
        );
    }

    #[test]
    fn a_filter_matching_nothing_yields_an_empty_result() {
        let result = run(
            SALES,
            "SELECT region, SUM(amount) FROM t WHERE amount > 10000 GROUP BY region",
        );
        assert_eq!(result.nrows(), 0);
        assert_eq!(result.ncols(), 2);
    }

    #[test]
    fn building_is_separate_from_running() {
        // The property the benchmark depends on: conversion happens once, queries reuse it.
        let table = table(SALES);
        let plan =
            parse("SELECT region, SUM(amount) FROM t WHERE price > 2 GROUP BY region").unwrap();

        let store = build(&table, &plan).unwrap();
        assert_eq!(store.rows(), 5, "every source row is materialized");

        let first = store.run(&plan).unwrap();
        let second = store.run(&plan).unwrap();
        assert_eq!(first, second, "running twice must not consume the store");
    }

    #[test]
    fn rejects_the_same_plans_the_other_arms_reject() {
        let table = table(SALES);
        for sql in [
            "SELECT region, SUM(missing) FROM t WHERE amount > 1 GROUP BY region",
            "SELECT price, SUM(amount) FROM t WHERE amount > 1 GROUP BY price",
            "SELECT region, SUM(region) FROM t WHERE amount > 1 GROUP BY region",
            "SELECT region, SUM(amount) FROM t WHERE region > 1 GROUP BY region",
            "SELECT region, SUM(amount) FROM t WHERE amount > 'x' GROUP BY region",
        ] {
            let plan = parse(sql).unwrap();
            assert!(build(&table, &plan).is_err(), "accepted {sql:?}");
        }
    }
}
