//! `LogicalPlan` -> operator tree.
//!
//! This is where the plan meets real data for the first time, so it is where every check that
//! needs the table's schema happens: that each named column exists, that the group column can
//! be a key, that the aggregated column is numeric, and that the filter literal can be
//! compared to its column. Getting all of that out of the way here is what lets
//! [`Operator::next_batch`] be infallible (`exec` module docs).
//!
//! # One enum switch, then no dispatch
//! `Aggregate` is monomorphized per value type, so summing an `Int64` column and a `Float64`
//! column are two different concrete types. [`BatchPipeline`] is the single place that
//! difference is resolved -- one match, once per query, after which every per-row call is a
//! direct, inlined one. That is what "monomorphized, not `Box<dyn Accumulator>`"
//! (`techstack.md`) costs in practice: one enum at the top instead of a virtual call per row.

use crate::error::{Error, Result};
use crate::exec::{
    Aggregate, Filter, FilterKind, GroupKind, Operator, Project, Scan, SortAggregate,
    SumAccumulator,
};
use crate::plan::logical::{CompareOp, Literal, LogicalPlan};
use crate::storage::{Column, Table};

/// The fixed shape of every pipeline this engine builds, below the aggregate.
type Source<'a> = Project<Filter<Scan<'a>>>;

/// Which group-by strategy to build.
///
/// Both are real, selectable strategies rather than one replacing the other -- the benchmark
/// needs to drive either over identical input (`systemDesign.md` "Sort-Based Grouping").
/// [`GroupStrategy::Hash`] is the default everywhere except where a benchmark asks otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GroupStrategy {
    /// Hash map from key to accumulator slot, then a data-dependent scatter. O(n), O(groups)
    /// memory, unvectorizable second phase.
    #[default]
    Hash,
    /// Sort by key, then reduce contiguous runs. O(n log n), O(rows) memory, and the only path
    /// on which the SIMD sum kernel reaches a real query.
    Sort,
}

/// A built, validated pipeline, specialized to the aggregated column's type and the group-by
/// strategy.
///
/// Four variants for two binary choices. Both are resolved **once per query**, here, after
/// which every per-row call is direct and inlined.
pub enum BatchPipeline<'a> {
    HashInt64(Aggregate<Source<'a>, i64, SumAccumulator<i64>>),
    HashFloat64(Aggregate<Source<'a>, f64, SumAccumulator<f64>>),
    SortInt64(SortAggregate<Source<'a>, i64>),
    SortFloat64(SortAggregate<Source<'a>, f64>),
}

impl BatchPipeline<'_> {
    /// Drain the pipeline and materialize the result.
    pub fn execute(mut self) -> Result<Table> {
        let batch = match &mut self {
            BatchPipeline::HashInt64(agg) => agg.next_batch(),
            BatchPipeline::HashFloat64(agg) => agg.next_batch(),
            BatchPipeline::SortInt64(agg) => agg.next_batch(),
            BatchPipeline::SortFloat64(agg) => agg.next_batch(),
        };

        batch
            .expect("aggregate always produces exactly one result batch")
            .into_table()
    }
}

/// Build the operator tree for `plan` over `table`, using the default hash group-by.
pub fn build<'a>(table: &'a Table, plan: &LogicalPlan) -> Result<BatchPipeline<'a>> {
    build_with(table, plan, GroupStrategy::default())
}

/// Build the operator tree with an explicit group-by strategy.
pub fn build_with<'a>(
    table: &'a Table,
    plan: &LogicalPlan,
    strategy: GroupStrategy,
) -> Result<BatchPipeline<'a>> {
    let group_kind = group_kind(table, &plan.group_by)?;
    let filter_kind = filter_kind(table, plan)?;

    // Scan reads only what the query touches. The filter column is included because `Filter`
    // needs it, then dropped by `Project` before `Aggregate` ever sees it.
    let mut scan_columns = vec![plan.group_by.clone(), plan.aggregation.input.clone()];
    if !scan_columns.contains(&plan.filter.column) {
        scan_columns.push(plan.filter.column.clone());
    }

    let project_columns = if plan.group_by == plan.aggregation.input {
        // `SELECT k, SUM(k) ... GROUP BY k` is legal; asking for the same column twice would
        // make `Project` try to move it out of the batch a second time.
        vec![plan.group_by.clone()]
    } else {
        vec![plan.group_by.clone(), plan.aggregation.input.clone()]
    };

    let scan = Scan::new(table, scan_columns.clone());
    // Compaction visits columns in this order rather than the map's, so heap layout does not
    // depend on a hashbrown implementation detail (`Filter::column_order`).
    let filter = Filter::new(scan, plan.filter.column.clone(), filter_kind, scan_columns);
    let source = Project::new(filter, project_columns);

    let group_columns = vec![plan.group_by.clone()];
    let value_column = plan.aggregation.input.clone();
    let output_column = plan.aggregation.output_name();

    // The one dispatch point: pick the monomorphized aggregate for this column's type and
    // the requested strategy.
    Ok(match (strategy, value_column_kind(table, &value_column)?) {
        (GroupStrategy::Hash, ValueKind::Int64) => BatchPipeline::HashInt64(Aggregate::new(
            source,
            group_columns,
            group_kind,
            value_column,
            output_column,
        )),
        (GroupStrategy::Hash, ValueKind::Float64) => BatchPipeline::HashFloat64(Aggregate::new(
            source,
            group_columns,
            group_kind,
            value_column,
            output_column,
        )),
        (GroupStrategy::Sort, ValueKind::Int64) => BatchPipeline::SortInt64(SortAggregate::new(
            source,
            group_columns,
            group_kind,
            value_column,
            output_column,
        )),
        (GroupStrategy::Sort, ValueKind::Float64) => {
            BatchPipeline::SortFloat64(SortAggregate::new(
                source,
                group_columns,
                group_kind,
                value_column,
                output_column,
            ))
        }
    })
}

/// Build and run in one step, mirroring [`naive_query`](crate::bench::naive_query) so the
/// engines are called the same way.
pub fn batch_query(table: &Table, plan: &LogicalPlan) -> Result<Table> {
    build(table, plan)?.execute()
}

/// Build and run with an explicit group-by strategy.
pub fn batch_query_with(
    table: &Table,
    plan: &LogicalPlan,
    strategy: GroupStrategy,
) -> Result<Table> {
    build_with(table, plan, strategy)?.execute()
}

enum ValueKind {
    Int64,
    Float64,
}

/// Look up a column, naming the alternatives when it is missing.
///
/// Duplicated from the naive baseline rather than shared. `agents.md` forbids the baseline
/// growing shared code with the batch engine, and a bright line is worth more than the twenty
/// lines it costs -- the parity test catches any drift in behaviour immediately.
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

fn group_kind(table: &Table, name: &str) -> Result<GroupKind> {
    match column(table, name)? {
        Column::Int64(_) => Ok(GroupKind::Int64),
        Column::Utf8Dict { .. } => Ok(GroupKind::Utf8),
        Column::Float64(_) => Err(Error::TypeError(format!(
            "cannot GROUP BY column {name:?}: it is Float64, and group keys must be \
             Int64 or Utf8"
        ))),
    }
}

fn value_column_kind(table: &Table, name: &str) -> Result<ValueKind> {
    match column(table, name)? {
        Column::Int64(_) => Ok(ValueKind::Int64),
        Column::Float64(_) => Ok(ValueKind::Float64),
        Column::Utf8Dict { .. } => Err(Error::TypeError(format!(
            "cannot SUM column {name:?}: it is Utf8"
        ))),
    }
}

/// Resolve the filter to a typed comparison, doing every lookup that can be done once.
fn filter_kind(table: &Table, plan: &LogicalPlan) -> Result<FilterKind> {
    let name = &plan.filter.column;
    let op = plan.filter.op;

    match (column(table, name)?, &plan.filter.literal) {
        (Column::Int64(_), Literal::Int64(x)) => Ok(FilterKind::IntVsInt(op, *x)),
        (Column::Int64(_), Literal::Float64(x)) => Ok(FilterKind::IntVsFloat(op, *x)),
        (Column::Float64(_), Literal::Float64(x)) => Ok(FilterKind::FloatVsFloat(op, *x)),
        (Column::Float64(_), Literal::Int64(x)) => Ok(FilterKind::FloatVsFloat(op, *x as f64)),
        (Column::Utf8Dict { dict, .. }, Literal::Utf8(literal)) => match op {
            // Resolve the literal to a dictionary code exactly once, here, rather than
            // re-scanning the dictionary for every batch. Safe because `materialize` copies
            // dictionaries verbatim and never renumbers codes.
            CompareOp::Eq => Ok(FilterKind::Utf8Eq(
                dict.iter().position(|d| d == literal).map(|i| i as u32),
            )),
            _ => Ok(FilterKind::Utf8Ord(op, literal.clone())),
        },
        (column, literal) => Err(Error::TypeError(format!(
            "cannot compare column {name:?} ({}) against {}",
            column.data_type(),
            describe(literal)
        ))),
    }
}

fn describe(literal: &Literal) -> String {
    match literal {
        Literal::Int64(v) => format!("the integer {v}"),
        Literal::Float64(v) => format!("the float {v}"),
        Literal::Utf8(v) => format!("the string {v:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;
    use crate::storage::{infer_schema, read_csv};

    const ROWS: &str = concat!(
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

    fn error(sql: &str) -> Error {
        let plan = parse(sql).unwrap();
        build(&table(ROWS), &plan)
            .err()
            .expect("expected build to fail")
    }

    #[test]
    fn resolves_a_string_equality_literal_to_a_code_once() {
        let plan =
            parse("SELECT region, SUM(amount) FROM t WHERE region = 'south' GROUP BY region")
                .unwrap();
        // "north" is code 0, "south" code 1 in first-seen order.
        assert_eq!(
            filter_kind(&table(ROWS), &plan).unwrap(),
            FilterKind::Utf8Eq(Some(1))
        );
    }

    #[test]
    fn a_literal_outside_the_dictionary_resolves_to_no_code() {
        let plan = parse("SELECT region, SUM(amount) FROM t WHERE region = 'west' GROUP BY region")
            .unwrap();
        assert_eq!(
            filter_kind(&table(ROWS), &plan).unwrap(),
            FilterKind::Utf8Eq(None)
        );
    }

    #[test]
    fn string_ordering_keeps_the_literal_for_per_row_comparison() {
        let plan =
            parse("SELECT region, SUM(amount) FROM t WHERE region > 'north' GROUP BY region")
                .unwrap();
        assert_eq!(
            filter_kind(&table(ROWS), &plan).unwrap(),
            FilterKind::Utf8Ord(CompareOp::Gt, "north".to_string())
        );
    }

    #[test]
    fn widens_mixed_numeric_comparisons() {
        let table = table(ROWS);

        let int_col =
            parse("SELECT region, SUM(amount) FROM t WHERE amount > 2.5 GROUP BY region").unwrap();
        assert_eq!(
            filter_kind(&table, &int_col).unwrap(),
            FilterKind::IntVsFloat(CompareOp::Gt, 2.5)
        );

        let float_col =
            parse("SELECT region, SUM(amount) FROM t WHERE price > 2 GROUP BY region").unwrap();
        assert_eq!(
            filter_kind(&table, &float_col).unwrap(),
            FilterKind::FloatVsFloat(CompareOp::Gt, 2.0)
        );
    }

    #[test]
    fn reports_unknown_columns_with_the_available_names() {
        match error("SELECT region, SUM(nope) FROM t WHERE price > 1 GROUP BY region") {
            Error::UnknownColumn { column, available } => {
                assert_eq!(column, "nope");
                assert_eq!(available, vec!["amount", "price", "region"]);
            }
            other => panic!("expected UnknownColumn, got {other:?}"),
        }
    }

    #[test]
    fn rejects_column_types_that_cannot_fill_their_role() {
        let cases = [
            (
                "SELECT price, SUM(amount) FROM t WHERE amount > 1 GROUP BY price",
                "GROUP BY",
            ),
            (
                "SELECT region, SUM(region) FROM t WHERE amount > 1 GROUP BY region",
                "SUM",
            ),
            (
                "SELECT region, SUM(amount) FROM t WHERE region > 1 GROUP BY region",
                "compare",
            ),
            (
                "SELECT region, SUM(amount) FROM t WHERE amount > 'x' GROUP BY region",
                "compare",
            ),
        ];

        for (sql, expected) in cases {
            match error(sql) {
                Error::TypeError(detail) => assert!(
                    detail.contains(expected),
                    "for {sql:?}\n  expected {expected:?} in: {detail}"
                ),
                other => panic!("for {sql:?} expected TypeError, got {other:?}"),
            }
        }
    }

    #[test]
    fn grouping_and_summing_the_same_column_works() {
        // `Project` must not be asked to move the same column out of a batch twice.
        let table = table("k,v\n2,9\n3,9\n2,9\n");
        let plan = parse("SELECT k, SUM(k) FROM t WHERE v > 0 GROUP BY k").unwrap();
        let result = batch_query(&table, &plan).unwrap();

        assert_eq!(result.nrows(), 2);
        assert_eq!(result.ncols(), 2);
        let sums = result.column("SUM(k)").unwrap().as_i64().unwrap();
        assert_eq!(sums.iter().sum::<i64>(), 2 + 2 + 3);
    }
}
