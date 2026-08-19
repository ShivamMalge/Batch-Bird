//! The internal logical plan: what the engine will execute, stripped of SQL syntax.
//!
//! This is the boundary type between `parser/` (SQL text -> `LogicalPlan`) and, in Phase 4,
//! the operator-tree builder that turns a `LogicalPlan` into `Scan -> Filter -> Project ->
//! Aggregate` (`architecture.md`).
//!
//! # Why it has no tree shape
//! A real engine's logical plan is a tree of relational nodes, because a real engine accepts
//! arbitrarily nested queries. Batchbird accepts exactly one query shape, so a plan is a
//! fixed record with named parts rather than a tree. That is not a shortcut around a hard
//! problem -- it is the shape the scope cut *implies*, and inventing a general plan tree here
//! would be inventing a general planner nobody asked for (`agents.md`: scope cuts are
//! decisions, not gaps to fill). Every field below is non-optional because every clause in
//! the supported shape is mandatory.

/// A parsed, validated query.
///
/// ```sql
/// SELECT col1, SUM(col2) FROM t WHERE col3 <op> x GROUP BY col1
/// --     ^^^^  ^^^^^^^^^      ^        ^^^^^^^^^^          ^^^^
/// --     |     aggregation    table    filter              group_by (== the SELECT column)
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct LogicalPlan {
    /// `FROM t`.
    pub table: String,
    /// `GROUP BY col1`, which the supported shape also requires as the first SELECT item.
    /// Single column only -- `GroupKey` is a single-value newtype (`agents.md` guardrail).
    pub group_by: String,
    /// `SUM(col2)`.
    pub aggregation: Aggregation,
    /// `WHERE col3 <op> x`. Not an `Option`: the supported shape always has a WHERE.
    pub filter: Predicate,
}

/// An aggregate call over one column.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Aggregation {
    pub func: AggFunc,
    /// The aggregated column (`col2`).
    pub input: String,
}

/// Aggregate functions the engine knows.
///
/// One variant today. It exists for the same reason `systemDesign.md` keeps the
/// `Accumulator` trait around with a single impl: it makes the extension point visible, and
/// lets the parser reject `AVG`/`COUNT`/`MIN`/`MAX` by name instead of by falling off the
/// end of a match.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggFunc {
    Sum,
}

/// `WHERE column <op> literal`.
///
/// Deliberately not an expression tree: `AND`/`OR`/`NOT` are out of scope, so a predicate is
/// exactly one comparison. Widening this later is where a real expression evaluator would
/// come in, and that is a separate project.
#[derive(Debug, Clone, PartialEq)]
pub struct Predicate {
    pub column: String,
    pub op: CompareOp,
    pub literal: Literal,
}

/// The three comparison operators in scope.
///
/// No `>=`, `<=`, or `!=`. They are trivial to add to a scalar loop, but each one is another
/// SIMD comparison kernel to write and benchmark in Phase 5, and they demonstrate nothing the
/// existing three do not (`prd.md` Supported Query Shape).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareOp {
    Eq,
    Lt,
    Gt,
}

/// A literal on the right-hand side of the filter.
///
/// Mirrors `storage::DataType` minus any null or boolean: `prd.md` rules out a `Bool` column
/// type, and Phase 1 established that the engine has no NULL representation. Both are
/// rejected at parse time rather than becoming a runtime surprise.
#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Int64(i64),
    Float64(f64),
    Utf8(String),
}
