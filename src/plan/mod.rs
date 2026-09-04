//! Query plans: the logical shape, and the operator tree built from it.
//!
//! - [`logical`] is pure data -- what the query asks for, with SQL syntax stripped away.
//! - [`physical`] turns that into a runnable `Scan -> Filter -> Project -> Aggregate` tree,
//!   validating it against a real table's schema on the way (`architecture.md`).

mod logical;
mod physical;

pub use logical::{AggFunc, Aggregation, CompareOp, Literal, LogicalPlan, Predicate};
pub use physical::{BatchPipeline, batch_query, build};
