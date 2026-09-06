//! Batch execution: `Scan -> Filter -> Project -> Aggregate`, pulled one batch at a time.
//!
//! # The pull model
//! Every operator implements [`Operator`] and holds its input. Execution is driven from the
//! top: `Aggregate` asks `Project` for a batch, which asks `Filter`, which asks `Scan`. No
//! operator ever pushes, and no operator sees more than one batch at a time, so peak memory
//! is a batch rather than a table.
//!
//! # Why `next_batch` cannot fail
//! `architecture.md` pins the signature as returning `Option<RecordBatch>`, with no error
//! channel -- and that is a design statement, not an omission. Everything that *can* go wrong
//! (a missing column, a type that cannot fill its role, a literal that cannot be compared to
//! its column) is caught when the pipeline is **built**, by
//! [`plan::build`](crate::plan::build). By the time batches flow, the pipeline is known-valid,
//! so the hot loop carries no error handling at all. `None` means exhausted, never "failed".
//!
//! # Generic over the input, not boxed
//! Each operator is generic over its input type, so a built pipeline is one concrete type and
//! every `next_batch` call inlines. `Box<dyn Operator>` would cost only one virtual call per
//! 1024 rows -- genuinely negligible -- but generics cost nothing at all here, and it keeps
//! the whole pipeline free of dispatch so no one can attribute a benchmark result to it.

pub mod aggregate;
pub mod batch;
pub mod bitset;
pub mod filter;
pub mod kernels;
pub mod materialize;
pub mod project;
pub mod scan;

pub use aggregate::{Accumulator, Aggregate, GroupKey, GroupKind, SumAccumulator, Summable};
pub use batch::RecordBatch;
pub use bitset::Bitset;
pub use filter::{Filter, FilterKind};
pub use project::Project;
pub use scan::Scan;

/// Rows per batch.
///
/// Big enough to amortize per-batch overhead over many rows, small enough that a batch's
/// working set stays in cache during a vectorized sweep (`systemDesign.md`). Real engines sit
/// in the same range -- DuckDB uses 2048-row vectors, DataFusion defaults to 8192 -- and
/// [`Scan::with_batch_size`] exists so the choice can be swept rather than asserted.
pub const BATCH_SIZE: usize = 1024;

/// One stage of the execution pipeline.
///
/// `None` means the input is exhausted. An operator may return batches of any length, and
/// `Filter` in particular returns batches smaller than its input's.
pub trait Operator {
    fn next_batch(&mut self) -> Option<RecordBatch>;
}
