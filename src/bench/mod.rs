//! Benchmark support: the two baselines the batch engine is measured against.
//!
//! - [`naive_query`] — row-at-a-time over *columnar* storage, which isolates the execution
//!   model by holding layout constant.
//! - [`row_query`] — row-at-a-time over *row-oriented* storage, which supplies the other half:
//!   the gap between the two is attributable to layout alone.
//!
//! Library code, deliberately not a `criterion` harness. The baseline has to be callable
//! from correctness tests -- it is the oracle every later execution strategy must agree with
//! -- and cargo only lets `benches/` targets be benchmarks, not dependencies. The criterion
//! harnesses that time this code live in `benches/` at the crate root (Phase 6).

pub mod data;
pub mod harness;
mod naive;
mod row_store;

pub use naive::naive_query;
pub use row_store::{RowEngine, RowStore, build as build_row_store, row_query};
