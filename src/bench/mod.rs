//! Benchmark support: the row-at-a-time baseline the batch engine is measured against.
//!
//! Library code, deliberately not a `criterion` harness. The baseline has to be callable
//! from correctness tests -- it is the oracle every later execution strategy must agree with
//! -- and cargo only lets `benches/` targets be benchmarks, not dependencies. The criterion
//! harnesses that time this code live in `benches/` at the crate root (Phase 6).

mod naive;

pub use naive::naive_query;
