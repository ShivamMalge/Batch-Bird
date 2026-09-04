//! Batchbird — a mini columnar query engine built to demonstrate *why* vectorized,
//! column-at-a-time execution beats row-at-a-time processing, with benchmarks to prove it.
//!
//! Design docs are the source of truth — read them before adding code:
//! `prd.md` (scope), `architecture.md` (shapes), `systemDesign.md` (rationale),
//! `phases.md` (build order), `agents.md` (guardrails).
//!
//! **Current state: Phase 4 (batch execution pipeline) complete.** SIMD lands in Phase 5.

// `std::simd` is still nightly-only (2026-08). Applying the feature attribute conditionally
// is what lets the default build, and every correctness test, stay on stable.
// See Cargo.toml [features] and techstack.md.
#![cfg_attr(feature = "simd", feature(portable_simd))]

pub mod bench;
pub mod error;
pub mod exec;
pub mod parser;
pub mod plan;
pub mod storage;

#[cfg(feature = "simd")]
pub mod simd;
