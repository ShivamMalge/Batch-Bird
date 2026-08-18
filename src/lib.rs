//! Batchbird — a mini columnar query engine built to demonstrate *why* vectorized,
//! column-at-a-time execution beats row-at-a-time processing, with benchmarks to prove it.
//!
//! Design docs are the source of truth — read them before adding code:
//! `prd.md` (scope), `architecture.md` (shapes), `systemDesign.md` (rationale),
//! `phases.md` (build order), `agents.md` (guardrails).
//!
//! **Current state: Phase 0 (setup).** No engine code yet; the storage, parser, plan, and
//! exec modules land in Phases 1-4.

// `std::simd` is still nightly-only (2026-08). Applying the feature attribute conditionally
// is what lets the default build, and every correctness test, stay on stable.
// See Cargo.toml [features] and techstack.md.
#![cfg_attr(feature = "simd", feature(portable_simd))]

#[cfg(feature = "simd")]
pub mod simd;
