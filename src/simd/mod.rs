//! SIMD kernels — feature-gated, nightly-only (`cargo +nightly ... --features simd`).
//!
//! # Phase 0 scope
//! This module currently holds **only a toolchain smoke test**. The real kernels — filter
//! comparison and `SUM` reduction over `Int64`/`Float64` — are Phase 5 work (`phases.md`).
//! The smoke test exists because "confirm the toolchain supports `std::simd`" was an open
//! question in the design docs; failing that check now is far cheaper than failing it in
//! Phase 5 with the whole engine already built on the assumption.
//!
//! # The invariant this module establishes
//! Every SIMD kernel has a scalar twin that produces identical results and is covered by the
//! same tests (`agents.md` "Testing Expectations"). SIMD is an optimization path, never the
//! only implementation. From Phase 5 on, the scalar twins live ungated in `exec/` and only
//! the vectorized variants are gated here; the smoke test keeps both side by side because
//! there is no `exec/` yet.

use std::simd::Simd;
use std::simd::num::SimdInt;

/// Lane count for the smoke test. Phase 5 will pick lane widths per kernel and per element
/// type; 8 is just a width every target we care about supports.
const LANES: usize = 8;

/// Scalar reference sum. Deliberately the dumbest correct implementation.
pub fn sum_i64_scalar(values: &[i64]) -> i64 {
    values.iter().sum()
}

/// Vectorized sum — a *smoke test*, not the Phase 5 kernel.
///
/// Proves three things compile and run correctly on this toolchain: `Simd` construction from
/// a slice, lane-wise arithmetic, and a horizontal reduction. Uses `wrapping_add` because
/// SIMD lane arithmetic wraps rather than panicking on overflow the way debug-mode scalar
/// `+` does — keeping the two twins bit-identical on overflow is a real Phase 5 concern, and
/// noting it here is cheaper than rediscovering it later.
pub fn sum_i64_simd(values: &[i64]) -> i64 {
    let (chunks, tail) = values.as_chunks::<LANES>();

    let mut acc = Simd::<i64, LANES>::splat(0);
    for chunk in chunks {
        acc += Simd::from_array(*chunk);
    }

    // Horizontal reduce, then fold in the remainder the scalar way. Every real kernel has
    // this same shape: a vectorized body plus a scalar tail for the ragged end.
    acc.reduce_sum()
        .wrapping_add(tail.iter().fold(0i64, |a, b| a.wrapping_add(*b)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The twins must agree — including on inputs shorter than one vector, and on inputs
    /// whose length is not a multiple of `LANES` (exercising the scalar tail).
    #[test]
    fn simd_sum_matches_scalar() {
        for len in [0, 1, 7, 8, 9, 63, 64, 1000] {
            let values: Vec<i64> = (0..len as i64).map(|i| i * 3 - 7).collect();
            assert_eq!(
                sum_i64_simd(&values),
                sum_i64_scalar(&values),
                "twins disagreed at len {len}"
            );
        }
    }
}
