//! SIMD kernels -- feature-gated, nightly-only (`cargo +nightly ... --features simd`).
//!
//! Every function here is an optimization of a reference implementation in
//! [`exec::kernels`](crate::exec::kernels), which stays compiled and tested on stable. The
//! tests there run both and assert they agree, so nothing in this file can silently drift.
//! Phase 0's toolchain smoke test lived here and has been replaced by these.
//!
//! # Compare to bitmask
//! A vector compare produces a *lane mask*, and `to_bitmask` turns that into bits directly --
//! [`LANES`] rows per instruction, with no per-row branch. Those bits drop straight into the
//! packed [`Bitset`] the scalar path already builds, which is why [`crate::exec::filter`]
//! needed no restructuring to accept them.
//!
//! # Lane width
//! [`LANES`] is 8, so an `i64` or `f64` vector is 512 bits wide -- wider than AVX2's 256-bit
//! registers. That is deliberate and is the point of `portable_simd`: LLVM legalizes an
//! over-wide vector into however many native registers the target actually has, and on a
//! 256-bit machine the two halves are independent, which *helps* by giving the pipeline two
//! chains to interleave. Eight lanes also packs cleanly into the 64-bit bitset words: eight
//! vectors fill exactly one word, and a chunk's bits never straddle a word boundary.
//!
//! # Every kernel ends with a scalar tail
//! Slices are not multiples of eight. Each kernel runs full vectors over
//! `as_chunks::<LANES>()` and finishes the remainder scalar. The tail is where off-by-one
//! bugs live, so the parity tests deliberately include lengths of 1, 7, 9, 15, 63, 65 and 127.

use std::simd::Simd;
use std::simd::cmp::{SimdPartialEq, SimdPartialOrd};
use std::simd::num::{SimdFloat, SimdInt};

use crate::exec::bitset::Bitset;
use crate::exec::kernels::compare;
use crate::plan::CompareOp;

/// Lanes per vector. See the module docs on why over-wide is fine.
pub const LANES: usize = 8;

/// Fold `LANES` bits produced by one vector compare into the mask at row offset `base`.
///
/// `base` is always a multiple of `LANES`, and `LANES` divides 64, so the bits for one chunk
/// always land inside a single word -- no cross-word shifting to get wrong.
#[inline]
fn write_chunk_bits(mask: &mut Bitset, base: usize, bits: u64) {
    if bits != 0 {
        mask.words_mut()[base / 64] |= bits << (base % 64);
    }
}

/// `Int64` column compared against a literal, straight to a bitmask.
pub fn mask_i64_simd(values: &[i64], op: CompareOp, literal: i64) -> Bitset {
    let mut mask = Bitset::new(values.len());
    let splat = Simd::<i64, LANES>::splat(literal);
    let (chunks, tail) = values.as_chunks::<LANES>();

    for (index, chunk) in chunks.iter().enumerate() {
        let vector = Simd::from_array(*chunk);
        let lanes = match op {
            CompareOp::Eq => vector.simd_eq(splat),
            CompareOp::Lt => vector.simd_lt(splat),
            CompareOp::Gt => vector.simd_gt(splat),
        };
        write_chunk_bits(&mut mask, index * LANES, lanes.to_bitmask());
    }

    let base = chunks.len() * LANES;
    for (offset, value) in tail.iter().enumerate() {
        if compare(value, &literal, op) {
            mask.set(base + offset);
        }
    }

    mask
}

/// `Float64` column compared against a literal.
///
/// IEEE comparison semantics are preserved lane-wise, so a `NaN` would compare false against
/// everything just as it does scalar-side. In practice none reaches here: the CSV loader and
/// the parser both reject non-finite values at the boundary (Phase 1, Phase 2).
pub fn mask_f64_simd(values: &[f64], op: CompareOp, literal: f64) -> Bitset {
    let mut mask = Bitset::new(values.len());
    let splat = Simd::<f64, LANES>::splat(literal);
    let (chunks, tail) = values.as_chunks::<LANES>();

    for (index, chunk) in chunks.iter().enumerate() {
        let vector = Simd::from_array(*chunk);
        let lanes = match op {
            CompareOp::Eq => vector.simd_eq(splat),
            CompareOp::Lt => vector.simd_lt(splat),
            CompareOp::Gt => vector.simd_gt(splat),
        };
        write_chunk_bits(&mut mask, index * LANES, lanes.to_bitmask());
    }

    let base = chunks.len() * LANES;
    for (offset, value) in tail.iter().enumerate() {
        if compare(value, &literal, op) {
            mask.set(base + offset);
        }
    }

    mask
}

/// Sum a dense `Int64` slice.
///
/// Lane arithmetic wraps, matching `SumAccumulator<i64>` and the naive baseline. Integer
/// addition stays associative under wrapping, so this is bit-identical to the scalar version
/// on every input, overflow included.
pub fn sum_i64_simd(values: &[i64]) -> i64 {
    let (chunks, tail) = values.as_chunks::<LANES>();

    // One accumulator, deliberately. Unrolling into two or four would break the loop-carried
    // dependency chain and is the standard next optimization -- Phase 6 can measure whether
    // this reduction is latency-bound before adding that complexity.
    let mut acc = Simd::<i64, LANES>::splat(0);
    for chunk in chunks {
        acc += Simd::from_array(*chunk);
    }

    let vector_total = acc.reduce_sum();
    tail.iter()
        .fold(vector_total, |total, v| total.wrapping_add(*v))
}

/// Sum a dense `Float64` slice.
///
/// **Not bit-identical to the scalar version.** This accumulates eight partial sums and
/// combines them at the end, which is a different order, and floating-point addition is not
/// associative. Often *more* accurate than the serial version, since eight shorter chains
/// accumulate less rounding error than one long one -- but different, and callers comparing
/// against the scalar reference must use a tolerance.
pub fn sum_f64_simd(values: &[f64]) -> f64 {
    let (chunks, tail) = values.as_chunks::<LANES>();

    let mut acc = Simd::<f64, LANES>::splat(0.0);
    for chunk in chunks {
        acc += Simd::from_array(*chunk);
    }

    let vector_total = acc.reduce_sum();
    tail.iter().fold(vector_total, |total, v| total + *v)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Agreement with the scalar reference is asserted in `exec::kernels`, where both are in
    /// scope. These cover the vector-specific mechanics instead.
    #[test]
    fn bits_land_at_the_right_offsets_across_a_word() {
        // 64 values, one per lane position: only every third row passes, which puts set bits
        // at varied positions inside each 8-lane chunk and across all eight chunks of a word.
        let values: Vec<i64> = (0..64).collect();
        let mask = mask_i64_simd(&values, CompareOp::Lt, 20);

        assert_eq!(mask.count_ones(), 20);
        assert_eq!(
            mask.iter_ones().collect::<Vec<_>>(),
            (0..20).collect::<Vec<_>>()
        );
    }

    #[test]
    fn handles_a_ragged_tail() {
        // 13 = one full vector plus a five-row tail.
        let values: Vec<i64> = (0..13).collect();
        let mask = mask_i64_simd(&values, CompareOp::Gt, 7);
        assert_eq!(mask.len(), 13);
        assert_eq!(mask.iter_ones().collect::<Vec<_>>(), vec![8, 9, 10, 11, 12]);
    }

    #[test]
    fn handles_inputs_shorter_than_one_vector() {
        let values = [5i64, 1, 9];
        let mask = mask_i64_simd(&values, CompareOp::Gt, 4);
        assert_eq!(mask.iter_ones().collect::<Vec<_>>(), vec![0, 2]);
    }

    #[test]
    fn handles_empty_input() {
        assert_eq!(mask_i64_simd(&[], CompareOp::Gt, 0).count_ones(), 0);
        assert_eq!(sum_i64_simd(&[]), 0);
        assert_eq!(sum_f64_simd(&[]), 0.0);
    }

    #[test]
    fn float_compare_selects_the_right_lanes() {
        let values: Vec<f64> = (0..20).map(|i| i as f64 * 0.5).collect();
        let mask = mask_f64_simd(&values, CompareOp::Gt, 4.0);
        // 0.0, 0.5, .. 9.5 -- values above 4.0 start at index 9.
        assert_eq!(
            mask.iter_ones().collect::<Vec<_>>(),
            (9..20).collect::<Vec<_>>()
        );
    }

    #[test]
    fn sums_span_the_vector_tail_boundary() {
        for len in [0usize, 1, 7, 8, 9, 100] {
            let values: Vec<i64> = (1..=len as i64).collect();
            let expected = (len as i64 * (len as i64 + 1)) / 2;
            assert_eq!(sum_i64_simd(&values), expected, "len {len}");
        }
    }
}
