//! The hot loops, and the single place that decides whether SIMD runs them.
//!
//! # Scalar is the implementation; SIMD is an optimization of it
//! Every kernel here has a `_scalar` version that is **always compiled and always tested**,
//! on stable and on nightly alike (`agents.md`: "SIMD is an optimization path, never the only
//! implementation of a given operation"). The `simd` feature only changes which one the
//! dispatchers call. With the feature on, the tests below run *both* and assert they agree,
//! so the vector path can never silently drift from the reference.
//!
//! # What is vectorized, and what deliberately is not
//! Per `systemDesign.md` "SIMD Scope":
//!
//! | kernel | vectorized | why |
//! |---|---|---|
//! | `Int64` / `Float64` filter compare | yes | dense numeric compare straight to a bitmask |
//! | `Int64` / `Float64` sum reduction | yes | dense sequential read, one accumulator |
//! | int-column vs float-literal compare | no | mixed-type; see [`crate::exec::filter`] |
//! | `Utf8` filtering | no | `agents.md` hard guardrail |
//! | group-by scatter-accumulate | no | data-dependent store addresses |
//!
//! The last row is the headline finding the whole benchmark exists to demonstrate, and it is
//! a property of the *operation*, not a gap in this module: SIMD can load eight values at
//! once but cannot store them to eight computed addresses.
//!
//! An observation worth the write-up: `Utf8` **equality** filtering reduces to comparing
//! `u32` dictionary codes, which is a dense numeric compare that would vectorize perfectly.
//! The guardrail forecloses it, so the cost of "no SIMD on strings" is larger than it first
//! looks -- dictionary encoding had already turned the string problem into an integer one.

use crate::exec::bitset::Bitset;
use crate::plan::CompareOp;

/// Which implementation of a kernel to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kernel {
    Scalar,
    Simd,
}

impl Kernel {
    pub fn name(self) -> &'static str {
        match self {
            Kernel::Scalar => "scalar",
            Kernel::Simd => "simd",
        }
    }
}

/// Runtime kernel selection, compiled in only under the `bench-dispatch` feature.
///
/// # Why this exists
/// Comparing scalar against SIMD across two `cargo` invocations was measured and found
/// unusable: two runs of an *identical* binary on identical data differ by a median of 7.5%,
/// which is larger than the effect. The only reliable comparison alternates the two inside one
/// process, sample by sample — and that requires choosing the kernel at runtime.
///
/// # What it costs, and where
/// The branch is resolved **per kernel call**, never per row. A kernel call processes a whole
/// batch (filter) or a whole run (sum), so the cost is one relaxed atomic load amortized over
/// hundreds or thousands of elements. `SortAggregate::aggregate_runs` hoists it further,
/// reading the mode once for the entire aggregation rather than once per run, because at high
/// cardinality runs are only a couple of rows long and per-run overhead would land close
/// enough to per-row to matter.
///
/// Without the feature, [`active_kernel`] is a `const`-foldable function returning whatever
/// `cfg` selected, so the match compiles away entirely and the default build carries **no
/// runtime branch**.
#[cfg(feature = "bench-dispatch")]
pub mod dispatch {
    use super::Kernel;
    use std::sync::atomic::{AtomicU8, Ordering};

    static ACTIVE: AtomicU8 = AtomicU8::new(1);

    /// Select the kernel every subsequent call will use.
    pub fn set(kernel: Kernel) {
        ACTIVE.store(
            match kernel {
                Kernel::Scalar => 0,
                Kernel::Simd => 1,
            },
            Ordering::Relaxed,
        );
    }

    /// Relaxed because this is read far more often than written, and a stale read is
    /// impossible in practice: the benchmark sets the mode, then runs, single-threaded.
    #[inline]
    pub fn get() -> Kernel {
        match ACTIVE.load(Ordering::Relaxed) {
            0 => Kernel::Scalar,
            _ => Kernel::Simd,
        }
    }
}

/// The kernel to run right now.
#[cfg(feature = "bench-dispatch")]
#[inline]
pub fn active_kernel() -> Kernel {
    dispatch::get()
}

/// Compile-time selection: the default. Folds to a constant, so every `match` on it
/// disappears and the chosen kernel is called directly.
#[cfg(not(feature = "bench-dispatch"))]
#[inline(always)]
pub fn active_kernel() -> Kernel {
    if cfg!(feature = "simd") {
        Kernel::Simd
    } else {
        Kernel::Scalar
    }
}

#[inline]
pub fn compare<T: PartialOrd + ?Sized>(a: &T, b: &T, op: CompareOp) -> bool {
    match op {
        CompareOp::Eq => a == b,
        CompareOp::Lt => a < b,
        CompareOp::Gt => a > b,
    }
}

/// Evaluate `predicate` for every row, packing 64 results into each word.
///
/// The generic scalar mask builder, used directly for the comparisons that stay scalar.
/// Writing whole words rather than setting bits one at a time is what makes the scalar and
/// vector paths write through the same interface: a SIMD compare also produces bits, just
/// eight at a time instead of one.
#[inline]
pub fn mask_from<F: FnMut(usize) -> bool>(len: usize, mut predicate: F) -> Bitset {
    let mut mask = Bitset::new(len);

    for (word_index, word) in mask.words_mut().iter_mut().enumerate() {
        let start = word_index * 64;
        let end = (start + 64).min(len);
        let mut bits = 0u64;
        for row in start..end {
            // Branchless on the store side: the bool becomes a bit, not a jump.
            bits |= (predicate(row) as u64) << (row - start);
        }
        *word = bits;
    }

    mask
}

// ---- Reference implementations (always compiled, always tested) --------------------

pub fn mask_i64_scalar(values: &[i64], op: CompareOp, literal: i64) -> Bitset {
    mask_from(values.len(), |row| compare(&values[row], &literal, op))
}

pub fn mask_f64_scalar(values: &[f64], op: CompareOp, literal: f64) -> Bitset {
    mask_from(values.len(), |row| compare(&values[row], &literal, op))
}

/// Wrapping, to match `SumAccumulator<i64>` and the naive baseline. SIMD lane arithmetic
/// wraps and cannot panic, so wrapping everywhere is what keeps all three paths agreeing
/// bit-for-bit on overflow instead of one panicking in a debug build.
pub fn sum_i64_scalar(values: &[i64]) -> i64 {
    values.iter().fold(0i64, |acc, v| acc.wrapping_add(*v))
}

pub fn sum_f64_scalar(values: &[f64]) -> f64 {
    values.iter().sum()
}

// ---- Dispatch: the only place the feature flag changes behaviour --------------------

pub fn mask_i64(values: &[i64], op: CompareOp, literal: i64) -> Bitset {
    mask_i64_with(active_kernel(), values, op, literal)
}

/// One branch per call, outside the row loop the kernel then runs.
pub fn mask_i64_with(kernel: Kernel, values: &[i64], op: CompareOp, literal: i64) -> Bitset {
    match kernel {
        Kernel::Scalar => mask_i64_scalar(values, op, literal),
        #[cfg(feature = "simd")]
        Kernel::Simd => crate::simd::mask_i64_simd(values, op, literal),
        #[cfg(not(feature = "simd"))]
        Kernel::Simd => unreachable!("SIMD kernels are not compiled into this build"),
    }
}

pub fn mask_f64(values: &[f64], op: CompareOp, literal: f64) -> Bitset {
    mask_f64_with(active_kernel(), values, op, literal)
}

pub fn mask_f64_with(kernel: Kernel, values: &[f64], op: CompareOp, literal: f64) -> Bitset {
    match kernel {
        Kernel::Scalar => mask_f64_scalar(values, op, literal),
        #[cfg(feature = "simd")]
        Kernel::Simd => crate::simd::mask_f64_simd(values, op, literal),
        #[cfg(not(feature = "simd"))]
        Kernel::Simd => unreachable!("SIMD kernels are not compiled into this build"),
    }
}

/// Sum a dense slice.
///
/// Not used by the hash group-by, which scatters into per-group accumulators rather than
/// reducing a slice. Its consumer is Phase 6's **sort-based grouping**: once rows are sorted
/// by key, each group's values are contiguous, so run-length aggregation becomes exactly this
/// dense reduction. That is the mechanism behind `systemDesign.md`'s prediction that
/// sort-group's aggregation phase beats hash-group's scatter -- and the reason the sort's
/// `O(n log n)` may still lose overall.
pub fn sum_i64(values: &[i64]) -> i64 {
    sum_i64_with(active_kernel(), values)
}

pub fn sum_i64_with(kernel: Kernel, values: &[i64]) -> i64 {
    match kernel {
        Kernel::Scalar => sum_i64_scalar(values),
        #[cfg(feature = "simd")]
        Kernel::Simd => crate::simd::sum_i64_simd(values),
        #[cfg(not(feature = "simd"))]
        Kernel::Simd => unreachable!("SIMD kernels are not compiled into this build"),
    }
}

/// Sum a dense slice of floats.
///
/// **Not bit-identical to [`sum_f64_scalar`].** Floating-point addition is not associative,
/// and the vector version accumulates in lanes -- eight partial sums combined at the end --
/// so results can differ in the last ULP. Neither is "wrong"; the lane-wise version is often
/// *more* accurate, because eight shorter chains lose less precision than one long one.
/// Integer sums have no such caveat, which is why the tests below assert exact equality for
/// `i64` and a relative tolerance for `f64`.
pub fn sum_f64(values: &[f64]) -> f64 {
    sum_f64_with(active_kernel(), values)
}

pub fn sum_f64_with(kernel: Kernel, values: &[f64]) -> f64 {
    match kernel {
        Kernel::Scalar => sum_f64_scalar(values),
        #[cfg(feature = "simd")]
        Kernel::Simd => crate::simd::sum_f64_simd(values),
        #[cfg(not(feature = "simd"))]
        Kernel::Simd => unreachable!("SIMD kernels are not compiled into this build"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Lengths chosen to straddle every boundary that matters: below one vector, exactly one,
    /// ragged tails, exactly one bitset word, and multiple words with a partial last one.
    const LENGTHS: [usize; 12] = [0, 1, 7, 8, 9, 15, 16, 63, 64, 65, 127, 1000];

    fn i64_data(len: usize) -> Vec<i64> {
        // Spans negatives, zero, and values on both sides of every threshold tested.
        (0..len as i64).map(|i| (i * 7) % 23 - 11).collect()
    }

    fn f64_data(len: usize) -> Vec<f64> {
        (0..len).map(|i| (i as f64) * 0.37 - 9.5).collect()
    }

    const OPS: [CompareOp; 3] = [CompareOp::Eq, CompareOp::Lt, CompareOp::Gt];

    /// The reference behaviour, spelled out independently of `mask_from` so the test does not
    /// just re-run the implementation.
    fn expected_mask<T: PartialOrd + Copy>(values: &[T], op: CompareOp, literal: T) -> Vec<usize> {
        values
            .iter()
            .enumerate()
            .filter(|(_, v)| compare(*v, &literal, op))
            .map(|(i, _)| i)
            .collect()
    }

    #[test]
    fn scalar_int_mask_matches_a_plain_filter() {
        for len in LENGTHS {
            let values = i64_data(len);
            for op in OPS {
                for literal in [-11, -5, 0, 5, 11, 100] {
                    let mask = mask_i64_scalar(&values, op, literal);
                    assert_eq!(mask.len(), len);
                    assert_eq!(
                        mask.iter_ones().collect::<Vec<_>>(),
                        expected_mask(&values, op, literal),
                        "len {len}, op {op:?}, literal {literal}"
                    );
                }
            }
        }
    }

    #[test]
    fn scalar_float_mask_matches_a_plain_filter() {
        for len in LENGTHS {
            let values = f64_data(len);
            for op in OPS {
                for literal in [-9.5, -1.0, 0.0, 3.33, 500.0] {
                    let mask = mask_f64_scalar(&values, op, literal);
                    assert_eq!(
                        mask.iter_ones().collect::<Vec<_>>(),
                        expected_mask(&values, op, literal),
                        "len {len}, op {op:?}, literal {literal}"
                    );
                }
            }
        }
    }

    #[test]
    fn scalar_int_sum_wraps_rather_than_panicking() {
        assert_eq!(sum_i64_scalar(&[]), 0);
        assert_eq!(sum_i64_scalar(&[1, 2, 3]), 6);
        assert_eq!(sum_i64_scalar(&[i64::MAX, 1]), i64::MIN);
    }

    // ---- The twins must agree (only meaningful with the feature on) -----------------

    #[cfg(feature = "simd")]
    mod simd_parity {
        use super::*;
        use crate::simd;

        #[test]
        fn int_masks_are_identical() {
            for len in LENGTHS {
                let values = i64_data(len);
                for op in OPS {
                    for literal in [-11, -5, 0, 5, 11, 100] {
                        assert_eq!(
                            simd::mask_i64_simd(&values, op, literal),
                            mask_i64_scalar(&values, op, literal),
                            "len {len}, op {op:?}, literal {literal}"
                        );
                    }
                }
            }
        }

        #[test]
        fn float_masks_are_identical() {
            for len in LENGTHS {
                let values = f64_data(len);
                for op in OPS {
                    for literal in [-9.5, -1.0, 0.0, 3.33, 500.0] {
                        assert_eq!(
                            simd::mask_f64_simd(&values, op, literal),
                            mask_f64_scalar(&values, op, literal),
                            "len {len}, op {op:?}, literal {literal}"
                        );
                    }
                }
            }
        }

        #[test]
        fn int_sums_are_identical_including_on_overflow() {
            // Integer addition is associative even when it wraps, so lane-wise accumulation
            // must reproduce the scalar result exactly -- no tolerance, ever.
            for len in LENGTHS {
                let values = i64_data(len);
                assert_eq!(
                    simd::sum_i64_simd(&values),
                    sum_i64_scalar(&values),
                    "len {len}"
                );
            }

            let overflowing = vec![i64::MAX; 9];
            assert_eq!(
                simd::sum_i64_simd(&overflowing),
                sum_i64_scalar(&overflowing),
                "wrapping must agree across the lane/tail split"
            );
        }

        #[test]
        fn float_sums_agree_to_within_rounding() {
            // Lane-wise accumulation is a different summation order, so the last ULP may
            // differ. That is expected and documented on `sum_f64`, not a defect.
            for len in LENGTHS {
                let values = f64_data(len);
                let vector = simd::sum_f64_simd(&values);
                let scalar = sum_f64_scalar(&values);

                let tolerance = scalar.abs() * 1e-12 + 1e-12;
                assert!(
                    (vector - scalar).abs() <= tolerance,
                    "len {len}: simd {vector} vs scalar {scalar}"
                );
            }
        }

        #[test]
        fn float_sum_order_actually_differs_somewhere() {
            // Guards against the tolerance test being vacuous. Values engineered so that a
            // long serial chain loses a small addend that lane-wise accumulation keeps:
            // 1.0 is below the ULP of 2^53, but eight of them summed separately are not.
            let mut values = vec![1.0f64; 64];
            values[0] = 9_007_199_254_740_992.0; // 2^53

            let vector = simd::sum_f64_simd(&values);
            let scalar = sum_f64_scalar(&values);
            assert_ne!(
                vector, scalar,
                "expected the two summation orders to disagree on this input"
            );
        }
    }
}
