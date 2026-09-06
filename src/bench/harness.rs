//! An alternating, minimum-of-N measurement harness.
//!
//! # Why not just use criterion
//! Criterion stays, for the reproducible artifact and the CSV export. It is not what the
//! comparisons are read off, for two reasons this project measured rather than assumed:
//!
//! 1. **It runs A to completion, then B.** Any drift over the run — thermal, frequency,
//!    background load — lands on the two arms unequally, and drift walks across a group even
//!    inside a single process. Alternating per sample gives both arms the same machine.
//! 2. **Its estimator assumes symmetric noise.** Interference is one-sided: a sample can be
//!    slowed by an interrupt, never sped up. The mean absorbs every such event. The minimum is
//!    the observation closest to the machine's actual capability, and on this project's data
//!    it cut a 19.7% apparent difference on provably identical work down to 0.3%.
//!
//! This follows the evidence rather than the convention. What held up across Phase 5 and 6 was
//! `examples/kernels.rs` — alternate within one process, take the best of N. What failed
//! repeatedly was comparing criterion runs across invocations, where a median 7.5% and maximum
//! 66.8% drift between *identical* runs made every effect under study unresolvable.
//!
//! Recorded as a deliberate deviation from `phases.md`, which specified criterion for the
//! comparisons. The reasoning is above; the numbers are in `phases.md` Phase 6.
//!
//! # The resolution of this harness is itself measured
//! [`Arm`]s are boxed closures, so two registrations of the same work cannot be merged by the
//! optimizer into a single call. That makes an **A/A null test** possible: register one arm
//! twice, alternate, and read the spread. The true difference is zero, so whatever comes back
//! is the harness's resolution — the smallest effect it can honestly claim. See
//! `examples/aa_null.rs`.

use std::time::Instant;

use crate::storage::{Column, Table};

/// An **order-insensitive** checksum of a query result.
///
/// Order-insensitive because group order is unspecified and the strategies genuinely differ:
/// hash group-by emits first-seen order, sort group-by emits key order. A positional rolling
/// hash would report those as disagreeing results when they are the same answer -- which it
/// did, the first time this was written.
///
/// Per-row contributions are combined with `wrapping_add`, which is commutative, so the same
/// set of `(label, sum)` pairs hashes identically whatever order they arrive in.
pub fn result_checksum(table: &Table, group: &str, agg: &str) -> u64 {
    let group_column = table.column(group);
    let agg_column = table.column(agg);

    let mut total = table.nrows() as u64;
    for row in 0..table.nrows() {
        let label = match group_column {
            Some(column @ Column::Utf8Dict { .. }) => {
                fnv1a(column.utf8_value(row).unwrap_or("").as_bytes())
            }
            Some(Column::Int64(v)) => v[row] as u64,
            _ => 0,
        };
        let sum = match agg_column {
            Some(Column::Int64(v)) => v[row] as u64,
            Some(Column::Float64(v)) => v[row].to_bits(),
            _ => 0,
        };
        total = total.wrapping_add(label.wrapping_mul(0x1000_0000_01b3).wrapping_add(sum));
    }
    total
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    hash
}

/// One thing to measure.
///
/// The closure returns a `u64` checksum. That is not decoration: it stops the optimizer
/// deleting work whose result is unused, and it lets the harness verify that arms which should
/// agree actually do, at measurement time rather than only in tests.
pub struct Arm<'a> {
    pub name: String,
    run: Box<dyn FnMut() -> u64 + 'a>,
}

impl<'a> Arm<'a> {
    pub fn new(name: impl Into<String>, run: impl FnMut() -> u64 + 'a) -> Self {
        Arm {
            name: name.into(),
            run: Box::new(run),
        }
    }
}

/// Every sample for one arm, in the order they were taken.
#[derive(Debug, Clone)]
pub struct Measurement {
    pub name: String,
    /// Milliseconds per call.
    pub samples: Vec<f64>,
    /// The checksum every call returned. Constant by construction, or the arm is not
    /// deterministic and nothing it reports means anything.
    pub checksum: u64,
}

impl Measurement {
    /// The headline statistic. See the module docs on why not the mean.
    pub fn min(&self) -> f64 {
        self.samples.iter().cloned().fold(f64::INFINITY, f64::min)
    }

    pub fn max(&self) -> f64 {
        self.samples.iter().cloned().fold(0.0, f64::max)
    }

    pub fn median(&self) -> f64 {
        self.quantile(0.5)
    }

    pub fn quantile(&self, q: f64) -> f64 {
        let mut sorted = self.samples.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).expect("no NaN timings"));
        let index = ((sorted.len() - 1) as f64 * q).round() as usize;
        sorted[index]
    }

    /// How far the worst sample sits above the best, as a percentage.
    ///
    /// A measure of the run's own instability, not of the thing being measured.
    pub fn spread_percent(&self) -> f64 {
        (self.max() / self.min() - 1.0) * 100.0
    }
}

/// Run every arm once per round, rotating, for `samples` rounds.
///
/// **ABAB, never A-block then B-block.** Blocking the arms is what lets drift attach itself to
/// one of them; rotating means any drift over the run is shared. `warmup` rounds run first and
/// are discarded, so page faults and cold caches are not charged to the first arm.
///
/// Panics if an arm returns different checksums on different calls, which would mean its work
/// is not reproducible and its timings are not comparable.
pub fn alternate(mut arms: Vec<Arm<'_>>, samples: usize, warmup: usize) -> Vec<Measurement> {
    assert!(!arms.is_empty(), "nothing to measure");
    assert!(samples > 0, "need at least one sample");

    let mut recorded: Vec<Vec<f64>> = vec![Vec::with_capacity(samples); arms.len()];
    let mut checksums: Vec<Option<u64>> = vec![None; arms.len()];

    for _ in 0..warmup {
        for arm in arms.iter_mut() {
            std::hint::black_box((arm.run)());
        }
    }

    for _ in 0..samples {
        for (index, arm) in arms.iter_mut().enumerate() {
            let start = Instant::now();
            let checksum = (arm.run)();
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            std::hint::black_box(checksum);

            match checksums[index] {
                None => checksums[index] = Some(checksum),
                Some(previous) => assert_eq!(
                    previous, checksum,
                    "arm {:?} is not deterministic; its timings mean nothing",
                    arm.name
                ),
            }
            recorded[index].push(elapsed);
        }
    }

    arms.into_iter()
        .zip(recorded)
        .zip(checksums)
        .map(|((arm, samples), checksum)| Measurement {
            name: arm.name,
            samples,
            checksum: checksum.expect("at least one sample"),
        })
        .collect()
}

/// Print a table of measurements, with the first as the reference.
pub fn report(measurements: &[Measurement]) {
    println!(
        "  {:<24} {:>9} {:>9} {:>9} {:>8} {:>9}",
        "arm", "min", "median", "max", "spread", "vs first"
    );

    let reference = measurements.first().map(|m| m.min()).unwrap_or(1.0);
    for m in measurements {
        println!(
            "  {:<24} {:>8.3} {:>8.3} {:>8.3} {:>7.1}% {:>8.3}x",
            m.name,
            m.min(),
            m.median(),
            m.max(),
            m.spread_percent(),
            m.min() / reference,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_every_sample_for_every_arm() {
        let arms = vec![Arm::new("a", || 1), Arm::new("b", || 2)];
        let out = alternate(arms, 5, 1);

        assert_eq!(out.len(), 2);
        assert_eq!(out[0].samples.len(), 5);
        assert_eq!(out[1].samples.len(), 5);
        assert_eq!(out[0].checksum, 1);
        assert_eq!(out[1].checksum, 2);
    }

    #[test]
    #[should_panic(expected = "is not deterministic")]
    fn rejects_an_arm_whose_answer_changes() {
        // Guards the whole measurement model: an arm that does different work each call
        // cannot be compared against anything.
        let mut counter = 0u64;
        let arms = vec![Arm::new("drifting", move || {
            counter += 1;
            counter
        })];
        alternate(arms, 3, 0);
    }

    #[test]
    fn statistics_are_order_independent() {
        let m = Measurement {
            name: "x".to_string(),
            samples: vec![3.0, 1.0, 2.0, 5.0, 4.0],
            checksum: 0,
        };
        assert_eq!(m.min(), 1.0);
        assert_eq!(m.max(), 5.0);
        assert_eq!(m.median(), 3.0);
        assert!((m.spread_percent() - 400.0).abs() < 1e-9);
    }
}
