//! Deterministic synthetic data, and the environment record that makes timings citable.
//!
//! Library code rather than something buried in a harness target, so every benchmark and every
//! test draws from the same generator. Regenerating with the same seed produces byte-identical
//! data, which is what lets a rerun be compared against an earlier one at all.
//!
//! # Tables are built directly, not through CSV
//! Loading is not what the engine is being measured on, and parsing a few million rows of CSV
//! would dominate setup for every parameter combination. [`generate`] builds the columnar
//! `Table` straight away. The CSV loader has its own tests; nothing here needs to re-exercise
//! it, and keeping it out of the setup path keeps the sweeps affordable.
//!
//! # The knobs, and why these three
//! - **Group cardinality.** The axis this project has already been burned on: the `Arc`
//!   dictionary bug was invisible at 8 distinct values and 84x at 10,000. It is also where
//!   hash-group and sort-group should trade places, and where sort-group's run lengths shrink
//!   until the SIMD sum kernel stops paying.
//! - **Filter selectivity.** Compaction cost scales with how many rows survive, so a 1% filter
//!   and a 90% filter exercise different halves of the pipeline.
//! - **Row count.** Enough points to show whether a curve is bandwidth-bound rather than
//!   compute-bound — Phase 5's sum result predicts a ceiling, and the suite should confirm or
//!   refute it rather than leave it asserted.

use std::sync::Arc;

use crate::storage::{Column, Table};

/// xorshift64*, so a seed reproduces a dataset exactly.
///
/// Not a dependency and not statistically rigorous — the data needs to be varied and
/// repeatable, which this is.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        // Zero is a fixed point for xorshift; nudge it rather than silently producing zeros.
        Rng(if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        })
    }

    /// Named `next_u64` rather than `next` so it cannot be mistaken for `Iterator::next`.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    pub fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }
}

/// `amount` is uniform over `0..AMOUNT_RANGE`, which is what makes selectivity a dial rather
/// than a guess — see [`amount_threshold`].
pub const AMOUNT_RANGE: u64 = 1_000_000;

/// The `WHERE amount > threshold` value that passes approximately `fraction` of rows.
///
/// Exact in expectation because `amount` is uniform. A benchmark that says "selectivity 10%"
/// and means it is worth the two lines.
pub fn amount_threshold(fraction: f64) -> i64 {
    assert!(
        (0.0..=1.0).contains(&fraction),
        "selectivity must be a fraction"
    );
    ((1.0 - fraction) * AMOUNT_RANGE as f64) as i64
}

/// The default seed. Named so every benchmark uses the same one without repeating a literal.
pub const SEED: u64 = 0xBA7C_B12D;

/// Build a dataset with `cardinality` distinct group labels.
///
/// Columns:
/// - `region` — `Utf8`, `cardinality` distinct values, the dictionary-encoded group key
/// - `bucket` — `Int64`, `cardinality` distinct values, the same grouping without a dictionary
/// - `amount` — `Int64`, uniform over `0..AMOUNT_RANGE`, the filter and sum column
/// - `price`  — `Float64`, the float sum column
///
/// `region` and `bucket` carry the *same* cardinality deliberately: grouping by one and then
/// the other isolates the cost of dictionary-encoded keys from everything else, which is
/// exactly how the Phase 4 dictionary bug was diagnosed.
pub fn generate(rows: usize, cardinality: u64, seed: u64) -> Table {
    assert!(cardinality > 0, "cardinality must be positive");

    let mut rng = Rng::new(seed);

    let dict: Vec<String> = (0..cardinality).map(|i| format!("r{i}")).collect();
    let mut codes = Vec::with_capacity(rows);
    let mut buckets = Vec::with_capacity(rows);
    let mut amounts = Vec::with_capacity(rows);
    let mut prices = Vec::with_capacity(rows);

    for _ in 0..rows {
        let group = rng.below(cardinality);
        codes.push(group as u32);
        buckets.push(group as i64);
        amounts.push(rng.below(AMOUNT_RANGE) as i64);
        prices.push((rng.below(100_000) as f64) / 100.0 - 500.0);
    }

    let mut columns = crate::hash::map_with_capacity(4);
    columns.insert(
        "region".to_string(),
        Column::Utf8Dict {
            dict: Arc::from(dict),
            codes,
        },
    );
    columns.insert("bucket".to_string(), Column::Int64(buckets));
    columns.insert("amount".to_string(), Column::Int64(amounts));
    columns.insert("price".to_string(), Column::Float64(prices));

    Table::new(columns, rows).expect("generated columns are equal length by construction")
}

/// A record of what produced a set of timings.
///
/// `phases.md` and any Phase 7 chart are worthless without this. It caught a real error: a
/// scalar-vs-SIMD comparison that had silently also changed compiler, which the recorded
/// `rustc` line made obvious.
///
/// Split into **detected** and **declared** because they have different provenance. The
/// process can read its own toolchain and CPU; it cannot verify that the operator pinned it to
/// a core or capped the clock. Those are declared through `BATCHBIRD_RUN_CONFIG` and reported
/// as declarations, never as facts the program checked.
pub fn environment() -> String {
    let rustc = std::process::Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    let cpu = std::env::var("PROCESSOR_IDENTIFIER")
        .ok()
        .or_else(|| {
            // Linux/macOS runners: first model-name line, if there is one.
            std::fs::read_to_string("/proc/cpuinfo").ok().and_then(|s| {
                s.lines()
                    .find(|l| l.starts_with("model name"))
                    .and_then(|l| l.split(':').nth(1))
                    .map(|v| v.trim().to_string())
            })
        })
        .unwrap_or_else(|| "unknown".to_string());

    let simd = if cfg!(feature = "simd") {
        "on (std::simd kernels)"
    } else {
        "off (scalar kernels)"
    };

    let declared = std::env::var("BATCHBIRD_RUN_CONFIG").unwrap_or_else(|_| {
        "(none declared -- set BATCHBIRD_RUN_CONFIG with the measurement conditions)".to_string()
    });

    format!(
        "-- detected ------------------------------------------------
         os:          {} / {}
         cpu:         {cpu}
         rustc:       {rustc}
         profile:     release, codegen-units=1, lto=thin, debug=true
         simd:        {simd}
         data seed:   {SEED:#x}
         hasher seed: {:#x}  (fixed; see crate::hash)
         -- declared -----------------------------------------------
         {declared}",
        std::env::consts::OS,
        std::env::consts::ARCH,
        crate::hash::DEFAULT_SEED,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::DataType;

    #[test]
    fn the_same_seed_reproduces_the_same_table() {
        // The property every recorded timing depends on.
        let a = generate(1_000, 16, 42);
        let b = generate(1_000, 16, 42);
        assert_eq!(a, b);

        let different = generate(1_000, 16, 43);
        assert_ne!(a, different, "a different seed must produce different data");
    }

    #[test]
    fn columns_have_the_documented_types() {
        let table = generate(100, 4, SEED);
        assert_eq!(table.nrows(), 100);
        assert_eq!(table.ncols(), 4);
        assert_eq!(table.column("region").unwrap().data_type(), DataType::Utf8);
        assert_eq!(table.column("bucket").unwrap().data_type(), DataType::Int64);
        assert_eq!(table.column("amount").unwrap().data_type(), DataType::Int64);
        assert_eq!(
            table.column("price").unwrap().data_type(),
            DataType::Float64
        );
    }

    #[test]
    fn cardinality_is_respected() {
        for cardinality in [1u64, 8, 500] {
            let table = generate(20_000, cardinality, SEED);
            let (dict, _) = table.column("region").unwrap().as_utf8_dict().unwrap();
            assert_eq!(dict.len() as u64, cardinality, "cardinality {cardinality}");
        }
    }

    #[test]
    fn region_and_bucket_encode_the_same_grouping() {
        // What makes "dictionary key vs integer key, all else equal" a valid comparison.
        let table = generate(5_000, 32, SEED);
        let (_, codes) = table.column("region").unwrap().as_utf8_dict().unwrap();
        let buckets = table.column("bucket").unwrap().as_i64().unwrap();

        assert!(
            codes.iter().zip(buckets).all(|(c, b)| *c as i64 == *b),
            "region codes and bucket values must agree row for row"
        );
    }

    #[test]
    fn selectivity_thresholds_are_close_to_their_target() {
        let table = generate(50_000, 8, SEED);
        let amounts = table.column("amount").unwrap().as_i64().unwrap();

        for target in [0.01, 0.1, 0.5, 0.9] {
            let threshold = amount_threshold(target);
            let passing = amounts.iter().filter(|v| **v > threshold).count();
            let actual = passing as f64 / amounts.len() as f64;

            assert!(
                (actual - target).abs() < 0.02,
                "selectivity {target}: got {actual:.4}"
            );
        }
    }

    #[test]
    fn a_zero_seed_still_produces_varied_data() {
        // Zero is a fixed point for xorshift; the constructor nudges it rather than emitting
        // a column of zeros that would silently make every benchmark meaningless.
        let table = generate(1_000, 8, 0);
        let amounts = table.column("amount").unwrap().as_i64().unwrap();
        assert!(amounts.iter().any(|v| *v != amounts[0]));
    }

    #[test]
    fn environment_reports_the_simd_state() {
        let described = environment();
        assert!(described.contains("rustc:"));
        assert!(described.contains("hasher seed:"));
        assert!(described.contains("-- declared"));
        if cfg!(feature = "simd") {
            assert!(described.contains("simd:        on"));
        } else {
            assert!(described.contains("simd:        off"));
        }
    }
}
