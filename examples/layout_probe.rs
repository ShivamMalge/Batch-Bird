//! What mechanism produces the layout gap?
//!
//! ```text
//! cargo run --release --example layout_probe
//! ```
//!
//! The row-oriented arm is ~1.7x slower than the columnar one at fixed cardinality, and the
//! obvious explanation — memory bandwidth — is dead: the pipeline runs at 0.75-1.4 GB/s
//! against a demonstrated ~20 GB/s ceiling, and per-row cost is flat across a 32x range of
//! working set. So the gap needs a mechanism, and this tests the remaining candidate.
//!
//! # The hypothesis
//! **Cache-line utilization.** A row store reads one field but drags the whole row into cache
//! with it. A 64-byte line holds two 32-byte rows, so scanning one 8-byte field wastes 3/4 of
//! every line fetched. Columnar wastes nothing: consecutive values of the scanned field are
//! adjacent.
//!
//! If that is the mechanism, padding the row while **holding the field count fixed** should
//! make it worse in proportion — wider rows, fewer useful bytes per line, more lines touched
//! for the same logical work. Flat would mean something else is responsible and we need to
//! know before Phase 7.
//!
//! One axis: row width. Same three fields read, same predicate, same grouping, same data.
//! Padding is never touched by the loop; it exists only to push the useful bytes apart.

use batchbird::bench::data::{AMOUNT_RANGE, Rng, SEED, environment};
use batchbird::bench::harness::{Arm, alternate};
use batchbird::hash::{Map, map_with_capacity};

const ROWS: usize = 2_000_000;
const CARDINALITY: u32 = 128;
const SAMPLES: usize = 25;
const WARMUP: usize = 3;

/// One row, with `PAD` bytes of dead weight after the fields the loop actually reads.
///
/// `repr(C)` so the padding cannot be reordered away, and the fields stay where they are put.
#[repr(C)]
#[derive(Clone, Copy)]
struct Row<const PAD: usize> {
    group: u32,
    filter: i64,
    value: i64,
    _pad: [u8; PAD],
}

/// The row-oriented loop: filter, group, accumulate — reading three fields of each struct.
fn row_scan<const PAD: usize>(rows: &[Row<PAD>], threshold: i64) -> u64 {
    let mut slots: Map<u32, usize> = map_with_capacity(CARDINALITY as usize);
    let mut totals: Vec<i64> = Vec::new();

    for row in rows {
        if row.filter <= threshold {
            continue;
        }
        let slot = match slots.get(&row.group) {
            Some(slot) => *slot,
            None => {
                let slot = totals.len();
                slots.insert(row.group, slot);
                totals.push(0);
                slot
            }
        };
        totals[slot] = totals[slot].wrapping_add(row.value);
    }

    totals.iter().fold(totals.len() as u64, |acc, t| {
        acc.wrapping_mul(31).wrapping_add(*t as u64)
    })
}

/// The columnar equivalent: identical logic, three separate arrays.
fn columnar_scan(groups: &[u32], filters: &[i64], values: &[i64], threshold: i64) -> u64 {
    let mut slots: Map<u32, usize> = map_with_capacity(CARDINALITY as usize);
    let mut totals: Vec<i64> = Vec::new();

    for row in 0..filters.len() {
        if filters[row] <= threshold {
            continue;
        }
        let slot = match slots.get(&groups[row]) {
            Some(slot) => *slot,
            None => {
                let slot = totals.len();
                slots.insert(groups[row], slot);
                totals.push(0);
                slot
            }
        };
        totals[slot] = totals[slot].wrapping_add(values[row]);
    }

    totals.iter().fold(totals.len() as u64, |acc, t| {
        acc.wrapping_mul(31).wrapping_add(*t as u64)
    })
}

fn build<const PAD: usize>() -> (Vec<Row<PAD>>, Vec<u32>, Vec<i64>, Vec<i64>) {
    let mut rng = Rng::new(SEED);
    let mut rows = Vec::with_capacity(ROWS);
    let (mut groups, mut filters, mut values) = (
        Vec::with_capacity(ROWS),
        Vec::with_capacity(ROWS),
        Vec::with_capacity(ROWS),
    );

    for _ in 0..ROWS {
        let group = (rng.next_u64() % CARDINALITY as u64) as u32;
        let filter = (rng.next_u64() % AMOUNT_RANGE) as i64;
        let value = (rng.next_u64() % 1_000) as i64;

        rows.push(Row {
            group,
            filter,
            value,
            _pad: [0u8; PAD],
        });
        groups.push(group);
        filters.push(filter);
        values.push(value);
    }

    (rows, groups, filters, values)
}

/// Measure one padding width against the columnar baseline, alternated.
fn probe<const PAD: usize>(threshold: i64) -> (usize, f64, f64) {
    let (rows, groups, filters, values) = build::<PAD>();
    let width = std::mem::size_of::<Row<PAD>>();

    let arms = vec![
        Arm::new("row", || row_scan(&rows, threshold)),
        Arm::new("columnar", || {
            columnar_scan(&groups, &filters, &values, threshold)
        }),
    ];

    let m = alternate(arms, SAMPLES, WARMUP);
    assert_eq!(m[0].checksum, m[1].checksum, "the two layouts disagree");

    let (row_ns, col_ns) = (
        m[0].min() * 1e6 / ROWS as f64,
        m[1].min() * 1e6 / ROWS as f64,
    );
    println!(
        "  {width:>4} B/row  {:>6} lines/1k rows   row {row_ns:6.2} ns   columnar {col_ns:6.2} ns   \
         gap {:.2}x",
        (width * 1000).div_ceil(64),
        row_ns / col_ns,
    );
    (width, row_ns, row_ns / col_ns)
}

fn main() {
    println!("{}\n", environment());
    println!(
        "Layout probe: {ROWS} rows, cardinality {CARDINALITY}, 50% selectivity.\n\
         Three fields read in every case; only the dead padding between rows changes.\n\
         The columnar arm is identical across rows -- it has no padding to grow.\n"
    );

    let threshold = (AMOUNT_RANGE / 2) as i64;
    println!("  width      cache lines touched per 1000 rows scanned");

    let results = [
        probe::<0>(threshold),
        probe::<8>(threshold),
        probe::<24>(threshold),
        probe::<56>(threshold),
        probe::<120>(threshold),
    ];

    println!("\n=== reading the result ===");
    let (narrow_width, narrow_ns, narrow_gap) = results[0];
    let (wide_width, wide_ns, wide_gap) = results[results.len() - 1];
    let width_ratio = wide_width as f64 / narrow_width as f64;
    let cost_ratio = wide_ns / narrow_ns;

    println!(
        "row width grew {width_ratio:.1}x ({narrow_width} -> {wide_width} B); \
         row-arm cost grew {cost_ratio:.2}x"
    );
    println!("layout gap went {narrow_gap:.2}x -> {wide_gap:.2}x");
    println!(
        "\nIf cost tracks width, cache-line utilization is the mechanism: the loop reads the\n\
         same three fields either way, so only the bytes dragged along with them changed.\n\
         If cost is flat, width is not what the row arm is paying for, and the gap needs\n\
         another explanation before Phase 7 can claim one."
    );
}
