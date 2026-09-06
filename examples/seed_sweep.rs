//! Item 3: how much bias does one fixed hasher seed buy us?
//!
//! Fixing the seed traded variance for bias. Variance is gone — every run now does identical
//! work — but the seed we happened to pick is one particular collision pattern, and it could
//! be luckier or unluckier than typical. This measures that spread so it is a reported number
//! rather than a hidden assumption.
//!
//! Run at the **highest** cardinality, where the group table is largest and collision
//! behaviour matters most. Low cardinality would understate it: eight keys in a table sized
//! for eight barely collide whatever the seed.
//!
//! ```text
//! cargo run --release --example seed_sweep
//! ```
//!
//! Reports the minimum of N, not the mean: interference is one-sided, so the minimum is the
//! observation closest to the machine's real capability.

use std::time::Instant;

use batchbird::bench::data::{SEED, amount_threshold, generate};
use batchbird::exec::GroupKey;
use batchbird::hash::{DEFAULT_SEED, FixedState, Map};

/// The group-by inner loop, parameterized by hasher seed.
///
/// Deliberately a replica of `Aggregate`'s two phases rather than a call into it: `Aggregate`
/// is hard-wired to the default seed, and making it generic over `BuildHasher` purely to run
/// this sweep would put a type parameter on the engine to serve a diagnostic.
fn group_by_with_seed(codes: &[u32], values: &[i64], seed: u64) -> (usize, i64) {
    let mut index: Map<GroupKey, usize> = Map::with_hasher(FixedState::with_seed(seed));
    let mut totals: Vec<i64> = Vec::new();

    for (code, value) in codes.iter().zip(values) {
        let key = GroupKey(*code as u64);
        let slot = match index.get(&key) {
            Some(slot) => *slot,
            None => {
                let slot = totals.len();
                index.insert(key, slot);
                totals.push(0);
                slot
            }
        };
        totals[slot] = totals[slot].wrapping_add(*value);
    }

    // Returned so the work cannot be optimized away, and as a correctness check: every seed
    // must produce the same groups and the same sum.
    (
        totals.len(),
        totals.iter().fold(0i64, |a, b| a.wrapping_add(*b)),
    )
}

fn main() {
    const ROWS: usize = 1_000_000;
    const CARDINALITY: u64 = 250_000;
    const REPEATS: usize = 7;

    let table = generate(ROWS, CARDINALITY, SEED);
    let threshold = amount_threshold(0.5);

    // Pre-filter once, outside timing, so this measures grouping and nothing else.
    let (_, all_codes) = table.column("region").unwrap().as_utf8_dict().unwrap();
    let all_values = table.column("amount").unwrap().as_i64().unwrap();
    let (codes, values): (Vec<u32>, Vec<i64>) = all_codes
        .iter()
        .zip(all_values)
        .filter(|(_, v)| **v > threshold)
        .map(|(c, v)| (*c, *v))
        .unzip();

    println!(
        "{} rows surviving, cardinality {CARDINALITY}, min of {REPEATS}\n",
        codes.len()
    );

    // The default first, then four arbitrary others.
    let seeds = [
        DEFAULT_SEED,
        1,
        0x9E37_79B9_7F4A_7C15,
        0xDEAD_BEEF,
        u64::MAX,
    ];
    let mut timings = Vec::new();
    let mut expected: Option<(usize, i64)> = None;

    for seed in seeds {
        let mut best = f64::MAX;
        for _ in 0..REPEATS {
            let start = Instant::now();
            let result = group_by_with_seed(&codes, &values, seed);
            best = best.min(start.elapsed().as_secs_f64() * 1000.0);
            std::hint::black_box(result);

            match expected {
                None => expected = Some(result),
                Some(e) => assert_eq!(e, result, "seed {seed:#x} changed the answer"),
            }
        }
        let label = if seed == DEFAULT_SEED {
            " <- default"
        } else {
            ""
        };
        println!("  seed {seed:#018x}  {best:8.3} ms{label}");
        timings.push((seed, best));
    }

    let times: Vec<f64> = timings.iter().map(|(_, t)| *t).collect();
    let fastest = times.iter().cloned().fold(f64::MAX, f64::min);
    let slowest = times.iter().cloned().fold(0.0, f64::max);
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let default = timings[0].1;

    println!(
        "\nspread across seeds: {:.1}%",
        (slowest / fastest - 1.0) * 100.0
    );
    println!(
        "default vs mean:     {:+.1}%",
        (default / mean - 1.0) * 100.0
    );
    println!(
        "\nThe first number is how much the choice of seed can matter at all; the second is the\n\
         bias our particular seed introduces. A small second number means fixing the seed cost\n\
         us accuracy we can ignore. A large one means collision pattern is a real variable and\n\
         belongs in the write-up rather than buried in a constant."
    );
}
