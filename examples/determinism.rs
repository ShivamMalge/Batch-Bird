//! Diagnostic: is anything about a benchmark run non-reproducible *inside the program*?
//!
//! Run it twice and diff. Two independent questions:
//!
//! 1. **Is the generated dataset byte-identical across processes?** If not, benchmark variance
//!    is a seeding bug rather than machine noise, and no amount of pinning or cooling fixes it.
//! 2. **Are the hash maps seeded per process?** `std::collections::HashMap` and
//!    `hashbrown::HashMap` both default to a randomly-seeded hasher. That changes probe
//!    sequences, collision distribution, and iteration order on every run — which changes both
//!    group-by cost and the order columns are allocated during compaction.
//!
//! ```text
//! cargo run --release --example determinism > a.txt
//! cargo run --release --example determinism > b.txt
//! diff a.txt b.txt
//! ```

use batchbird::bench::data::{SEED, generate};
use batchbird::hash;
use batchbird::storage::Column;

/// FNV-1a. Not cryptographic — just needs to be stable across processes, which is exactly the
/// property under test, so it cannot itself use a randomly-seeded hasher.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    hash
}

fn hash_column(column: &Column) -> u64 {
    match column {
        Column::Int64(v) => fnv1a(unsafe {
            std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(&v[..]))
        }),
        Column::Float64(v) => fnv1a(unsafe {
            std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(&v[..]))
        }),
        Column::Utf8Dict { dict, codes } => {
            let mut hash = fnv1a(unsafe {
                std::slice::from_raw_parts(
                    codes.as_ptr() as *const u8,
                    std::mem::size_of_val(&codes[..]),
                )
            });
            for entry in dict.iter() {
                hash ^= fnv1a(entry.as_bytes());
            }
            hash
        }
    }
}

fn main() {
    println!("=== 1. dataset reproducibility ===");
    for (rows, cardinality) in [(100_000usize, 8u64), (1_000_000, 2_048)] {
        let table = generate(rows, cardinality, SEED);
        // Sorted, so the report itself does not depend on map iteration order.
        let mut names: Vec<&str> = table.column_names().collect();
        names.sort();
        for name in names {
            let column = table.column(name).expect("named column");
            println!(
                "rows={rows:<9} card={cardinality:<6} {name:<8} {:016x}",
                hash_column(column)
            );
        }
    }

    println!("\n=== 2. hash-map seeding ===");
    // Same keys, same insertion order, every run. If the printed order differs between
    // processes, the hasher is randomly seeded and probe sequences differ run to run.
    let keys: Vec<String> = (0..12).map(|i| format!("r{i}")).collect();

    let mut fixed: hash::Map<&str, usize> = hash::map();
    let mut random: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for (i, key) in keys.iter().enumerate() {
        fixed.insert(key, i);
        random.insert(key, i);
    }

    let order = |iter: Vec<&str>| iter.join(",");
    // The engine's map: must be identical on every run.
    println!(
        "engine (fixed seed): {}",
        order(fixed.keys().copied().collect())
    );
    // A stdlib map, kept only to show the default this project deliberately does not use.
    // Expected to differ between runs -- that is the behaviour `crate::hash` exists to avoid.
    println!(
        "stdlib (random seed): {}",
        order(random.keys().copied().collect())
    );
}
