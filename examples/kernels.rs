//! Does vectorizing the kernels actually do anything?
//!
//! Times each scalar reference kernel against whatever the dispatcher picks, so running it
//! twice answers the question directly:
//!
//! ```text
//! cargo run --release --example kernels                            # dispatcher = scalar
//! cargo +nightly run --release --example kernels --features simd   # dispatcher = SIMD
//! ```
//!
//! Indicative only -- best-of-N wall clock, not `criterion`. Phase 6 does this properly, with
//! the group-by phases separated. The point here is a sanity check before declaring Phase 5
//! done: an "optimization" that turns out slower is worth discovering now.

use std::time::Instant;

use batchbird::exec::kernels;
use batchbird::plan::CompareOp;

#[cfg(feature = "simd")]
const MODE: &str = "SIMD (nightly, --features simd)";
#[cfg(not(feature = "simd"))]
const MODE: &str = "scalar (the dispatcher falls back)";

const ROWS: usize = 16_000_000;

fn time<T, F: FnMut() -> T>(label: &str, mut f: F) -> f64 {
    f();
    let mut best = f64::MAX;
    for _ in 0..5 {
        let start = Instant::now();
        let out = f();
        best = best.min(start.elapsed().as_secs_f64() * 1000.0);
        std::hint::black_box(out);
    }
    println!("  {label:<22} {best:>8.2} ms");
    best
}

fn main() {
    println!("{ROWS} rows, best of 5\ndispatcher resolves to: {MODE}\n");

    let ints: Vec<i64> = (0..ROWS as i64).map(|i| (i * 7) % 1000 - 500).collect();
    let floats: Vec<f64> = (0..ROWS).map(|i| (i as f64) * 0.37 - 500.0).collect();

    println!("filter compare -> bitmask (Int64):");
    let s = time("scalar", || {
        kernels::mask_i64_scalar(&ints, CompareOp::Gt, 0)
    });
    let d = time("dispatched", || kernels::mask_i64(&ints, CompareOp::Gt, 0));
    println!("  {:<22} {:>8.2}x\n", "speedup", s / d);

    println!("filter compare -> bitmask (Float64):");
    let s = time("scalar", || {
        kernels::mask_f64_scalar(&floats, CompareOp::Gt, 0.0)
    });
    let d = time("dispatched", || {
        kernels::mask_f64(&floats, CompareOp::Gt, 0.0)
    });
    println!("  {:<22} {:>8.2}x\n", "speedup", s / d);

    println!("sum reduction (Int64):");
    let s = time("scalar", || kernels::sum_i64_scalar(&ints));
    let d = time("dispatched", || kernels::sum_i64(&ints));
    println!("  {:<22} {:>8.2}x\n", "speedup", s / d);

    println!("sum reduction (Float64):");
    let s = time("scalar", || kernels::sum_f64_scalar(&floats));
    let d = time("dispatched", || kernels::sum_f64(&floats));
    println!("  {:<22} {:>8.2}x\n", "speedup", s / d);
}
