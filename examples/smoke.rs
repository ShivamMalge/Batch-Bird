//! Scaling check: does the batch pipeline hold up as group cardinality grows?
//!
//! Groups the same data by a `Utf8` column and by an `Int64` column of identical cardinality,
//! so the slot count and scatter pattern match and the only variable is whether the group
//! column carries a dictionary.
//!
//! This found a real bug. When `Column::Utf8Dict` owned its dictionary, every batch cloned it
//! twice and the Utf8 path degraded from 0.80x to **84x** slower than the naive baseline as
//! cardinality went 8 -> 10,000, while the Int64 path stayed flat at ~1.1x. Sharing the
//! dictionary behind an `Arc` fixed it; both paths now sit at ~0.70x everywhere. Keep this
//! around as a regression guard -- the failure mode is invisible at low cardinality, which is
//! exactly where a casual benchmark would look.
//!
//! Indicative timings only; `criterion` benchmarks arrive in Phase 6.
//!     cargo run --release --example smoke

use std::time::Instant;

use batchbird::bench::naive_query;
use batchbird::parser::parse;
use batchbird::plan::batch_query;
use batchbird::storage::{DataType, Field, Schema, Table, read_csv};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

fn build(rows: usize, cardinality: u64) -> Table {
    let mut rng = Rng(0x5EED);
    let mut csv = String::from("region,gid,amount,price\n");
    for _ in 0..rows {
        let r = rng.next() % cardinality;
        let a = rng.next() % 2000;
        let p = (rng.next() % 100_000) as f64 / 100.0;
        csv.push_str(&format!("r{r},{r},{a},{p}\n"));
    }
    let schema = Schema::new(vec![
        Field::new("region", DataType::Utf8),
        Field::new("gid", DataType::Int64),
        Field::new("amount", DataType::Int64),
        Field::new("price", DataType::Float64),
    ]);
    read_csv(csv.as_bytes(), &schema).unwrap()
}

fn time<F: FnMut() -> usize>(label: &str, mut f: F) -> f64 {
    f();
    let mut best = f64::MAX;
    for _ in 0..3 {
        let t = Instant::now();
        let groups = f();
        best = best.min(t.elapsed().as_secs_f64() * 1000.0);
        std::hint::black_box(groups);
    }
    println!("  {label:<24} {best:>9.2} ms");
    best
}

fn main() {
    let rows = 2_000_000;
    let utf8 =
        parse("SELECT region, SUM(amount) FROM t WHERE amount > 500 GROUP BY region").unwrap();
    let int = parse("SELECT gid, SUM(amount) FROM t WHERE amount > 500 GROUP BY gid").unwrap();

    println!("{rows} rows, WHERE amount > 500, best of 3\n");
    for cardinality in [8u64, 100, 1_000, 10_000] {
        let table = build(rows, cardinality);
        println!("cardinality {cardinality}:");
        let n = time("naive,   Utf8 group", || {
            naive_query(&table, &utf8).unwrap().nrows()
        });
        let b = time("batched, Utf8 group", || {
            batch_query(&table, &utf8).unwrap().nrows()
        });
        let ni = time("naive,   Int64 group", || {
            naive_query(&table, &int).unwrap().nrows()
        });
        let bi = time("batched, Int64 group", || {
            batch_query(&table, &int).unwrap().nrows()
        });
        println!("  {:<24} {:>9.2}x", "Utf8:  batched/naive", b / n);
        println!("  {:<24} {:>9.2}x\n", "Int64: batched/naive", bi / ni);
    }
}
