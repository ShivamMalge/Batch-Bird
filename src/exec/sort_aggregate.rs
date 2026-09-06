//! `SortAggregate` -- sort by group key, then run-length aggregate.
//!
//! The second group-by strategy (`systemDesign.md` "Sort-Based Grouping"), alongside — never
//! replacing — the hash path. Both are selectable at the physical-plan level so the benchmark
//! can drive either over identical input.
//!
//! # Why bother, when hash group-by is O(n)
//! Because of what sorting does to *memory access*. Hash group-by's second phase scatters:
//! `accumulators[slot] += value` with a data-dependent slot, which no amount of vectorization
//! helps. Sorting makes every group's values contiguous, so aggregation becomes a sequential
//! scan over dense slices — and a dense slice is exactly what the Phase 5 SIMD sum kernel
//! wants. This is the only path in the engine through which that kernel reaches a real query.
//!
//! The bet is that a vectorizable O(n) aggregation phase can outrun a scatter-bound one by
//! enough to pay for an O(n log n) sort. `systemDesign.md` predicts it will not, at this
//! scale. Phase 6 measures rather than assumes.
//!
//! # Three decisions, recorded
//!
//! **What gets sorted: `(key, value)` pairs.** The alternatives both leave the payload
//! scattered, which defeats the entire point:
//! - *Sorting an index permutation* moves 8 bytes per swap instead of 16, but aggregation then
//!   reads values through the permutation — random access, no contiguous run, nothing for the
//!   sum kernel to eat.
//! - *Sorting keys and gathering payload afterwards* has the same problem one step later.
//!
//! Sorting pairs costs more memory traffic during the sort and buys dense runs after it. Since
//! dense runs are the reason this strategy exists, that is the right side of the trade.
//!
//! One consequence: after sorting, the pairs are split into parallel `keys` and `values`
//! arrays, because a `&[(GroupKey, T)]` has the values *strided* by the key — a sum kernel
//! cannot take a slice of it. That split is one extra sequential pass and one extra allocation,
//! and it is a real cost sort-group pays that hash-group does not.
//!
//! **Which sort: `sort_unstable_by_key`, a comparison sort — and this is a scope cut, not the
//! better algorithm.** Keys are `u64`, and for `u64` keys an LSD radix sort is O(n · passes)
//! rather than O(n log n); for dictionary codes, whose range is `0..cardinality`, a counting
//! sort is O(n + k) and nearly trivial. Either would likely flip the headline comparison,
//! because `systemDesign.md`'s prediction that sort-group loses rests specifically on the
//! `O(n log n)` term this implementation keeps. Reported as a limitation of the measurement,
//! not as evidence that sorting cannot win.
//!
//! **Memory: O(rows), not O(groups).** Hash group-by holds one accumulator per group; this
//! holds every surviving row until the input is exhausted. At high selectivity that is the
//! whole dataset. A genuine cost of the strategy, and one a benchmark measuring only time
//! would miss entirely.

use std::sync::Arc;

use crate::exec::Operator;
use crate::exec::aggregate::{GroupKey, GroupKind, Summable};
use crate::exec::batch::RecordBatch;
use crate::storage::Column;

pub struct SortAggregate<I, T> {
    input: I,
    group_column: String,
    group_kind: GroupKind,
    value_column: String,
    output_column: String,

    /// Every surviving row, keyed. The O(rows) memory this strategy trades for dense runs.
    pairs: Vec<(GroupKey, T)>,

    /// Filled by [`sort_pairs`](Self::sort_pairs): the same data, sorted and split so each
    /// run's values are a contiguous slice.
    sorted_keys: Vec<GroupKey>,
    sorted_values: Vec<T>,

    /// Ping-pong buffer for the radix sort. Held on the struct so repeated sorts reuse one
    /// allocation rather than making the allocator part of what is being measured.
    scratch: Vec<(GroupKey, T)>,

    dictionary: Arc<[String]>,
    drained: bool,

    /// Which sort `finish` runs. A benchmark axis; the engine's default is `Comparison`.
    algorithm: SortAlgorithm,
}

impl<I: Operator, T: Summable> SortAggregate<I, T> {
    pub fn new(
        input: I,
        group_columns: Vec<String>,
        group_kind: GroupKind,
        value_column: String,
        output_column: String,
    ) -> Self {
        assert_eq!(
            group_columns.len(),
            1,
            "only single-column GROUP BY is supported; GroupKey is a single-value newtype"
        );

        SortAggregate {
            input,
            group_column: group_columns.into_iter().next().expect("length checked"),
            group_kind,
            value_column,
            output_column,
            pairs: Vec::new(),
            sorted_keys: Vec::new(),
            sorted_values: Vec::new(),
            scratch: Vec::new(),
            dictionary: Vec::new().into(),
            drained: false,
            algorithm: SortAlgorithm::default(),
        }
    }

    /// Choose the sort. Only the benchmark calls this; see [`SortAlgorithm`].
    pub fn with_algorithm(mut self, algorithm: SortAlgorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// **Phase 1.** Append this batch's `(key, value)` pairs.
    ///
    /// Sequential reads, sequential appends, no hashing at all — the phase where sort-group
    /// is cheaper than hash-group, which pays a probe per row here.
    pub fn collect_batch(&mut self, batch: &RecordBatch) {
        let group = batch
            .column(&self.group_column)
            .expect("group column is validated when the pipeline is built");
        let value_column = batch
            .column(&self.value_column)
            .expect("value column is validated when the pipeline is built");
        let values = T::slice_of(value_column).expect("value type is validated at build time");

        self.pairs.reserve(batch.len());

        match group {
            Column::Int64(keys) => {
                for (key, value) in keys.iter().zip(values) {
                    self.pairs.push((GroupKey(*key as u64), *value));
                }
            }
            Column::Utf8Dict { dict, codes } => {
                if self.dictionary.is_empty() {
                    self.dictionary = Arc::clone(dict);
                }
                for (code, value) in codes.iter().zip(values) {
                    self.pairs.push((GroupKey(*code as u64), *value));
                }
            }
            Column::Float64(_) => {
                unreachable!("float group keys are rejected when the pipeline is built")
            }
        }
    }

    /// **Phase 2.** Sort by key, then split into parallel dense arrays.
    ///
    /// Public so the benchmark can time it apart from aggregation — this is the phase expected
    /// to dominate, and reporting one combined number would hide that.
    ///
    /// The split is what makes phase 3 vectorizable; see the module docs.
    pub fn sort_pairs(&mut self) {
        self.sort_pairs_with(SortAlgorithm::default());
    }

    /// Sort with an explicitly chosen algorithm.
    ///
    /// Exists so the benchmark can answer a question the default alone cannot: is "sort-group
    /// loses" a statement about *sort-based grouping*, or about *pdqsort on 16-byte pairs*?
    /// Those are different claims and only one of them is interesting.
    pub fn sort_pairs_with(&mut self, algorithm: SortAlgorithm) {
        match algorithm {
            SortAlgorithm::Comparison => self.pairs.sort_unstable_by_key(|(key, _)| key.0),
            SortAlgorithm::Radix => radix_sort_pairs(&mut self.pairs, &mut self.scratch),
        }

        self.sorted_keys.clear();
        self.sorted_values.clear();
        self.sorted_keys.reserve(self.pairs.len());
        self.sorted_values.reserve(self.pairs.len());

        for (key, value) in &self.pairs {
            self.sorted_keys.push(*key);
            self.sorted_values.push(*value);
        }
    }

    /// **Phase 3.** Reduce each run of equal keys.
    ///
    /// The payoff: every run is a contiguous `&[T]`, so `T::sum_slice` dispatches to the SIMD
    /// kernel. How much that is worth depends entirely on run length — at low cardinality runs
    /// are long and vectorize well, and as cardinality rises toward one row per group the runs
    /// shrink until per-call overhead outweighs the vector work. The cardinality where that
    /// crossover happens is a measurement, not a prediction.
    ///
    /// Takes `&self` so the benchmark can time it repeatedly without re-sorting.
    pub fn aggregate_runs(&self) -> (Vec<GroupKey>, Vec<T>) {
        // Read once for the whole aggregation, not once per run. Without the
        // `bench-dispatch` feature this folds to a constant and the match below disappears.
        let kernel = crate::exec::kernels::active_kernel();

        let mut keys = Vec::new();
        let mut totals = Vec::new();

        let mut start = 0;
        while start < self.sorted_keys.len() {
            let key = self.sorted_keys[start];
            let mut end = start + 1;
            while end < self.sorted_keys.len() && self.sorted_keys[end] == key {
                end += 1;
            }

            keys.push(key);
            totals.push(T::sum_slice_with(kernel, &self.sorted_values[start..end]));
            start = end;
        }

        (keys, totals)
    }

    /// How many rows are being held. Exposed so the benchmark can report the O(rows) memory
    /// this strategy costs, rather than only its time.
    pub fn buffered_rows(&self) -> usize {
        self.pairs.len()
    }

    fn finish(&mut self) -> RecordBatch {
        self.sort_pairs_with(self.algorithm);
        let (keys, totals) = self.aggregate_runs();

        let group_column = match self.group_kind {
            GroupKind::Int64 => Column::Int64(keys.iter().map(|k| k.0 as i64).collect()),
            GroupKind::Utf8 => Column::Utf8Dict {
                dict: keys
                    .iter()
                    .map(|k| self.dictionary[k.0 as usize].clone())
                    .collect::<Vec<_>>()
                    .into(),
                codes: (0..keys.len() as u32).collect(),
            },
        };

        let mut columns = crate::hash::map_with_capacity(2);
        columns.insert(self.group_column.clone(), group_column);
        columns.insert(self.output_column.clone(), T::into_column(totals));

        RecordBatch::new(columns, keys.len())
    }
}

impl<I: Operator, T: Summable> Operator for SortAggregate<I, T> {
    fn next_batch(&mut self) -> Option<RecordBatch> {
        if self.drained {
            return None;
        }

        while let Some(batch) = self.input.next_batch() {
            self.collect_batch(&batch);
        }
        self.drained = true;

        Some(self.finish())
    }
}

/// Which sort to use inside [`SortAggregate::sort_pairs_with`].
///
/// A benchmark axis, not a query option. The engine always uses the default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SortAlgorithm {
    /// `sort_unstable_by_key`, Rust's pdqsort. O(n log n) comparisons.
    #[default]
    Comparison,
    /// LSD radix, O(n * passes) with the pass count adapted to the largest key present.
    Radix,
}

/// Least-significant-digit radix sort on the `u64` key, 8 bits per pass.
///
/// # Why this is worth having
/// The interesting claim is about *sort-based grouping*, and a comparison sort would have let
/// `O(n log n)` stand in for it. Group keys here are `u64` — dictionary codes or bit-cast
/// `i64` — which is exactly the shape radix handles in `O(n)`. Without this, "sort-group
/// loses" could not be separated from "pdqsort on 16-byte pairs loses".
///
/// # Passes are adapted to the data
/// Only bytes up to the largest key present are processed. Dictionary codes for 250k groups
/// occupy three bytes, so this makes three passes rather than eight — the difference between
/// competitive and hopeless, and it costs one scan to discover.
///
/// Stable by construction, which the run-length aggregation downstream does not require but
/// which makes the result identical to the comparison sort's on equal keys.
fn radix_sort_pairs<T: Copy>(pairs: &mut Vec<(GroupKey, T)>, scratch: &mut Vec<(GroupKey, T)>) {
    if pairs.len() < 2 {
        return;
    }

    let max_key = pairs.iter().map(|(key, _)| key.0).max().unwrap_or(0);
    let passes = if max_key == 0 {
        1
    } else {
        (64 - max_key.leading_zeros()).div_ceil(8) as usize
    };

    scratch.clear();
    scratch.resize(pairs.len(), pairs[0]);

    let mut counts = [0usize; 256];
    for pass in 0..passes {
        let shift = pass * 8;

        counts.fill(0);
        for (key, _) in pairs.iter() {
            counts[((key.0 >> shift) & 0xFF) as usize] += 1;
        }

        // A pass whose digit is constant would only copy the data back and forth.
        if counts.contains(&pairs.len()) {
            continue;
        }

        let mut offset = 0;
        for count in counts.iter_mut() {
            let current = *count;
            *count = offset;
            offset += current;
        }

        for pair in pairs.iter() {
            let digit = ((pair.0.0 >> shift) & 0xFF) as usize;
            scratch[counts[digit]] = *pair;
            counts[digit] += 1;
        }

        std::mem::swap(pairs, scratch);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::scan::Scan;
    use crate::storage::{Table, infer_schema, read_csv};

    const ROWS: &str = concat!(
        "region,amount,price\n",
        "north,10,1.5\n",
        "south,-20,2.25\n",
        "north,30,3.0\n",
        "east,40,4.75\n",
        "north,50,5.5\n",
    );

    fn table(csv: &str) -> Table {
        let schema = infer_schema(csv.as_bytes()).unwrap();
        read_csv(csv.as_bytes(), &schema).unwrap()
    }

    fn pairs_of(batch: &RecordBatch, group: &str, agg: &str) -> Vec<(String, String)> {
        let group_col = batch.column(group).expect("group column");
        let agg_col = batch.column(agg).expect("agg column");

        let mut rows: Vec<(String, String)> = (0..batch.len())
            .map(|row| {
                let key = match group_col {
                    Column::Int64(v) => v[row].to_string(),
                    Column::Utf8Dict { .. } => group_col.utf8_value(row).unwrap().to_string(),
                    Column::Float64(_) => unreachable!(),
                };
                let sum = match agg_col {
                    Column::Int64(v) => v[row].to_string(),
                    Column::Float64(v) => v[row].to_string(),
                    Column::Utf8Dict { .. } => unreachable!(),
                };
                (key, sum)
            })
            .collect();
        rows.sort();
        rows
    }

    fn owned(rows: Vec<(&str, &str)>) -> Vec<(String, String)> {
        rows.into_iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    fn sort_agg<'a>(
        table: &'a Table,
        group: &str,
        kind: GroupKind,
        value: &str,
        batch_size: usize,
    ) -> SortAggregate<Scan<'a>, i64> {
        let scan = Scan::with_batch_size(
            table,
            vec![group.to_string(), value.to_string()],
            batch_size,
        );
        SortAggregate::new(
            scan,
            vec![group.to_string()],
            kind,
            value.to_string(),
            format!("SUM({value})"),
        )
    }

    #[test]
    fn groups_by_string_and_sums() {
        let table = table(ROWS);
        let mut agg = sort_agg(&table, "region", GroupKind::Utf8, "amount", 1024);
        let result = agg.next_batch().unwrap();

        assert_eq!(
            pairs_of(&result, "region", "SUM(amount)"),
            owned(vec![("east", "40"), ("north", "90"), ("south", "-20")])
        );
    }

    #[test]
    fn groups_by_integer_including_negatives() {
        // Keys are bit-cast to u64 before sorting, so negatives sort *above* positives. That
        // is fine and is exactly why the cast is documented as equality-preserving only:
        // grouping needs equal keys adjacent, not keys in numeric order.
        let table = table("k,v\n-1,10\n2,20\n-1,30\n2,40\n");
        let mut agg = sort_agg(&table, "k", GroupKind::Int64, "v", 1024);
        let result = agg.next_batch().unwrap();

        assert_eq!(
            pairs_of(&result, "k", "SUM(v)"),
            owned(vec![("-1", "40"), ("2", "60")])
        );
    }

    #[test]
    fn accumulates_across_batch_boundaries() {
        let table = table(ROWS);
        for size in [1, 2, 3, 4, 5, 1024] {
            let mut agg = sort_agg(&table, "region", GroupKind::Utf8, "amount", size);
            let result = agg.next_batch().unwrap();
            assert_eq!(
                pairs_of(&result, "region", "SUM(amount)"),
                owned(vec![("east", "40"), ("north", "90"), ("south", "-20")]),
                "batch size {size}"
            );
        }
    }

    #[test]
    fn the_phases_can_be_driven_separately() {
        // What the benchmark needs: sort timed apart from aggregation, since the sort is
        // expected to dominate and a combined number would hide that.
        let table = table(ROWS);
        let mut agg = sort_agg(&table, "region", GroupKind::Utf8, "amount", 1024);

        let mut scan = Scan::with_batch_size(
            &table,
            vec!["region".to_string(), "amount".to_string()],
            1024,
        );
        while let Some(batch) = scan.next_batch() {
            agg.collect_batch(&batch);
        }
        assert_eq!(
            agg.buffered_rows(),
            5,
            "every row is buffered, not just groups"
        );

        agg.sort_pairs();
        // north(code 0) x3, south(1), east(2) -- sorted by code, so runs are contiguous.
        assert_eq!(
            agg.sorted_keys.iter().map(|k| k.0).collect::<Vec<_>>(),
            vec![0, 0, 0, 1, 2]
        );

        let (keys, totals) = agg.aggregate_runs();
        assert_eq!(keys.len(), 3);
        assert_eq!(totals, vec![90, -20, 40]);
    }

    #[test]
    fn aggregate_runs_is_repeatable() {
        // Takes &self so the benchmark can time it without re-sorting each iteration.
        let table = table(ROWS);
        let mut agg = sort_agg(&table, "region", GroupKind::Utf8, "amount", 1024);
        let mut scan = Scan::with_batch_size(
            &table,
            vec!["region".to_string(), "amount".to_string()],
            1024,
        );
        while let Some(batch) = scan.next_batch() {
            agg.collect_batch(&batch);
        }
        agg.sort_pairs();

        assert_eq!(agg.aggregate_runs(), agg.aggregate_runs());
    }

    #[test]
    fn is_blocking_and_yields_exactly_one_batch() {
        let table = table(ROWS);
        let mut agg = sort_agg(&table, "region", GroupKind::Utf8, "amount", 2);
        assert!(agg.next_batch().is_some());
        assert!(agg.next_batch().is_none());
    }

    #[test]
    fn an_empty_input_yields_a_correctly_typed_empty_result() {
        let table = table("region,amount\n");
        let scan = Scan::new(&table, vec!["region".to_string(), "amount".to_string()]);
        let mut agg: SortAggregate<_, i64> = SortAggregate::new(
            scan,
            vec!["region".to_string()],
            GroupKind::Utf8,
            "amount".to_string(),
            "SUM(amount)".to_string(),
        );

        let result = agg.next_batch().unwrap();
        assert_eq!(result.len(), 0);
        assert_eq!(result.ncols(), 2);
    }

    #[test]
    fn every_row_its_own_group_still_works() {
        // The degenerate case for this strategy: runs of length one, so the sum kernel is
        // called once per row and vectorization has nothing to chew on.
        let mut csv = String::from("k,v\n");
        for i in 0..50 {
            csv.push_str(&format!("{i},{}\n", i * 2));
        }
        let table = table(&csv);
        let mut agg = sort_agg(&table, "k", GroupKind::Int64, "v", 8);
        let result = agg.next_batch().unwrap();

        assert_eq!(result.len(), 50);
        let sums: i64 = result
            .column("SUM(v)")
            .unwrap()
            .as_i64()
            .unwrap()
            .iter()
            .sum();
        assert_eq!(sums, (0..50i64).map(|i| i * 2).sum::<i64>());
    }

    #[test]
    fn radix_and_comparison_sorts_agree() {
        // The radix path is new correctness surface, and its adaptive pass count means the
        // number of passes depends on the data -- so the cases below span one byte, three
        // bytes, and the full eight.
        for max_key in [0u64, 1, 255, 256, 250_000, u64::MAX] {
            for len in [0usize, 1, 2, 17, 1000] {
                let mut rng = crate::bench::data::Rng::new(0xC0FFEE ^ max_key ^ len as u64);
                let pairs: Vec<(GroupKey, i64)> = (0..len)
                    .map(|i| {
                        let key = if max_key == 0 {
                            0
                        } else {
                            rng.next_u64() % max_key.max(1)
                        };
                        (GroupKey(key), i as i64)
                    })
                    .collect();

                let mut expected = pairs.clone();
                expected.sort_by_key(|(key, _)| key.0);

                let mut actual = pairs;
                let mut scratch = Vec::new();
                radix_sort_pairs(&mut actual, &mut scratch);

                assert_eq!(
                    actual, expected,
                    "max_key {max_key}, len {len}: radix disagrees with a comparison sort"
                );
            }
        }
    }

    #[test]
    fn both_algorithms_produce_the_same_aggregate() {
        // What actually matters: the strategy's answer must not depend on which sort ran.
        let table = table(ROWS);
        let mut results = Vec::new();

        for algorithm in [SortAlgorithm::Comparison, SortAlgorithm::Radix] {
            let mut agg = sort_agg(&table, "region", GroupKind::Utf8, "amount", 2);
            let mut scan =
                Scan::with_batch_size(&table, vec!["region".to_string(), "amount".to_string()], 2);
            while let Some(batch) = scan.next_batch() {
                agg.collect_batch(&batch);
            }
            agg.sort_pairs_with(algorithm);
            results.push(agg.aggregate_runs());
        }

        assert_eq!(
            results[0], results[1],
            "the sorts disagree on the aggregate"
        );
    }

    #[test]
    #[should_panic(expected = "only single-column GROUP BY is supported")]
    fn multi_column_group_by_is_an_internal_error() {
        let table = table(ROWS);
        let scan = Scan::new(&table, vec!["region".to_string(), "amount".to_string()]);
        let _: SortAggregate<_, i64> = SortAggregate::new(
            scan,
            vec!["region".to_string(), "amount".to_string()],
            GroupKind::Utf8,
            "amount".to_string(),
            "SUM(amount)".to_string(),
        );
    }
}
