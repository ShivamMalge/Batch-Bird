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

    dictionary: Arc<[String]>,
    drained: bool,
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
            dictionary: Vec::new().into(),
            drained: false,
        }
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
        self.pairs.sort_unstable_by_key(|(key, _)| key.0);

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
            totals.push(T::sum_slice(&self.sorted_values[start..end]));
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
        self.sort_pairs();
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

        let mut columns = std::collections::HashMap::with_capacity(2);
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
