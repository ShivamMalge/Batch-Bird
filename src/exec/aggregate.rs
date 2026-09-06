//! `Aggregate` -- hash group-by with per-group accumulators.
//!
//! # Two phases, kept separate on purpose
//! Per batch:
//! 1. [`build_group_index`](Aggregate::build_group_index) -- sweep the group column, mapping
//!    each row's [`GroupKey`] to an accumulator slot. A sequential read plus a hash probe.
//! 2. [`scatter_accumulate`](Aggregate::scatter_accumulate) -- `accumulators[slot].update(v)`.
//!    The destination index is *data-dependent*, which is what makes this phase unvectorizable:
//!    SIMD can load eight values at once but cannot store them to eight computed addresses.
//!
//! `systemDesign.md` insists these be profiled separately, and both are public so Phase 6 can
//! time them individually. Reporting one combined "group-by" number would muddy the headline
//! finding, because phase 1 benefits a little from vectorization and phase 2 not at all.
//!
//! # Blocking operator
//! Aggregation cannot stream: no group's total is final until the input is exhausted. So
//! `next_batch` drains its input on the first call and returns the whole result as one batch,
//! then `None` forever. That is how a blocking operator fits a pull-based pipeline.
//!
//! # Monomorphized, not boxed
//! `Aggregate` is generic over `A: Accumulator<T>` and instantiated per value type
//! (`architecture.md`, `techstack.md`), so `update` inlines to a single add. `Box<dyn
//! Accumulator>` would put a virtual call on the innermost loop -- once per row -- and would
//! only pay for itself if a single query needed heterogeneous accumulators chosen at plan
//! time, which is out of scope.

use std::marker::PhantomData;
use std::sync::Arc;

use crate::exec::Operator;
use crate::exec::batch::RecordBatch;
use crate::storage::Column;

/// A single-column group key.
///
/// Holds a dictionary code (widened from `u32`) or a raw `i64` bit-cast with `as u64`. The
/// bit-cast is bijective, so hashing and equality stay exact; it does not preserve numeric
/// order, which does not matter because grouping needs only equality.
///
/// Deliberately not generic over tuples: multi-column keys would need heterogeneous hashing
/// and equality for a feature that is an explicit non-goal (`agents.md` guardrail).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GroupKey(pub u64);

/// Accumulates values of type `T` into one aggregate result.
///
/// Generic over `T` rather than fixed to `f64` (confirmed 2026-08-18, `systemDesign.md`) so
/// summing an `Int64` column stays exact instead of silently losing precision past 2^53.
///
/// Only `SumAccumulator` implements it. The trait exists so "why no AVG or COUNT?" has the
/// answer "they are new accumulators", not "they would need a rearchitecture".
pub trait Accumulator<T> {
    fn update(&mut self, val: T);
    fn finalize(&self) -> T;
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SumAccumulator<T> {
    total: T,
}

impl Accumulator<i64> for SumAccumulator<i64> {
    #[inline]
    fn update(&mut self, val: i64) {
        // Wrapping, matching the naive baseline. SIMD lane arithmetic wraps and cannot panic,
        // so this is what lets all three implementations agree bit-for-bit on overflow rather
        // than one of them panicking in a debug build.
        self.total = self.total.wrapping_add(val);
    }

    #[inline]
    fn finalize(&self) -> i64 {
        self.total
    }
}

impl Accumulator<f64> for SumAccumulator<f64> {
    #[inline]
    fn update(&mut self, val: f64) {
        self.total += val;
    }

    #[inline]
    fn finalize(&self) -> f64 {
        self.total
    }
}

/// A value type that can be summed out of a [`Column`].
///
/// Bridges the runtime `Column` enum to the compile-time `T` that `Accumulator<T>` is generic
/// over. The enum is matched **once per batch**, not per row.
pub trait Summable: Copy + Default {
    fn slice_of(column: &Column) -> Option<&[Self]>;
    fn into_column(values: Vec<Self>) -> Column;

    /// Sum a dense, contiguous slice.
    ///
    /// Hash group-by never calls this -- it scatters into per-group accumulators, and a
    /// scatter has no slice to hand over. [`SortAggregate`](crate::exec::SortAggregate) does:
    /// sorting makes each group's values contiguous, so a run *is* a dense slice. This is the
    /// single seam through which the Phase 5 SIMD sum kernel reaches the query path, and it is
    /// why sort-group can vectorize its aggregation where hash-group cannot.
    fn sum_slice(values: &[Self]) -> Self;
}

impl Summable for i64 {
    fn slice_of(column: &Column) -> Option<&[Self]> {
        column.as_i64()
    }

    fn into_column(values: Vec<Self>) -> Column {
        Column::Int64(values)
    }

    fn sum_slice(values: &[Self]) -> Self {
        crate::exec::kernels::sum_i64(values)
    }
}

impl Summable for f64 {
    fn slice_of(column: &Column) -> Option<&[Self]> {
        column.as_f64()
    }

    fn into_column(values: Vec<Self>) -> Column {
        Column::Float64(values)
    }

    fn sum_slice(values: &[Self]) -> Self {
        crate::exec::kernels::sum_f64(values)
    }
}

/// The group column's storage type, fixed when the pipeline is built.
///
/// Known up front rather than sniffed from the first batch, so an empty result still has the
/// correct schema instead of an arbitrary one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroupKind {
    Int64,
    Utf8,
}

pub struct Aggregate<I, T, A> {
    input: I,
    group_column: String,
    group_kind: GroupKind,
    value_column: String,
    output_column: String,

    /// Group key -> accumulator slot. `hashbrown` because std's SipHash on an 8-byte key
    /// would dominate phase 1's timing (`techstack.md`), and a **fixed** seed because a
    /// random one gives every process a different collision pattern and made runs
    /// incomparable (`crate::hash`).
    index: crate::hash::Map<GroupKey, usize>,
    /// Slot order, so results come out in first-seen order and the key for each slot is known.
    keys: Vec<GroupKey>,
    accumulators: Vec<A>,

    /// Phase 1's output, read by phase 2. Reused across batches so the per-batch cost is a
    /// `clear`, not an allocation.
    slots: Vec<usize>,

    /// Captured from the first string batch, to label groups at finalize time. Empty for
    /// `Int64` groups and for a query that matched no rows. An `Arc` clone, so capturing it
    /// is a refcount bump rather than a copy of every label.
    dictionary: Arc<[String]>,

    drained: bool,
    /// `T` appears only in method signatures, so the compiler needs it named here to know
    /// which `Accumulator<T>` impl `A` is being used through.
    _value: PhantomData<T>,
}

impl<I: Operator, T: Summable, A: Accumulator<T> + Default> Aggregate<I, T, A> {
    /// Build an aggregate over exactly one group column.
    ///
    /// `group_columns` is a `Vec` rather than a single name to leave the multi-column
    /// extension seam visible at the signature level, exactly as `systemDesign.md` describes
    /// -- and it panics on anything but one, because the parser already rejects multi-column
    /// `GROUP BY` and reaching here with two would be an internal bug, not user error.
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

        Aggregate {
            input,
            group_column: group_columns.into_iter().next().expect("length checked"),
            group_kind,
            value_column,
            output_column,
            index: crate::hash::map(),
            keys: Vec::new(),
            accumulators: Vec::new(),
            slots: Vec::new(),
            dictionary: Vec::new().into(),
            drained: false,
            _value: PhantomData,
        }
    }

    /// **Phase 1.** Map every row of the batch to an accumulator slot, creating slots for
    /// keys not seen before. Results land in `self.slots`.
    ///
    /// The read side is a sequential sweep of the group column and partially vectorizes on
    /// its own; the hash probe in the middle does not.
    pub fn build_group_index(&mut self, batch: &RecordBatch) {
        let column = batch
            .column(&self.group_column)
            .expect("group column is validated when the pipeline is built");

        self.slots.clear();
        self.slots.reserve(batch.len());

        match column {
            Column::Int64(values) => {
                for value in values {
                    let key = GroupKey(*value as u64);
                    let slot = self.slot_for(key);
                    self.slots.push(slot);
                }
            }
            Column::Utf8Dict { dict, codes } => {
                if self.dictionary.is_empty() {
                    // Once per query, not per batch: every batch shares the one dictionary,
                    // so any batch's is the whole dictionary.
                    self.dictionary = Arc::clone(dict);
                }
                for code in codes {
                    let key = GroupKey(*code as u64);
                    let slot = self.slot_for(key);
                    self.slots.push(slot);
                }
            }
            Column::Float64(_) => {
                unreachable!("float group keys are rejected when the pipeline is built")
            }
        }
    }

    #[inline]
    fn slot_for(&mut self, key: GroupKey) -> usize {
        match self.index.get(&key) {
            Some(slot) => *slot,
            None => {
                let slot = self.keys.len();
                self.index.insert(key, slot);
                self.keys.push(key);
                self.accumulators.push(A::default());
                slot
            }
        }
    }

    /// **Phase 2.** Add each row's value into its group's accumulator.
    ///
    /// This is the scatter: `slots[row]` is data-dependent, so consecutive rows write to
    /// unpredictable addresses. No amount of vectorization helps -- the headline asymmetry
    /// the benchmark exists to demonstrate.
    pub fn scatter_accumulate(&mut self, batch: &RecordBatch) {
        let column = batch
            .column(&self.value_column)
            .expect("value column is validated when the pipeline is built");
        let values = T::slice_of(column).expect("value column type is validated at build time");

        debug_assert_eq!(values.len(), self.slots.len(), "phase 1 must run first");

        for (row, value) in values.iter().enumerate() {
            let slot = self.slots[row];
            self.accumulators[slot].update(*value);
        }
    }

    /// Both phases, in order.
    pub fn consume(&mut self, batch: &RecordBatch) {
        self.build_group_index(batch);
        self.scatter_accumulate(batch);
    }

    /// Materialize one row per group.
    fn finish(&mut self) -> RecordBatch {
        let group_column = match self.group_kind {
            // Recovering the i64 by casting back is exact -- the widening was a bit-cast.
            GroupKind::Int64 => Column::Int64(self.keys.iter().map(|k| k.0 as i64).collect()),
            GroupKind::Utf8 => Column::Utf8Dict {
                // One row per group, so the result dictionary is exactly the group labels and
                // the codes are 0..n.
                // A fresh dictionary rather than a share of the input's: the result holds
                // one row per group, so its labels are exactly the surviving groups.
                dict: self
                    .keys
                    .iter()
                    .map(|k| self.dictionary[k.0 as usize].clone())
                    .collect::<Vec<_>>()
                    .into(),
                codes: (0..self.keys.len() as u32).collect(),
            },
        };

        let totals: Vec<T> = self.accumulators.iter().map(|a| a.finalize()).collect();

        let mut columns = crate::hash::map_with_capacity(2);
        columns.insert(self.group_column.clone(), group_column);
        columns.insert(self.output_column.clone(), T::into_column(totals));

        RecordBatch::new(columns, self.keys.len())
    }
}

impl<I: Operator, T: Summable, A: Accumulator<T> + Default> Operator for Aggregate<I, T, A> {
    fn next_batch(&mut self) -> Option<RecordBatch> {
        if self.drained {
            return None;
        }

        while let Some(batch) = self.input.next_batch() {
            self.consume(&batch);
        }
        self.drained = true;

        Some(self.finish())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::filter::{Filter, FilterKind};
    use crate::exec::scan::Scan;
    use crate::plan::CompareOp;
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

    /// Group/sum pairs, sorted so nothing depends on group order.
    fn pairs(batch: &RecordBatch, group: &str, agg: &str) -> Vec<(String, String)> {
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

    /// Aggregate straight off a scan, no filter.
    fn aggregate_all<'a>(
        table: &'a Table,
        group: &str,
        kind: GroupKind,
        value: &str,
        batch_size: usize,
    ) -> Aggregate<Scan<'a>, i64, SumAccumulator<i64>> {
        let scan = Scan::with_batch_size(
            table,
            vec![group.to_string(), value.to_string()],
            batch_size,
        );
        Aggregate::new(
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
        let mut agg = aggregate_all(&table, "region", GroupKind::Utf8, "amount", 1024);
        let result = agg.next_batch().unwrap();

        assert_eq!(
            pairs(&result, "region", "SUM(amount)"),
            owned(vec![("east", "40"), ("north", "90"), ("south", "-20")])
        );
    }

    #[test]
    fn groups_by_integer_including_negatives() {
        // Exercises the i64 -> u64 bit-cast round trip.
        let table = table("k,v\n-1,10\n2,20\n-1,30\n2,40\n");
        let mut agg = aggregate_all(&table, "k", GroupKind::Int64, "v", 1024);
        let result = agg.next_batch().unwrap();

        assert_eq!(
            pairs(&result, "k", "SUM(v)"),
            owned(vec![("-1", "40"), ("2", "60")])
        );
    }

    #[test]
    fn accumulates_across_batch_boundaries() {
        // The property that makes aggregation blocking: a group spanning batches must end up
        // in one slot, with one total.
        let table = table(ROWS);
        for size in [1, 2, 3, 4, 5, 1024] {
            let mut agg = aggregate_all(&table, "region", GroupKind::Utf8, "amount", size);
            let result = agg.next_batch().unwrap();
            assert_eq!(
                pairs(&result, "region", "SUM(amount)"),
                owned(vec![("east", "40"), ("north", "90"), ("south", "-20")]),
                "batch size {size}"
            );
        }
    }

    #[test]
    fn sums_floats() {
        let table = table(ROWS);
        let scan =
            Scan::with_batch_size(&table, vec!["region".to_string(), "price".to_string()], 2);
        let mut agg: Aggregate<_, f64, SumAccumulator<f64>> = Aggregate::new(
            scan,
            vec!["region".to_string()],
            GroupKind::Utf8,
            "price".to_string(),
            "SUM(price)".to_string(),
        );

        let result = agg.next_batch().unwrap();
        assert_eq!(
            pairs(&result, "region", "SUM(price)"),
            owned(vec![("east", "4.75"), ("north", "10"), ("south", "2.25")])
        );
    }

    #[test]
    fn is_blocking_and_yields_exactly_one_batch() {
        let table = table(ROWS);
        let mut agg = aggregate_all(&table, "region", GroupKind::Utf8, "amount", 2);

        assert!(agg.next_batch().is_some());
        assert!(agg.next_batch().is_none(), "the result is produced once");
        assert!(agg.next_batch().is_none());
    }

    #[test]
    fn an_empty_input_still_produces_a_correctly_typed_empty_result() {
        // Group kind is fixed at build time precisely so this schema is right rather than
        // guessed from a batch that never arrives.
        let table = table("region,amount\nnorth,1\n");
        let scan = Scan::with_batch_size(
            &table,
            vec!["region".to_string(), "amount".to_string()],
            1024,
        );
        let filter = Filter::new(
            scan,
            "amount".to_string(),
            FilterKind::IntVsInt(CompareOp::Gt, 100),
            vec!["region".to_string(), "amount".to_string()],
        );
        let mut agg: Aggregate<_, i64, SumAccumulator<i64>> = Aggregate::new(
            filter,
            vec!["region".to_string()],
            GroupKind::Utf8,
            "amount".to_string(),
            "SUM(amount)".to_string(),
        );

        let result = agg.next_batch().unwrap();
        assert_eq!(result.len(), 0);
        assert_eq!(result.ncols(), 2);
        assert_eq!(
            result.column("region").unwrap().data_type(),
            crate::storage::DataType::Utf8
        );
    }

    #[test]
    fn the_two_phases_can_be_driven_separately() {
        // Phase 6 times these independently; if they could not be called apart, the
        // "SIMD helps phase 1 a little and phase 2 not at all" finding could not be measured.
        let table = table(ROWS);
        let mut scan = Scan::with_batch_size(
            &table,
            vec!["region".to_string(), "amount".to_string()],
            1024,
        );
        let batch = scan.next_batch().unwrap();

        let mut agg = aggregate_all(&table, "region", GroupKind::Utf8, "amount", 1024);

        agg.build_group_index(&batch);
        assert_eq!(
            agg.slots,
            vec![0, 1, 0, 2, 0],
            "north, south, north, east, north"
        );
        assert_eq!(agg.keys.len(), 3, "three distinct groups");

        agg.scatter_accumulate(&batch);
        let result = agg.finish();
        assert_eq!(
            pairs(&result, "region", "SUM(amount)"),
            owned(vec![("east", "40"), ("north", "90"), ("south", "-20")])
        );
    }

    #[test]
    fn integer_sums_wrap_like_the_naive_baseline() {
        let csv = format!("k,v\n1,{}\n1,1\n", i64::MAX);
        let table = table(&csv);
        let mut agg = aggregate_all(&table, "k", GroupKind::Int64, "v", 1024);
        let result = agg.next_batch().unwrap();

        assert_eq!(
            pairs(&result, "k", "SUM(v)"),
            vec![("1".to_string(), i64::MIN.to_string())]
        );
    }

    #[test]
    #[should_panic(expected = "only single-column GROUP BY is supported")]
    fn multi_column_group_by_is_an_internal_error() {
        // The extension seam systemDesign.md describes: the signature admits several columns,
        // the implementation refuses them.
        let table = table(ROWS);
        let scan = Scan::new(&table, vec!["region".to_string(), "amount".to_string()]);
        let _: Aggregate<_, i64, SumAccumulator<i64>> = Aggregate::new(
            scan,
            vec!["region".to_string(), "amount".to_string()],
            GroupKind::Utf8,
            "amount".to_string(),
            "SUM(amount)".to_string(),
        );
    }
}
