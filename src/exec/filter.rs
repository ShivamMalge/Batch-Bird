//! `Filter` -- bitset evaluation, then compaction.
//!
//! # Two steps, deliberately separate
//! 1. Sweep the filter column and build a packed [`Bitset`] -- one bit per row, no branches
//!    on the output side.
//! 2. **Compact**: materialize a new, smaller batch holding only the selected rows.
//!
//! `systemDesign.md` chose compaction over passing a selection vector downstream. The cost is
//! one copy per surviving row; the benefit is that `Project` and `Aggregate` see plain batches
//! and never need to know filtering happened, so no operator signature is coupled to
//! always-filtered execution. DuckDB takes the other road; that trade is the documented
//! road not taken.
//!
//! # Where the mask comes from
//! Numeric comparisons go through [`crate::exec::kernels`], which picks the scalar or SIMD
//! implementation depending on the `simd` feature. This operator never learns which ran --
//! that was the point of building the scalar mask a word at a time in Phase 4: a vector
//! compare produces bits too, just eight at a time, so wiring SIMD in changed nothing here
//! but two call sites. String comparisons stay scalar by guardrail (`agents.md`).

use std::collections::HashMap;

use crate::exec::Operator;
use crate::exec::batch::RecordBatch;
use crate::exec::bitset::Bitset;
use crate::exec::kernels::{compare, mask_f64, mask_from, mask_i64};
use crate::exec::materialize::gather_column;
use crate::plan::CompareOp;
use crate::storage::Column;

/// A filter comparison, resolved against the source table's types when the pipeline is built.
///
/// Resolving once here is what keeps the per-row work down to a compare: no type dispatch, no
/// dictionary lookup, no literal parsing inside the scan.
#[derive(Debug, Clone, PartialEq)]
pub enum FilterKind {
    IntVsInt(CompareOp, i64),
    /// `WHERE int_col > 2.5` -- the column value widens to `f64` per row, lossy past 2^53.
    IntVsFloat(CompareOp, f64),
    FloatVsFloat(CompareOp, f64),
    /// String equality against a dictionary code resolved **once**, at plan time.
    ///
    /// `None` means the literal is absent from the dictionary, so no row can match and the
    /// whole scan is a constant `false`. This is safe across batches only because
    /// `materialize` copies dictionaries verbatim and never renumbers codes.
    Utf8Eq(Option<u32>),
    /// String ordering. Codes are in first-seen order, not lexical order, so `<` and `>` must
    /// decode and compare the strings themselves -- the documented cost of an unsorted
    /// dictionary (`systemDesign.md`), and the reason Phase 5 vectorizes numeric filters only.
    Utf8Ord(CompareOp, String),
}

pub struct Filter<I> {
    input: I,
    column: String,
    kind: FilterKind,
}

impl<I: Operator> Filter<I> {
    pub fn new(input: I, column: String, kind: FilterKind) -> Self {
        Filter {
            input,
            column,
            kind,
        }
    }

    /// Build the selection mask for one batch.
    ///
    /// Public so Phase 6 can time mask construction on its own, separately from compaction --
    /// they are different costs with different scaling, and reporting one number for "filter"
    /// would hide which half SIMD actually helps.
    pub fn mask(&self, batch: &RecordBatch) -> Bitset {
        let column = batch
            .column(&self.column)
            .expect("filter column is validated when the pipeline is built");

        match (&self.kind, column) {
            // The two vectorizable comparisons (systemDesign.md "SIMD Scope"). The dispatcher
            // picks scalar or SIMD; this operator does not know or care which.
            (FilterKind::IntVsInt(op, literal), Column::Int64(values)) => {
                mask_i64(values, *op, *literal)
            }
            (FilterKind::FloatVsFloat(op, literal), Column::Float64(values)) => {
                mask_f64(values, *op, *literal)
            }
            // Stays scalar: widening each i64 to f64 makes this a mixed-type comparison
            // rather than the dense same-type compare systemDesign.md scopes SIMD to. It is
            // vectorizable in principle (a lane-wise int-to-float convert, then compare), but
            // widening the SIMD surface past what the docs pin needs sign-off.
            (FilterKind::IntVsFloat(op, literal), Column::Int64(values)) => {
                mask_from(values.len(), |row| {
                    compare(&(values[row] as f64), literal, *op)
                })
            }
            (FilterKind::Utf8Eq(code), Column::Utf8Dict { codes, .. }) => match code {
                Some(code) => mask_from(codes.len(), |row| codes[row] == *code),
                // The literal is not in the dictionary at all: nothing matches, and there is
                // no point touching the data.
                None => Bitset::new(codes.len()),
            },
            (FilterKind::Utf8Ord(op, literal), Column::Utf8Dict { dict, codes }) => {
                mask_from(codes.len(), |row| {
                    compare(&dict[codes[row] as usize].as_str(), &literal.as_str(), *op)
                })
            }
            _ => unreachable!("filter kind and column type are matched when the pipeline is built"),
        }
    }
}

impl<I: Operator> Operator for Filter<I> {
    fn next_batch(&mut self) -> Option<RecordBatch> {
        // Loops rather than forwarding an empty batch: a selective filter can reject whole
        // batches, and waking `Aggregate` up to do nothing is pure overhead. Returning `None`
        // still means "input exhausted", never "this batch was empty".
        loop {
            let batch = self.input.next_batch()?;
            let mask = self.mask(&batch);
            let selected = mask.count_ones();

            if selected == 0 {
                continue;
            }

            // Compaction: every column, not just the filtered one, so downstream sees a plain
            // batch with no idea a filter ran.
            let mut columns = HashMap::with_capacity(batch.ncols());
            for name in batch.column_names() {
                let column = batch.column(name).expect("name came from this batch");
                columns.insert(name.to_string(), gather_column(column, &mask, selected));
            }

            return Some(RecordBatch::new(columns, selected));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::scan::Scan;
    use crate::storage::{Table, infer_schema, read_csv};

    const ROWS: &str = concat!(
        "n,region,price\n",
        "1,north,1.5\n",
        "2,south,2.5\n",
        "3,north,3.5\n",
        "4,east,4.5\n",
        "5,north,5.5\n",
    );

    fn table(csv: &str) -> Table {
        let schema = infer_schema(csv.as_bytes()).unwrap();
        read_csv(csv.as_bytes(), &schema).unwrap()
    }

    fn all_columns(table: &Table) -> Vec<String> {
        let mut names: Vec<String> = table.column_names().map(str::to_string).collect();
        names.sort();
        names
    }

    /// Run a filter over the whole table and return the surviving `n` values in order.
    fn survivors(table: &Table, column: &str, kind: FilterKind, batch_size: usize) -> Vec<i64> {
        let scan = Scan::with_batch_size(table, all_columns(table), batch_size);
        let mut filter = Filter::new(scan, column.to_string(), kind);

        let mut out = Vec::new();
        while let Some(batch) = filter.next_batch() {
            out.extend_from_slice(batch.column("n").unwrap().as_i64().unwrap());
        }
        out
    }

    #[test]
    fn keeps_rows_matching_a_numeric_predicate() {
        let table = table(ROWS);
        let kind = FilterKind::IntVsInt(CompareOp::Gt, 2);
        assert_eq!(survivors(&table, "n", kind, 1024), vec![3, 4, 5]);
    }

    #[test]
    fn all_three_operators_select_the_right_rows() {
        let table = table(ROWS);
        let cases = [
            (CompareOp::Gt, vec![4, 5]),
            (CompareOp::Lt, vec![1, 2]),
            (CompareOp::Eq, vec![3]),
        ];
        for (op, expected) in cases {
            let kind = FilterKind::IntVsInt(op, 3);
            assert_eq!(survivors(&table, "n", kind, 1024), expected, "op {op:?}");
        }
    }

    #[test]
    fn compaction_shrinks_the_batch_and_keeps_every_column() {
        let table = table(ROWS);
        let scan = Scan::with_batch_size(&table, all_columns(&table), 1024);
        let mut filter = Filter::new(
            scan,
            "n".to_string(),
            FilterKind::IntVsInt(CompareOp::Gt, 3),
        );

        let batch = filter.next_batch().expect("a batch survives");
        assert_eq!(batch.len(), 2, "batch is compacted to the surviving rows");
        assert_eq!(
            batch.ncols(),
            3,
            "all columns come through, not just the filtered one"
        );
        assert_eq!(batch.column("n").unwrap().as_i64(), Some(&[4i64, 5][..]));
        assert_eq!(
            batch.column("price").unwrap().as_f64(),
            Some(&[4.5f64, 5.5][..])
        );
        assert_eq!(batch.column("region").unwrap().utf8_value(0), Some("east"));
    }

    #[test]
    fn batching_does_not_change_the_result() {
        // Compaction happens per batch, so a row near a boundary is where an off-by-one shows.
        let table = table(ROWS);
        for size in [1, 2, 3, 4, 5, 6, 1024] {
            let kind = FilterKind::IntVsInt(CompareOp::Gt, 1);
            assert_eq!(
                survivors(&table, "n", kind, size),
                vec![2, 3, 4, 5],
                "batch size {size}"
            );
        }
    }

    #[test]
    fn string_equality_compares_dictionary_codes() {
        let table = table(ROWS);
        // "north" is the first distinct value, so code 0.
        let kind = FilterKind::Utf8Eq(Some(0));
        assert_eq!(survivors(&table, "region", kind, 1024), vec![1, 3, 5]);
    }

    #[test]
    fn a_literal_absent_from_the_dictionary_matches_nothing() {
        let table = table(ROWS);
        assert!(survivors(&table, "region", FilterKind::Utf8Eq(None), 1024).is_empty());
    }

    #[test]
    fn string_ordering_compares_strings_not_codes() {
        // Dictionary order is north(0), south(1), east(2). Comparing codes against "north"
        // would wrongly select east; lexical comparison selects only south.
        let table = table(ROWS);
        let kind = FilterKind::Utf8Ord(CompareOp::Gt, "north".to_string());
        assert_eq!(survivors(&table, "region", kind, 1024), vec![2]);
    }

    #[test]
    fn mixed_numeric_comparisons_work() {
        let table = table(ROWS);
        let by_float = FilterKind::IntVsFloat(CompareOp::Gt, 3.5);
        assert_eq!(survivors(&table, "n", by_float, 1024), vec![4, 5]);

        let by_float_col = FilterKind::FloatVsFloat(CompareOp::Lt, 3.0);
        assert_eq!(survivors(&table, "price", by_float_col, 1024), vec![1, 2]);
    }

    #[test]
    fn rejecting_every_row_yields_no_batches() {
        let table = table(ROWS);
        let scan = Scan::with_batch_size(&table, all_columns(&table), 2);
        let mut filter = Filter::new(
            scan,
            "n".to_string(),
            FilterKind::IntVsInt(CompareOp::Gt, 100),
        );

        assert!(
            filter.next_batch().is_none(),
            "empty batches are skipped, not forwarded"
        );
    }

    #[test]
    fn mask_packs_the_expected_bits() {
        let table = table(ROWS);
        let mut scan = Scan::with_batch_size(&table, all_columns(&table), 1024);
        let filter = Filter::new(
            Scan::with_batch_size(&table, all_columns(&table), 1024),
            "n".to_string(),
            FilterKind::IntVsInt(CompareOp::Gt, 3),
        );

        let batch = scan.next_batch().unwrap();
        let mask = filter.mask(&batch);

        assert_eq!(mask.len(), 5);
        assert_eq!(mask.count_ones(), 2);
        assert_eq!(mask.iter_ones().collect::<Vec<_>>(), vec![3, 4]);
    }

    #[test]
    fn mask_is_correct_across_word_boundaries() {
        // 130 rows spans three words; only rows 60..70 pass, straddling the 64-bit edge.
        let mut csv = String::from("n\n");
        for i in 0..130 {
            csv.push_str(&format!("{i}\n"));
        }
        let table = table(&csv);
        let mut scan = Scan::with_batch_size(&table, vec!["n".to_string()], 1024);
        let filter = Filter::new(
            Scan::with_batch_size(&table, vec!["n".to_string()], 1024),
            "n".to_string(),
            FilterKind::IntVsInt(CompareOp::Lt, 70),
        );

        let batch = scan.next_batch().unwrap();
        let mask = filter.mask(&batch);
        assert_eq!(mask.count_ones(), 70);
        assert_eq!(
            mask.iter_ones().collect::<Vec<_>>(),
            (0..70).collect::<Vec<_>>()
        );
    }
}
