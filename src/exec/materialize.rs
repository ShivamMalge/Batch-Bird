//! The two places the batch pipeline copies data, kept together so the cost is visible.
//!
//! [`slice_column`] is what `Scan` pays to cut a `Table` into batches; [`gather_column`] is
//! what `Filter` pays to compact a batch down to its surviving rows (`systemDesign.md` chose
//! compaction over selection-vector passthrough, and this is that choice's bill).
//!
//! # Dictionaries are shared, not copied
//! Both operations `Arc::clone` the dictionary rather than duplicating it. That is load-bearing,
//! not tidiness. When `Column::Utf8Dict` owned a `Vec<String>`, every batch cloned the whole
//! dictionary twice -- once here for `Scan`, once again for `Filter` compaction -- and measured
//! **84x slower than the row-at-a-time baseline** at 10k distinct values, while an `Int64`
//! group column over identical data stayed at 1.1x. The benchmark would have been reporting
//! allocator throughput rather than anything about vectorized execution. `examples/smoke.rs`
//! reproduces the measurement.
//!
//! Two properties downstream operators rely on, both guaranteed here:
//! - **Dictionary identity is preserved.** Slicing and gathering share `dict` and never
//!   renumber codes, so a code resolved once against the source `Table` stays valid in every
//!   batch. `Filter` uses this to resolve a string literal to a code exactly once, at plan
//!   time, instead of re-scanning the dictionary per batch.
//! - **Row order is preserved.** Both operations keep rows in their original relative order.

use std::sync::Arc;

use crate::exec::bitset::Bitset;
use crate::storage::Column;

/// Copy rows `start..end` of a column into a new one.
pub fn slice_column(column: &Column, start: usize, end: usize) -> Column {
    match column {
        Column::Int64(values) => Column::Int64(values[start..end].to_vec()),
        Column::Float64(values) => Column::Float64(values[start..end].to_vec()),
        Column::Utf8Dict { dict, codes } => Column::Utf8Dict {
            // An `Arc` clone: the batch shares the dictionary rather than copying it.
            dict: Arc::clone(dict),
            codes: codes[start..end].to_vec(),
        },
    }
}

/// Copy the rows whose bit is set, in ascending row order.
///
/// `selected` is passed in rather than recomputed from `mask.count_ones()` because `Filter`
/// already needs the count to decide whether the batch is worth forwarding at all; this lets
/// the output vectors be allocated exactly once, at the right size.
pub fn gather_column(column: &Column, mask: &Bitset, selected: usize) -> Column {
    debug_assert_eq!(selected, mask.count_ones(), "selected must match the mask");

    match column {
        Column::Int64(values) => {
            let mut out = Vec::with_capacity(selected);
            out.extend(mask.iter_ones().map(|row| values[row]));
            Column::Int64(out)
        }
        Column::Float64(values) => {
            let mut out = Vec::with_capacity(selected);
            out.extend(mask.iter_ones().map(|row| values[row]));
            Column::Float64(out)
        }
        Column::Utf8Dict { dict, codes } => {
            let mut out = Vec::with_capacity(selected);
            out.extend(mask.iter_ones().map(|row| codes[row]));
            Column::Utf8Dict {
                // Not rebuilt to only the surviving values: renumbering would invalidate
                // every code resolved against the source table, and would cost a copy where
                // sharing costs a refcount bump.
                dict: Arc::clone(dict),
                codes: out,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn utf8(values: &[&str]) -> Column {
        let mut dict: Vec<String> = Vec::new();
        let mut codes = Vec::new();
        for value in values {
            let code = match dict.iter().position(|d| d == value) {
                Some(i) => i as u32,
                None => {
                    dict.push((*value).to_string());
                    (dict.len() - 1) as u32
                }
            };
            codes.push(code);
        }
        Column::Utf8Dict {
            dict: dict.into(),
            codes,
        }
    }

    fn mask_of(len: usize, rows: &[usize]) -> Bitset {
        let mut mask = Bitset::new(len);
        for row in rows {
            mask.set(*row);
        }
        mask
    }

    #[test]
    fn slices_numeric_columns() {
        let column = Column::Int64(vec![10, 20, 30, 40, 50]);
        assert_eq!(
            slice_column(&column, 1, 4).as_i64(),
            Some(&[20i64, 30, 40][..])
        );
        assert_eq!(slice_column(&column, 0, 0).len(), 0);
        assert_eq!(slice_column(&column, 0, 5).len(), 5);
    }

    #[test]
    fn slicing_preserves_dictionary_codes() {
        // The invariant `Filter` depends on: a code resolved against the source table stays
        // valid in every batch, so the literal lookup happens once at plan time.
        let column = utf8(&["north", "south", "north", "east"]);
        let sliced = slice_column(&column, 2, 4);

        let (source_dict, _) = column.as_utf8_dict().unwrap();
        let (sliced_dict, sliced_codes) = sliced.as_utf8_dict().unwrap();

        assert_eq!(source_dict, sliced_dict, "dictionary must be verbatim");
        assert_eq!(sliced_codes, [0, 2], "codes must not be renumbered");
        assert_eq!(sliced.utf8_value(0), Some("north"));
        assert_eq!(sliced.utf8_value(1), Some("east"));
    }

    #[test]
    fn gathers_selected_rows_in_order() {
        let column = Column::Float64(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let mask = mask_of(5, &[0, 3, 4]);
        assert_eq!(
            gather_column(&column, &mask, 3).as_f64(),
            Some(&[1.0f64, 4.0, 5.0][..])
        );
    }

    #[test]
    fn gathering_preserves_dictionary_codes() {
        let column = utf8(&["north", "south", "north", "east"]);
        let mask = mask_of(4, &[1, 3]);
        let gathered = gather_column(&column, &mask, 2);

        let (dict, codes) = gathered.as_utf8_dict().unwrap();
        assert_eq!(dict, column.as_utf8_dict().unwrap().0);
        assert_eq!(codes, [1, 2], "codes are copied, not renumbered");
        assert_eq!(gathered.utf8_value(0), Some("south"));
        assert_eq!(gathered.utf8_value(1), Some("east"));
    }

    #[test]
    fn gathering_nothing_yields_an_empty_column() {
        let column = Column::Int64(vec![1, 2, 3]);
        let empty = gather_column(&column, &mask_of(3, &[]), 0);
        assert_eq!(empty.len(), 0);
        assert_eq!(empty.data_type(), column.data_type());
    }

    #[test]
    fn gathering_everything_is_a_copy() {
        let column = Column::Int64(vec![1, 2, 3]);
        let all = gather_column(&column, &mask_of(3, &[0, 1, 2]), 3);
        assert_eq!(all, column);
    }
}
