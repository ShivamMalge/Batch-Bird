//! [`RecordBatch`] -- the unit of work flowing between operators.
//!
//! Same shape as a [`Table`](crate::storage::Table), just smaller: a set of equal-length named
//! columns. 1024 rows by default ([`BATCH_SIZE`](super::BATCH_SIZE)), sized so a batch's
//! working set sits in cache while an operator sweeps it.
//!
//! # It owns its data, and that costs something
//! `architecture.md` pins `RecordBatch` as owning `HashMap<String, Column>`, and `Column` owns
//! its `Vec`s. So `Scan` cannot hand out a view into the source `Table` -- it must **copy**
//! 1024 values per column per batch, and `Filter` copies again when it compacts. The naive
//! baseline copies nothing.
//!
//! That is a real, measurable tax on the batch pipeline, and Phase 6 should report it rather
//! than hide it. It is also exactly why production columnar engines (Arrow, DuckDB) pass
//! reference-counted buffers with offsets instead of materializing: the batching win is
//! supposed to come from cache behaviour and amortized dispatch, not from copying. Changing
//! `Column` to hold shared buffers is a change to a pinned type, so it needs sign-off.

use std::collections::HashMap;

use crate::error::Result;
use crate::storage::{Column, Table};

/// A slice of rows in columnar form.
#[derive(Debug, Clone, PartialEq)]
pub struct RecordBatch {
    columns: HashMap<String, Column>,
    len: usize,
}

impl RecordBatch {
    /// Build a batch. In debug builds every column is checked against `len`; operators are
    /// built once and validated then, so release builds do not re-pay for it per batch.
    pub fn new(columns: HashMap<String, Column>, len: usize) -> Self {
        debug_assert!(
            columns.values().all(|c| c.len() == len),
            "every column in a batch must have exactly {len} rows"
        );
        RecordBatch { columns, len }
    }

    pub fn column(&self, name: &str) -> Option<&Column> {
        self.columns.get(name)
    }

    /// Move a column out, leaving the batch without it.
    ///
    /// This is how `Project` drops columns without copying: it takes what it wants and lets
    /// the rest drop.
    pub fn take_column(&mut self, name: &str) -> Option<Column> {
        self.columns.remove(name)
    }

    /// Rows in this batch. Note this is the row count, not the column count.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Unordered, like `Table` -- columns are reached by name, never positionally.
    pub fn column_names(&self) -> impl Iterator<Item = &str> {
        self.columns.keys().map(String::as_str)
    }

    pub fn ncols(&self) -> usize {
        self.columns.len()
    }

    /// Promote a batch to a standalone `Table`.
    ///
    /// The final aggregate result is one small batch, and `architecture.md` specifies the
    /// query result as a `Table`; the two types have identical shape, so this is a move.
    pub fn into_table(self) -> Result<Table> {
        Table::new(self.columns, self.len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn batch() -> RecordBatch {
        let mut columns = HashMap::new();
        columns.insert("a".to_string(), Column::Int64(vec![1, 2, 3]));
        columns.insert("b".to_string(), Column::Float64(vec![1.0, 2.0, 3.0]));
        RecordBatch::new(columns, 3)
    }

    #[test]
    fn exposes_columns_by_name() {
        let batch = batch();
        assert_eq!(batch.len(), 3);
        assert_eq!(batch.ncols(), 2);
        assert_eq!(batch.column("a").unwrap().as_i64(), Some(&[1i64, 2, 3][..]));
        assert!(batch.column("missing").is_none());
    }

    #[test]
    fn taking_a_column_removes_it() {
        let mut batch = batch();
        let taken = batch.take_column("a").expect("column a");

        assert_eq!(taken.as_i64(), Some(&[1i64, 2, 3][..]));
        assert_eq!(batch.ncols(), 1, "the column is gone, not copied");
        assert!(batch.column("a").is_none());
        assert_eq!(
            batch.len(),
            3,
            "row count is unchanged by dropping a column"
        );
    }

    #[test]
    fn converts_to_a_table() {
        let table = batch().into_table().unwrap();
        assert_eq!(table.nrows(), 3);
        assert_eq!(table.ncols(), 2);
        assert_eq!(
            table.column("b").unwrap().as_f64(),
            Some(&[1.0f64, 2.0, 3.0][..])
        );
    }

    #[test]
    fn an_empty_batch_is_valid() {
        let batch = RecordBatch::new(HashMap::new(), 0);
        assert!(batch.is_empty());
        assert_eq!(batch.ncols(), 0);
    }

    #[test]
    #[should_panic(expected = "every column in a batch must have exactly")]
    fn debug_builds_catch_ragged_batches() {
        let mut columns = HashMap::new();
        columns.insert("a".to_string(), Column::Int64(vec![1, 2]));
        RecordBatch::new(columns, 3);
    }
}
