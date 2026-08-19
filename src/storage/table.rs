//! `Table` — a set of equal-length named columns held in memory.

use std::collections::HashMap;

use crate::error::{Error, Result};
use crate::storage::Column;

/// A columnar, in-memory table.
///
/// Columns live in a `HashMap` keyed by name (`architecture.md`), which means **column order
/// is not preserved**. That is fine for the supported query shape, which only ever reaches
/// columns by name; nothing in the engine iterates columns positionally.
///
/// `nrows` is stored rather than derived so an empty table still knows its row count, and so
/// callers can ask for the row count without picking an arbitrary column to measure.
#[derive(Debug, Clone, PartialEq)]
pub struct Table {
    columns: HashMap<String, Column>,
    nrows: usize,
}

impl Table {
    /// Build a table, checking the invariant every operator will later assume: all columns
    /// have exactly `nrows` rows. Checked once here so `Scan` can slice batches without
    /// re-validating per batch.
    pub fn new(columns: HashMap<String, Column>, nrows: usize) -> Result<Self> {
        for (name, column) in &columns {
            if column.len() != nrows {
                return Err(Error::ColumnLengthMismatch {
                    column: name.clone(),
                    expected: nrows,
                    found: column.len(),
                });
            }
        }
        Ok(Table { columns, nrows })
    }

    pub fn column(&self, name: &str) -> Option<&Column> {
        self.columns.get(name)
    }

    pub fn nrows(&self) -> usize {
        self.nrows
    }

    pub fn ncols(&self) -> usize {
        self.columns.len()
    }

    /// Unordered, for the reason given on the struct.
    pub fn column_names(&self) -> impl Iterator<Item = &str> {
        self.columns.keys().map(String::as_str)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table_of(pairs: Vec<(&str, Column)>, nrows: usize) -> Result<Table> {
        let map = pairs
            .into_iter()
            .map(|(n, c)| (n.to_string(), c))
            .collect::<HashMap<_, _>>();
        Table::new(map, nrows)
    }

    #[test]
    fn accepts_equal_length_columns() {
        let t = table_of(
            vec![
                ("a", Column::Int64(vec![1, 2, 3])),
                ("b", Column::Float64(vec![1.0, 2.0, 3.0])),
            ],
            3,
        )
        .unwrap();

        assert_eq!(t.nrows(), 3);
        assert_eq!(t.ncols(), 2);
        assert_eq!(t.column("a").unwrap().as_i64(), Some(&[1i64, 2, 3][..]));
        assert!(t.column("missing").is_none());
    }

    #[test]
    fn rejects_ragged_columns() {
        let err = table_of(
            vec![
                ("a", Column::Int64(vec![1, 2, 3])),
                ("b", Column::Int64(vec![1, 2])),
            ],
            3,
        )
        .unwrap_err();

        match err {
            Error::ColumnLengthMismatch {
                column,
                expected,
                found,
            } => {
                assert_eq!(column, "b");
                assert_eq!((expected, found), (3, 2));
            }
            other => panic!("expected ColumnLengthMismatch, got {other:?}"),
        }
    }

    #[test]
    fn empty_table_still_reports_its_row_count() {
        let t = table_of(vec![], 0).unwrap();
        assert_eq!(t.nrows(), 0);
        assert_eq!(t.ncols(), 0);
    }
}
