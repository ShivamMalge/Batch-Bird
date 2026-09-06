//! `Scan` -- the leaf: cuts a [`Table`] into fixed-size batches.

use crate::exec::batch::RecordBatch;
use crate::exec::materialize::slice_column;
use crate::exec::{BATCH_SIZE, Operator};
use crate::storage::Table;

/// Reads a `Table` in `batch_size`-row slices.
///
/// Only the columns named in `columns` are read. That is the projection pushdown a real
/// planner would do, and here it is not an optimization so much as a correctness-of-measurement
/// concern: copying columns the query never mentions would tax the batch pipeline for work the
/// naive baseline does not do.
pub struct Scan<'a> {
    table: &'a Table,
    columns: Vec<String>,
    batch_size: usize,
    next_row: usize,
}

impl<'a> Scan<'a> {
    /// Scan `columns` of `table` at the default [`BATCH_SIZE`].
    pub fn new(table: &'a Table, columns: Vec<String>) -> Self {
        Scan::with_batch_size(table, columns, BATCH_SIZE)
    }

    /// Scan with an explicit batch size.
    ///
    /// Exists for two reasons: tests need small batches to exercise boundaries cheaply, and
    /// `systemDesign.md` suggests a batch-size sweep (512 / 1024 / 4096) as a secondary
    /// benchmark axis -- 1024 is a reasoned default, not a magic constant.
    pub fn with_batch_size(table: &'a Table, columns: Vec<String>, batch_size: usize) -> Self {
        assert!(batch_size > 0, "batch size must be positive");
        Scan {
            table,
            columns,
            batch_size,
            next_row: 0,
        }
    }
}

impl Operator for Scan<'_> {
    fn next_batch(&mut self) -> Option<RecordBatch> {
        if self.next_row >= self.table.nrows() {
            return None;
        }

        let start = self.next_row;
        // The final batch is short whenever the row count is not a multiple of the batch size.
        let end = (start + self.batch_size).min(self.table.nrows());
        self.next_row = end;

        let mut columns = crate::hash::map_with_capacity(self.columns.len());
        for name in &self.columns {
            let column = self
                .table
                .column(name)
                .expect("scan columns are validated when the pipeline is built");
            columns.insert(name.clone(), slice_column(column, start, end));
        }

        Some(RecordBatch::new(columns, end - start))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{infer_schema, read_csv};

    fn table(rows: usize) -> Table {
        let mut csv = String::from("a,b\n");
        for i in 0..rows {
            csv.push_str(&format!("{i},{}\n", i * 10));
        }
        let schema = infer_schema(csv.as_bytes()).unwrap();
        read_csv(csv.as_bytes(), &schema).unwrap()
    }

    /// Every value the scan emits, in order, for column `a`.
    fn drain(scan: &mut Scan<'_>) -> (Vec<i64>, Vec<usize>) {
        let mut values = Vec::new();
        let mut sizes = Vec::new();
        while let Some(batch) = scan.next_batch() {
            sizes.push(batch.len());
            values.extend_from_slice(batch.column("a").unwrap().as_i64().unwrap());
        }
        (values, sizes)
    }

    #[test]
    fn emits_every_row_exactly_once() {
        let table = table(10);
        let mut scan = Scan::with_batch_size(&table, vec!["a".to_string()], 4);
        let (values, sizes) = drain(&mut scan);

        assert_eq!(values, (0..10).collect::<Vec<i64>>());
        assert_eq!(sizes, vec![4, 4, 2], "the last batch is short");
    }

    #[test]
    fn handles_exact_batch_multiples() {
        let table = table(8);
        let mut scan = Scan::with_batch_size(&table, vec!["a".to_string()], 4);
        let (values, sizes) = drain(&mut scan);

        assert_eq!(values.len(), 8);
        assert_eq!(sizes, vec![4, 4], "no trailing empty batch");
    }

    #[test]
    fn handles_fewer_rows_than_one_batch() {
        let table = table(3);
        let mut scan = Scan::with_batch_size(&table, vec!["a".to_string()], 1024);
        let (values, sizes) = drain(&mut scan);

        assert_eq!(values, vec![0, 1, 2]);
        assert_eq!(sizes, vec![3]);
    }

    #[test]
    fn an_empty_table_yields_no_batches() {
        let table = table(0);
        let mut scan = Scan::new(&table, vec!["a".to_string()]);
        assert!(scan.next_batch().is_none());
    }

    #[test]
    fn is_exhausted_permanently() {
        let table = table(2);
        let mut scan = Scan::with_batch_size(&table, vec!["a".to_string()], 2);
        assert!(scan.next_batch().is_some());
        assert!(scan.next_batch().is_none());
        assert!(scan.next_batch().is_none(), "still None on a second ask");
    }

    #[test]
    fn reads_only_the_requested_columns() {
        let table = table(4);
        let mut scan = Scan::with_batch_size(&table, vec!["b".to_string()], 4);
        let batch = scan.next_batch().unwrap();

        assert_eq!(batch.ncols(), 1);
        assert!(
            batch.column("a").is_none(),
            "unrequested columns are not copied"
        );
        assert_eq!(
            batch.column("b").unwrap().as_i64(),
            Some(&[0i64, 10, 20, 30][..])
        );
    }

    #[test]
    fn batch_size_changes_batching_but_not_output() {
        let table = table(100);
        let expected: Vec<i64> = (0..100).collect();

        for size in [1, 7, 64, 99, 100, 101, 1024] {
            let mut scan = Scan::with_batch_size(&table, vec!["a".to_string()], size);
            let (values, sizes) = drain(&mut scan);
            assert_eq!(values, expected, "batch size {size}");
            assert_eq!(sizes.iter().sum::<usize>(), 100, "batch size {size}");
        }
    }
}
