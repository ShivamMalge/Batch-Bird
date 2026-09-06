//! `Project` -- column selection, nothing more.
//!
//! No expressions: the supported query shape has none (`prd.md`), so this operator exists to
//! drop the filter column once `Filter` is done with it, leaving `Aggregate` a batch of
//! exactly the two columns it uses.
//!
//! It is the one operator in the pipeline that copies nothing. Columns are **moved** out of
//! the incoming batch and the rest are dropped, so projecting is a few pointer moves plus
//! freeing what is no longer needed.

use crate::exec::Operator;
use crate::exec::batch::RecordBatch;

pub struct Project<I> {
    input: I,
    columns: Vec<String>,
}

impl<I: Operator> Project<I> {
    pub fn new(input: I, columns: Vec<String>) -> Self {
        Project { input, columns }
    }
}

impl<I: Operator> Operator for Project<I> {
    fn next_batch(&mut self) -> Option<RecordBatch> {
        let mut batch = self.input.next_batch()?;
        let len = batch.len();

        let mut columns = crate::hash::map_with_capacity(self.columns.len());
        for name in &self.columns {
            let column = batch
                .take_column(name)
                .expect("projected columns are validated when the pipeline is built");
            columns.insert(name.clone(), column);
        }

        Some(RecordBatch::new(columns, len))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::scan::Scan;
    use crate::storage::{Table, infer_schema, read_csv};

    const ROWS: &str = concat!("a,b,c\n", "1,10,100\n", "2,20,200\n", "3,30,300\n");

    fn table() -> Table {
        let schema = infer_schema(ROWS.as_bytes()).unwrap();
        read_csv(ROWS.as_bytes(), &schema).unwrap()
    }

    fn all() -> Vec<String> {
        vec!["a".to_string(), "b".to_string(), "c".to_string()]
    }

    #[test]
    fn keeps_only_the_named_columns() {
        let table = table();
        let scan = Scan::with_batch_size(&table, all(), 1024);
        let mut project = Project::new(scan, vec!["a".to_string(), "c".to_string()]);

        let batch = project.next_batch().unwrap();
        assert_eq!(batch.ncols(), 2);
        assert_eq!(batch.len(), 3, "dropping a column does not drop rows");
        assert_eq!(batch.column("a").unwrap().as_i64(), Some(&[1i64, 2, 3][..]));
        assert_eq!(
            batch.column("c").unwrap().as_i64(),
            Some(&[100i64, 200, 300][..])
        );
        assert!(batch.column("b").is_none());
    }

    #[test]
    fn passes_every_batch_through() {
        let table = table();
        let scan = Scan::with_batch_size(&table, all(), 2);
        let mut project = Project::new(scan, vec!["a".to_string()]);

        let mut values = Vec::new();
        while let Some(batch) = project.next_batch() {
            values.extend_from_slice(batch.column("a").unwrap().as_i64().unwrap());
        }
        assert_eq!(values, vec![1, 2, 3], "batching is preserved end to end");
    }

    #[test]
    fn projecting_everything_is_a_no_op() {
        let table = table();
        let scan = Scan::with_batch_size(&table, all(), 1024);
        let mut project = Project::new(scan, all());

        let batch = project.next_batch().unwrap();
        assert_eq!(batch.ncols(), 3);
    }

    #[test]
    fn an_exhausted_input_yields_nothing() {
        let table = table();
        let scan = Scan::with_batch_size(&table, all(), 1024);
        let mut project = Project::new(scan, vec!["a".to_string()]);

        assert!(project.next_batch().is_some());
        assert!(project.next_batch().is_none());
    }
}
