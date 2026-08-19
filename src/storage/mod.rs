//! Columnar storage: typed columns, tables, and the CSV loader.

mod column;
mod csv_loader;
mod table;

pub use column::{Column, DataType};
pub use csv_loader::{Field, Schema, infer_schema, read_csv, read_csv_path};
pub use table::Table;
