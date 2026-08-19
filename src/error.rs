//! Crate-wide error type.
//!
//! Deliberately a plain enum rather than a `Box<dyn Error>`: Phase 2 must *reject*
//! out-of-scope SQL with a clear message (`phases.md`), which is much easier to test
//! against a typed error than a stringly-typed one.

use std::fmt;

use crate::storage::DataType;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Csv(csv::Error),

    /// `Table` stores columns in a map keyed by name, so two identically-named CSV columns
    /// would silently shadow each other. Caught at load time instead.
    DuplicateColumn(String),

    /// Every column in a `Table` must agree on the row count.
    ColumnLengthMismatch {
        column: String,
        expected: usize,
        found: usize,
    },

    /// Only reachable when a caller supplies an explicit schema that the data contradicts;
    /// an inferred schema is correct by construction.
    Parse {
        column: String,
        row: usize,
        value: String,
        expected: DataType,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "io error: {e}"),
            Error::Csv(e) => write!(f, "csv error: {e}"),
            Error::DuplicateColumn(name) => {
                write!(f, "duplicate column name in CSV header: {name:?}")
            }
            Error::ColumnLengthMismatch {
                column,
                expected,
                found,
            } => write!(
                f,
                "column {column:?} has {found} rows, but the table has {expected}"
            ),
            Error::Parse {
                column,
                row,
                value,
                expected,
            } => write!(
                f,
                "could not parse {value:?} as {expected} for column {column:?} (row {row})"
            ),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            Error::Csv(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

impl From<csv::Error> for Error {
    fn from(e: csv::Error) -> Self {
        Error::Csv(e)
    }
}
