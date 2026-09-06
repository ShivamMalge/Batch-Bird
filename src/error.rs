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

    /// The text is not valid SQL at all -- `sqlparser` could not read it.
    Sql(sqlparser::parser::ParserError),

    /// Perfectly valid SQL that falls outside the one supported query shape.
    ///
    /// Kept distinct from [`Error::Sql`] because they mean opposite things to a user: one is
    /// "you typed it wrong", the other is "this engine deliberately does not do that"
    /// (`agents.md` Hard Guardrails). The payload names the specific construct, and the
    /// `Display` impl appends the shape that *is* supported.
    Unsupported(String),

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
        /// **1-based line number as the CSV reader counts them**, with the header as line 1.
        ///
        /// Taken from the reader's own record position, not a row counter, so it lines up
        /// with the source file rather than being an index into the parsed rows.
        ///
        /// One caveat, measured rather than assumed: the reader does not count wholly blank
        /// lines, so in a file containing them this can be lower than the physical line in a
        /// text editor. The physical number is not recoverable through the reader's API --
        /// blank lines are consumed invisibly -- so this reports what can actually be known,
        /// and says so. Pinned by test in `csv_loader`.
        line: u64,
        value: String,
        expected: DataType,
    },

    /// A column named in the query is not in the table.
    ///
    /// The parser validates *shape*; it has never seen the data, so it cannot know which
    /// names exist. That check lands here, when the plan first meets a `Table`.
    UnknownColumn {
        column: String,
        /// Sorted, so the message is deterministic despite `Table` keying columns by hash.
        available: Vec<String>,
    },

    /// A column exists, but its type cannot serve the role the query gives it -- summing a
    /// string, grouping by a float, comparing a number against a quoted literal.
    TypeError(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "io error: {e}"),
            Error::Csv(e) => write!(f, "csv error: {e}"),
            Error::Sql(e) => write!(f, "could not parse SQL: {e}"),
            Error::Unsupported(what) => write!(
                f,
                "unsupported query: {what}\n\
                 batchbird supports exactly one query shape:\n  \
                 SELECT col1, SUM(col2) FROM t WHERE col3 <op> x GROUP BY col1\n  \
                 (op is one of =, <, >)"
            ),
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
                line,
                value,
                expected,
            } => write!(
                f,
                "could not parse {value:?} as {expected} for column {column:?} (line {line})"
            ),
            Error::UnknownColumn { column, available } => write!(
                f,
                "no column named {column:?}; the table has: {}",
                available.join(", ")
            ),
            Error::TypeError(detail) => write!(f, "type error: {detail}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            Error::Csv(e) => Some(e),
            Error::Sql(e) => Some(e),
            _ => None,
        }
    }
}

impl From<sqlparser::parser::ParserError> for Error {
    fn from(e: sqlparser::parser::ParserError) -> Self {
        Error::Sql(e)
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
