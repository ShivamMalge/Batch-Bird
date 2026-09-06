//! CSV -> `Table`: schema inference and columnar loading.
//!
//! # Two passes, on purpose
//! `phases.md` allows inferring *or* declaring column types. Inference here is a separate
//! pass ([`infer_schema`]) that runs before loading ([`read_csv`]), rather than a single
//! pass that promotes a column type as it goes.
//!
//! Single-pass promotion sounds cheaper but has a nasty case: a column that parses as
//! `Int64` for two million rows and then hits `"N/A"`. Promoting it to `Utf8` at that point
//! requires turning the already-collected numbers back into strings, and re-formatting a
//! parsed float does not reliably reproduce its original text (`3.10` comes back as `3.1`).
//! Reading the file twice costs I/O that is never benchmarked -- loading happens before the
//! timer starts -- and buys exact, order-independent typing. Cheap trade.
//!
//! # No NULL / missing-value support
//! **Not specified anywhere in the design docs**, so this takes the least inventive option:
//! empty fields get no special meaning. `""` simply fails numeric parsing, so a column with
//! blanks infers as `Utf8` and the blank becomes an ordinary dictionary entry. Nothing here
//! silently coerces a missing value to `0`. Real NULL semantics (a validity bitmap, plus
//! null-skipping in every operator and aggregate) is a scope decision to make deliberately,
//! not something to backfill by accident.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use crate::error::{Error, Result};
use crate::storage::{Column, DataType, Table};

/// One named, typed column in a schema.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Field {
    pub name: String,
    pub data_type: DataType,
}

impl Field {
    pub fn new(name: impl Into<String>, data_type: DataType) -> Self {
        Field {
            name: name.into(),
            data_type,
        }
    }
}

/// Column names and types, in CSV column order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Schema {
    pub fields: Vec<Field>,
}

impl Schema {
    pub fn new(fields: Vec<Field>) -> Self {
        Schema { fields }
    }

    pub fn data_type(&self, name: &str) -> Option<DataType> {
        self.fields
            .iter()
            .find(|f| f.name == name)
            .map(|f| f.data_type)
    }
}

/// Parse an integer, tolerating surrounding whitespace.
///
/// Whitespace is trimmed for the *numeric* attempt only; `Utf8` values are stored verbatim.
/// A leading space in a CSV is nearly always an artifact of formatting, not data, and
/// letting `" 5"` force an entire column to `Utf8` would be a surprising trap.
fn parse_i64(field: &str) -> Option<i64> {
    field.trim().parse::<i64>().ok()
}

/// Parse a float, rejecting non-finite results.
///
/// `"inf"` and `"NaN"` both parse successfully as `f64` in Rust, which would quietly retype
/// a text column that happens to contain the word "nan". Requiring finiteness prevents that,
/// and keeps `NaN` -- which breaks equality, and therefore grouping -- out of the engine at
/// the boundary rather than deep inside an aggregate.
fn parse_f64(field: &str) -> Option<f64> {
    field.trim().parse::<f64>().ok().filter(|v| v.is_finite())
}

/// Extract column names from the header, rejecting duplicates.
///
/// `Table` keys columns by name, so two identically-named CSV columns would silently shadow
/// each other. Rejecting here turns a wrong-answer bug into a load-time error.
fn column_names(headers: &csv::StringRecord) -> Result<Vec<String>> {
    let names: Vec<String> = headers.iter().map(str::to_string).collect();

    let mut seen: HashMap<&str, ()> = HashMap::new();
    for name in &names {
        if seen.insert(name.as_str(), ()).is_some() {
            return Err(Error::DuplicateColumn(name.clone()));
        }
    }
    Ok(names)
}

/// Infer each column type by scanning every row.
///
/// A column is `Int64` if every value parses as `i64`, else `Float64` if every value parses
/// as a finite `f64`, else `Utf8`. Scanning the whole file rather than sampling the first N
/// rows is what makes the result trustworthy: a sampled schema can be contradicted by data
/// further down, and that failure would surface as a load error long after the sample.
///
/// A header-only file infers every column as `Utf8` -- with no values there is no evidence
/// for a narrower type, and `Utf8` is the one that can represent whatever arrives later.
pub fn infer_schema<R: Read>(reader: R) -> Result<Schema> {
    let mut rdr = csv::Reader::from_reader(reader);
    let names = column_names(rdr.headers()?)?;
    let ncols = names.len();

    let mut still_int = vec![true; ncols];
    let mut still_float = vec![true; ncols];
    let mut saw_data = false;

    let mut record = csv::StringRecord::new();
    while rdr.read_record(&mut record)? {
        saw_data = true;
        for (i, field) in record.iter().enumerate().take(ncols) {
            if still_int[i] && parse_i64(field).is_none() {
                still_int[i] = false;
            }
            if still_float[i] && parse_f64(field).is_none() {
                still_float[i] = false;
            }
        }
        // Once every column has bottomed out at Utf8 there is nothing left to learn --
        // types only ever widen, never narrow back down.
        if still_float.iter().all(|ok| !ok) {
            break;
        }
    }

    let fields = names
        .into_iter()
        .enumerate()
        .map(|(i, name)| {
            let data_type = if !saw_data {
                DataType::Utf8
            } else if still_int[i] {
                DataType::Int64
            } else if still_float[i] {
                DataType::Float64
            } else {
                DataType::Utf8
            };
            Field { name, data_type }
        })
        .collect();

    Ok(Schema::new(fields))
}

/// Per-column accumulation state during a load.
enum Builder {
    Int64(Vec<i64>),
    Float64(Vec<f64>),
    Utf8 {
        dict: Vec<String>,
        codes: Vec<u32>,
        /// String -> code, so encoding a row is a hash lookup instead of a linear scan of
        /// `dict`. Without it, loading would be O(rows x distinct values).
        index: HashMap<String, u32>,
    },
}

impl Builder {
    fn new(data_type: DataType) -> Self {
        match data_type {
            DataType::Int64 => Builder::Int64(Vec::new()),
            DataType::Float64 => Builder::Float64(Vec::new()),
            DataType::Utf8 => Builder::Utf8 {
                dict: Vec::new(),
                codes: Vec::new(),
                index: HashMap::new(),
            },
        }
    }

    fn push(&mut self, field: &str, column: &str, line: u64) -> Result<()> {
        match self {
            Builder::Int64(values) => {
                let v = parse_i64(field).ok_or_else(|| Error::Parse {
                    column: column.to_string(),
                    line,
                    value: field.to_string(),
                    expected: DataType::Int64,
                })?;
                values.push(v);
            }
            Builder::Float64(values) => {
                let v = parse_f64(field).ok_or_else(|| Error::Parse {
                    column: column.to_string(),
                    line,
                    value: field.to_string(),
                    expected: DataType::Float64,
                })?;
                values.push(v);
            }
            Builder::Utf8 { dict, codes, index } => {
                // Look up by `&str` so only a first sighting allocates. High-cardinality
                // columns pay one allocation per distinct value, not one per row.
                let code = match index.get(field) {
                    Some(code) => *code,
                    None => {
                        let code = dict.len() as u32;
                        dict.push(field.to_string());
                        index.insert(field.to_string(), code);
                        code
                    }
                };
                codes.push(code);
            }
        }
        Ok(())
    }

    fn finish(self) -> Column {
        match self {
            Builder::Int64(values) => Column::Int64(values),
            Builder::Float64(values) => Column::Float64(values),
            Builder::Utf8 { dict, codes, .. } => Column::Utf8Dict {
                dict: dict.into(),
                codes,
            },
        }
    }
}

/// Load CSV into a `Table` using an explicit schema.
///
/// The schema supplies types; the CSV header supplies names. A value contradicting the
/// schema is an [`Error::Parse`] naming the column, file line, and offending text -- a case that
/// cannot arise when the schema came from [`infer_schema`] over the same data.
///
/// Columns absent from the schema fall back to `Utf8`, which is lossless: a partial schema
/// narrows the columns it mentions and leaves the rest readable rather than failing.
pub fn read_csv<R: Read>(reader: R, schema: &Schema) -> Result<Table> {
    let mut rdr = csv::Reader::from_reader(reader);
    let names = column_names(rdr.headers()?)?;

    let mut builders: Vec<Builder> = names
        .iter()
        .map(|name| Builder::new(schema.data_type(name).unwrap_or(DataType::Utf8)))
        .collect();

    // `csv::Reader` is non-flexible by default, so a row with the wrong field count is an
    // error rather than a silently short row.
    let mut nrows = 0usize;
    let mut record = csv::StringRecord::new();
    while rdr.read_record(&mut record)? {
        // The reader's own line number for this record, so an error points at the real file
        // line even though blank lines are skipped and never become rows.
        let line = record.position().map_or(0, |p| p.line());
        for (i, field) in record.iter().enumerate().take(names.len()) {
            builders[i].push(field, &names[i], line)?;
        }
        nrows += 1;
    }

    let mut columns = crate::hash::map_with_capacity(names.len());
    for (name, builder) in names.into_iter().zip(builders) {
        columns.insert(name, builder.finish());
    }

    Table::new(columns, nrows)
}

/// Load CSV from a path, inferring the schema first (two passes -- see the module docs).
pub fn read_csv_path<P: AsRef<Path>>(path: P) -> Result<Table> {
    let path = path.as_ref();
    let schema = infer_schema(BufReader::new(File::open(path)?))?;
    read_csv(BufReader::new(File::open(path)?), &schema)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mixed types plus a repeating string column, which is what dictionary encoding is for.
    const MIXED: &str = "\
region,amount,price
north,10,1.5
south,-20,2.25
north,30,3.0
east,40,4.75
north,50,5.5
";

    fn load(csv: &str) -> Table {
        let schema = infer_schema(csv.as_bytes()).expect("inference failed");
        read_csv(csv.as_bytes(), &schema).expect("load failed")
    }

    fn types(csv: &str) -> Vec<(String, DataType)> {
        infer_schema(csv.as_bytes())
            .unwrap()
            .fields
            .into_iter()
            .map(|f| (f.name, f.data_type))
            .collect()
    }

    #[test]
    fn infers_int_float_and_utf8() {
        assert_eq!(
            types(MIXED),
            vec![
                ("region".to_string(), DataType::Utf8),
                ("amount".to_string(), DataType::Int64),
                ("price".to_string(), DataType::Float64),
            ]
        );
    }

    #[test]
    fn loads_column_contents() {
        let t = load(MIXED);
        assert_eq!(t.nrows(), 5);
        assert_eq!(t.ncols(), 3);

        assert_eq!(
            t.column("amount").unwrap().as_i64(),
            Some(&[10i64, -20, 30, 40, 50][..])
        );
        assert_eq!(
            t.column("price").unwrap().as_f64(),
            Some(&[1.5f64, 2.25, 3.0, 4.75, 5.5][..])
        );
    }

    #[test]
    fn dictionary_holds_distinct_values_in_first_seen_order() {
        let t = load(MIXED);
        let (dict, codes) = t.column("region").unwrap().as_utf8_dict().unwrap();

        // Three distinct values across five rows -- the compression dictionary encoding buys.
        assert_eq!(
            dict,
            ["north".to_string(), "south".to_string(), "east".to_string()]
        );
        assert_eq!(codes, [0, 1, 0, 2, 0]);

        // Not sorted: "east" arrives last despite sorting first. `systemDesign.md` leans on
        // this to justify the `<`/`>` string-filter fallback.
        assert_ne!(dict[0], "east");
    }

    #[test]
    fn every_row_decodes_back_to_its_original_string() {
        let t = load(MIXED);
        let region = t.column("region").unwrap();
        let expected = ["north", "south", "north", "east", "north"];
        for (row, want) in expected.iter().enumerate() {
            assert_eq!(region.utf8_value(row), Some(*want));
        }
    }

    #[test]
    fn int_column_with_one_float_becomes_float() {
        let csv = "a\n1\n2\n3.5\n";
        assert_eq!(types(csv)[0].1, DataType::Float64);
        assert_eq!(
            load(csv).column("a").unwrap().as_f64(),
            Some(&[1.0f64, 2.0, 3.5][..])
        );
    }

    #[test]
    fn a_single_late_non_numeric_value_retypes_the_whole_column() {
        // The exact case that motivates two-pass inference: sampling the first rows would
        // have typed this Int64 and then failed on the last row.
        let csv = "a\n1\n2\nN/A\n";
        assert_eq!(types(csv)[0].1, DataType::Utf8);

        let t = load(csv);
        assert_eq!(t.column("a").unwrap().utf8_value(2), Some("N/A"));
    }

    #[test]
    fn empty_fields_are_strings_not_nulls() {
        // Documented consequence of having no NULL support -- see the module docs. The
        // blank is an ordinary dictionary entry, and it drags the column to Utf8 rather
        // than being coerced to 0.
        let csv = "a,b\n1,x\n,y\n3,z\n";
        assert_eq!(types(csv)[0].1, DataType::Utf8);

        let t = load(csv);
        assert_eq!(t.nrows(), 3);
        assert_eq!(t.column("a").unwrap().utf8_value(1), Some(""));

        let (dict, _) = t.column("a").unwrap().as_utf8_dict().unwrap();
        assert!(
            dict.contains(&String::new()),
            "blank should be a dict entry"
        );
    }

    #[test]
    fn wholly_blank_lines_are_skipped_not_read_as_empty_rows() {
        // `csv` treats a blank line as no record at all, so it does not contribute a row.
        // Pinned because it means nrows can be lower than the file's line count minus one,
        // which matters when reconciling against a reference tool in Phase 7.
        let t = load("a\n1\n\n3\n");
        assert_eq!(t.nrows(), 2);
        assert_eq!(t.column("a").unwrap().as_i64(), Some(&[1i64, 3][..]));
    }

    #[test]
    fn non_finite_text_does_not_become_float() {
        // "inf" and "NaN" both parse as f64; typing this column Float64 would smuggle NaN
        // into the engine, where it breaks the equality that grouping depends on.
        for csv in ["a\n1.5\ninf\n", "a\n1.5\nNaN\n"] {
            assert_eq!(types(csv)[0].1, DataType::Utf8, "csv: {csv:?}");
        }
    }

    #[test]
    fn numeric_fields_tolerate_surrounding_whitespace() {
        let csv = "a,b\n 1 , 2.5 \n";
        assert_eq!(
            types(csv),
            vec![
                ("a".to_string(), DataType::Int64),
                ("b".to_string(), DataType::Float64),
            ]
        );

        let t = load(csv);
        assert_eq!(t.column("a").unwrap().as_i64(), Some(&[1i64][..]));
        assert_eq!(t.column("b").unwrap().as_f64(), Some(&[2.5f64][..]));
    }

    #[test]
    fn string_values_are_stored_verbatim() {
        // Trimming applies only to the numeric parse attempt; Utf8 data keeps its spaces.
        let csv = "a\n x \ny\n";
        let t = load(csv);
        assert_eq!(t.column("a").unwrap().utf8_value(0), Some(" x "));
    }

    #[test]
    fn header_only_file_is_utf8_with_zero_rows() {
        let t = load("a,b\n");
        assert_eq!(t.nrows(), 0);
        assert_eq!(t.ncols(), 2);
        assert_eq!(t.column("a").unwrap().data_type(), DataType::Utf8);
        assert!(t.column("a").unwrap().is_empty());
    }

    #[test]
    fn duplicate_column_names_are_rejected() {
        // Table keys columns by name, so this would otherwise silently drop a column.
        let err = infer_schema("a,a\n1,2\n".as_bytes()).unwrap_err();
        match err {
            Error::DuplicateColumn(name) => assert_eq!(name, "a"),
            other => panic!("expected DuplicateColumn, got {other:?}"),
        }
    }

    #[test]
    fn ragged_rows_are_rejected() {
        let err = infer_schema("a,b\n1,2\n3\n".as_bytes()).unwrap_err();
        assert!(matches!(err, Error::Csv(_)), "got {err:?}");
    }

    #[test]
    fn explicit_schema_contradicted_by_data_reports_where() {
        let csv = "a\n1\noops\n";
        let schema = Schema::new(vec![Field::new("a", DataType::Int64)]);

        match read_csv(csv.as_bytes(), &schema).unwrap_err() {
            Error::Parse {
                column,
                line,
                value,
                expected,
            } => {
                assert_eq!(column, "a");
                assert_eq!(line, 3, "header is line 1, so the bad value sits on line 3");
                assert_eq!(value, "oops");
                assert_eq!(expected, DataType::Int64);
            }
            other => panic!("expected Parse, got {other:?}"),
        }
    }

    #[test]
    fn reported_line_does_not_count_blank_lines() {
        // Discovered by writing this test expecting 5. The csv reader's line counter skips
        // wholly blank lines just as its record iterator does, so a file containing them
        // reports lower than the physical editor line. The physical number is not available
        // through the reader's API -- blank lines are consumed invisibly -- so
        // `Error::Parse::line` documents this limit rather than claiming a precision it
        // cannot deliver.
        //
        // "oops" sits on physical line 5 here, and is reported as line 3.
        let csv = "a\n1\n\n\noops\n";
        let schema = Schema::new(vec![Field::new("a", DataType::Int64)]);

        match read_csv(csv.as_bytes(), &schema).unwrap_err() {
            Error::Parse { line, value, .. } => {
                assert_eq!(value, "oops");
                assert_eq!(line, 3, "blank lines are not counted");
            }
            other => panic!("expected Parse, got {other:?}"),
        }
    }

    #[test]
    fn explicit_schema_may_cover_only_some_columns() {
        let csv = "a,b\n1,2\n";
        let schema = Schema::new(vec![Field::new("b", DataType::Int64)]);
        let t = read_csv(csv.as_bytes(), &schema).unwrap();

        assert_eq!(t.column("b").unwrap().as_i64(), Some(&[2i64][..]));
        // Unmentioned columns fall back to Utf8, which is lossless.
        assert_eq!(t.column("a").unwrap().data_type(), DataType::Utf8);
    }

    #[test]
    fn reads_from_a_real_file_path() {
        // Covers the two-open path in `read_csv_path`, which the in-memory tests bypass.
        let path = std::env::temp_dir().join("batchbird_loader_test.csv");
        std::fs::write(&path, MIXED).unwrap();

        let from_path = read_csv_path(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(from_path, load(MIXED));
    }
}
