//! `Column` — the unit of columnar storage: one type, one contiguous buffer.
//!
//! Contiguity is the whole point. A row-oriented layout interleaves a row's fields in
//! memory, so scanning one field touches every cache line; a column puts the values a scan
//! actually reads next to each other. That is what the Phase 6 benchmark measures, and it
//! is also the precondition for SIMD in Phase 5 — you cannot load 8 lanes at once out of a
//! struct-of-fields layout.

use std::fmt;

/// The three types in scope. No `Bool` — see `prd.md` Non-Goals: no source data in scope is
/// boolean-shaped, and the only boolean artifact in the engine is the filter mask, which is
/// a bitset rather than a stored column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    Int64,
    Float64,
    Utf8,
}

impl fmt::Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            DataType::Int64 => "Int64",
            DataType::Float64 => "Float64",
            DataType::Utf8 => "Utf8",
        };
        f.write_str(s)
    }
}

/// A typed, contiguous run of values.
///
/// There is no null/missing representation — see `Utf8Dict` below and the loader docs.
#[derive(Debug, Clone, PartialEq)]
pub enum Column {
    Int64(Vec<i64>),
    Float64(Vec<f64>),

    /// Dictionary-encoded strings: `codes[row]` indexes into `dict`.
    ///
    /// Two payoffs, both load-bearing later:
    /// - **Group-by** hashes a `u32` code instead of a string — integer hashing and equality
    ///   rather than pointer-chasing and byte comparison, per row (`systemDesign.md`).
    /// - **Equality filters** resolve the literal to a code once, then compare integers.
    ///
    /// The dictionary is in **first-seen order, not sorted**. That is a deliberate cost:
    /// sorting it would make code order match lexical order and let `<`/`>` filters compare
    /// codes, but sorting is out of scope, so ordering filters fall back to comparing the
    /// strings themselves (`systemDesign.md` "String Columns").
    Utf8Dict {
        dict: Vec<String>,
        codes: Vec<u32>,
    },
}

impl Column {
    pub fn len(&self) -> usize {
        match self {
            Column::Int64(v) => v.len(),
            Column::Float64(v) => v.len(),
            Column::Utf8Dict { codes, .. } => codes.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn data_type(&self) -> DataType {
        match self {
            Column::Int64(_) => DataType::Int64,
            Column::Float64(_) => DataType::Float64,
            Column::Utf8Dict { .. } => DataType::Utf8,
        }
    }

    /// Borrow the raw values. These return slices rather than iterators precisely because
    /// Phase 5 needs a contiguous `&[T]` to feed `Simd::from_slice`.
    pub fn as_i64(&self) -> Option<&[i64]> {
        match self {
            Column::Int64(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<&[f64]> {
        match self {
            Column::Float64(v) => Some(v),
            _ => None,
        }
    }

    /// Borrow the dictionary and the per-row codes.
    pub fn as_utf8_dict(&self) -> Option<(&[String], &[u32])> {
        match self {
            Column::Utf8Dict { dict, codes } => Some((dict, codes)),
            _ => None,
        }
    }

    /// Resolve one row of a `Utf8Dict` column back to its string.
    ///
    /// Decoding is for correctness checks and for materializing results — never for the hot
    /// loops, which stay on codes.
    pub fn utf8_value(&self, row: usize) -> Option<&str> {
        match self {
            Column::Utf8Dict { dict, codes } => {
                let code = *codes.get(row)? as usize;
                dict.get(code).map(String::as_str)
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn utf8(values: &[&str]) -> Column {
        let mut dict: Vec<String> = Vec::new();
        let mut codes = Vec::new();
        for v in values {
            let code = match dict.iter().position(|d| d == v) {
                Some(i) => i as u32,
                None => {
                    dict.push((*v).to_string());
                    (dict.len() - 1) as u32
                }
            };
            codes.push(code);
        }
        Column::Utf8Dict { dict, codes }
    }

    #[test]
    fn len_and_type_agree_across_variants() {
        assert_eq!(Column::Int64(vec![1, 2, 3]).len(), 3);
        assert_eq!(Column::Int64(vec![]).data_type(), DataType::Int64);
        assert_eq!(Column::Float64(vec![1.0]).data_type(), DataType::Float64);

        // A dictionary column's length is its row count, NOT its distinct-value count.
        let c = utf8(&["a", "b", "a", "a"]);
        assert_eq!(c.len(), 4);
        assert_eq!(c.data_type(), DataType::Utf8);
        assert_eq!(c.as_utf8_dict().unwrap().0.len(), 2);
    }

    #[test]
    fn accessors_are_type_gated() {
        let i = Column::Int64(vec![7]);
        assert_eq!(i.as_i64(), Some(&[7i64][..]));
        assert!(i.as_f64().is_none());
        assert!(i.as_utf8_dict().is_none());
        assert!(i.utf8_value(0).is_none());
    }

    #[test]
    fn utf8_decodes_back_to_original_values() {
        let values = ["north", "south", "north", "east", "north"];
        let c = utf8(&values);
        for (row, expected) in values.iter().enumerate() {
            assert_eq!(c.utf8_value(row), Some(*expected));
        }
        assert_eq!(c.utf8_value(values.len()), None, "out-of-range row");
    }

    #[test]
    fn dictionary_is_first_seen_order_not_sorted() {
        // Load-bearing: `systemDesign.md` justifies the `<`/`>` string-filter fallback on
        // exactly this property. If this ever starts passing sorted, that scope cut is
        // no longer needed and the doc is stale.
        let c = utf8(&["zebra", "apple", "zebra"]);
        let (dict, codes) = c.as_utf8_dict().unwrap();
        assert_eq!(dict, ["zebra".to_string(), "apple".to_string()]);
        assert_eq!(codes, [0, 1, 0]);
    }
}
