//! A packed bitset: one bit per row, 64 rows per `u64` word.
//!
//! # Why not `Vec<bool>`
//! Rust's `Vec<bool>` is one *byte* per element, so it has no memory-bandwidth advantage over
//! a plain byte mask (`systemDesign.md`). Since the benchmark's whole point is demonstrating a
//! bandwidth effect, only a genuinely bit-packed mask can show it: 1024 rows fit in 128 bytes
//! -- two cache lines -- instead of a kilobyte.
//!
//! # Why words are written in bulk
//! [`Bitset::words_mut`] exists so a filter can build 64 results, OR them into one `u64`, and
//! store it once. That is exactly the shape a SIMD comparison produces in Phase 5 (a compare
//! yields a lane mask, which becomes bits), so the scalar and vector filters can write through
//! the same interface rather than the SIMD path needing a different mask type.

/// Bits beyond `len` are always zero, which is what lets [`count_ones`](Bitset::count_ones)
/// and [`iter_ones`](Bitset::iter_ones) read whole words without masking the tail.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bitset {
    words: Vec<u64>,
    len: usize,
}

impl Bitset {
    /// A bitset of `len` bits, all clear.
    pub fn new(len: usize) -> Self {
        Bitset {
            words: vec![0; len.div_ceil(64)],
            len,
        }
    }

    /// Number of bits (rows), not words.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn set(&mut self, index: usize) {
        debug_assert!(
            index < self.len,
            "bit {index} is past the end ({})",
            self.len
        );
        self.words[index / 64] |= 1u64 << (index % 64);
    }

    #[inline]
    pub fn get(&self, index: usize) -> bool {
        debug_assert!(
            index < self.len,
            "bit {index} is past the end ({})",
            self.len
        );
        self.words[index / 64] >> (index % 64) & 1 == 1
    }

    /// How many rows survived. One `popcnt` per 64 rows rather than a per-row branch, which
    /// is what lets `Filter` size the compacted batch exactly and allocate once.
    pub fn count_ones(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// The set bit positions, ascending.
    ///
    /// Skips 64 clear rows at a time and finds each set bit with a `trailing_zeros`, so a
    /// highly selective filter costs roughly one instruction per *surviving* row instead of
    /// one per scanned row.
    pub fn iter_ones(&self) -> IterOnes<'_> {
        IterOnes {
            words: &self.words,
            word: self.words.first().copied().unwrap_or(0),
            index: 0,
        }
    }

    /// Raw word access for bulk construction.
    ///
    /// The caller must leave bits at or beyond [`len`](Bitset::len) clear; every read path
    /// here assumes it.
    pub fn words_mut(&mut self) -> &mut [u64] {
        &mut self.words
    }

    pub fn words(&self) -> &[u64] {
        &self.words
    }
}

pub struct IterOnes<'a> {
    words: &'a [u64],
    /// The current word with already-yielded bits cleared.
    word: u64,
    index: usize,
}

impl Iterator for IterOnes<'_> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        loop {
            if self.word != 0 {
                let bit = self.word.trailing_zeros() as usize;
                // Clear the lowest set bit: the standard `x & (x - 1)` trick.
                self.word &= self.word - 1;
                return Some(self.index * 64 + bit);
            }
            self.index += 1;
            self.word = *self.words.get(self.index)?;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starts_clear() {
        let bits = Bitset::new(100);
        assert_eq!(bits.len(), 100);
        assert_eq!(bits.count_ones(), 0);
        assert!(bits.iter_ones().next().is_none());
        assert!((0..100).all(|i| !bits.get(i)));
    }

    #[test]
    fn packs_64_rows_per_word() {
        // The whole reason for this type: 1024 rows in 128 bytes, not 1024.
        assert_eq!(Bitset::new(64).words().len(), 1);
        assert_eq!(Bitset::new(65).words().len(), 2);
        assert_eq!(Bitset::new(1024).words().len(), 16);
        assert_eq!(Bitset::new(0).words().len(), 0);
    }

    #[test]
    fn set_bits_round_trip() {
        let mut bits = Bitset::new(200);
        let set = [0, 1, 63, 64, 65, 127, 128, 199];
        for i in set {
            bits.set(i);
        }

        assert_eq!(bits.count_ones(), set.len());
        assert_eq!(bits.iter_ones().collect::<Vec<_>>(), set);
        assert!(set.iter().all(|i| bits.get(*i)));
        assert!(!bits.get(2));
    }

    #[test]
    fn iterates_across_word_boundaries_in_order() {
        // Crossing 64 is where an off-by-one in the word/bit split would show up.
        let mut bits = Bitset::new(130);
        for i in 60..70 {
            bits.set(i);
        }
        assert_eq!(
            bits.iter_ones().collect::<Vec<_>>(),
            (60..70).collect::<Vec<_>>()
        );
    }

    #[test]
    fn bulk_word_construction_matches_setting_bits() {
        // The path a SIMD filter will use in Phase 5.
        let mut bulk = Bitset::new(128);
        bulk.words_mut()[0] = 0b1011;
        bulk.words_mut()[1] = 1 << 5;

        let mut individual = Bitset::new(128);
        for i in [0, 1, 3, 69] {
            individual.set(i);
        }

        assert_eq!(bulk, individual);
        assert_eq!(bulk.iter_ones().collect::<Vec<_>>(), vec![0, 1, 3, 69]);
    }

    #[test]
    fn all_ones_is_dense_and_ordered() {
        let mut bits = Bitset::new(1024);
        for i in 0..1024 {
            bits.set(i);
        }
        assert_eq!(bits.count_ones(), 1024);
        assert_eq!(bits.iter_ones().count(), 1024);
        assert_eq!(bits.iter_ones().last(), Some(1023));
    }

    #[test]
    fn an_empty_bitset_is_well_behaved() {
        let bits = Bitset::new(0);
        assert!(bits.is_empty());
        assert_eq!(bits.count_ones(), 0);
        assert_eq!(bits.iter_ones().count(), 0);
    }
}
