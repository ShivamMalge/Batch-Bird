//! A fixed-seed hasher, so a run is reproducible.
//!
//! # Why not the default
//! `std::collections::HashMap` and `hashbrown::HashMap` both seed their hashers randomly per
//! process. That is the right default for a server parsing untrusted input, where predictable
//! hashing invites collision-flooding attacks. It is the wrong default here, and measurably so.
//!
//! A random seed means every run gets a different collision pattern, a different probe
//! sequence, and a different iteration order. Measured on this project: two runs of the *same
//! binary* on the *same data* differed by a median of 7.5% and a maximum of 66.8%. Pinning to
//! one core and capping the clock brought that to 3.1% / 19.7%, and taking the minimum of N
//! samples instead of the mean brought it to 2.6% / 8.8% — but neither fixes it, because a
//! different seed is not interference, it is genuinely different work. Only a fixed seed does.
//!
//! The trade is real and named: **variance for bias.** One fixed seed is one particular
//! collision pattern, which might be luckier or unluckier than average. `examples/seed_sweep.rs`
//! measures that bias across several seeds so it is a reported number rather than a hidden
//! assumption.
//!
//! Safe here because the crate has `publish = false`, takes no untrusted input, and exists to
//! be measured. Determinism beats DoS resistance for exactly this workload and no other.
//!
//! # The algorithm
//! FxHash, the hasher rustc uses internally, for the same reason: it is very fast on the small
//! integer keys that dominate here (`GroupKey` is a `u64`), and it is deterministic. It is not
//! a strong hash and is not meant to be.

use std::hash::{BuildHasher, Hasher};

/// The multiplier from rustc's FxHash — the fractional bits of the golden ratio.
const MULTIPLIER: u64 = 0x517c_c1b7_2722_0a95;

/// The project-wide seed. Arbitrary, but fixed forever: changing it changes every collision
/// pattern and invalidates comparisons against previously recorded timings.
pub const DEFAULT_SEED: u64 = 0xba7c_b18d_0000_0001;

#[derive(Clone, Copy, Default)]
pub struct FxHasher {
    hash: u64,
}

impl FxHasher {
    #[inline]
    fn add(&mut self, word: u64) {
        self.hash = (self.hash.rotate_left(5) ^ word).wrapping_mul(MULTIPLIER);
    }
}

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let (chunks, tail) = bytes.as_chunks::<8>();
        for chunk in chunks {
            // Little-endian rather than native, so a hash is reproducible across machines and
            // not only across runs on this one.
            self.add(u64::from_le_bytes(*chunk));
        }
        if !tail.is_empty() {
            let mut buffer = [0u8; 8];
            buffer[..tail.len()].copy_from_slice(tail);
            self.add(u64::from_le_bytes(buffer));
        }
    }

    #[inline]
    fn write_u8(&mut self, value: u8) {
        self.add(value as u64);
    }

    #[inline]
    fn write_u32(&mut self, value: u32) {
        self.add(value as u64);
    }

    #[inline]
    fn write_u64(&mut self, value: u64) {
        self.add(value);
    }

    #[inline]
    fn write_usize(&mut self, value: usize) {
        self.add(value as u64);
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }
}

/// A `BuildHasher` with a seed chosen at construction rather than from the OS.
#[derive(Clone, Copy)]
pub struct FixedState {
    seed: u64,
}

impl FixedState {
    /// For the seed sweep. Everything in the engine uses [`FixedState::default`].
    pub const fn with_seed(seed: u64) -> Self {
        FixedState { seed }
    }
}

impl Default for FixedState {
    fn default() -> Self {
        FixedState::with_seed(DEFAULT_SEED)
    }
}

impl BuildHasher for FixedState {
    type Hasher = FxHasher;

    #[inline]
    fn build_hasher(&self) -> FxHasher {
        FxHasher { hash: self.seed }
    }
}

/// The one map type used throughout the crate.
///
/// `hashbrown` rather than `std` because std's SipHash would dominate the group-by timing
/// (`techstack.md`), and one alias rather than two so there is a single answer to "which map".
pub type Map<K, V> = hashbrown::HashMap<K, V, FixedState>;

/// A map with the default fixed seed.
pub fn map<K, V>() -> Map<K, V> {
    Map::with_hasher(FixedState::default())
}

/// A map with the default fixed seed and room for `capacity` entries.
pub fn map_with_capacity<K, V>(capacity: usize) -> Map<K, V> {
    Map::with_capacity_and_hasher(capacity, FixedState::default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::hash::Hash;

    fn hash_of<T: Hash>(value: &T) -> u64 {
        FixedState::default().hash_one(value)
    }

    #[test]
    fn hashing_is_stable_within_a_process() {
        assert_eq!(hash_of(&42u64), hash_of(&42u64));
        assert_eq!(hash_of(&"region"), hash_of(&"region"));
    }

    #[test]
    fn different_values_hash_differently() {
        assert_ne!(hash_of(&1u64), hash_of(&2u64));
        assert_ne!(hash_of(&"north"), hash_of(&"south"));
    }

    #[test]
    fn the_seed_actually_changes_the_hash() {
        // What makes the seed sweep meaningful: a different seed really is a different
        // collision pattern, not the same one relabelled.
        let a = FixedState::with_seed(1).hash_one(7u64);
        let b = FixedState::with_seed(2).hash_one(7u64);
        assert_ne!(a, b);
    }

    #[test]
    fn iteration_order_is_deterministic_for_a_given_seed() {
        // The property the whole module exists for. Two maps built identically must iterate
        // identically -- with the stdlib default this fails across processes, which is what
        // made benchmark runs incomparable.
        let build = || {
            let mut m: Map<String, usize> = map();
            for i in 0..32 {
                m.insert(format!("r{i}"), i);
            }
            m.keys().cloned().collect::<Vec<_>>()
        };
        assert_eq!(build(), build());
    }

    #[test]
    fn known_hashes_are_pinned() {
        // Locks the algorithm and the default seed together. If either changes, every
        // previously recorded benchmark number becomes incomparable, so that should be a
        // deliberate act that breaks a test rather than a silent drift.
        assert_eq!(hash_of(&0u64), 12_207_218_417_661_396_483);
        assert_eq!(hash_of(&1u64), 6_335_437_411_097_394_030);
    }

    #[test]
    fn byte_and_integer_paths_both_work() {
        let mut m: Map<u64, &str> = map();
        m.insert(1, "one");
        assert_eq!(m.get(&1), Some(&"one"));

        let mut s: Map<String, u32> = map_with_capacity(4);
        s.insert("north".to_string(), 0);
        assert_eq!(s.get("north"), Some(&0));
    }
}
