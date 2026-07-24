//! Primitives shared by every half of the crate, with no natural owner among them.

/// FNV-1a, the crate's one hash.
///
/// Genome identity, archive keys, campaign checksums, the tuning digest, world state hashes,
/// and the renderer's snapshot fingerprint all use it. Several of those are written into durable
/// formats, so the constants and the fold order here are a compatibility surface: changing
/// either invalidates every archive, campaign state, and checkpoint already on disk.
pub(crate) struct Fnv(u64);

const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const PRIME: u64 = 0x0000_0100_0000_01b3;

impl Fnv {
    pub(crate) fn new() -> Fnv {
        Fnv(OFFSET)
    }

    /// Continue from a digest produced earlier, for callers that thread a running hash.
    #[cfg(feature = "viewer")]
    pub(crate) fn resume(hash: u64) -> Fnv {
        Fnv(hash)
    }

    pub(crate) fn byte(&mut self, byte: u8) {
        self.0 ^= byte as u64;
        self.0 = self.0.wrapping_mul(PRIME);
    }

    pub(crate) fn bytes(&mut self, bytes: impl IntoIterator<Item = u8>) {
        for byte in bytes {
            self.byte(byte);
        }
    }

    /// Absorb a whole word in one step. This is *not* the same as folding its eight bytes, and a
    /// caller that needs one form cannot substitute the other without changing the hash.
    pub(crate) fn word(&mut self, word: u64) {
        self.0 ^= word;
        self.0 = self.0.wrapping_mul(PRIME);
    }

    pub(crate) fn float(&mut self, value: f32) {
        self.bytes(value.to_bits().to_le_bytes());
    }

    pub(crate) fn finish(&self) -> u64 {
        self.0
    }
}

impl Default for Fnv {
    fn default() -> Fnv {
        Fnv::new()
    }
}

/// Replace a non-finite value with a fallback, so one `NaN` gene cannot poison a decode.
pub(crate) fn finite(value: f32, fallback: f32) -> f32 {
    if value.is_finite() {
        value
    } else {
        fallback
    }
}

/// Deterministic xorshift64. Seeding and mutation are the only things needing randomness, and reproducibility
/// matters more than statistical quality. Fitness must be deterministic or the search optimizes noise.
pub struct Rng { state: u64 }

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng { state: seed.max(1) ^ 0x9e3779b97f4a7c15 }
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
        self.state = x; x.wrapping_mul(0x2545f4914f6cdd1d)
    }
    /// Uniform on [0, 1)
    pub fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    /// Uniform on [low, high)
    pub fn range(&mut self, low: f32, high: f32) -> f32 {
        low + self.unit() * (high - low)
    }
    pub fn below(&mut self, limit: usize) -> usize {
        if limit == 0 { 0 } else { (self.next_u64() % limit as u64) as usize }
    }
    /// A standard normal distribution treats two uniform draws as a radius and angle, so their
    /// combo lands on a Gaussian bell instead of a flat range. Sample can't be 0 to prevent ln(0).
    pub fn normal(&mut self) -> f32 {
        let u1 = self.unit().max(1e-7); let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}