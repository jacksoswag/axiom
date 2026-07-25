//! Primitives shared by every half of the crate, with no natural owner among them.

/// Replace a non-finite value with a fallback, so one NaN gene cannot poison a decode.
pub fn finite(value: f32, fallback: f32) -> f32 { if value.is_finite() { value } else { fallback } }

/// Straight Euclidean distance. Flat space, so not interchangeable with the engine's periodic
/// distance_sq, which folds every axis into a box and adds a softening term.
pub fn distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().enumerate().map(|(k, x)| (x - b[k]).powi(2)).sum::<f32>().sqrt()
}

/// Deterministic xorshift64. Reproducibility matters more than statistical quality: fitness must be
/// deterministic or the search optimizes noise.
pub struct Rng { state: u64 }
impl Rng {
    pub fn new(seed: u64) -> Rng { Rng { state: seed.max(1) ^ 0x9e37_79b9_7f4a_7c15 } }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
        self.state = x; x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    /// Uniform on [0, 1)
    pub fn unit(&mut self) -> f32 { (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32 }
    /// Uniform on [low, high)
    pub fn range(&mut self, low: f32, high: f32) -> f32 { low + self.unit() * (high - low) }
    pub fn below(&mut self, limit: usize) -> usize {
        if limit == 0 { 0 } else { (self.next_u64() % limit as u64) as usize }
    }
    /// Two uniform draws read as a radius and an angle land on a Gaussian bell instead of a flat
    /// range. The radius draw is floored to keep ln(0) out.
    pub fn normal(&mut self) -> f32 {
        let (u1, u2) = (self.unit().max(1e-7), self.unit());
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}
