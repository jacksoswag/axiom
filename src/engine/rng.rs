//! Deterministic xorshift64. Seeding and mutation are the only things needing randomness, and reproducibility 
//! matters more than statistical quality. Fitness must be deterministic or the search optimizes noise.

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
