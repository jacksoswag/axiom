//! Growth / activation functions (§2.5) — the map from potential `U` to a
//! signed growth increment in `[-1, 1]`. `μ` (center) and `σ` (width) are the
//! primary "species" axis of Lenia.

#[derive(Debug, Clone, Copy)]
pub enum GrowthKind {
    Gauss,
    Poly,
    Exp,
}

impl GrowthKind {
    pub fn parse(s: &str) -> Self {
        match s {
            "poly" => GrowthKind::Poly,
            "exp" => GrowthKind::Exp,
            _ => GrowthKind::Gauss,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Growth {
    pub kind: GrowthKind,
    pub mu: f32,
    pub sigma: f32,
}

impl Growth {
    #[inline]
    pub fn apply(&self, u: f32) -> f32 {
        match self.kind {
            // Gaussian bell mapped to [-1, 1].
            GrowthKind::Gauss => 2.0 * (-0.5 * ((u - self.mu) / self.sigma).powi(2)).exp() - 1.0,
            // Polynomial bump, same support, cheaper tails.
            GrowthKind::Poly => {
                let d = (u - self.mu).abs() / (3.0 * self.sigma);
                if d >= 1.0 { -1.0 } else { 2.0 * (1.0 - d * d).powi(4) - 1.0 }
            }
            // One-sided exponential (reaction-style).
            GrowthKind::Exp => 2.0 * (-((u - self.mu).abs() / self.sigma)).exp() - 1.0,
        }
    }
}
