//! Perception kernels and toroidal convolution.
//!
//! The kernel *core* is pluggable per §2.3 ("Never hardcode Lenia's default").
//! A kernel is built from a radius `R`, a ring skeleton `β`, and a core function
//! evaluated on the normalised radius `r ∈ [0,1]`, then sum-normalised so the
//! potential `U = K * A` stays in a comparable range regardless of `R`.

use rayon::prelude::*;

/// Which analytic shape the kernel rings use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoreKind {
    /// Gaussian bump on each ring — the classic Lenia shell.
    GaussRing,
    /// Smooth polynomial bump `(4 r (1-r))^4`.
    Poly,
    /// Hard step ring (Larger-than-Life style).
    Step,
}

impl CoreKind {
    pub fn parse(s: &str) -> Self {
        match s {
            "poly" | "quad4" => CoreKind::Poly,
            "step" => CoreKind::Step,
            _ => CoreKind::GaussRing,
        }
    }
}

#[inline]
fn core(kind: CoreKind, r: f32, mu: f32, sigma: f32) -> f32 {
    match kind {
        CoreKind::GaussRing => (-0.5 * ((r - mu) / sigma).powi(2)).exp(),
        CoreKind::Poly => {
            let x = 4.0 * r * (1.0 - r);
            if x <= 0.0 { 0.0 } else { x.powi(4) }
        }
        CoreKind::Step => {
            if (0.25..=0.75).contains(&r) { 1.0 } else { 0.0 }
        }
    }
}

/// A precomputed square kernel, row-major over `(2R+1)^2`, sum-normalised to 1.
#[derive(Debug, Clone)]
pub struct Kernel {
    pub radius: usize,
    pub weights: Vec<f32>,
}

pub struct KernelParams<'a> {
    pub radius: usize,
    pub core: CoreKind,
    /// Ring skeleton β — one weight per concentric ring.
    pub beta: &'a [f32],
    pub core_mu: f32,
    pub core_sigma: f32,
}

impl Kernel {
    pub fn build(p: &KernelParams) -> Kernel {
        let r = p.radius as i32;
        let size = (2 * p.radius + 1) as i32;
        let mut weights = vec![0.0f32; (size * size) as usize];
        let bands = p.beta.len().max(1);
        let mut sum = 0.0f64;
        for dy in -r..=r {
            for dx in -r..=r {
                let dist = ((dx * dx + dy * dy) as f32).sqrt();
                let rn = if r > 0 { dist / r as f32 } else { 0.0 };
                let val = if rn <= 1.0 {
                    let br = rn * bands as f32;
                    let ring = (br.floor() as usize).min(bands - 1);
                    let frac = br - br.floor();
                    let beta = p.beta.get(ring).copied().unwrap_or(1.0);
                    beta * core(p.core, frac, p.core_mu, p.core_sigma)
                } else {
                    0.0
                };
                let ix = ((dy + r) * size + (dx + r)) as usize;
                weights[ix] = val;
                sum += val as f64;
            }
        }
        if sum > 0.0 {
            let inv = (1.0 / sum) as f32;
            for v in weights.iter_mut() {
                *v *= inv;
            }
        }
        Kernel { radius: p.radius, weights }
    }
}

/// Direct convolution with wrap-around (torus) or zero (bounded) boundaries.
///
/// Kernels are small (`R ≈ 13`), so a direct `O(N·k²)` pass parallelised across
/// rows beats FFT setup cost at this scale and is the correctness oracle FFT would
/// later be validated against (§5.2). Output length must equal `h*w`.
pub fn convolve(input: &[f32], h: usize, w: usize, k: &Kernel, torus: bool, out: &mut [f32]) {
    let r = k.radius as i32;
    let ks = (2 * k.radius + 1) as i32;
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let mut acc = 0.0f32;
            for dy in -r..=r {
                let yy = y as i32 + dy;
                let yy = if torus {
                    yy.rem_euclid(h as i32)
                } else if yy < 0 || yy >= h as i32 {
                    continue;
                } else {
                    yy
                };
                let base = yy as usize * w;
                let krow = ((dy + r) * ks) as usize;
                for dx in -r..=r {
                    let xx = x as i32 + dx;
                    let xx = if torus {
                        xx.rem_euclid(w as i32)
                    } else if xx < 0 || xx >= w as i32 {
                        continue;
                    } else {
                        xx
                    };
                    acc += input[base + xx as usize] * k.weights[krow + (dx + r) as usize];
                }
            }
            row[x] = acc;
        }
    });
}
