//! Particle substrate (§2.1, §3) — a swarm is a graph with proximity edges.
//!
//! Continuous-position particles with a preferred-distance attraction (a Lenia
//! kernel bell) plus short-range repulsion, on a torus world. They self-organize
//! into stable cells/lattices, and their proximity graph feeds the same PageRank
//! observer used on grids and graphs — the particle realization of the seam.
//!
//! ponytail: cohesion/separation swarm with a Lenia-bell attraction kernel; the
//! full energy-based Particle-Lenia (growth field + repulsion energy descent) is
//! the upgrade path.

use crate::substrate::Xorshift;
use rayon::prelude::*;

pub struct Particles {
    pub s: f32,
    pub pos: Vec<(f32, f32)>,
}

impl Particles {
    pub fn new(n: usize, s: f32, seed: u64) -> Particles {
        let mut rng = Xorshift::new(seed);
        let pos = (0..n).map(|_| (rng.unit() * s, rng.unit() * s)).collect();
        Particles { s, pos }
    }

    #[inline]
    fn delta(&self, a: (f32, f32), b: (f32, f32)) -> (f32, f32, f32) {
        // Minimal-image displacement b→a on the torus.
        let mut dx = a.0 - b.0;
        let mut dy = a.1 - b.1;
        if dx > self.s * 0.5 { dx -= self.s } else if dx < -self.s * 0.5 { dx += self.s }
        if dy > self.s * 0.5 { dy -= self.s } else if dy < -self.s * 0.5 { dy += self.s }
        (dx, dy, (dx * dx + dy * dy).sqrt())
    }

    pub fn step(&mut self, dt: f32, k_mu: f32, k_sigma: f32, attract: f32, repel: f32, rep_r: f32) {
        let n = self.pos.len();
        let pos = &self.pos;
        let forces: Vec<(f32, f32)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let (mut fx, mut fy) = (0.0f32, 0.0f32);
                for j in 0..n {
                    if i == j {
                        continue;
                    }
                    let (dx, dy, d) = self.delta(pos[i], pos[j]);
                    if d < 1e-4 || d > k_mu + 4.0 * k_sigma {
                        continue;
                    }
                    let inv = 1.0 / d;
                    // Attraction peaks at the preferred distance k_mu.
                    let att = attract * (-0.5 * ((d - k_mu) / k_sigma).powi(2)).exp();
                    // Short-range repulsion.
                    let rep = repel * (-d * d / (rep_r * rep_r)).exp();
                    let mag = rep - att; // >0 pushes apart, <0 pulls together
                    fx += mag * dx * inv;
                    fy += mag * dy * inv;
                }
                (fx, fy)
            })
            .collect();
        for (p, f) in self.pos.iter_mut().zip(forces) {
            p.0 = (p.0 + dt * f.0).rem_euclid(self.s);
            p.1 = (p.1 + dt * f.1).rem_euclid(self.s);
        }
    }

    /// Particle-Lenia energy descent (§2.4). Each particle descends
    /// `E = R − G(U)`: a growth field `G` over the kernel potential `U` pulls
    /// particles together at the preferred distance, a short-range repulsion `R`
    /// keeps them apart. Self-organizes into stable cells. `O(N²)`.
    #[allow(clippy::too_many_arguments)]
    pub fn step_lenia(&mut self, dt: f32, k_mu: f32, k_sigma: f32, g_mu: f32, g_sigma: f32, rep_c: f32, rep_sigma: f32) {
        let n = self.pos.len();
        let pos = &self.pos;
        // Potential U at each particle (sum of distance-bells over the others).
        let u: Vec<f32> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut acc = 0.0f32;
                for j in 0..n {
                    if i == j {
                        continue;
                    }
                    let (_, _, d) = self.delta(pos[i], pos[j]);
                    acc += (-0.5 * ((d - k_mu) / k_sigma).powi(2)).exp();
                }
                acc
            })
            .collect();
        let forces: Vec<(f32, f32)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let gp = -(u[i] - g_mu) / (g_sigma * g_sigma) * (-0.5 * ((u[i] - g_mu) / g_sigma).powi(2)).exp();
                let (mut fx, mut fy) = (0.0f32, 0.0f32);
                for j in 0..n {
                    if i == j {
                        continue;
                    }
                    let (dx, dy, d) = self.delta(pos[i], pos[j]);
                    if d < 1e-4 {
                        continue;
                    }
                    let inv = 1.0 / d;
                    // ∇U contribution: dK/dd along the separation.
                    let kd = -(d - k_mu) / (k_sigma * k_sigma) * (-0.5 * ((d - k_mu) / k_sigma).powi(2)).exp();
                    // ∇R contribution: short-range repulsion.
                    let rd = rep_c * (-2.0 * d / (rep_sigma * rep_sigma)) * (-(d * d) / (rep_sigma * rep_sigma)).exp();
                    // Force = −∇E = −∇R + G'(U)·∇U, projected onto the unit separation.
                    let mag = -rd + gp * kd;
                    fx += mag * dx * inv;
                    fy += mag * dy * inv;
                }
                (fx, fy)
            })
            .collect();
        for (p, f) in self.pos.iter_mut().zip(forces) {
            p.0 = (p.0 + dt * f.0).rem_euclid(self.s);
            p.1 = (p.1 + dt * f.1).rem_euclid(self.s);
        }
    }

    /// Per-particle local features for a learned rule: `[U, |∇U|, neighbor_count]`
    /// plus a bias term, computed over a neighborhood radius.
    fn features(&self, radius: f32, k_mu: f32, k_sigma: f32) -> Vec<[f32; 4]> {
        let n = self.pos.len();
        let pos = &self.pos;
        (0..n)
            .into_par_iter()
            .map(|i| {
                let (mut u, mut gx, mut gy, mut cnt) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                for j in 0..n {
                    if i == j {
                        continue;
                    }
                    let (dx, dy, d) = self.delta(pos[i], pos[j]);
                    if d > radius || d < 1e-4 {
                        continue;
                    }
                    let inv = 1.0 / d;
                    u += (-0.5 * ((d - k_mu) / k_sigma).powi(2)).exp();
                    let kd = -(d - k_mu) / (k_sigma * k_sigma) * (-0.5 * ((d - k_mu) / k_sigma).powi(2)).exp();
                    gx += kd * dx * inv;
                    gy += kd * dy * inv;
                    cnt += 1.0;
                }
                [u, (gx * gx + gy * gy).sqrt(), cnt * 0.05, 1.0]
            })
            .collect()
    }

    /// Particle-NCA step: a small per-particle MLP maps local features to a
    /// velocity. `weights` is `[hidden×4 | hidden→2]` flattened; random weights
    /// give a parameterized swarm, and the same layout is trainable by the ES /
    /// gradient machinery. Demonstrates a learned rule on the particle substrate.
    pub fn step_nca(&mut self, dt: f32, radius: f32, k_mu: f32, k_sigma: f32, weights: &[f32], hidden: usize) {
        let feats = self.features(radius, k_mu, k_sigma);
        let n = self.pos.len();
        let vel: Vec<(f32, f32)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let x = feats[i];
                let mut h = vec![0.0f32; hidden];
                for (j, hj) in h.iter_mut().enumerate() {
                    let wb = j * 4;
                    let mut acc = 0.0f32;
                    for k in 0..4 {
                        acc += weights[wb + k] * x[k];
                    }
                    *hj = acc.tanh();
                }
                let o = hidden * 4;
                let (mut vx, mut vy) = (0.0f32, 0.0f32);
                for j in 0..hidden {
                    vx += weights[o + j] * h[j];
                    vy += weights[o + hidden + j] * h[j];
                }
                (vx, vy)
            })
            .collect();
        for (p, v) in self.pos.iter_mut().zip(vel) {
            p.0 = (p.0 + dt * v.0).rem_euclid(self.s);
            p.1 = (p.1 + dt * v.1).rem_euclid(self.s);
        }
    }

    pub fn nca_weight_count(hidden: usize) -> usize {
        hidden * 4 + hidden * 2
    }

    /// Splat particles onto a `grid×grid` density field (torus).
    pub fn density_field(&self, grid: usize) -> Vec<f32> {
        let mut f = vec![0.0f32; grid * grid];
        let scale = grid as f32 / self.s;
        for &(x, y) in &self.pos {
            let gx = ((x * scale) as usize).min(grid - 1);
            let gy = ((y * scale) as usize).min(grid - 1);
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let px = (gx as i32 + dx).rem_euclid(grid as i32) as usize;
                    let py = (gy as i32 + dy).rem_euclid(grid as i32) as usize;
                    let w = if dx == 0 && dy == 0 { 1.0 } else { 0.4 };
                    f[py * grid + px] = (f[py * grid + px] + w).min(1.0);
                }
            }
        }
        f
    }

    /// Proximity interaction graph: an edge for every pair within `radius`.
    pub fn proximity_graph(&self, radius: f32) -> Vec<Vec<(usize, f32)>> {
        let n = self.pos.len();
        let mut adj = vec![Vec::new(); n];
        for i in 0..n {
            for j in (i + 1)..n {
                let (_, _, d) = self.delta(self.pos[i], self.pos[j]);
                if d <= radius && d > 0.0 {
                    let w = 1.0 / d;
                    adj[i].push((j, w));
                    adj[j].push((i, w));
                }
            }
        }
        adj
    }
}
