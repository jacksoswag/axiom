//! Neural Cellular Automata + learning (§2.4, §2.7, §4) — the ML integration.
//!
//! The rule is a small per-cell MLP over a 4-filter perception (identity,
//! Sobel-x, Sobel-y, Laplacian) — the Laplacian lets one 3×3 step represent a
//! diffusion rule. Weights come from training.
//!
//! `train_imitate` fits the NCA to a target rule's one-step transition using
//! gradient-free **evolution strategies** (antithetic finite-difference), so
//! "rule = a learned net" is real and needs no autodiff backend. Loss dropping
//! over generations is the evidence of learning; `rollout_error` then measures
//! the learned rule as a world model — prediction error vs. horizon (§4).

use crate::field::Field;
use crate::rule::Rule;
use crate::substrate::Xorshift;
use rayon::prelude::*;
use std::f32::consts::TAU;
use std::sync::atomic::{AtomicU64, Ordering};

/// Perception filters per channel: identity, Sobel-x, Sobel-y, Laplacian.
const PERC: usize = 4;

pub struct Nca {
    pub c: usize,
    pub hidden: usize,
    pub update_rate: f32,
    pub w1: Vec<f32>, // hidden × (PERC·c + 1 bias)
    pub w2: Vec<f32>, // c × (hidden + 1 bias)
    salt: AtomicU64,
}

impl Nca {
    pub fn param_count(c: usize, hidden: usize) -> usize {
        hidden * (PERC * c + 1) + c * (hidden + 1)
    }

    pub fn from_theta(c: usize, hidden: usize, update_rate: f32, theta: &[f32]) -> Nca {
        let split = hidden * (PERC * c + 1);
        Nca { c, hidden, update_rate, w1: theta[..split].to_vec(), w2: theta[split..].to_vec(), salt: AtomicU64::new(0x1234_5678) }
    }

    pub fn random(c: usize, hidden: usize, update_rate: f32, seed: u64) -> Nca {
        let mut rng = Xorshift::new(seed);
        let np = PERC * c;
        let r1 = (1.0 / (np as f32 + 1.0)).sqrt();
        let r2 = (1.0 / (hidden as f32 + 1.0)).sqrt();
        let w1 = (0..hidden * (np + 1)).map(|_| (rng.unit() * 2.0 - 1.0) * r1).collect();
        // Second layer starts near zero so the initial rule is close to identity.
        let w2 = (0..c * (hidden + 1)).map(|_| (rng.unit() * 2.0 - 1.0) * r2 * 0.1).collect();
        Nca { c, hidden, update_rate, w1, w2, salt: AtomicU64::new(seed | 1) }
    }

    pub fn theta(&self) -> Vec<f32> {
        let mut t = self.w1.clone();
        t.extend_from_slice(&self.w2);
        t
    }

    fn perceive(&self, s: &Field) -> Vec<f32> {
        let (h, w) = (s.h, s.w);
        let plane = h * w;
        let np = PERC * self.c;
        let mut out = vec![0.0f32; plane * np];
        for ch in 0..self.c {
            let a = s.channel(ch);
            for y in 0..h {
                let yl = if y == 0 { h - 1 } else { y - 1 };
                let yr = if y + 1 == h { 0 } else { y + 1 };
                for x in 0..w {
                    let xl = if x == 0 { w - 1 } else { x - 1 };
                    let xr = if x + 1 == w { 0 } else { x + 1 };
                    let i = y * w + x;
                    let c = a[i];
                    let sx = (a[yl * w + xr] + 2.0 * a[y * w + xr] + a[yr * w + xr]
                        - a[yl * w + xl] - 2.0 * a[y * w + xl] - a[yr * w + xl]) / 8.0;
                    let sy = (a[yr * w + xl] + 2.0 * a[yr * w + x] + a[yr * w + xr]
                        - a[yl * w + xl] - 2.0 * a[yl * w + x] - a[yl * w + xr]) / 8.0;
                    let lap = a[yl * w + x] + a[yr * w + x] + a[y * w + xl] + a[y * w + xr] - 4.0 * c;
                    let b = i * np + ch * PERC;
                    out[b] = c;
                    out[b + 1] = sx;
                    out[b + 2] = sy;
                    out[b + 3] = lap;
                }
            }
        }
        out
    }
}

impl Rule for Nca {
    fn step(&self, state: &Field, out: &mut Field, _torus: bool) {
        let plane = state.plane();
        let (c, hid, np) = (self.c, self.hidden, PERC * self.c);
        let percept = self.perceive(state);
        let deltas: Vec<f32> = (0..plane)
            .into_par_iter()
            .flat_map_iter(|i| {
                let base = i * np;
                let mut h = vec![0.0f32; hid];
                for (j, hj) in h.iter_mut().enumerate() {
                    let wb = j * (np + 1);
                    let mut acc = self.w1[wb + np];
                    for k in 0..np {
                        acc += self.w1[wb + k] * percept[base + k];
                    }
                    *hj = acc.max(0.0);
                }
                let mut d = vec![0.0f32; c];
                for (ch, dc) in d.iter_mut().enumerate() {
                    let wb = ch * (hid + 1);
                    let mut acc = self.w2[wb + hid];
                    for j in 0..hid {
                        acc += self.w2[wb + j] * h[j];
                    }
                    *dc = acc;
                }
                d.into_iter()
            })
            .collect();
        let async_update = self.update_rate < 1.0;
        let salt = if async_update { self.salt.fetch_add(0x9e37_79b9, Ordering::Relaxed) } else { 0 };
        for i in 0..plane {
            let mask = if async_update {
                let hsh = (i as u64 ^ salt).wrapping_mul(0x2545_f491_4f6c_dd1d) >> 40;
                if (hsh as f32 / (1u64 << 24) as f32) < self.update_rate { 1.0 } else { 0.0 }
            } else {
                1.0
            };
            for ch in 0..c {
                let v = state.data[ch * plane + i] + mask * deltas[i * c + ch];
                out.data[ch * plane + i] = v.clamp(0.0, 1.0);
            }
        }
    }
    fn name(&self) -> &'static str {
        "nca"
    }
}

// --- Training (evolution strategies) ------------------------------------------

pub struct TrainReport {
    pub theta: Vec<f32>,
    pub c: usize,
    pub hidden: usize,
    pub loss_history: Vec<f32>,
    pub val_loss: f32,
}

fn gauss(rng: &mut Xorshift) -> f32 {
    let u1 = rng.unit().max(1e-7);
    let u2 = rng.unit();
    (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos()
}

/// Mean-squared error of the NCA's one-step prediction against a target rule.
pub fn one_step_mse(nca: &Nca, target: &dyn Rule, torus: bool, batch: &[Field]) -> f32 {
    let total: f32 = batch
        .par_iter()
        .map(|f| {
            let mut pred = f.clone();
            let mut truth = f.clone();
            nca.step(f, &mut pred, torus);
            target.step(f, &mut truth, torus);
            let n = pred.data.len();
            pred.data.iter().zip(truth.data.iter()).map(|(a, b)| (a - b) * (a - b)).sum::<f32>() / n as f32
        })
        .sum();
    total / batch.len() as f32
}

/// Fit an NCA to `target`'s one-step transition via antithetic evolution strategies.
#[allow(clippy::too_many_arguments)]
pub fn train_imitate(
    target: &dyn Rule,
    torus: bool,
    train: &[Field],
    val: &[Field],
    c: usize,
    hidden: usize,
    gens: usize,
    pop: usize,
    sigma: f32,
    lr: f32,
    seed: u64,
) -> TrainReport {
    let n = Nca::param_count(c, hidden);
    let mut theta = Nca::random(c, hidden, 1.0, seed).theta();
    let mut rng = Xorshift::new(seed ^ 0xf00d);
    let pairs = (pop / 2).max(1);
    let mut loss_history = Vec::with_capacity(gens);

    for _ in 0..gens {
        // Antithetic perturbation directions.
        let dirs: Vec<Vec<f32>> = (0..pairs).map(|_| (0..n).map(|_| gauss(&mut rng)).collect()).collect();
        // Evaluate ± each direction in parallel.
        let results: Vec<(f32, f32)> = dirs
            .par_iter()
            .map(|d| {
                let mut tp = theta.clone();
                let mut tm = theta.clone();
                for k in 0..n {
                    tp[k] += sigma * d[k];
                    tm[k] -= sigma * d[k];
                }
                let lp = one_step_mse(&Nca::from_theta(c, hidden, 1.0, &tp), target, torus, train);
                let lm = one_step_mse(&Nca::from_theta(c, hidden, 1.0, &tm), target, torus, train);
                (lp, lm)
            })
            .collect();
        // Finite-difference gradient of loss, averaged over directions; descend.
        let mut grad = vec![0.0f32; n];
        for (d, (lp, lm)) in dirs.iter().zip(results.iter()) {
            let coeff = (lp - lm) / (2.0 * sigma);
            for k in 0..n {
                grad[k] += coeff * d[k];
            }
        }
        let scale = lr / pairs as f32;
        for k in 0..n {
            theta[k] -= scale * grad[k];
        }
        let cur = one_step_mse(&Nca::from_theta(c, hidden, 1.0, &theta), target, torus, train);
        loss_history.push(cur);
    }

    let val_loss = one_step_mse(&Nca::from_theta(c, hidden, 1.0, &theta), target, torus, val);
    TrainReport { theta, c, hidden, loss_history, val_loss }
}

/// Analytic gradient of the one-step imitation loss for one field, accumulated
/// over its cells, in `theta` layout (w1 block, then w2 block). Returns
/// `(grad, loss)`. Clip is treated as straight-through in the backward pass.
fn field_grad(nca: &Nca, target: &dyn Rule, torus: bool, f: &Field) -> (Vec<f32>, f32) {
    let plane = f.plane();
    let (c, hid, np) = (nca.c, nca.hidden, PERC * nca.c);
    let w1len = hid * (np + 1);
    let n = w1len + c * (hid + 1);
    let mut truth = f.clone();
    target.step(f, &mut truth, torus);
    let percept = nca.perceive(f);
    let mut grad = vec![0.0f32; n];
    let mut loss = 0.0f32;
    let norm = (c * plane) as f32;
    let mut z1 = vec![0.0f32; hid];
    let mut h = vec![0.0f32; hid];
    for i in 0..plane {
        let base = i * np;
        // Forward.
        for j in 0..hid {
            let wb = j * (np + 1);
            let mut acc = nca.w1[wb + np];
            for k in 0..np {
                acc += nca.w1[wb + k] * percept[base + k];
            }
            z1[j] = acc;
            h[j] = acc.max(0.0);
        }
        // Output + loss gradient per channel.
        let mut g_delta = vec![0.0f32; c];
        for ch in 0..c {
            let wb = ch * (hid + 1);
            let mut delta = nca.w2[wb + hid];
            for j in 0..hid {
                delta += nca.w2[wb + j] * h[j];
            }
            let a = f.data[ch * plane + i];
            let out = (a + delta).clamp(0.0, 1.0);
            let e = out - truth.data[ch * plane + i];
            loss += e * e;
            g_delta[ch] = 2.0 * e / norm;
        }
        // Backward through W2 and into hidden.
        let mut g_h = vec![0.0f32; hid];
        for ch in 0..c {
            let wb = ch * (hid + 1);
            let gd = g_delta[ch];
            grad[w1len + wb + hid] += gd; // b2
            for j in 0..hid {
                grad[w1len + wb + j] += gd * h[j];
                g_h[j] += nca.w2[wb + j] * gd;
            }
        }
        // Backward through relu and W1.
        for j in 0..hid {
            if z1[j] <= 0.0 {
                continue;
            }
            let gz = g_h[j];
            let wb = j * (np + 1);
            grad[wb + np] += gz; // b1
            for k in 0..np {
                grad[wb + k] += gz * percept[base + k];
            }
        }
    }
    (grad, loss / norm)
}

/// Fit an NCA to `target`'s one-step transition by gradient descent (Adam) with
/// analytic backprop. Converges faster and lower than evolution strategies.
#[allow(clippy::too_many_arguments)]
pub fn train_imitate_grad(
    target: &dyn Rule,
    torus: bool,
    train: &[Field],
    val: &[Field],
    c: usize,
    hidden: usize,
    epochs: usize,
    lr: f32,
    seed: u64,
) -> TrainReport {
    let n = Nca::param_count(c, hidden);
    let mut theta = Nca::random(c, hidden, 1.0, seed).theta();
    let (mut m, mut v) = (vec![0.0f32; n], vec![0.0f32; n]);
    let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
    let mut loss_history = Vec::with_capacity(epochs);
    for epoch in 1..=epochs {
        let nca = Nca::from_theta(c, hidden, 1.0, &theta);
        let (grad, loss) = train
            .par_iter()
            .map(|f| field_grad(&nca, target, torus, f))
            .reduce(
                || (vec![0.0f32; n], 0.0f32),
                |(mut ga, la), (gb, lb)| {
                    for k in 0..n {
                        ga[k] += gb[k];
                    }
                    (ga, la + lb)
                },
            );
        let inv = 1.0 / train.len() as f32;
        let bc1 = 1.0 - b1.powi(epoch as i32);
        let bc2 = 1.0 - b2.powi(epoch as i32);
        for k in 0..n {
            let g = grad[k] * inv;
            m[k] = b1 * m[k] + (1.0 - b1) * g;
            v[k] = b2 * v[k] + (1.0 - b2) * g * g;
            theta[k] -= lr * (m[k] / bc1) / ((v[k] / bc2).sqrt() + eps);
        }
        loss_history.push(loss * inv);
    }
    let val_loss = one_step_mse(&Nca::from_theta(c, hidden, 1.0, &theta), target, torus, val);
    TrainReport { theta, c, hidden, loss_history, val_loss }
}

/// Roll out both rules from the same seed and report per-step prediction MSE —
/// the learned rule evaluated as a world model.
pub fn rollout_error(nca: &Nca, target: &dyn Rule, torus: bool, seed_field: &Field, horizon: usize) -> Vec<f32> {
    let mut np = seed_field.clone();
    let mut tp = seed_field.clone();
    let mut nbuf = seed_field.clone();
    let mut tbuf = seed_field.clone();
    let mut errs = Vec::with_capacity(horizon);
    for _ in 0..horizon {
        nca.step(&np, &mut nbuf, torus);
        target.step(&tp, &mut tbuf, torus);
        std::mem::swap(&mut np, &mut nbuf);
        std::mem::swap(&mut tp, &mut tbuf);
        let n = np.data.len();
        let e: f32 = np.data.iter().zip(tp.data.iter()).map(|(a, b)| (a - b) * (a - b)).sum::<f32>() / n as f32;
        errs.push(e);
    }
    errs
}
