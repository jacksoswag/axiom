//! The update law (§2.4) — the core swappable trait.
//!
//! A `Rule` reads the current `Field` and writes the next one. Two realizations
//! ship here: multi-kernel / multi-channel **Lenia** (classic Lenia is the
//! 1-kernel/1-channel special case) and **Gray-Scott** reaction-diffusion, which
//! exists mainly to prove a second, structurally different rule drops into the
//! same trait with no engine changes.

use crate::config::{GrayScottConfig, GrowthConfig, KernelConfig, LeniaConfig};
use crate::field::Field;
use crate::growth::{Growth, GrowthKind};
use crate::kernel::{convolve, CoreKind, Kernel, KernelParams};
use rayon::prelude::*;

pub trait Rule: Send + Sync {
    /// Advance `state` into `out` by one step.
    fn step(&self, state: &Field, out: &mut Field, torus: bool);
    fn name(&self) -> &'static str;
}

// --- Lenia --------------------------------------------------------------------

struct Spec {
    source: usize,
    target: usize,
    kernel: Kernel,
    growth: Growth,
    weight: f32,
}

pub struct LeniaRule {
    specs: Vec<Spec>,
    dt: f32,
    clamp: (f32, f32),
    channels: usize,
}

impl LeniaRule {
    pub fn from_config(cfg: &LeniaConfig, channels: usize) -> LeniaRule {
        LeniaRule { specs: build_specs(cfg), dt: cfg.dt, clamp: (cfg.clamp_lo, cfg.clamp_hi), channels }
    }
}

fn build_specs(cfg: &LeniaConfig) -> Vec<Spec> {
    cfg.kernels.iter().map(build_spec).collect()
}

fn build_spec(k: &KernelConfig) -> Spec {
    let kernel = Kernel::build(&KernelParams {
        radius: k.radius,
        core: CoreKind::parse(&k.core),
        beta: &k.beta,
        core_mu: k.core_mu,
        core_sigma: k.core_sigma,
    });
    Spec {
        source: k.source,
        target: k.target,
        kernel,
        growth: to_growth(&k.growth),
        weight: k.weight,
    }
}

fn to_growth(g: &GrowthConfig) -> Growth {
    Growth { kind: GrowthKind::parse(&g.kind), mu: g.mu, sigma: g.sigma }
}

impl Rule for LeniaRule {
    fn step(&self, state: &Field, out: &mut Field, torus: bool) {
        let plane = state.plane();
        // Accumulate weighted growth per target channel.
        let mut deltas = vec![0.0f32; self.channels * plane];
        let mut conv = vec![0.0f32; plane];
        for spec in &self.specs {
            convolve(state.channel(spec.source), state.h, state.w, &spec.kernel, torus, &mut conv);
            let dslice = &mut deltas[spec.target * plane..(spec.target + 1) * plane];
            let g = spec.growth;
            let wgt = spec.weight;
            dslice
                .par_iter_mut()
                .zip(conv.par_iter())
                .for_each(|(d, &u)| *d += wgt * g.apply(u));
        }
        // out = clip(state + dt * delta)
        let (lo, hi) = self.clamp;
        let dt = self.dt;
        out.data
            .par_iter_mut()
            .zip(state.data.par_iter())
            .zip(deltas.par_iter())
            .for_each(|((o, &s), &d)| *o = (s + dt * d).clamp(lo, hi));
    }

    fn name(&self) -> &'static str {
        "lenia"
    }
}

// --- Asymptotic Lenia (relaxation toward target) ------------------------------

pub struct AsymptoticLeniaRule {
    specs: Vec<Spec>,
    dt: f32,
    clamp: (f32, f32),
    channels: usize,
}

impl AsymptoticLeniaRule {
    pub fn from_config(cfg: &LeniaConfig, channels: usize) -> Self {
        AsymptoticLeniaRule { specs: build_specs(cfg), dt: cfg.dt, clamp: (cfg.clamp_lo, cfg.clamp_hi), channels }
    }
}

impl Rule for AsymptoticLeniaRule {
    fn step(&self, state: &Field, out: &mut Field, torus: bool) {
        let plane = state.plane();
        // Accumulate a target activation in [0,1] per channel, then relax toward it.
        let mut target = vec![0.0f32; self.channels * plane];
        let mut conv = vec![0.0f32; plane];
        for spec in &self.specs {
            convolve(state.channel(spec.source), state.h, state.w, &spec.kernel, torus, &mut conv);
            let tslice = &mut target[spec.target * plane..(spec.target + 1) * plane];
            let g = spec.growth;
            let wgt = spec.weight;
            tslice
                .par_iter_mut()
                .zip(conv.par_iter())
                .for_each(|(t, &u)| *t += wgt * 0.5 * (g.apply(u) + 1.0));
        }
        let (lo, hi) = self.clamp;
        let dt = self.dt;
        out.data
            .par_iter_mut()
            .zip(state.data.par_iter())
            .zip(target.par_iter())
            .for_each(|((o, &s), &t)| *o = (s + dt * (t.clamp(0.0, 1.0) - s)).clamp(lo, hi));
    }
    fn name(&self) -> &'static str {
        "asymptotic_lenia"
    }
}

// --- Flow Lenia (mass-conserving advection) -----------------------------------

/// Mass is transported along a flow derived from the potential gradient, using
/// reintegration tracking (bilinear splat). Total mass is conserved by
/// construction — the validation for this rule (§2.4, "mass-conserving").
///
/// ponytail: single-channel advection up ∇U minus a concentration term; the full
/// Flow-Lenia energy (growth-modulated, multi-species) is the upgrade path.
pub struct FlowLeniaRule {
    specs: Vec<Spec>,
    dt: f32,
    flow: f32,
    concentration: f32,
}

impl FlowLeniaRule {
    pub fn from_config(cfg: &crate::config::FlowLeniaConfig) -> Self {
        FlowLeniaRule { specs: build_specs(&cfg.base), dt: cfg.base.dt, flow: cfg.flow, concentration: cfg.concentration }
    }
}

#[inline]
fn wrap(v: i32, n: usize) -> usize {
    v.rem_euclid(n as i32) as usize
}

impl Rule for FlowLeniaRule {
    fn step(&self, state: &Field, out: &mut Field, torus: bool) {
        let (h, w) = (state.h, state.w);
        let plane = h * w;
        // Potential U = Σ weight · (K * A_source).
        let mut u = vec![0.0f32; plane];
        let mut conv = vec![0.0f32; plane];
        for spec in &self.specs {
            convolve(state.channel(spec.source), h, w, &spec.kernel, torus, &mut conv);
            for (uu, &c) in u.iter_mut().zip(conv.iter()) {
                *uu += spec.weight * c;
            }
        }
        let a = state.channel(0);
        // Flow field F = flow·∇U − concentration·∇A (central differences, torus).
        let out0 = &mut out.data[0..plane];
        out0.fill(0.0);
        let (flow, conc, dt) = (self.flow, self.concentration, self.dt);
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                let m = a[i];
                if m == 0.0 {
                    continue;
                }
                let (yl, yr) = (wrap(y as i32 - 1, h), wrap(y as i32 + 1, h));
                let (xl, xr) = (wrap(x as i32 - 1, w), wrap(x as i32 + 1, w));
                let fy = flow * 0.5 * (u[yr * w + x] - u[yl * w + x]) - conc * 0.5 * (a[yr * w + x] - a[yl * w + x]);
                let fx = flow * 0.5 * (u[y * w + xr] - u[y * w + xl]) - conc * 0.5 * (a[y * w + xr] - a[y * w + xl]);
                // Clamp displacement below one cell for a stable splat.
                let ty = y as f32 + (dt * fy).clamp(-0.9, 0.9);
                let tx = x as f32 + (dt * fx).clamp(-0.9, 0.9);
                let (y0, x0) = (ty.floor(), tx.floor());
                let (gy, gx) = (ty - y0, tx - x0);
                let (iy0, ix0) = (wrap(y0 as i32, h), wrap(x0 as i32, w));
                let (iy1, ix1) = (wrap(y0 as i32 + 1, h), wrap(x0 as i32 + 1, w));
                out0[iy0 * w + ix0] += m * (1.0 - gy) * (1.0 - gx);
                out0[iy0 * w + ix1] += m * (1.0 - gy) * gx;
                out0[iy1 * w + ix0] += m * gy * (1.0 - gx);
                out0[iy1 * w + ix1] += m * gy * gx;
            }
        }
        // Extra channels (if any) pass through unchanged.
        for ch in 1..state.c {
            out.channel_mut(ch).copy_from_slice(state.channel(ch));
        }
    }
    fn name(&self) -> &'static str {
        "flow_lenia"
    }
}

// --- Gray-Scott reaction-diffusion --------------------------------------------

pub struct GrayScottRule {
    dt: f32,
    du: f32,
    dv: f32,
    feed: f32,
    kill: f32,
}

impl GrayScottRule {
    pub fn from_config(cfg: &GrayScottConfig) -> GrayScottRule {
        GrayScottRule { dt: cfg.dt, du: cfg.du, dv: cfg.dv, feed: cfg.feed, kill: cfg.kill }
    }
}

#[inline]
fn laplacian(f: &[f32], h: usize, w: usize, y: usize, x: usize, torus: bool) -> f32 {
    // 9-point stencil (weights from the standard Gray-Scott discretization).
    let wrap = |v: i32, n: usize| -> Option<usize> {
        if torus {
            Some(v.rem_euclid(n as i32) as usize)
        } else if v < 0 || v >= n as i32 {
            None
        } else {
            Some(v as usize)
        }
    };
    let at = |yy: i32, xx: i32| -> f32 {
        match (wrap(yy, h), wrap(xx, w)) {
            (Some(a), Some(b)) => f[a * w + b],
            _ => 0.0,
        }
    };
    let (yi, xi) = (y as i32, x as i32);
    let center = f[y * w + x];
    let ortho = at(yi - 1, xi) + at(yi + 1, xi) + at(yi, xi - 1) + at(yi, xi + 1);
    let diag = at(yi - 1, xi - 1) + at(yi - 1, xi + 1) + at(yi + 1, xi - 1) + at(yi + 1, xi + 1);
    0.2 * ortho + 0.05 * diag - center
}

impl Rule for GrayScottRule {
    fn step(&self, state: &Field, out: &mut Field, torus: bool) {
        let (h, w) = (state.h, state.w);
        let u = state.channel(0);
        let v = state.channel(1);
        let plane = h * w;
        let (out_u, out_v) = out.data.split_at_mut(plane);
        let (du, dv, feed, kill, dt) = (self.du, self.dv, self.feed, self.kill, self.dt);
        out_u
            .par_chunks_mut(w)
            .zip(out_v.par_chunks_mut(w))
            .enumerate()
            .for_each(|(y, (ru, rv))| {
                for x in 0..w {
                    let i = y * w + x;
                    let (uu, vv) = (u[i], v[i]);
                    let reaction = uu * vv * vv;
                    let lu = laplacian(u, h, w, y, x, torus);
                    let lv = laplacian(v, h, w, y, x, torus);
                    ru[x] = (uu + dt * (du * lu - reaction + feed * (1.0 - uu))).clamp(0.0, 1.0);
                    rv[x] = (vv + dt * (dv * lv + reaction - (kill + feed) * vv)).clamp(0.0, 1.0);
                }
            });
    }

    fn name(&self) -> &'static str {
        "gray_scott"
    }
}
