//! Bundled presets and field seeding.
//!
//! `ORBIUM` is the reference oracle (§7.1): the canonical Lenia glider. Seeding a
//! field with it and confirming it translates while conserving mass validates the
//! whole convolution → growth → integrate pipeline against a known attractor.

use crate::config::*;
use crate::field::Field;
use crate::substrate::Xorshift;

/// The Orbium glider (Bert Chan's Lenia), a 20×20 cell block. Under R=13,
/// μ=0.15, σ=0.015 it is a stable translating soliton.
#[rustfmt::skip]
pub const ORBIUM: [[f32; 20]; 20] = [
    [0.0,0.0,0.0,0.0,0.0,0.0,0.1,0.14,0.1,0.0,0.0,0.03,0.03,0.0,0.0,0.3,0.0,0.0,0.0,0.0],
    [0.0,0.0,0.0,0.0,0.0,0.08,0.24,0.3,0.3,0.18,0.14,0.15,0.16,0.15,0.09,0.2,0.0,0.0,0.0,0.0],
    [0.0,0.0,0.0,0.0,0.0,0.15,0.34,0.44,0.46,0.38,0.18,0.14,0.11,0.13,0.19,0.18,0.45,0.0,0.0,0.0],
    [0.0,0.0,0.0,0.0,0.06,0.13,0.39,0.5,0.5,0.37,0.06,0.0,0.0,0.0,0.02,0.16,0.68,0.0,0.0,0.0],
    [0.0,0.0,0.0,0.11,0.17,0.17,0.33,0.4,0.38,0.28,0.14,0.0,0.0,0.0,0.0,0.0,0.18,0.42,0.0,0.0],
    [0.0,0.0,0.09,0.18,0.13,0.06,0.08,0.26,0.32,0.32,0.27,0.0,0.0,0.0,0.0,0.0,0.0,0.82,0.0,0.0],
    [0.27,0.0,0.16,0.12,0.0,0.0,0.0,0.25,0.38,0.44,0.45,0.34,0.0,0.0,0.0,0.0,0.0,0.22,0.17,0.0],
    [0.0,0.07,0.2,0.02,0.0,0.0,0.0,0.31,0.48,0.57,0.6,0.57,0.0,0.0,0.0,0.0,0.0,0.0,0.49,0.0],
    [0.0,0.59,0.19,0.0,0.0,0.0,0.0,0.2,0.57,0.69,0.76,0.76,0.49,0.0,0.0,0.0,0.0,0.0,0.36,0.0],
    [0.0,0.58,0.19,0.0,0.0,0.0,0.0,0.0,0.67,0.83,0.9,0.92,0.87,0.12,0.0,0.0,0.0,0.0,0.22,0.07],
    [0.0,0.0,0.46,0.0,0.0,0.0,0.0,0.0,0.7,0.93,1.0,1.0,1.0,0.61,0.0,0.0,0.0,0.0,0.18,0.11],
    [0.0,0.0,0.82,0.0,0.0,0.0,0.0,0.0,0.47,1.0,1.0,0.98,1.0,0.96,0.27,0.0,0.0,0.0,0.19,0.1],
    [0.0,0.0,0.46,0.0,0.0,0.0,0.0,0.0,0.25,1.0,1.0,0.84,0.92,0.97,0.54,0.14,0.04,0.1,0.21,0.05],
    [0.0,0.0,0.0,0.4,0.0,0.0,0.0,0.0,0.09,0.8,1.0,0.82,0.8,0.85,0.63,0.31,0.18,0.19,0.2,0.01],
    [0.0,0.0,0.0,0.36,0.1,0.0,0.0,0.0,0.05,0.54,0.86,0.79,0.74,0.72,0.6,0.39,0.28,0.24,0.13,0.0],
    [0.0,0.0,0.0,0.01,0.3,0.07,0.0,0.0,0.08,0.36,0.64,0.7,0.64,0.6,0.51,0.39,0.29,0.19,0.04,0.0],
    [0.0,0.0,0.0,0.0,0.1,0.24,0.14,0.1,0.15,0.29,0.45,0.53,0.52,0.46,0.4,0.31,0.21,0.08,0.0,0.0],
    [0.0,0.0,0.0,0.0,0.0,0.08,0.21,0.21,0.22,0.29,0.36,0.39,0.37,0.33,0.26,0.18,0.09,0.0,0.0,0.0],
    [0.0,0.0,0.0,0.0,0.0,0.0,0.03,0.13,0.19,0.22,0.24,0.24,0.23,0.18,0.13,0.05,0.0,0.0,0.0,0.0],
    [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.02,0.06,0.08,0.09,0.07,0.05,0.01,0.0,0.0,0.0,0.0,0.0],
];

fn stamp_orbium(field: &mut Field, top: usize, left: usize) {
    for (dy, row) in ORBIUM.iter().enumerate() {
        for (dx, &v) in row.iter().enumerate() {
            let y = (top + dy) % field.h;
            let x = (left + dx) % field.w;
            let cur = field.get(0, y, x);
            field.set(0, y, x, (cur + v).min(1.0));
        }
    }
}

fn stamp_blob(field: &mut Field, ch: usize, cy: f32, cx: f32, radius: f32, amp: f32) {
    let r = radius.ceil() as i32;
    let (cyi, cxi) = (cy as i32, cx as i32);
    for dy in -r..=r {
        for dx in -r..=r {
            let d2 = (dx * dx + dy * dy) as f32;
            let v = amp * (-d2 / (2.0 * radius * radius)).exp();
            let y = (cyi + dy).rem_euclid(field.h as i32) as usize;
            let x = (cxi + dx).rem_euclid(field.w as i32) as usize;
            let cur = field.get(ch, y, x);
            field.set(ch, y, x, (cur + v).min(1.0));
        }
    }
}

/// Apply an `InitConfig` to a freshly-zeroed field.
pub fn apply_init(field: &mut Field, init: &InitConfig, seed: u64) {
    let mut rng = Xorshift::new(seed);
    match init {
        InitConfig::Zeros => {}
        InitConfig::Random { density } => {
            // Sparse random soup over the central region — self-organizes under Lenia.
            let (h, w) = (field.h, field.w);
            let (y0, y1) = (h / 5, 4 * h / 5);
            let (x0, x1) = (w / 5, 4 * w / 5);
            for y in y0..y1 {
                for x in x0..x1 {
                    if rng.unit() < *density {
                        field.set(0, y, x, rng.unit());
                    }
                }
            }
        }
        InitConfig::Orbium { dx, dy } => {
            let top = (field.h as i32 / 2 - 10 + dy).rem_euclid(field.h as i32) as usize;
            let left = (field.w as i32 / 2 - 10 + dx).rem_euclid(field.w as i32) as usize;
            stamp_orbium(field, top, left);
        }
        InitConfig::Blobs { count, radius } => {
            for _ in 0..*count {
                let cy = rng.below(field.h) as f32;
                let cx = rng.below(field.w) as f32;
                stamp_blob(field, 0, cy, cx, *radius, 0.9);
            }
        }
        InitConfig::OrbiumSwarm { count } => {
            // Jittered grid placement so the gliders stay separated.
            let cols = (*count as f32).sqrt().ceil().max(1.0) as usize;
            let rows = (count + cols - 1) / cols;
            for k in 0..*count {
                let (gx, gy) = (k % cols, k / cols);
                let jx = (rng.unit() - 0.5) * (field.w as f32 / cols as f32) * 0.4;
                let jy = (rng.unit() - 0.5) * (field.h as f32 / rows as f32) * 0.4;
                let cx = (gx as f32 + 0.5) / cols as f32 * field.w as f32 + jx;
                let cy = (gy as f32 + 0.5) / rows as f32 * field.h as f32 + jy;
                let top = (cy as i32 - 10).rem_euclid(field.h as i32) as usize;
                let left = (cx as i32 - 10).rem_euclid(field.w as i32) as usize;
                stamp_orbium(field, top, left);
            }
        }
        InitConfig::GrayScottSeed => {
            // u = 1 everywhere; a grid of jittered v-blocks seeds pattern formation
            // across the whole field so structure develops everywhere, not just a
            // slowly-spreading center.
            for v in field.channel_mut(0).iter_mut() {
                *v = 1.0;
            }
            let n = 4;
            let half = 5i32;
            for gy in 0..n {
                for gx in 0..n {
                    let jx = (rng.unit() - 0.5) * (field.w as f32 / n as f32) * 0.5;
                    let jy = (rng.unit() - 0.5) * (field.h as f32 / n as f32) * 0.5;
                    let ccx = ((gx as f32 + 0.5) / n as f32 * field.w as f32 + jx) as i32;
                    let ccy = ((gy as f32 + 0.5) / n as f32 * field.h as f32 + jy) as i32;
                    for dy in -half..=half {
                        for dx in -half..=half {
                            let y = (ccy + dy).rem_euclid(field.h as i32) as usize;
                            let x = (ccx + dx).rem_euclid(field.w as i32) as usize;
                            field.set(0, y, x, 0.5);
                            field.set(1, y, x, 0.25 + 0.02 * (rng.unit() - 0.5));
                        }
                    }
                }
            }
        }
    }
}

// --- Preset configs -----------------------------------------------------------

fn lenia_kernel(radius: usize, growth_mu: f32, growth_sigma: f32) -> KernelConfig {
    KernelConfig {
        source: 0,
        target: 0,
        radius,
        core: "gauss_ring".into(),
        beta: vec![1.0],
        core_mu: 0.5,
        core_sigma: 0.15,
        weight: 1.0,
        growth: GrowthConfig { kind: "gauss".into(), mu: growth_mu, sigma: growth_sigma },
    }
}

fn grid(name: &str, h: usize, w: usize, ch: usize) -> SubstrateConfig {
    let _ = name;
    SubstrateConfig { kind: "grid".into(), width: w, height: h, channels: ch, topology: "torus".into() }
}

/// Orbium — the reference oracle.
pub fn orbium() -> Config {
    Config {
        name: "orbium".into(),
        schema_version: 1,
        substrate: grid("orbium", 128, 128, 1),
        rule: RuleConfig::Lenia(LeniaConfig {
            dt: 0.1,
            clamp_lo: 0.0,
            clamp_hi: 1.0,
            kernels: vec![lenia_kernel(13, 0.15, 0.015)],
        }),
        init: InitConfig::Orbium { dx: 0, dy: 0 },
        render: RenderConfig { colormap: "viridis".into(), channel: 0, scale: 1.0 },
        analysis: vec![AnalysisConfig::Descriptors],
        steps: Some(400),
        seed: 1,
    }
}

/// Random Lenia soup — self-organizing dynamics from noise.
pub fn soup() -> Config {
    Config {
        name: "soup".into(),
        schema_version: 1,
        substrate: grid("soup", 192, 192, 1),
        rule: RuleConfig::Lenia(LeniaConfig {
            dt: 0.1,
            clamp_lo: 0.0,
            clamp_hi: 1.0,
            kernels: vec![lenia_kernel(13, 0.15, 0.017)],
        }),
        init: InitConfig::Random { density: 0.5 },
        render: RenderConfig { colormap: "viridis".into(), channel: 0, scale: 1.0 },
        analysis: vec![AnalysisConfig::Descriptors],
        steps: Some(600),
        seed: 7,
    }
}

/// Multi-organism scene for the detection + PageRank observer.
pub fn life() -> Config {
    Config {
        name: "life".into(),
        schema_version: 1,
        substrate: grid("life", 160, 160, 1),
        rule: RuleConfig::Lenia(LeniaConfig {
            dt: 0.1,
            clamp_lo: 0.0,
            clamp_hi: 1.0,
            kernels: vec![lenia_kernel(13, 0.15, 0.016)],
        }),
        init: InitConfig::OrbiumSwarm { count: 6 },
        render: RenderConfig { colormap: "turbo".into(), channel: 0, scale: 1.0 },
        analysis: vec![
            AnalysisConfig::Detect { threshold: 0.15 },
            AnalysisConfig::PageRank { threshold: 0.15, link_radius: 55.0 },
        ],
        steps: Some(200),
        seed: 3,
    }
}

/// Multi-ring Lenia — two concentric kernels, exercising the multi-kernel path.
pub fn multiring() -> Config {
    let mut k1 = lenia_kernel(13, 0.15, 0.017);
    k1.beta = vec![1.0, 0.4];
    k1.weight = 0.6;
    let mut k2 = lenia_kernel(18, 0.16, 0.018);
    k2.weight = 0.4;
    Config {
        name: "multiring".into(),
        schema_version: 1,
        substrate: grid("multiring", 192, 192, 1),
        rule: RuleConfig::Lenia(LeniaConfig { dt: 0.1, clamp_lo: 0.0, clamp_hi: 1.0, kernels: vec![k1, k2] }),
        init: InitConfig::Random { density: 0.5 },
        render: RenderConfig { colormap: "turbo".into(), channel: 0, scale: 1.0 },
        analysis: vec![AnalysisConfig::Descriptors],
        steps: Some(500),
        seed: 13,
    }
}

/// Asymptotic Lenia — relaxation toward target, smoother and more stable.
pub fn asymptotic() -> Config {
    Config {
        name: "asymptotic".into(),
        schema_version: 1,
        substrate: grid("asymptotic", 160, 160, 1),
        rule: RuleConfig::AsymptoticLenia(LeniaConfig {
            dt: 0.25,
            clamp_lo: 0.0,
            clamp_hi: 1.0,
            kernels: vec![lenia_kernel(13, 0.15, 0.017)],
        }),
        init: InitConfig::Random { density: 0.5 },
        render: RenderConfig { colormap: "magma".into(), channel: 0, scale: 1.0 },
        analysis: vec![AnalysisConfig::Descriptors],
        steps: Some(400),
        seed: 5,
    }
}

/// Flow Lenia — mass-conserving advection. Total mass stays constant by construction.
pub fn flow() -> Config {
    Config {
        name: "flow".into(),
        schema_version: 1,
        substrate: grid("flow", 160, 160, 1),
        rule: RuleConfig::FlowLenia(FlowLeniaConfig {
            base: LeniaConfig {
                dt: 0.4,
                clamp_lo: 0.0,
                clamp_hi: 1.0,
                kernels: vec![lenia_kernel(13, 0.15, 0.02)],
            },
            flow: 2.0,
            concentration: 0.4,
        }),
        init: InitConfig::Blobs { count: 8, radius: 9.0 },
        render: RenderConfig { colormap: "turbo".into(), channel: 0, scale: 1.0 },
        analysis: vec![AnalysisConfig::Descriptors],
        steps: Some(400),
        seed: 9,
    }
}

/// Gray-Scott reaction-diffusion — a structurally different rule, same engine.
pub fn gray_scott() -> Config {
    Config {
        name: "gray_scott".into(),
        schema_version: 1,
        substrate: grid("gray_scott", 200, 200, 2),
        rule: RuleConfig::GrayScott(GrayScottConfig {
            dt: 1.0,
            du: 0.16,
            dv: 0.08,
            // "Labyrinth" regime — v fingers out and fills the field with maze structure.
            feed: 0.039,
            kill: 0.058,
        }),
        init: InitConfig::GrayScottSeed,
        render: RenderConfig { colormap: "magma".into(), channel: 1, scale: 2.5 },
        analysis: vec![AnalysisConfig::Descriptors],
        steps: Some(6000),
        seed: 11,
    }
}

pub fn by_name(name: &str) -> Option<Config> {
    match name {
        "orbium" => Some(orbium()),
        "soup" => Some(soup()),
        "life" => Some(life()),
        "multiring" => Some(multiring()),
        "asymptotic" => Some(asymptotic()),
        "flow" => Some(flow()),
        "gray_scott" | "grayscott" => Some(gray_scott()),
        _ => None,
    }
}

pub const NAMES: &[&str] =
    &["orbium", "soup", "life", "multiring", "asymptotic", "flow", "gray_scott"];
