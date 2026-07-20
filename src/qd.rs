//! MAP-Elites quality-diversity (§2.7, §5.4) — the discovery mode.
//!
//! Searches Lenia growth-parameter space `(μ, σ)` and illuminates it: an archive
//! binned by *behavior* (final mass × mobility) keeps the most structured pattern
//! found in each bin. The output is not one "best" rule but a map of the diverse
//! behaviors the rule family can produce — rendered as a montage over behavior
//! space.

use crate::analysis::circular_centroid;
use crate::config::*;
use crate::engine::Engine;
use crate::substrate::Xorshift;
use rayon::prelude::*;

pub struct Elite {
    pub mu: f32,
    pub sigma: f32,
    pub fitness: f32,
    pub mass: f32,
    pub mobility: f32,
    pub field: Vec<f32>,
}

pub struct QdReport {
    pub bins: usize,
    pub grid: usize,
    pub archive: Vec<Option<Elite>>,
    pub evals: usize,
}

impl QdReport {
    pub fn coverage(&self) -> f32 {
        self.archive.iter().filter(|e| e.is_some()).count() as f32 / self.archive.len() as f32
    }
    pub fn best(&self) -> Option<&Elite> {
        self.archive.iter().flatten().max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap())
    }
}

const MASS_MAX: f32 = 0.6;
const MOB_MAX: f32 = 1.5;

fn bin(mass: f32, mobility: f32, bins: usize) -> (usize, usize) {
    let mb = ((mass / MASS_MAX).clamp(0.0, 0.999) * bins as f32) as usize;
    let ob = ((mobility / MOB_MAX).clamp(0.0, 0.999) * bins as f32) as usize;
    (mb, ob)
}

/// Run one Lenia genome to a behavior descriptor + a fitness (structuredness).
fn evaluate(mu: f32, sigma: f32, grid: usize, steps: usize) -> Elite {
    let cfg = Config {
        name: "qd".into(),
        schema_version: 1,
        substrate: SubstrateConfig { kind: "grid".into(), width: grid, height: grid, channels: 1, topology: "torus".into() },
        rule: RuleConfig::Lenia(LeniaConfig {
            dt: 0.1,
            clamp_lo: 0.0,
            clamp_hi: 1.0,
            kernels: vec![KernelConfig {
                source: 0, target: 0, radius: 13, core: "gauss_ring".into(), beta: vec![1.0],
                core_mu: 0.5, core_sigma: 0.15, weight: 1.0,
                growth: GrowthConfig { kind: "gauss".into(), mu, sigma },
            }],
        }),
        // Fixed seed pattern → deterministic behavior per genome.
        init: InitConfig::Random { density: 0.4 },
        render: RenderConfig::default(),
        analysis: vec![],
        steps: Some(steps as u64),
        seed: 42,
    };
    let mut e = Engine::from_config(cfg);
    let (mut px, mut py) = circular_centroid(&e.field, 0);
    let mut path = 0.0f64;
    for _ in 0..steps {
        e.step();
        let (cx, cy) = circular_centroid(&e.field, 0);
        path += tor(px, cx, grid) + tor(py, cy, grid);
        px = cx;
        py = cy;
    }
    let field = e.field.channel(0).to_vec();
    let mean = field.iter().sum::<f32>() / field.len() as f32;
    let var = field.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / field.len() as f32;
    Elite { mu, sigma, fitness: var.sqrt(), mass: mean, mobility: (path / steps as f64) as f32, field }
}

fn tor(a: f64, b: f64, n: usize) -> f64 {
    let d = (a - b).abs();
    d.min(n as f64 - d)
}

/// Batched MAP-Elites: mutate random elites, evaluate a batch in parallel, insert.
pub fn run(grid: usize, steps: usize, iterations: usize, batch: usize, bins: usize, seed: u64) -> QdReport {
    let mut archive: Vec<Option<Elite>> = (0..bins * bins).map(|_| None).collect();
    let mut rng = Xorshift::new(seed);

    for it in 0..iterations {
        // Genomes: random early (fill), then mutations of existing elites.
        let filled: Vec<usize> = archive.iter().enumerate().filter(|(_, e)| e.is_some()).map(|(i, _)| i).collect();
        let genomes: Vec<(f32, f32)> = (0..batch)
            .map(|_| {
                if filled.len() < 8 || rng.unit() < 0.15 {
                    // µ∈[0.05,0.35], σ∈[0.005,0.06]
                    (0.05 + rng.unit() * 0.30, 0.005 + rng.unit() * 0.055)
                } else {
                    let e = archive[filled[rng.below(filled.len())]].as_ref().unwrap();
                    ((e.mu + (rng.unit() - 0.5) * 0.04).clamp(0.03, 0.4),
                     (e.sigma + (rng.unit() - 0.5) * 0.01).clamp(0.003, 0.08))
                }
            })
            .collect();
        let evaluated: Vec<Elite> = genomes.par_iter().map(|&(mu, sg)| evaluate(mu, sg, grid, steps)).collect();
        for elite in evaluated {
            let (mb, ob) = bin(elite.mass, elite.mobility, bins);
            let idx = mb * bins + ob;
            let better = archive[idx].as_ref().map(|e| elite.fitness > e.fitness).unwrap_or(true);
            if better {
                archive[idx] = Some(elite);
            }
        }
        let _ = it;
    }

    QdReport { bins, grid, archive, evals: iterations * batch }
}
