//! The genome codec: a flat, bounded Vec<f32> on one side, engine Params on the other.
//! Nothing here simulates or measures, which keeps mutation operators free to treat the
//! genotype as an anonymous vector. Layout is [coordination][one logit per anchor][pair blocks],
//! and the pair-block stride comes from Params so the formula lives in exactly one place.

use crate::engine::params::Params;
use crate::engine::kernel::CUTOFF_SIGMA;
use crate::engine::resolve::Probe;
use crate::util::{finite, Rng};

pub const COORDINATION_BOUNDS: (f32, f32) = (3.0, 20.0);
const TRAIT_LOGIT_BOUNDS: (f32, f32) = (-4.0, 4.0);
const MAX_KERNEL_REACH: f32 = 0.375; // of the box, so a shell never wraps into itself

/// Campaign-level envelope: the fixed parts of a world no genome may vary, plus the shape used
/// to size freshly generated genomes. Every genome in a campaign shares this one shape; parsers
/// verify a stored genome's length against gene_len before decoding.
#[derive(Clone, Debug, PartialEq)]
pub struct Caps {
    pub particle_count: usize,
    pub dimensions: usize,
    pub anchor_count: usize,
    pub radius: f32,
    pub rate: f32, // integration rate, dt is its reciprocal
    pub seed: u64,
    pub shells: usize,
    pub bumps: usize,
}
impl Default for Caps {
    fn default() -> Self {
        Caps { particle_count: 1200, dimensions: 3, anchor_count: 4, radius: 12.0, rate: 10.0, seed: 1, shells: 3, bumps: 2 }
    }
}
impl Caps {
    /// A Params with every scalar filled and the gene vectors empty: enough for the layout
    /// helpers and the density probe, which read counts and radius only.
    fn skeleton(&self) -> Params {
        Params { particle_count: self.particle_count, dimensions: self.dimensions, coordination: 0.0,
            radius: self.radius, dt: 1.0 / self.rate, seed: self.seed, anchor_count: self.anchor_count,
            shells: self.shells, bumps: self.bumps, trait_distribution: Vec::new(),
            interactions: Vec::new(), box_len: 0.0 }
    }
    /// Full genome length for this shape.
    pub fn gene_len(&self) -> usize { self.skeleton().gene_len() }
    /// Runs the expensive reference measurement, once per campaign.
    pub fn probe(&self) -> Probe { Probe::new(&self.skeleton()) }
    /// Decode a full genome into runtime Params, splitting the world prefix off the pair genes,
    /// clamping coordination and logits, and resolving the box from the probe. Callers verify
    /// the genome length against gene_len first; a short genome panics here.
    pub fn params(&self, genome: &[f32], probe: &Probe) -> Params {
        let (world, interactions) = genome.split_at(1 + self.anchor_count);
        let coordination = finite(world[0], COORDINATION_BOUNDS.0).clamp(COORDINATION_BOUNDS.0, COORDINATION_BOUNDS.1);
        let mut params = self.skeleton();
        params.coordination = coordination;
        params.trait_distribution = world[1..].iter()
            .map(|&gene| finite(gene, 0.0).clamp(TRAIT_LOGIT_BOUNDS.0, TRAIT_LOGIT_BOUNDS.1)).collect();
        params.interactions = interactions.to_vec();
        params.box_len = probe.box_len(coordination);
        params
    }
    /// Per-gene bounds for the full genome. Pair-block bounds are widest at the largest box, so
    /// they are taken at maximum coordination and stay valid for every genome a search produces.
    pub fn bounds(&self, probe: &Probe) -> Vec<(f32, f32)> {
        let mut bounds = Vec::with_capacity(self.gene_len());
        bounds.push(COORDINATION_BOUNDS);
        bounds.extend(std::iter::repeat_n(TRAIT_LOGIT_BOUNDS, self.anchor_count));
        bounds.extend(self.pair_bounds(probe.box_len(COORDINATION_BOUNDS.1)));
        bounds
    }
    /// Bounds for the pair-block genes alone, at one specific box size. Rollout re-clamps a
    /// genome's pair genes at its own resolved box, where reach is tighter than the widest-box
    /// bounds a search mutates under.
    pub fn pair_bounds(&self, box_len: f32) -> Vec<(f32, f32)> {
        let reach = MAX_KERNEL_REACH * box_len;
        let peak_max = (2.0 * self.radius).min(reach * 0.5).max(1e-3);
        let width_min = (0.05 * self.radius).min(peak_max * 0.25).max(1e-4);
        let width_max = self.radius.min(reach / (2.0 * CUTOFF_SIGMA)).max(width_min * 2.0);
        let mut bounds = Vec::with_capacity(self.gene_len() - 1 - self.anchor_count);
        for _ in 0..self.anchor_count * self.anchor_count {
            for _ in 0..self.shells { bounds.extend([(0.0, 1.0), (0.0, peak_max), (width_min, width_max)]); } // amp, peak, width
            for _ in 0..self.bumps { bounds.extend([(0.0, 1.0), (0.0, 3.0), (0.05, 1.5)]); } // over sensed density
            bounds.push((-100.0, 100.0)); // directed weight
        }
        bounds
    }
    /// A reproducible starting probe: one cohesion shell per pair, growth peaked at moderate
    /// density, self-interaction weighted. It passes the same gate as everything else.
    pub fn default_genome(&self, probe: &Probe) -> Vec<f32> {
        let mut genome = vec![9.0]; // coordination
        genome.extend(std::iter::repeat_n(0.0, self.anchor_count)); // uniform trait logits
        let skeleton = self.skeleton();
        let mut pairs = vec![0.0; skeleton.gene_len() - 1 - self.anchor_count];
        let stride = skeleton.pair_stride();
        for pair in 0..self.anchor_count * self.anchor_count {
            let base = pair * stride;
            for shell in 0..self.shells {
                pairs[base + shell * 3] = if shell == 0 { 1.0 } else { 0.0 };
                pairs[base + shell * 3 + 1] = self.radius;
                pairs[base + shell * 3 + 2] = self.radius * 0.5;
            }
            let bump_base = base + self.shells * 3;
            for bump in 0..self.bumps {
                pairs[bump_base + bump * 3] = if bump == 0 { 1.0 } else { 0.0 };
                pairs[bump_base + bump * 3 + 1] = 1.5;
                pairs[bump_base + bump * 3 + 2] = 0.5;
            }
            let diagonal = pair / self.anchor_count == pair % self.anchor_count;
            pairs[bump_base + self.bumps * 3] = if diagonal { 40.0 } else { 0.0 };
        }
        genome.extend(pairs);
        clamp(&mut genome, &self.bounds(probe));
        genome
    }
}

/// Clamp a genome into its bounds in place, replacing any non-finite gene with the lower bound.
pub fn clamp(genome: &mut [f32], bounds: &[(f32, f32)]) {
    for (gene, &(low, high)) in genome.iter_mut().zip(bounds) {
        *gene = if gene.is_finite() { gene.clamp(low, high) } else { low };
    }
}

pub fn random_genome(bounds: &[(f32, f32)], rng: &mut Rng) -> Vec<f32> {
    bounds.iter().map(|&(low, high)| rng.range(low, high)).collect()
}
