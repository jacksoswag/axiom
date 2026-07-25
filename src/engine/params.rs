//! A run's parameters in two halves: FixedGenome, the shape a campaign holds still,
//! and Genome, the per-run values search tunes. Layout, the flat-to-Genome split, uniform genome
//! draws, and the legal range of every gene live here, so nothing outside can disagree about what
//! a gene means or how far it may travel.

use crate::engine::kernel::CUTOFF_SIGMA;
use crate::engine::resolve::{Probe, COORDINATION_BOUNDS};
use crate::util::{finite, Rng};

const TRAIT_LOGIT_BOUNDS: (f32, f32) = (-4.0, 4.0);
const MAX_KERNEL_REACH: f32 = 0.375; // of the box, so a shell never wraps into itself

#[derive(Clone)]
pub struct FixedGenome {
    pub particle_count: usize,
    pub dimensions: usize,
    pub anchor_count: usize,
    pub shells: usize,
    pub bumps: usize,
    pub radius: f32,
    pub dt: f32, // integration timestep
    pub seed: u64,
}
impl FixedGenome {
    /// Genes in one pair block: amp/peak/width per shell and per bump, then the directed weight.
    pub fn pair_stride(&self) -> usize { 3 * (self.shells + self.bumps) + 1 }
    /// Full genome length: coordination, one logit per anchor, then anchor_count^2 pair blocks.
    pub fn gene_len(&self) -> usize { 1 + self.anchor_count + self.anchor_count * self.anchor_count * self.pair_stride() }
    /// Split a flat genome into its coordination, per-anchor logits, and pair blocks.
    pub fn decode(&self, genes: &[f32]) -> Genome {
        Genome {
            coordination: finite(genes[0], COORDINATION_BOUNDS.0).clamp(COORDINATION_BOUNDS.0, COORDINATION_BOUNDS.1),
            trait_distribution: genes[1..1 + self.anchor_count].iter()
                .map(|&gene| finite(gene, 0.0).clamp(TRAIT_LOGIT_BOUNDS.0, TRAIT_LOGIT_BOUNDS.1)).collect(),
            interactions: genes[1 + self.anchor_count..].to_vec(),
        }
    }
    /// Per-gene bounds for a full genome. Pair-block bounds are widest at the largest box, so they
    /// are taken at maximum coordination and stay valid for every genome a search can produce.
    pub fn bounds(&self, probe: &Probe) -> Vec<(f32, f32)> {
        let mut bounds = Vec::with_capacity(self.gene_len());
        bounds.push(COORDINATION_BOUNDS);
        bounds.extend(std::iter::repeat_n(TRAIT_LOGIT_BOUNDS, self.anchor_count));
        bounds.extend(self.pair_bounds(probe.box_len(COORDINATION_BOUNDS.1)));
        bounds
    }
    /// Bounds for the pair-block genes alone, at one specific box size. Rollout re-clamps at the
    /// box its genome resolves to, where reach is tighter than the widest-box bounds a search mutates under.
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
}
#[derive(Clone)]
pub struct Genome {
    pub coordination: f32,
    pub trait_distribution: Vec<f32>, // anchor distribution
    pub interactions: Vec<f32>, // flat pair-block genes, source-major, decoded by Matrix::derive()
}
impl Genome {
    /// Generate deterministic vector Genome from seed
    pub fn build_random(bounds: &[(f32, f32)], seed: u64) -> Vec<f32> {
        let mut rng = Rng::new(seed);
        bounds.iter().map(|&(low, high)| rng.range(low, high)).collect()
    }
}
/// Clamp a flat genome into its bounds in place
pub fn clamp(genome: &mut [f32], bounds: &[(f32, f32)]) {
    for (gene, &(low, high)) in genome.iter_mut().zip(bounds) {
        *gene = if gene.is_finite() { gene.clamp(low, high) } else { low };
    }
}
