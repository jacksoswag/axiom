//! Matrix is one directed AnchorInteraction law per ordered anchor pair, derived from a genome and
//! calibrated against a uniform reference population. Pair position in Matrix is its (source, destination)
//! identity, so neither the interaction nor its callers check indices against each other.

use crate::engine::params::{FixedGenome, Genome};
use crate::engine::kernel::{strength_and_slope, distance_sq, distance_cutoff, Shell};
use crate::engine::substrate::Substrate;

/// How two anchors interact: shells sense over distance, bumps respond over density, weight scales it.
/// The two mixtures live in Matrix's flat arrays instead of here: every pair carries the same count of
/// each, and one allocation reads better in the step loop than A^2 x 2 separate little ones.
pub struct AnchorInteraction {
    pub weight: f32, // scales growth response to an actual force
    pub norm: f32, // source-anchor potential for destination anchor
    pub reach_sq: f32, // squared significant interaction radius, so the pair check skips a sqrt
}
/// An [anchors x anchors] matrix, complexity scales with anchor-count
pub struct Matrix {
    pub anchor_count: usize,
    pub interactions: Vec<AnchorInteraction>, // one entry per pair of anchors
    pub shells: Vec<Shell>, // K(d) over distance, every pair's end to end, shell_count apiece
    pub bumps: Vec<Shell>, // G(u) over sensed potential, the same way
    pub shell_count: usize, // stride into shells
    pub bump_count: usize, // stride into bumps
    pub max_reach: f32, // widest kernel reach in the matrix, the smallest cell size rebuild_grid may use
}
impl Matrix {
    /// Derive the matrix from a genome's pair-block genes, source-major over genome.interactions.
    pub fn derive(fg: &FixedGenome, g: &Genome) -> Matrix {
        let stride = fg.pair_stride(); // indices in gene pair: 3/shell, 3/bump, 1 weight
        let bump_start = fg.shells * 3;
        let pair_count = fg.anchor_count * fg.anchor_count;
        let mut matrix = Matrix { anchor_count: fg.anchor_count, max_reach: 0.0,
            interactions: Vec::with_capacity(pair_count),
            shells: Vec::with_capacity(pair_count * fg.shells),
            bumps: Vec::with_capacity(pair_count * fg.bumps),
            shell_count: fg.shells, bump_count: fg.bumps };
        for pair in 0..pair_count {
            let genes = &g.interactions[pair * stride..][..stride];
            Self::read_triplets(genes, 0, fg.shells, &mut matrix.shells);
            Self::read_triplets(genes, bump_start, fg.bumps, &mut matrix.bumps);
            let reach = distance_cutoff(&matrix.shells[pair * fg.shells..][..fg.shells]);
            matrix.max_reach = matrix.max_reach.max(reach);
            matrix.interactions.push(AnchorInteraction {
                weight: genes[bump_start + fg.bumps * 3], norm: 1.0, reach_sq: reach * reach });
        }
        matrix
    }
    /// One pair's sensing mixture, out of the flat array every pair shares
    pub fn shells_of(&self, pair: usize) -> &[Shell] { &self.shells[pair * self.shell_count..][..self.shell_count] }
    /// One pair's growth mixture, likewise
    pub fn bumps_of(&self, pair: usize) -> &[Shell] { &self.bumps[pair * self.bump_count..][..self.bump_count] }
    /// Measure each interaction's density norm on substrate: the mean weighted potential a
    /// receiver anchor senses from a source anchor, over every neighbor within kernel reach.
    /// One sweep of the population, not one per pair: a neighbor visit lands in the at most four
    /// (source, destination) pairs the two memberships pick out, so the walk and its distance math are
    /// paid once where pair-major order paid them A^2 times over.
    pub fn norm_densities(&mut self, substrate: &mut Substrate) {
        let anchor_count = self.anchor_count;
        substrate.rebuild_grid(self.max_reach); // grid w best cell size
        let memberships = &substrate.memberships; // every particle's anchor memberships
        let mut total = vec![0.0f64; self.interactions.len()]; // sensed potential, per pair
        let mut receiver_mass = vec![0.0f64; anchor_count]; // how much of the population each anchor holds

        for i in 0..substrate.traits.len() { // for each particle
            let receiver = memberships[i]; // this particle's up-to-2 active anchors
            for &(destination_anchor, receiver_weight) in &receiver.entries {
                if receiver_weight > 0.0 { receiver_mass[destination_anchor as usize] += receiver_weight as f64; }
            }
            let i_pos = substrate.pos(i);
            let mut visit = |j: usize| {
                if i == j { return; } // self-relationships don't count
                let source = memberships[j]; // neighbor's 2 active anchors
                let distance_sq = distance_sq(i_pos, substrate.pos(j), substrate, &mut []);
                let mut distance = -1.0; // lazy: same value for every pair below, sqrt at most once
                for &(source_anchor, source_weight) in &source.entries {
                    if source_weight <= 0.0 { continue; } // only neighbors adjacent to the source anchor
                    for &(destination_anchor, receiver_weight) in &receiver.entries {
                        if receiver_weight <= 0.0 { continue; }
                        let pair = source_anchor as usize * anchor_count + destination_anchor as usize;
                        if distance_sq > self.interactions[pair].reach_sq { continue; } // squared compare skips the sqrt
                        if distance < 0.0 { distance = distance_sq.sqrt(); }
                        total[pair] += (receiver_weight * source_weight // interaction magnitude
                            * strength_and_slope(distance, self.shells_of(pair)).0) as f64;
                    }
                }
            };
            substrate.visit_neighbors(i_pos, &mut visit);
        }
        for (pair, interaction) in self.interactions.iter_mut().enumerate() {
            let mass = receiver_mass[pair % anchor_count]; // this pair's destination anchor holds it
            let norm = total[pair] / mass.max(1e-9); // mean per unit receiver mass, floor only guards 0/0
            interaction.norm = if norm > 1e-6 { norm as f32 } else { 1.0 };
        }
    }
    /// Helper for derive. Decodes amp/peak/width triplets onto the end of a mixture, shells and bumps
    /// differ only in start index
    fn read_triplets(genes: &[f32], start: usize, count: usize, into: &mut Vec<Shell>) {
        for slot in 0..count {
            into.push(Shell::new(genes[start + slot * 3], genes[start + slot * 3 + 1], genes[start + slot * 3 + 2]));
        }
    }
}
