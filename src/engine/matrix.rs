//! Matrix is one directed AnchorInteraction law per ordered anchor pair, derived from a genome and
//! calibrated against a uniform reference population. Pair position in Matrix is its (source, destination) 
//! identity, so neither the interaction nor its callers check indices against each other.

use crate::engine::params::{FixedGenome, Genome};
use crate::engine::kernel::{strength_and_slope, distance_sq, distance_cutoff, Shell};
use crate::engine::r#trait::memberships;
use crate::engine::substrate::Substrate;

/// How two anchors interact: shells sense over distance, bumps respond over density, weight scales it
pub struct AnchorInteraction {
    pub shells: Vec<Shell>, // K(d), sensed potential over distance
    pub bumps: Vec<Shell>, // G(u), growth response over sensed potential
    pub weight: f32, // scales growth response to an actual force
    pub norm: f32, // source-anchor potential for destination anchor
    pub reach: f32, // significant interaction radius, CUTOFF_SIGMA
}
/// An [anchors x anchors] matrix, complexity scales with anchor-count
pub struct Matrix {
    pub anchor_count: usize,
    pub interactions: Vec<AnchorInteraction>, // one entry per pair of anchors
}
impl Matrix {
    /// Derive the matrix from a genome's pair-block genes, source-major over genome.interactions.
    pub fn derive(fg: &FixedGenome, g: &Genome) -> Matrix {
        let stride = fg.pair_stride(); // indices in gene pair: 3/shell, 3/bump, 1 weight
        let bump_start = fg.shells * 3;
        let pairs = (0..fg.anchor_count * fg.anchor_count).map(|pair| {
            let genes = &g.interactions[pair * stride..][..stride];
            let shells = Self::read_triplets(genes, 0, fg.shells);
            let reach = distance_cutoff(&shells);
            AnchorInteraction {
                bumps: Self::read_triplets(genes, bump_start, fg.bumps),
                weight: genes[bump_start + fg.bumps * 3],
                norm: 1.0, shells, reach}}).collect();
        Matrix { anchor_count: fg.anchor_count, interactions: pairs }
    }
    /// Measure each interaction's density norm on substrate: the mean weighted potential a
    /// receiver anchor senses from a source anchor, over every neighbor within kernel reach.
    pub fn norm_densities(&mut self, substrate: &mut Substrate) {
        let anchor_count = self.anchor_count;
        substrate.rebuild_grid(self.max_reach()); // grid w best cell size
        let memberships = memberships(&substrate.traits, anchor_count); // every particle's anchor memberships

        for (pair, interaction) in self.interactions.iter_mut().enumerate() { // for each slot in matrix
            let (source_anchor, destination_anchor) = (pair / anchor_count, pair % anchor_count); // parse from flat to 2D
            let (mut total, mut receiver_mass) = (0.0f64, 0.0f64);
            for i in 0..substrate.traits.len() { // for each particle
                let receiver_weight = memberships[i].weight(destination_anchor);
                if receiver_weight <= 0.0 { continue; } // only particles adjacent to the current anchor
                receiver_mass += receiver_weight as f64;
                let i_pos = substrate.pos(i);
                let mut visit = |j: usize| {
                    if i == j { return; } // self-relationships don't count
                    let source_weight = memberships[j].weight(source_anchor);
                    if source_weight <= 0.0 { return; }
                    let distance = distance_sq(i_pos, substrate.pos(j), substrate, &mut []).sqrt();
                    if distance <= interaction.reach {
                        total += (receiver_weight * source_weight // interaction magnitude
                            * strength_and_slope(distance, &interaction.shells).0) as f64;
                    }
                };
                substrate.visit_neighbors(i_pos, &mut visit);
            }
            let norm = total / receiver_mass.max(1e-9); // mean per unit receiver mass, floor only guards 0/0
            interaction.norm = if norm > 1e-6 { norm as f32 } else { 1.0 };
        }
    }
    /// Widest kernel reach in the matrix, the smallest cell size rebuild_grid may use
    pub fn max_reach(&self) -> f32 {
        self.interactions.iter().map(|interaction| interaction.reach).fold(0.0, f32::max)
    }
    /// Helper for derive. Decodes amp/peak/width triplets, shells and bumps differ only in start index
    fn read_triplets(genes: &[f32], start: usize, count: usize) -> Vec<Shell> {
    (0..count).map(|slot| Shell {
        amp: genes[start + slot * 3], peak: genes[start + slot * 3 + 1],
        width: genes[start + slot * 3 + 2]}).collect()
    }
}
