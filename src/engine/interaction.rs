//! The pair control net: one directed interaction law per ordered anchor pair, decoded from a
//! genome and calibrated against a uniform reference population.
//!
//! A pair's position in [`Net`] *is* its `(source, destination)` identity, so neither the
//! interaction nor its callers carry those indices around to be checked against each other.

use crate::engine::genome::Layout;
use crate::engine::grid::Grid;
use crate::engine::kernel::{bump_and_slope, displacement, displacement_cutoff, Shell};
use crate::engine::r#trait::{membership, Membership};
use crate::engine::substrate::Substrate;

/// One directed law: what a receiver anchor senses from a source anchor, and how it responds.
pub struct Interaction {
    pub shells: Vec<Shell>,
    pub bumps: Vec<Shell>,
    pub weight: f32,
    /// Uniform-seed mean of this source-anchor potential for this destination anchor.
    pub norm: f32,
    pub reach: f32,
}

impl Interaction {
    fn refresh_reach(&mut self) {
        self.reach = displacement_cutoff(&self.shells);
    }
}

/// An `anchors × anchors` source-major control net.
pub struct Net {
    anchors: usize,
    pairs: Vec<Interaction>,
}

impl Net {
    /// Decode a genome's pair-block genes, source-major per [`Layout::index`].
    pub fn from_genome(layout: &Layout, genome: &[f32]) -> Net {
        let stride = layout.genes_per_interaction();
        let bump_base = layout.shells * 3;
        let pairs = (0..layout.anchors * layout.anchors)
            .map(|pair| {
                let genes = &genome[pair * stride..][..stride];
                let read = |base: usize, count: usize| {
                    (0..count)
                        .map(|slot| Shell {
                            amp: genes[base + slot * 3],
                            peak: genes[base + slot * 3 + 1],
                            width: genes[base + slot * 3 + 2],
                        })
                        .collect()
                };
                let mut interaction = Interaction {
                    shells: read(0, layout.shells),
                    bumps: read(bump_base, layout.bumps),
                    weight: genes[bump_base + layout.bumps * 3],
                    norm: 1.0,
                    reach: 0.0,
                };
                interaction.refresh_reach();
                interaction
            })
            .collect();
        Net {
            anchors: layout.anchors,
            pairs,
        }
    }

    pub fn anchors(&self) -> usize {
        self.anchors
    }

    pub fn pairs(&self) -> &[Interaction] {
        &self.pairs
    }

    fn index(&self, source: usize, destination: usize) -> usize {
        source * self.anchors + destination
    }

    pub fn pair(&self, source: usize, destination: usize) -> &Interaction {
        &self.pairs[self.index(source, destination)]
    }

    /// Widest kernel reach in the net, which sets the neighbour-search cutoff.
    pub fn max_reach(&self) -> f32 {
        self.pairs
            .iter()
            .map(|interaction| interaction.reach)
            .fold(0.0, f32::max)
    }

    /// Replace measured norms restored from a checkpoint, validating them at that boundary.
    pub fn set_norms(&mut self, norms: &[f32]) {
        assert_eq!(norms.len(), self.pairs.len());
        for (interaction, &norm) in self.pairs.iter_mut().zip(norms) {
            assert!(norm.is_finite() && norm > 0.0);
            interaction.norm = norm;
        }
    }

    pub fn norms(&self) -> impl Iterator<Item = f32> + '_ {
        self.pairs.iter().map(|interaction| interaction.norm)
    }

    /// Measure each interaction's density norm on `substrate`: the mean weighted potential a
    /// receiver anchor senses from a source anchor, over every neighbour within kernel reach.
    pub fn calibrate(&mut self, substrate: &Substrate, extent: f32, softening: f32) {
        for interaction in self.pairs.iter_mut() {
            interaction.refresh_reach();
        }
        let mut grid = Grid::default();
        grid.rebuild(
            &substrate.positions,
            substrate.dims(),
            extent,
            self.max_reach(),
        );
        let inverse_extent = 1.0 / extent;
        let softening_squared = softening * softening;
        let memberships: Vec<Membership> = substrate
            .traits
            .iter()
            .map(|&trait_value| membership(trait_value, self.anchors))
            .collect();

        let anchors = self.anchors;
        for (pair, interaction) in self.pairs.iter_mut().enumerate() {
            let (source_anchor, destination_anchor) = (pair / anchors, pair % anchors);
            let mut total = 0.0f64;
            let mut receiver_mass = 0.0f64;
            for i in 0..substrate.len() {
                let receiver_weight = memberships[i].weight(destination_anchor);
                if receiver_weight <= 0.0 {
                    continue;
                }
                receiver_mass += receiver_weight as f64;
                let position_i = substrate.at(i);
                let mut visit = |j: usize| {
                    if i == j {
                        return;
                    }
                    let source_weight = memberships[j].weight(source_anchor);
                    if source_weight <= 0.0 {
                        return;
                    }
                    let distance = displacement(
                        position_i,
                        substrate.at(j),
                        extent,
                        inverse_extent,
                        softening_squared,
                    );
                    if distance <= interaction.reach {
                        total += (receiver_weight
                            * source_weight
                            * bump_and_slope(distance, &interaction.shells).0)
                            as f64;
                    }
                };
                grid.for_each_candidate(position_i, substrate.len(), &mut visit);
            }
            let mean = total / receiver_mass.max(1.0);
            interaction.norm = if mean > 1e-6 { mean as f32 } else { 1.0 };
        }
    }
}
