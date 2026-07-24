//! How a search campaign sizes a world before evaluating any genome: one measured density
//! reference, shared by every candidate, then cheaply rescaled to each genome's own
//! coordination gene. Deciding a world's box size is a campaign-setup concern, not something
//! the simulation core does to itself -- `engine` only ever consumes the resulting `f32` extent.

use crate::engine::genome::{Caps, Params, COORDINATION_BOUNDS};
use crate::engine::kernel::{bump_and_slope, displacement, Shell};
use crate::engine::substrate::{Substrate, CALIBRATION_SEED};

const TRIAL_EXTENT_RADII: f32 = 6.0;
const PROBE_PARTICLES: usize = 1500;

/// Mean nominal potential a particle senses from a uniform population at extent, using a
/// canonical shell shaped by radius alone -- not any one genome's actual interaction law, so
/// every candidate in a search campaign can share this reference regardless of its own shells.
/// Measured on a capped sample and scaled back up, so the cost does not grow with the world.
fn measure_coordination(particles: usize, dimensions: usize, radius: f32, extent: f32) -> f32 {
    let total = particles.max(1);
    let probe_count = total.min(PROBE_PARTICLES);
    let density_ratio = total as f64 / probe_count as f64;
    let substrate = Substrate::new(probe_count, extent, dimensions, CALIBRATION_SEED);
    let nominal = [Shell { amp: 1.0, peak: radius, width: radius * 0.5 }];
    let inverse_extent = 1.0 / extent;
    let softening_squared = substrate.softening * substrate.softening;
    let mut total_potential = 0.0f64;
    for i in 0..substrate.traits.len() {
        let position_i = substrate.at(i);
        for j in 0..substrate.traits.len() {
            if i == j { continue; }
            total_potential += bump_and_slope(
                displacement(position_i, substrate.at(j), extent, inverse_extent, softening_squared),
                &nominal,
            ).0 as f64;
        }
    }
    (total_potential / probe_count as f64 * density_ratio) as f32
}

/// Measured density reference shared by every genome in a search campaign.
///
/// The expensive probe depends on particle count, dimensionality, and radius only. A genome's
/// coordination gene only rescales this reference, so measuring it per candidate adds cost
/// without changing the answer. Build one and pass it down.
#[derive(Clone, Copy, Debug)]
pub struct DensityProbe {
    trial_extent: f32,
    measured_coordination: f32,
    dimensions: usize,
}

impl DensityProbe {
    pub fn probe(particles: usize, dimensions: usize, radius: f32) -> DensityProbe {
        let trial_extent = radius * TRIAL_EXTENT_RADII;
        debug_assert!(2.5 * radius < trial_extent * 0.5, "the density probe wraps");
        DensityProbe {
            trial_extent,
            measured_coordination: measure_coordination(particles, dimensions, radius, trial_extent),
            dimensions,
        }
    }

    /// Rescale the reference to the box size that hits `coordination` neighbours per particle.
    pub fn bound_len(self, coordination: f32) -> f32 {
        if self.measured_coordination > 1e-9 {
            self.trial_extent
                * (self.measured_coordination / coordination.max(1e-3))
                    .powf(1.0 / self.dimensions as f32)
        } else {
            self.trial_extent
        }
    }
}

/// Probe once at campaign creation and pass the result to every candidate evaluation.
pub fn probe_for(caps: &Caps) -> DensityProbe {
    DensityProbe::probe(caps.particles, caps.dimensions, caps.radius)
}

/// Convenience for the common case: derive one `Params`'s own box size directly.
pub fn bound_len_for(params: &Params) -> f32 {
    DensityProbe::probe(params.particles, params.dimensions, params.radius).bound_len(params.coordination)
}

/// The widest box any genome a `Caps` can produce will need, at maximum coordination.
pub fn widest_bound_len(caps: &Caps) -> f32 {
    probe_for(caps).bound_len(COORDINATION_BOUNDS.1)
}
