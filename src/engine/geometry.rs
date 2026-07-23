//! Derived world geometry. `extent` is never a free parameter: it falls out of particle count,
//! interaction radius, and the genome's target coordination, against one measured density
//! reference.

use crate::engine::kernel::{bump_and_slope, displacement, Shell};
use crate::engine::rng::Rng;
use crate::engine::substrate::Substrate;

const TRIAL_EXTENT_RADII: f32 = 6.0;
const PROBE_PARTICLES: usize = 1500;

/// Seed of the reference population every genome calibrates against, so a clumped live state
/// can never set the density reference.
pub const CALIBRATION_SEED: u64 = 0xA710_5EED_D15C_0A11;

#[derive(Clone, Copy, Debug)]
pub struct Geometry {
    pub extent: f32,
    pub softening: f32,
}

/// Measured density reference shared by every genome in a search campaign.
///
/// The expensive probe depends on particle count, dimensionality, and radius only. A genome's
/// coordination gene only rescales this reference, so measuring it per candidate adds cost
/// without changing the answer. Build one and pass it down.
#[derive(Clone, Copy, Debug)]
pub struct GeometryScale {
    trial_extent: f32,
    measured_coordination: f32,
    dimensions: usize,
}

impl GeometryScale {
    pub fn probe(particles: usize, dimensions: usize, radius: f32) -> GeometryScale {
        let trial_extent = radius * TRIAL_EXTENT_RADII;
        debug_assert!(2.5 * radius < trial_extent * 0.5, "the density probe wraps");
        GeometryScale {
            trial_extent,
            measured_coordination: measure_coordination(
                particles,
                dimensions,
                radius,
                trial_extent,
            ),
            dimensions,
        }
    }

    /// Rescale the reference to the box size that hits `coordination` neighbours per particle.
    pub fn geometry(self, coordination: f32) -> Geometry {
        let extent = if self.measured_coordination > 1e-9 {
            self.trial_extent
                * (self.measured_coordination / coordination.max(1e-3))
                    .powf(1.0 / self.dimensions as f32)
        } else {
            self.trial_extent
        };
        Geometry {
            extent,
            softening: extent * 1e-3,
        }
    }
}

/// Mean nominal potential one particle senses from a uniform population at the trial extent.
/// Measured on a capped sample and scaled back up, so the cost does not grow with the world.
fn measure_coordination(particles: usize, dimensions: usize, radius: f32, extent: f32) -> f32 {
    let total = particles.max(1);
    let probe_count = total.min(PROBE_PARTICLES);
    let density_ratio = total as f64 / probe_count as f64;
    let mut substrate = Substrate::new(probe_count, dimensions);
    let mut rng = Rng::new(CALIBRATION_SEED);
    for position in substrate.positions.iter_mut() {
        *position = rng.unit() * extent;
    }

    let nominal = [Shell {
        amp: 1.0,
        peak: radius,
        width: radius * 0.5,
    }];
    let inverse_extent = 1.0 / extent;
    let softening_squared = (extent * 1e-3).powi(2);
    let mut total_potential = 0.0f64;
    for i in 0..substrate.len() {
        let position_i = substrate.at(i);
        for j in 0..substrate.len() {
            if i == j {
                continue;
            }
            total_potential += bump_and_slope(
                displacement(
                    position_i,
                    substrate.at(j),
                    extent,
                    inverse_extent,
                    softening_squared,
                ),
                &nominal,
            )
            .0 as f64;
        }
    }
    (total_potential / probe_count as f64 * density_ratio) as f32
}
