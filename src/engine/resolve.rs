//! Deriving a genome's box size from its coordination gene: how many neighbors a particle
//! should sense, converted into how big the box needs to be for that to hold.

use crate::engine::kernel::{strength_and_slope, distance_sq, Shell};
use crate::engine::params::FixedGenome;
use crate::engine::substrate::Substrate;

/// Fixed seed for the shared reference population both calibrations use: Probe's per-campaign box
/// measurement here, and Sim::new's per-run uniform-trait population that norm_densities measures against.
/// A wandering seed would turn either calibration into noise.
pub const CALIBRATION_SEED: u64 = 0xA710_5EED_D15C_0A11;
/// How many neighbors a particle may be asked to sense. The gene's range lives here because this
/// is what converts it into a box, and a value outside it has no box to resolve to.
pub const COORDINATION_BOUNDS: (f32, f32) = (3.0, 20.0);
const TRIAL_BOX_RADII: f32 = 6.0;
const PROBE_PARTICLES: usize = 1500;

/// Measured density reference shared by every genome in a campaign. The expensive probe depends
/// on particle count, dimensions, and radius only, none of which vary per run, so build one
/// per campaign and reuse it; each genome's own coordination gene only cheaply rescales it.
pub struct Probe {
    trial_box_len: f32,
    measured_coordination: f32,
    dimensions: usize,
}
impl Probe {
    /// Runs the expensive O(n^2) reference measurement once per campaign.
    pub fn new(fixed_genome: &FixedGenome) -> Probe {
        let trial_box_len = fixed_genome.radius * TRIAL_BOX_RADII;
        Probe { trial_box_len, measured_coordination: measure_coordination(fixed_genome, trial_box_len), dimensions: fixed_genome.dimensions }
    }
    /// Cheap per-run rescale: the box size that hits 'coordination' neighbors per particle.
    pub fn box_len(&self, coordination: f32) -> f32 {
        if self.measured_coordination > 1e-9 {
            self.trial_box_len * (self.measured_coordination / coordination.max(1e-3)).powf(1.0 / self.dimensions as f32)
        } else { self.trial_box_len }
    }
}
/// Mean nominal potential a particle senses from a uniform population, using a shell shaped by radius alone, 
/// not any one genome's actual interaction law, so all campaign candidates measure the same reference.
fn measure_coordination(fixed_genome: &FixedGenome, trial_box_len: f32) -> f32 {
    let probe_count = fixed_genome.particle_count.min(PROBE_PARTICLES);
    let density_ratio = fixed_genome.particle_count.max(1) as f64 / probe_count as f64;
    let mut probe_genome = fixed_genome.clone();
    probe_genome.particle_count = probe_count; probe_genome.seed = CALIBRATION_SEED;
    let substrate = Substrate::build(&probe_genome, trial_box_len);
    let nominal = [Shell { amp: 1.0, peak: fixed_genome.radius, width: fixed_genome.radius * 0.5 }];
    let mut total_potential = 0.0f64;
    for i in 0..substrate.traits.len() {
        let position_i = substrate.pos(i);
        for j in 0..substrate.traits.len() {
            if i == j { continue; }
            total_potential += strength_and_slope(
                distance_sq(position_i, substrate.pos(j), &substrate, &mut []).sqrt(),
                &nominal).0 as f64;
        }
    }
    (total_potential / probe_count as f64 * density_ratio) as f32
}
