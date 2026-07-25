//! How far a sim sits from randomness, in one number. Mean absolute departure of the radial ratios
//! from 1, so a gas reads 0 and any preferred spacing pushes it up. Reads the pair histogram through
//! rdf rather than measuring anything itself.

use super::{rdf, Blocks, Metric, Spec};

const WIDTH: usize = 1;
pub const METRIC: Metric = &Spec { pairs: true, ..Spec::of("structure", WIDTH, measure) };

pub fn measure(base: &Blocks, _: &[f32]) -> Vec<f32> {
    let ratios = rdf::ratios(base.pairs(), base.rollout.baseline);
    // clamped like every other slot: raw departure runs toward MAX_RATIO - 1 and would swamp a distance
    vec![(ratios.iter().map(|v| (v - 1.0).abs()).sum::<f32>() / ratios.len() as f32).clamp(0.0, 1.0)]
}
