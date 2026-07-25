//! How one-sided the interaction law is. Every other metric reads the sim, this one reads the law
//! that produced it, so a run can tell a mutual attraction from a chase where one anchor pursues
//! another that ignores it back.

use super::{Blocks, Metric, Spec};

const WIDTH: usize = 1;
pub const SPEC: Spec = Spec::of("asymmetry", WIDTH, measure);
pub const METRIC: Metric = &SPEC;

/// Departure of the directed weight matrix from its own transpose, normalized by the weights, so it
/// reads the same for a gentle law and a violent one: 0 when every pair pulls exactly as hard as it
/// is pulled, rising as the two disagree.
pub fn measure(base: &Blocks, _: &[f32]) -> Vec<f32> {
    let matrix = base.rollout.matrix;
    let anchor_count = matrix.anchor_count;
    let (mut difference, mut total) = (0.0f32, 0.0f32);
    for source in 0..anchor_count {
        for destination in 0..anchor_count {
            let forward = matrix.interactions[source * anchor_count + destination].weight;
            let back = matrix.interactions[destination * anchor_count + source].weight;
            difference += (forward - back).powi(2); total += forward.powi(2);
        }
    }
    vec![if total < 1e-12 { 0.0 } else { (difference / total).sqrt().min(1.0) }]
}
