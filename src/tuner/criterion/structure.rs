//! Reward a genome for holding regions that differ, keeping open space between them, surviving
//! damage, and still turning over. Absolute, so unlike novelty it can be climbed toward and unlike
//! novelty it can be fooled: a static lattice satisfies the contrast term outright.

use crate::tuner::metrics::{connectivity, heterogeneity, robustness, scalar, slice, turnover, Metric};

/// Exactly what score() reads below, beside it so the two move in one edit. A metric missing from a
/// plan reads 0 and silently zeroes the product, so drift here is a bug that never announces itself
pub const METRICS: [Metric; 4] = [heterogeneity::METRIC, connectivity::METRIC, turnover::METRIC,
    robustness::METRIC];

/// A clause reading zero costs FLOOR rather than annihilating the product. Multiplying raw, a sim
/// that misses one clause scored exactly 0 no matter how it did on the other four, and since a
/// tournament breaks ties toward its first draw, a population of zeroes made selection random.
/// Floored, the AND still costs an order of magnitude and the other four stay rankable.
const FLOOR: f32 = 0.02;

/// Clauses multiply, so weakness anywhere costs everywhere; a sum would let beautiful contrast and
/// no motion whatsoever place well. Contrast is read at the scale that shows it most, since
/// sub-structure lives at one scale and averaging across them hides it.
pub fn score(values: &[f32], plan: &[Metric]) -> f32 {
    let contrast = slice(values, plan, heterogeneity::METRIC).chunks_exact(heterogeneity::VALUES)
        .map(|scale| scale[0].min(scale[2])) // density contrast and trait contrast at this scale
        .fold(0.0, f32::max);
    let connectivity = slice(values, plan, connectivity::METRIC);
    let half = connectivity.len() / 2; // dense fractions first, then void
    let material = connectivity[..half].iter().copied().fold(1.0, f32::min);
    let void = connectivity[half..].iter().copied().fold(1.0, f32::min);
    [contrast, material, void, scalar(values, plan, robustness::METRIC),
        scalar(values, plan, turnover::METRIC)].into_iter().map(|clause| clause.max(FLOOR)).product()
}
