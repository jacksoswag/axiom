//! How much of a sim's mass changed cells between samples. Mobility asks how far particles moved,
//! this asks whether the arrangement itself changed, so a body drifting intact reads high on one
//! and low on the other.

use super::{Blocks, Metric, Spec};

const WIDTH: usize = 1;
pub const SPEC: Spec = Spec { motion: true, sides: &[SIDE], ..Spec::of("turnover", WIDTH, measure) };
pub const METRIC: Metric = &SPEC;
/// The finest grid, where a body shifting one cell still registers. Blocks::carry knows this side by
/// name: it is the one grid a tick hands forward instead of the next sample rebuilding it.
pub const SIDE: usize = 16;

pub fn measure(base: &Blocks, _: &[f32]) -> Vec<f32> {
    let after = base.field(SIDE);
    let moved = base.motion.and_then(|was| was.field.as_ref()).map_or(0.0, |before| {
        before.density.iter().zip(&after.density).map(|(a, b)| (a - b).abs()).sum::<f32>()
            / (before.density.iter().sum::<f32>() + after.density.iter().sum::<f32>()).max(1.0)
    });
    vec![moved.clamp(0.0, 1.0)]
}
