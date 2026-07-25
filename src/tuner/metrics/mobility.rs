//! How far particles actually travel between samples, as a fraction of the box. Minimum-image, so a
//! particle crossing the wrap seam reads as a small step rather than a box-length leap. The engine's
//! softened distance floors this near 1e-3 of the box even when frozen, so a gate sits above that.

use crate::engine::kernel::distance_sq;
use super::{Blocks, Metric, Spec};

const WIDTH: usize = 1;
pub const METRIC: Metric = &Spec { motion: true, ..Spec::of("mobility", WIDTH, measure) };
/// Travel of a tenth of the box between samples is already fast; past that the axis saturates
const CEILING: f32 = 0.1;

pub fn measure(base: &Blocks, _: &[f32]) -> Vec<f32> {
    let substrate = base.substrate;
    let dims = substrate.dimensions;
    let travelled = base.motion.as_ref().map_or(0.0, |was| {
        let count = was.positions.len() / dims;
        if was.positions.len() != substrate.positions.len() || count == 0 { return 0.0; }
        (0..count).map(|i| distance_sq(substrate.pos(i), &was.positions[i * dims..(i + 1) * dims],
            substrate, &mut []).sqrt()).sum::<f32>() / count as f32 / substrate.box_len
    });
    vec![(travelled / CEILING).clamp(0.0, 1.0)]
}
