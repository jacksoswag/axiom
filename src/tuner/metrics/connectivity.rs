//! Is there a connected medium, and connected empty space to move through. A phase only counts if
//! it holds enough of the grid to be a carrier, and then only the part looping around the torus: a
//! compact blob is disconnected here no matter how many of its cells touch.

use super::field::{SpatialField, GRID_AXES};
use super::{Blocks, Metric, Spec};

/// Multiples of mean density separating void from material, read at both so faint structure and
/// sharp structure are told apart
const THRESHOLDS: [f32; 2] = [0.5, 1.5];
/// Coarse enough for a few-hundred-particle sim to show corridors, fine enough to resolve bodies at
/// a few thousand
pub const SIDE: usize = 8;
const WIDTH: usize = THRESHOLDS.len() * 2;
pub const METRIC: Metric = &Spec { sides: &[SIDE], ..Spec::of("connectivity", WIDTH, measure) };
/// A qualifying phase holds at least 8 percent of the grid
const MIN_PHASE_VOLUME: f32 = 0.08;

/// Dense fractions first, then void, which is the order criterion::structure splits them on.
pub fn measure(base: &Blocks, _: &[f32]) -> Vec<f32> {
    let field = base.field(SIDE);
    let density = field.smoothed();
    let mean = density.iter().sum::<f32>() / density.len().max(1) as f32;
    // One set of scratch for all four fills. Each one used to allocate and throw away a grid's worth
    // of flags and lifts, four times per sample, for the same answer.
    let mut work = Work { included: vec![false; density.len()], lifts: vec![None; density.len()], stack: Vec::new() };
    let dense = THRESHOLDS.map(|edge| winding_fraction(field, &density, |v| v >= edge * mean, &mut work));
    let void = THRESHOLDS.map(|edge| winding_fraction(field, &density, |v| v < edge * mean, &mut work));
    dense.into_iter().chain(void).collect()
}

/// The three grid-sized buffers a flood fill walks on, kept apart from the walk so four fills share one set
struct Work { included: Vec<bool>, lifts: Vec<Option<[i32; GRID_AXES]>>, stack: Vec<usize> }

/// Largest winding-component fraction, or 0 without both volume and a torus loop. Flood fill
/// carrying a lift: the integer count of steps taken along each axis to reach a cell without
/// wrapping. Arriving somewhere already visited with a disagreeing lift means the two walks differ
/// by a full loop around the torus, which detects a box-spanning phase using nothing but integers.
fn winding_fraction(field: &SpatialField, density: &[f32], include: impl Fn(f32) -> bool, work: &mut Work) -> f32 {
    let Work { included, lifts, stack } = work;
    for (cell, &value) in density.iter().enumerate() { included[cell] = include(value); }
    let total = included.iter().filter(|v| **v).count();
    if total == 0 || (total as f32 / included.len() as f32) < MIN_PHASE_VOLUME { return 0.0; }
    lifts.fill(None);
    let mut largest_winding = 0;
    for root in 0..included.len() {
        if !included[root] || lifts[root].is_some() { continue; }
        stack.clear(); stack.push(root);
        lifts[root] = Some([0i32; GRID_AXES]);
        let (mut size, mut winds) = (0, [false; GRID_AXES]);
        while let Some(cell) = stack.pop() {
            size += 1;
            for axis in 0..GRID_AXES {
                for direction in [-1, 1] {
                    let next = field.neighbor(cell, axis, direction);
                    if !included[next] { continue; }
                    let mut proposed = lifts[cell].expect("visited cell has a lift");
                    proposed[axis] += direction as i32;
                    if let Some(existing) = lifts[next] {
                        for winding_axis in 0..GRID_AXES { winds[winding_axis] |= proposed[winding_axis] != existing[winding_axis]; }
                    } else { lifts[next] = Some(proposed); stack.push(next); }
                }
            }
        }
        if winds.into_iter().any(|wrapped| wrapped) { largest_winding = largest_winding.max(size); }
    }
    largest_winding as f32 / total as f32
}
