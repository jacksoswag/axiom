//! Base block: the periodic density and trait field on a cubic grid. Not a metric. Several metrics
//! read one, and the plan's union of requested resolutions decides how many get built, so no metric
//! ever constructs its own.

use crate::engine::substrate::Substrate;
use super::alive;

/// The grid is a cubic 3-axis lattice: index and neighbor take x, y, z. A const rather than a read
/// off the substrate because it sizes the per-axis arrays a flood fill carries. Search::new turns
/// away a genome shaped any other way.
pub const GRID_AXES: usize = 3;

/// Fields are public because the metrics reading them live in sibling files, and hiding them would
/// only mean forwarding methods.
pub struct SpatialField {
    pub side: usize,
    pub density: Vec<f32>,
    pub trait_sum: Vec<f32>,
    pub trait_square_sum: Vec<f32>,
    pub trait_bins: Vec<[u32; 4]>,
}

pub fn build(substrate: &Substrate, side: usize) -> SpatialField {
    let cells = side.pow(GRID_AXES as u32);
    let mut field = SpatialField {
        side, density: vec![0.0; cells], trait_sum: vec![0.0; cells],
        trait_square_sum: vec![0.0; cells], trait_bins: vec![[0; 4]; cells]};
    let box_len = substrate.box_len;
    for (i, pos) in substrate.positions.chunks_exact(substrate.dimensions).enumerate() {
        if !alive(pos) { continue; }
        let cell = |axis: usize| ((pos[axis].rem_euclid(box_len) / box_len * side as f32) as usize).min(side - 1);
        let index = field.index(cell(0), cell(1), cell(2));
        let trait_value = substrate.traits[i].clamp(0.0, 1.0);
        field.density[index] += 1.0;
        field.trait_sum[index] += trait_value;
        field.trait_square_sum[index] += trait_value * trait_value;
        field.trait_bins[index][((trait_value * 4.0) as usize).min(3)] += 1;
    }
    field
}

impl SpatialField {
    pub fn index(&self, x: usize, y: usize, z: usize) -> usize { (z * self.side + y) * self.side + x }
    /// Step one cell along one axis, wrapping. The only thing making every grid measurement
    /// periodic, so nothing else needs to know the box is a torus.
    pub fn neighbor(&self, index: usize, axis: usize, delta: isize) -> usize {
        let side = self.side as isize;
        let mut xyz = [(index % self.side) as isize, ((index / self.side) % self.side) as isize,
            (index / (self.side * self.side)) as isize];
        xyz[axis] = (xyz[axis] + delta).rem_euclid(side);
        self.index(xyz[0] as usize, xyz[1] as usize, xyz[2] as usize)
    }
    pub fn local_mean_trait(&self, index: usize) -> f32 { self.trait_sum[index] / self.density[index].max(1.0) }
    /// Separable 3-tap blur on every axis. Runs before any threshold, so one stray particle cannot
    /// register as a dense phase. One pass per axis reaches the same 27-cell cube as a single nested
    /// sweep would, for nine taps of work instead of twenty-seven and no chained index decoding.
    pub fn smoothed(&self) -> Vec<f32> {
        let mut values = self.density.clone();
        let mut blurred = vec![0.0; values.len()];
        for axis in 0..GRID_AXES {
            for cell in 0..values.len() {
                blurred[cell] = 0.5 * values[cell]
                    + 0.25 * (values[self.neighbor(cell, axis, -1)] + values[self.neighbor(cell, axis, 1)]);
            }
            std::mem::swap(&mut values, &mut blurred);
        }
        values
    }
}

/// How much a cell's value resembles its neighbor's, averaged over axes. Near 1 is smooth blobs,
/// near 0 is speckle. No variance reads 1.0: perfectly flat is perfectly self-similar.
pub fn axial_correlation(values: &[f32], field: &SpatialField) -> f32 {
    let mean = values.iter().sum::<f32>() / values.len().max(1) as f32;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>();
    if variance < 1e-12 { return 1.0; }
    let covariance: f32 = (0..values.len())
        .flat_map(|i| (0..GRID_AXES).map(move |axis| (values[i] - mean) * (values[field.neighbor(i, axis, 1)] - mean)))
        .sum();
    (covariance / (GRID_AXES as f32 * variance)).clamp(-1.0, 1.0)
}
