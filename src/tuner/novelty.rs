//! Fixed-space descriptor distances used by discovery search.

use crate::tuner::metrics::{descriptor_len, HETEROGENEITY_SIDES, HETEROGENEITY_VALUES, LAGS, RDF_BINS, TRAIT_BANDS};
use crate::tuner::persistence::BARS;
use crate::util::Fnv;

/// Neighbors averaged by novelty. Part of the search contract, never a per-run knob, so archived
/// runs stay comparable.
pub const NEIGHBORS: usize = 15;

/// Stable coarse cell used for diversity quotas and stall diagnostics: one quantized mean per
/// semantic descriptor block, folded into one hash.
pub fn neighborhood_key(descriptor: &[f32]) -> u64 {
    let rdf_end = TRAIT_BANDS * RDF_BINS;
    let heterogeneity_end = rdf_end + HETEROGENEITY_SIDES.len() * HETEROGENEITY_VALUES;
    let barcode_end = heterogeneity_end + BARS;
    let connectivity_end = barcode_end + 4;
    let temporal_end = connectivity_end + 1 + LAGS.len();
    let ranges = [0..rdf_end, rdf_end..heterogeneity_end, heterogeneity_end..barcode_end,
        barcode_end..barcode_end + 2, barcode_end + 2..connectivity_end,
        connectivity_end..temporal_end, temporal_end..temporal_end + 2, temporal_end + 2..descriptor_len()];
    let mut hash = Fnv::new();
    for range in ranges { // one quantized block mean -> one hashed word
        let width = range.len().max(1) as f32;
        let mean = descriptor[range].iter().sum::<f32>() / width;
        hash.word((mean.clamp(0.0, 2.0) * 4.0).round() as u64);
    }
    hash.finish()
}

/// Euclidean distance in the versioned descriptor space.
pub fn distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().enumerate().map(|(k, x)| (x - b[k]).powi(2)).sum::<f32>().sqrt()
}

/// Mean distance to the nearest k neighbors. An empty reference set is deliberately zero:
/// bootstrap admission is controlled by capacity and the viability gate, never an invented novelty.
pub fn novelty(descriptor: &[f32], population: &[Vec<f32>], k: usize) -> f32 {
    if population.is_empty() || k == 0 { return 0.0; }
    let mut distances: Vec<f32> = population.iter().map(|other| distance(descriptor, other)).collect();
    let count = k.min(distances.len());
    distances.select_nth_unstable_by(count - 1, f32::total_cmp);
    distances[..count].iter().sum::<f32>() / count as f32
}

/// Distance to the closest other point, a deliberately local crowding measure for capacity
/// pruning: two entries can be similarly novel while one is a near duplicate.
pub fn crowding(descriptor: &[f32], population: &[Vec<f32>]) -> f32 {
    population.iter().map(|other| distance(descriptor, other)).min_by(|a, b| a.total_cmp(b)).unwrap_or(f32::INFINITY)
}
