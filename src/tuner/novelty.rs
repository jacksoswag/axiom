//! Fixed-space descriptor distances used by discovery search.

use crate::tuner::metrics::{
    descriptor_len, HETEROGENEITY_SIDES, HETEROGENEITY_VALUES, LAGS, RDF_BINS, TRAIT_BANDS,
};
use crate::tuner::persistence::BARS;
use crate::util::Fnv;

/// Neighbours averaged by novelty.  The value is part of the search contract, rather than a
/// per-run tuning knob, so archived runs remain comparable.
pub const NEIGHBOURS: usize = 15;

/// Stable coarse cell used for diversity quotas and stall diagnostics.
pub fn neighborhood_key(descriptor: &[f32]) -> u64 {
    assert_eq!(descriptor.len(), descriptor_len());
    let rdf_end = TRAIT_BANDS * RDF_BINS;
    let heterogeneity_end = rdf_end + HETEROGENEITY_SIDES.len() * HETEROGENEITY_VALUES;
    let barcode_end = heterogeneity_end + BARS;
    let connectivity_end = barcode_end + 4;
    let temporal_end = connectivity_end + 1 + LAGS.len();
    let ranges = [
        0..rdf_end,
        rdf_end..heterogeneity_end,
        heterogeneity_end..barcode_end,
        barcode_end..barcode_end + 2,
        barcode_end + 2..connectivity_end,
        connectivity_end..temporal_end,
        temporal_end..temporal_end + 2,
        temporal_end + 2..descriptor_len(),
    ];
    let mut hash = Fnv::new();
    for range in ranges {
        let width = range.len().max(1) as f32;
        let mean = descriptor[range].iter().sum::<f32>() / width;
        hash.word((mean.clamp(0.0, 2.0) * 4.0).round() as u64);
    }
    hash.finish()
}

/// Euclidean distance in the versioned descriptor space.
pub fn distance(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "descriptor versions do not match");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Mean distance to the nearest `k` neighbours.  An empty reference set is intentionally
/// zero: bootstrap admission is controlled by capacity and the viability gate, not an
/// invented novelty value.
pub fn novelty(descriptor: &[f32], population: &[Vec<f32>], k: usize) -> f32 {
    if population.is_empty() || k == 0 {
        return 0.0;
    }
    let mut distances: Vec<f32> = population
        .iter()
        .map(|other| distance(descriptor, other))
        .collect();
    let count = k.min(distances.len());
    distances.select_nth_unstable_by(count - 1, f32::total_cmp);
    distances[..count].iter().sum::<f32>() / count as f32
}

/// Distance to the closest other point.  This is a deliberately local crowding measure for
/// capacity pruning: two entries can be similarly novel while one is a near duplicate.
pub fn crowding(descriptor: &[f32], population: &[Vec<f32>]) -> f32 {
    population
        .iter()
        .map(|other| distance(descriptor, other))
        .min_by(|a, b| a.total_cmp(b))
        .unwrap_or(f32::INFINITY)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn averages_exactly_the_nearest_neighbours() {
        let population = vec![vec![1.0], vec![2.0], vec![3.0], vec![100.0]];
        assert!((novelty(&[0.0], &population, 3) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn crowding_exposes_near_duplicates() {
        let population = vec![vec![0.1], vec![4.0]];
        assert!((crowding(&[0.0], &population) - 0.1).abs() < 1e-6);
    }

    #[test]
    fn neighborhood_spans_every_semantic_block() {
        let baseline = vec![0.0; descriptor_len()];
        let baseline_key = neighborhood_key(&baseline);
        let representatives = [0, 18, 33, 41, 43, 45, 49, 51];
        for start in representatives {
            let mut changed = baseline.clone();
            let end = representatives
                .into_iter()
                .find(|&candidate| candidate > start)
                .unwrap_or(descriptor_len());
            changed[start..end].fill(2.0);
            assert_ne!(baseline_key, neighborhood_key(&changed), "block at {start}");
        }
    }
}
