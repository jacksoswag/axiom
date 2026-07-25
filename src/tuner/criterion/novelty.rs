//! A genome is good exactly insofar as nothing found so far behaves like it. No target and no
//! gradient toward one, so it dodges the trivial attractors an absolute score falls into, and
//! equally never climbs toward a goal you had in mind.

use crate::util::distance;
use crate::tuner::specimen::Specimen;

/// Neighbors averaged into the score. Fixed, not a per-run setting, so two searches compare
pub const NEIGHBORS: usize = 15;

/// Mean distance to the nearest few behaviors already found. Reads the whole descriptor, never one
/// metric, so it needs no plan. An empty population scores 0: an opening generation survives on
/// capacity and the gate, never on invented novelty.
pub fn score(values: &[f32], population: &[Specimen]) -> f32 {
    if population.is_empty() { return 0.0; }
    let mut distances: Vec<f32> = population.iter()
        .map(|other| distance(values, &other.metrics)).collect();
    let count = NEIGHBORS.min(distances.len());
    distances.select_nth_unstable_by(count - 1, f32::total_cmp);
    distances[..count].iter().sum::<f32>() / count as f32
}
