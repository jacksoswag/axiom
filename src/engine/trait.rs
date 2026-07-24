//! Particle traits are assigned from genome logits, and their piecewise-linear anchor basis.
//! Anchors sit evenly spaced on a circle, so a trait belongs to at most two adjacent anchors
//! and wraps at the 0/1 seam instead of clamping.

use crate::util::Rng;
use crate::engine::substrate::Substrate;
use crate::util::finite;

/// Assigns each particle a trait so the population's anchor distribution matches logits.
pub fn init_particle_traits(substrate: &mut Substrate, logits: &[f32], seed: u64) {
    let weights: Vec<f32> = logits.iter() // softmax logits to per-bin target shares
        .map(|&logit| finite(logit, 0.0).exp() + 1e-3).collect();
    let total: f32 = weights.iter().sum();
    let exact: Vec<f32> = weights
        .iter().map(|weight| weight / total * substrate.traits.len() as f32).collect();
    // Floor each share, then hand out leftover particles to bins with biggest fractional remainder
    let mut counts: Vec<usize> = exact.iter().map(|value| value.floor() as usize).collect();
    let mut order: Vec<usize> = (0..counts.len()).collect();
    order.sort_by(|&left, &right| {
        (exact[right] - exact[right].floor())
            .total_cmp(&(exact[left] - exact[left].floor()))
            .then(left.cmp(&right))
    });
    for &bin in order.iter().cycle()
        .take(substrate.traits.len() - counts.iter().sum::<usize>())
    { counts[bin] += 1; }

    // Fill each bin's quota with a trait drawn uniformly from the bin's slice of [0, 1).
    let mut rng = Rng::new(seed ^ 0xD1B5_4A32_D192_ED03);
    let width = 1.0 / logits.len() as f32;
    let mut index = 0usize;
    for (bin, count) in counts.into_iter().enumerate() {
        let low = bin as f32 * width;
        for _ in 0..count {
            substrate.traits[index] = (low + width * rng.unit()).min(1.0);
            index += 1;
        }
    }
}

/// Active memberships in a fixed anchor basis.
#[derive(Clone, Copy)] // Copy: weight(self) takes it by value
pub struct Membership { pub entries: [(usize, f32); 2] }

impl Membership {
    /// Particle's membership weight for an anchor, or 0.0 if the anchor isn't the 2 active entries.
    pub fn weight(self, anchor: usize) -> f32 {
        self.entries.iter().find_map(|&(index, weight)| (index == anchor).then_some(weight)).unwrap_or(0.0)
    }
}

/// Piecewise-linear hat memberships over anchors evenly spaced on a circle.
pub fn membership(trait_value: f32, anchor_count: usize) -> Membership {
    let scaled = trait_value.clamp(0.0, 1.0) * anchor_count as f32;
    let lower = (scaled.floor() as usize) % anchor_count;
    let upper = (lower + 1) % anchor_count;
    let upper_weight = scaled - scaled.floor();
    Membership { entries: [(lower, 1.0 - upper_weight), (upper, upper_weight)] }
}

/// Every particle's membership, in trait order. Shared by callers that walk the whole population.
pub fn memberships(traits: &[f32], anchor_count: usize) -> Vec<Membership> {
    traits.iter().map(|&trait_value| membership(trait_value, anchor_count)).collect()
}
