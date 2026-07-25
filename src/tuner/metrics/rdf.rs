//! Radial distribution: stand on a particle and ask how many others sit at each distance, against
//! how many a uniform cloud would put there. Flat at 1.0 is a gas, a peak means particles prefer
//! that spacing. The trait axis splits it further, so it also answers whether similar particles sit
//! at different spacings than dissimilar ones. Owns the pair histogram, the expensive base block.

use crate::engine::kernel::distance_sq;
use crate::engine::substrate::Substrate;
use rayon::prelude::*;
use super::{alive, Blocks, Metric, Spec};

const TRAIT_BANDS: usize = 3;
const BINS: usize = 6;
const WIDTH: usize = TRAIT_BANDS * BINS;
pub const METRIC: Metric = &Spec { pairs: true, ..Spec::of("rdf", WIDTH, measure) };
/// Ratios past this are indistinguishable for search and would dominate a distance
const MAX_RATIO: f32 = 20.0;

/// Unordered pair counts per trait-distance band and log-radial bin. f64 because a thousand-particle
/// sim contributes half a million pairs and f32 starts dropping counts.
pub struct Histogram { pub bins: [f64; WIDTH] }

/// Particles a histogram walks pairs among. Below this the swarm is walked whole and the answer is
/// exact; above it an evenly strided sample stands in, and the block costs the same at four thousand
/// particles as at four hundred thousand. Every other block here is linear in the swarm and this one
/// is quadratic, so without a ceiling it is the whole cost of a campaign at any interesting size.
///
/// Strided rather than drawn: it needs no rng, it lands on the same particles in the measured cloud
/// and in the uniform one it is divided by, and index order is trait order, since seeding lays traits
/// down in anchor blocks and a stride crosses all of them evenly.
const SAMPLED: usize = 4096;

/// All pairs among the sample, deliberately: the far bins reach the torus half-diagonal, well past any
/// kernel cutoff, so the neighbor grid cannot help here. Measured and baseline take the same stride
/// over the same population, so they contribute the same pair count and their ratio needs no rescale.
///
/// ponytail: a uniform sample, not a stratified one. The tightest bin holds about a six-thousandth of
/// the pairs, so at this sample size it still lands a few hundred and reads to within a few percent.
/// Stratifying by radius would tighten the near bins specifically, and is the upgrade if the first
/// radial bin ever starts deciding a search on its own noise.
pub fn build(substrate: &Substrate) -> Histogram {
    let (low, high) = bin_range(substrate);
    let edges_sq = bin_edges_sq(low, high);
    let count = substrate.traits.len();
    let stride = count.div_ceil(SAMPLED).max(1);
    let taken = count.div_ceil(stride); // particles this histogram actually stands on
    // Blocks of rows rather than one task per row, and summed back in block order, so the totals never
    // depend on how the work landed across cores. A small swarm comes out as a single block and runs
    // where it stands: inside a search the batch already owns every core, and splitting a cheap
    // histogram underneath it would only pay for the privilege.
    const BLOCK: usize = 1024;
    let blocks: Vec<[f64; WIDTH]> = (0..taken.div_ceil(BLOCK).max(1)).into_par_iter().map(|block| {
        let mut bins = [0.0f64; WIDTH];
        for slot in block * BLOCK..((block + 1) * BLOCK).min(taken) {
            let i = slot * stride;
            let a = substrate.pos(i);
            if !alive(a) { continue; }
            for other in slot + 1..taken {
                let j = other * stride;
                let b = substrate.pos(j);
                if !alive(b) { continue; }
                let radial = radial_bin(distance_sq(a, b, substrate, &mut []), &edges_sq);
                bins[trait_band(substrate.traits[i], substrate.traits[j]) * BINS + radial] += 1.0;
            }
        }
        bins
    }).collect();
    let mut out = Histogram { bins: [0.0; WIDTH] };
    for block in blocks { for (total, value) in out.bins.iter_mut().zip(block) { *total += value; } }
    out
}

/// The log-spaced radial bins as squared edges. Squared is the whole point: a pair is classified by
/// comparing its squared distance against these, so the pair loop takes neither the square root nor
/// the two logarithms that reading the bin off a distance costs.
fn bin_edges_sq(low: f32, high: f32) -> [f32; BINS] {
    std::array::from_fn(|bin| (low * (high / low).powf(bin as f32 / BINS as f32)).powi(2))
}
/// Which bin a squared distance falls in: the last edge it clears. Under the first edge is bin 0,
/// which is where anything closer than one mean spacing belongs anyway.
fn radial_bin(distance_sq: f32, edges_sq: &[f32; BINS]) -> usize {
    edges_sq.iter().filter(|&&edge| distance_sq >= edge).count().saturating_sub(1)
}

/// What a plan that never reads a baseline is handed instead of paying for one. Every bin sits below
/// the floor ratios() treats as empty, so anything that somehow asked would read 1.0 across the board:
/// no information, rather than a division by almost nothing.
pub fn blank() -> Histogram { Histogram { bins: [0.0; WIDTH] } }

/// Below one mean particle spacing a bin holds too few pairs to trust; the far edge reaches the
/// torus corner, the furthest two points can be.
fn bin_range(substrate: &Substrate) -> (f32, f32) {
    let dims = substrate.dimensions as f32;
    let spacing = substrate.box_len / (substrate.traits.len().max(1) as f32).powf(1.0 / dims);
    let low = spacing.max(1e-6);
    (low, (substrate.box_len * 0.5 * dims.sqrt()).max(low * 4.0))
}

fn trait_band(a: f32, b: f32) -> usize {
    (((a - b).abs().clamp(0.0, 1.0) * TRAIT_BANDS as f32) as usize).min(TRAIT_BANDS - 1)
}

/// Measured over uniform. A bin the baseline left near empty reports 1.0, meaning no information
/// rather than a division by almost nothing.
pub fn ratios(measured: &Histogram, baseline: &Histogram) -> Vec<f32> {
    measured.bins.iter().zip(baseline.bins)
        .map(|(&value, reference)| if reference < 0.5 { 1.0 } else { (value / reference) as f32 }.clamp(0.0, MAX_RATIO))
        .collect()
}

/// Multiplicative ratios onto a symmetric log axis: 1 sits at the midpoint, MAX_RATIO at the top,
/// its reciprocal at the bottom, so twice-as-dense and half-as-dense land equally far out.
fn scale(ratio: f32) -> f32 {
    if ratio <= 0.0 { 0.0 } else { ((1.0 + ratio.ln() / MAX_RATIO.ln()) * 0.5).clamp(0.0, 1.0) }
}

pub fn measure(base: &Blocks, _: &[f32]) -> Vec<f32> {
    ratios(base.pairs(), base.rollout.baseline).into_iter().map(scale).collect()
}
