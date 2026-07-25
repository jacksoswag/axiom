//! Radial distribution: stand on a particle and ask how many others sit at each distance, against
//! how many a uniform cloud would put there. Flat at 1.0 is a gas, a peak means particles prefer
//! that spacing. The trait axis splits it further, so it also answers whether similar particles sit
//! at different spacings than dissimilar ones. Owns the pair histogram, the expensive base block.

use crate::engine::kernel::distance_sq;
use crate::engine::substrate::Substrate;
use super::{alive, log_bin, Blocks, Metric, Spec};

const TRAIT_BANDS: usize = 3;
const BINS: usize = 6;
const WIDTH: usize = TRAIT_BANDS * BINS;
pub const SPEC: Spec = Spec { pairs: true, ..Spec::of("rdf", WIDTH, measure) };
pub const METRIC: Metric = &SPEC;
/// Ratios past this are indistinguishable for search and would dominate a distance
const MAX_RATIO: f32 = 20.0;

/// Unordered pair counts per trait-distance band and log-radial bin. f64 because a thousand-particle
/// sim contributes half a million pairs and f32 starts dropping counts.
pub struct Histogram { pub bins: [f64; WIDTH] }

/// All pairs, deliberately: the far bins reach the torus half-diagonal, well past any kernel cutoff,
/// so the neighbor grid cannot help here.
pub fn build(substrate: &Substrate) -> Histogram {
    let (low, high) = bin_range(substrate);
    let mut out = Histogram { bins: [0.0; WIDTH] };
    for i in 0..substrate.traits.len() {
        let a = substrate.pos(i);
        if !alive(a) { continue; }
        for j in i + 1..substrate.traits.len() {
            let b = substrate.pos(j);
            if !alive(b) { continue; }
            let radial = log_bin(distance_sq(a, b, substrate, &mut []).sqrt(), low, high, BINS);
            out.bins[trait_band(substrate.traits[i], substrate.traits[j]) * BINS + radial] += 1.0;
        }
    }
    out
}

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
