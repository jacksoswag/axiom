//! Do different regions look different, and at what scale. Five readings per requested grid: how
//! uneven the density is, how much of the box is empty, whether regions hold different species
//! mixes, whether a region is mixed or pure, and whether density varies smoothly or as speckle.
//! Two of the five subtract what a random cloud shows, or a fine grid calls every swarm structured.

use super::field::{axial_correlation, SpatialField};
use super::{Blocks, Metric, Spec};

/// Coarse to fine. Sub-structure lives at one scale, so all three stay separate rather than averaged
pub const SIDES: [usize; 3] = [4, 8, 16];
pub const VALUES: usize = 5;
const WIDTH: usize = SIDES.len() * VALUES;
pub const SPEC: Spec = Spec { sides: &SIDES, ..Spec::of("heterogeneity", WIDTH, measure) };
pub const METRIC: Metric = &SPEC;

pub fn measure(base: &Blocks, _: &[f32]) -> Vec<f32> {
    SIDES.into_iter().flat_map(|side| of(base.field(side))).enumerate().map(scale).collect()
}

fn of(field: &SpatialField) -> [f32; VALUES] {
    let cells = field.density.len() as f32;
    let mean = field.density.iter().sum::<f32>() / cells;
    let raw_variance = field.density.iter().map(|value| (value - mean).powi(2)).sum::<f32>() / cells;
    // Remove the Poisson sampling variance: a uniform finite swarm should read near zero instead
    // of looking heterogeneous merely because the grid is fine.
    let density_variance = ((raw_variance - mean) / mean.max(1e-6).powi(2)).max(0.0);
    let void = field.density.iter().filter(|v| **v == 0.0).count() as f32 / cells;
    let particles = field.density.iter().sum::<f32>();
    let occupied = field.density.iter().filter(|&&n| n > 0.0).count();
    let trait_mean = field.trait_sum.iter().sum::<f32>() / particles.max(1.0);
    let between_variance = field.density.iter().enumerate().filter(|(_, n)| **n > 0.0)
        .map(|(i, &n)| n * (field.local_mean_trait(i) - trait_mean).powi(2))
        .sum::<f32>() / particles.max(1.0);
    let population_variance =
        (field.trait_square_sum.iter().sum::<f32>() / particles.max(1.0) - trait_mean * trait_mean).max(0.0);
    // Under random mixing, grouping N traits into C occupied cells contributes exactly this
    // finite-population between-cell variance in expectation. Subtracting it removes the
    // single-particle-cell false biomes seen on fine grids.
    let shot_noise = if particles > 1.0 {
        population_variance * occupied.saturating_sub(1) as f32 / (particles - 1.0)
    } else { 0.0 };
    // As a share of the trait variance there is to explain, not a raw variance. Between-cell
    // variance can never exceed the population's own, so this is natively [0, 1] and needs no
    // magic multiplier: a raw reading here runs about 0.004 and read as ~0.02 on the axis, which
    // pinned every downstream product that multiplies by it.
    let trait_variance = ((between_variance - shot_noise).max(0.0) / population_variance.max(1e-6)).min(1.0);
    let entropy = field.density.iter().enumerate().filter(|(_, n)| **n > 0.0)
        .map(|(i, &total)| {
            let nonempty = field.trait_bins[i].iter().filter(|&&n| n > 0).count();
            let plug_in = -field.trait_bins[i].iter().filter(|&&n| n > 0)
                .map(|&n| { let p = n as f32 / total; p * p.ln() }).sum::<f32>();
            let corrected = plug_in + nonempty.saturating_sub(1) as f32 / (2.0 * total); // small-sample correction
            total * (corrected / 4.0f32.ln()).min(1.0)
        })
        .sum::<f32>() / particles.max(1.0);
    [density_variance, void, trait_variance, entropy, axial_correlation(&field.density, field)]
}

/// Each reading has its own natural range, so each gets its own map onto the shared axis. Position
/// in the group, not grid resolution, decides which map applies.
fn scale((slot, value): (usize, f32)) -> f32 {
    match slot % VALUES {
        0 => (value / 4.0).clamp(0.0, 1.0), // density CV^2
        1 | 2 | 3 => value.clamp(0.0, 1.0), // void, trait share and entropy are already [0,1]
        _ => ((value + 1.0) * 0.5).clamp(0.0, 1.0), // correlation from [-1,1]
    }
}
