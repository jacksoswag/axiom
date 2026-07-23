//! Fixed, three-dimensional behaviour measurements used by novelty and qualification.
//!
//! Descriptor version 5 is deliberately a world descriptor: it contains no projection or
//! renderer-only statistic.  Its layout is fixed at 53 values so an archive never changes
//! coordinate meaning while it is running.

use crate::engine::kernel::displacement;
use crate::engine::interaction::Net;
use crate::tuner::persistence::BARS;
use crate::engine::substrate::Substrate;

/// Descriptor space is 3-D-specific (winding axes, side-4/8/16 heterogeneity). Generalizing it
/// is deferred training work, so measurement stays pinned to three dimensions for now.
const DIMENSIONS: usize = 3;

pub const TRAIT_BANDS: usize = 3;
pub const RDF_BINS: usize = 6;
pub const HETEROGENEITY_SIDES: [usize; 3] = [4, 8, 16];
pub const HETEROGENEITY_VALUES: usize = 5;
pub const DENSITY_THRESHOLDS: [f32; 2] = [0.5, 1.5];
pub const LAGS: [usize; 3] = [1, 2, 4];
pub const DESCRIPTOR_VERSION: u32 = 5;
pub const MOBILITY_CEILING: f32 = 0.1;
const AXIS_SPAN: f32 = 2.0;
const MAX_RATIO: f32 = 20.0;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Connectivity {
    /// Fraction held by the largest component, or zero unless the phase has enough volume
    /// and a non-contractible loop around the torus.
    pub dense: [f32; 2],
    /// The corresponding winding-component fraction for thresholded void cells.
    pub void: [f32; 2],
}

#[derive(Clone, Debug, Default)]
pub struct Metrics {
    pub mobility: f32,
    pub temporal_variance: f32,
    pub autocorrelation: [f32; LAGS.len()],
    pub turnover: f32,
    pub robustness: f32,
    /// Mean departure of the trait-conditioned RDF from its uniform baseline.
    pub structure: f32,
    /// Named qualification evidence for promotion tiers.
    pub heterogeneity: [f32; HETEROGENEITY_SIDES.len() * HETEROGENEITY_VALUES],
    pub connectivity: Connectivity,
    pub descriptor: Vec<f32>,
    pub alive: bool,
}

/// Descriptor v5 layout:
///
/// ```text
/// [ 0..18) trait-distance (3) × radial (6) RDF ratios
/// [18..33) periodic 3-D heterogeneity at sides 4, 8, 16 (five each)
/// [33..41) seven H0 death-scale masses plus cutoff-separated component mass
/// [41..45) dense then void winding-component fractions at two thresholds
/// [45..53) temporal variance, 3 lags, mobility, turnover, asymmetry, repair
/// ```
pub const fn descriptor_len() -> usize {
    TRAIT_BANDS * RDF_BINS
        + HETEROGENEITY_SIDES.len() * HETEROGENEITY_VALUES
        + BARS
        + 4
        + 1
        + LAGS.len()
        + 1
        + 1
        + 1
        + 1
}

#[derive(Clone, Debug)]
pub struct Rdf {
    pub bins: [f64; TRAIT_BANDS * RDF_BINS],
}

impl Rdf {
    pub fn zeros() -> Self {
        Self {
            bins: [0.0; TRAIT_BANDS * RDF_BINS],
        }
    }
    pub fn add(&mut self, other: &Self) {
        for (a, b) in self.bins.iter_mut().zip(other.bins) {
            *a += b;
        }
    }
    pub fn scale(&mut self, factor: f64) {
        self.bins.iter_mut().for_each(|v| *v *= factor);
    }
}

fn bin_range(substrate: &Substrate, extent: f32) -> (f32, f32) {
    let spacing = extent / (substrate.len().max(1) as f32).powf(1.0 / DIMENSIONS as f32);
    let low = spacing.max(1e-6);
    (low, (extent * 0.5 * 3.0f32.sqrt()).max(low * 4.0))
}

fn radial_bin(distance: f32, low: f32, high: f32) -> usize {
    if !distance.is_finite() || distance <= low {
        return 0;
    }
    (((distance / low).ln() / (high / low).ln() * RDF_BINS as f32) as usize).min(RDF_BINS - 1)
}

fn trait_band(a: f32, b: f32) -> usize {
    (((a - b).abs().clamp(0.0, 1.0) * TRAIT_BANDS as f32) as usize).min(TRAIT_BANDS - 1)
}

/// Counts every unordered pair into a trait-distance and log-radial band.
pub fn raw_rdf(substrate: &Substrate, extent: f32, softening: f32) -> Rdf {
    let (low, high) = bin_range(substrate, extent);
    let mut out = Rdf::zeros();
    for i in 0..substrate.len() {
        let a = &substrate.positions[i * DIMENSIONS..(i + 1) * DIMENSIONS];
        if !a.iter().all(|v| v.is_finite()) {
            continue;
        }
        for j in i + 1..substrate.len() {
            let b = &substrate.positions[j * DIMENSIONS..(j + 1) * DIMENSIONS];
            if !b.iter().all(|v| v.is_finite()) {
                continue;
            }
            let radial = radial_bin(
                displacement(a, b, extent, 1.0 / extent, softening * softening),
                low,
                high,
            );
            let index = trait_band(substrate.traits[i], substrate.traits[j]) * RDF_BINS + radial;
            out.bins[index] += 1.0;
        }
    }
    out
}

pub fn normalized_rdf(raw: &Rdf, baseline: &Rdf) -> Vec<f32> {
    raw.bins
        .iter()
        .zip(baseline.bins)
        .map(|(&value, reference)| {
            if reference < 0.5 {
                1.0
            } else {
                (value / reference) as f32
            }
            .clamp(0.0, MAX_RATIO)
        })
        .collect()
}

/// Multiplicative ratios use a symmetric log scale: `1` remains the uniform midpoint,
/// `MAX_RATIO` maps to the upper edge, and its reciprocal maps to the lower edge.
fn scale_ratio(ratio: f32) -> f32 {
    if ratio <= 0.0 {
        0.0
    } else {
        (1.0 + ratio.ln() / MAX_RATIO.ln()).clamp(0.0, AXIS_SPAN)
    }
}

pub fn structure(ratios: &[f32]) -> f32 {
    if ratios.is_empty() {
        0.0
    } else {
        ratios.iter().map(|v| (v - 1.0).abs()).sum::<f32>() / ratios.len() as f32
    }
}

/// A periodic density/trait field at a fixed descriptor resolution.
#[derive(Clone, Debug)]
pub struct SpatialField {
    pub side: usize,
    pub density: Vec<f32>,
    trait_sum: Vec<f32>,
    trait_square_sum: Vec<f32>,
    trait_bins: Vec<[u32; 4]>,
}

pub fn spatial_field(substrate: &Substrate, extent: f32, side: usize) -> SpatialField {
    let cells = side.pow(3);
    let mut field = SpatialField {
        side,
        density: vec![0.0; cells],
        trait_sum: vec![0.0; cells],
        trait_square_sum: vec![0.0; cells],
        trait_bins: vec![[0; 4]; cells],
    };
    for (i, pos) in substrate.positions.chunks_exact(DIMENSIONS).enumerate() {
        if !pos.iter().all(|v| v.is_finite()) {
            continue;
        }
        let xyz = [
            ((pos[0].rem_euclid(extent) / extent * side as f32) as usize).min(side - 1),
            ((pos[1].rem_euclid(extent) / extent * side as f32) as usize).min(side - 1),
            ((pos[2].rem_euclid(extent) / extent * side as f32) as usize).min(side - 1),
        ];
        let index = field.index(xyz[0], xyz[1], xyz[2]);
        let trait_value = substrate.traits[i].clamp(0.0, 1.0);
        field.density[index] += 1.0;
        field.trait_sum[index] += trait_value;
        field.trait_square_sum[index] += trait_value * trait_value;
        field.trait_bins[index][((trait_value * 4.0) as usize).min(3)] += 1;
    }
    field
}

impl SpatialField {
    fn index(&self, x: usize, y: usize, z: usize) -> usize {
        (z * self.side + y) * self.side + x
    }
    fn neighbour(&self, index: usize, axis: usize, delta: isize) -> usize {
        let side = self.side as isize;
        let x = (index % self.side) as isize;
        let y = ((index / self.side) % self.side) as isize;
        let z = (index / (self.side * self.side)) as isize;
        let mut xyz = [x, y, z];
        xyz[axis] = (xyz[axis] + delta).rem_euclid(side);
        self.index(xyz[0] as usize, xyz[1] as usize, xyz[2] as usize)
    }
    pub fn local_mean_trait(&self, index: usize) -> f32 {
        self.trait_sum[index] / self.density[index].max(1.0)
    }
    pub fn heterogeneity(&self) -> [f32; HETEROGENEITY_VALUES] {
        let cells = self.density.len() as f32;
        let mean = self.density.iter().sum::<f32>() / cells;
        let raw_variance = self
            .density
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f32>()
            / cells;
        // Remove the Poisson sampling variance. A uniform finite swarm should read near zero
        // instead of looking more heterogeneous merely because the grid is fine.
        let density_variance = ((raw_variance - mean) / mean.max(1e-6).powi(2)).max(0.0);
        let void = self.density.iter().filter(|v| **v == 0.0).count() as f32 / cells;
        let particles = self.density.iter().sum::<f32>();
        let occupied = self.density.iter().filter(|&&n| n > 0.0).count();
        let trait_total = self.trait_sum.iter().sum::<f32>();
        let trait_mean = trait_total / particles.max(1.0);
        let between_variance = self
            .density
            .iter()
            .enumerate()
            .filter(|(_, n)| **n > 0.0)
            .map(|(i, &n)| n * (self.local_mean_trait(i) - trait_mean).powi(2))
            .sum::<f32>()
            / particles.max(1.0);
        let trait_square_total = self.trait_square_sum.iter().sum::<f32>();
        let population_variance =
            (trait_square_total / particles.max(1.0) - trait_mean * trait_mean).max(0.0);
        // Under random mixing, grouping N traits into C occupied cells contributes exactly
        // this finite-population between-cell variance in expectation. Subtracting it removes
        // the single-particle-cell false biomes seen on fine grids.
        let shot_noise = if particles > 1.0 {
            population_variance * (occupied.saturating_sub(1)) as f32 / (particles - 1.0)
        } else {
            0.0
        };
        let trait_variance = (between_variance - shot_noise).max(0.0);
        let entropy = self
            .density
            .iter()
            .enumerate()
            .filter(|(_, n)| **n > 0.0)
            .map(|(i, &total)| {
                let nonempty = self.trait_bins[i].iter().filter(|&&n| n > 0).count();
                let plug_in = -self.trait_bins[i]
                    .iter()
                    .filter(|&&n| n > 0)
                    .map(|&n| {
                        let p = n as f32 / total;
                        p * p.ln()
                    })
                    .sum::<f32>();
                let corrected = plug_in + nonempty.saturating_sub(1) as f32 / (2.0 * total);
                total * (corrected / 4.0f32.ln()).min(1.0)
            })
            .sum::<f32>()
            / particles.max(1.0);
        [
            density_variance,
            void,
            trait_variance,
            entropy,
            axial_correlation(&self.density, self),
        ]
    }
    pub fn connectivity(&self) -> Connectivity {
        let density = smoothed_density(self);
        let mean = density.iter().sum::<f32>() / density.len().max(1) as f32;
        let mut dense = [0.0; 2];
        let mut void = [0.0; 2];
        for (slot, threshold) in DENSITY_THRESHOLDS.into_iter().enumerate() {
            dense[slot] = winding_component_fraction(self, &density, |v| v >= threshold * mean);
            void[slot] = winding_component_fraction(self, &density, |v| v < threshold * mean);
        }
        Connectivity { dense, void }
    }
}

fn axial_correlation(values: &[f32], field: &SpatialField) -> f32 {
    let mean = values.iter().sum::<f32>() / values.len().max(1) as f32;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>();
    if variance < 1e-12 {
        return 1.0;
    }
    let covariance: f32 = (0..values.len())
        .flat_map(|i| {
            (0..DIMENSIONS)
                .map(move |axis| (values[i] - mean) * (values[field.neighbour(i, axis, 1)] - mean))
        })
        .sum();
    (covariance / (DIMENSIONS as f32 * variance)).clamp(-1.0, 1.0)
}

fn smoothed_density(field: &SpatialField) -> Vec<f32> {
    let weights = [0.25, 0.5, 0.25];
    (0..field.density.len())
        .map(|cell| {
            let mut value = 0.0;
            for (xi, dx) in [-1, 0, 1].into_iter().enumerate() {
                for (yi, dy) in [-1, 0, 1].into_iter().enumerate() {
                    for (zi, dz) in [-1, 0, 1].into_iter().enumerate() {
                        let next = field.neighbour(
                            field.neighbour(field.neighbour(cell, 0, dx), 1, dy),
                            2,
                            dz,
                        );
                        value += weights[xi] * weights[yi] * weights[zi] * field.density[next];
                    }
                }
            }
            value
        })
        .collect()
}

const MIN_PHASE_VOLUME: f32 = 0.08;

fn winding_component_fraction(
    field: &SpatialField,
    density: &[f32],
    include: impl Fn(f32) -> bool,
) -> f32 {
    let included = density.iter().map(|&v| include(v)).collect::<Vec<_>>();
    let total = included.iter().filter(|v| **v).count();
    if total == 0 || total as f32 / (included.len() as f32) < MIN_PHASE_VOLUME {
        return 0.0;
    }
    let mut lifts = vec![None; included.len()];
    let mut largest_winding = 0;
    for root in 0..included.len() {
        if !included[root] || lifts[root].is_some() {
            continue;
        }
        let mut stack = vec![root];
        lifts[root] = Some([0i32; DIMENSIONS]);
        let mut size = 0;
        let mut winds = [false; DIMENSIONS];
        while let Some(cell) = stack.pop() {
            size += 1;
            for axis in 0..DIMENSIONS {
                for direction in [-1, 1] {
                    let next = field.neighbour(cell, axis, direction);
                    if !included[next] {
                        continue;
                    }
                    let mut proposed = lifts[cell].expect("visited cell has a lift");
                    proposed[axis] += direction as i32;
                    if let Some(existing) = lifts[next] {
                        for winding_axis in 0..DIMENSIONS {
                            winds[winding_axis] |= proposed[winding_axis] != existing[winding_axis];
                        }
                    } else {
                        lifts[next] = Some(proposed);
                        stack.push(next);
                    }
                }
            }
        }
        if winds.into_iter().any(|wrapped| wrapped) {
            largest_winding = largest_winding.max(size);
        }
    }
    largest_winding as f32 / total as f32
}

pub fn heterogeneity(
    substrate: &Substrate,
    extent: f32,
) -> [f32; HETEROGENEITY_SIDES.len() * HETEROGENEITY_VALUES] {
    let mut out = [0.0; HETEROGENEITY_SIDES.len() * HETEROGENEITY_VALUES];
    for (scale, side) in HETEROGENEITY_SIDES.into_iter().enumerate() {
        out[scale * HETEROGENEITY_VALUES..][..HETEROGENEITY_VALUES]
            .copy_from_slice(&spatial_field(substrate, extent, side).heterogeneity());
    }
    out
}

/// Topology uses an eight-cell side and a periodic compact density deposit. That is coarse
/// enough for 320-particle discovery worlds to represent corridors rather than isolated
/// point deposits, while still resolving distinct bodies in 1,200-particle worlds.
pub fn connectivity(substrate: &Substrate, extent: f32) -> Connectivity {
    spatial_field(substrate, extent, 8).connectivity()
}

pub fn temporal(samples: &[Vec<f32>]) -> (f32, [f32; LAGS.len()]) {
    let mut lags = [1.0; LAGS.len()];
    if samples.len() < 2 || samples[0].is_empty() {
        return (0.0, lags);
    }
    let width = samples[0].len();
    let variance = (0..width)
        .map(|i| {
            let mean = samples.iter().map(|s| s[i]).sum::<f32>() / samples.len() as f32;
            samples.iter().map(|s| (s[i] - mean).powi(2)).sum::<f32>() / samples.len() as f32
        })
        .sum::<f32>()
        / width as f32;
    for (slot, lag) in lags.iter_mut().zip(LAGS) {
        let spans = samples.len().saturating_sub(lag);
        if spans > 0 {
            *slot = (0..spans)
                .map(|i| correlation(&samples[i], &samples[i + lag]))
                .sum::<f32>()
                / spans as f32;
        }
    }
    (variance, lags)
}

/// The spatial blocks in descriptor units. Keeping rollout samples and repair comparisons in
/// this fixed `[0, 2]` space prevents a high-variance raw statistic from dominating either.
pub fn scaled_spatial_features(
    rdf: &Rdf,
    rdf_baseline: &Rdf,
    heterogeneity: [f32; HETEROGENEITY_SIDES.len() * HETEROGENEITY_VALUES],
    connectivity: Connectivity,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(TRAIT_BANDS * RDF_BINS + heterogeneity.len() + 4);
    out.extend(
        normalized_rdf(rdf, rdf_baseline)
            .into_iter()
            .map(scale_ratio),
    );
    out.extend(scale_heterogeneity(heterogeneity));
    out.extend(connectivity.dense.map(|v| v * AXIS_SPAN));
    out.extend(connectivity.void.map(|v| v * AXIS_SPAN));
    out
}

fn correlation(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 1.0;
    }
    let (ma, mb) = (
        a[..n].iter().sum::<f32>() / n as f32,
        b[..n].iter().sum::<f32>() / n as f32,
    );
    let (mut cov, mut va, mut vb) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let (x, y) = (a[i] - ma, b[i] - mb);
        cov += x * y;
        va += x * x;
        vb += y * y;
    }
    if va < 1e-12 || vb < 1e-12 {
        1.0
    } else {
        (cov / (va * vb).sqrt()).clamp(-1.0, 1.0)
    }
}

pub fn mobility(before: &[f32], after: &[f32], extent: f32) -> f32 {
    if before.is_empty() || before.len() != after.len() {
        return 0.0;
    }
    (0..before.len() / DIMENSIONS)
        .map(|i| {
            displacement(
                &after[i * DIMENSIONS..(i + 1) * DIMENSIONS],
                &before[i * DIMENSIONS..(i + 1) * DIMENSIONS],
                extent,
                1.0 / extent,
                0.0,
            )
        })
        .sum::<f32>()
        / (before.len() / DIMENSIONS) as f32
        / extent
}

pub fn turnover(before: &SpatialField, after: &SpatialField) -> f32 {
    before
        .density
        .iter()
        .zip(&after.density)
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / (before.density.iter().sum::<f32>() + after.density.iter().sum::<f32>()).max(1.0)
}

pub fn asymmetry(net: &Net) -> f32 {
    let anchors = net.anchors();
    let mut difference = 0.0f32;
    let mut total = 0.0f32;
    for source in 0..anchors {
        for destination in 0..anchors {
            let a = net.pair(source, destination).weight;
            let b = net.pair(destination, source).weight;
            difference += (a - b).powi(2);
            total += a.powi(2);
        }
    }
    if total < 1e-12 {
        0.0
    } else {
        (difference / total).sqrt().min(AXIS_SPAN)
    }
}

pub struct Observations<'a> {
    pub rdf: &'a Rdf,
    pub rdf_baseline: &'a Rdf,
    pub spatial_samples: &'a [Vec<f32>],
    pub heterogeneity: [f32; HETEROGENEITY_SIDES.len() * HETEROGENEITY_VALUES],
    pub connectivity: Connectivity,
    pub barcode: &'a crate::tuner::persistence::Barcode,
    pub mobility: f32,
    pub turnover: f32,
    pub asymmetry: f32,
    pub robustness: f32,
}

fn scale_heterogeneity(
    values: [f32; HETEROGENEITY_SIDES.len() * HETEROGENEITY_VALUES],
) -> impl Iterator<Item = f32> {
    values
        .into_iter()
        .enumerate()
        .map(|(i, v)| match i % HETEROGENEITY_VALUES {
            0 => (v / 4.0 * AXIS_SPAN).clamp(0.0, AXIS_SPAN), // density CV²
            1 | 3 => (v * AXIS_SPAN).clamp(0.0, AXIS_SPAN),
            2 => (v * 8.0).clamp(0.0, AXIS_SPAN), // trait variance <= .25
            _ => (v + 1.0).clamp(0.0, AXIS_SPAN),
        })
}

pub fn descriptor(observed: &Observations) -> Vec<f32> {
    let mut out = Vec::with_capacity(descriptor_len());
    let spatial = scaled_spatial_features(
        observed.rdf,
        observed.rdf_baseline,
        observed.heterogeneity,
        observed.connectivity,
    );
    out.extend(&spatial[..TRAIT_BANDS * RDF_BINS + observed.heterogeneity.len()]);
    out.extend(
        crate::tuner::persistence::mass(observed.barcode)
            .into_iter()
            .map(|value| value.clamp(0.0, AXIS_SPAN)),
    );
    out.extend(&spatial[TRAIT_BANDS * RDF_BINS + observed.heterogeneity.len()..]);
    let (variance, lags) = temporal(observed.spatial_samples);
    out.push(variance.clamp(0.0, AXIS_SPAN));
    out.extend(lags.map(|value| (value + 1.0).clamp(0.0, AXIS_SPAN)));
    out.push((observed.mobility / MOBILITY_CEILING * AXIS_SPAN).clamp(0.0, AXIS_SPAN));
    out.push((observed.turnover * AXIS_SPAN).clamp(0.0, AXIS_SPAN));
    out.push(observed.asymmetry.clamp(0.0, AXIS_SPAN));
    out.push((observed.robustness * AXIS_SPAN).clamp(0.0, AXIS_SPAN));
    debug_assert_eq!(out.len(), descriptor_len());
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tuner::persistence::Barcode;
    use crate::engine::rng::Rng;

    fn uniform(count: usize, extent: f32, seed: u64) -> Substrate {
        let mut s = Substrate::new(count, DIMENSIONS);
        let mut rng = Rng::new(seed);
        for p in &mut s.positions {
            *p = rng.unit() * extent;
        }
        for t in &mut s.traits {
            *t = rng.unit();
        }
        s
    }

    #[test]
    fn uniform_trait_conditioned_rdf_is_flat_against_measured_baseline() {
        let baseline = uniform(900, 100.0, 1);
        let other = uniform(900, 100.0, 2);
        for value in normalized_rdf(
            &raw_rdf(&other, 100.0, 0.1),
            &raw_rdf(&baseline, 100.0, 0.1),
        ) {
            assert!((value - 1.0).abs() < 0.3, "{value}");
        }
    }

    #[test]
    fn separated_biomes_differ_from_a_uniform_blend() {
        let mut biomes = Substrate::new(256, DIMENSIONS);
        let mut blend = Substrate::new(256, DIMENSIONS);
        for i in 0..256 {
            let side = if i < 128 { 0.2 } else { 0.8 };
            biomes.traits[i] = side;
            blend.traits[i] = if i % 2 == 0 { 0.2 } else { 0.8 };
            let x = if i < 128 { 20.0 } else { 80.0 };
            for axis in 0..3 {
                biomes.positions[i * 3 + axis] = x + (i % 5) as f32;
                blend.positions[i * 3 + axis] = biomes.positions[i * 3 + axis];
            }
        }
        let raw_a = raw_rdf(&biomes, 100.0, 0.1);
        let raw_b = raw_rdf(&blend, 100.0, 0.1);
        for radial in 0..RDF_BINS {
            let aggregate = |rdf: &Rdf| {
                (0..TRAIT_BANDS)
                    .map(|band| rdf.bins[band * RDF_BINS + radial])
                    .sum::<f64>()
            };
            assert_eq!(aggregate(&raw_a), aggregate(&raw_b));
        }
        let local_a = spatial_field(&biomes, 100.0, 4).heterogeneity();
        let local_b = spatial_field(&blend, 100.0, 4).heterogeneity();
        assert!(local_a[2] > local_b[2] + 0.05, "{local_a:?} vs {local_b:?}");
        assert!(local_a[3] + 0.45 < local_b[3], "{local_a:?} vs {local_b:?}");
    }

    #[test]
    fn uniformly_mixed_traits_do_not_become_fine_scale_biomes() {
        for seed in 1..=12 {
            let local = heterogeneity(&uniform(1_200, 100.0, seed), 100.0);
            for scale in local.chunks_exact(HETEROGENEITY_VALUES) {
                assert!(scale[2] < 0.01, "seed {seed}: {local:?}");
            }
        }
    }

    #[test]
    fn compact_blob_fails_and_bicontinuous_slab_passes() {
        let planted = |include: fn(usize, usize, usize) -> bool| {
            let side = 8;
            let mut field = SpatialField {
                side,
                density: vec![0.0; side.pow(3)],
                trait_sum: vec![0.0; side.pow(3)],
                trait_square_sum: vec![0.0; side.pow(3)],
                trait_bins: vec![[0; 4]; side.pow(3)],
            };
            for z in 0..side {
                for y in 0..side {
                    for x in 0..side {
                        if include(x, y, z) {
                            let index = field.index(x, y, z);
                            field.density[index] = 1.0;
                        }
                    }
                }
            }
            field.connectivity()
        };

        let compact =
            planted(|x, y, z| (2..6).contains(&x) && (2..6).contains(&y) && (2..6).contains(&z));
        assert_eq!(compact.dense, [0.0; 2], "{compact:?}");

        let bicontinuous = planted(|x, _, _| x < 4);
        for score in bicontinuous.dense.into_iter().chain(bicontinuous.void) {
            assert!(score > 0.75, "{bicontinuous:?}");
        }
    }

    #[test]
    fn radial_range_reaches_the_three_dimensional_torus_corner() {
        let substrate = Substrate::new(1_000, DIMENSIONS);
        let (_, high) = bin_range(&substrate, 100.0);
        assert!((high - 50.0 * 3.0f32.sqrt()).abs() < 1e-5, "{high}");
    }

    #[test]
    fn every_descriptor_axis_uses_the_frozen_zero_to_two_span() {
        assert!((scale_ratio(1.0) - 1.0).abs() < 1e-6);
        assert!((scale_ratio(MAX_RATIO) - AXIS_SPAN).abs() < 1e-6);
        assert!(scale_ratio(1.0 / MAX_RATIO) < 1e-6);

        let observed = Observations {
            rdf: &Rdf {
                bins: [20.0; TRAIT_BANDS * RDF_BINS],
            },
            rdf_baseline: &Rdf {
                bins: [1.0; TRAIT_BANDS * RDF_BINS],
            },
            spatial_samples: &[vec![0.0; 37], vec![100.0; 37]],
            heterogeneity: [100.0; HETEROGENEITY_SIDES.len() * HETEROGENEITY_VALUES],
            connectivity: Connectivity {
                dense: [1.0; 2],
                void: [1.0; 2],
            },
            barcode: &Barcode {
                bins: vec![0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                components: 1,
            },
            mobility: 100.0,
            turnover: 100.0,
            asymmetry: 100.0,
            robustness: 100.0,
        };
        let values = descriptor(&observed);
        assert_eq!(values.len(), descriptor_len());
        assert!(values.iter().all(|value| (0.0..=AXIS_SPAN).contains(value)));
    }
}
