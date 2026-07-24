//! Fixed, three-dimensional behavior measurements used by novelty and qualification.
//! Descriptor version 5 is deliberately a world descriptor: no projection, no renderer-only
//! statistic, and a fixed 53-value layout so an archive never changes coordinate meaning mid-run.
//! Distances here are the engine's softened distance (softening is 1e-3 of box), so a perfectly
//! frozen world reads mobility near 1e-3 rather than 0; gate floors sit above that bias.

use crate::engine::kernel::distance_sq;
use crate::engine::matrix::Matrix;
use crate::engine::substrate::Substrate;
use crate::tuner::persistence::{mass, Barcode, BARS};

/// Descriptor space is 3-D-specific (winding axes, side-4/8/16 heterogeneity). Generalizing it
/// is deferred training work, so measurement stays pinned to three dimensions.
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
    pub dense: [f32; 2], // largest winding-component fraction, or 0 without volume + a torus loop
    pub void: [f32; 2], // the same fraction for thresholded void cells
}

#[derive(Clone, Debug, Default)]
pub struct Metrics {
    pub mobility: f32,
    pub temporal_variance: f32,
    pub autocorrelation: [f32; LAGS.len()],
    pub turnover: f32,
    pub robustness: f32,
    pub structure: f32, // mean departure of the trait-conditioned RDF from its uniform baseline
    pub heterogeneity: [f32; HETEROGENEITY_SIDES.len() * HETEROGENEITY_VALUES],
    pub connectivity: Connectivity,
    pub descriptor: Vec<f32>,
    pub alive: bool,
}

/// Descriptor v5 layout:
/// [ 0..18) trait-distance (3) x radial (6) RDF ratios
/// [18..33) periodic 3-D heterogeneity at sides 4, 8, 16 (five each)
/// [33..41) seven H0 death-scale masses plus cutoff-separated component mass
/// [41..45) dense then void winding-component fractions at two thresholds
/// [45..53) temporal variance, 3 lags, mobility, turnover, asymmetry, repair
pub const fn descriptor_len() -> usize {
    TRAIT_BANDS * RDF_BINS + HETEROGENEITY_SIDES.len() * HETEROGENEITY_VALUES + BARS + 4 + 1 + LAGS.len() + 1 + 1 + 1 + 1
}

#[derive(Clone, Debug)]
pub struct Rdf { pub bins: [f64; TRAIT_BANDS * RDF_BINS] }
impl Rdf {
    pub fn zeros() -> Self { Self { bins: [0.0; TRAIT_BANDS * RDF_BINS] } }
    pub fn add(&mut self, other: &Self) { for (a, b) in self.bins.iter_mut().zip(other.bins) { *a += b; } }
    pub fn scale(&mut self, factor: f64) { self.bins.iter_mut().for_each(|v| *v *= factor); }
}

fn bin_range(substrate: &Substrate) -> (f32, f32) {
    let spacing = substrate.box_len / (substrate.traits.len().max(1) as f32).powf(1.0 / DIMENSIONS as f32);
    let low = spacing.max(1e-6); // below one mean spacing a bin holds too few pairs to trust
    (low, (substrate.box_len * 0.5 * 3.0f32.sqrt()).max(low * 4.0)) // high edge reaches the torus corner
}

fn radial_bin(distance: f32, low: f32, high: f32) -> usize {
    if !distance.is_finite() || distance <= low { return 0; }
    (((distance / low).ln() / (high / low).ln() * RDF_BINS as f32) as usize).min(RDF_BINS - 1)
}

fn trait_band(a: f32, b: f32) -> usize {
    (((a - b).abs().clamp(0.0, 1.0) * TRAIT_BANDS as f32) as usize).min(TRAIT_BANDS - 1)
}

/// Counts every unordered pair into a trait-distance and log-radial band.
pub fn raw_rdf(substrate: &Substrate) -> Rdf {
    let (low, high) = bin_range(substrate);
    let mut out = Rdf::zeros();
    for i in 0..substrate.traits.len() {
        let a = substrate.pos(i);
        if !a.iter().all(|v| v.is_finite()) { continue; }
        for j in i + 1..substrate.traits.len() {
            let b = substrate.pos(j);
            if !b.iter().all(|v| v.is_finite()) { continue; }
            let radial = radial_bin(distance_sq(a, b, substrate, &mut []).sqrt(), low, high);
            out.bins[trait_band(substrate.traits[i], substrate.traits[j]) * RDF_BINS + radial] += 1.0;
        }
    }
    out
}

pub fn normalized_rdf(raw: &Rdf, baseline: &Rdf) -> Vec<f32> {
    raw.bins.iter().zip(baseline.bins)
        .map(|(&value, reference)| if reference < 0.5 { 1.0 } else { (value / reference) as f32 }.clamp(0.0, MAX_RATIO))
        .collect()
}

/// Multiplicative ratios use a symmetric log scale: 1 remains the uniform midpoint, MAX_RATIO
/// maps to the upper edge, and its reciprocal maps to the lower edge.
fn scale_ratio(ratio: f32) -> f32 {
    if ratio <= 0.0 { 0.0 } else { (1.0 + ratio.ln() / MAX_RATIO.ln()).clamp(0.0, AXIS_SPAN) }
}

pub fn structure(ratios: &[f32]) -> f32 {
    if ratios.is_empty() { 0.0 } else { ratios.iter().map(|v| (v - 1.0).abs()).sum::<f32>() / ratios.len() as f32 }
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

pub fn spatial_field(substrate: &Substrate, side: usize) -> SpatialField {
    let cells = side.pow(3);
    let mut field = SpatialField {
        side, density: vec![0.0; cells], trait_sum: vec![0.0; cells],
        trait_square_sum: vec![0.0; cells], trait_bins: vec![[0; 4]; cells]};
    let box_len = substrate.box_len;
    for (i, pos) in substrate.positions.chunks_exact(DIMENSIONS).enumerate() {
        if !pos.iter().all(|v| v.is_finite()) { continue; }
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
    fn index(&self, x: usize, y: usize, z: usize) -> usize { (z * self.side + y) * self.side + x }
    fn neighbor(&self, index: usize, axis: usize, delta: isize) -> usize {
        let side = self.side as isize;
        let mut xyz = [(index % self.side) as isize, ((index / self.side) % self.side) as isize,
            (index / (self.side * self.side)) as isize];
        xyz[axis] = (xyz[axis] + delta).rem_euclid(side);
        self.index(xyz[0] as usize, xyz[1] as usize, xyz[2] as usize)
    }
    fn local_mean_trait(&self, index: usize) -> f32 { self.trait_sum[index] / self.density[index].max(1.0) }
    pub fn heterogeneity(&self) -> [f32; HETEROGENEITY_VALUES] {
        let cells = self.density.len() as f32;
        let mean = self.density.iter().sum::<f32>() / cells;
        let raw_variance = self.density.iter().map(|value| (value - mean).powi(2)).sum::<f32>() / cells;
        // Remove the Poisson sampling variance: a uniform finite swarm should read near zero
        // instead of looking heterogeneous merely because the grid is fine.
        let density_variance = ((raw_variance - mean) / mean.max(1e-6).powi(2)).max(0.0);
        let void = self.density.iter().filter(|v| **v == 0.0).count() as f32 / cells;
        let particles = self.density.iter().sum::<f32>();
        let occupied = self.density.iter().filter(|&&n| n > 0.0).count();
        let trait_mean = self.trait_sum.iter().sum::<f32>() / particles.max(1.0);
        let between_variance = self.density.iter().enumerate().filter(|(_, n)| **n > 0.0)
            .map(|(i, &n)| n * (self.local_mean_trait(i) - trait_mean).powi(2))
            .sum::<f32>() / particles.max(1.0);
        let population_variance =
            (self.trait_square_sum.iter().sum::<f32>() / particles.max(1.0) - trait_mean * trait_mean).max(0.0);
        // Under random mixing, grouping N traits into C occupied cells contributes exactly this
        // finite-population between-cell variance in expectation. Subtracting it removes the
        // single-particle-cell false biomes seen on fine grids.
        let shot_noise = if particles > 1.0 {
            population_variance * occupied.saturating_sub(1) as f32 / (particles - 1.0)
        } else { 0.0 };
        let trait_variance = (between_variance - shot_noise).max(0.0);
        let entropy = self.density.iter().enumerate().filter(|(_, n)| **n > 0.0)
            .map(|(i, &total)| {
                let nonempty = self.trait_bins[i].iter().filter(|&&n| n > 0).count();
                let plug_in = -self.trait_bins[i].iter().filter(|&&n| n > 0)
                    .map(|&n| { let p = n as f32 / total; p * p.ln() }).sum::<f32>();
                let corrected = plug_in + nonempty.saturating_sub(1) as f32 / (2.0 * total); // small-sample correction
                total * (corrected / 4.0f32.ln()).min(1.0)
            })
            .sum::<f32>() / particles.max(1.0);
        [density_variance, void, trait_variance, entropy, axial_correlation(&self.density, self)]
    }
    pub fn connectivity(&self) -> Connectivity {
        let density = smoothed_density(self);
        let mean = density.iter().sum::<f32>() / density.len().max(1) as f32;
        let (mut dense, mut void) = ([0.0; 2], [0.0; 2]);
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
    if variance < 1e-12 { return 1.0; }
    let covariance: f32 = (0..values.len())
        .flat_map(|i| (0..DIMENSIONS).map(move |axis| (values[i] - mean) * (values[field.neighbor(i, axis, 1)] - mean)))
        .sum();
    (covariance / (DIMENSIONS as f32 * variance)).clamp(-1.0, 1.0)
}

fn smoothed_density(field: &SpatialField) -> Vec<f32> {
    let weights = [0.25, 0.5, 0.25];
    (0..field.density.len()).map(|cell| {
        let mut value = 0.0;
        for (xi, dx) in [-1, 0, 1].into_iter().enumerate() {
            for (yi, dy) in [-1, 0, 1].into_iter().enumerate() {
                for (zi, dz) in [-1, 0, 1].into_iter().enumerate() {
                    let next = field.neighbor(field.neighbor(field.neighbor(cell, 0, dx), 1, dy), 2, dz);
                    value += weights[xi] * weights[yi] * weights[zi] * field.density[next];
                }
            }
        }
        value
    }).collect()
}

/// A phase counts only when it holds enough of the grid to be a carrier, then only its winding
/// component. A compact blob is disconnected for product purposes even with every cell touching.
fn winding_component_fraction(field: &SpatialField, density: &[f32], include: impl Fn(f32) -> bool) -> f32 {
    let included = density.iter().map(|&v| include(v)).collect::<Vec<_>>();
    let total = included.iter().filter(|v| **v).count();
    if total == 0 || (total as f32 / included.len() as f32) < MIN_PHASE_VOLUME { return 0.0; }
    let mut lifts = vec![None; included.len()]; // per-cell integer unwrap of the periodic walk
    let mut largest_winding = 0;
    for root in 0..included.len() {
        if !included[root] || lifts[root].is_some() { continue; }
        let mut stack = vec![root];
        lifts[root] = Some([0i32; DIMENSIONS]);
        let (mut size, mut winds) = (0, [false; DIMENSIONS]);
        while let Some(cell) = stack.pop() { // flood fill one component
            size += 1;
            for axis in 0..DIMENSIONS {
                for direction in [-1, 1] {
                    let next = field.neighbor(cell, axis, direction);
                    if !included[next] { continue; }
                    let mut proposed = lifts[cell].expect("visited cell has a lift");
                    proposed[axis] += direction as i32;
                    if let Some(existing) = lifts[next] {
                        // Reaching a cell by two walks that disagree on the lift = a loop around the torus
                        for winding_axis in 0..DIMENSIONS { winds[winding_axis] |= proposed[winding_axis] != existing[winding_axis]; }
                    } else { lifts[next] = Some(proposed); stack.push(next); }
                }
            }
        }
        if winds.into_iter().any(|wrapped| wrapped) { largest_winding = largest_winding.max(size); }
    }
    largest_winding as f32 / total as f32
}
/// A qualifying phase holds at least 8 percent of the descriptor grid.
const MIN_PHASE_VOLUME: f32 = 0.08;

pub fn heterogeneity(substrate: &Substrate) -> [f32; HETEROGENEITY_SIDES.len() * HETEROGENEITY_VALUES] {
    let mut out = [0.0; HETEROGENEITY_SIDES.len() * HETEROGENEITY_VALUES];
    for (scale, side) in HETEROGENEITY_SIDES.into_iter().enumerate() {
        out[scale * HETEROGENEITY_VALUES..][..HETEROGENEITY_VALUES]
            .copy_from_slice(&spatial_field(substrate, side).heterogeneity());
    }
    out
}

/// Topology uses an eight-cell side and a periodic compact density deposit: coarse enough for
/// 320-particle discovery worlds to show corridors, fine enough to resolve bodies at 1,200.
pub fn connectivity(substrate: &Substrate) -> Connectivity { spatial_field(substrate, 8).connectivity() }

pub fn temporal(samples: &[Vec<f32>]) -> (f32, [f32; LAGS.len()]) {
    let mut lags = [1.0; LAGS.len()];
    if samples.len() < 2 || samples[0].is_empty() { return (0.0, lags); }
    let width = samples[0].len();
    let variance = (0..width).map(|i| {
        let mean = samples.iter().map(|s| s[i]).sum::<f32>() / samples.len() as f32;
        samples.iter().map(|s| (s[i] - mean).powi(2)).sum::<f32>() / samples.len() as f32
    }).sum::<f32>() / width as f32;
    for (slot, lag) in lags.iter_mut().zip(LAGS) {
        let spans = samples.len().saturating_sub(lag);
        if spans > 0 {
            *slot = (0..spans).map(|i| correlation(&samples[i], &samples[i + lag])).sum::<f32>() / spans as f32;
        }
    }
    (variance, lags)
}

/// The spatial blocks in descriptor units. Keeping rollout samples and repair comparisons in
/// this fixed [0, 2] space prevents a high-variance raw statistic from dominating either.
pub fn scaled_spatial_features(rdf: &Rdf, rdf_baseline: &Rdf,
    heterogeneity: [f32; HETEROGENEITY_SIDES.len() * HETEROGENEITY_VALUES], connectivity: Connectivity) -> Vec<f32>
{
    let mut out = Vec::with_capacity(TRAIT_BANDS * RDF_BINS + heterogeneity.len() + 4);
    out.extend(normalized_rdf(rdf, rdf_baseline).into_iter().map(scale_ratio));
    out.extend(scale_heterogeneity(heterogeneity));
    out.extend(connectivity.dense.map(|v| v * AXIS_SPAN));
    out.extend(connectivity.void.map(|v| v * AXIS_SPAN));
    out
}

fn correlation(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 { return 1.0; }
    let (ma, mb) = (a[..n].iter().sum::<f32>() / n as f32, b[..n].iter().sum::<f32>() / n as f32);
    let (mut cov, mut va, mut vb) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let (x, y) = (a[i] - ma, b[i] - mb);
        cov += x * y; va += x * x; vb += y * y;
    }
    if va < 1e-12 || vb < 1e-12 { 1.0 } else { (cov / (va * vb).sqrt()).clamp(-1.0, 1.0) }
}

/// Mean minimum-image travel since 'before', as a fraction of the box. 'after' is the
/// substrate's current positions, so callers snapshot once and hand the substrate back later.
pub fn mobility(before: &[f32], substrate: &Substrate) -> f32 {
    if before.is_empty() || before.len() != substrate.positions.len() { return 0.0; }
    let count = before.len() / DIMENSIONS;
    (0..count)
        .map(|i| distance_sq(substrate.pos(i), &before[i * DIMENSIONS..(i + 1) * DIMENSIONS], substrate, &mut []).sqrt())
        .sum::<f32>() / count as f32 / substrate.box_len
}

pub fn turnover(before: &SpatialField, after: &SpatialField) -> f32 {
    before.density.iter().zip(&after.density).map(|(a, b)| (a - b).abs()).sum::<f32>()
        / (before.density.iter().sum::<f32>() + after.density.iter().sum::<f32>()).max(1.0)
}

/// Normalized asymmetry of the directed weight matrix: 0 for symmetric laws, capped at AXIS_SPAN.
pub fn asymmetry(matrix: &Matrix) -> f32 {
    let anchor_count = matrix.anchor_count;
    let (mut difference, mut total) = (0.0f32, 0.0f32);
    for source in 0..anchor_count {
        for destination in 0..anchor_count {
            let a = matrix.interactions[source * anchor_count + destination].weight;
            let b = matrix.interactions[destination * anchor_count + source].weight;
            difference += (a - b).powi(2); total += a.powi(2);
        }
    }
    if total < 1e-12 { 0.0 } else { (difference / total).sqrt().min(AXIS_SPAN) }
}

/// Everything one rollout observed, gathered for descriptor assembly.
pub struct Observations<'a> {
    pub rdf: &'a Rdf,
    pub rdf_baseline: &'a Rdf,
    pub spatial_samples: &'a [Vec<f32>],
    pub heterogeneity: [f32; HETEROGENEITY_SIDES.len() * HETEROGENEITY_VALUES],
    pub connectivity: Connectivity,
    pub barcode: &'a Barcode,
    pub mobility: f32,
    pub turnover: f32,
    pub asymmetry: f32,
    pub robustness: f32,
}

fn scale_heterogeneity(values: [f32; HETEROGENEITY_SIDES.len() * HETEROGENEITY_VALUES]) -> impl Iterator<Item = f32> {
    values.into_iter().enumerate().map(|(i, v)| match i % HETEROGENEITY_VALUES {
        0 => (v / 4.0 * AXIS_SPAN).clamp(0.0, AXIS_SPAN), // density CV^2
        1 | 3 => (v * AXIS_SPAN).clamp(0.0, AXIS_SPAN),
        2 => (v * 8.0).clamp(0.0, AXIS_SPAN), // trait variance tops out at .25
        _ => (v + 1.0).clamp(0.0, AXIS_SPAN), // correlation from [-1,1]
    })
}

pub fn descriptor(observed: &Observations) -> Vec<f32> {
    let mut out = Vec::with_capacity(descriptor_len());
    let spatial = scaled_spatial_features(observed.rdf, observed.rdf_baseline, observed.heterogeneity, observed.connectivity);
    out.extend(&spatial[..TRAIT_BANDS * RDF_BINS + observed.heterogeneity.len()]);
    out.extend(mass(observed.barcode).into_iter().map(|value| value.clamp(0.0, AXIS_SPAN)));
    out.extend(&spatial[TRAIT_BANDS * RDF_BINS + observed.heterogeneity.len()..]);
    let (variance, lags) = temporal(observed.spatial_samples);
    out.push(variance.clamp(0.0, AXIS_SPAN));
    out.extend(lags.map(|value| (value + 1.0).clamp(0.0, AXIS_SPAN)));
    out.push((observed.mobility / MOBILITY_CEILING * AXIS_SPAN).clamp(0.0, AXIS_SPAN));
    out.push((observed.turnover * AXIS_SPAN).clamp(0.0, AXIS_SPAN));
    out.push(observed.asymmetry.clamp(0.0, AXIS_SPAN));
    out.push((observed.robustness * AXIS_SPAN).clamp(0.0, AXIS_SPAN));
    out
}
