//! Is the sim still changing, and is the change structured or noise. The one metric reading other
//! metrics: it takes their outputs at every sampled tick, asks how far that picture moved, then how
//! much it still resembles itself a few samples later. A lag correlation near 1 means it looks the
//! same later, so a frozen crystal and a slow drift both read high; near 0 means samples are unrelated.

use super::{connectivity, heterogeneity, rdf, slice, Blocks, Metric, Reduce, Spec};

/// Short, medium, and long relative to a rollout's sampling, so a fast oscillation and a slow drift
/// do not read alike
const LAGS: [usize; 3] = [1, 2, 4];
const WIDTH: usize = 1 + LAGS.len();
/// The axes one tick's picture is drawn on, declared once and read twice: depends makes the plan
/// measure them ahead of this metric, and measure pulls the same three back out of the descriptor.
const READS: [Metric; 3] = [rdf::METRIC, heterogeneity::METRIC, connectivity::METRIC];
pub const SPEC: Spec = Spec { reduce: Reduce::Last, depends: &READS, ..Spec::of("temporal", WIDTH, measure) };
pub const METRIC: Metric = &SPEC;

/// Its axes sit in the descriptor already in shared units, so one tick's picture is their slices
/// laid end to end and nothing needs rescaling before comparison.
pub fn measure(base: &Blocks, current: &[f32]) -> Vec<f32> {
    let picture = |tick: &[f32]| -> Vec<f32> {
        READS.iter().flat_map(|&metric| slice(tick, base.plan, metric)).copied().collect()
    };
    let samples: Vec<Vec<f32>> = base.history.iter().map(|tick| picture(tick))
        .chain(std::iter::once(picture(current))).collect();
    let (variance, lags) = of(&samples);
    std::iter::once(variance.clamp(0.0, 1.0))
        .chain(lags.map(|value| ((value + 1.0) * 0.5).clamp(0.0, 1.0))).collect()
}

/// Defaults of 1.0 with too few samples mean "assume static", which fails a variance floor rather
/// than sneaking past it.
fn of(samples: &[Vec<f32>]) -> (f32, [f32; LAGS.len()]) {
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
