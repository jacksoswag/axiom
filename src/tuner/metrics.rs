//! What a run measures, as a list it is handed rather than a struct baked in. Every metric owns a
//! file; this one routes to them, builds the blocks they share, and folds a plan into a descriptor.
//! A descriptor is a bare Vec<f32> and the plan is its layout, so reading one slot needs both.
//! Every slot lands in [0, 1], so no raw statistic with a wide range dominates a distance.

pub mod asymmetry;
pub mod connectivity;
pub mod field;
pub mod heterogeneity;
pub mod mobility;
pub mod rdf;
pub mod robustness;
pub mod structure;
pub mod temporal;
pub mod topology;
pub mod turnover;

use crate::engine::matrix::Matrix;
use crate::engine::params::FixedGenome;
use crate::engine::substrate::Substrate;
use field::SpatialField;
use rdf::Histogram;
use topology::Barcode;

/// Non-finite coordinates mean explicit Euler blew up. Takes any slice, one particle or a whole sim
pub fn alive(positions: &[f32]) -> bool { positions.iter().all(|value| value.is_finite()) }

/// Which of 'bins' log-spaced buckets between low and high a value lands in. Log so close-range
/// structure gets the resolution and long range does not waste it.
pub fn log_bin(value: f32, low: f32, high: f32, bins: usize) -> usize {
    if !value.is_finite() || value <= low { return 0; }
    (((value / low).ln() / (high / low).ln() * bins as f32) as usize).min(bins - 1)
}

/// One measurement, as a pointer at the Spec its own file wrote
pub type Metric = &'static Spec;

/// What a metric declares about itself, written in its own file
pub struct Spec {
    pub key: &'static str, // stable name, and what two handles compare on
    pub width: usize, // slots it owns in a descriptor
    pub sides: &'static [usize], // grid resolutions it reads
    pub depends: &'static [Metric], // measured ahead of it, so its slots are already filled
    pub reduce: Reduce,
    pub pairs: bool, // wants the all-pairs trait-by-radial histogram
    pub graph: bool, // wants the neighbor-edge barcode
    pub motion: bool, // wants the previous sample's positions and field
    pub measure: fn(&Blocks, &[f32]) -> Vec<f32>, // arg two is this tick's descriptor so far
}
impl Spec {
    /// The three a metric cannot omit; the rest default and are overridden with ..Spec::of()
    pub const fn of(key: &'static str, width: usize, measure: fn(&Blocks, &[f32]) -> Vec<f32>) -> Spec {
        Spec { key, width, measure, sides: &[], depends: &[], reduce: Reduce::Mean,
            pairs: false, graph: false, motion: false }
    }
}

/// Mean smooths an unlucky sample, Last is for anything that read the whole history, Once is for an
/// experiment too costly to repeat
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Reduce { Mean, Last, Once }

/// Every measurement there is, in dependency order, which is what lets plan() filter instead of sort
pub const ALL: [Metric; 10] = [rdf::METRIC, structure::METRIC, heterogeneity::METRIC,
    connectivity::METRIC, topology::METRIC, mobility::METRIC, turnover::METRIC,
    asymmetry::METRIC, temporal::METRIC, robustness::METRIC];

/// 'wanted' plus everything those metrics read, dependencies ahead of dependents. Walking ALL
/// backwards closes the set in one pass, since a dependency only ever sits earlier in ALL.
pub fn plan(wanted: &[Metric]) -> Vec<Metric> {
    let mut keys: Vec<&str> = wanted.iter().map(|metric| metric.key).collect();
    for metric in ALL.into_iter().rev() {
        if keys.contains(&metric.key) { keys.extend(metric.depends.iter().map(|read| read.key)); }
    }
    ALL.into_iter().filter(|metric| keys.contains(&metric.key)).collect()
}

/// One metric's slots in a descriptor this plan produced. A metric the plan skipped, or a descriptor
/// from a sim that died, both read empty. The key is the identity here and comparing the handles
/// would not do: a Spec is a const, so every mention of one is free to be its own copy at its own
/// address, and two handles on the same metric routinely are.
pub fn slice<'a>(values: &'a [f32], plan: &[Metric], metric: Metric) -> &'a [f32] {
    let mut start = 0;
    for named in plan {
        if named.key == metric.key { return values.get(start..start + named.width).unwrap_or(&[]); }
        start += named.width; // walk past the metrics ahead of this one
    }
    &[]
}

/// Whether a plan will ever read the uniform-start pair histogram. Building one is the most expensive
/// thing a rollout can do before it has taken a single step, and a plan with no pairs metric and no
/// repair experiment never looks at it. Robustness counts because its signature reads one.
pub fn wants_baseline(plan: &[Metric]) -> bool {
    plan.iter().any(|metric| metric.pairs || metric.key == robustness::METRIC.key)
}

/// A one-slot metric's value, or 0 when missing
pub fn scalar(values: &[f32], plan: &[Metric], metric: Metric) -> f32 {
    slice(values, plan, metric).first().copied().unwrap_or(0.0)
}

/// True of one genome for its whole run, so an experiment can re-point measurement at another
/// substrate without restating what it belongs to
pub struct Rollout<'a> {
    pub fixed_genome: &'a FixedGenome,
    pub matrix: &'a Matrix,
    pub baseline: &'a Histogram, // the same pair histogram on the uniform cloud the run started from
}

/// One sampled tick: the substrate as it stands plus the blocks this plan paid for. Asking for one
/// the plan never requested panics rather than inventing a reading.
pub struct Blocks<'a> {
    pub rollout: &'a Rollout<'a>,
    pub substrate: &'a Substrate,
    pub plan: &'a [Metric],
    pub history: &'a [Vec<f32>], // every earlier sample's descriptor, for anything reading change
    pub motion: Option<Motion>, // absent on the first sample, which has no interval behind it
    pairs: Option<Histogram>,
    fields: Vec<(usize, SpatialField)>, // ascending by side
    barcode: Option<Barcode>,
}
impl<'a> Blocks<'a> {
    /// Every block the plan asks for and nothing else, so no metric builds its own. The previous
    /// sample's motion arrives owned rather than borrowed, which is what lets carry() hand its
    /// buffers straight back instead of copying every coordinate again.
    pub fn build(rollout: &'a Rollout<'a>, substrate: &'a mut Substrate, plan: &'a [Metric],
        motion: Option<Motion>, history: &'a [Vec<f32>]) -> Blocks<'a>
    {
        // h0 re-buckets the grid at the widest interaction radius; step() re-buckets it again next
        // tick, so the index is scratch either way and this is the only mutation here.
        let barcode = plan.iter().any(|metric| metric.graph)
            .then(|| topology::h0(substrate, rollout.matrix.max_reach));
        let substrate: &'a Substrate = substrate;
        let pairs = plan.iter().any(|metric| metric.pairs).then(|| rdf::build(substrate));
        // The union of grid resolutions, on the stack: two metrics wanting one resolution cost one
        // grid, and settling that is a fixed property of the plan asked on every sample of every
        // candidate. Past MAX_SIDES resolutions this indexes off the end, which is four more than
        // every metric in the crate asks for between them.
        const MAX_SIDES: usize = 8;
        let (mut sides, mut side_count) = ([0usize; MAX_SIDES], 0);
        for metric in plan {
            for &side in metric.sides {
                if !sides[..side_count].contains(&side) { sides[side_count] = side; side_count += 1; }
            }
        }
        sides[..side_count].sort_unstable(); // ascending, which is the order fields are read back in
        let fields = sides[..side_count].iter().map(|&side| (side, field::build(substrate, side))).collect();
        Blocks { rollout, substrate, plan, history, motion, pairs, fields, barcode }
    }
    /// What the next sample needs from this one, grids included, or None when nothing reads change
    pub fn carry(mut self) -> Option<Motion> {
        if !self.plan.iter().any(|metric| metric.motion) { return None; }
        let field = self.fields.iter().position(|(side, _)| *side == turnover::SIDE)
            .map(|slot| self.fields.swap_remove(slot).1);
        // The sample before this one left a positions buffer of exactly this size behind, so refill
        // it rather than allocate a second copy of every coordinate every time a sample lands.
        let mut motion = self.motion.take().unwrap_or(Motion { positions: Vec::new(), field: None });
        motion.positions.clear();
        motion.positions.extend_from_slice(&self.substrate.positions);
        motion.field = field;
        Some(motion)
    }
    /// The same genome on another substrate under another plan, for a metric running an experiment
    pub fn probe<'b>(&self, substrate: &'b mut Substrate, plan: &'b [Metric]) -> Blocks<'b> where 'a: 'b {
        Blocks::build(self.rollout, substrate, plan, None, &[])
    }
    pub fn field(&self, side: usize) -> &SpatialField {
        &self.fields.iter().find(|(built, _)| *built == side).expect("plan did not request this grid").1
    }
    pub fn pairs(&self) -> &Histogram { self.pairs.as_ref().expect("plan did not request the pair histogram") }
    pub fn barcode(&self) -> &Barcode { self.barcode.as_ref().expect("plan did not request the neighbor graph") }
}

/// What the previous sampled tick left behind, for anything measuring change
pub struct Motion {
    pub positions: Vec<f32>,
    pub field: Option<SpatialField>, // carried only when a plan measures turnover
}

/// Run a plan against one tick, in order; each metric reads its dependencies out of the descriptor
/// being built, which the plan's ordering has already filled. Before the final sample a Once metric
/// holds its slots at zero instead of paying for them.
pub fn evaluate(base: &Blocks, full: bool) -> Vec<f32> {
    let mut out: Vec<f32> = Vec::with_capacity(base.plan.iter().map(|metric| metric.width).sum());
    for metric in base.plan {
        let mut values = if metric.reduce == Reduce::Once && !full { Vec::new() }
            else { (metric.measure)(base, &out) };
        values.resize(metric.width, 0.0); // width IS the layout, so a miscount can't shift the slots behind it
        out.extend(values);
    }
    out
}

/// One descriptor from a run's per-tick readings, each metric by its own rule. A run that never
/// sampled yields nothing rather than a plausible row of numbers.
pub fn fold(plan: &[Metric], ticks: &[Vec<f32>]) -> Vec<f32> {
    let Some(last) = ticks.last() else { return Vec::new() };
    let mut out = last.clone();
    let mut start = 0;
    for metric in plan {
        if metric.reduce == Reduce::Mean { // Last and Once are already the final tick's values
            for slot in start..start + metric.width {
                out[slot] = ticks.iter().map(|tick| tick[slot]).sum::<f32>() / ticks.len() as f32;
            }
        }
        start += metric.width;
    }
    out
}
