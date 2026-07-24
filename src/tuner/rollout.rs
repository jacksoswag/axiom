//! Deterministic, headless rollout and promotion evaluation.

use crate::engine::kernel::distance_sq;
use crate::engine::params::Params;
use crate::engine::sim::Sim;
use crate::engine::substrate::Substrate;
use crate::tuner::genome::clamp;
use crate::tuner::metrics::{asymmetry, connectivity, descriptor, heterogeneity, mobility, normalized_rdf,
    raw_rdf, scaled_spatial_features, spatial_field, structure, temporal, turnover, Connectivity, Metrics,
    Observations, Rdf, SpatialField, HETEROGENEITY_SIDES, HETEROGENEITY_VALUES};
use crate::tuner::novelty::distance;
use crate::tuner::persistence::{h0, Barcode, BARS};
use crate::engine::resolve::Probe;
use crate::tuner::tuning::{Repair, Tuning};
use crate::tuner::viability::{margins, viable, world_qualified, GateMargins, Rejection, WorldRejection};
use crate::util::Rng;

/// A genome pairing a large weight with a small dt blows explicit Euler up to NaN, which ends
/// the rollout. The check lives here rather than on Substrate, which holds no runtime logic.
fn finite(substrate: &Substrate) -> bool {
    substrate.positions.iter().all(|value| value.is_finite())
}

/// Time-local samples taken per rollout. This sizes the temporal block of the descriptor, so
/// it is part of the descriptor contract rather than a per-run knob.
pub const SAMPLES: usize = 8;

/// Seed count, pass requirement, and whether the world gate applies are properties of the tier
/// itself. A run chooses which tiers to spend on and how long each rollout is; it cannot
/// weaken what a tier means.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvaluationTier { Discovery, Persistence, Certification }
impl EvaluationTier {
    pub const ALL: [Self; 3] = [Self::Discovery, Self::Persistence, Self::Certification];
    pub const fn seeds(self) -> usize {
        match self { Self::Discovery => 1, Self::Persistence => 3, Self::Certification => 5 }
    }
    pub const fn required_passes(self) -> usize {
        match self { Self::Discovery => 1, Self::Persistence => 2, Self::Certification => 4 }
    }
    pub const fn requires_world(self) -> bool { matches!(self, Self::Persistence | Self::Certification) }
    /// Shortest rollout that still earns the tier's name.
    pub const fn minimum_steps(self) -> usize {
        match self { Self::Discovery => 1_500, Self::Persistence => 10_000, Self::Certification => 100_000 }
    }
    pub const fn label(self) -> &'static str {
        match self { Self::Discovery => "discovery", Self::Persistence => "persistence", Self::Certification => "certification" }
    }
}

/// A tier and how long its rollouts run. Nothing else about a tier is configurable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EvaluationBudget {
    pub tier: EvaluationTier,
    pub steps: usize,
}
impl EvaluationBudget {
    pub const fn new(tier: EvaluationTier, steps: usize) -> Self { Self { tier, steps } }
    pub const fn seeds(self) -> usize { self.tier.seeds() }
    pub const fn required_passes(self) -> usize { self.tier.required_passes() }
    pub const fn requires_world(self) -> bool { self.tier.requires_world() }
    /// Whether this budget still earns its tier's name.
    pub const fn honors_tier(self) -> bool { self.steps >= self.tier.minimum_steps() }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EarlyStop { NonFinite, SustainedDispersion, SustainedCollapse }

#[derive(Clone, Debug)]
pub struct SeedRecord {
    pub seed: u64,
    pub metrics: Metrics,
    pub windows: Vec<WindowRecord>,
    pub margins: GateMargins,
    pub base: Result<(), Rejection>,
    pub world: Result<(), WorldRejection>,
    pub early_stop: Option<EarlyStop>,
}

/// One time-local observation retained for drift checks and learned scheduling.
#[derive(Clone, Debug)]
pub struct WindowRecord {
    pub tick: usize,
    pub spatial: Vec<f32>,
    pub structure: f32,
    pub heterogeneity: [f32; HETEROGENEITY_SIDES.len() * HETEROGENEITY_VALUES],
    pub connectivity: Connectivity,
    pub mobility: f32,
    pub turnover: f32,
}

#[derive(Clone, Debug)]
struct RolloutOutcome {
    metrics: Metrics,
    windows: Vec<WindowRecord>,
    early_stop: Option<EarlyStop>,
}

#[derive(Clone, Debug)]
pub struct TierEvaluation {
    pub budget: EvaluationBudget,
    pub records: Vec<SeedRecord>,
    pub passes: usize,
}
impl TierEvaluation {
    pub fn passed(&self) -> bool { self.passes >= self.budget.required_passes() }
}

/// Resolve a full genome against the tuned world, then re-clamp its pair genes at its own
/// resolved box, so any genome (archived, curated, hand-fed) is safe to simulate.
fn resolve_clamped(tuning: &Tuning, probe: &Probe, genome: &[f32]) -> Params {
    let mut params = tuning.world.params(genome, probe);
    let bounds = tuning.world.pair_bounds(params.box_len);
    clamp(&mut params.interactions, &bounds);
    params
}

/// Campaign path: one discovery rollout of a genome.
pub fn discovery_metrics(tuning: &Tuning, probe: &Probe, genome: &[f32], steps: usize) -> Metrics {
    rollout(tuning, &resolve_clamped(tuning, probe, genome), steps)
}

/// Campaign path: all candidates share one measured density reference.
pub fn evaluate_tier(tuning: &Tuning, probe: &Probe, genome: &[f32], budget: EvaluationBudget) -> TierEvaluation {
    let template = resolve_clamped(tuning, probe, genome);
    let records = (0..budget.seeds()).map(|offset| {
        let mut params = template.clone();
        params.seed = tier_seed(template.seed, budget.tier, offset);
        let outcome = rollout_status(tuning, &params, budget.steps);
        let metrics = outcome.metrics;
        let gate_margins = margins(&metrics, &tuning.gates);
        let base = viable(&metrics, &tuning.gates);
        let world = if budget.requires_world() { world_qualified(&metrics, &tuning.gates) } else { Ok(()) };
        SeedRecord { seed: params.seed, metrics, windows: outcome.windows, margins: gate_margins,
            base, world, early_stop: outcome.early_stop }
    }).collect::<Vec<_>>();
    let passes = records.iter().filter(|r| r.base.is_ok() && r.world.is_ok()).count();
    TierEvaluation { budget, records, passes }
}

pub fn rollout(tuning: &Tuning, params: &Params, steps: usize) -> Metrics {
    rollout_status(tuning, params, steps).metrics
}

/// Disjoint seed spaces per tier, so certification always sees unseen seeds.
fn tier_seed(base: u64, tier: EvaluationTier, offset: usize) -> u64 {
    let tier_offset = match tier {
        EvaluationTier::Discovery => 0,
        EvaluationTier::Persistence => 1u64 << 32,
        EvaluationTier::Certification => 2u64 << 32,
    };
    base.wrapping_add(tier_offset).wrapping_add(offset as u64)
}

fn rollout_status(tuning: &Tuning, params: &Params, steps: usize) -> RolloutOutcome {
    let mut sim = Sim::new(params);
    let cutoff = sim.matrix.max_reach();
    let baseline = raw_rdf(&sim.substrate); // uniform start is the structure reference
    let interval = (steps / (2 * SAMPLES)).max(1); // samples spread across the latter half
    let first_sample = steps.saturating_sub(interval * SAMPLES);
    let watchdog_interval = (steps / 16).max(1);
    let watchdog_start = steps / 2;
    let (mut dispersed, mut collapsed) = (0, 0);
    let mut rdf = Rdf::zeros();
    let mut barcode_bins = vec![0.0; BARS];
    let mut barcode_components = 0usize;
    let mut heterogeneity_total = [0.0; HETEROGENEITY_SIDES.len() * HETEROGENEITY_VALUES];
    let (mut dense, mut void) = ([0.0; 2], [0.0; 2]);
    let mut spatial_samples = Vec::with_capacity(SAMPLES);
    let (mut mobility_total, mut turnover_total) = (0.0, 0.0);
    let mut previous_positions: Option<Vec<f32>> = None;
    let mut previous_field: Option<SpatialField> = None;
    let (mut samples, mut intervals) = (0usize, 0usize);
    let mut windows = Vec::with_capacity(SAMPLES);

    for tick in 0..steps {
        sim.step();
        if !finite(&sim.substrate) {
            return RolloutOutcome { metrics: Metrics::default(), windows, early_stop: Some(EarlyStop::NonFinite) };
        }
        if tick >= watchdog_start && tick % watchdog_interval == 0 {
            let current = structure(&normalized_rdf(&raw_rdf(&sim.substrate), &baseline));
            dispersed = if current <= tuning.gates.dispersed_structure { dispersed + 1 } else { 0 };
            collapsed = if current >= tuning.gates.collapsed_structure { collapsed + 1 } else { 0 };
            if dispersed >= tuning.gates.sustained_windows {
                return RolloutOutcome { metrics: Metrics::default(), windows, early_stop: Some(EarlyStop::SustainedDispersion) };
            }
            if collapsed >= tuning.gates.sustained_windows {
                return RolloutOutcome { metrics: Metrics::default(), windows, early_stop: Some(EarlyStop::SustainedCollapse) };
            }
        }
        if tick < first_sample || (tick - first_sample) % interval != 0 { continue; }
        let sample_rdf = raw_rdf(&sim.substrate);
        rdf.add(&sample_rdf);
        let field = spatial_field(&sim.substrate, 16);
        let local = heterogeneity(&sim.substrate);
        let connected = connectivity(&sim.substrate);
        spatial_samples.push(scaled_spatial_features(&sample_rdf, &baseline, local, connected));
        for (total, value) in heterogeneity_total.iter_mut().zip(local) { *total += value; }
        for (total, value) in dense.iter_mut().zip(connected.dense) { *total += value; }
        for (total, value) in void.iter_mut().zip(connected.void) { *total += value; }
        let barcode = h0(&mut sim.substrate, cutoff); // rebuilds the grid; step() re-buckets next tick
        for (total, value) in barcode_bins.iter_mut().zip(barcode.bins) { *total += value; }
        barcode_components += barcode.components;
        let window_mobility = previous_positions.as_ref().map_or(0.0, |previous| mobility(previous, &sim.substrate));
        let window_turnover = previous_field.as_ref().map_or(0.0, |previous| turnover(previous, &field));
        if previous_positions.is_some() { // first sample has no interval behind it
            mobility_total += window_mobility; turnover_total += window_turnover; intervals += 1;
        }
        windows.push(WindowRecord {
            tick: tick + 1,
            spatial: spatial_samples.last().cloned().unwrap_or_default(),
            structure: structure(&normalized_rdf(&sample_rdf, &baseline)),
            heterogeneity: local, connectivity: connected,
            mobility: window_mobility, turnover: window_turnover,
        });
        previous_positions = Some(sim.substrate.positions.clone());
        previous_field = Some(field);
        samples += 1;
    }
    if samples == 0 {
        return RolloutOutcome { metrics: Metrics::default(), windows, early_stop: None };
    }
    rdf.scale(1.0 / samples as f64);
    barcode_bins.iter_mut().for_each(|v| *v /= samples as f64);
    heterogeneity_total.iter_mut().for_each(|v| *v /= samples as f32);
    dense.iter_mut().for_each(|v| *v /= samples as f32);
    void.iter_mut().for_each(|v| *v /= samples as f32);
    let mobility = mobility_total / intervals.max(1) as f32;
    let turnover = turnover_total / intervals.max(1) as f32;
    let robustness = self_maintenance(&sim, &RepairProtocol {
        params, baseline: &baseline, config: tuning.repair,
        steps: (steps / 4).max(50), shake: cutoff * tuning.repair.shake, seed: params.seed,
    });
    let (temporal_variance, autocorrelation) = temporal(&spatial_samples);
    let heterogeneity = heterogeneity_total;
    let connectivity = Connectivity { dense, void };
    let descriptor = descriptor(&Observations {
        rdf: &rdf, rdf_baseline: &baseline, spatial_samples: &spatial_samples,
        heterogeneity, connectivity,
        barcode: &Barcode {
            bins: barcode_bins,
            components: ((barcode_components as f32 / samples as f32).round() as usize).max(1),
        },
        mobility, turnover, asymmetry: asymmetry(&sim.matrix), robustness,
    });
    RolloutOutcome {
        metrics: Metrics {
            mobility, temporal_variance, autocorrelation, turnover, robustness,
            structure: structure(&normalized_rdf(&rdf, &baseline)),
            heterogeneity, connectivity, descriptor, alive: true,
        },
        windows, early_stop: None,
    }
}

struct RepairProtocol<'a> {
    params: &'a Params,
    baseline: &'a Rdf,
    config: Repair,
    steps: usize,
    shake: f32,
    seed: u64,
}

/// A fresh Sim carrying a snapshot of another run: same law, same tick, copied state.
fn from_snapshot(params: &Params, positions: &[f32], traits: &[f32], tick: u64) -> Sim {
    let mut sim = Sim::new(params);
    sim.substrate.positions = positions.to_vec();
    sim.substrate.traits = traits.to_vec();
    sim.tick = tick;
    sim
}

/// The repair fraction: shake a local region, run on, and ask whether the world returns to
/// where its own undamaged control went. Compared against the future control rather than the
/// present state, so ordinary drift is not billed as damage.
fn self_maintenance(sim: &Sim, protocol: &RepairProtocol<'_>) -> f32 {
    let box_len = sim.substrate.box_len;
    let positions = sim.substrate.positions.clone();
    let traits = sim.substrate.traits.clone();
    let initial_control = recovery_signature(&sim.substrate, protocol.baseline);
    let mut control = from_snapshot(protocol.params, &positions, &traits, sim.tick);
    for _ in 0..protocol.steps {
        control.step();
        if !finite(&control.substrate) { return 0.0; }
    }
    let future_control = recovery_signature(&control.substrate, protocol.baseline);
    let mut recovered = 0;
    for probe in 0..protocol.config.probes {
        let mut damaged = from_snapshot(protocol.params, &positions, &traits, sim.tick);
        let mut rng = Rng::new(protocol.seed.wrapping_mul(0x9E37_79B9).wrapping_add(probe as u64 + 1));
        if damaged.substrate.traits.is_empty() { continue; }
        let dims = damaged.substrate.dimensions;
        let center = damaged.substrate.pos(rng.below(damaged.substrate.traits.len())).to_vec();
        let region = (2.0 * protocol.shake).min(box_len * 0.25); // damage stays local to one neighborhood
        let hit: Vec<usize> = (0..damaged.substrate.traits.len())
            .filter(|&i| distance_sq(damaged.substrate.pos(i), &center, &damaged.substrate, &mut []).sqrt() <= region)
            .collect();
        for i in hit { // same particle order as the scan, so rng draws stay reproducible
            for coordinate in &mut damaged.substrate.positions[i * dims..(i + 1) * dims] {
                *coordinate = (*coordinate + protocol.shake * rng.normal()).rem_euclid(box_len);
            }
        }
        let damage = distance(&initial_control, &recovery_signature(&damaged.substrate, protocol.baseline));
        for _ in 0..protocol.steps {
            damaged.step();
            if !finite(&damaged.substrate) { break; }
        }
        if damage >= protocol.config.min_effective_damage
            && finite(&damaged.substrate)
            && distance(&future_control, &recovery_signature(&damaged.substrate, protocol.baseline))
                < protocol.config.recovery * damage
        { recovered += 1; }
    }
    recovered as f32 / protocol.config.probes.max(1) as f32
}

fn recovery_signature(substrate: &Substrate, baseline: &Rdf) -> Vec<f32> {
    scaled_spatial_features(&raw_rdf(substrate), baseline, heterogeneity(substrate), connectivity(substrate))
}
