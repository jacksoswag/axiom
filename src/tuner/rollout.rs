//! Deterministic, headless rollout and promotion evaluation.

use crate::engine::grid::Grid;
use crate::tuner::metrics::{
    asymmetry, connectivity, descriptor, heterogeneity, mobility, normalized_rdf, raw_rdf,
    scaled_spatial_features, spatial_field, structure, temporal, turnover, Connectivity, Metrics,
    Observations, Rdf, SpatialField, HETEROGENEITY_SIDES, HETEROGENEITY_VALUES,
};
use crate::tuner::novelty::distance;
use crate::tuner::persistence::{h0, Barcode, BARS};
use crate::engine::rng::Rng;
use crate::tuner::tuning::{Repair, Tuning};
use crate::tuner::viability::{margins, viable, world_qualified, GateMargins, Rejection, WorldRejection};
use crate::engine::world::World;
use crate::engine::genome::Params;
use crate::engine::geometry::{Geometry, GeometryScale};

/// A genome pairing a large weight with a small dt blows explicit Euler up to NaN, which ends
/// the rollout. The check lives here rather than on `Substrate`, which holds no runtime logic.
fn finite(substrate: &crate::engine::substrate::Substrate) -> bool {
    substrate.positions.iter().all(|value| value.is_finite())
}

/// Time-local samples taken per rollout. This sizes the temporal block of the descriptor, so
/// it is part of the descriptor contract rather than a per-run knob.
pub const SAMPLES: usize = 8;

struct RepairProtocol<'a> {
    params: &'a Params,
    geometry: &'a Geometry,
    genome: &'a [f32],
    baseline: &'a Rdf,
    config: Repair,
    steps: usize,
    shake: f32,
    seed: u64,
}

/// Seed count, pass requirement, and whether the world gate applies are properties of the tier
/// itself. A run chooses which tiers to spend on and how long each rollout is; it cannot
/// weaken what a tier means.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvaluationTier {
    Discovery,
    Persistence,
    Certification,
}

impl EvaluationTier {
    pub const ALL: [Self; 3] = [Self::Discovery, Self::Persistence, Self::Certification];

    pub const fn seeds(self) -> usize {
        match self {
            Self::Discovery => 1,
            Self::Persistence => 3,
            Self::Certification => 5,
        }
    }

    pub const fn required_passes(self) -> usize {
        match self {
            Self::Discovery => 1,
            Self::Persistence => 2,
            Self::Certification => 4,
        }
    }

    pub const fn requires_world(self) -> bool {
        matches!(self, Self::Persistence | Self::Certification)
    }

    /// Shortest rollout that still earns the tier's name.
    pub const fn minimum_steps(self) -> usize {
        match self {
            Self::Discovery => 1_500,
            Self::Persistence => 10_000,
            Self::Certification => 100_000,
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::Discovery => "discovery",
            Self::Persistence => "persistence",
            Self::Certification => "certification",
        }
    }
}

/// A tier and how long its rollouts run. Nothing else about a tier is configurable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EvaluationBudget {
    pub tier: EvaluationTier,
    pub steps: usize,
}

impl EvaluationBudget {
    pub const fn new(tier: EvaluationTier, steps: usize) -> Self {
        Self { tier, steps }
    }
    pub const fn discovery() -> Self {
        Self::new(EvaluationTier::Discovery, EvaluationTier::Discovery.minimum_steps())
    }
    pub const fn persistence() -> Self {
        Self::new(
            EvaluationTier::Persistence,
            EvaluationTier::Persistence.minimum_steps(),
        )
    }
    pub const fn certification() -> Self {
        Self::new(
            EvaluationTier::Certification,
            EvaluationTier::Certification.minimum_steps(),
        )
    }
    pub const fn seeds(self) -> usize {
        self.tier.seeds()
    }
    pub const fn required_passes(self) -> usize {
        self.tier.required_passes()
    }
    pub const fn requires_world(self) -> bool {
        self.tier.requires_world()
    }
    /// Whether this budget still earns its tier's name.
    pub const fn honours_tier(self) -> bool {
        self.steps >= self.tier.minimum_steps()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EarlyStop {
    NonFinite,
    SustainedDispersion,
    SustainedCollapse,
}

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
    pub fn passed(&self) -> bool {
        self.passes >= self.budget.required_passes()
    }
}

/// Resolve a genome against the tuned world and clamp it to its own bounds. Callers retain the
/// scale, avoiding the density probe per genome.
fn resolve_clamped(
    tuning: &Tuning,
    scale: GeometryScale,
    genome: &[f32],
) -> (Params, Geometry, Vec<f32>) {
    let (params, interactions) = tuning.world.resolve(genome);
    let geometry = params.geometry_with(scale);
    let mut genes = interactions.to_vec();
    params
        .layout()
        .clamp(&mut genes, params.radius, geometry.extent);
    (params, geometry, genes)
}

/// Campaign path: one discovery rollout of a genome.
pub fn from_genome_with_scale(
    tuning: &Tuning,
    scale: GeometryScale,
    genome: &[f32],
    steps: usize,
) -> Metrics {
    let (params, geometry, genes) = resolve_clamped(tuning, scale, genome);
    rollout(tuning, &params, &geometry, &genes, steps)
}

/// Campaign path: all candidates use one measured density reference.
pub fn evaluate_tier_with_scale(
    tuning: &Tuning,
    scale: GeometryScale,
    genome: &[f32],
    budget: EvaluationBudget,
) -> TierEvaluation {
    let (template, geometry, genes) = resolve_clamped(tuning, scale, genome);
    let records = (0..budget.seeds())
        .map(|offset| {
            let mut params = template.clone();
            params.seed = tier_seed(template.seed, budget.tier, offset);
            let outcome = rollout_status(tuning, &params, &geometry, &genes, budget.steps);
            let metrics = outcome.metrics;
            let gate_margins = margins(&metrics, &tuning.gates);
            let base = viable(&metrics, &tuning.gates);
            let world = if budget.requires_world() {
                world_qualified(&metrics, &tuning.gates)
            } else {
                Ok(())
            };
            SeedRecord {
                seed: params.seed,
                metrics,
                windows: outcome.windows,
                margins: gate_margins,
                base,
                world,
                early_stop: outcome.early_stop,
            }
        })
        .collect::<Vec<_>>();
    let passes = records
        .iter()
        .filter(|r| r.base.is_ok() && r.world.is_ok())
        .count();
    TierEvaluation {
        budget,
        records,
        passes,
    }
}

pub fn rollout(
    tuning: &Tuning,
    params: &Params,
    geometry: &Geometry,
    genome: &[f32],
    steps: usize,
) -> Metrics {
    rollout_status(tuning, params, geometry, genome, steps).metrics
}

fn tier_seed(base: u64, tier: EvaluationTier, offset: usize) -> u64 {
    let tier_offset = match tier {
        EvaluationTier::Discovery => 0,
        EvaluationTier::Persistence => 1u64 << 32,
        EvaluationTier::Certification => 2u64 << 32,
    };
    base.wrapping_add(tier_offset).wrapping_add(offset as u64)
}

fn rollout_status(
    tuning: &Tuning,
    params: &Params,
    geometry: &Geometry,
    genome: &[f32],
    steps: usize,
) -> RolloutOutcome {
    let mut world = World::with_geometry(params, geometry, genome);
    let (extent, softening, cutoff) = (
        world.geometry.extent,
        world.geometry.softening,
        world.net.max_reach(),
    );
    let baseline = raw_rdf(&world.substrate, extent, softening);
    let interval = (steps / (2 * SAMPLES)).max(1);
    let first_sample = steps.saturating_sub(interval * SAMPLES);
    let watchdog_interval = (steps / 16).max(1);
    let watchdog_start = steps / 2;
    let mut dispersed = 0;
    let mut collapsed = 0;
    let mut grid = Grid::default();
    let mut rdf = Rdf::zeros();
    let mut barcode_bins = vec![0.0; BARS];
    let mut barcode_components = 0usize;
    let mut heterogeneity_total = [0.0; HETEROGENEITY_SIDES.len() * HETEROGENEITY_VALUES];
    let mut dense = [0.0; 2];
    let mut void = [0.0; 2];
    let mut spatial_samples = Vec::with_capacity(SAMPLES);
    let mut mobility_total = 0.0;
    let mut turnover_total = 0.0;
    let mut previous_positions: Option<Vec<f32>> = None;
    let mut previous_field: Option<SpatialField> = None;
    let mut samples = 0usize;
    let mut intervals = 0usize;
    let mut windows = Vec::with_capacity(SAMPLES);

    for tick in 0..steps {
        world.step();
        if !finite(&world.substrate) {
            return RolloutOutcome {
                metrics: Metrics::default(),
                windows,
                early_stop: Some(EarlyStop::NonFinite),
            };
        }
        if tick >= watchdog_start && tick % watchdog_interval == 0 {
            let current = structure(&normalized_rdf(
                &raw_rdf(&world.substrate, extent, softening),
                &baseline,
            ));
            dispersed = if current <= tuning.gates.dispersed_structure {
                dispersed + 1
            } else {
                0
            };
            collapsed = if current >= tuning.gates.collapsed_structure {
                collapsed + 1
            } else {
                0
            };
            if dispersed >= tuning.gates.sustained_windows {
                return RolloutOutcome {
                    metrics: Metrics::default(),
                    windows,
                    early_stop: Some(EarlyStop::SustainedDispersion),
                };
            }
            if collapsed >= tuning.gates.sustained_windows {
                return RolloutOutcome {
                    metrics: Metrics::default(),
                    windows,
                    early_stop: Some(EarlyStop::SustainedCollapse),
                };
            }
        }
        if tick < first_sample || !(tick - first_sample).is_multiple_of(interval) {
            continue;
        }
        let sample_rdf = raw_rdf(&world.substrate, extent, softening);
        rdf.add(&sample_rdf);
        let field = spatial_field(&world.substrate, extent, 16);
        let h = heterogeneity(&world.substrate, extent);
        let c = connectivity(&world.substrate, extent);
        let spatial = scaled_spatial_features(&sample_rdf, &baseline, h, c);
        spatial_samples.push(spatial);
        for (total, value) in heterogeneity_total.iter_mut().zip(h) {
            *total += value;
        }
        for (total, value) in dense.iter_mut().zip(c.dense) {
            *total += value;
        }
        for (total, value) in void.iter_mut().zip(c.void) {
            *total += value;
        }
        grid.rebuild(&world.substrate.positions, world.substrate.dims(), extent, cutoff);
        let barcode = h0(&world.substrate, extent, cutoff, &grid);
        for (total, value) in barcode_bins.iter_mut().zip(barcode.bins) {
            *total += value;
        }
        barcode_components += barcode.components;
        let window_mobility = previous_positions.as_ref().map_or(0.0, |previous| {
            mobility(previous, &world.substrate.positions, extent)
        });
        let window_turnover = previous_field
            .as_ref()
            .map_or(0.0, |previous| turnover(previous, &field));
        if previous_positions.is_some() {
            mobility_total += window_mobility;
            turnover_total += window_turnover;
            intervals += 1;
        }
        windows.push(WindowRecord {
            tick: tick + 1,
            spatial: spatial_samples.last().cloned().unwrap_or_default(),
            structure: structure(&normalized_rdf(&sample_rdf, &baseline)),
            heterogeneity: h,
            connectivity: c,
            mobility: window_mobility,
            turnover: window_turnover,
        });
        previous_positions = Some(world.substrate.positions.clone());
        previous_field = Some(field);
        samples += 1;
    }
    if samples == 0 {
        return RolloutOutcome {
            metrics: Metrics::default(),
            windows,
            early_stop: None,
        };
    }
    rdf.scale(1.0 / samples as f64);
    barcode_bins.iter_mut().for_each(|v| *v /= samples as f64);
    heterogeneity_total
        .iter_mut()
        .for_each(|v| *v /= samples as f32);
    dense.iter_mut().for_each(|v| *v /= samples as f32);
    void.iter_mut().for_each(|v| *v /= samples as f32);
    let mobility = mobility_total / intervals.max(1) as f32;
    let turnover = turnover_total / intervals.max(1) as f32;
    let robustness = self_maintenance(
        &world,
        &RepairProtocol {
            params,
            geometry,
            genome,
            baseline: &baseline,
            config: tuning.repair,
            steps: (steps / 4).max(50),
            shake: cutoff * tuning.repair.shake,
            seed: params.seed,
        },
    );
    let (temporal_variance, autocorrelation) = temporal(&spatial_samples);
    let heterogeneity = heterogeneity_total;
    let connectivity = Connectivity { dense, void };
    let descriptor = descriptor(&Observations {
        rdf: &rdf,
        rdf_baseline: &baseline,
        spatial_samples: &spatial_samples,
        heterogeneity,
        connectivity,
        barcode: &Barcode {
            bins: barcode_bins,
            components: ((barcode_components as f32 / samples as f32).round() as usize).max(1),
        },
        mobility,
        turnover,
        asymmetry: asymmetry(&world.net),
        robustness,
    });
    RolloutOutcome {
        metrics: Metrics {
            mobility,
            temporal_variance,
            autocorrelation,
            turnover,
            robustness,
            structure: structure(&normalized_rdf(&rdf, &baseline)),
            heterogeneity,
            connectivity,
            descriptor,
            alive: true,
        },
        windows,
        early_stop: None,
    }
}

fn self_maintenance(world: &World, protocol: &RepairProtocol<'_>) -> f32 {
    let extent = world.geometry.extent;
    let positions = world.substrate.positions.clone();
    let traits = world.substrate.traits.clone();
    let initial_control = recovery_signature(world, protocol.baseline);
    let mut control = World::from_snapshot(
        protocol.params,
        protocol.geometry,
        protocol.genome,
        positions.clone(),
        traits.clone(),
        world.tick,
    );
    for _ in 0..protocol.steps {
        control.step();
        if !finite(&control.substrate) {
            return 0.0;
        }
    }
    let future_control = recovery_signature(&control, protocol.baseline);
    let mut recovered = 0;
    for probe in 0..protocol.config.probes {
        let mut damaged = World::from_snapshot(
            protocol.params,
            protocol.geometry,
            protocol.genome,
            positions.clone(),
            traits.clone(),
            world.tick,
        );
        let mut rng = Rng::new(
            protocol
                .seed
                .wrapping_mul(0x9E37_79B9)
                .wrapping_add(probe as u64 + 1),
        );
        if damaged.substrate.is_empty() {
            continue;
        }
        let dims = damaged.substrate.dims();
        let centre_index = rng.below(damaged.substrate.len());
        let centre = damaged.substrate.at(centre_index).to_vec();
        for position in damaged.substrate.positions.chunks_exact_mut(dims) {
            let distance = crate::engine::kernel::displacement(position, &centre, extent, 1.0 / extent, 0.0);
            if distance <= (2.0 * protocol.shake).min(extent * 0.25) {
                for coordinate in position {
                    *coordinate = (*coordinate + protocol.shake * rng.normal()).rem_euclid(extent);
                }
            }
        }
        let damage = distance(
            &initial_control,
            &recovery_signature(&damaged, protocol.baseline),
        );
        for _ in 0..protocol.steps {
            damaged.step();
            if !finite(&damaged.substrate) {
                break;
            }
        }
        if damage >= protocol.config.min_effective_damage
            && finite(&damaged.substrate)
            && distance(
                &future_control,
                &recovery_signature(&damaged, protocol.baseline),
            ) < protocol.config.recovery * damage
        {
            recovered += 1;
        }
    }
    recovered as f32 / protocol.config.probes.max(1) as f32
}

fn recovery_signature(world: &World, baseline: &Rdf) -> Vec<f32> {
    let rdf = raw_rdf(&world.substrate, world.geometry.extent, world.geometry.softening);
    let local = heterogeneity(&world.substrate, world.geometry.extent);
    let connected = connectivity(&world.substrate, world.geometry.extent);
    scaled_spatial_features(&rdf, baseline, local, connected)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tuner::metrics::descriptor_len;

    fn small_params() -> Params {
        Params {
            particles: 120,
            coordination: 9.0,
            ..Default::default()
        }
    }

    fn small_tuning(particles: usize) -> Tuning {
        Tuning {
            world: crate::engine::genome::Caps {
                particles,
                ..Default::default()
            },
            ..Tuning::default()
        }
    }

    #[test]
    fn rollout_is_deterministic_and_v5_sized() {
        let params = small_params();
        let geometry = params.geometry();
        let genome = params
            .layout()
            .default_genome(params.radius, geometry.extent);
        let tuning = small_tuning(params.particles);
        let a = rollout(&tuning, &params, &geometry, &genome, 120);
        let b = rollout(&tuning, &params, &geometry, &genome, 120);
        assert_eq!(a.descriptor, b.descriptor);
        assert_eq!(a.descriptor.len(), descriptor_len());
        assert!(a.descriptor.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn tier_records_every_small_test_seed() {
        let tuning = small_tuning(80);
        let caps = &tuning.world;
        let scale = caps.geometry_scale();
        let budget = EvaluationBudget::new(EvaluationTier::Persistence, 80);
        let report =
            evaluate_tier_with_scale(&tuning, scale, &caps.default_genome(scale), budget);
        assert_eq!(report.records.len(), 3);
        assert!(report
            .records
            .iter()
            .all(|record| record.metrics.descriptor.is_empty()
                || record.metrics.descriptor.len() == descriptor_len()));
    }

    #[test]
    fn promotion_tiers_use_disjoint_seed_sets() {
        let base = 17;
        let discovery = (0..1)
            .map(|offset| tier_seed(base, EvaluationTier::Discovery, offset))
            .collect::<std::collections::HashSet<_>>();
        let persistence = (0..3)
            .map(|offset| tier_seed(base, EvaluationTier::Persistence, offset))
            .collect::<std::collections::HashSet<_>>();
        let certification = (0..5)
            .map(|offset| tier_seed(base, EvaluationTier::Certification, offset))
            .collect::<std::collections::HashSet<_>>();
        assert!(discovery.is_disjoint(&persistence));
        assert!(discovery.is_disjoint(&certification));
        assert!(persistence.is_disjoint(&certification));
    }

    #[test]
    fn ineffective_damage_never_counts_as_repair() {
        let params = small_params();
        let geometry = params.geometry();
        let genome = params
            .layout()
            .default_genome(params.radius, geometry.extent);
        let world = World::with_geometry(&params, &geometry, &genome);
        let baseline = raw_rdf(&world.substrate, world.geometry.extent, world.geometry.softening);
        assert_eq!(
            self_maintenance(
                &world,
                &RepairProtocol {
                    config: Repair::default(),
                    params: &params,
                    geometry: &geometry,
                    genome: &genome,
                    baseline: &baseline,
                    steps: 1,
                    shake: 0.0,
                    seed: params.seed,
                },
            ),
            0.0
        );
    }
}
