//! Append-only campaign evidence: one record per evaluated seed, with the numeric feature row
//! written at evaluation time so later model versions can never silently reinterpret old
//! observations. Failed and early-stopped runs are stored on purpose.

use crate::tuner::archive::{EvaluationRecord, ParentSource, SourceTier};
use crate::tuner::checkpoint::{GENOME_LAYOUT_VERSION, SIMULATOR_VERSION};
use crate::tuner::learning::{FeatureRow, FeatureSchema};
use crate::tuner::metrics::{descriptor_len, Metrics, DESCRIPTOR_VERSION};
use crate::tuner::novelty::neighborhood_key;
use crate::tuner::rollout::{EarlyStop, EvaluationBudget, EvaluationTier, TierEvaluation, WindowRecord};
use crate::tuner::search::PromotionRecord;
use crate::tuner::tuning::{Gates, Tuning};
use crate::tuner::viability::{margins, GateMargins, Rejection, WorldRejection};
use std::collections::BTreeMap;

pub const FEATURE_SCHEMA_VERSION: u32 = 1;

/// Exact experiment contract carried by durable campaign evidence. Search scheduling knobs are
/// deliberately absent: a resumed campaign may use a new search seed, batch size, generation
/// count, or archive capacity without mixing simulator or evaluation regimes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExperimentIdentity {
    pub simulator_version: u32,
    pub descriptor_version: u32,
    pub genome_layout_version: u32,
    pub feature_schema_version: u32,
    pub particle_count: usize,
    pub anchor_count: usize,
    pub radius_bits: u32,
    pub rate_bits: u32,
    pub evaluation_seed: u64,
    pub shells: usize,
    pub bumps: usize,
    pub discovery_steps: usize,
    pub tuning_digest: u64, // every gate, repair, and mutation setting folded into one value
}
impl ExperimentIdentity {
    pub fn for_tuning(tuning: &Tuning) -> Self {
        Self {
            simulator_version: SIMULATOR_VERSION,
            descriptor_version: DESCRIPTOR_VERSION,
            genome_layout_version: GENOME_LAYOUT_VERSION,
            feature_schema_version: FEATURE_SCHEMA_VERSION,
            particle_count: tuning.world.particle_count,
            anchor_count: tuning.world.anchor_count,
            radius_bits: tuning.world.radius.to_bits(),
            rate_bits: tuning.world.rate.to_bits(),
            evaluation_seed: tuning.world.seed,
            shells: tuning.world.shells,
            bumps: tuning.world.bumps,
            discovery_steps: tuning.discovery.steps,
            tuning_digest: tuning.digest(),
        }
    }
}

/// One complete seed result.
#[derive(Clone, Debug)]
pub struct CampaignRecord {
    pub evaluation_id: u64,
    pub genome_hash: u64, // individual identity for the exact genome
    pub lineage_id: u64, // evolutionary-family identity inherited by mutated descendants
    pub genome: Vec<f32>,
    pub source: ParentSource,
    pub source_tier: SourceTier,
    pub budget: EvaluationBudget,
    pub seed: u64,
    pub metrics: Metrics,
    pub feature_schema_version: u32,
    pub features: Vec<f32>,
    pub margins: GateMargins,
    pub base_gate: Result<(), Rejection>,
    pub world_gate: Result<(), WorldRejection>,
    pub early_stop: Option<EarlyStop>,
    pub windows: Vec<WindowRecord>,
    pub tier_passed: bool, // repeated on every seed of a requested tier, preserving the multi-seed label
}

/// A candidate carried between tiers: the genome plus the discovery evidence scheduling needs.
#[derive(Clone, Debug)]
pub struct CampaignCandidate {
    pub genome: Vec<f32>,
    pub genome_hash: u64,
    pub lineage_id: u64,
    pub source: ParentSource,
    pub descriptor: Vec<f32>,
    pub features: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct CampaignLedger {
    schema: FeatureSchema,
    records: Vec<CampaignRecord>,
    next_id: u64,
}
impl Default for CampaignLedger {
    fn default() -> Self { Self { schema: feature_schema(), records: Vec::new(), next_id: 1 } }
}
impl CampaignLedger {
    pub fn schema(&self) -> &FeatureSchema { &self.schema }
    pub fn records(&self) -> &[CampaignRecord] { &self.records }
    /// Rebuild from deserialized parts. The loader validates records before calling this.
    pub fn restore(records: Vec<CampaignRecord>, next_id: u64) -> Self {
        Self { schema: feature_schema(), records, next_id }
    }
    /// Imports the complete discovery ledger, including rejected candidates. Search records do
    /// not retain window snapshots, so those fields are empty instead of being fabricated.
    pub fn append_search_discovery(&mut self, record: &EvaluationRecord, discovery_steps: usize, gates: &Gates) {
        if self.records.iter().any(|existing| {
            existing.source_tier == SourceTier::Discovery
                && existing.genome_hash == record.genome_hash
                && existing.lineage_id == record.lineage_id
                && existing.seed == record.seed
                && existing.budget.steps == discovery_steps
        }) { return; } // resumed batches replay identical discoveries; keep one
        let margins = margins(&record.metrics, gates);
        let features = feature_values(&record.metrics, margins, &[], None);
        let id = self.next_id;
        self.next_id += 1;
        self.records.push(CampaignRecord {
            evaluation_id: id,
            genome_hash: record.genome_hash,
            lineage_id: record.lineage_id,
            genome: Vec::new(),
            source: record.parent_source,
            source_tier: SourceTier::Discovery,
            budget: EvaluationBudget::new(EvaluationTier::Discovery, discovery_steps),
            seed: record.seed,
            metrics: record.metrics.clone(),
            feature_schema_version: FEATURE_SCHEMA_VERSION,
            features, margins,
            base_gate: record.gate,
            world_gate: Ok(()),
            early_stop: None,
            windows: Vec::new(),
            tier_passed: record.gate.is_ok(),
        });
    }
    pub fn append_tier(&mut self, candidate: &CampaignCandidate, evaluation: TierEvaluation) -> Vec<u64> {
        let passed = evaluation.passed();
        let mut ids = Vec::with_capacity(evaluation.records.len());
        for seed in evaluation.records {
            let id = self.next_id;
            self.next_id += 1;
            let features = feature_values(&seed.metrics, seed.margins, &seed.windows, seed.early_stop);
            self.records.push(CampaignRecord {
                evaluation_id: id,
                genome_hash: candidate.genome_hash,
                lineage_id: candidate.lineage_id,
                genome: candidate.genome.clone(),
                source: candidate.source,
                source_tier: source_tier(evaluation.budget.tier),
                budget: evaluation.budget,
                seed: seed.seed,
                metrics: seed.metrics,
                feature_schema_version: FEATURE_SCHEMA_VERSION,
                features,
                margins: seed.margins,
                base_gate: seed.base,
                world_gate: seed.world,
                early_stop: seed.early_stop,
                windows: seed.windows,
                tier_passed: passed,
            });
            ids.push(id);
        }
        ids
    }
    /// Labels come solely from canonical-or-longer persistence rollouts. A discovery feature
    /// row appears once per (genome, evolutionary family) pair, and repeated persistence
    /// attempts combine conservatively: one failed tier makes the label false.
    pub fn persistence_rows(&self) -> Vec<FeatureRow> {
        let mut discovery = BTreeMap::new();
        let mut labels = BTreeMap::new();
        for record in &self.records {
            let key = (record.genome_hash, record.lineage_id);
            match record.source_tier {
                SourceTier::Discovery => {
                    if budget_valid_for_source(record.budget, SourceTier::Discovery) {
                        discovery.entry(key).or_insert(record);
                    }
                }
                SourceTier::Persistence => {
                    if budget_valid_for_source(record.budget, SourceTier::Persistence) {
                        labels.entry(key).and_modify(|survives| *survives &= record.tier_passed).or_insert(record.tier_passed);
                    }
                }
                SourceTier::Certification => {}
            }
        }
        discovery.into_iter().filter_map(|(key, record)| {
            labels.get(&key).map(|survives| FeatureRow {
                id: record.evaluation_id,
                lineage_id: record.lineage_id,
                neighborhood: neighborhood_key(&record.metrics.descriptor),
                features: record.features.clone(),
                survives: *survives,
            })
        }).collect()
    }
    pub fn discovery_features(&self, record: &PromotionRecord) -> Option<Vec<f32>> {
        let key = (record.genome_hash, record.lineage_id);
        self.records.iter()
            .find(|entry| entry.source_tier == SourceTier::Discovery && (entry.genome_hash, entry.lineage_id) == key)
            .map(|entry| entry.features.clone())
    }
}

/// Whether a recorded budget still counts as evidence for its tier.
pub fn budget_valid_for_source(budget: EvaluationBudget, source: SourceTier) -> bool {
    budget.tier == evaluation_tier(source) && budget.honors_tier()
}

pub fn source_tier(tier: EvaluationTier) -> SourceTier {
    match tier {
        EvaluationTier::Discovery => SourceTier::Discovery,
        EvaluationTier::Persistence => SourceTier::Persistence,
        EvaluationTier::Certification => SourceTier::Certification,
    }
}

pub fn evaluation_tier(source: SourceTier) -> EvaluationTier {
    match source {
        SourceTier::Discovery => EvaluationTier::Discovery,
        SourceTier::Persistence => EvaluationTier::Persistence,
        SourceTier::Certification => EvaluationTier::Certification,
    }
}

pub fn feature_schema() -> FeatureSchema {
    let mut names = (0..descriptor_len())
        .map(|index| format!("v{FEATURE_SCHEMA_VERSION}.descriptor.{index}")).collect::<Vec<_>>();
    names.extend([
        "structure", "mobility", "temporal_variance", "turnover", "robustness",
        "margin_structure_floor", "margin_structure_ceiling", "margin_mobility", "margin_temporal",
        "margin_repair", "margin_heterogeneity", "margin_material", "margin_void",
        "window_count", "window_structure_drift", "window_mobility_mean", "window_turnover_mean",
        "early_stop",
    ].map(str::to_owned));
    FeatureSchema::new(names)
}

/// One fixed-width numeric row per seed. Early stops carry Metrics::default with an empty
/// descriptor; their failure label is preserved rather than silently dropped from training.
pub fn feature_values(metrics: &Metrics, margins: GateMargins, windows: &[WindowRecord],
    early_stop: Option<EarlyStop>) -> Vec<f32>
{
    let mut values = metrics.descriptor.clone();
    values.truncate(descriptor_len());
    values.resize(descriptor_len(), 0.0);
    values.extend([
        metrics.structure, metrics.mobility, metrics.temporal_variance, metrics.turnover, metrics.robustness,
        margins.structure_floor, margins.structure_ceiling, margins.mobility, margins.temporal,
        margins.repair, margins.heterogeneity, margins.material, margins.void,
        windows.len() as f32,
        windows.first().zip(windows.last()).map_or(0.0, |(first, last)| last.structure - first.structure),
        windows.iter().map(|window| window.mobility).sum::<f32>() / windows.len().max(1) as f32,
        windows.iter().map(|window| window.turnover).sum::<f32>() / windows.len().max(1) as f32,
        f32::from(early_stop.is_some()),
    ]);
    values
}
