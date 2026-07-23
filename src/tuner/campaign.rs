//! Campaign orchestration for costly, evidence-backed world promotion.
//!
//! Discovery remains novelty plus the binary viability gate.  This module records the
//! evidence needed to learn where to spend later rollouts; it never changes archive
//! admission, viability, or the definition of a certified preset.

use crate::tuner::archive::{Archive, EvaluationRecord, ParentSource, SourceTier};
use crate::tuner::checkpoint::{save_world, snapshot_world, SnapshotMetadata, WorldManifest, WorldState};
use crate::tuner::learning::{
    fit_persistence, lineage_split, select_continuations, ContinuationCandidate, FeatureRow,
    FeatureSchema, PersistenceEnsemble,
};
use crate::tuner::metrics::{descriptor_len, Metrics, DESCRIPTOR_VERSION};
use crate::tuner::novelty::neighborhood_key;
use crate::tuner::rollout::{
    evaluate_tier_with_scale, EarlyStop, EvaluationBudget, EvaluationTier, TierEvaluation,
    WindowRecord,
};
use crate::tuner::search::{run_with_promotions, PromotionRecord, Report, SearchResult};
use crate::tuner::learning::ContinuationQuotas;
use crate::tuner::tuning::{Gates, Learning, Tuning};
use crate::tuner::viability::{margins, GateMargins, Rejection, WorldRejection};
use crate::engine::world::World;
use crate::engine::genome::Caps;
use crate::engine::geometry::GeometryScale;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

pub use crate::tuner::checkpoint::{GENOME_LAYOUT_VERSION, SIMULATOR_VERSION};
pub use crate::render_recipe::VERSION as RENDER_RECIPE_VERSION;

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
    pub particles: usize,
    pub anchors: usize,
    pub radius_bits: u32,
    pub rate_bits: u32,
    pub evaluation_seed: u64,
    pub shells: usize,
    pub bumps: usize,
    pub discovery_steps: usize,
    /// Every gate, repair, and mutation setting folded into one value. Two campaigns that
    /// measure or judge differently cannot share a ledger, even when their worlds match.
    pub tuning_digest: u64,
}

impl ExperimentIdentity {
    pub fn for_tuning(tuning: &Tuning) -> Self {
        Self {
            simulator_version: SIMULATOR_VERSION,
            descriptor_version: DESCRIPTOR_VERSION,
            genome_layout_version: GENOME_LAYOUT_VERSION,
            feature_schema_version: FEATURE_SCHEMA_VERSION,
            particles: tuning.world.particles,
            anchors: tuning.world.anchors,
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

/// One complete seed result.  The numeric row is written at evaluation time so later model
/// versions can never silently reinterpret old observations.
#[derive(Clone, Debug)]
pub struct CampaignRecord {
    pub evaluation_id: u64,
    /// Individual identity for the exact genome.
    pub genome_hash: u64,
    /// Evolutionary-family identity inherited by mutated descendants.
    pub lineage_id: u64,
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
    /// Repeated on every seed of a requested tier, preserving the tier's multi-seed label.
    pub tier_passed: bool,
}

/// Append-only campaign evidence.  It intentionally stores failed and early-stopped runs.
#[derive(Clone, Debug)]
pub struct CampaignLedger {
    schema: FeatureSchema,
    records: Vec<CampaignRecord>,
    next_id: u64,
}

impl Default for CampaignLedger {
    fn default() -> Self {
        Self {
            schema: feature_schema(),
            records: Vec::new(),
            next_id: 1,
        }
    }
}

impl CampaignLedger {
    pub fn schema(&self) -> &FeatureSchema {
        &self.schema
    }

    pub fn records(&self) -> &[CampaignRecord] {
        &self.records
    }

    /// Imports the complete discovery ledger, including rejected candidates. Search records do
    /// not retain window snapshots, so those fields are empty instead of being fabricated.
    pub fn append_search_discovery(
        &mut self,
        record: &EvaluationRecord,
        discovery_steps: usize,
        gates: &Gates,
    ) {
        if self.records.iter().any(|existing| {
            existing.source_tier == SourceTier::Discovery
                && existing.genome_hash == record.genome_hash
                && existing.lineage_id == record.lineage_id
                && existing.seed == record.seed
                && existing.budget.steps == discovery_steps
        }) {
            return;
        }
        let margins = margins(&record.metrics, gates);
        let features = feature_values(&record.metrics, margins, &[], None);
        let budget = EvaluationBudget::new(EvaluationTier::Discovery, discovery_steps);
        let id = self.next_id;
        self.next_id += 1;
        self.records.push(CampaignRecord {
            evaluation_id: id,
            genome_hash: record.genome_hash,
            lineage_id: record.lineage_id,
            genome: Vec::new(),
            source: record.parent_source,
            source_tier: SourceTier::Discovery,
            budget,
            seed: record.seed,
            metrics: record.metrics.clone(),
            feature_schema_version: FEATURE_SCHEMA_VERSION,
            features,
            margins,
            base_gate: record.gate,
            world_gate: Ok(()),
            early_stop: None,
            windows: Vec::new(),
            tier_passed: record.gate.is_ok(),
        });
    }

    pub(crate) fn append_tier(
        &mut self,
        candidate: &CampaignCandidate,
        evaluation: TierEvaluation,
    ) -> Vec<u64> {
        let passed = evaluation.passed();
        let mut ids = Vec::with_capacity(evaluation.records.len());
        for seed in evaluation.records {
            let id = self.next_id;
            self.next_id += 1;
            let features =
                feature_values(&seed.metrics, seed.margins, &seed.windows, seed.early_stop);
            debug_assert!(self.schema.accepts(&features));
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

    /// Labels are derived solely from canonical-or-longer persistence rollouts. A discovery
    /// feature row appears once per `(genome, evolutionary family)` pair. Repeated persistence
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
                        labels
                            .entry(key)
                            .and_modify(|survives| *survives &= record.tier_passed)
                            .or_insert(record.tier_passed);
                    }
                }
                SourceTier::Certification => {}
            }
        }
        discovery
            .into_iter()
            .filter_map(|(key, record)| {
                labels.get(&key).map(|survives| FeatureRow {
                    id: record.evaluation_id,
                    lineage_id: record.lineage_id,
                    neighborhood: neighborhood_key(&record.metrics.descriptor),
                    features: record.features.clone(),
                    survives: *survives,
                })
            })
            .collect()
    }

    fn discovery_features(&self, record: &PromotionRecord) -> Option<Vec<f32>> {
        let key = (record.genome_hash, record.lineage_id);
        self.records
            .iter()
            .find(|entry| {
                entry.source_tier == SourceTier::Discovery
                    && (entry.genome_hash, entry.lineage_id) == key
            })
            .map(|entry| entry.features.clone())
    }
}

#[derive(Clone, Debug)]
pub(crate) struct CampaignCandidate {
    genome: Vec<f32>,
    genome_hash: u64,
    lineage_id: u64,
    source: ParentSource,
    descriptor: Vec<f32>,
    features: Vec<f32>,
}

/// Durable output paths; absent paths keep the corresponding evidence in memory only.
#[derive(Clone, Debug, Default)]
pub struct CampaignPersistence {
    pub archive_path: Option<PathBuf>,
    pub state_path: Option<PathBuf>,
    pub preset_dir: Option<PathBuf>,
}

#[derive(Debug)]
pub enum CampaignError {
    InvalidConfiguration(String),
    ExperimentMismatch,
    State(CampaignStateError),
    Archive { path: PathBuf, source: io::Error },
    Checkpoint(crate::tuner::checkpoint::Error),
}

impl fmt::Display for CampaignError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfiguration(problem) => write!(f, "{problem}"),
            Self::ExperimentMismatch => {
                write!(f, "campaign state belongs to a different experiment")
            }
            Self::State(problem) => write!(f, "{problem}"),
            Self::Archive { path, source } => {
                write!(f, "could not save archive {}: {source}", path.display())
            }
            Self::Checkpoint(problem) => write!(f, "could not save certified preset: {problem}"),
        }
    }
}

impl std::error::Error for CampaignError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::State(problem) => Some(problem),
            Self::Archive { source, .. } => Some(source),
            Self::Checkpoint(problem) => Some(problem),
            Self::InvalidConfiguration(_) | Self::ExperimentMismatch => None,
        }
    }
}

impl From<CampaignStateError> for CampaignError {
    fn from(problem: CampaignStateError) -> Self {
        Self::State(problem)
    }
}

impl From<crate::tuner::checkpoint::Error> for CampaignError {
    fn from(problem: crate::tuner::checkpoint::Error) -> Self {
        Self::Checkpoint(problem)
    }
}

/// Validate the evidence contract before any rollout work begins.
pub fn validate(tuning: &Tuning) -> Result<(), CampaignError> {
    let check = |budget: EvaluationBudget| {
        if budget.honours_tier() {
            Ok(())
        } else {
            Err(CampaignError::InvalidConfiguration(format!(
                "{} requires at least {} steps",
                budget.tier.label(),
                budget.tier.minimum_steps()
            )))
        }
    };
    check(EvaluationBudget::new(
        EvaluationTier::Discovery,
        tuning.discovery.steps,
    ))?;
    if let Some(steps) = tuning.tiers.persistence_steps {
        check(EvaluationBudget::new(EvaluationTier::Persistence, steps))?;
    }
    if let Some(steps) = tuning.tiers.certification_steps {
        if tuning.tiers.persistence_steps.is_none() {
            return Err(CampaignError::InvalidConfiguration(
                "certification requires persistence evaluation".into(),
            ));
        }
        check(EvaluationBudget::new(EvaluationTier::Certification, steps))?;
    }
    Ok(())
}

/// Whether a recorded budget still counts as evidence for its tier.
fn budget_valid_for_source(budget: EvaluationBudget, source: SourceTier) -> bool {
    budget.tier == evaluation_tier(source) && budget.honours_tier()
}

#[derive(Clone, Debug)]
pub struct PersistenceModel {
    pub model: Option<PersistenceEnsemble>,
    pub rows: usize,
}

impl PersistenceModel {
    pub fn authority_active(&self) -> bool {
        self.model
            .as_ref()
            .is_some_and(|model| model.authority.active)
    }
}

/// The reproducible state recipe belonging to a real certification outcome.
#[derive(Clone, Debug)]
pub struct CertifiedPreset {
    pub world_id: String,
    pub genome_hash: u64,
    pub lineage_id: u64,
    pub manifest: WorldManifest,
    pub state: WorldState,
}

/// Durable in-memory campaign state. Keep this object between search batches so persistence
/// labels eventually meet the lineage-held-out training threshold.
#[derive(Clone, Debug, Default)]
pub struct CampaignState {
    identity: Option<ExperimentIdentity>,
    pub ledger: CampaignLedger,
}

const STATE_MAGIC: &[u8; 8] = b"AXIOMCP1";
const STATE_VERSION: u32 = 4;

#[derive(Debug)]
pub enum CampaignStateError {
    Io(io::Error),
    InvalidFormat(&'static str),
    CorruptChecksum { expected: u64, actual: u64 },
}

impl fmt::Display for CampaignStateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "campaign state I/O: {error}"),
            Self::InvalidFormat(field) => write!(f, "invalid campaign state {field}"),
            Self::CorruptChecksum { expected, actual } => write!(
                f,
                "campaign state checksum {actual:016x} does not match {expected:016x}"
            ),
        }
    }
}

impl std::error::Error for CampaignStateError {}

impl From<io::Error> for CampaignStateError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl CampaignState {
    pub fn for_tuning(tuning: &Tuning) -> Self {
        Self {
            identity: Some(ExperimentIdentity::for_tuning(tuning)),
            ..Self::default()
        }
    }

    pub fn identity(&self) -> Option<&ExperimentIdentity> {
        self.identity.as_ref()
    }

    fn bind(&mut self, identity: ExperimentIdentity) -> Result<(), CampaignError> {
        match &self.identity {
            Some(bound) if bound != &identity => Err(CampaignError::ExperimentMismatch),
            Some(_) => Ok(()),
            None => {
                self.identity = Some(identity);
                Ok(())
            }
        }
    }

    /// Lossless durable state for long-running campaigns. Floats are written as raw bits,
    /// preserving feature rows and checkpoint scheduling decisions exactly across restarts.
    pub fn save(&self, path: &Path) -> Result<(), CampaignStateError> {
        let identity = self
            .identity
            .as_ref()
            .ok_or(CampaignStateError::InvalidFormat("experiment identity"))?;
        let mut out = Vec::new();
        out.extend_from_slice(STATE_MAGIC);
        put_u32(&mut out, STATE_VERSION);
        put_identity(&mut out, identity);
        put_u64(&mut out, self.ledger.next_id);
        put_u64(&mut out, self.ledger.records.len() as u64);
        for record in &self.ledger.records {
            put_record(&mut out, record);
        }
        let checksum = campaign_checksum(&out);
        put_u64(&mut out, checksum);
        write_atomic(path, &out).map_err(CampaignStateError::Io)
    }

    pub fn load(path: &Path) -> Result<Self, CampaignStateError> {
        let bytes = fs::read(path)?;
        let checksum_at = bytes
            .len()
            .checked_sub(8)
            .ok_or(CampaignStateError::InvalidFormat("checksum"))?;
        let expected = u64::from_le_bytes(
            bytes[checksum_at..]
                .try_into()
                .map_err(|_| CampaignStateError::InvalidFormat("checksum"))?,
        );
        let actual = campaign_checksum(&bytes[..checksum_at]);
        if expected != actual {
            return Err(CampaignStateError::CorruptChecksum { expected, actual });
        }
        let mut reader = StateReader::new(&bytes[..checksum_at]);
        if reader.take(8)? != STATE_MAGIC {
            return Err(CampaignStateError::InvalidFormat("magic"));
        }
        if reader.u32()? != STATE_VERSION {
            return Err(CampaignStateError::InvalidFormat("version"));
        }
        let identity = reader.identity()?;
        let next_id = reader.u64()?;
        let records = (0..reader.len()?)
            .map(|_| reader.record())
            .collect::<Result<Vec<_>, _>>()?;
        if !reader.finished() {
            return Err(CampaignStateError::InvalidFormat("trailing bytes"));
        }
        if records.iter().any(|record| {
            record.feature_schema_version != FEATURE_SCHEMA_VERSION
                || !feature_schema().accepts(&record.features)
                || !budget_valid_for_source(record.budget, record.source_tier)
        }) {
            return Err(CampaignStateError::InvalidFormat(
                "feature schema or evaluation budget",
            ));
        }
        Ok(Self {
            identity: Some(identity),
            ledger: CampaignLedger {
                schema: feature_schema(),
                records,
                next_id,
            },
        })
    }
}

pub struct CampaignRun {
    pub search: SearchResult,
    pub persistence_model: Option<PersistenceModel>,
    pub certifications: Vec<CertifiedPreset>,
}

/// Run one campaign: discovery search, then whichever expensive tiers `tuning.tiers` enables.
///
/// This is the only entry point. Everything a run can vary lives in `tuning`; `paths` decides
/// only what gets written to disk. Resuming against prior evidence is the same call with a
/// loaded `state`: search batches are independent while the ledger stays append-only.
pub fn run(
    tuning: &Tuning,
    paths: &CampaignPersistence,
    state: &mut CampaignState,
    on_generation: &mut impl FnMut(usize, &Archive, &Report),
) -> Result<CampaignRun, CampaignError> {
    validate(tuning)?;
    state.bind(ExperimentIdentity::for_tuning(tuning))?;
    let mut callback = |generation, archive: &Archive, report: &Report, _: &_| {
        on_generation(generation, archive, report)
    };
    let search_result = run_with_promotions(tuning, &mut callback);
    let mut candidates = Vec::new();
    let scale = tuning.world.geometry_scale();

    for record in search_result.ledger.records() {
        state
            .ledger
            .append_search_discovery(record, tuning.discovery.steps, &tuning.gates);
    }

    let archive_result = persist_archive(paths, &search_result.archive);
    let state_result = persist_state(paths, state);
    archive_result?;
    state_result?;

    let already_persisted = state
        .ledger
        .records()
        .iter()
        .filter(|record| record.source_tier == SourceTier::Persistence)
        .map(|record| (record.genome_hash, record.lineage_id))
        .collect::<BTreeSet<_>>();

    for promotion in search_result.promotions.records() {
        if already_persisted.contains(&(promotion.genome_hash, promotion.lineage_id)) {
            continue;
        }
        if let Some(features) = state.ledger.discovery_features(promotion) {
            candidates.push(CampaignCandidate {
                genome: promotion.genome.clone(),
                genome_hash: promotion.genome_hash,
                lineage_id: promotion.lineage_id,
                source: promotion.source,
                descriptor: promotion.descriptor.clone(),
                features,
            });
        }
    }

    let mut certifications = Vec::new();
    let Some(persistence_budget) = tuning
        .tiers
        .persistence_steps
        .map(|steps| EvaluationBudget::new(EvaluationTier::Persistence, steps))
    else {
        return Ok(CampaignRun {
            search: search_result,
            persistence_model: None,
            certifications,
        });
    };
    let model = train_persistence(&state.ledger, &tuning.learning);
    let scheduled = schedule(&candidates, &model, tuning.learning.quotas);
    for candidate in scheduled {
        let evaluation =
            evaluate_tier_with_scale(tuning, scale, &candidate.genome, persistence_budget);
        state.ledger.append_tier(&candidate, evaluation);
        persist_state(paths, state)?;
    }
    let model = train_persistence(&state.ledger, &tuning.learning);

    if let Some(certification_budget) = tuning
        .tiers
        .certification_steps
        .map(|steps| EvaluationBudget::new(EvaluationTier::Certification, steps))
    {
        let promoted = passing_persistence_candidates(&state.ledger);
        for candidate in promoted {
            let evaluation =
                evaluate_tier_with_scale(tuning, scale, &candidate.genome, certification_budget);
            let passed = evaluation.passed();
            let passing_seed = evaluation
                .records
                .iter()
                .find(|record| record.base.is_ok() && record.world.is_ok())
                .map(|record| record.seed);
            state.ledger.append_tier(&candidate, evaluation);
            persist_state(paths, state)?;
            if passed {
                let preset = certified_preset(
                    &tuning.world,
                    scale,
                    &candidate,
                    certification_budget,
                    passing_seed.expect("a passed tier has a passing seed"),
                )?;
                if let Some(root) = &paths.preset_dir {
                    save_world(root, &preset.manifest, &preset.state)?;
                }
                certifications.push(preset);
            }
        }
    }
    Ok(CampaignRun {
        search: search_result,
        persistence_model: Some(model),
        certifications,
    })
}

fn persist_archive(paths: &CampaignPersistence, archive: &Archive) -> Result<(), CampaignError> {
    let Some(path) = &paths.archive_path else {
        return Ok(());
    };
    write_atomic(path, archive.to_text().as_bytes()).map_err(|source| CampaignError::Archive {
        path: path.clone(),
        source,
    })
}

fn persist_state(paths: &CampaignPersistence, state: &CampaignState) -> Result<(), CampaignError> {
    if let Some(path) = &paths.state_path {
        state.save(path)?;
    }
    Ok(())
}

/// Fit only after enough outcomes exist.  The model may be present yet unauthoritative; the
/// held-out authority report remains the only switch that changes scheduling.
pub fn train_persistence(ledger: &CampaignLedger, config: &Learning) -> PersistenceModel {
    let rows = ledger.persistence_rows();
    if rows.len() < config.minimum_labeled_rows {
        return PersistenceModel {
            model: None,
            rows: rows.len(),
        };
    }
    let split = lineage_split(&rows, config.model_seed);
    let model = fit_persistence(
        ledger.schema().clone(),
        &rows,
        &split,
        config.logistic,
        config.model_seed,
    )
    .ok();
    PersistenceModel {
        model,
        rows: rows.len(),
    }
}

/// Return the candidates chosen under the named quotas.  No model means a deterministic,
/// neighborhood-uniform fallback.
pub(crate) fn schedule(
    candidates: &[CampaignCandidate],
    model: &PersistenceModel,
    quotas: ContinuationQuotas,
) -> Vec<CampaignCandidate> {
    let authority = model.authority_active();
    let ranked = candidates
        .iter()
        .map(|candidate| {
            let prediction = model
                .model
                .as_ref()
                .and_then(|model| model.predict(&candidate.features).ok())
                .unwrap_or(crate::tuner::learning::PersistencePrediction {
                    probability: 0.5,
                    uncertainty: 0.0,
                });
            ContinuationCandidate {
                id: candidate.genome_hash,
                neighborhood: neighborhood_key(&candidate.descriptor),
                prediction,
            }
        })
        .collect::<Vec<_>>();
    let selected = select_continuations(&ranked, quotas, authority);
    selected
        .into_iter()
        .filter_map(|id| {
            candidates
                .iter()
                .find(|candidate| candidate.genome_hash == id)
        })
        .cloned()
        .collect()
}

fn passing_persistence_candidates(ledger: &CampaignLedger) -> Vec<CampaignCandidate> {
    struct Aggregate {
        candidate: CampaignCandidate,
        all_passed: bool,
    }

    let certification_evaluated = ledger
        .records()
        .iter()
        .filter(|record| record.source_tier == SourceTier::Certification)
        .map(|record| (record.genome_hash, record.lineage_id))
        .collect::<BTreeSet<_>>();
    let mut candidates: BTreeMap<(u64, u64), Aggregate> = BTreeMap::new();
    for record in ledger.records() {
        let key = (record.genome_hash, record.lineage_id);
        if record.source_tier == SourceTier::Persistence
            && budget_valid_for_source(record.budget, SourceTier::Persistence)
            && !certification_evaluated.contains(&key)
        {
            candidates
                .entry(key)
                .and_modify(|aggregate| aggregate.all_passed &= record.tier_passed)
                .or_insert_with(|| Aggregate {
                    candidate: CampaignCandidate {
                        genome: record.genome.clone(),
                        genome_hash: record.genome_hash,
                        lineage_id: record.lineage_id,
                        source: record.source,
                        descriptor: record.metrics.descriptor.clone(),
                        features: record.features.clone(),
                    },
                    all_passed: record.tier_passed,
                });
        }
    }
    candidates
        .into_values()
        .filter(|aggregate| aggregate.all_passed)
        .map(|aggregate| aggregate.candidate)
        .collect()
}

fn certified_preset(
    caps: &Caps,
    scale: GeometryScale,
    candidate: &CampaignCandidate,
    budget: EvaluationBudget,
    passing_seed: u64,
) -> Result<CertifiedPreset, crate::tuner::checkpoint::Error> {
    let (mut params, interactions) = caps.resolve(&candidate.genome);
    params.seed = passing_seed;
    let geometry = params.geometry_with(scale);
    let mut world = World::with_geometry(&params, &geometry, interactions);
    for _ in 0..budget.steps {
        world.step();
    }
    let world_id = format!("world-{:016x}", candidate.genome_hash);
    let checkpoint_id = format!("certified-{}", budget.steps);
    let mut checkpoint_caps = caps.clone();
    checkpoint_caps.seed = passing_seed;
    let mut render_recipe = crate::render_recipe::RenderRecipe::default();
    render_recipe.support = render_recipe.support.min(geometry.extent * 0.25);
    let (manifest, state) = snapshot_world(
        SnapshotMetadata {
            world_id: world_id.clone(),
            checkpoint_id,
            parent_checkpoint_id: None,
            simulator_version: SIMULATOR_VERSION,
            descriptor_version: DESCRIPTOR_VERSION,
            genome_layout_version: GENOME_LAYOUT_VERSION,
            render_recipe_version: RENDER_RECIPE_VERSION,
            render_recipe,
        },
        &checkpoint_caps,
        candidate.genome.clone(),
        &world,
    )?;
    Ok(CertifiedPreset {
        world_id,
        genome_hash: candidate.genome_hash,
        lineage_id: candidate.lineage_id,
        manifest,
        state,
    })
}

fn put_identity(out: &mut Vec<u8>, identity: &ExperimentIdentity) {
    for version in [
        identity.simulator_version,
        identity.descriptor_version,
        identity.genome_layout_version,
        identity.feature_schema_version,
    ] {
        put_u32(out, version);
    }
    put_u64(out, identity.particles as u64);
    put_u64(out, identity.anchors as u64);
    put_u32(out, identity.radius_bits);
    put_u32(out, identity.rate_bits);
    put_u64(out, identity.evaluation_seed);
    put_u64(out, identity.shells as u64);
    put_u64(out, identity.bumps as u64);
    put_u64(out, identity.discovery_steps as u64);
    put_u64(out, identity.tuning_digest);
}

fn put_record(out: &mut Vec<u8>, record: &CampaignRecord) {
    put_u64(out, record.evaluation_id);
    put_u64(out, record.genome_hash);
    put_u64(out, record.lineage_id);
    put_f32s(out, &record.genome);
    out.push(parent_source_code(record.source));
    out.push(source_tier_code(record.source_tier));
    out.push(evaluation_tier_code(record.budget.tier));
    put_u64(out, record.budget.steps as u64);
    put_u64(out, record.seed);
    put_metrics(out, &record.metrics);
    put_u32(out, record.feature_schema_version);
    put_f32s(out, &record.features);
    put_margins(out, record.margins);
    out.push(rejection_code(record.base_gate));
    out.push(world_rejection_code(record.world_gate));
    out.push(early_stop_code(record.early_stop));
    out.push(u8::from(record.tier_passed));
    put_u64(out, record.windows.len() as u64);
    for window in &record.windows {
        put_u64(out, window.tick as u64);
        put_f32s(out, &window.spatial);
        put_f32(out, window.structure);
        for value in window.heterogeneity {
            put_f32(out, value);
        }
        for value in window.connectivity.dense {
            put_f32(out, value);
        }
        for value in window.connectivity.void {
            put_f32(out, value);
        }
        put_f32(out, window.mobility);
        put_f32(out, window.turnover);
    }
}

fn put_metrics(out: &mut Vec<u8>, metrics: &Metrics) {
    for value in [
        metrics.mobility,
        metrics.temporal_variance,
        metrics.turnover,
        metrics.robustness,
        metrics.structure,
    ] {
        put_f32(out, value);
    }
    for value in metrics.autocorrelation {
        put_f32(out, value);
    }
    for value in metrics.heterogeneity {
        put_f32(out, value);
    }
    for value in metrics.connectivity.dense {
        put_f32(out, value);
    }
    for value in metrics.connectivity.void {
        put_f32(out, value);
    }
    put_f32s(out, &metrics.descriptor);
    out.push(u8::from(metrics.alive));
}

fn put_margins(out: &mut Vec<u8>, margins: GateMargins) {
    for value in [
        margins.structure_floor,
        margins.structure_ceiling,
        margins.mobility,
        margins.temporal,
        margins.repair,
        margins.heterogeneity,
        margins.material,
        margins.void,
    ] {
        put_f32(out, value);
    }
}

fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_f32(out: &mut Vec<u8>, value: f32) {
    put_u32(out, value.to_bits());
}

fn put_f32s(out: &mut Vec<u8>, values: &[f32]) {
    put_u64(out, values.len() as u64);
    for value in values {
        put_f32(out, *value);
    }
}

fn campaign_checksum(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ *byte as u64).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn write_atomic(path: &Path, bytes: &[u8]) -> io::Result<()> {
    // A bare filename yields `Some("")`, which is not a directory anything can be created in.
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("campaign");
    let temporary = parent.join(format!(".{name}.tmp-{}", std::process::id()));
    let result = (|| {
        let mut file = fs::File::create(&temporary)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        fs::rename(&temporary, path)?;
        fs::File::open(parent)?.sync_all()?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

struct StateReader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> StateReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, at: 0 }
    }

    fn take(&mut self, length: usize) -> Result<&'a [u8], CampaignStateError> {
        let end = self
            .at
            .checked_add(length)
            .filter(|end| *end <= self.bytes.len())
            .ok_or(CampaignStateError::InvalidFormat("truncated"))?;
        let slice = &self.bytes[self.at..end];
        self.at = end;
        Ok(slice)
    }

    fn byte(&mut self) -> Result<u8, CampaignStateError> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32, CampaignStateError> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("four bytes"),
        ))
    }

    fn u64(&mut self) -> Result<u64, CampaignStateError> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("eight bytes"),
        ))
    }

    fn usize(&mut self) -> Result<usize, CampaignStateError> {
        usize::try_from(self.u64()?).map_err(|_| CampaignStateError::InvalidFormat("length"))
    }

    fn len(&mut self) -> Result<usize, CampaignStateError> {
        let length = self.usize()?;
        if length > 1 << 28 {
            return Err(CampaignStateError::InvalidFormat("length"));
        }
        Ok(length)
    }

    fn f32(&mut self) -> Result<f32, CampaignStateError> {
        Ok(f32::from_bits(self.u32()?))
    }

    fn f32s(&mut self) -> Result<Vec<f32>, CampaignStateError> {
        (0..self.len()?)
            .map(|_| self.f32())
            .collect::<Result<Vec<_>, _>>()
    }

    fn f32_array<const N: usize>(&mut self) -> Result<[f32; N], CampaignStateError> {
        let values = (0..N).map(|_| self.f32()).collect::<Result<Vec<_>, _>>()?;
        values
            .try_into()
            .map_err(|_| CampaignStateError::InvalidFormat("fixed float array"))
    }

    fn metrics(&mut self) -> Result<Metrics, CampaignStateError> {
        let mobility = self.f32()?;
        let temporal_variance = self.f32()?;
        let turnover = self.f32()?;
        let robustness = self.f32()?;
        let structure = self.f32()?;
        Ok(Metrics {
            mobility,
            temporal_variance,
            turnover,
            robustness,
            structure,
            autocorrelation: self.f32_array()?,
            heterogeneity: self.f32_array()?,
            connectivity: crate::tuner::metrics::Connectivity {
                dense: self.f32_array()?,
                void: self.f32_array()?,
            },
            descriptor: self.f32s()?,
            alive: self.bool()?,
        })
    }

    fn margins(&mut self) -> Result<GateMargins, CampaignStateError> {
        Ok(GateMargins {
            structure_floor: self.f32()?,
            structure_ceiling: self.f32()?,
            mobility: self.f32()?,
            temporal: self.f32()?,
            repair: self.f32()?,
            heterogeneity: self.f32()?,
            material: self.f32()?,
            void: self.f32()?,
        })
    }

    fn bool(&mut self) -> Result<bool, CampaignStateError> {
        match self.byte()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(CampaignStateError::InvalidFormat("boolean")),
        }
    }

    fn identity(&mut self) -> Result<ExperimentIdentity, CampaignStateError> {
        Ok(ExperimentIdentity {
            simulator_version: self.u32()?,
            descriptor_version: self.u32()?,
            genome_layout_version: self.u32()?,
            feature_schema_version: self.u32()?,
            particles: self.usize()?,
            anchors: self.usize()?,
            radius_bits: self.u32()?,
            rate_bits: self.u32()?,
            evaluation_seed: self.u64()?,
            shells: self.usize()?,
            bumps: self.usize()?,
            discovery_steps: self.usize()?,
            tuning_digest: self.u64()?,
        })
    }

    fn record(&mut self) -> Result<CampaignRecord, CampaignStateError> {
        let evaluation_id = self.u64()?;
        let genome_hash = self.u64()?;
        let lineage_id = self.u64()?;
        let genome = self.f32s()?;
        let source = parent_source_from_code(self.byte()?)?;
        let source_tier = source_tier_from_code(self.byte()?)?;
        let tier = evaluation_tier_from_code(self.byte()?)?;
        let budget = EvaluationBudget {
            tier,
            steps: self.usize()?,
        };
        let seed = self.u64()?;
        let metrics = self.metrics()?;
        let feature_schema_version = self.u32()?;
        let features = self.f32s()?;
        let margins = self.margins()?;
        let base_gate = rejection_from_code(self.byte()?)?;
        let world_gate = world_rejection_from_code(self.byte()?)?;
        let early_stop = early_stop_from_code(self.byte()?)?;
        let tier_passed = self.bool()?;
        let windows = (0..self.len()?)
            .map(|_| {
                Ok(WindowRecord {
                    tick: self.usize()?,
                    spatial: self.f32s()?,
                    structure: self.f32()?,
                    heterogeneity: self.f32_array()?,
                    connectivity: crate::tuner::metrics::Connectivity {
                        dense: self.f32_array()?,
                        void: self.f32_array()?,
                    },
                    mobility: self.f32()?,
                    turnover: self.f32()?,
                })
            })
            .collect::<Result<Vec<_>, CampaignStateError>>()?;
        Ok(CampaignRecord {
            evaluation_id,
            genome_hash,
            lineage_id,
            genome,
            source,
            source_tier,
            budget,
            seed,
            metrics,
            feature_schema_version,
            features,
            margins,
            base_gate,
            world_gate,
            early_stop,
            windows,
            tier_passed,
        })
    }

    fn finished(&self) -> bool {
        self.at == self.bytes.len()
    }
}

fn parent_source_code(source: ParentSource) -> u8 {
    match source {
        ParentSource::Bootstrap => 0,
        ParentSource::Novelty => 1,
        ParentSource::Expedition => 2,
        ParentSource::Random => 3,
        ParentSource::Curated => 4,
    }
}

fn parent_source_from_code(code: u8) -> Result<ParentSource, CampaignStateError> {
    match code {
        0 => Ok(ParentSource::Bootstrap),
        1 => Ok(ParentSource::Novelty),
        2 => Ok(ParentSource::Expedition),
        3 => Ok(ParentSource::Random),
        4 => Ok(ParentSource::Curated),
        _ => Err(CampaignStateError::InvalidFormat("parent source")),
    }
}

fn source_tier_code(tier: SourceTier) -> u8 {
    match tier {
        SourceTier::Discovery => 0,
        SourceTier::Persistence => 1,
        SourceTier::Certification => 2,
    }
}

fn source_tier_from_code(code: u8) -> Result<SourceTier, CampaignStateError> {
    match code {
        0 => Ok(SourceTier::Discovery),
        1 => Ok(SourceTier::Persistence),
        2 => Ok(SourceTier::Certification),
        _ => Err(CampaignStateError::InvalidFormat("source tier")),
    }
}

fn evaluation_tier_code(tier: EvaluationTier) -> u8 {
    match tier {
        EvaluationTier::Discovery => 0,
        EvaluationTier::Persistence => 1,
        EvaluationTier::Certification => 2,
    }
}

fn evaluation_tier_from_code(code: u8) -> Result<EvaluationTier, CampaignStateError> {
    match code {
        0 => Ok(EvaluationTier::Discovery),
        1 => Ok(EvaluationTier::Persistence),
        2 => Ok(EvaluationTier::Certification),
        _ => Err(CampaignStateError::InvalidFormat("evaluation tier")),
    }
}

fn rejection_code(value: Result<(), Rejection>) -> u8 {
    match value {
        Ok(()) => 0,
        Err(Rejection::Dead) => 1,
        Err(Rejection::Dispersed) => 2,
        Err(Rejection::Frozen) => 3,
        Err(Rejection::Collapsed) => 4,
        Err(Rejection::Fragile) => 5,
    }
}

fn rejection_from_code(code: u8) -> Result<Result<(), Rejection>, CampaignStateError> {
    match code {
        0 => Ok(Ok(())),
        1 => Ok(Err(Rejection::Dead)),
        2 => Ok(Err(Rejection::Dispersed)),
        3 => Ok(Err(Rejection::Frozen)),
        4 => Ok(Err(Rejection::Collapsed)),
        5 => Ok(Err(Rejection::Fragile)),
        _ => Err(CampaignStateError::InvalidFormat("base gate")),
    }
}

fn world_rejection_code(value: Result<(), WorldRejection>) -> u8 {
    match value {
        Ok(()) => 0,
        Err(WorldRejection::Homogeneous) => 1,
        Err(WorldRejection::MaterialDisconnected) => 2,
        Err(WorldRejection::VoidDisconnected) => 3,
    }
}

fn world_rejection_from_code(code: u8) -> Result<Result<(), WorldRejection>, CampaignStateError> {
    match code {
        0 => Ok(Ok(())),
        1 => Ok(Err(WorldRejection::Homogeneous)),
        2 => Ok(Err(WorldRejection::MaterialDisconnected)),
        3 => Ok(Err(WorldRejection::VoidDisconnected)),
        _ => Err(CampaignStateError::InvalidFormat("world gate")),
    }
}

fn early_stop_code(value: Option<EarlyStop>) -> u8 {
    match value {
        None => 0,
        Some(EarlyStop::NonFinite) => 1,
        Some(EarlyStop::SustainedDispersion) => 2,
        Some(EarlyStop::SustainedCollapse) => 3,
    }
}

fn early_stop_from_code(code: u8) -> Result<Option<EarlyStop>, CampaignStateError> {
    match code {
        0 => Ok(None),
        1 => Ok(Some(EarlyStop::NonFinite)),
        2 => Ok(Some(EarlyStop::SustainedDispersion)),
        3 => Ok(Some(EarlyStop::SustainedCollapse)),
        _ => Err(CampaignStateError::InvalidFormat("early stop")),
    }
}

fn feature_schema() -> FeatureSchema {
    let mut names = (0..descriptor_len())
        .map(|index| format!("v{FEATURE_SCHEMA_VERSION}.descriptor.{index}"))
        .collect::<Vec<_>>();
    names.extend(
        [
            "structure",
            "mobility",
            "temporal_variance",
            "turnover",
            "robustness",
            "margin_structure_floor",
            "margin_structure_ceiling",
            "margin_mobility",
            "margin_temporal",
            "margin_repair",
            "margin_heterogeneity",
            "margin_material",
            "margin_void",
            "window_count",
            "window_structure_drift",
            "window_mobility_mean",
            "window_turnover_mean",
            "early_stop",
        ]
        .map(str::to_owned),
    );
    FeatureSchema::new(names)
}

fn feature_values(
    metrics: &Metrics,
    margins: GateMargins,
    windows: &[WindowRecord],
    early_stop: Option<EarlyStop>,
) -> Vec<f32> {
    // Early stops carry `Metrics::default()` with an empty descriptor. Preserve their failure
    // label in a fixed-width row rather than silently dropping them from future training.
    let mut values = metrics.descriptor.clone();
    values.truncate(descriptor_len());
    values.resize(descriptor_len(), 0.0);
    values.extend([
        metrics.structure,
        metrics.mobility,
        metrics.temporal_variance,
        metrics.turnover,
        metrics.robustness,
        margins.structure_floor,
        margins.structure_ceiling,
        margins.mobility,
        margins.temporal,
        margins.repair,
        margins.heterogeneity,
        margins.material,
        margins.void,
        windows.len() as f32,
        windows
            .first()
            .zip(windows.last())
            .map_or(0.0, |(first, last)| last.structure - first.structure),
        windows.iter().map(|window| window.mobility).sum::<f32>() / windows.len().max(1) as f32,
        windows.iter().map(|window| window.turnover).sum::<f32>() / windows.len().max(1) as f32,
        f32::from(early_stop.is_some()),
    ]);
    values
}

fn source_tier(tier: EvaluationTier) -> SourceTier {
    match tier {
        EvaluationTier::Discovery => SourceTier::Discovery,
        EvaluationTier::Persistence => SourceTier::Persistence,
        EvaluationTier::Certification => SourceTier::Certification,
    }
}

fn evaluation_tier(source: SourceTier) -> EvaluationTier {
    match source {
        SourceTier::Discovery => EvaluationTier::Discovery,
        SourceTier::Persistence => EvaluationTier::Persistence,
        SourceTier::Certification => EvaluationTier::Certification,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tuner::tuning::{Discovery, Tiers};
    use crate::tuner::metrics::{Connectivity, Metrics};
    use crate::tuner::rollout::{EvaluationBudget, SeedRecord};
    use crate::tuner::viability::{margins, viable};

    fn metrics(value: f32) -> Metrics {
        Metrics {
            mobility: 0.02,
            temporal_variance: 0.1,
            autocorrelation: [0.9, 0.8, 0.7],
            turnover: value,
            robustness: 1.0,
            structure: 0.4,
            heterogeneity: [0.2; 15],
            connectivity: Connectivity {
                dense: [1.0; 2],
                void: [1.0; 2],
            },
            descriptor: vec![value; descriptor_len()],
            alive: true,
        }
    }

    fn seed_record(value: f32) -> SeedRecord {
        let metrics = metrics(value);
        SeedRecord {
            seed: value.to_bits() as u64,
            margins: margins(&metrics, &Gates::default()),
            base: viable(&metrics, &Gates::default()),
            world: Ok(()),
            metrics,
            early_stop: None,
            windows: Vec::new(),
        }
    }

    fn candidate(id: u64) -> CampaignCandidate {
        CampaignCandidate {
            genome: vec![id as f32],
            genome_hash: id,
            lineage_id: id,
            source: ParentSource::Random,
            descriptor: vec![id as f32; descriptor_len()],
            features: feature_values(&metrics(id as f32), margins(&metrics(id as f32), &Gates::default()), &[], None),
        }
    }

    #[test]
    fn ledger_keeps_failed_seed_windows_and_versioned_features() {
        let mut ledger = CampaignLedger::default();
        let mut bad = seed_record(0.0);
        bad.early_stop = Some(EarlyStop::SustainedCollapse);
        bad.windows.push(WindowRecord {
            tick: 3,
            spatial: vec![0.0],
            structure: 0.2,
            heterogeneity: [0.0; 15],
            connectivity: Connectivity::default(),
            mobility: 0.0,
            turnover: 0.0,
        });
        let candidate = candidate(1);
        ledger.append_tier(
            &candidate,
            TierEvaluation {
                budget: EvaluationBudget::new(EvaluationTier::Discovery, 3),
                records: vec![bad],
                passes: 0,
            },
        );
        let record = &ledger.records()[0];
        assert_eq!(record.feature_schema_version, FEATURE_SCHEMA_VERSION);
        assert_eq!(record.windows.len(), 1);
        assert!(record.early_stop.is_some());
        assert!(ledger.schema().accepts(&record.features));
    }

    #[test]
    fn discovery_ledger_records_the_steps_actually_run() {
        let metrics = metrics(0.4);
        let record = EvaluationRecord {
            genome_hash: 4,
            lineage_id: 2,
            seed: 7,
            parent_source: ParentSource::Bootstrap,
            source_tier: SourceTier::Discovery,
            gate: viable(&metrics, &Gates::default()),
            metrics,
        };
        let mut ledger = CampaignLedger::default();
        ledger.append_search_discovery(&record, 2_345, &Gates::default());
        assert_eq!(ledger.records()[0].budget.steps, 2_345);
        assert!(budget_valid_for_source(
            ledger.records()[0].budget,
            SourceTier::Discovery
        ));
        ledger.append_search_discovery(&record, 2_345, &Gates::default());
        assert_eq!(ledger.records().len(), 1);
    }

    #[test]
    fn schedule_falls_back_uniformly_without_authority() {
        let choices = (1..=6).map(candidate).collect::<Vec<_>>();
        let picked = schedule(
            &choices,
            &PersistenceModel {
                model: None,
                rows: 0,
            },
            ContinuationQuotas {
                high_survival: 2,
                high_uncertainty: 2,
                uniform_neighborhood: 2,
            },
        );
        assert_eq!(picked.len(), 6);
        assert_eq!(picked[0].genome_hash, 1);
    }

    #[test]
    fn persistence_feature_rows_join_only_actual_tier_outcomes() {
        let mut ledger = CampaignLedger::default();
        let candidate = candidate(12);
        ledger.append_tier(
            &candidate,
            TierEvaluation {
                budget: EvaluationBudget::discovery(),
                records: vec![seed_record(0.3)],
                passes: 1,
            },
        );
        ledger.append_tier(
            &candidate,
            TierEvaluation {
                budget: EvaluationBudget::persistence(),
                records: vec![seed_record(0.3), seed_record(0.4), seed_record(0.5)],
                passes: 3,
            },
        );
        let rows = ledger.persistence_rows();
        assert_eq!(rows.len(), 1);
        assert!(rows[0].survives);
        assert_eq!(rows[0].lineage_id, 12);
    }

    #[test]
    fn short_persistence_is_not_a_label_and_repeated_outcomes_are_conservative() {
        let mut ledger = CampaignLedger::default();
        let candidate = candidate(12);
        ledger.append_tier(
            &candidate,
            TierEvaluation {
                budget: EvaluationBudget::discovery(),
                records: vec![seed_record(0.3)],
                passes: 1,
            },
        );
        let mut short = EvaluationBudget::persistence();
        short.steps -= 1;
        ledger.append_tier(
            &candidate,
            TierEvaluation {
                budget: short,
                records: vec![seed_record(0.3); 3],
                passes: 3,
            },
        );
        assert!(ledger.persistence_rows().is_empty());

        ledger.append_tier(
            &candidate,
            TierEvaluation {
                budget: EvaluationBudget::persistence(),
                records: vec![seed_record(0.3); 3],
                passes: 3,
            },
        );
        ledger.append_tier(
            &candidate,
            TierEvaluation {
                budget: EvaluationBudget::persistence(),
                records: vec![seed_record(0.3); 3],
                passes: 1,
            },
        );
        let rows = ledger.persistence_rows();
        assert_eq!(rows.len(), 1);
        assert!(!rows[0].survives);
        assert!(passing_persistence_candidates(&ledger).is_empty());
    }

    #[test]
    fn certification_candidates_are_not_repeated() {
        let mut ledger = CampaignLedger::default();
        let candidate = candidate(21);
        ledger.append_tier(
            &candidate,
            TierEvaluation {
                budget: EvaluationBudget::persistence(),
                records: vec![seed_record(0.3); 3],
                passes: 3,
            },
        );
        assert_eq!(passing_persistence_candidates(&ledger).len(), 1);
        ledger.append_tier(
            &candidate,
            TierEvaluation {
                budget: EvaluationBudget::certification(),
                records: vec![seed_record(0.3); 5],
                passes: 0,
            },
        );
        assert!(passing_persistence_candidates(&ledger).is_empty());
    }

    #[test]
    fn campaign_rejects_tiers_that_would_not_earn_their_name() {
        let short_discovery = Tuning {
            discovery: Discovery {
                steps: EvaluationTier::Discovery.minimum_steps() - 1,
                ..Discovery::default()
            },
            ..Tuning::default()
        };
        assert!(validate(&short_discovery).is_err());

        let short_persistence = Tuning {
            tiers: Tiers {
                persistence_steps: Some(EvaluationTier::Persistence.minimum_steps() - 1),
                ..Tiers::default()
            },
            ..Tuning::default()
        };
        assert!(validate(&short_persistence).is_err());

        let certification_without_persistence = Tuning {
            tiers: Tiers {
                persistence_steps: None,
                certification_steps: Some(EvaluationTier::Certification.minimum_steps()),
            },
            ..Tuning::default()
        };
        assert!(validate(&certification_without_persistence).is_err());

        let short_certification = Tuning {
            tiers: Tiers {
                persistence_steps: Some(EvaluationTier::Persistence.minimum_steps()),
                certification_steps: Some(EvaluationTier::Certification.minimum_steps() - 1),
            },
            ..Tuning::default()
        };
        assert!(validate(&short_certification).is_err());

        assert!(validate(&Tuning::default()).is_ok());
    }

    #[test]
    fn experiment_identity_excludes_search_scheduling_knobs() {
        let base = Tuning::default();
        let mut rescheduled = base.clone();
        rescheduled.discovery.seed = 999;
        rescheduled.discovery.batch = 3;
        rescheduled.discovery.initial = 7;
        rescheduled.discovery.capacity = 9;
        rescheduled.discovery.promotion_budget = 2;
        assert_eq!(
            ExperimentIdentity::for_tuning(&base),
            ExperimentIdentity::for_tuning(&rescheduled)
        );

        let mut reseeded = base.clone();
        reseeded.world.seed += 1;
        assert_ne!(
            ExperimentIdentity::for_tuning(&base),
            ExperimentIdentity::for_tuning(&reseeded)
        );
    }

    /// A gate is a per-run setting now, so it must split evidence the way the world does.
    #[test]
    fn experiment_identity_separates_worlds_judged_by_different_gates() {
        let base = Tuning::default();
        let mut strict = base.clone();
        strict.gates.robustness_floor += 0.1;
        assert_ne!(
            ExperimentIdentity::for_tuning(&base),
            ExperimentIdentity::for_tuning(&strict)
        );

        let mut shaken = base.clone();
        shaken.repair.shake += 0.1;
        assert_ne!(
            ExperimentIdentity::for_tuning(&base),
            ExperimentIdentity::for_tuning(&shaken)
        );
    }

    #[test]
    fn mismatched_state_is_rejected_before_search_work() {
        let tuning = Tuning::default();
        let mut state = CampaignState::for_tuning(&tuning);
        let mut changed = tuning.clone();
        changed.world.rate += 1.0;
        changed.discovery.generations = 0;
        let mut called = false;
        let result = run(
            &changed,
            &CampaignPersistence::default(),
            &mut state,
            &mut |_, _, _| called = true,
        );
        assert!(matches!(result, Err(CampaignError::ExperimentMismatch)));
        assert!(!called);
        assert!(state.ledger.records().is_empty());
    }

    #[test]
    fn campaign_state_round_trip_preserves_failure_evidence() {
        let tuning = Tuning::default();
        let mut state = CampaignState::for_tuning(&tuning);
        let mut failed = seed_record(0.0);
        failed.early_stop = Some(EarlyStop::NonFinite);
        state.ledger.append_tier(
            &candidate(7),
            TierEvaluation {
                budget: EvaluationBudget::discovery(),
                records: vec![failed],
                passes: 0,
            },
        );
        let path = std::env::temp_dir().join(format!("axiom-campaign-{}", std::process::id()));
        state.save(&path).unwrap();
        let loaded = CampaignState::load(&path).unwrap();
        let _ = std::fs::remove_file(path);
        assert_eq!(loaded.ledger.records().len(), 1);
        assert_eq!(
            loaded.ledger.records()[0].features[0].to_bits(),
            state.ledger.records()[0].features[0].to_bits()
        );
        assert_eq!(loaded.identity(), state.identity());
    }

    #[test]
    fn corrupt_campaign_state_is_rejected_by_checksum() {
        let state = CampaignState::for_tuning(&Tuning::default());
        let path =
            std::env::temp_dir().join(format!("axiom-campaign-corrupt-{}", std::process::id()));
        state.save(&path).unwrap();
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[20] ^= 0x40;
        std::fs::write(&path, bytes).unwrap();
        assert!(matches!(
            CampaignState::load(&path),
            Err(CampaignStateError::CorruptChecksum { .. })
        ));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn certified_recipe_scales_visual_support_to_small_worlds() {
        let caps = Caps {
            particles: 1,
            anchors: 2,
            shells: 1,
            bumps: 1,
            ..Caps::default()
        };
        let genome = caps.default_genome(caps.geometry_scale());
        let candidate = CampaignCandidate {
            genome_hash: crate::tuner::archive::genome_hash(&genome),
            lineage_id: 1,
            source: ParentSource::Random,
            descriptor: vec![0.0; descriptor_len()],
            features: vec![0.0; feature_schema().width()],
            genome,
        };
        let preset = certified_preset(
            &caps,
            caps.geometry_scale(),
            &candidate,
            EvaluationBudget::new(EvaluationTier::Certification, 1),
            9,
        )
        .unwrap();
        assert!(preset.manifest.render_recipe.support < preset.manifest.extent * 0.5);
    }
}
