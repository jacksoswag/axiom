//! Campaign orchestration for costly, evidence-backed world promotion. Discovery remains
//! novelty plus the binary viability gate; this module spends the expensive tiers and records
//! the evidence needed to learn where to spend them. It never changes archive admission,
//! viability, or the definition of a certified preset.

use crate::engine::resolve::Probe;
use crate::engine::sim::Sim;
use crate::render_recipe::RenderRecipe;
use crate::tuner::archive::{Archive, SourceTier};
use crate::tuner::checkpoint::{save_world, snapshot_world, SnapshotMetadata, WorldManifest, WorldState,
    GENOME_LAYOUT_VERSION, SIMULATOR_VERSION};
use crate::tuner::learning::{lineage_split, fit_persistence, select_continuations, ContinuationCandidate,
    ContinuationQuotas, PersistenceEnsemble, PersistencePrediction};
use crate::tuner::ledger::{budget_valid_for_source, CampaignCandidate, CampaignLedger};
use crate::tuner::metrics::DESCRIPTOR_VERSION;
use crate::tuner::novelty::neighborhood_key;
use crate::tuner::rollout::{evaluate_tier, EvaluationBudget, EvaluationTier};
use crate::tuner::search::{run_with_promotions, Report, SearchResult};
use crate::tuner::state::{write_atomic, CampaignState, CampaignStateError, ExperimentMismatch};
use crate::tuner::tuning::{Learning, Tuning};
use crate::tuner::ledger::ExperimentIdentity;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::io;
use std::path::PathBuf;

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
            Self::ExperimentMismatch => write!(f, "campaign state belongs to a different experiment"),
            Self::State(problem) => write!(f, "{problem}"),
            Self::Archive { path, source } => write!(f, "could not save archive {}: {source}", path.display()),
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
    fn from(problem: CampaignStateError) -> Self { Self::State(problem) }
}
impl From<crate::tuner::checkpoint::Error> for CampaignError {
    fn from(problem: crate::tuner::checkpoint::Error) -> Self { Self::Checkpoint(problem) }
}
impl From<ExperimentMismatch> for CampaignError {
    fn from(_: ExperimentMismatch) -> Self { Self::ExperimentMismatch }
}

/// Validate the evidence contract before any rollout work begins.
pub fn validate(tuning: &Tuning) -> Result<(), CampaignError> {
    let check = |budget: EvaluationBudget| {
        if budget.honors_tier() { Ok(()) } else {
            Err(CampaignError::InvalidConfiguration(format!(
                "{} requires at least {} steps", budget.tier.label(), budget.tier.minimum_steps())))
        }
    };
    check(EvaluationBudget::new(EvaluationTier::Discovery, tuning.discovery.steps))?;
    if let Some(steps) = tuning.tiers.persistence_steps {
        check(EvaluationBudget::new(EvaluationTier::Persistence, steps))?;
    }
    if let Some(steps) = tuning.tiers.certification_steps {
        if tuning.tiers.persistence_steps.is_none() {
            return Err(CampaignError::InvalidConfiguration("certification requires persistence evaluation".into()));
        }
        check(EvaluationBudget::new(EvaluationTier::Certification, steps))?;
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub struct PersistenceModel {
    pub model: Option<PersistenceEnsemble>,
    pub rows: usize,
}
impl PersistenceModel {
    pub fn authority_active(&self) -> bool {
        self.model.as_ref().is_some_and(|model| model.authority.active)
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

pub struct CampaignRun {
    pub search: SearchResult,
    pub persistence_model: Option<PersistenceModel>,
    pub certifications: Vec<CertifiedPreset>,
}

/// Run one campaign: discovery search, then whichever expensive tiers tuning.tiers enables.
/// This is the only entry point. Everything a run can vary lives in tuning; paths decides only
/// what gets written to disk. Resuming against prior evidence is the same call with a loaded
/// state: search batches are independent while the ledger stays append-only.
pub fn run(tuning: &Tuning, paths: &CampaignPersistence, state: &mut CampaignState,
    on_generation: &mut impl FnMut(usize, &Archive, &Report)) -> Result<CampaignRun, CampaignError>
{
    validate(tuning)?;
    state.bind(ExperimentIdentity::for_tuning(tuning))?;
    let mut callback = |generation, archive: &Archive, report: &Report, _: &_| {
        on_generation(generation, archive, report)
    };
    let search_result = run_with_promotions(tuning, &mut callback);
    let probe = tuning.world.probe(); // one measured density reference for every tier below

    for record in search_result.ledger.records() {
        state.ledger.append_search_discovery(record, tuning.discovery.steps, &tuning.gates);
    }
    let archive_result = persist_archive(paths, &search_result.archive);
    let state_result = persist_state(paths, state);
    archive_result?;
    state_result?;

    let already_persisted = state.ledger.records().iter()
        .filter(|record| record.source_tier == SourceTier::Persistence)
        .map(|record| (record.genome_hash, record.lineage_id))
        .collect::<BTreeSet<_>>();
    let mut candidates = Vec::new();
    for promotion in search_result.promotions.records() {
        if already_persisted.contains(&(promotion.genome_hash, promotion.lineage_id)) { continue; }
        if let Some(features) = state.ledger.discovery_features(promotion) {
            candidates.push(CampaignCandidate {
                genome: promotion.genome.clone(),
                genome_hash: promotion.genome_hash,
                lineage_id: promotion.lineage_id,
                source: promotion.source,
                descriptor: promotion.descriptor.clone(),
                features });
        }
    }

    let mut certifications = Vec::new();
    let Some(persistence_budget) = tuning.tiers.persistence_steps
        .map(|steps| EvaluationBudget::new(EvaluationTier::Persistence, steps))
    else {
        return Ok(CampaignRun { search: search_result, persistence_model: None, certifications });
    };
    let model = train_persistence(&state.ledger, &tuning.learning);
    let scheduled = schedule(&candidates, &model, tuning.learning.quotas);
    for candidate in scheduled {
        let evaluation = evaluate_tier(tuning, &probe, &candidate.genome, persistence_budget);
        state.ledger.append_tier(&candidate, evaluation);
        persist_state(paths, state)?;
    }
    let model = train_persistence(&state.ledger, &tuning.learning);

    if let Some(certification_budget) = tuning.tiers.certification_steps
        .map(|steps| EvaluationBudget::new(EvaluationTier::Certification, steps))
    {
        for candidate in passing_persistence_candidates(&state.ledger) {
            let evaluation = evaluate_tier(tuning, &probe, &candidate.genome, certification_budget);
            let passed = evaluation.passed();
            let passing_seed = evaluation.records.iter()
                .find(|record| record.base.is_ok() && record.world.is_ok()).map(|record| record.seed);
            state.ledger.append_tier(&candidate, evaluation);
            persist_state(paths, state)?;
            if passed {
                let preset = certified_preset(tuning, &probe, &candidate, certification_budget,
                    passing_seed.expect("a passed tier has a passing seed"))?;
                if let Some(root) = &paths.preset_dir {
                    save_world(root, &preset.manifest, &preset.state)?;
                }
                certifications.push(preset);
            }
        }
    }
    Ok(CampaignRun { search: search_result, persistence_model: Some(model), certifications })
}

fn persist_archive(paths: &CampaignPersistence, archive: &Archive) -> Result<(), CampaignError> {
    let Some(path) = &paths.archive_path else { return Ok(()); };
    write_atomic(path, archive.to_text().as_bytes())
        .map_err(|source| CampaignError::Archive { path: path.clone(), source })
}

fn persist_state(paths: &CampaignPersistence, state: &CampaignState) -> Result<(), CampaignError> {
    if let Some(path) = &paths.state_path { state.save(path)?; }
    Ok(())
}

/// Fit only after enough outcomes exist. The model may be present yet unauthoritative; the
/// held-out authority report remains the only switch that changes scheduling.
pub fn train_persistence(ledger: &CampaignLedger, config: &Learning) -> PersistenceModel {
    let rows = ledger.persistence_rows();
    if rows.len() < config.minimum_labeled_rows {
        return PersistenceModel { model: None, rows: rows.len() };
    }
    let split = lineage_split(&rows, config.model_seed);
    let model = fit_persistence(ledger.schema().clone(), &rows, &split, config.logistic, config.model_seed).ok();
    PersistenceModel { model, rows: rows.len() }
}

/// The candidates chosen under the named quotas. No model means a deterministic,
/// neighborhood-uniform fallback.
pub fn schedule(candidates: &[CampaignCandidate], model: &PersistenceModel,
    quotas: ContinuationQuotas) -> Vec<CampaignCandidate>
{
    let authority = model.authority_active();
    let ranked = candidates.iter().map(|candidate| {
        let prediction = model.model.as_ref()
            .and_then(|model| model.predict(&candidate.features).ok())
            .unwrap_or(PersistencePrediction { probability: 0.5, uncertainty: 0.0 });
        ContinuationCandidate {
            id: candidate.genome_hash,
            neighborhood: neighborhood_key(&candidate.descriptor),
            prediction }
    }).collect::<Vec<_>>();
    select_continuations(&ranked, quotas, authority).into_iter()
        .filter_map(|id| candidates.iter().find(|candidate| candidate.genome_hash == id))
        .cloned().collect()
}

/// Candidates whose every canonical persistence attempt passed and which certification has not
/// yet evaluated.
fn passing_persistence_candidates(ledger: &CampaignLedger) -> Vec<CampaignCandidate> {
    struct Aggregate { candidate: CampaignCandidate, all_passed: bool }
    let certification_evaluated = ledger.records().iter()
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
            candidates.entry(key)
                .and_modify(|aggregate| aggregate.all_passed &= record.tier_passed)
                .or_insert_with(|| Aggregate {
                    candidate: CampaignCandidate {
                        genome: record.genome.clone(),
                        genome_hash: record.genome_hash,
                        lineage_id: record.lineage_id,
                        source: record.source,
                        descriptor: record.metrics.descriptor.clone(),
                        features: record.features.clone() },
                    all_passed: record.tier_passed });
        }
    }
    candidates.into_values().filter(|aggregate| aggregate.all_passed).map(|aggregate| aggregate.candidate).collect()
}

/// Replay the passing seed for the tier budget and snapshot it. The replay decodes through the
/// same genome -> Params path as the tier evaluation, so the certified world is the world that
/// passed. Render support is clamped so the checkpoint stays valid for small worlds.
fn certified_preset(tuning: &Tuning, probe: &Probe, candidate: &CampaignCandidate,
    budget: EvaluationBudget, passing_seed: u64) -> Result<CertifiedPreset, crate::tuner::checkpoint::Error>
{
    let mut params = tuning.world.params(&candidate.genome, probe);
    params.seed = passing_seed;
    let mut sim = Sim::new(&params);
    for _ in 0..budget.steps { sim.step(); }
    let world_id = format!("world-{:016x}", candidate.genome_hash);
    let mut caps = tuning.world.clone();
    caps.seed = passing_seed;
    let mut render_recipe = RenderRecipe::default();
    render_recipe.support = render_recipe.support.min(params.box_len * 0.25);
    let (manifest, state) = snapshot_world(
        SnapshotMetadata {
            world_id: world_id.clone(),
            checkpoint_id: format!("certified-{}", budget.steps),
            parent_checkpoint_id: None,
            simulator_version: SIMULATOR_VERSION,
            descriptor_version: DESCRIPTOR_VERSION,
            genome_layout_version: GENOME_LAYOUT_VERSION,
            render_recipe_version: crate::render_recipe::VERSION,
            render_recipe },
        &caps, candidate.genome.clone(), &sim)?;
    Ok(CertifiedPreset {
        world_id,
        genome_hash: candidate.genome_hash,
        lineage_id: candidate.lineage_id,
        manifest, state })
}
