//! Deterministic novelty-led evolutionary generations.

use crate::engine::resolve::Probe;
use crate::tuner::archive::{genome_hash, Admission, Archive, Entry, EvaluationLedger, EvaluationRecord,
    ParentSource, SourceTier};
use crate::tuner::genome::random_genome;
use crate::tuner::metrics::Metrics;
use crate::tuner::novelty::{distance, neighborhood_key, novelty, NEIGHBORS};
use crate::tuner::rollout::discovery_metrics;
use crate::tuner::tuning::{Gates, LaneCounts, Mutation, Tuning};
use crate::tuner::viability::{viable, Rejection};
use crate::util::Rng;
use rayon::prelude::*;
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lane { Novelty, Expedition, Random }

#[derive(Clone, Debug)]
pub struct PromotionRecord {
    pub genome: Vec<f32>,
    pub genome_hash: u64, // this exact genome; families are lineage_id, so learning never splits relatives
    pub lineage_id: u64, // evolutionary family, inherited from the primary mutation parent
    pub descriptor: Vec<f32>,
    pub source_generation: u64,
    pub source: ParentSource,
}

/// A learner-free persistence handoff. The queue is deterministic and stores compact discovery
/// evidence, leaving learning authoritative for model decisions.
#[derive(Clone, Default, Debug)]
pub struct PromotionQueue { records: Vec<PromotionRecord> }
impl PromotionQueue {
    pub fn records(&self) -> &[PromotionRecord] { &self.records }
    /// Queue up to budget admissions: one pass diversified by neighborhood, a second filling
    /// whatever budget remains in novelty order.
    fn push_generation(&mut self, admissions: &[Admission], budget: usize) -> usize {
        if budget == 0 { return 0; }
        let mut ranked: Vec<&Admission> = admissions.iter().collect();
        ranked.sort_by(admission_order);
        let mut seen: BTreeSet<u64> = self.records.iter().map(|record| record.genome_hash).collect();
        let mut queued = 0;
        let mut selected = BTreeSet::new();
        for candidate in &ranked {
            if queued == budget || !selected.insert(neighborhood_key(&candidate.metrics.descriptor)) { continue; }
            queued += push_promotion(&mut self.records, &mut seen, candidate);
        }
        for candidate in ranked {
            if queued == budget { break; }
            queued += push_promotion(&mut self.records, &mut seen, candidate);
        }
        queued
    }
}

fn admission_order(left: &&Admission, right: &&Admission) -> std::cmp::Ordering {
    right.admission_novelty.total_cmp(&left.admission_novelty).then_with(|| left.lineage_id.cmp(&right.lineage_id))
}

fn push_promotion(records: &mut Vec<PromotionRecord>, seen: &mut BTreeSet<u64>, candidate: &Admission) -> usize {
    let genome_hash = genome_hash(&candidate.genome);
    if !seen.insert(genome_hash) { return 0; }
    records.push(PromotionRecord {
        genome_hash, genome: candidate.genome.clone(), lineage_id: candidate.lineage_id,
        descriptor: candidate.metrics.descriptor.clone(), source_generation: candidate.birth_generation,
        source: candidate.parent_source });
    1
}

#[derive(Clone, Copy, Default, Debug)]
pub struct StallDiagnostics {
    pub median_current_novelty: f32,
    pub occupied_neighborhoods: usize,
    pub stagnant_generations: usize,
    pub stalled: bool,
}

#[derive(Clone, Copy, Default, Debug)]
pub struct Report {
    pub evaluated: usize,
    pub viable: usize,
    pub lanes: LaneCounts,
    pub fallback_births: usize,
    pub promotions_queued: usize,
    pub stall: StallDiagnostics,
    rejected: [usize; Rejection::ALL.len()],
}
impl Report {
    fn record(&mut self, lane: Lane, outcome: Result<(), Rejection>) {
        self.evaluated += 1;
        match lane {
            Lane::Novelty => self.lanes.novelty += 1,
            Lane::Expedition => self.lanes.expedition += 1,
            Lane::Random => self.lanes.random += 1,
        }
        match outcome {
            Ok(()) => self.viable += 1,
            Err(reason) => {
                if let Some(slot) = Rejection::ALL.iter().position(|candidate| *candidate == reason) {
                    self.rejected[slot] += 1;
                }
            }
        }
    }
    pub fn count(&self, reason: Rejection) -> usize {
        Rejection::ALL.iter().position(|candidate| *candidate == reason).map_or(0, |slot| self.rejected[slot])
    }
    pub fn viable_fraction(&self) -> f32 {
        if self.evaluated == 0 { 0.0 } else { self.viable as f32 / self.evaluated as f32 }
    }
}

pub struct SearchResult {
    pub archive: Archive,
    pub promotions: PromotionQueue,
    pub ledger: EvaluationLedger,
}

#[derive(Clone)]
struct Candidate {
    genome: Vec<f32>,
    lineage_id: u64, // family, never individual: the genome hash is the individual identity
    parent_lineage_id: Option<u64>,
    generation: u64,
    source: ParentSource,
    lane: Lane,
}

pub fn run(tuning: &Tuning, mut on_generation: impl FnMut(usize, &Archive, &Report)) -> Archive {
    let mut callback = |generation, archive: &Archive, report: &Report, _: &PromotionQueue| {
        on_generation(generation, archive, report)
    };
    run_with_promotions(tuning, &mut callback).archive
}

pub fn run_with_promotions(
    tuning: &Tuning,
    on_generation: &mut impl FnMut(usize, &Archive, &Report, &PromotionQueue),
) -> SearchResult {
    let probe = tuning.world.probe(); // one measured density reference for the whole campaign
    let bounds = tuning.world.bounds(&probe);
    let mut rng = Rng::new(tuning.discovery.seed);
    let mut archive = Archive::new(tuning.clone());
    let mut promotions = PromotionQueue::default();
    let mut ledger = EvaluationLedger::default();
    let bootstrap: Vec<Candidate> = std::iter::once(tuning.world.default_genome(&probe))
        .chain((0..tuning.discovery.initial).map(|_| random_genome(&bounds, &mut rng)))
        .map(|genome| {
            let lineage_id = founder_lineage_id(&genome);
            Candidate { genome, lineage_id, parent_lineage_id: None, generation: 0,
                source: ParentSource::Bootstrap, lane: Lane::Random }
        })
        .collect();
    let bootstrap_snapshot = archive.descriptors();
    let (_, _, records) = absorb(&mut archive, evaluate(tuning, &probe, bootstrap),
        &bootstrap_snapshot, tuning.world.seed, &tuning.gates);
    for record in records { ledger.append(record); }
    let mut history = Vec::new();
    let mut prior_stalled = false;

    for generation in 0..tuning.discovery.generations {
        // merge() already refreshed current novelty; the snapshot below is that frozen view.
        let snapshot = archive.descriptors();
        let (batch, fallback_births) =
            offspring(tuning, &archive, &bounds, &mut rng, generation as u64 + 1, prior_stalled);
        let (mut report, viable, records) = absorb(&mut archive, evaluate(tuning, &probe, batch),
            &snapshot, tuning.world.seed, &tuning.gates);
        for record in records { ledger.append(record); }
        report.fallback_births = fallback_births;
        report.stall = stall_diagnostics(&archive, &mut history, tuning.discovery.stall_window);
        prior_stalled = report.stall.stalled;
        report.promotions_queued = promotions.push_generation(&viable, tuning.discovery.promotion_budget);
        on_generation(generation, &archive, &report, &promotions);
    }
    SearchResult { archive, promotions, ledger }
}

fn offspring(tuning: &Tuning, archive: &Archive, bounds: &[(f32, f32)], rng: &mut Rng,
    generation: u64, stalled: bool) -> (Vec<Candidate>, usize)
{
    let counts = tuning.discovery.lanes.counts(tuning.discovery.batch, stalled);
    let mut candidates = Vec::with_capacity(tuning.discovery.batch);
    let mut fallback = 0;
    for lane in std::iter::repeat_n(Lane::Novelty, counts.novelty)
        .chain(std::iter::repeat_n(Lane::Expedition, counts.expedition))
        .chain(std::iter::repeat_n(Lane::Random, counts.random))
    {
        let (genome, inherited_lineage_id, parent_lineage_id, source) = match lane {
            Lane::Random => (random_genome(bounds, rng), None, None, ParentSource::Random),
            Lane::Novelty if archive.len() >= 2 => {
                let (genome, parent_lineage_id, source) =
                    mutate_from(archive, pick_novelty(archive.entries(), rng), bounds, rng, tuning.discovery.mutation, stalled);
                (genome, parent_lineage_id, parent_lineage_id, source)
            }
            Lane::Expedition if archive.len() >= 2 => {
                let expedition_slot = candidates.iter().filter(|candidate: &&Candidate| candidate.lane == Lane::Expedition).count();
                let pinned = (expedition_slot < counts.expedition / 2) // curated pins take at most half the lane
                    .then_some(tuning.discovery.curated_parent_indices.get(expedition_slot))
                    .flatten().and_then(|index| archive.entries().get(*index));
                let parent = pinned.unwrap_or_else(|| goal_parent(archive.entries(), rng));
                let (genome, lineage, _) = mutate_from(archive, parent, bounds, rng, tuning.discovery.mutation, stalled);
                (genome, lineage, lineage, if pinned.is_some() { ParentSource::Curated } else { ParentSource::Expedition })
            }
            _ => { // archive too small to mutate from: random escape route
                fallback += 1;
                (random_genome(bounds, rng), None, None, ParentSource::Random)
            }
        };
        let lineage_id = inherited_lineage_id.unwrap_or_else(|| founder_lineage_id(&genome));
        candidates.push(Candidate { genome, lineage_id, parent_lineage_id, generation, source, lane });
    }
    (candidates, fallback)
}

/// A founder's exact genome identifies its family across independently resumed search batches.
/// Descendants keep this value while their own genome hashes remain individual IDs.
fn founder_lineage_id(genome: &[f32]) -> u64 {
    let hash = genome_hash(genome);
    if hash == 0 { u64::MAX } else { hash }
}

fn mutate_from(archive: &Archive, first: &Entry, bounds: &[(f32, f32)], rng: &mut Rng,
    mutation: Mutation, stalled: bool) -> (Vec<f32>, Option<u64>, ParentSource)
{
    let second = pick_novelty(archive.entries(), rng);
    let mut genome = iso_line_dd_scaled(&first.genome, &second.genome, bounds, rng, mutation,
        if stalled { mutation.stalled_spread } else { 1.0 });
    for (gene, &(low, high)) in genome.iter_mut().zip(bounds) { *gene = gene.clamp(low, high); }
    (genome, Some(first.lineage_id), ParentSource::Novelty)
}

/// Iso+LineDD: isotropic noise scaled to each gene's range, plus one shared draw along the
/// difference vector toward a second parent.
fn iso_line_dd_scaled(a: &[f32], b: &[f32], bounds: &[(f32, f32)], rng: &mut Rng,
    mutation: Mutation, spread: f32) -> Vec<f32>
{
    let line = rng.normal();
    a.iter().zip(b).zip(bounds).map(|((&a, &b), &(low, high))| {
        a + spread * mutation.iso * (high - low) * rng.normal() + spread * mutation.line * line * (b - a)
    }).collect()
}

/// Binary tournament on current novelty, ties broken by stable key.
fn pick_novelty<'a>(entries: &'a [Entry], rng: &mut Rng) -> &'a Entry {
    let left = &entries[rng.below(entries.len())];
    let right = &entries[rng.below(entries.len())];
    if right.current_novelty > left.current_novelty { right }
    else if right.current_novelty < left.current_novelty { left }
    else if right.stable_key() < left.stable_key() { right }
    else { left }
}

/// Sample a uniform descriptor target and mutate the archived behavior nearest it. In 53
/// dimensions this is random goal direction, not coverage-aware illumination, and that is the
/// design: distant intentions without letting a model invent simulation state.
fn goal_parent<'a>(entries: &'a [Entry], rng: &mut Rng) -> &'a Entry {
    let width = entries[0].metrics.descriptor.len();
    let target: Vec<f32> = (0..width).map(|_| rng.range(0.0, 2.0)).collect();
    entries.iter().min_by(|left, right| {
        distance(&left.metrics.descriptor, &target).total_cmp(&distance(&right.metrics.descriptor, &target))
            .then_with(|| left.stable_key().cmp(&right.stable_key()))
    }).unwrap()
}

/// Batch evaluation, order-preserving: rayon computes in parallel, collect keeps batch order.
fn evaluate(tuning: &Tuning, probe: &Probe, candidates: Vec<Candidate>) -> Vec<(Candidate, Metrics)> {
    candidates.into_par_iter().map(|candidate| {
        let metrics = discovery_metrics(tuning, probe, &candidate.genome, tuning.discovery.steps);
        (candidate, metrics)
    }).collect()
}

fn absorb(archive: &mut Archive, evaluated: Vec<(Candidate, Metrics)>, snapshot: &[Vec<f32>],
    seed: u64, gates: &Gates) -> (Report, Vec<Admission>, Vec<EvaluationRecord>)
{
    let mut report = Report::default();
    let mut viable_admissions = Vec::new();
    let mut records = Vec::new();
    for (candidate, metrics) in evaluated {
        let outcome = viable(&metrics, gates);
        report.record(candidate.lane, outcome);
        records.push(EvaluationRecord {
            genome_hash: genome_hash(&candidate.genome), lineage_id: candidate.lineage_id, seed,
            parent_source: candidate.source, source_tier: SourceTier::Discovery,
            metrics: metrics.clone(), gate: outcome });
        if outcome.is_ok() {
            viable_admissions.push(Admission {
                admission_novelty: novelty(&metrics.descriptor, snapshot, NEIGHBORS),
                genome: candidate.genome, metrics, lineage_id: candidate.lineage_id,
                parent_lineage_id: candidate.parent_lineage_id, birth_generation: candidate.generation,
                parent_source: candidate.source });
        }
    }
    archive.merge(viable_admissions.clone());
    (report, viable_admissions, records)
}

/// A stall is both signals flat across the window: median current novelty AND occupied
/// neighborhood count. Either one improving resets the clock.
fn stall_diagnostics(archive: &Archive, history: &mut Vec<(f32, usize)>, window: usize) -> StallDiagnostics {
    let mut novelty: Vec<f32> = archive.entries().iter().map(|entry| entry.current_novelty).collect();
    novelty.sort_by(f32::total_cmp);
    let median_current_novelty = novelty.get(novelty.len() / 2).copied().unwrap_or(0.0);
    let occupied_neighborhoods = archive.entries().iter()
        .map(|entry| neighborhood_key(&entry.metrics.descriptor)).collect::<BTreeSet<_>>().len();
    history.push((median_current_novelty, occupied_neighborhoods));
    let (mut best_novelty, mut best_neighborhoods) = history[0];
    let mut last_improvement = 0usize;
    for (index, &(value, neighborhoods)) in history.iter().enumerate().skip(1) {
        if value > best_novelty + 1e-6 || neighborhoods > best_neighborhoods { last_improvement = index; }
        best_novelty = best_novelty.max(value);
        best_neighborhoods = best_neighborhoods.max(neighborhoods);
    }
    let stagnant_generations = history.len() - 1 - last_improvement;
    let window = window.max(1);
    StallDiagnostics {
        median_current_novelty, occupied_neighborhoods, stagnant_generations,
        stalled: history.len() >= window && stagnant_generations + 1 >= window,
    }
}
