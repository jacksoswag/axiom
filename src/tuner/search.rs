//! Deterministic novelty-led evolutionary generations.

use crate::tuner::archive::{
    genome_hash, Admission, Archive, Entry, EvaluationLedger, EvaluationRecord, ParentSource,
    SourceTier,
};
use crate::tuner::metrics::Metrics;
use crate::tuner::novelty::{distance, neighborhood_key, novelty, NEIGHBOURS};
use crate::util::Rng;
use crate::tuner::rollout::from_genome_with_scale;
use crate::tuner::viability::{viable, Rejection};
use crate::engine::genome::{random_genome, COORDINATION_BOUNDS};
use crate::tuner::density::{self, DensityProbe};
use crate::tuner::tuning::{LaneCounts, Mutation, Tuning};
use rayon::prelude::*;
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lane {
    Novelty,
    Expedition,
    Random,
}

#[derive(Clone, Debug)]
pub struct PromotionRecord {
    pub genome: Vec<f32>,
    /// Stable identifier for this exact genome. Evolutionary families are identified by
    /// `lineage_id`, so held-out learning never splits close mutated relatives.
    pub genome_hash: u64,
    /// Stable evolutionary-family identifier inherited from the primary mutation parent.
    pub lineage_id: u64,
    pub descriptor: Vec<f32>,
    pub source_generation: u64,
    pub source: ParentSource,
}

/// A learner-free persistence handoff.  The queue is deterministic and stores compact
/// discovery evidence, leaving `learning.rs` authoritative for model decisions.
#[derive(Clone, Default, Debug)]
pub struct PromotionQueue {
    records: Vec<PromotionRecord>,
}

impl PromotionQueue {
    pub fn records(&self) -> &[PromotionRecord] {
        &self.records
    }
    fn push_generation(&mut self, admissions: &[Admission], budget: usize) -> usize {
        if budget == 0 {
            return 0;
        }
        let mut ranked: Vec<&Admission> = admissions.iter().collect();
        ranked.sort_by(admission_order);
        let seen: BTreeSet<u64> = self
            .records
            .iter()
            .map(|record| record.genome_hash)
            .collect();
        let mut seen = seen;
        let mut queued = 0;
        let mut selected = BTreeSet::new();
        for candidate in &ranked {
            if queued == budget || !selected.insert(neighborhood_key(&candidate.metrics.descriptor))
            {
                continue;
            }
            queued += push_promotion(&mut self.records, &mut seen, candidate);
        }
        for candidate in ranked {
            if queued == budget {
                break;
            }
            queued += push_promotion(&mut self.records, &mut seen, candidate);
        }
        queued
    }
}

fn admission_order(left: &&Admission, right: &&Admission) -> std::cmp::Ordering {
    right
        .admission_novelty
        .total_cmp(&left.admission_novelty)
        .then_with(|| left.lineage_id.cmp(&right.lineage_id))
}

fn push_promotion(
    records: &mut Vec<PromotionRecord>,
    seen: &mut BTreeSet<u64>,
    candidate: &Admission,
) -> usize {
    let genome_hash = genome_hash(&candidate.genome);
    if !seen.insert(genome_hash) {
        return 0;
    }
    records.push(PromotionRecord {
        genome_hash,
        genome: candidate.genome.clone(),
        lineage_id: candidate.lineage_id,
        descriptor: candidate.metrics.descriptor.clone(),
        source_generation: candidate.birth_generation,
        source: candidate.parent_source,
    });
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
                if let Some(slot) = Rejection::ALL
                    .iter()
                    .position(|candidate| *candidate == reason)
                {
                    self.rejected[slot] += 1;
                }
            }
        }
    }
    pub fn count(&self, reason: Rejection) -> usize {
        Rejection::ALL
            .iter()
            .position(|candidate| *candidate == reason)
            .map_or(0, |slot| self.rejected[slot])
    }
    pub fn viable_fraction(&self) -> f32 {
        if self.evaluated == 0 {
            0.0
        } else {
            self.viable as f32 / self.evaluated as f32
        }
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
    /// Evolutionary-family identifier, not an individual identifier. The genome hash is the
    /// individual identity at evaluation and promotion boundaries.
    lineage_id: u64,
    parent_lineage_id: Option<u64>,
    generation: u64,
    source: ParentSource,
    lane: Lane,
}

pub fn run(
    tuning: &Tuning,
    mut on_generation: impl FnMut(usize, &Archive, &Report),
) -> Archive {
    let mut callback = |generation, archive: &Archive, report: &Report, _: &PromotionQueue| {
        on_generation(generation, archive, report)
    };
    run_with_promotions(tuning, &mut callback).archive
}

pub fn run_with_promotions(
    tuning: &Tuning,
    on_generation: &mut impl FnMut(usize, &Archive, &Report, &PromotionQueue),
) -> SearchResult {
    let geometry_scale = density::probe_for(&tuning.world);
    let widest_extent = geometry_scale.bound_len(COORDINATION_BOUNDS.1);
    let bounds = tuning.world.bounds(widest_extent);
    let mut rng = Rng::new(tuning.discovery.seed);
    let mut archive = Archive::new(tuning.clone());
    let mut promotions = PromotionQueue::default();
    let mut ledger = EvaluationLedger::default();
    let bootstrap: Vec<Candidate> = std::iter::once(tuning.world.default_genome(widest_extent))
        .chain((0..tuning.discovery.initial).map(|_| random_genome(&bounds, &mut rng)))
        .map(|genome| {
            let lineage_id = founder_lineage_id(&genome);
            Candidate {
                genome,
                lineage_id,
                parent_lineage_id: None,
                generation: 0,
                source: ParentSource::Bootstrap,
                lane: Lane::Random,
            }
        })
        .collect();
    let bootstrap_snapshot = archive.descriptors();
    let (_, _, records) = absorb(
        &mut archive,
        evaluate(tuning, geometry_scale, bootstrap),
        &bootstrap_snapshot,
        tuning.world.seed,
        &tuning.gates,
    );
    for record in records {
        ledger.append(record);
    }
    let mut history = Vec::new();
    let mut prior_stalled = false;

    for generation in 0..tuning.discovery.generations {
        archive.refresh_current_novelty();
        let snapshot = archive.descriptors();
        let (batch, fallback_births) = offspring(
            tuning,
            &archive,
            &bounds,
            &mut rng,
            generation as u64 + 1,
            prior_stalled,
        );
        let (mut report, viable, records) = absorb(
            &mut archive,
            evaluate(tuning, geometry_scale, batch),
            &snapshot,
            tuning.world.seed,
            &tuning.gates,
        );
        for record in records {
            ledger.append(record);
        }
        report.fallback_births = fallback_births;
        report.stall = stall_diagnostics(&archive, &mut history, tuning.discovery.stall_window);
        prior_stalled = report.stall.stalled;
        report.promotions_queued = promotions.push_generation(&viable, tuning.discovery.promotion_budget);
        on_generation(generation, &archive, &report, &promotions);
    }
    SearchResult {
        archive,
        promotions,
        ledger,
    }
}

fn offspring(
    tuning: &Tuning,
    archive: &Archive,
    bounds: &[(f32, f32)],
    rng: &mut Rng,
    generation: u64,
    stalled: bool,
) -> (Vec<Candidate>, usize) {
    let counts = tuning.discovery.lanes.counts(tuning.discovery.batch, stalled);
    let mut candidates = Vec::with_capacity(tuning.discovery.batch);
    let mut fallback = 0;
    for lane in std::iter::repeat_n(Lane::Novelty, counts.novelty)
        .chain(std::iter::repeat_n(Lane::Expedition, counts.expedition))
        .chain(std::iter::repeat_n(Lane::Random, counts.random))
    {
        let (genome, inherited_lineage_id, parent_lineage_id, source) = match lane {
            Lane::Random => (
                random_genome(bounds, rng),
                None,
                None,
                ParentSource::Random,
            ),
            Lane::Novelty if archive.len() >= 2 => {
                let (genome, parent_lineage_id, source) = mutate_from(
                    archive,
                    pick_novelty(archive.entries(), rng),
                    bounds,
                    rng,
                    tuning.discovery.mutation,
                    stalled,
                );
                (genome, parent_lineage_id, parent_lineage_id, source)
            }
            Lane::Expedition if archive.len() >= 2 => {
                let expedition_slot = candidates
                    .iter()
                    .filter(|candidate: &&Candidate| candidate.lane == Lane::Expedition)
                    .count();
                let pinned = (expedition_slot < counts.expedition / 2)
                    .then_some(tuning.discovery.curated_parent_indices.get(expedition_slot))
                    .flatten()
                    .and_then(|index| archive.entries().get(*index));
                let parent = pinned.unwrap_or_else(|| goal_parent(archive.entries(), rng));
                let (genome, lineage, _) = mutate_from(archive, parent, bounds, rng, tuning.discovery.mutation, stalled);
                (
                    genome,
                    lineage,
                    lineage,
                    if pinned.is_some() {
                        ParentSource::Curated
                    } else {
                        ParentSource::Expedition
                    },
                )
            }
            _ => {
                fallback += 1;
                (
                    random_genome(bounds, rng),
                    None,
                    None,
                    ParentSource::Random,
                )
            }
        };
        let lineage_id = inherited_lineage_id.unwrap_or_else(|| founder_lineage_id(&genome));
        candidates.push(Candidate {
            genome,
            lineage_id,
            parent_lineage_id,
            generation,
            source,
            lane,
        });
    }
    (candidates, fallback)
}

fn founder_lineage_id(genome: &[f32]) -> u64 {
    // A founder's exact genome identifies its family across independently resumed search
    // batches. Descendants keep this value while their own genome hashes remain individual IDs.
    let hash = genome_hash(genome);
    if hash == 0 {
        u64::MAX
    } else {
        hash
    }
}

fn mutate_from(
    archive: &Archive,
    first: &Entry,
    bounds: &[(f32, f32)],
    rng: &mut Rng,
    mutation: Mutation,
    stalled: bool,
) -> (Vec<f32>, Option<u64>, ParentSource) {
    let second = pick_novelty(archive.entries(), rng);
    let mut genome = iso_line_dd_scaled(
        &first.genome,
        &second.genome,
        bounds,
        rng,
        mutation,
        if stalled { mutation.stalled_spread } else { 1.0 },
    );
    for (gene, &(low, high)) in genome.iter_mut().zip(bounds) {
        *gene = gene.clamp(low, high);
    }
    (genome, Some(first.lineage_id), ParentSource::Novelty)
}

fn iso_line_dd_scaled(
    a: &[f32],
    b: &[f32],
    bounds: &[(f32, f32)],
    rng: &mut Rng,
    mutation: Mutation,
    spread: f32,
) -> Vec<f32> {
    let line = rng.normal();
    a.iter()
        .zip(b)
        .zip(bounds)
        .map(|((&a, &b), &(low, high))| {
            a + spread * mutation.iso * (high - low) * rng.normal()
                + spread * mutation.line * line * (b - a)
        })
        .collect()
}

fn pick_novelty<'a>(entries: &'a [Entry], rng: &mut Rng) -> &'a Entry {
    let left = &entries[rng.below(entries.len())];
    let right = &entries[rng.below(entries.len())];
    if right.current_novelty > left.current_novelty {
        right
    } else if right.current_novelty < left.current_novelty {
        left
    } else if right.stable_key() < left.stable_key() {
        right
    } else {
        left
    }
}

fn goal_parent<'a>(entries: &'a [Entry], rng: &mut Rng) -> &'a Entry {
    let width = entries[0].metrics.descriptor.len();
    let target: Vec<f32> = (0..width).map(|_| rng.range(0.0, 2.0)).collect();
    entries
        .iter()
        .min_by(|left, right| {
            distance(&left.metrics.descriptor, &target)
                .total_cmp(&distance(&right.metrics.descriptor, &target))
                .then_with(|| left.stable_key().cmp(&right.stable_key()))
        })
        .unwrap()
}

fn evaluate(
    tuning: &Tuning,
    geometry_scale: DensityProbe,
    candidates: Vec<Candidate>,
) -> Vec<(Candidate, Metrics)> {
    candidates
        .into_par_iter()
        .map(|candidate| {
            let metrics = from_genome_with_scale(
                tuning,
                geometry_scale,
                &candidate.genome,
                tuning.discovery.steps,
            );
            (candidate, metrics)
        })
        .collect()
}

fn absorb(
    archive: &mut Archive,
    evaluated: Vec<(Candidate, Metrics)>,
    snapshot: &[Vec<f32>],
    seed: u64,
    gates: &crate::tuner::tuning::Gates,
) -> (Report, Vec<Admission>, Vec<EvaluationRecord>) {
    let mut report = Report::default();
    let mut viable_admissions = Vec::new();
    let mut records = Vec::new();
    for (candidate, metrics) in evaluated {
        let outcome = viable(&metrics, gates);
        report.record(candidate.lane, outcome);
        records.push(EvaluationRecord {
            genome_hash: genome_hash(&candidate.genome),
            lineage_id: candidate.lineage_id,
            seed,
            parent_source: candidate.source,
            source_tier: SourceTier::Discovery,
            metrics: metrics.clone(),
            gate: outcome,
        });
        if outcome.is_ok() {
            viable_admissions.push(Admission {
                admission_novelty: novelty(&metrics.descriptor, snapshot, NEIGHBOURS),
                genome: candidate.genome,
                metrics,
                lineage_id: candidate.lineage_id,
                parent_lineage_id: candidate.parent_lineage_id,
                birth_generation: candidate.generation,
                parent_source: candidate.source,
            });
        }
    }
    archive.merge(viable_admissions.clone());
    (report, viable_admissions, records)
}

fn stall_diagnostics(
    archive: &Archive,
    history: &mut Vec<(f32, usize)>,
    window: usize,
) -> StallDiagnostics {
    let mut novelty: Vec<f32> = archive
        .entries()
        .iter()
        .map(|entry| entry.current_novelty)
        .collect();
    novelty.sort_by(f32::total_cmp);
    let median_current_novelty = novelty.get(novelty.len() / 2).copied().unwrap_or(0.0);
    let occupied_neighborhoods = archive
        .entries()
        .iter()
        .map(|entry| neighborhood_key(&entry.metrics.descriptor))
        .collect::<BTreeSet<_>>()
        .len();
    history.push((median_current_novelty, occupied_neighborhoods));
    let mut best_novelty = history[0].0;
    let mut best_neighborhoods = history[0].1;
    let mut last_improvement = 0usize;
    for (index, &(value, neighborhoods)) in history.iter().enumerate().skip(1) {
        if value > best_novelty + 1e-6 || neighborhoods > best_neighborhoods {
            last_improvement = index;
        }
        best_novelty = best_novelty.max(value);
        best_neighborhoods = best_neighborhoods.max(neighborhoods);
    }
    let stagnant_generations = history.len() - 1 - last_improvement;
    let window = window.max(1);
    StallDiagnostics {
        median_current_novelty,
        occupied_neighborhoods,
        stagnant_generations,
        stalled: history.len() >= window && stagnant_generations + 1 >= window,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::genome::Caps;
    use crate::tuner::tuning::{Discovery, Lanes};
    use crate::tuner::metrics::{descriptor_len, Connectivity, HETEROGENEITY_SIDES, HETEROGENEITY_VALUES};

    fn viable_metrics(value: f32) -> Metrics {
        let mut descriptor = vec![1.0; descriptor_len()];
        descriptor[0] = value;
        Metrics {
            mobility: 0.02,
            temporal_variance: 0.1,
            autocorrelation: [0.9, 0.7, 0.4],
            turnover: 0.1,
            robustness: 1.0,
            structure: 0.5,
            heterogeneity: [0.2; HETEROGENEITY_SIDES.len() * HETEROGENEITY_VALUES],
            connectivity: Connectivity {
                dense: [1.0; 2],
                void: [1.0; 2],
            },
            descriptor,
            alive: true,
        }
    }

    fn capped(capacity: usize) -> Tuning {
        Tuning {
            discovery: Discovery {
                capacity,
                ..Default::default()
            },
            ..Tuning::default()
        }
    }

    fn tuned(batch: usize) -> Tuning {
        Tuning {
            discovery: Discovery {
                batch,
                ..Default::default()
            },
            ..Tuning::default()
        }
    }

    #[test]
    fn lane_quotas_and_empty_archive_fallback_are_deterministic() {
        let lanes = Lanes::default();
        assert_eq!(
            lanes.counts(10, false),
            LaneCounts {
                novelty: 7,
                expedition: 2,
                random: 1
            }
        );
        assert_eq!(
            lanes.counts(10, true),
            LaneCounts {
                novelty: 5,
                expedition: 3,
                random: 2,
            }
        );
        let tuning = tuned(10);
        let mut rng = Rng::new(1);
        let (batch, fallback) = offspring(
            &tuning,
            &Archive::new(Tuning::default()),
            &tuning.world.bounds(density::widest_bound_len(&tuning.world)),
            &mut rng,
            1,
            false,
        );
        assert_eq!(batch.len(), 10);
        assert_eq!(fallback, 9);
    }

    #[test]
    fn an_exact_novelty_plateau_triggers_stall_handling() {
        let archive = Archive::new(Tuning::default());
        let mut history = Vec::new();
        for _ in 0..7 {
            assert!(!stall_diagnostics(&archive, &mut history, 8).stalled);
        }
        let report = stall_diagnostics(&archive, &mut history, 8);
        assert!(report.stalled);
        assert_eq!(report.stagnant_generations, 7);
    }

    #[test]
    fn mutations_inherit_family_while_random_births_start_families() {
        let tuning = tuned(10);
        let mut archive = Archive::new(Tuning::default());
        archive.merge([11u64, 22].map(|lineage_id| Admission {
            genome: tuning.world.default_genome(density::widest_bound_len(&tuning.world)),
            metrics: viable_metrics(lineage_id as f32 / 100.0),
            lineage_id,
            parent_lineage_id: None,
            birth_generation: 0,
            parent_source: ParentSource::Bootstrap,
            admission_novelty: lineage_id as f32,
        }));
        let mut rng = Rng::new(17);
        let (batch, fallback) =
            offspring(&tuning, &archive, &tuning.world.bounds(density::widest_bound_len(&tuning.world)), &mut rng, 1, false);
        assert_eq!(fallback, 0);
        let inherited = batch
            .iter()
            .filter(|candidate| candidate.parent_lineage_id.is_some())
            .collect::<Vec<_>>();
        assert_eq!(inherited.len(), 9);
        assert!(inherited.iter().all(|candidate| {
            candidate.parent_lineage_id == Some(candidate.lineage_id)
                && [11, 22].contains(&candidate.lineage_id)
        }));
        let random = batch
            .iter()
            .find(|candidate| candidate.parent_lineage_id.is_none())
            .expect("the random lane starts a new family");
        assert_eq!(random.lineage_id, founder_lineage_id(&random.genome));
        assert!(![11, 22].contains(&random.lineage_id));
    }

    #[test]
    fn capacity_is_independent_of_viable_batch_order() {
        let admissions = |order: &[u64]| {
            order
                .iter()
                .map(|&id| Admission {
                    genome: vec![id as f32],
                    metrics: viable_metrics(id as f32),
                    lineage_id: id,
                    parent_lineage_id: None,
                    birth_generation: 1,
                    parent_source: ParentSource::Novelty,
                    admission_novelty: id as f32,
                })
                .collect::<Vec<_>>()
        };
        let mut left = Archive::new(capped(2));
        left.merge(admissions(&[1, 2, 5, 9]));
        let mut right = Archive::new(capped(2));
        right.merge(admissions(&[9, 5, 2, 1]));
        assert_eq!(
            left.entries()
                .iter()
                .map(|entry| entry.lineage_id)
                .collect::<Vec<_>>(),
            right
                .entries()
                .iter()
                .map(|entry| entry.lineage_id)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn seeded_parallel_generations_reproduce() {
        let tuning = Tuning {
            world: Caps {
                particles: 30,
                anchors: 2,
                shells: 1,
                bumps: 1,
                ..Default::default()
            },
            discovery: Discovery {
                steps: 20,
                initial: 2,
                batch: 4,
                capacity: 8,
                seed: 3,
                generations: 1,
                ..Default::default()
            },
            ..Tuning::default()
        };
        let first = run(&tuning, |_, _, _| {});
        let second = run(&tuning, |_, _, _| {});
        assert_eq!(
            first
                .entries()
                .iter()
                .map(|entry| (&entry.genome, entry.lineage_id, entry.current_novelty))
                .collect::<Vec<_>>(),
            second
                .entries()
                .iter()
                .map(|entry| (&entry.genome, entry.lineage_id, entry.current_novelty))
                .collect::<Vec<_>>()
        );
    }
}
