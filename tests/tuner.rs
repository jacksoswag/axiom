//! Tuner behavior: measurement, gates, the knob registry, search determinism, evidence
//! separation, and both durable formats.

use axiom::engine::params::Params;
use axiom::engine::sim::Sim;
use axiom::engine::substrate::Substrate;
use axiom::tuner::archive::{Admission, Archive, EvaluationRecord, ParentSource, SourceTier};
use axiom::tuner::checkpoint::{self, load_manifest, load_state, restore_world, save_manifest, save_state,
    save_world, snapshot_world, SnapshotMetadata, WorldState};
use axiom::tuner::genome::Caps;
use axiom::tuner::learning::{fit_persistence, lineage_split, select_continuations, ContinuationCandidate,
    ContinuationQuotas, FeatureRow, FeatureSchema, LogisticConfig, Partition, PersistencePrediction};
use axiom::tuner::ledger::{feature_values, CampaignCandidate, CampaignLedger, ExperimentIdentity};
use axiom::tuner::metrics::{descriptor_len, heterogeneity, normalized_rdf, raw_rdf, spatial_field,
    Connectivity, Metrics, HETEROGENEITY_SIDES, HETEROGENEITY_VALUES};
use axiom::tuner::novelty::{crowding, neighborhood_key, novelty};
use axiom::tuner::persistence::{h0, mass, Barcode, BARS};
use axiom::tuner::rollout::{evaluate_tier, EvaluationBudget, EvaluationTier, TierEvaluation, SeedRecord};
use axiom::tuner::search::run;
use axiom::tuner::state::CampaignState;
use axiom::tuner::tuning::{self, Discovery, Gates, Lanes, Tiers, Tuning, KNOBS};
use axiom::tuner::viability::{margins, viable, world_qualified, Rejection, WorldRejection};
use axiom::util::Rng;

const DIMENSIONS: usize = 3;

fn substrate_at(count: usize, box_len: f32, seed: u64) -> Substrate {
    let params = Params { particle_count: count, dimensions: DIMENSIONS, coordination: 9.0, radius: 12.0,
        dt: 0.1, seed, anchor_count: 2, shells: 1, bumps: 1, trait_distribution: vec![0.0; 2],
        interactions: Vec::new(), box_len };
    Substrate::build(&params)
}

fn uniform_traits(substrate: &mut Substrate, seed: u64) {
    let mut rng = Rng::new(seed.wrapping_add(1));
    for value in &mut substrate.traits { *value = rng.unit(); }
}

fn living() -> Metrics {
    Metrics {
        mobility: 0.02, temporal_variance: 0.05, autocorrelation: [0.9, 0.7, 0.4], turnover: 0.1,
        robustness: 1.0, structure: 0.5, heterogeneity: [0.2; 15],
        connectivity: Connectivity { dense: [1.0; 2], void: [1.0; 2] },
        descriptor: vec![1.0; descriptor_len()], alive: true }
}

fn small_tuning() -> Tuning {
    Tuning {
        world: Caps { particle_count: 30, anchor_count: 2, shells: 1, bumps: 1, ..Caps::default() },
        discovery: Discovery { steps: 20, initial: 2, batch: 4, capacity: 8, seed: 3, generations: 1,
            ..Discovery::default() },
        ..Tuning::default()
    }
}

// novelty

#[test]
fn novelty_averages_exactly_the_nearest_neighbors() {
    let population = vec![vec![1.0], vec![2.0], vec![3.0], vec![100.0]];
    assert!((novelty(&[0.0], &population, 3) - 2.0).abs() < 1e-6);
}

#[test]
fn crowding_exposes_near_duplicates() {
    let population = vec![vec![0.1], vec![4.0]];
    assert!((crowding(&[0.0], &population) - 0.1).abs() < 1e-6);
}

#[test]
fn neighborhood_key_spans_every_semantic_block() {
    let baseline = vec![0.0; descriptor_len()];
    let baseline_key = neighborhood_key(&baseline);
    let starts = [0, 18, 33, 41, 43, 45, 49, 51];
    for start in starts {
        let end = starts.into_iter().find(|&candidate| candidate > start).unwrap_or(descriptor_len());
        let mut changed = baseline.clone();
        changed[start..end].fill(2.0);
        assert_ne!(baseline_key, neighborhood_key(&changed), "block at {start}");
    }
}

// viability

#[test]
fn base_gate_reports_the_first_failing_named_clause() {
    let gates = Gates::default();
    for (metrics, rejection) in [
        (Metrics { alive: false, ..living() }, Rejection::Dead),
        (Metrics { structure: 0.001, ..living() }, Rejection::Dispersed),
        (Metrics { structure: 1.4, ..living() }, Rejection::Collapsed),
        (Metrics { mobility: 0.0, ..living() }, Rejection::Frozen),
        (Metrics { temporal_variance: 0.0, ..living() }, Rejection::Frozen),
        (Metrics { robustness: 0.0, ..living() }, Rejection::Fragile),
    ] {
        assert_eq!(viable(&metrics, &gates), Err(rejection));
    }
    assert_eq!(viable(&living(), &gates), Ok(()));
}

#[test]
fn world_clauses_are_named_and_a_single_blob_reads_homogeneous() {
    let gates = Gates::default();
    assert_eq!(world_qualified(&Metrics { heterogeneity: [0.0; 15], ..living() }, &gates),
        Err(WorldRejection::Homogeneous));
    assert_eq!(world_qualified(&Metrics {
            connectivity: Connectivity { dense: [0.2; 2], void: [1.0; 2] }, ..living() }, &gates),
        Err(WorldRejection::MaterialDisconnected));
    assert_eq!(world_qualified(&Metrics {
            connectivity: Connectivity { dense: [1.0; 2], void: [0.2; 2] }, ..living() }, &gates),
        Err(WorldRejection::VoidDisconnected));
    let mut blob = living();
    for scale in blob.heterogeneity.chunks_exact_mut(5) { scale[0] = 1.0; scale[2] = 0.0; }
    assert_eq!(world_qualified(&blob, &gates), Err(WorldRejection::Homogeneous));
    assert_eq!(world_qualified(&living(), &gates), Ok(()));
}

#[test]
fn raising_a_floor_rejects_a_world_that_passed_the_default() {
    assert_eq!(viable(&living(), &Gates::default()), Ok(()));
    let strict = Gates { robustness_floor: 1.5, ..Gates::default() };
    assert_eq!(viable(&living(), &strict), Err(Rejection::Fragile));
}

// tuning registry

#[test]
fn every_knob_round_trips_through_its_own_text() {
    let mut tuning = Tuning::default();
    for knob in KNOBS {
        let shown = knob.show(&tuning);
        knob.set(&mut tuning, &shown).unwrap_or_else(|problem| panic!("{}: {problem}", knob.key));
        assert_eq!(knob.show(&tuning), shown, "{} did not round trip", knob.key);
    }
}

#[test]
fn knob_keys_are_unique() {
    let mut keys: Vec<&str> = KNOBS.iter().map(|knob| knob.key).collect();
    let count = keys.len();
    keys.sort_unstable();
    keys.dedup();
    assert_eq!(keys.len(), count, "duplicate knob key");
}

#[test]
fn the_digest_separates_evidence_regimes_from_search_effort() {
    let base = Tuning::default();
    for key in ["batch", "generations", "seed", "capacity", "promotion_budget"] {
        let mut other = base.clone();
        tuning::apply(&mut other, key, "99").unwrap();
        assert_eq!(base.digest(), other.digest(), "{key} should not split evidence");
    }
    for key in ["robustness_floor", "mutation_iso", "repair_shake", "lane_novelty"] {
        let mut other = base.clone();
        tuning::apply(&mut other, key, "0.42").unwrap();
        assert_ne!(base.digest(), other.digest(), "{key} must split evidence");
    }
}

#[test]
fn lane_counts_always_sum_to_the_batch() {
    let lanes = Lanes::default();
    for batch in 0..64 {
        for stalled in [false, true] {
            let counts = lanes.counts(batch, stalled);
            assert_eq!(counts.novelty + counts.expedition + counts.random, batch, "batch {batch} stalled {stalled}");
        }
    }
}

// persistence (H0)

#[test]
fn bars_and_components_account_for_every_particle() {
    let mut substrate = substrate_at(500, 150.0, 5);
    let barcode = h0(&mut substrate, 40.0);
    let bars: f64 = barcode.bins.iter().sum();
    assert_eq!(bars as usize + barcode.components, 500);
}

#[test]
fn planted_blobs_separate_where_a_gas_connects() {
    let mut rng = Rng::new(3);
    let mut blobs = substrate_at(270, 300.0, 3);
    let mut at = 0;
    for center in [40.0f32, 140.0, 240.0] {
        for _ in 0..90 {
            for axis in 0..3 { blobs.positions[at * 3 + axis] = center + rng.range(-4.0, 4.0); }
            at += 1;
        }
    }
    let clustered = h0(&mut blobs, 70.0);
    assert_eq!(clustered.components, 3, "three blobs did not read as three components");
    let mut gas = substrate_at(270, 300.0, 8);
    let flat = h0(&mut gas, 70.0);
    assert!(flat.components < clustered.components);
    let center_of_mass = |bins: &[f64]| -> f64 {
        let total: f64 = bins.iter().sum();
        bins.iter().enumerate().map(|(i, v)| i as f64 * v).sum::<f64>() / total.max(1.0)
    };
    assert!(center_of_mass(&clustered.bins) < center_of_mass(&flat.bins) - 1.0,
        "blob and gas edge scales did not separate");
}

#[test]
fn mass_is_deterministic_and_preserves_the_component_count() {
    let mut substrate = substrate_at(400, 150.0, 17);
    let first = h0(&mut substrate, 40.0);
    let second = h0(&mut substrate, 40.0);
    assert_eq!(first.bins, second.bins);
    let profile = mass(&first);
    assert_eq!(profile.len(), BARS);
    let total: f32 = profile[..BARS - 1].iter().sum();
    assert!((total - (BARS - 1) as f32).abs() < 1e-3, "death mass did not sum to {}", BARS - 1);
    assert_eq!(mass(&Barcode { bins: first.bins.clone(), components: 1 })[BARS - 1], 0.0);
    assert!(mass(&Barcode { bins: first.bins, components: 16 })[BARS - 1] > 0.5);
}

#[test]
fn h0_survives_an_empty_swarm_and_non_finite_positions() {
    let mut empty = substrate_at(0, 100.0, 1);
    assert_eq!(h0(&mut empty, 10.0).components, 0);
    let mut broken = substrate_at(3, 100.0, 1);
    broken.positions[0] = f32::NAN;
    assert!(h0(&mut broken, 10.0).components <= 3);
}

// metrics

#[test]
fn uniform_trait_conditioned_rdf_is_flat_against_a_measured_baseline() {
    let mut baseline = substrate_at(900, 100.0, 1);
    uniform_traits(&mut baseline, 1);
    let mut other = substrate_at(900, 100.0, 2);
    uniform_traits(&mut other, 2);
    for value in normalized_rdf(&raw_rdf(&other), &raw_rdf(&baseline)) {
        assert!((value - 1.0).abs() < 0.3, "{value}");
    }
}

#[test]
fn separated_biomes_differ_from_a_uniform_blend() {
    let mut biomes = substrate_at(256, 100.0, 1);
    let mut blend = substrate_at(256, 100.0, 1);
    for i in 0..256 {
        biomes.traits[i] = if i < 128 { 0.2 } else { 0.8 };
        blend.traits[i] = if i % 2 == 0 { 0.2 } else { 0.8 };
        let x = if i < 128 { 20.0 } else { 80.0 };
        for axis in 0..3 {
            biomes.positions[i * 3 + axis] = x + (i % 5) as f32;
            blend.positions[i * 3 + axis] = biomes.positions[i * 3 + axis];
        }
    }
    let local_biomes = spatial_field(&biomes, 4).heterogeneity();
    let local_blend = spatial_field(&blend, 4).heterogeneity();
    assert!(local_biomes[2] > local_blend[2] + 0.05, "{local_biomes:?} vs {local_blend:?}");
    assert!(local_biomes[3] + 0.45 < local_blend[3], "{local_biomes:?} vs {local_blend:?}");
}

#[test]
fn uniformly_mixed_traits_do_not_become_fine_scale_biomes() {
    for seed in 1..=12 {
        let mut substrate = substrate_at(1_200, 100.0, seed);
        uniform_traits(&mut substrate, seed);
        let local = heterogeneity(&substrate);
        for scale in local.chunks_exact(HETEROGENEITY_VALUES) {
            assert!(scale[2] < 0.01, "seed {seed}: {local:?}");
        }
    }
}

#[test]
fn a_compact_blob_fails_where_a_bicontinuous_slab_passes() {
    let box_len = 100.0;
    let planted = |place: fn(&mut Rng, f32) -> [f32; 3]| {
        let mut substrate = substrate_at(4_000, box_len, 9);
        let mut rng = Rng::new(11);
        for i in 0..4_000 {
            let position = place(&mut rng, box_len);
            for axis in 0..3 { substrate.positions[i * 3 + axis] = position[axis]; }
        }
        axiom::tuner::metrics::connectivity(&substrate)
    };
    let blob = planted(|rng, len| { // a ball around the box center, far from any face
        [len * 0.5 + rng.range(-12.0, 12.0), len * 0.5 + rng.range(-12.0, 12.0), len * 0.5 + rng.range(-12.0, 12.0)]
    });
    assert_eq!(blob.dense, [0.0; 2], "a compact blob wound the torus: {blob:?}");
    let slab = planted(|rng, len| [rng.unit() * len * 0.45, rng.unit() * len, rng.unit() * len]);
    for score in slab.dense.into_iter().chain(slab.void) {
        assert!(score > 0.5, "slab did not read bicontinuous: {slab:?}");
    }
}

#[test]
fn every_descriptor_axis_stays_in_the_frozen_span() {
    let observed = axiom::tuner::metrics::Observations {
        rdf: &axiom::tuner::metrics::Rdf { bins: [20.0; 18] },
        rdf_baseline: &axiom::tuner::metrics::Rdf { bins: [1.0; 18] },
        spatial_samples: &[vec![0.0; 37], vec![100.0; 37]],
        heterogeneity: [100.0; HETEROGENEITY_SIDES.len() * HETEROGENEITY_VALUES],
        connectivity: Connectivity { dense: [1.0; 2], void: [1.0; 2] },
        barcode: &Barcode { bins: vec![0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0], components: 1 },
        mobility: 100.0, turnover: 100.0, asymmetry: 100.0, robustness: 100.0,
    };
    let values = axiom::tuner::metrics::descriptor(&observed);
    assert_eq!(values.len(), descriptor_len());
    assert!(values.iter().all(|value| (0.0..=2.0).contains(value)));
}

// rollout

#[test]
fn tier_evaluation_is_deterministic_and_v5_sized_with_disjoint_tier_seeds() {
    let tuning = small_tuning();
    let probe = tuning.world.probe();
    let genome = tuning.world.default_genome(&probe);
    let budget = EvaluationBudget::new(EvaluationTier::Persistence, 40);
    let a = evaluate_tier(&tuning, &probe, &genome, budget);
    let b = evaluate_tier(&tuning, &probe, &genome, budget);
    assert_eq!(a.records.len(), 3);
    for (left, right) in a.records.iter().zip(&b.records) {
        assert_eq!(left.seed, right.seed);
        assert_eq!(left.metrics.descriptor, right.metrics.descriptor);
        assert!(left.metrics.descriptor.is_empty() || left.metrics.descriptor.len() == descriptor_len());
    }
    let persistence: Vec<u64> = a.records.iter().map(|record| record.seed).collect();
    let certification = evaluate_tier(&tuning, &probe, &genome,
        EvaluationBudget::new(EvaluationTier::Certification, 40));
    assert!(certification.records.iter().all(|record| !persistence.contains(&record.seed)),
        "certification reused a persistence seed");
}

// archive

#[test]
fn archive_round_trips_and_refreshes_current_novelty() {
    let tuning = small_tuning();
    let probe = tuning.world.probe();
    let genome = tuning.world.default_genome(&probe);
    let mut archive = Archive::new(tuning);
    let admission = |id: u64, x: f32, genome: Vec<f32>| {
        let mut descriptor = vec![1.0; descriptor_len()];
        descriptor[0] = x;
        Admission { genome, metrics: Metrics { descriptor, ..living() }, lineage_id: id,
            parent_lineage_id: None, birth_generation: 1, parent_source: ParentSource::Novelty,
            admission_novelty: x.abs() }
    };
    let mut second_genome = genome.clone();
    second_genome[0] = 10.0;
    archive.merge([admission(7, 0.0, genome), admission(9, 3.0, second_genome)]);
    let text = archive.to_text();
    let (restored, read_tuning) = Archive::from_text(&text).unwrap();
    assert_eq!(read_tuning.world.anchor_count, 2);
    assert!(restored.entries().iter().any(|entry| entry.lineage_id == 7));
    assert!(restored.entries().iter().all(|entry| entry.current_novelty.is_finite()));
}

#[test]
fn admission_novelty_is_immutable_provenance() {
    let tuning = small_tuning();
    let probe = tuning.world.probe();
    let genome = tuning.world.default_genome(&probe);
    let mut archive = Archive::new(tuning);
    let admission = |id: u64, x: f32| {
        let mut descriptor = vec![1.0; descriptor_len()];
        descriptor[0] = x;
        let mut entry_genome = genome.clone();
        entry_genome[0] = 3.0 + x;
        Admission { genome: entry_genome, metrics: Metrics { descriptor, ..living() }, lineage_id: id,
            parent_lineage_id: None, birth_generation: 1, parent_source: ParentSource::Novelty,
            admission_novelty: x }
    };
    archive.merge([admission(1, 0.5), admission(2, 1.0)]);
    let before = archive.entries().iter().find(|entry| entry.lineage_id == 1).unwrap().clone();
    archive.merge([admission(3, 0.6)]);
    let after = archive.entries().iter().find(|entry| entry.lineage_id == 1).unwrap();
    assert_eq!(after.admission_novelty, before.admission_novelty);
    assert_ne!(after.current_novelty, before.current_novelty);
}

#[test]
fn an_old_archive_format_is_rejected() {
    assert!(Archive::from_text("axiom-archive 5\n").is_err());
}

#[test]
fn the_archive_header_reads_without_parsing_entry_bodies() {
    let tuning = small_tuning();
    let expected = tuning.world.clone();
    let archive = Archive::new(tuning);
    let text = format!("{}entry deliberately malformed\n", archive.to_text());
    let restored = Archive::header(&text).unwrap();
    assert_eq!(restored.world.particle_count, expected.particle_count);
    assert_eq!(restored.world.anchor_count, expected.anchor_count);
}

// search

#[test]
fn seeded_parallel_generations_reproduce() {
    let tuning = small_tuning();
    let first = run(&tuning, |_, _, _| {});
    let second = run(&tuning, |_, _, _| {});
    let picture = |archive: &Archive| archive.entries().iter()
        .map(|entry| (entry.genome.clone(), entry.lineage_id, entry.current_novelty)).collect::<Vec<_>>();
    assert_eq!(picture(&first), picture(&second));
}

#[test]
fn archive_capacity_is_independent_of_viable_batch_order() {
    let capped = Tuning {
        discovery: Discovery { capacity: 2, ..Discovery::default() },
        ..Tuning::default()
    };
    let admissions = |order: &[u64]| order.iter().map(|&id| {
        let mut descriptor = vec![1.0; descriptor_len()];
        descriptor[0] = id as f32;
        Admission { genome: vec![id as f32], metrics: Metrics { descriptor, ..living() },
            lineage_id: id, parent_lineage_id: None, birth_generation: 1,
            parent_source: ParentSource::Novelty, admission_novelty: id as f32 }
    }).collect::<Vec<_>>();
    let mut left = Archive::new(capped.clone());
    left.merge(admissions(&[1, 2, 5, 9]));
    let mut right = Archive::new(capped);
    right.merge(admissions(&[9, 5, 2, 1]));
    let ids = |archive: &Archive| archive.entries().iter().map(|entry| entry.lineage_id).collect::<Vec<_>>();
    assert_eq!(ids(&left), ids(&right));
}

// ledger and campaign state

fn seed_record(value: f32) -> SeedRecord {
    let metrics = Metrics { turnover: value, descriptor: vec![value; descriptor_len()], ..living() };
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
    let metrics = Metrics { descriptor: vec![id as f32; descriptor_len()], ..living() };
    CampaignCandidate {
        genome: vec![id as f32],
        genome_hash: id,
        lineage_id: id,
        source: ParentSource::Random,
        descriptor: metrics.descriptor.clone(),
        features: feature_values(&metrics, margins(&metrics, &Gates::default()), &[], None),
    }
}

#[test]
fn discovery_imports_deduplicate_and_record_the_steps_actually_run() {
    let metrics = living();
    let record = EvaluationRecord {
        genome_hash: 4, lineage_id: 2, seed: 7,
        parent_source: ParentSource::Bootstrap, source_tier: SourceTier::Discovery,
        gate: viable(&metrics, &Gates::default()), metrics,
    };
    let mut ledger = CampaignLedger::default();
    ledger.append_search_discovery(&record, 2_345, &Gates::default());
    assert_eq!(ledger.records()[0].budget.steps, 2_345);
    ledger.append_search_discovery(&record, 2_345, &Gates::default());
    assert_eq!(ledger.records().len(), 1);
}

#[test]
fn persistence_rows_join_only_canonical_tier_outcomes_conservatively() {
    let mut ledger = CampaignLedger::default();
    let candidate = candidate(12);
    ledger.append_tier(&candidate, TierEvaluation {
        budget: EvaluationBudget::new(EvaluationTier::Discovery, 1_500),
        records: vec![seed_record(0.3)], passes: 1 });
    let mut short = EvaluationBudget::new(EvaluationTier::Persistence, 10_000);
    short.steps -= 1;
    ledger.append_tier(&candidate, TierEvaluation { budget: short, records: vec![seed_record(0.3); 3], passes: 3 });
    assert!(ledger.persistence_rows().is_empty(), "a short persistence run became a label");
    ledger.append_tier(&candidate, TierEvaluation {
        budget: EvaluationBudget::new(EvaluationTier::Persistence, 10_000),
        records: vec![seed_record(0.3); 3], passes: 3 });
    ledger.append_tier(&candidate, TierEvaluation {
        budget: EvaluationBudget::new(EvaluationTier::Persistence, 10_000),
        records: vec![seed_record(0.3); 3], passes: 1 });
    let rows = ledger.persistence_rows();
    assert_eq!(rows.len(), 1);
    assert!(!rows[0].survives, "one failed tier must make the label false");
    assert!(ledger.schema().accepts(&rows[0].features));
}

#[test]
fn campaign_validate_rejects_tiers_that_would_not_earn_their_name() {
    use axiom::tuner::campaign::validate;
    let short_discovery = Tuning {
        discovery: Discovery { steps: 1_499, ..Discovery::default() }, ..Tuning::default() };
    assert!(validate(&short_discovery).is_err());
    let short_persistence = Tuning {
        tiers: Tiers { persistence_steps: Some(9_999), ..Tiers::default() }, ..Tuning::default() };
    assert!(validate(&short_persistence).is_err());
    let certification_without_persistence = Tuning {
        tiers: Tiers { persistence_steps: None, certification_steps: Some(100_000) }, ..Tuning::default() };
    assert!(validate(&certification_without_persistence).is_err());
    assert!(validate(&Tuning::default()).is_ok());
}

#[test]
fn experiment_identity_excludes_search_scheduling_and_splits_on_gates() {
    let base = Tuning::default();
    let mut rescheduled = base.clone();
    rescheduled.discovery.seed = 999;
    rescheduled.discovery.batch = 3;
    rescheduled.discovery.capacity = 9;
    assert_eq!(ExperimentIdentity::for_tuning(&base), ExperimentIdentity::for_tuning(&rescheduled));
    let mut strict = base.clone();
    strict.gates.robustness_floor += 0.1;
    assert_ne!(ExperimentIdentity::for_tuning(&base), ExperimentIdentity::for_tuning(&strict));
    let mut reseeded = base.clone();
    reseeded.world.seed += 1;
    assert_ne!(ExperimentIdentity::for_tuning(&base), ExperimentIdentity::for_tuning(&reseeded));
}

#[test]
fn mismatched_state_is_rejected_before_search_work() {
    let tuning = small_tuning();
    let mut state = CampaignState::for_tuning(&tuning);
    let mut changed = tuning.clone();
    changed.world.rate += 1.0;
    changed.discovery.steps = 1_500;
    let mut called = false;
    let result = axiom::tuner::campaign::run(&changed, &Default::default(), &mut state, &mut |_, _, _| called = true);
    assert!(matches!(result, Err(axiom::tuner::campaign::CampaignError::ExperimentMismatch)));
    assert!(!called);
    assert!(state.ledger.records().is_empty());
}

#[test]
fn campaign_state_round_trips_failure_evidence_bit_for_bit() {
    let tuning = Tuning::default();
    let mut state = CampaignState::for_tuning(&tuning);
    let mut failed = seed_record(0.0);
    failed.early_stop = Some(axiom::tuner::rollout::EarlyStop::NonFinite);
    state.ledger.append_tier(&candidate(7), TierEvaluation {
        budget: EvaluationBudget::new(EvaluationTier::Discovery, 1_500),
        records: vec![failed], passes: 0 });
    let path = std::env::temp_dir().join(format!("axiom-campaign-{}", std::process::id()));
    state.save(&path).unwrap();
    let loaded = CampaignState::load(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    assert_eq!(loaded.ledger.records().len(), 1);
    assert_eq!(loaded.ledger.records()[0].features[0].to_bits(), state.ledger.records()[0].features[0].to_bits());
    assert_eq!(loaded.identity(), state.identity());
}

#[test]
fn corrupt_campaign_state_is_rejected_by_checksum() {
    let state = CampaignState::for_tuning(&Tuning::default());
    let path = std::env::temp_dir().join(format!("axiom-campaign-corrupt-{}", std::process::id()));
    state.save(&path).unwrap();
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[20] ^= 0x40;
    std::fs::write(&path, bytes).unwrap();
    assert!(matches!(CampaignState::load(&path),
        Err(axiom::tuner::state::CampaignStateError::CorruptChecksum { .. })));
    let _ = std::fs::remove_file(&path);
}

#[test]
fn scheduling_falls_back_uniformly_without_authority() {
    use axiom::tuner::campaign::{schedule, PersistenceModel};
    let choices = (1..=6).map(candidate).collect::<Vec<_>>();
    let picked = schedule(&choices, &PersistenceModel { model: None, rows: 0 },
        ContinuationQuotas { high_survival: 2, high_uncertainty: 2, uniform_neighborhood: 2 });
    assert_eq!(picked.len(), 6);
    assert_eq!(picked[0].genome_hash, 1);
}

// learning

fn mix(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[test]
fn grouped_split_keeps_lineages_whole() {
    let rows = (0..80u64).flat_map(|lineage_id| (0..3).map(move |copy| FeatureRow {
        id: lineage_id * 10 + copy, lineage_id, neighborhood: 0,
        features: vec![copy as f32], survives: copy % 2 == 0,
    })).collect::<Vec<_>>();
    let split = lineage_split(&rows, 9);
    for lineage in 0..80u64 {
        let memberships: Vec<Partition> = rows.iter().enumerate()
            .filter(|(_, row)| row.lineage_id == lineage)
            .filter_map(|(index, _)| split.partition(index)).collect();
        assert_eq!(memberships.len(), 3);
        assert!(memberships.iter().all(|partition| *partition == memberships[0]));
    }
}

#[test]
fn persistence_signal_beats_uniform_and_gains_authority() {
    let schema = FeatureSchema::new(["signal", "noise"]);
    let rows: Vec<FeatureRow> = (0..160u64).map(|lineage_id| {
        let signal = (lineage_id % 16) as f32 - 7.5;
        FeatureRow { id: lineage_id, lineage_id, neighborhood: lineage_id % 5,
            features: vec![signal, (mix(lineage_id) % 7) as f32], survives: signal > 0.0 }
    }).collect();
    let split = lineage_split(&rows, 31);
    let model = fit_persistence(schema, &rows, &split, LogisticConfig::default(), 4).unwrap();
    assert!(model.authority.active, "{:#?}", model.authority);
    assert!(model.authority.brier < model.authority.uniform_brier);
    let low = model.predict(&[-6.0, 2.0]).unwrap();
    let high = model.predict(&[6.0, 2.0]).unwrap();
    assert!(high.probability > low.probability + 0.5);
}

#[test]
fn a_useless_persistence_model_loses_authority() {
    let schema = FeatureSchema::new(["signal", "noise"]);
    let rows: Vec<FeatureRow> = (0..240u64).map(|lineage_id| FeatureRow {
        id: lineage_id, lineage_id, neighborhood: lineage_id % 3,
        features: vec![1.0, 1.0], survives: mix(lineage_id) & 1 == 0,
    }).collect();
    let split = lineage_split(&rows, 11);
    let model = fit_persistence(schema, &rows, &split, LogisticConfig::default(), 2).unwrap();
    assert!(!model.authority.active, "{:#?}", model.authority);
}

#[test]
fn continuation_selection_keeps_fixed_quotas_and_uniform_fallback() {
    let entry = |id, neighborhood, probability, uncertainty| ContinuationCandidate {
        id, neighborhood, prediction: PersistencePrediction { probability, uncertainty } };
    let candidates = vec![entry(1, 0, 0.9, 0.1), entry(2, 0, 0.8, 0.9), entry(3, 1, 0.2, 0.2), entry(4, 2, 0.1, 0.3)];
    let quotas = ContinuationQuotas { high_survival: 1, high_uncertainty: 1, uniform_neighborhood: 1 };
    let learned = select_continuations(&candidates, quotas, true);
    assert_eq!(learned.len(), 3);
    assert!(learned.contains(&1) && learned.contains(&2));
    assert_eq!(select_continuations(&candidates, quotas, false), vec![1, 3, 4]);
}

// checkpoints

fn tiny_caps() -> Caps {
    Caps { particle_count: 2, anchor_count: 2, radius: 2.0, rate: 10.0, seed: 41, shells: 1, bumps: 1, ..Caps::default() }
}

fn tiny_snapshot() -> (axiom::tuner::checkpoint::WorldManifest, WorldState, Sim, Caps, Vec<f32>) {
    let caps = tiny_caps();
    let probe = caps.probe();
    let genome = caps.default_genome(&probe);
    let mut sim = Sim::new(&caps.params(&genome, &probe));
    sim.run(12);
    let recipe = axiom::render_recipe::RenderRecipe {
        support: sim.params.box_len * 0.1, ..axiom::render_recipe::RenderRecipe::default() };
    let (manifest, state) = snapshot_world(SnapshotMetadata {
        world_id: "tiny-reef".into(), checkpoint_id: "tick-12".into(), parent_checkpoint_id: None,
        simulator_version: checkpoint::SIMULATOR_VERSION,
        descriptor_version: axiom::tuner::metrics::DESCRIPTOR_VERSION,
        genome_layout_version: checkpoint::GENOME_LAYOUT_VERSION,
        render_recipe_version: axiom::render_recipe::VERSION,
        render_recipe: recipe,
    }, &caps, genome.clone(), &sim).unwrap();
    (manifest, state, sim, caps, genome)
}

#[test]
fn checkpoint_state_round_trips_every_bit() {
    let (_, state, _, _, _) = tiny_snapshot();
    let path = std::env::temp_dir().join(format!("axiom-state-{}", std::process::id()));
    save_state(&path, &state).unwrap();
    let loaded = load_state(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    let bits = |values: &[f32]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    assert_eq!(loaded.tick, state.tick);
    assert_eq!(bits(&loaded.positions), bits(&state.positions));
    assert_eq!(bits(&loaded.traits), bits(&state.traits));
    assert_eq!(bits(&loaded.genome), bits(&state.genome));
}

#[test]
fn snapshot_restore_continues_bit_exactly_for_ten_thousand_steps() {
    let (manifest, state, mut original, _, _) = tiny_snapshot();
    let root = std::env::temp_dir().join(format!("axiom-continuation-{}", std::process::id()));
    let (manifest_path, state_path) = save_world(&root, &manifest, &state).unwrap();
    let (loaded_manifest, loaded_state) = checkpoint::load_world(&manifest_path, &state_path).unwrap();
    let mut restored = restore_world(&loaded_manifest, &loaded_state).unwrap();
    let bits = |sim: &Sim| sim.substrate.positions.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&original), bits(&restored));
    original.run(10_000);
    restored.run(10_000);
    assert_eq!(original.tick, restored.tick);
    assert_eq!(bits(&original), bits(&restored));
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn checkpoint_corruption_and_version_mismatches_are_named_errors() {
    use axiom::tuner::checkpoint::Error;
    let (mut manifest, state, _, _, _) = tiny_snapshot();
    let state_path = std::env::temp_dir().join(format!("axiom-corrupt-state-{}", std::process::id()));
    save_state(&state_path, &state).unwrap();
    let mut bytes = std::fs::read(&state_path).unwrap();
    bytes[20] ^= 0x40;
    std::fs::write(&state_path, &bytes).unwrap();
    assert!(matches!(load_state(&state_path), Err(Error::CorruptChecksum { .. })));
    let _ = std::fs::remove_file(&state_path);

    let manifest_path = std::env::temp_dir().join(format!("axiom-corrupt-manifest-{}", std::process::id()));
    save_manifest(&manifest_path, &manifest).unwrap();
    let mut bytes = std::fs::read(&manifest_path).unwrap();
    bytes[20] ^= 0x40;
    std::fs::write(&manifest_path, &bytes).unwrap();
    assert!(matches!(load_manifest(&manifest_path), Err(Error::CorruptChecksum { .. })));
    let _ = std::fs::remove_file(&manifest_path);

    manifest.simulator_version += 1;
    assert!(matches!(manifest.validate(), Err(Error::UnsupportedVersion { kind: "simulator", .. })));
}

#[test]
fn checkpoint_ids_are_immutable_while_the_manifest_updates() {
    let (manifest, state, _, _, _) = tiny_snapshot();
    let root = std::env::temp_dir().join(format!("axiom-immutable-{}", std::process::id()));
    save_world(&root, &manifest, &state).unwrap();
    save_world(&root, &manifest, &state).unwrap(); // identical bytes may republish
    let mut conflicting = state.clone();
    conflicting.positions[0] += 1.0;
    let mut conflicting_manifest = manifest.clone();
    conflicting_manifest.tick = conflicting.tick;
    assert!(matches!(save_world(&root, &conflicting_manifest, &conflicting),
        Err(axiom::tuner::checkpoint::Error::ImmutableCheckpoint { .. })));
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn a_stale_genome_cannot_be_snapshotted_over_a_different_world() {
    let (_, _, sim, caps, genome) = tiny_snapshot();
    let mut stale = genome;
    let last = stale.len() - 1;
    stale[last] = (stale[last] + 1.0).min(100.0);
    let result = snapshot_world(SnapshotMetadata {
        world_id: "stale".into(), checkpoint_id: "tick-0".into(), parent_checkpoint_id: None,
        simulator_version: checkpoint::SIMULATOR_VERSION,
        descriptor_version: axiom::tuner::metrics::DESCRIPTOR_VERSION,
        genome_layout_version: checkpoint::GENOME_LAYOUT_VERSION,
        render_recipe_version: axiom::render_recipe::VERSION,
        render_recipe: axiom::render_recipe::RenderRecipe {
            support: sim.params.box_len * 0.1, ..axiom::render_recipe::RenderRecipe::default() },
    }, &caps, stale, &sim);
    assert!(matches!(result, Err(axiom::tuner::checkpoint::Error::ManifestStateMismatch { field: "world recipe" })));
}

#[test]
fn non_finite_and_mismatched_state_have_named_errors() {
    use axiom::tuner::checkpoint::Error;
    let (_, state, _, _, _) = tiny_snapshot();
    let mut bad = state.clone();
    bad.traits[0] = f32::NAN;
    assert!(matches!(bad.validate(), Err(Error::NonFinite { field: "traits" })));
    let mut bad = state.clone();
    bad.traits.pop();
    assert!(matches!(bad.validate(), Err(Error::ParticleCountMismatch { .. })));
    let mut bad = state;
    bad.traits[0] = 1.1;
    assert!(matches!(bad.validate(), Err(Error::TraitOutOfRange)));
}

// render recipe

#[test]
fn a_derived_render_recipe_is_valid_and_resolves_its_own_support() {
    use axiom::render_recipe::{RenderRecipe, MAX_RESOLUTION};
    for count in [300usize, 1_000, 2_500, 10_000, 50_000] {
        let caps = Caps { particle_count: count, ..Caps::default() };
        let box_len = caps.probe().box_len(9.0);
        let recipe = RenderRecipe::for_world(box_len, count);
        assert!(recipe.valid(), "count {count} -> {recipe:?}");
        let voxel = box_len / recipe.resolution as f32;
        assert!(recipe.resolution == MAX_RESOLUTION || voxel <= recipe.support / 6.0 * 1.001,
            "count {count}: voxel {voxel} undersamples support {}", recipe.support);
    }
    assert!(!RenderRecipe { resolution: MAX_RESOLUTION + 1, ..RenderRecipe::default() }.valid());
    assert!(RenderRecipe::default().valid());
}

// genome codec

#[test]
fn the_codec_is_shaped_right_and_decoding_is_idempotent() {
    let caps = tiny_caps();
    let probe = caps.probe();
    let genome = caps.default_genome(&probe);
    assert_eq!(genome.len(), caps.gene_len());
    let bounds = caps.bounds(&probe);
    assert_eq!(bounds.len(), caps.gene_len());
    for (gene, &(low, high)) in genome.iter().zip(&bounds) {
        assert!(*gene >= low && *gene <= high, "default gene {gene} escapes [{low}, {high}]");
    }
    let bits = |values: &[f32]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    let once = caps.params(&genome, &probe);
    let twice = caps.params_at(&genome, once.box_len);
    assert_eq!(once.coordination.to_bits(), twice.coordination.to_bits());
    assert_eq!(bits(&once.trait_distribution), bits(&twice.trait_distribution));
    assert_eq!(bits(&once.interactions), bits(&twice.interactions), "decode is not idempotent");
}
