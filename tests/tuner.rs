//! Tuner behavior: how a plan is assembled, what a gate actually rejects, how a metric that reads
//! other metrics finds them, and that a rollout is reproducible. Every case here covers a mechanism
//! that has silently done nothing before.

use axiom::engine::matrix::Matrix;
use axiom::engine::params::{FixedGenome, Genome};
use axiom::engine::resolve::Probe;
use axiom::engine::substrate::Substrate;
use axiom::tuner::gate::{valid, Gate};
use axiom::tuner::metrics::{evaluate, heterogeneity, mobility, plan, rdf, robustness, scalar,
    slice, structure, temporal, Blocks, Metric, Rollout, ALL};
use axiom::run;

fn small_genome() -> FixedGenome {
    FixedGenome { particle_count: 120, dimensions: 3, anchor_count: 2, shells: 1, bumps: 1,
        radius: 2.0, dt: 0.05, seed: 7 }
}

/// The uniform starting cloud and the law over it. No stepping, so every case built on this is
/// exact rather than dependent on where a genome happens to go.
struct Fixture {
    fixed_genome: FixedGenome,
    substrate: Substrate,
    matrix: Matrix,
    baseline: rdf::Histogram,
}
fn fixture() -> Fixture {
    let fixed_genome = small_genome();
    let probe = Probe::new(&fixed_genome);
    let genes = Genome::build_random(&fixed_genome.bounds(&probe), 11);
    let genome = fixed_genome.decode(&genes);
    let box_len = probe.box_len(genome.coordination);
    let substrate = Substrate::build(&fixed_genome, box_len);
    let matrix = Matrix::derive(&fixed_genome, &genome);
    let baseline = rdf::build(&substrate);
    Fixture { fixed_genome, substrate, matrix, baseline }
}
/// One sampled tick of a plan on that cloud. 'full' is the final-sample flag a rollout passes.
fn measured(assembled: &[Metric], full: bool) -> Vec<f32> {
    let mut start = fixture();
    let rollout = Rollout { fixed_genome: &start.fixed_genome, matrix: &start.matrix,
        baseline: &start.baseline };
    evaluate(&Blocks::build(&rollout, &mut start.substrate, assembled, None, &[]), full)
}
fn width_of(assembled: &[Metric]) -> usize { assembled.iter().map(|m| m.width).sum() }

// plan assembly

#[test]
fn plan_pulls_in_what_a_wanted_metric_reads() {
    let assembled = plan(&[temporal::METRIC]);
    for dependency in temporal::METRIC.depends {
        assert!(assembled.iter().any(|m| m.key == dependency.key), "the plan is missing {}", dependency.key);
    }
}

#[test]
fn plan_places_every_dependency_ahead_of_its_dependent() {
    let assembled = plan(&ALL);
    for (index, metric) in assembled.iter().enumerate() {
        for dependency in metric.depends {
            let at = assembled.iter().position(|other| other.key == dependency.key).expect("dependency present");
            assert!(at < index, "{} reads {} but sits ahead of it", metric.key, dependency.key);
        }
    }
}

#[test]
fn plan_names_each_metric_once_however_many_asked_for_it() {
    let assembled = plan(&[temporal::METRIC, rdf::METRIC, rdf::METRIC, structure::METRIC]);
    let named: Vec<&str> = assembled.iter().map(|metric| metric.key).collect();
    let mut once = named.clone();
    once.dedup();
    assert_eq!(once, named, "a metric appears twice in {named:?}");
}

#[test]
fn descriptor_width_is_the_sum_of_the_plans_metrics() {
    let assembled = plan(&[rdf::METRIC, structure::METRIC]);
    assert_eq!(measured(&assembled, true).len(), width_of(&assembled));
}

#[test]
fn every_metric_reads_onto_the_shared_axis() {
    let assembled = plan(&ALL);
    let probe = Probe::new(&small_genome());
    let bounds = small_genome().bounds(&probe);
    let values = (0..8)
        .map(|seed| run::sim(&small_genome(), &assembled, &[], &probe, &Genome::build_random(&bounds, seed), 60))
        .find(|values| !values.is_empty())
        .expect("no genome out of eight produced a measurable sim");
    for metric in &assembled {
        for &slot in slice(&values, &assembled, metric) {
            assert!((0.0..=1.0).contains(&slot), "{} read {slot}, off the shared axis", metric.key);
        }
    }
}

// gates

#[test]
fn a_clause_rejects_a_sim_below_its_floor_and_above_its_ceiling() {
    let assembled = plan(&[rdf::METRIC, structure::METRIC]);
    let values = measured(&assembled, true);
    let reading = scalar(&values, &assembled, structure::METRIC);

    let inside = [Gate { metric: structure::METRIC, floor: reading - 0.1, ceiling: reading + 0.1 }];
    let too_high = [Gate { metric: structure::METRIC, floor: reading + 0.1, ceiling: 1.0 }];
    let too_low = [Gate { metric: structure::METRIC, floor: 0.0, ceiling: reading - 0.1 }];
    assert!(valid(&values, &assembled, &inside, false), "structure {reading} rejected by a band around it");
    assert!(!valid(&values, &assembled, &too_high, false), "structure {reading} cleared a floor above it");
    assert!(!valid(&values, &assembled, &too_low, false), "structure {reading} cleared a ceiling below it");
}

#[test]
fn an_empty_sim_clears_no_clause_with_a_floor() {
    let gates = [Gate { metric: structure::METRIC, floor: 0.05, ceiling: 1.0 }];
    assert!(!valid(&[], &plan(&[structure::METRIC]), &gates, false));
}

// the once schedule

#[test]
fn a_once_metric_holds_its_slots_before_the_final_sample() {
    let assembled = plan(&[rdf::METRIC, robustness::METRIC]);
    let partial = measured(&assembled, false);
    assert_eq!(slice(&partial, &assembled, robustness::METRIC).len(), robustness::METRIC.width,
        "spans would drift between samples");
    assert!(slice(&partial, &assembled, robustness::METRIC).iter().all(|&value| value == 0.0),
        "the experiment ran early");
    assert_eq!(partial.len(), width_of(&assembled));
}

#[test]
fn a_partial_reading_skips_a_once_clause_and_still_judges_the_rest() {
    let assembled = plan(&[rdf::METRIC, structure::METRIC, robustness::METRIC]);
    let partial = measured(&assembled, false);
    let reading = scalar(&partial, &assembled, structure::METRIC);
    let repair = [Gate { metric: robustness::METRIC, floor: 0.5, ceiling: 1.0 }];
    let both = [Gate { metric: robustness::METRIC, floor: 0.5, ceiling: 1.0 },
        Gate { metric: structure::METRIC, floor: reading + 0.1, ceiling: 1.0 }];
    assert!(valid(&partial, &assembled, &repair, true), "an unmeasured experiment failed a mid-rollout check");
    assert!(!valid(&partial, &assembled, &repair, false), "an unmeasured experiment cleared a final check");
    assert!(!valid(&partial, &assembled, &both, true), "a partial check skipped a clause it could judge");
}

// reading one metric out of another

#[test]
fn a_dependent_metric_finds_its_axes_in_the_descriptor_it_is_handed() {
    let assembled = plan(&[temporal::METRIC]);
    let mut start = fixture();
    let rollout = Rollout { fixed_genome: &start.fixed_genome, matrix: &start.matrix,
        baseline: &start.baseline };
    let first = evaluate(&Blocks::build(&rollout, &mut start.substrate, &assembled, None, &[]), true);
    // squeeze the cloud into a corner, so the second tick genuinely looks unlike the first
    for coordinate in start.substrate.positions.iter_mut() { *coordinate *= 0.5; }
    let history = vec![first];
    let second = evaluate(&Blocks::build(&rollout, &mut start.substrate, &assembled, None, &history), true);
    let spread = slice(&second, &assembled, temporal::METRIC)[0];
    assert!(spread > 0.0, "temporal read a flat picture, so it never found its axes in the descriptor");
}

// rollouts

#[test]
fn the_same_genome_measures_identically_twice() {
    let fixed = small_genome();
    let probe = Probe::new(&fixed);
    let assembled = plan(&[rdf::METRIC, structure::METRIC, heterogeneity::METRIC, mobility::METRIC]);
    let genes = Genome::build_random(&fixed.bounds(&probe), 11);
    let first = run::sim(&fixed, &assembled, &[], &probe, &genes, 40);
    let second = run::sim(&fixed, &assembled, &[], &probe, &genes, 40);
    assert_eq!(first.is_empty(), second.is_empty());
    assert_eq!(first, second, "a rollout is not reproducible");
}

#[test]
fn some_genome_survives_a_rollout_and_fills_its_descriptor() {
    let fixed = small_genome();
    let probe = Probe::new(&fixed);
    let bounds = fixed.bounds(&probe);
    let assembled = plan(&[rdf::METRIC, structure::METRIC, heterogeneity::METRIC, mobility::METRIC]);
    let measured = (0..8)
        .map(|seed| run::sim(&fixed, &assembled, &[], &probe, &Genome::build_random(&bounds, seed), 40))
        .find(|values| !values.is_empty())
        .expect("no genome out of eight produced a measurable sim");
    assert_eq!(measured.len(), width_of(&assembled));
    assert!(measured.iter().all(|value| value.is_finite()));
}

// scoring

/// The failure this guards: every clause of the structure score multiplied raw, so one zero
/// annihilated the product. Nearly every genome scored exactly 0, a tournament broke the ties
/// toward its first draw, and the search quietly became random.
#[test]
fn the_structure_criterion_tells_genomes_apart() {
    use axiom::tuner::criterion::structure as fitness;
    let fixed = small_genome();
    let probe = Probe::new(&fixed);
    let bounds = fixed.bounds(&probe);
    let assembled = plan(&fitness::METRICS);
    let scored: Vec<f32> = (0..6)
        .map(|seed| run::sim(&fixed, &assembled, &[], &probe, &Genome::build_random(&bounds, seed), 40))
        .filter(|values| !values.is_empty())
        .map(|values| fitness::score(&values, &assembled))
        .collect();

    assert!(scored.len() >= 4, "only {} of 6 genomes were measurable", scored.len());
    assert!(scored.iter().all(|score| *score > 0.0), "a clause annihilated the product: {scored:?}");
    let mut distinct: Vec<String> = scored.iter().map(|score| format!("{score:.9}")).collect();
    distinct.sort();
    distinct.dedup();
    assert!(distinct.len() >= scored.len() - 1, "scores barely differ, so a search cannot climb: {scored:?}");
}
