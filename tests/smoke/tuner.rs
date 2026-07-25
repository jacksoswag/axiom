//! Tuner behavior: how a plan is assembled, what each metric actually reads off a picture, what a
//! gate rejects, how a rollout gives up, and what a search keeps. Every case here covers a mechanism
//! that has silently done nothing before, or one whose answer is a number nobody could eyeball.

use axiom::engine::matrix::Matrix;
use axiom::engine::resolve::Probe;
use axiom::engine::params::Genome;
use axiom::harness::rollout;
use axiom::tuner::algorithms::evolve::{Evolve, Mutation};
use axiom::tuner::algorithms::Algorithm;
use axiom::tuner::criterion::{novelty, structure as fitness, Criterion};
use axiom::tuner::driver::{Search, Tally};
use axiom::tuner::gate::{valid, Gate};
use axiom::tuner::metrics::{evaluate, fold, log_bin, plan, scalar, slice, wants_baseline,
    asymmetry, connectivity, field, heterogeneity, mobility, rdf, robustness, structure, temporal,
    topology, turnover, Blocks, Metric, Motion, Reduce, Rollout, ALL};
use axiom::tuner::specimen::Specimen;
use axiom::util::Rng;

use crate::fixture;

/// One sampled tick of a plan on a world. 'full' is the final-sample flag a rollout passes, which is
/// what pays for the metrics too costly to read on every look.
fn read(world: &mut fixture::World, assembled: &[Metric], full: bool) -> Vec<f32> {
    let rollout = Rollout { fixed_genome: &world.fixed, matrix: &world.matrix, baseline: &world.baseline };
    evaluate(&Blocks::build(&rollout, &mut world.substrate, assembled, None, &[]), full)
}
/// The same, on the uniform starting cloud every case shares
fn measured(assembled: &[Metric], full: bool) -> Vec<f32> { read(&mut fixture::world(120, 11), assembled, full) }
fn width_of(assembled: &[Metric]) -> usize { assembled.iter().map(|metric| metric.width).sum() }
/// Scatter the population where a case wants it, so a metric is read on a picture with a known
/// answer rather than wherever a genome happened to wander.
fn place(world: &mut fixture::World, seed: u64, spread: impl Fn(usize, f32, &mut Rng) -> [f32; 3]) {
    let (box_len, mut rng) = (world.substrate.box_len, Rng::new(seed));
    for particle in 0..world.substrate.traits.len() {
        let placed = spread(particle, box_len, &mut rng);
        world.substrate.positions[particle * 3..particle * 3 + 3].copy_from_slice(&placed);
    }
}

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

/// Width is the layout: a metric that returns a different count of slots than it declared shifts
/// every metric behind it, and every reading of every one of them silently means something else.
#[test]
fn every_metric_fills_exactly_the_slots_it_declared() {
    let assembled = plan(&ALL);
    let values = measured(&assembled, true);
    assert_eq!(values.len(), width_of(&assembled));
    for metric in &assembled {
        let held = slice(&values, &assembled, metric);
        assert_eq!(held.len(), metric.width, "{} filled {} of its {} slots", metric.key, held.len(), metric.width);
        assert!(held.iter().all(|value| value.is_finite()), "{} read {held:?}", metric.key);
    }
}

/// The shared axis is what lets a distance compare two descriptors at all. One metric off it, on
/// one unlucky world, and every distance in the search is measured with a ruler of its own.
#[test]
fn every_metric_reads_onto_the_shared_axis() {
    let assembled = plan(&ALL);
    let fixed = fixture::shape(120);
    let probe = Probe::new(&fixed);
    let bounds = fixed.bounds(&probe);
    let mut seen = 0;
    for seed in 0..12u64 {
        let values = rollout::sim(&fixed, &assembled, &[], &probe, &Genome::build_random(&bounds, seed), 60);
        if values.is_empty() { continue; }
        seen += 1;
        for metric in &assembled {
            for &slot in slice(&values, &assembled, metric) {
                assert!((0.0..=1.0).contains(&slot), "{} read {slot} on seed {seed}, off the shared axis", metric.key);
            }
        }
    }
    assert!(seen >= 3, "only {seen} of twelve genomes were measurable, so this proved nothing");
}

#[test]
fn only_a_plan_that_compares_against_uniform_pays_for_a_baseline() {
    assert!(wants_baseline(&plan(&[structure::METRIC])), "structure reads the pair histogram");
    assert!(wants_baseline(&plan(&[robustness::METRIC])), "the repair experiment signs with one");
    assert!(!wants_baseline(&plan(&[mobility::METRIC])), "mobility paid for an all-pairs pass it never reads");
    assert!(!wants_baseline(&plan(&[asymmetry::METRIC])), "asymmetry paid for an all-pairs pass it never reads");
}

#[test]
fn a_log_bin_lands_between_the_edges_it_was_given() {
    assert_eq!(log_bin(0.5, 1.0, 100.0, 8), 0, "below the first edge belongs in the first bin");
    assert_eq!(log_bin(1.0, 1.0, 100.0, 8), 0);
    assert_eq!(log_bin(10.0, 1.0, 100.0, 8), 4, "the geometric middle sits in the middle bin");
    assert_eq!(log_bin(1000.0, 1.0, 100.0, 8), 7, "past the last edge belongs in the last bin");
    assert_eq!(log_bin(f32::NAN, 1.0, 100.0, 8), 0, "an unreadable distance indexed off the end");
    let bins: Vec<usize> = (1..=20).map(|step| log_bin(step as f32 * 5.0, 1.0, 100.0, 8)).collect();
    assert!(bins.windows(2).all(|pair| pair[0] <= pair[1]), "a farther distance landed in a nearer bin: {bins:?}");
}

// what each metric reads off a picture

/// A cloud measured against itself is by definition featureless, which is the one reading of the
/// radial ratios that can be worked out by hand: every bin exactly as full as the reference.
#[test]
fn a_cloud_measured_against_itself_reads_as_featureless() {
    let assembled = plan(&[rdf::METRIC, structure::METRIC]);
    let values = measured(&assembled, true);
    for &slot in slice(&values, &assembled, rdf::METRIC) {
        assert!((slot - 0.5).abs() < 1e-5, "a ratio of one read {slot} rather than the middle of the axis");
    }
    assert!(scalar(&values, &assembled, structure::METRIC) < 1e-5, "a gas read as structured");
}

/// The Poisson correction is what keeps a fine grid from calling every finite swarm heterogeneous,
/// and a swarm genuinely in one corner still has to outrank a uniform one at every scale.
#[test]
fn a_uniform_cloud_is_told_apart_from_a_clumped_one_at_every_scale() {
    let assembled = plan(&[heterogeneity::METRIC]);
    let mut world = fixture::world(2000, 11);
    place(&mut world, 5, |_, box_len, rng| [rng.unit() * box_len, rng.unit() * box_len, rng.unit() * box_len]);
    let uniform = read(&mut world, &assembled, true);
    place(&mut world, 5, |_, box_len, rng| std::array::from_fn(|_| rng.unit() * box_len * 0.25));
    let clumped = read(&mut world, &assembled, true);
    for (scale, side) in heterogeneity::SIDES.into_iter().enumerate() {
        let contrast = scale * heterogeneity::VALUES;
        assert!(uniform[contrast] < 0.05, "a uniform cloud read {} of density contrast at side {side}", uniform[contrast]);
        assert!(clumped[contrast] > 0.5, "a swarm in one corner read {} of density contrast at side {side}",
            clumped[contrast]);
        let void = contrast + 1;
        assert!(clumped[void] > uniform[void], "a corner swarm left no more of the box empty than a uniform one");
    }
}

/// A phase counts as a medium only where it comes back to itself around the torus. A compact body is
/// disconnected here however many of its cells touch, which is the whole difference between a
/// structure something can travel through and a lump.
#[test]
fn a_phase_counts_only_where_it_loops_the_box() {
    let assembled = plan(&[connectivity::METRIC]);
    let mut world = fixture::world(512, 11);
    // A slab: spread across two axes, thin on the third, so it wraps in x and y and nowhere else.
    place(&mut world, 5, |_, box_len, rng| [rng.unit() * box_len, rng.unit() * box_len, rng.unit() * box_len / 8.0]);
    let slab = read(&mut world, &assembled, true);
    // The same particles gathered into one corner: as many of them, touching, and going nowhere.
    place(&mut world, 5, |_, box_len, rng| std::array::from_fn(|_| rng.unit() * box_len / 8.0));
    let blob = read(&mut world, &assembled, true);
    let half = slab.len() / 2; // dense fractions first, then void
    assert!(slab[..half].iter().all(|&value| value > 0.0), "a slab that loops the box read as disconnected: {slab:?}");
    assert!(blob[..half].iter().all(|&value| value == 0.0), "a corner blob read as a connected medium: {blob:?}");
    assert!(blob[half..].iter().all(|&value| value > 0.0), "the empty space around a blob read as disconnected");
}

/// Bodies no kernel can reach across are not one structure at any scale, and the component count is
/// what the last barcode axis is built out of.
#[test]
fn separate_bodies_stay_separate_until_the_scale_reaches_across() {
    let mut world = fixture::world(200, 11);
    place(&mut world, 5, |particle, box_len, rng| {
        let corner = if particle < 100 { 0.0 } else { box_len * 0.5 }; // two clumps, half a box apart
        std::array::from_fn(|_| corner + rng.unit() * box_len * 0.05)
    });
    let box_len = world.substrate.box_len;
    let apart = topology::h0(&mut world.substrate, box_len * 0.1);
    assert_eq!(apart.components, 2, "two clumps a half box apart read as {} bodies", apart.components);
    assert_eq!(apart.bins.iter().sum::<f64>() as usize + apart.components, 200,
        "the bars and the components do not add up to the swarm");
    let across = topology::h0(&mut world.substrate, box_len);
    assert_eq!(across.components, 1, "a scale that reaches everything still read {} bodies", across.components);
    assert_eq!(topology::h0(&mut world.substrate, 0.0).components, 0, "a law reaching nowhere found bodies");
}

/// Both motion metrics read the sample before this one, and both have to say nothing rather than
/// something plausible when there is no sample behind them.
#[test]
fn the_motion_metrics_read_the_sample_behind_them() {
    let assembled = plan(&[mobility::METRIC, turnover::METRIC]);
    let mut world = fixture::world(400, 11);
    let opening = read(&mut world, &assembled, true);
    assert_eq!(scalar(&opening, &assembled, turnover::METRIC), 0.0, "the first sample turned over against nothing");
    assert!(scalar(&opening, &assembled, mobility::METRIC) < 0.05, "the first sample travelled somewhere");

    let was = Motion { positions: world.substrate.positions.clone(),
        field: Some(field::build(&world.substrate, turnover::SIDE)) };
    let still = { // the same population, not one particle moved
        let rollout = Rollout { fixed_genome: &world.fixed, matrix: &world.matrix, baseline: &world.baseline };
        let blocks = Blocks::build(&rollout, &mut world.substrate, &assembled, Some(was), &[]);
        evaluate(&blocks, true)
    };
    assert_eq!(scalar(&still, &assembled, turnover::METRIC), 0.0, "a frozen world turned over");
    assert!(scalar(&still, &assembled, mobility::METRIC) < 0.05, "a frozen world travelled");

    let was = Motion { positions: world.substrate.positions.clone(),
        field: Some(field::build(&world.substrate, turnover::SIDE)) };
    place(&mut world, 9, |_, box_len, rng| std::array::from_fn(|_| rng.unit() * box_len)); // redrawn everywhere
    let moved = {
        let rollout = Rollout { fixed_genome: &world.fixed, matrix: &world.matrix, baseline: &world.baseline };
        let blocks = Blocks::build(&rollout, &mut world.substrate, &assembled, Some(was), &[]);
        evaluate(&blocks, true)
    };
    assert!(scalar(&moved, &assembled, turnover::METRIC) > 0.3, "a redrawn cloud held its arrangement");
    assert!(scalar(&moved, &assembled, mobility::METRIC) > 0.5, "a redrawn cloud travelled nowhere");
}

/// The one metric that reads the law rather than the sim, so it is the only one whose answer can be
/// set exactly: a matrix equal to its own transpose is symmetric by construction.
#[test]
fn a_law_that_pulls_both_ways_alike_reads_symmetric() {
    let assembled = plan(&[asymmetry::METRIC]);
    let mut world = fixture::world(120, 11);
    let (_, mut genes) = fixture::drawn(&world.fixed, 3);
    let stride = world.fixed.pair_stride();
    let weight_at = |pair: usize| 1 + world.fixed.anchor_count + pair * stride + stride - 1;
    for pair in 0..world.fixed.anchor_count * world.fixed.anchor_count { genes[weight_at(pair)] = 4.0; }
    let symmetric = Matrix::derive(&world.fixed, &world.fixed.decode(&genes));
    genes[weight_at(1)] = -30.0; // one anchor chasing another that ignores it back
    let one_sided = Matrix::derive(&world.fixed, &world.fixed.decode(&genes));

    let of = |matrix: &Matrix, world: &mut fixture::World| {
        let rollout = Rollout { fixed_genome: &world.fixed, matrix, baseline: &world.baseline };
        evaluate(&Blocks::build(&rollout, &mut world.substrate, &assembled, None, &[]), true)[0]
    };
    assert_eq!(of(&symmetric, &mut world), 0.0, "a matrix equal to its own transpose read as one-sided");
    assert!(of(&one_sided, &mut world) > 0.5, "a chase in one direction read as mutual");
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

/// A metric outside the plan reads zero, which is the same thing a dead sim reads. Both have to fail
/// a floor rather than pass as a value nobody measured.
#[test]
fn a_gate_on_an_unmeasured_metric_reads_as_failed() {
    let assembled = plan(&[rdf::METRIC]);
    let values = measured(&assembled, true);
    assert!(slice(&values, &assembled, mobility::METRIC).is_empty(), "a metric outside the plan claimed slots");
    assert_eq!(scalar(&values, &assembled, mobility::METRIC), 0.0);
    let gates = [Gate { metric: mobility::METRIC, floor: 0.01, ceiling: 1.0 }];
    assert!(!valid(&values, &assembled, &gates, false), "an unmeasured metric cleared a gate on it");
}

// reading one metric out of another

#[test]
fn a_dependent_metric_finds_its_axes_in_the_descriptor_it_is_handed() {
    let assembled = plan(&[temporal::METRIC]);
    let mut world = fixture::world(120, 11);
    let first = read(&mut world, &assembled, true);
    // squeeze the cloud into a corner, so the second tick genuinely looks unlike the first
    for coordinate in world.substrate.positions.iter_mut() { *coordinate *= 0.5; }
    let history = vec![first];
    let second = {
        let rollout = Rollout { fixed_genome: &world.fixed, matrix: &world.matrix, baseline: &world.baseline };
        evaluate(&Blocks::build(&rollout, &mut world.substrate, &assembled, None, &history), true)
    };
    let spread = slice(&second, &assembled, temporal::METRIC)[0];
    assert!(spread > 0.0, "temporal read a flat picture, so it never found its axes in the descriptor");
}

// folding a run into one descriptor

#[test]
fn folding_averages_what_averages_and_takes_the_last_of_the_rest() {
    let assembled = plan(&[temporal::METRIC, robustness::METRIC]);
    let ticks: Vec<Vec<f32>> = (0..4).map(|tick| vec![tick as f32; width_of(&assembled)]).collect();
    let folded = fold(&assembled, &ticks);
    for metric in &assembled {
        let wanted = if metric.reduce == Reduce::Mean { 1.5 } else { 3.0 };
        for &slot in slice(&folded, &assembled, metric) {
            assert_eq!(slot, wanted, "{} folded four ticks to {slot}", metric.key);
        }
    }
    assert!(fold(&assembled, &[]).is_empty(), "a run that never sampled folded to a plausible row of numbers");
}

// rollouts

#[test]
fn the_same_genome_measures_identically_twice() {
    let fixed = fixture::shape(120);
    let probe = Probe::new(&fixed);
    let assembled = plan(&[rdf::METRIC, structure::METRIC, heterogeneity::METRIC, mobility::METRIC]);
    let genes = Genome::build_random(&fixed.bounds(&probe), 11);
    let first = rollout::sim(&fixed, &assembled, &[], &probe, &genes, 40);
    let second = rollout::sim(&fixed, &assembled, &[], &probe, &genes, 40);
    assert_eq!(first.is_empty(), second.is_empty());
    assert_eq!(first, second, "a rollout is not reproducible");
}

#[test]
fn some_genome_survives_a_rollout_and_fills_its_descriptor() {
    let fixed = fixture::shape(120);
    let assembled = plan(&[rdf::METRIC, structure::METRIC, heterogeneity::METRIC, mobility::METRIC]);
    let (_, measured) = fixture::measurable(&fixed, &assembled, 40);
    assert_eq!(measured.len(), width_of(&assembled));
    assert!(measured.iter().all(|value| value.is_finite()));
}

/// A rollout that keeps failing its clauses stops paying for the rest of the run. Two samples of
/// patience, so a sim that dips out of a band and climbs back is not thrown away for it.
#[test]
fn a_rollout_gives_up_on_a_genome_its_gates_keep_rejecting() {
    let fixed = fixture::shape(120);
    let probe = Probe::new(&fixed);
    let assembled = plan(&[structure::METRIC]);
    let (genes, _) = fixture::measurable(&fixed, &assembled, 40);
    let open = [Gate { metric: structure::METRIC, floor: 0.0, ceiling: 1.0 }];
    let shut = [Gate { metric: structure::METRIC, floor: 0.999, ceiling: 1.0 }];
    assert!(!rollout::sim(&fixed, &assembled, &open, &probe, &genes, 40).is_empty(),
        "a band the sim sits inside threw it away anyway");
    assert!(rollout::sim(&fixed, &assembled, &shut, &probe, &genes, 40).is_empty(),
        "a sim nowhere near its band ran to the end and came back measured");
}

// scoring

/// The failure this guards: every clause of the structure score multiplied raw, so one zero
/// annihilated the product. Nearly every genome scored exactly 0, a tournament broke the ties
/// toward its first draw, and the search quietly became random.
#[test]
fn the_structure_criterion_tells_genomes_apart() {
    let fixed = fixture::shape(120);
    let probe = Probe::new(&fixed);
    let bounds = fixed.bounds(&probe);
    let assembled = plan(&fitness::METRICS);
    let scored: Vec<f32> = (0..6)
        .map(|seed| rollout::sim(&fixed, &assembled, &[], &probe, &Genome::build_random(&bounds, seed), 40))
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

/// The same failure from the other side, on descriptors written by hand: a clause at zero has to
/// cost an order of magnitude and leave the other four rankable.
#[test]
fn a_missing_clause_costs_the_structure_score_without_erasing_it() {
    let assembled = plan(&fitness::METRICS);
    let descriptor = |strength: f32, missing: Option<Metric>| {
        let mut values = vec![strength; width_of(&assembled)];
        if let Some(metric) = missing {
            let start = fixture::offset_of(&assembled, metric);
            values[start..start + metric.width].fill(0.0);
        }
        fitness::score(&values, &assembled)
    };
    let whole = descriptor(0.8, None);
    let without = descriptor(0.8, Some(turnover::METRIC));
    assert!(whole > 0.3, "a descriptor strong everywhere scored {whole}");
    assert!(without > 0.0, "one empty clause annihilated the product");
    assert!(without < whole / 10.0, "an empty clause cost {without} against {whole}, which is no cost at all");
    assert!(descriptor(0.6, Some(turnover::METRIC)) < without,
        "two genomes missing the same clause scored alike, so the search cannot rank them");
}

#[test]
fn novelty_pays_for_distance_from_what_has_already_been_found() {
    let member = |metrics: Vec<f32>| Specimen { genome: vec![0.0], metrics, score: 0.0 };
    let population = vec![member(vec![0.0, 0.0]), member(vec![0.1, 0.0]), member(vec![0.0, 0.1])];
    assert_eq!(novelty::score(&[0.5, 0.5], &[]), 0.0, "an opening generation invented novelty for itself");
    let near = novelty::score(&[0.05, 0.05], &population);
    let far = novelty::score(&[1.0, 1.0], &population);
    assert!(far > near, "sitting on top of the population scored {near} against {far} out on its own");
    assert!(near > 0.0, "a genome at a real distance from every neighbor scored nothing");
}

// running a search

/// Capacity is a ceiling and the ranking is the contract a frontend draws from. Two runs of one
/// seed are the same run, or a recorded survivor cannot be found again.
#[test]
fn a_search_keeps_at_most_its_capacity_and_finds_the_same_thing_twice() {
    let run = || {
        let evolve = Evolve { generations: 2, batch: 6, capacity: 4,
            mutation: Mutation { tournament: 0.5, expedition: 0.25, iso: 0.05, line: 0.3 } };
        let search = Search::new(fixture::shape(120), Algorithm::Evolve(evolve), Criterion::Structure,
            Vec::new(), 40, 5);
        search.run(&mut |_, _, _| true)
    };
    let first = run();
    assert!(!first.is_empty(), "three generations of six candidates kept nothing at all");
    assert!(first.len() <= 4, "a capacity of four kept {}", first.len());
    assert!(first.windows(2).all(|pair| pair[0].score >= pair[1].score), "survivors came back unranked");
    let second = run();
    let genomes = |population: &[Specimen]| population.iter().map(|member| member.genome.clone()).collect::<Vec<_>>();
    assert_eq!(genomes(&first), genomes(&second), "one seed searched two different ways");
}

/// A caller tuning its gates needs to know which way a candidate failed: nothing survived and
/// everything was turned away ask for opposite adjustments. A clause on a once metric is the case
/// the rollout cannot judge early, so these all come back gated rather than dead.
#[test]
fn a_search_says_which_way_every_candidate_failed() {
    let evolve = Evolve { generations: 1, batch: 4, capacity: 4,
        mutation: Mutation { tournament: 0.5, expedition: 0.25, iso: 0.05, line: 0.3 } };
    let gates = vec![Gate { metric: robustness::METRIC, floor: 1.01, ceiling: 2.0 }]; // off the axis, so nothing clears it
    let search = Search::new(fixture::shape(120), Algorithm::Evolve(evolve), Criterion::Structure,
        gates, 40, 5);
    let mut counted = Tally::default();
    let population = search.run(&mut |_, _, tally| {
        counted = Tally { evaluated: counted.evaluated + tally.evaluated, died: counted.died + tally.died,
            gated: counted.gated + tally.gated, unscorable: counted.unscorable + tally.unscorable };
        true
    });
    assert!(population.is_empty(), "a gate nothing can clear kept {} genomes", population.len());
    assert_eq!(counted.evaluated, 8, "two generations of four judged {} candidates", counted.evaluated);
    assert_eq!(counted.died + counted.gated + counted.unscorable, counted.evaluated,
        "candidates went missing between the batch and the tally");
    assert!(counted.gated > 0, "a clause no genome cleared turned nobody away");
}
