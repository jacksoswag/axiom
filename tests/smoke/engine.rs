//! Engine behavior: the trait basis, the spatial index, the kernel and its slope, the box a
//! coordination gene resolves to, and that a step lands in the same place however it was scheduled.

use axiom::engine::kernel::{distance_cutoff, distance_sq, strength_and_slope, Shell, CUTOFF_SIGMA};
use axiom::engine::lenia::step;
use axiom::engine::matrix::Matrix;
use axiom::engine::params::{clamp, FixedGenome, Genome};
use axiom::engine::resolve::{Probe, COORDINATION_BOUNDS};
use axiom::engine::sim::Sim;
use axiom::engine::substrate::Substrate;
use axiom::engine::r#trait::{init_particle_traits, membership, memberships};
use axiom::tuner::metrics::alive;
use axiom::util::Rng;

use crate::fixture;

/// A decoded genome and the box its coordination gene resolves to, which is what every engine entry
/// point wants now that the two genome halves are separate.
fn resolved(fixed: &FixedGenome, seed: u64) -> (Genome, f32) {
    let (probe, genes) = fixture::drawn(fixed, seed);
    let genome = fixed.decode(&genes);
    let box_len = probe.box_len(genome.coordination);
    (genome, box_len)
}
/// The first genome out of twelve still finite after a short run. A blown-up sim is NaN everywhere,
/// and a case asking where its particles ended up would be asking about NaN.
fn living(particle_count: usize, steps: u64) -> Sim {
    let fixed = fixture::shape(particle_count);
    for seed in 0..12u64 {
        let (genome, box_len) = resolved(&fixed, seed);
        let mut sim = Sim::new(&fixed, genome, box_len);
        sim.run(steps);
        if alive(&sim.substrate.positions) { return sim; }
    }
    panic!("no genome out of twelve survived {steps} steps");
}

// the trait basis

#[test]
fn hat_memberships_sum_to_one_and_activate_at_most_two_adjacent_anchors() {
    for anchors in 2..=6 {
        for step in 0..=100 {
            let entries = membership(step as f32 / 100.0, anchors).entries;
            let total: f32 = entries.iter().map(|&(_, weight)| weight).sum();
            assert!((total - 1.0).abs() < 1e-5, "weights sum to {total}");
            let (lower, upper) = (entries[0].0 as usize, entries[1].0 as usize);
            assert!(upper == (lower + 1) % anchors, "anchors {lower} and {upper} are not adjacent");
        }
    }
}

#[test]
fn anchor_values_are_exactly_one_hot() {
    let anchors = 4;
    for anchor in 0..anchors {
        let entries = membership(anchor as f32 / anchors as f32, anchors).entries;
        assert_eq!(entries[0], (anchor as u32, 1.0));
        assert_eq!(entries[1].1, 0.0);
    }
}

#[test]
fn trait_seeding_matches_the_logit_histogram_exactly() {
    let fixed = fixture::shape(60);
    let (_, box_len) = resolved(&fixed, 3);
    let mut substrate = Substrate::build(&fixed, box_len);
    init_particle_traits(&mut substrate, &[0.0, 0.0, 0.0], 7); // three equal bins over 60 particles
    let mut counts = [0usize; 3];
    for &value in &substrate.traits {
        counts[((value * 3.0) as usize).min(2)] += 1;
    }
    assert_eq!(counts, [20, 20, 20], "equal logits must give equal shares");
}

/// Equal logits are the easy case. A share that only comes out right after rounding is where a
/// seeding that merely looks correct would hide, so these divide unevenly on purpose.
#[test]
fn uneven_logits_seed_the_shares_they_asked_for() {
    let fixed = fixture::shape(120);
    let (_, box_len) = resolved(&fixed, 3);
    let logits = [1.5f32, 0.0, -1.0];
    let mut substrate = Substrate::build(&fixed, box_len);
    init_particle_traits(&mut substrate, &logits, 7);
    let mut counts = [0usize; 3];
    for &value in &substrate.traits { counts[((value * 3.0) as usize).min(2)] += 1; }
    let total: f32 = logits.iter().map(|logit| logit.exp()).sum();
    for (bin, &logit) in logits.iter().enumerate() {
        let wanted = logit.exp() / total * 120.0;
        assert!((counts[bin] as f32 - wanted).abs() <= 1.0,
            "bin {bin} holds {} where its logit asks for {wanted:.1}", counts[bin]);
    }
    assert_eq!(counts.iter().sum::<usize>(), 120, "seeding lost or invented particles");
}

/// The step loop reads memberships a few million times a tick and never rebuilds them, so they have
/// to be exactly what the traits say from the moment seeding ends.
#[test]
fn memberships_are_the_traits_they_were_built_from() {
    let fixed = fixture::shape(120);
    let (genome, box_len) = resolved(&fixed, 3);
    let mut substrate = Substrate::build(&fixed, box_len);
    init_particle_traits(&mut substrate, &genome.trait_distribution, fixed.seed);
    let expected = memberships(&substrate.traits, fixed.anchor_count);
    assert_eq!(substrate.memberships.len(), substrate.traits.len());
    for (particle, held) in substrate.memberships.iter().enumerate() {
        assert_eq!(held.entries, expected[particle].entries, "particle {particle} carries a stale membership");
    }
}

// the spatial index

#[test]
fn grid_and_all_pairs_neighbor_walks_agree() {
    let fixed = fixture::shape(60);
    let (_, box_len) = resolved(&fixed, 3);
    let mut substrate = Substrate::build(&fixed, box_len);
    let cutoff = substrate.box_len / 5.0; // five cells per axis: the grid path is active
    let collect = |substrate: &Substrate| {
        let mut pairs = Vec::new();
        for i in 0..substrate.traits.len() {
            let position = substrate.pos(i).to_vec();
            let mut visit = |j: usize| {
                if j > i && distance_sq(&position, substrate.pos(j), substrate, &mut []).sqrt() <= cutoff {
                    pairs.push((i, j));
                }
            };
            substrate.visit_neighbors(&position, &mut visit);
        }
        pairs.sort_unstable();
        pairs
    };
    substrate.rebuild_grid(cutoff);
    let gridded = collect(&substrate);
    substrate.rebuild_grid(substrate.box_len); // one cell per axis is invalid, forcing all-pairs
    let brute = collect(&substrate);
    assert!(!gridded.is_empty(), "the fixture found no neighbor pairs at all");
    assert_eq!(gridded, brute);
}

/// The grid is a saving and never a requirement. Every regime it refuses has to fall back rather
/// than index a stencil that wraps onto itself.
#[test]
fn the_index_steps_aside_where_it_cannot_help() {
    let fixed = fixture::shape(120);
    let (_, box_len) = resolved(&fixed, 3);
    let substrate = Substrate::build(&fixed, box_len);
    assert!(substrate.gridded(substrate.box_len / 5.0), "an ordinary reach found no grid");
    assert!(!substrate.gridded(substrate.box_len), "one cell per axis is not a grid");
    assert!(!substrate.gridded(substrate.box_len / 2.5), "two cells per axis wrap the stencil onto itself");
    assert!(!substrate.gridded(0.0), "a law that reaches nowhere has nothing to bucket");
    assert!(!substrate.gridded(f32::INFINITY), "an infinite reach cast itself into a legal cell count");
}

#[test]
fn periodic_distance_uses_the_minimum_image() {
    let fixed = fixture::shape(60);
    let (_, box_len) = resolved(&fixed, 3);
    let mut substrate = Substrate::build(&fixed, box_len);
    let box_len = substrate.box_len;
    substrate.positions[..3].copy_from_slice(&[0.1, 0.0, 0.0]);
    substrate.positions[3..6].copy_from_slice(&[box_len - 0.1, 0.0, 0.0]);
    let distance = distance_sq(substrate.pos(0), substrate.pos(1), &substrate, &mut []).sqrt();
    assert!(distance < 0.3 + substrate.softening_sq.sqrt(), "wrap-around distance read as {distance}");
}

/// A copy is what the repair experiment damages, so it carries the population exactly and none of
/// the index built over it: the copy's particles are about to be somewhere else.
#[test]
fn a_copied_substrate_carries_the_population_and_not_the_index() {
    let fixed = fixture::shape(120);
    let (genome, box_len) = resolved(&fixed, 3);
    let mut substrate = Substrate::build(&fixed, box_len);
    init_particle_traits(&mut substrate, &genome.trait_distribution, fixed.seed);
    substrate.rebuild_grid(substrate.box_len / 5.0);
    let copy = substrate.copy();
    assert_eq!(copy.positions, substrate.positions);
    assert_eq!(copy.traits, substrate.traits);
    assert_eq!(copy.box_len, substrate.box_len);
    let mut seen = 0;
    copy.visit_neighbors(copy.pos(0), |_| seen += 1);
    assert_eq!(seen, copy.traits.len(), "a copy answered off an index it never built");
}

// the kernel

/// The step loop turns the slope into motion, so a slope that is not the slope of its own strength
/// is a force pointing somewhere the potential never said to go.
#[test]
fn a_kernels_slope_is_the_slope_of_its_own_strength() {
    let shells = [Shell::new(1.0, 2.0, 1.0), Shell::new(0.5, 4.0, 1.5)];
    let nudge = 0.01;
    for tenth in 5..60 {
        let x = tenth as f32 / 10.0;
        let (_, slope) = strength_and_slope(x, &shells);
        let ahead = strength_and_slope(x + nudge, &shells).0;
        let behind = strength_and_slope(x - nudge, &shells).0;
        let climb = (ahead - behind) / (2.0 * nudge);
        assert!((slope - climb).abs() < 2e-3, "at {x} the slope reads {slope} where the strength climbs at {climb}");
    }
}

#[test]
fn a_shell_peaks_at_its_own_peak() {
    let shells = [Shell::new(0.7, 3.0, 0.5)];
    let (peak, slope) = strength_and_slope(3.0, &shells);
    assert!((peak - 0.7).abs() < 1e-6, "a lone shell reads {peak} at its peak rather than its amplitude");
    assert!(slope.abs() < 1e-6, "the top of the bell is not flat");
    assert!(strength_and_slope(3.0 + CUTOFF_SIGMA * 0.5, &shells).0 < 0.007,
        "the cutoff sits where a hundredth of the amplitude is still left");
}

/// One inert term used to inflate the reach of the whole mixture, which walks every pair for a
/// contribution of zero.
#[test]
fn an_inert_shell_cannot_stretch_the_reach() {
    let live = Shell::new(1.0, 2.0, 0.5);
    let reach = distance_cutoff(&[Shell::new(0.0, 40.0, 10.0), Shell::new(1.0, 2.0, 0.5)]);
    assert!((reach - (live.peak + CUTOFF_SIGMA * live.width)).abs() < 1e-4, "reach came back {reach}");
    assert_eq!(distance_cutoff(&[Shell::new(0.0, 40.0, 10.0)]), 0.0, "a mixture of nothing reaches somewhere");
}

// stepping

#[test]
fn a_law_that_senses_nothing_moves_nothing() {
    let fixed = fixture::shape(120);
    let (probe, mut genes) = fixture::drawn(&fixed, 3);
    let stride = fixed.pair_stride();
    for pair in 0..fixed.anchor_count * fixed.anchor_count { // every shell amplitude to zero
        for shell in 0..fixed.shells { genes[1 + fixed.anchor_count + pair * stride + shell * 3] = 0.0; }
    }
    let genome = fixed.decode(&genes);
    let box_len = probe.box_len(genome.coordination);
    let mut sim = Sim::new(&fixed, genome, box_len);
    assert_eq!(sim.matrix.max_reach, 0.0, "a mixture of inert shells still claims a reach");
    let before = sim.substrate.positions.clone();
    sim.run(10);
    assert_eq!(sim.substrate.positions, before, "a law that senses nothing moved a particle anyway");
}

#[test]
fn every_particle_stays_inside_the_box() {
    let sim = living(120, 50);
    let box_len = sim.substrate.box_len;
    for (slot, &coordinate) in sim.substrate.positions.iter().enumerate() {
        assert!((0.0..box_len).contains(&coordinate),
            "coordinate {slot} sits at {coordinate} outside a box of {box_len}");
    }
}

/// Both paths accumulate per particle in the same fixed order, so they are the same arithmetic on
/// the same values. A search that judged a genome one way and a playground that replays it the
/// other would otherwise be showing a world the search never found.
#[test]
fn threaded_and_sequential_stepping_agree_bit_for_bit() {
    let fixed = fixture::shape(200);
    let (genome, box_len) = resolved(&fixed, 3);
    let mut threaded = Sim::new(&fixed, genome.clone(), box_len);
    let mut plain = Sim::new(&fixed, genome, box_len);
    for _ in 0..40 {
        step(&mut threaded.substrate, &threaded.matrix, fixed.dt, true);
        step(&mut plain.substrate, &plain.matrix, fixed.dt, false);
    }
    let bits = |sim: &Sim| sim.substrate.positions.iter().map(|value| value.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&threaded), bits(&plain));
}

#[test]
fn a_seeded_sim_is_bit_reproducible() {
    let fixed = fixture::shape(60);
    let (genome, box_len) = resolved(&fixed, 3);
    let mut a = Sim::new(&fixed, genome.clone(), box_len);
    let mut b = Sim::new(&fixed, genome, box_len);
    a.run(50);
    b.run(50);
    let bits = |sim: &Sim| sim.substrate.positions.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&a), bits(&b));
    assert_eq!(a.tick, 50);
}

// the law

#[test]
fn every_anchor_pair_calibrates_a_real_norm() {
    let fixed = fixture::shape(60);
    let (genome, box_len) = resolved(&fixed, 3);
    let sim = Sim::new(&fixed, genome, box_len);
    for (pair, interaction) in sim.matrix.interactions.iter().enumerate() {
        assert!(interaction.norm.is_finite() && interaction.norm > 0.0, "pair {pair} norm {}", interaction.norm);
    }
    // The uniform-trait reference population must give off-diagonal pairs measured norms too,
    // not the 1.0 fallback that an unseeded calibration substrate produces.
    let measured = sim.matrix.interactions.iter().filter(|i| (i.norm - 1.0).abs() > 1e-6).count();
    assert!(measured > 0, "every norm fell back to 1.0: calibration saw no anchor mass");
}

/// Pair position is the pair's identity, so a block read into the wrong row is a law nobody asked
/// for and nothing downstream is in a position to notice.
#[test]
fn a_pair_block_lands_in_the_row_its_position_names() {
    let fixed = fixture::shape(120);
    let (_, mut genes) = fixture::drawn(&fixed, 3);
    let stride = fixed.pair_stride();
    for pair in 0..fixed.anchor_count * fixed.anchor_count {
        genes[1 + fixed.anchor_count + pair * stride + stride - 1] = pair as f32; // this block's weight
    }
    let matrix = Matrix::derive(&fixed, &fixed.decode(&genes));
    for source in 0..fixed.anchor_count {
        for destination in 0..fixed.anchor_count {
            let pair = source * fixed.anchor_count + destination;
            assert_eq!(matrix.interactions[pair].weight, pair as f32,
                "the block for {source} onto {destination} read someone else's weight");
        }
    }
    assert_eq!(matrix.shells.len(), fixed.anchor_count * fixed.anchor_count * fixed.shells);
    assert_eq!(matrix.bumps.len(), fixed.anchor_count * fixed.anchor_count * fixed.bumps);
}

/// Past half the box a shell folds onto its own image and a particle senses itself around the torus,
/// which is a force from nowhere. The widest law a genome may legally hold stops short of that; it is
/// allowed past the third of a box the grid needs, and says so by falling back to every pair.
#[test]
fn no_legal_law_reaches_far_enough_to_fold_into_its_own_image() {
    let fixed = fixture::shape(320);
    let probe = Probe::new(&fixed);
    for coordination in [COORDINATION_BOUNDS.0, 8.0, COORDINATION_BOUNDS.1] {
        let box_len = probe.box_len(coordination);
        let mut genes: Vec<f32> = fixed.bounds(&probe).iter().map(|&(_, high)| high).collect();
        genes[0] = coordination;
        let mut genome = fixed.decode(&genes);
        clamp(&mut genome.interactions, &fixed.pair_bounds(box_len)); // what a rollout does at build time
        let matrix = Matrix::derive(&fixed, &genome);
        assert!(matrix.max_reach < box_len * 0.5,
            "a legal genome at {coordination} neighbors reaches {} in a box of {box_len}", matrix.max_reach);
        let substrate = Substrate::build(&fixed, box_len);
        assert_eq!(substrate.gridded(matrix.max_reach), box_len / matrix.max_reach >= 3.0,
            "a reach of {} in a box of {box_len} disagrees with the regime it reports", matrix.max_reach);
    }
}

// the box a genome resolves to

#[test]
fn the_box_shrinks_as_a_genome_asks_to_sense_more() {
    let probe = Probe::new(&fixture::shape(320));
    let mut previous = f32::INFINITY;
    for step in 0..=10 {
        let coordination = COORDINATION_BOUNDS.0
            + (COORDINATION_BOUNDS.1 - COORDINATION_BOUNDS.0) * step as f32 / 10.0;
        let box_len = probe.box_len(coordination);
        assert!(box_len.is_finite() && box_len > 0.0, "{coordination} neighbors resolved to a box of {box_len}");
        assert!(box_len < previous, "{coordination} neighbors wanted a box of {box_len}, no smaller than {previous}");
        previous = box_len;
    }
}

/// The probe is the expensive half of opening a world and a slider drag rebuilds the world on every
/// frame, so keeping it across the edits it does not depend on is what makes a drag affordable.
#[test]
fn a_probe_outlives_the_edits_it_does_not_depend_on() {
    let fixed = fixture::shape(320);
    let probe = Probe::new(&fixed);
    let with = |change: fn(&mut FixedGenome)| { let mut shape = fixed.clone(); change(&mut shape); probe.fits(&shape) };
    assert!(with(|shape| shape.seed = 99), "a reseed threw away a measurement it could keep");
    assert!(with(|shape| shape.anchor_count = 5), "an anchor count threw away a measurement it could keep");
    assert!(with(|shape| shape.dt = 0.01), "a timestep threw away a measurement it could keep");
    assert!(!with(|shape| shape.particle_count = 640), "a different swarm reused a stale density");
    assert!(!with(|shape| shape.radius = 4.0), "a different radius reused a stale density");
}

// the genome itself

#[test]
fn a_genome_decodes_into_the_slots_its_shape_promises() {
    for anchors in 2..=4 {
        for shells in 1..=3 {
            let mut fixed = fixture::shape(120);
            fixed.anchor_count = anchors; fixed.shells = shells; fixed.bumps = 2;
            let probe = Probe::new(&fixed);
            let bounds = fixed.bounds(&probe);
            assert_eq!(bounds.len(), fixed.gene_len(), "the bounds and the genome disagree about length");
            assert_eq!(fixed.pair_stride(), 3 * (shells + 2) + 1);
            let genome = fixed.decode(&Genome::build_random(&bounds, 5));
            assert_eq!(genome.trait_distribution.len(), anchors);
            assert_eq!(genome.interactions.len(), anchors * anchors * fixed.pair_stride());
        }
    }
}

/// A gene arrives from a frontend, a mutation, or a recorded run, and one that is not a number would
/// reach the kernel and spread to every particle inside a step.
#[test]
fn a_gene_that_is_not_a_number_decodes_to_a_legal_one() {
    let fixed = fixture::shape(120);
    let (probe, mut genes) = fixture::drawn(&fixed, 3);
    genes[0] = f32::NAN;
    genes[1] = f32::INFINITY;
    genes[2] = -1e9;
    let genome = fixed.decode(&genes);
    assert!((COORDINATION_BOUNDS.0..=COORDINATION_BOUNDS.1).contains(&genome.coordination),
        "coordination decoded to {}", genome.coordination);
    assert!(genome.trait_distribution.iter().all(|logit| logit.is_finite()), "a logit came through unreadable");
    assert!(probe.box_len(genome.coordination).is_finite(), "an unreadable gene resolved to an unreadable box");

    // Unreadable falls to the floor of its own range rather than to the nearest end: an infinite
    // gene has no nearest end, and a floor is the one answer that is legal for every gene there is.
    let mut wild = vec![f32::NAN, f32::INFINITY, -1e9, 1e9];
    clamp(&mut wild, &[(0.0, 1.0); 4]);
    assert_eq!(wild, vec![0.0, 0.0, 0.0, 1.0], "clamping left a gene the kernel cannot read");
}

// the draw underneath all of it

#[test]
fn the_draw_is_reproducible_and_stays_inside_its_range() {
    let sequence = |seed: u64| { let mut rng = Rng::new(seed); (0..64).map(|_| rng.next_u64()).collect::<Vec<_>>() };
    assert_eq!(sequence(11), sequence(11), "one seed drew two different runs");
    assert_ne!(sequence(11), sequence(12), "two seeds drew the same run");

    let mut rng = Rng::new(4);
    let mut bell = 0.0f64;
    for _ in 0..20_000 {
        let unit = rng.unit();
        assert!((0.0..1.0).contains(&unit), "a unit draw came back {unit}");
        assert!((-2.0..=5.0).contains(&rng.range(-2.0, 5.0)));
        assert!(rng.below(7) < 7);
        bell += rng.normal() as f64;
    }
    assert_eq!(rng.below(0), 0, "an empty range still picked something");
    assert!((bell / 20_000.0).abs() < 0.05, "the bell sits at {} rather than at zero", bell / 20_000.0);
}
