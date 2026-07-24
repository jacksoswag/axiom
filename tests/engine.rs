//! Engine behavior: the trait basis, the spatial index, seeding, and step determinism.

use axiom::engine::kernel::distance_sq;
use axiom::engine::params::Params;
use axiom::engine::sim::Sim;
use axiom::engine::substrate::Substrate;
use axiom::engine::r#trait::{init_particle_traits, membership};
use axiom::tuner::genome::Caps;

fn small_params() -> Params {
    let caps = Caps { particle_count: 60, radius: 2.0, anchor_count: 2, shells: 1, bumps: 1, ..Caps::default() };
    let probe = caps.probe();
    caps.params(&caps.default_genome(&probe), &probe)
}

#[test]
fn hat_memberships_sum_to_one_and_activate_at_most_two_adjacent_anchors() {
    for anchors in 2..=6 {
        for step in 0..=100 {
            let entries = membership(step as f32 / 100.0, anchors).entries;
            let total: f32 = entries.iter().map(|&(_, weight)| weight).sum();
            assert!((total - 1.0).abs() < 1e-5, "weights sum to {total}");
            let (lower, upper) = (entries[0].0, entries[1].0);
            assert!(upper == (lower + 1) % anchors, "anchors {lower} and {upper} are not adjacent");
        }
    }
}

#[test]
fn anchor_values_are_exactly_one_hot() {
    let anchors = 4;
    for anchor in 0..anchors {
        let entries = membership(anchor as f32 / anchors as f32, anchors).entries;
        assert_eq!(entries[0], (anchor, 1.0));
        assert_eq!(entries[1].1, 0.0);
    }
}

#[test]
fn trait_seeding_matches_the_logit_histogram_exactly() {
    let params = small_params();
    let mut substrate = Substrate::build(&params);
    init_particle_traits(&mut substrate, &[0.0, 0.0, 0.0], 7); // three equal bins over 60 particles
    let mut counts = [0usize; 3];
    for &value in &substrate.traits {
        counts[((value * 3.0) as usize).min(2)] += 1;
    }
    assert_eq!(counts, [20, 20, 20], "equal logits must give equal shares");
}

#[test]
fn grid_and_all_pairs_neighbor_walks_agree() {
    let params = small_params();
    let mut substrate = Substrate::build(&params);
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

#[test]
fn periodic_distance_uses_the_minimum_image() {
    let params = small_params();
    let mut substrate = Substrate::build(&params);
    let box_len = substrate.box_len;
    substrate.positions[..3].copy_from_slice(&[0.1, 0.0, 0.0]);
    substrate.positions[3..6].copy_from_slice(&[box_len - 0.1, 0.0, 0.0]);
    let distance = distance_sq(substrate.pos(0), substrate.pos(1), &substrate, &mut []).sqrt();
    assert!(distance < 0.3 + substrate.softening, "wrap-around distance read as {distance}");
}

#[test]
fn a_seeded_sim_is_bit_reproducible() {
    let params = small_params();
    let mut a = Sim::new(&params);
    let mut b = Sim::new(&params);
    a.run(50);
    b.run(50);
    let bits = |sim: &Sim| sim.substrate.positions.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&a), bits(&b));
    assert_eq!(a.tick, 50);
}

#[test]
fn every_anchor_pair_calibrates_a_real_norm() {
    let params = small_params();
    let sim = Sim::new(&params);
    for (pair, interaction) in sim.matrix.interactions.iter().enumerate() {
        assert!(interaction.norm.is_finite() && interaction.norm > 0.0, "pair {pair} norm {}", interaction.norm);
    }
    // The uniform-trait reference population must give off-diagonal pairs measured norms too,
    // not the 1.0 fallback that an unseeded calibration substrate produces.
    let measured = sim.matrix.interactions.iter().filter(|i| (i.norm - 1.0).abs() > 1e-6).count();
    assert!(measured > 0, "every norm fell back to 1.0: calibration saw no anchor mass");
}
