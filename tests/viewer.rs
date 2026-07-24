//! Reference renderer behavior: the density field's periodic seam, its determinism, and
//! connectivity over material cells. The field defines semantics any faster backend must match.
#![cfg(feature = "viewer")]

use axiom::engine::params::Params;
use axiom::engine::substrate::Substrate;
use axiom::render_recipe::RenderRecipe;
use axiom::util::Rng;
use axiom::viewer::material::DensityField;

fn swarm(count: usize, box_len: f32, seed: u64) -> Substrate {
    let params = Params { particle_count: count, dimensions: 3, coordination: 9.0, radius: 12.0,
        dt: 0.1, seed, anchor_count: 2, shells: 1, bumps: 1, trait_distribution: vec![0.0; 2],
        interactions: Vec::new(), box_len };
    let mut substrate = Substrate::build(&params);
    let mut rng = Rng::new(seed.wrapping_add(9));
    for value in &mut substrate.traits { *value = rng.unit(); }
    substrate
}

fn field(substrate: &Substrate, resolution: usize) -> DensityField {
    let recipe = RenderRecipe { resolution, support: substrate.box_len * 0.08,
        ..RenderRecipe::default() };
    DensityField::from_particles(&substrate.positions, &substrate.traits, substrate.box_len, &recipe)
        .expect("a valid swarm deposits")
}

#[test]
fn opposite_torus_faces_sample_equal_density() {
    let substrate = swarm(600, 100.0, 4);
    let field = field(&substrate, 32);
    let box_len = substrate.box_len;
    let gap = 0.05f32;
    for other in [7.0, 31.0, 68.5, 93.0] {
        // The same point mod box must be the same sample, interpolation included.
        assert_eq!(field.sample(0.0, other, other).to_bits(), field.sample(box_len, other, other).to_bits(),
            "x = 0 and x = box are one point, sampled differently at y=z={other}");
        // Crossing the seam must look like crossing anywhere else: the step across it stays
        // within the field's own smooth variation over the same distance, not a cliff.
        let crack = (field.sample(box_len - gap, other, other) - field.sample(gap, other, other)).abs();
        let smooth = (1..=8).map(|k| {
            let x = box_len * k as f32 / 9.0;
            (field.sample(x - gap, other, other) - field.sample(x + gap, other, other)).abs()
        }).fold(0.0f32, f32::max);
        assert!(crack <= smooth.max(1e-4) * 5.0, "seam crack at y=z={other}: step {crack} vs smooth {smooth}");
    }
}

#[test]
fn the_field_is_a_pure_function_of_particle_state() {
    let substrate = swarm(400, 80.0, 11);
    let first = field(&substrate, 24);
    let second = field(&substrate, 24);
    let mut probes = Rng::new(2);
    for _ in 0..64 {
        let (x, y, z) = (probes.unit() * 80.0, probes.unit() * 80.0, probes.unit() * 80.0);
        assert_eq!(first.sample(x, y, z).to_bits(), second.sample(x, y, z).to_bits());
    }
}

#[test]
fn a_particle_bridge_reads_connected_while_separated_clusters_keep_a_void() {
    let box_len = 100.0;
    let mut bridged = swarm(900, box_len, 3);
    let mut rng = Rng::new(5);
    for i in 0..900 { // a rod along x at the box middle: one component through the seam
        bridged.positions[i * 3] = rng.unit() * box_len;
        bridged.positions[i * 3 + 1] = box_len * 0.5 + rng.range(-4.0, 4.0);
        bridged.positions[i * 3 + 2] = box_len * 0.5 + rng.range(-4.0, 4.0);
    }
    let rod = field(&bridged, 24);
    let threshold = 0.5;
    assert_eq!(rod.connected_components(threshold), 1, "the rod fragmented");

    let mut separated = swarm(900, box_len, 6);
    let mut rng = Rng::new(7);
    for i in 0..900 { // two balls far apart: two components, void between
        let center = if i % 2 == 0 { box_len * 0.25 } else { box_len * 0.75 };
        for axis in 0..3 { separated.positions[i * 3 + axis] = center + rng.range(-6.0, 6.0); }
    }
    let clusters = field(&separated, 24);
    assert_eq!(clusters.connected_components(threshold), 2, "two balls did not read as two components");
}
