//! What a smoke case starts from: one small shape, a genome drawn inside it, the uniform cloud and
//! law that genome makes, and the hunt for a genome still measurable after a short run. Every case
//! below wants some of this and none of it is worth writing four times.

use axiom::engine::matrix::Matrix;
use axiom::engine::params::{FixedGenome, Genome};
use axiom::engine::resolve::Probe;
use axiom::engine::substrate::Substrate;
use axiom::engine::r#trait::init_particle_traits;
use axiom::harness::rollout;
use axiom::tuner::metrics::{rdf, Metric};

/// Small enough that a rollout is milliseconds, wide enough that two anchors interact and the
/// spatial index has cells worth walking.
pub fn shape(particle_count: usize) -> FixedGenome {
    FixedGenome { particle_count, dimensions: 3, anchor_count: 2, shells: 1, bumps: 1,
        radius: 2.0, dt: 0.05, seed: 7 }
}
/// A uniform draw inside a shape's own bounds, and the probe it was drawn against
pub fn drawn(fixed: &FixedGenome, seed: u64) -> (Probe, Vec<f32>) {
    let probe = Probe::new(fixed);
    let genes = Genome::build_random(&fixed.bounds(&probe), seed);
    (probe, genes)
}

/// The uniform starting cloud and the law over it, with nothing stepped. A case built on this reads
/// an exact value rather than wherever a genome happened to wander.
pub struct World {
    pub fixed: FixedGenome,
    pub substrate: Substrate,
    pub matrix: Matrix,
    pub baseline: rdf::Histogram,
}
pub fn world(particle_count: usize, seed: u64) -> World {
    let fixed = shape(particle_count);
    let (probe, genes) = drawn(&fixed, seed);
    let genome = fixed.decode(&genes);
    let mut substrate = Substrate::build(&fixed, probe.box_len(genome.coordination));
    init_particle_traits(&mut substrate, &genome.trait_distribution, fixed.seed);
    let matrix = Matrix::derive(&fixed, &genome);
    let baseline = rdf::build(&substrate);
    World { fixed, substrate, matrix, baseline }
}

/// The first seed whose rollout comes back with a descriptor, and that descriptor. A genome that
/// blew up or failed its gates measures empty, and a case asserting on one asserts on nothing.
pub fn measurable(fixed: &FixedGenome, plan: &[Metric], timesteps: usize) -> (Vec<f32>, Vec<f32>) {
    let probe = Probe::new(fixed);
    let bounds = fixed.bounds(&probe);
    (0..12u64).map(|seed| Genome::build_random(&bounds, seed))
        .map(|genes| { let measured = rollout::sim(fixed, plan, &[], &probe, &genes, timesteps); (genes, measured) })
        .find(|(_, measured)| !measured.is_empty())
        .expect("no genome out of twelve produced a measurable sim")
}

/// Where one metric's slots start in a descriptor this plan lays out. Reading a slice is the crate's
/// own job; writing a synthetic descriptor to score is this suite's, and both want the offset.
pub fn offset_of(plan: &[Metric], metric: Metric) -> usize {
    plan.iter().take_while(|named| named.key != metric.key).map(|named| named.width).sum()
}
