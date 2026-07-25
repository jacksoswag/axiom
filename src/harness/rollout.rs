//! One genome in, one descriptor out. The only place a Sim is built to be measured, so every
//! criterion reads behavior the same way. Decides when to look, never what at. Sits above both
//! halves: a search never builds a sim, and the engine never knows it is being measured.

use crate::engine::params::{clamp, FixedGenome};
use crate::engine::resolve::Probe;
use crate::engine::sim::Sim;
use crate::tuner::gate::{self, Gate};
use crate::tuner::metrics::{alive, evaluate, fold, rdf, wants_baseline, Blocks, Metric, Motion, Rollout};

/// Times a rollout stops to look, which sizes the history a metric reading across ticks gets. The
/// playground keeps the same count so a live reading and a searched one mean the same thing.
pub const SAMPLES: usize = 8;
/// Consecutive samples outside the clauses before giving up. One is too eager: a sim can dip out of
/// a band and climb back within a sample.
const PATIENCE: usize = 2;

/// The one way a Sim is built from a flat genome: settle the box the coordination gene asks for, then
/// pull the pair genes back inside what that box allows, since reach is tighter at a smaller box. A
/// searched genome and a played one go through here alike, so they cannot simulate different laws.
pub fn build(fixed_genome: &FixedGenome, genes: &[f32], probe: &Probe) -> Sim {
    let mut genome = fixed_genome.decode(genes);
    let box_len = probe.box_len(genome.coordination);
    clamp(&mut genome.interactions, &fixed_genome.pair_bounds(box_len));
    Sim::new(fixed_genome, genome, box_len)
}

/// Run and measure: sample the latter half of the run, judge the gates as they land, and fold what was
/// seen into one descriptor. A sim that blows up or fails its gates early comes back empty, which
/// reads zero everywhere and so clears nothing.
pub fn sim(fixed_genome: &FixedGenome, metrics: &[Metric], gates: &[Gate], probe: &Probe,
    genes: &[f32], timesteps: usize) -> Vec<f32>
{
    let mut sim = build(fixed_genome, genes, probe);
    // The uniform start is the structure reference, and building it is an all-pairs pass before the
    // run has taken a step, so a plan that will never read one does not pay for it.
    let baseline = if wants_baseline(metrics) { rdf::build(&sim.substrate) } else { rdf::blank() };
    let interval = (timesteps / (2 * SAMPLES)).max(1); // samples spread across the latter half
    let first = timesteps.saturating_sub(interval * SAMPLES);
    let last = timesteps.saturating_sub(interval);
    let mut history: Vec<Vec<f32>> = Vec::with_capacity(SAMPLES);
    let mut motion: Option<Motion> = None;
    let mut failures = 0;

    for tick in 0..timesteps {
        sim.step();
        if !alive(&sim.substrate.positions) { return Vec::new(); }
        if tick < first || (tick - first) % interval != 0 { continue; }

        let rollout = Rollout { fixed_genome, matrix: &sim.matrix, baseline: &baseline };
        let blocks = Blocks::build(&rollout, &mut sim.substrate, metrics, motion.take(), &history);
        let measured = evaluate(&blocks, tick == last);
        motion = blocks.carry();
        if !history.is_empty() { // the first sample has nothing behind it, so anything reading change reads zero
            failures = if gate::valid(&measured, metrics, gates, true) { 0 } else { failures + 1 };
        }
        history.push(measured);
        if failures >= PATIENCE { return Vec::new(); }
    }
    fold(metrics, &history) // a history that never filled folds to an empty descriptor
}
