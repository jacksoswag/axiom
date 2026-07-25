//! Entry point for tuning. Orchestrates sim instantiation, genome evaluation, scoring, and tuning.

use crate::engine::params::FixedGenome;
use crate::engine::resolve::Probe;
use crate::tuner::algorithms::Algorithm;
use crate::tuner::criterion::Criterion;
use crate::tuner::gate::{self, Gate};
use crate::tuner::metrics::{self, field, Metric};
use crate::run;
use crate::tuner::specimen::Specimen;
use crate::util::Rng;

/// One complete search. Fully configurable and runnable through harness/
pub struct Search {
    pub algorithm: Algorithm, // tuning algorithm (ie. evolution, RL, brute-force random, etc)
    pub criterion: Criterion, // success criteria, what simulated behavior rewards the algorithm
    pub gates: Vec<Gate>, // owned, because a harness builds these at runtime rather than declaring them
    pub fixed_genome: FixedGenome,
    pub timesteps: usize, // how long each sim runs before it is measured
    pub seed: u64,
    // Below are derived, not fed
    pub probe: Probe, // density reference, measured once per search
    pub bounds: Vec<(f32, f32)>,
    pub metrics: Vec<Metric>
}
impl Search {
    /// Used by harness to instantiate new search
    pub fn new(fixed_genome: FixedGenome, algorithm: Algorithm, criterion: Criterion, gates: Vec<Gate>, timesteps: usize, seed: u64) -> Search {
        // The measurement grid is a cubic 3-axis lattice, so any other shape indexes cells that do
        // not line up with the genome's own coordinates. Caught here, the one door into a search.
        if fixed_genome.dimensions != field::GRID_AXES {
            panic!("the measurement grid is {}-axis, genome is {}-D", field::GRID_AXES, fixed_genome.dimensions);
        }
        let probe = Probe::new(&fixed_genome);
        let bounds = fixed_genome.bounds(&probe);
        let mut wanted: Vec<Metric> = criterion.metrics().to_vec();
        wanted.extend(gates.iter().map(|gate| gate.metric));
        Search { metrics: metrics::plan(&wanted), fixed_genome, algorithm, criterion, gates, probe, bounds, timesteps, seed }
    }
    /// Run search to completion and hand back the surviving population, ranked best first. Called externally
    pub fn run(&self) -> Vec<Specimen> {
        let mut rng = Rng::new(self.seed);
        let mut population = self.algorithm.explore(self, &mut rng);
        population.sort_by(|left, right| right.score.total_cmp(&left.score));
        population
    }
    /// Simulates a genome, filters by the gates, and ranks survivors on criterion. Cost is the simulation.
    pub fn run_genome(&self, genes: &[f32], population: &[Specimen]) -> Option<Specimen> {
        let measured = run::sim(&self.fixed_genome, &self.metrics, &self.gates, &self.probe, genes, self.timesteps);
        if measured.is_empty() || measured.iter().any(|value| !value.is_finite()) { return None; } // empty is a sim that died
        if !gate::valid(&measured, &self.metrics, &self.gates, false) { return None; }
        let score = self.criterion.score(&measured, &self.metrics, population);
        score.is_finite().then(|| Specimen { genome: genes.to_vec(), metrics: measured, score })
    }
}
