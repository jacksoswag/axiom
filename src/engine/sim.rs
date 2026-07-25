//! Sim: the main entry point and the authoritative state of a run, reconstructible from its genome halves alone.
//! box_len must already be resolved by the caller. Sim only creates and runs the Lenia simulation.

use crate::engine::matrix::Matrix;
use crate::engine::lenia::step;
use crate::engine::params::{FixedGenome, Genome};
use crate::engine::resolve::CALIBRATION_SEED;
use crate::engine::r#trait::init_particle_traits;
use crate::engine::substrate::Substrate;

pub struct Sim {
    pub substrate: Substrate,
    pub matrix: Matrix, // interaction matrix between anchors
    pub fixed_genome: FixedGenome,
    pub genome: Genome,
    pub tick: u64,
    pub threaded: bool, // spread each tick over the cores, for the one sim that has them to itself
}
impl Sim {
    /// A fresh Sim, ready to run. Single-threaded stepping by default, since the common case is one
    /// of a batch of sims already filling every core; a lone sim raises threaded and takes them.
    pub fn new(fixed_genome: &FixedGenome, genome: Genome, box_len: f32) -> Sim {
        let mut substrate = Substrate::build(fixed_genome, box_len);
        init_particle_traits(&mut substrate, &genome.trait_distribution, fixed_genome.seed);
        let mut matrix = Matrix::derive(fixed_genome, &genome);
        matrix.norm_densities(&mut calibration_substrate(fixed_genome.clone(), box_len));
        Sim { substrate, matrix, fixed_genome: fixed_genome.clone(), genome, tick: 0, threaded: false }
    }
    /// Advance ticks steps (headless, as fast as possible)
    pub fn run(&mut self, ticks: u64) {
        for _ in 0..ticks { self.step(); }
    }
    /// Runs one Lenia step
    pub fn step(&mut self) {
        step(&mut self.substrate, &self.matrix, self.fixed_genome.dt, self.threaded);
        self.tick = self.tick.wrapping_add(1);
    }
}
/// Uniform population norm_densities measures against, baseline Matrix behavior
pub fn calibration_substrate(mut calib_fg: FixedGenome, box_len: f32) -> Substrate {
    calib_fg.seed = CALIBRATION_SEED;
    let mut calib_sub = Substrate::build(&calib_fg, box_len);
    init_particle_traits(&mut calib_sub, &vec![0.0; calib_fg.anchor_count], CALIBRATION_SEED); // uniform logits
    calib_sub
}