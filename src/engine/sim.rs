//! Sim: the authoritative state of a run, reconstructible from Params alone. params.box_len
//! must already be resolved by the caller. Sim only creates and runs the Lenia simulation.

use crate::engine::matrix::Matrix;
use crate::engine::lenia::step;
use crate::engine::params::Params;
use crate::engine::resolve::CALIBRATION_SEED;
use crate::engine::r#trait::init_particle_traits;
use crate::engine::substrate::Substrate;

pub struct Sim {
    pub substrate: Substrate,
    pub matrix: Matrix,
    pub params: Params,
    pub tick: u64,
}
impl Sim {
    /// A fresh Sim, ready to run.
    pub fn new(params: &Params) -> Sim {
        let mut substrate = Substrate::build(params);
        init_particle_traits(&mut substrate, &params.trait_distribution, params.seed);
        
        let mut calib_params = params.clone(); calib_params.seed = CALIBRATION_SEED;
        let mut calibration = Substrate::build(&calib_params);
        init_particle_traits(&mut calibration, &vec![0.0; params.anchor_count], CALIBRATION_SEED); // uniform logits: every anchor pair sees reference mass
        let mut matrix = Matrix::derive(params);
        matrix.norm_densities(&mut calibration);
        Sim { substrate, matrix, params: params.clone(), tick: 0 }
    }
    /// Advance ticks steps (headless, as fast as possible)
    pub fn run(&mut self, ticks: u64) {
        for _ in 0..ticks { self.step(); }
    }
    /// Runs one Lenia step
    pub fn step(&mut self) {
        step(&mut self.substrate, &self.matrix, self.params.dt);
        self.tick = self.tick.wrapping_add(1);
    }
}