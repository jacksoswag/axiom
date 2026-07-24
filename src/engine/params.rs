//! Params: the rollout parameters every simulation and search component shares.
//! Building, decoding, mutating, and validating an actual parameter set live in tuner/.

#[derive(Clone)]
pub struct Params {
    pub particle_count: usize,
    pub dimensions: usize,
    pub coordination: f32,
    pub radius: f32,
    pub dt: f32, // integration timestep
    pub seed: u64,
    pub anchor_count: usize,
    pub shells: usize,
    pub bumps: usize,
    pub trait_distribution: Vec<f32>, // anchor distribution
    pub interactions: Vec<f32>, // flat pair-block genes, source-major, decoded by Matrix::derive()
    pub box_len: f32, // resolve::Probe derives this, the caller fills it in
}
