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
impl Params {
    /// Genes in one pair block: amp/peak/width per shell and per bump, then the directed weight.
    pub fn pair_stride(&self) -> usize { 3 * (self.shells + self.bumps) + 1 }
    /// Full genome length: coordination, one logit per anchor, then anchor_count^2 pair blocks.
    pub fn gene_len(&self) -> usize { 1 + self.anchor_count + self.anchor_count * self.anchor_count * self.pair_stride() }
    /// Where pair (source, destination)'s block sits in a full genome.
    pub fn pair_genes(&self, source: usize, destination: usize) -> std::ops::Range<usize> {
        let start = 1 + self.anchor_count + (source * self.anchor_count + destination) * self.pair_stride();
        start..start + self.pair_stride()
    }
}
