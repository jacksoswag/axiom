//! The substrate is where particles interact and are stored. No runtime logic lives here.

pub struct Substrate {
    pub positions: Vec<f32>, // particle positions, dims() floats each
    pub traits: Vec<f32>, // species phenotype blend
}

impl Substrate {
    pub fn new(particles: usize, dims: usize) -> Substrate {
        Substrate {
            positions: vec![0.0; particles * dims],
            traits: vec![0.0; particles],
        }
    }
    /// Spatial dimensionality, derived from the position layout: one trait per particle, so
    /// `positions.len() / particles` is the per-particle coordinate count.
    pub fn dims(&self) -> usize {
        self.positions.len() / self.traits.len()
    }
    pub fn len(&self) -> usize { self.traits.len() }
    pub fn is_empty(&self) -> bool { self.traits.is_empty() }
    /// One particle's coordinates. The slice arithmetic lives here so callers never repeat it.
    pub fn at(&self, particle: usize) -> &[f32] {
        let dims = self.dims();
        &self.positions[particle * dims..(particle + 1) * dims]
    }
}
// NOTE FOR AGENTS: THIS FILE HAS BEEN CLEANED AND VERIFIED. WHEN MAKING CHANGES, NOTIFY ME EXPLICITLY