//! A world: the authoritative state a rollout advances, plus the deterministic seeding that
//! produces it. Everything here is reconstructible from a genome, a `Geometry`, and a seed.

use crate::engine::genome::Params;
use crate::engine::geometry::{Geometry, CALIBRATION_SEED};
use crate::engine::grid::Grid;
use crate::engine::interaction::Net;
use crate::engine::lenia;
use crate::engine::r#trait::seed_traits;
use crate::engine::rng::Rng;
use crate::engine::substrate::Substrate;

/// Deterministic initial particle state for a resolved genome: uniform positions from
/// `params.seed`, traits drawn to match `params.trait_logits`.
fn seeded_substrate(params: &Params, extent: f32) -> Substrate {
    let mut substrate = Substrate::new(params.particles, params.dimensions);
    let mut positions = Rng::new(params.seed);
    for coordinate in substrate.positions.iter_mut() {
        *coordinate = positions.unit() * extent;
    }
    seed_traits(&mut substrate, &params.trait_logits, params.seed);
    substrate
}

/// The shared, seed-fixed reference population every genome calibrates its norms against.
fn calibration_substrate(params: &Params, extent: f32) -> Substrate {
    let mut reference = params.clone();
    reference.seed = CALIBRATION_SEED;
    seeded_substrate(&reference, extent)
}

pub struct World {
    pub substrate: Substrate,
    pub net: Net,
    pub geometry: Geometry,
    pub dt: f32,
    pub tick: u64,
    pub seed: u64,
    /// The pair-block genes this world was built from. Kept so a recipe can be compared as the
    /// genome it is, rather than by re-deriving and diffing the decoded law.
    genome: Vec<f32>,
    grid: Grid,
}

impl World {
    pub fn new(params: &Params, genome: &[f32]) -> World {
        World::with_geometry(params, &params.geometry(), genome)
    }

    pub fn with_geometry(params: &Params, geometry: &Geometry, genome: &[f32]) -> World {
        let layout = params.layout();
        assert_eq!(
            genome.len(),
            layout.genome_len(),
            "genome does not match layout {layout:?}"
        );
        let mut world = World {
            substrate: seeded_substrate(params, geometry.extent),
            net: Net::from_genome(&layout, genome),
            geometry: *geometry,
            dt: params.dt(),
            tick: 0,
            seed: params.seed,
            genome: genome.to_vec(),
            grid: Grid::default(),
        };
        let reference = calibration_substrate(params, geometry.extent);
        world.calibrate(&reference);
        world
    }

    /// Rebuild a world from an authoritative particle snapshot. Norms are first calibrated from
    /// the deterministic initial seed, then may be replaced by the saved values by checkpoint
    /// restoration. The spatial index remains a cache and is rebuilt on the next step.
    pub fn from_snapshot(
        params: &Params,
        geometry: &Geometry,
        genome: &[f32],
        positions: Vec<f32>,
        traits: Vec<f32>,
        tick: u64,
    ) -> World {
        assert_eq!(positions.len(), traits.len() * params.dimensions);
        let mut world = World::with_geometry(params, geometry, genome);
        world.substrate = Substrate { positions, traits };
        world.tick = tick;
        world.grid = Grid::default();
        world
    }

    pub fn step(&mut self) {
        self.grid.rebuild(
            &self.substrate.positions,
            self.substrate.dims(),
            self.geometry.extent,
            self.net.max_reach(),
        );
        lenia::step(
            &mut self.substrate,
            &self.net,
            self.dt,
            self.geometry.extent,
            self.geometry.softening,
            &self.grid,
        );
        self.tick = self.tick.wrapping_add(1);
    }

    pub fn set_genome(&mut self, params: &Params, genome: &[f32]) {
        let layout = params.layout();
        assert_eq!(
            genome.len(),
            layout.genome_len(),
            "genome does not match layout {layout:?}"
        );
        self.net = Net::from_genome(&layout, genome);
        self.genome = genome.to_vec();
        let reference = calibration_substrate(params, self.geometry.extent);
        self.calibrate(&reference);
    }

    /// Whether a supplied recipe describes this world's law and geometry. Decoding is
    /// deterministic, so comparing genomes answers this exactly; measured norms are excluded
    /// because checkpoints save them separately.
    pub fn matches_recipe(&self, params: &Params, geometry: &Geometry, genome: &[f32]) -> bool {
        genome.len() == params.layout().genome_len()
            && self.genome.len() == genome.len()
            && self
                .genome
                .iter()
                .zip(genome)
                .all(|(held, given)| held.to_bits() == given.to_bits())
            && self.substrate.len() == params.particles
            && self.seed == params.seed
            && self.dt.to_bits() == params.dt().to_bits()
            && self.geometry.extent.to_bits() == geometry.extent.to_bits()
            && self.geometry.softening.to_bits() == geometry.softening.to_bits()
    }

    /// Stable hash of the authoritative dynamic state. Derived caches and the grid deliberately
    /// do not contribute.
    pub fn state_hash(&self) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for byte in self
            .tick
            .to_le_bytes()
            .into_iter()
            .chain(
                self.substrate
                    .positions
                    .iter()
                    .flat_map(|value| value.to_bits().to_le_bytes()),
            )
            .chain(
                self.substrate
                    .traits
                    .iter()
                    .flat_map(|value| value.to_bits().to_le_bytes()),
            )
        {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        hash
    }

    fn calibrate(&mut self, reference: &Substrate) {
        self.net
            .calibrate(reference, self.geometry.extent, self.geometry.softening);
    }
}
