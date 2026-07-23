//! The genome codec: a flat, bounded `Vec<f32>` on one side, typed runtime parameters on the
//! other. Nothing here simulates or measures, which is what keeps mutation operators free to
//! treat the genotype as an anonymous vector of numbers.
//!
//! ```text
//! [ coordination ]
//! [ trait-distribution logits ]
//! [ anchor × anchor pair blocks, source-major ]
//!
//! pair block = shells × [kernel beta, mu, sigma]
//!            + bumps  × [growth beta, mu, sigma]
//!            + [directed weight]
//! ```

use crate::engine::geometry::{Geometry, GeometryScale};
use crate::engine::kernel::CUTOFF_SIGMA;
use crate::engine::rng::Rng;
use crate::util::finite;

const MAX_KERNEL_REACH: f32 = 0.375;
const COORDINATION_BOUNDS: (f32, f32) = (3.0, 20.0);
const TRAIT_LOGIT_BOUNDS: (f32, f32) = (-4.0, 4.0);

/// Clamp a genome into its bounds in place, replacing any non-finite gene with the lower bound.
fn clamp_to(genome: &mut [f32], bounds: &[(f32, f32)]) {
    for (gene, &(low, high)) in genome.iter_mut().zip(bounds) {
        *gene = if gene.is_finite() {
            gene.clamp(low, high)
        } else {
            low
        };
    }
}

/// A decoded genome: everything one rollout needs except the box size, which `Geometry` derives.
#[derive(Clone, Debug)]
pub struct Params {
    pub particles: usize,
    pub dimensions: usize,
    pub coordination: f32,
    pub radius: f32,
    pub rate: f32, // integration rate; dt is its reciprocal
    pub seed: u64,
    pub anchors: usize,
    pub shells: usize,
    pub bumps: usize,
    // Resolved initial-density logits, one per equal-width trait bin.
    pub trait_logits: Vec<f32>,
}

impl Default for Params {
    /// Mirrors `Caps::default` so reference geometry and tests share one set of values.
    fn default() -> Params {
        let caps = Caps::default();
        Params {
            particles: caps.particles,
            dimensions: caps.dimensions,
            coordination: 9.0,
            radius: caps.radius,
            rate: caps.rate,
            seed: caps.seed,
            anchors: caps.anchors,
            shells: caps.shells,
            bumps: caps.bumps,
            trait_logits: vec![0.0; caps.anchors],
        }
    }
}

impl Params {
    pub fn dt(&self) -> f32 {
        1.0 / self.rate
    }

    pub fn layout(&self) -> Layout {
        Layout {
            anchors: self.anchors,
            shells: self.shells,
            bumps: self.bumps,
        }
    }

    /// Measure a fresh density reference. Callers evaluating more than one genome should probe
    /// once and reuse it through [`Params::geometry_with`].
    pub fn geometry_scale(&self) -> GeometryScale {
        GeometryScale::probe(self.particles, self.dimensions, self.radius)
    }

    pub fn geometry(&self) -> Geometry {
        self.geometry_scale().geometry(self.coordination)
    }

    pub fn geometry_with(&self, scale: GeometryScale) -> Geometry {
        scale.geometry(self.coordination)
    }
}

/// Shape of the pair-block region of a genome.
#[derive(Clone, Copy, Debug)]
pub struct Layout {
    pub anchors: usize,
    pub shells: usize,
    pub bumps: usize,
}

impl Layout {
    /// Anchor count implied by a full genome length, inverting `1 + N + N² · genes_per_interaction`.
    ///
    /// That quadratic is strictly increasing in `N`, so a length admits at most one anchor count.
    /// This makes the genome self-describing: nothing outside it has to declare its shape, which
    /// is what lets one search carry genomes of differing anchor counts.
    pub fn for_genome(genome_len: usize, shells: usize, bumps: usize) -> Option<Layout> {
        let per = 3 * (shells + bumps) + 1;
        let anchors = (genome_len.checked_sub(1)? / per).isqrt();
        (anchors > 0 && per * anchors * anchors + anchors + 1 == genome_len).then_some(Layout {
            anchors,
            shells,
            bumps,
        })
    }

    pub fn genes_per_interaction(&self) -> usize {
        3 * (self.shells + self.bumps) + 1
    }

    pub fn genome_len(&self) -> usize {
        self.anchors * self.anchors * self.genes_per_interaction()
    }

    pub fn index(&self, source: usize, destination: usize) -> usize {
        source * self.anchors + destination
    }

    pub fn bounds(&self, radius: f32, extent: f32) -> Vec<(f32, f32)> {
        let reach = MAX_KERNEL_REACH * extent;
        let mu_max = (2.0 * radius).min(reach * 0.5).max(1e-3);
        let sigma_min = (0.05 * radius).min(mu_max * 0.25).max(1e-4);
        let sigma_max = radius
            .min(reach / (2.0 * CUTOFF_SIGMA))
            .max(sigma_min * 2.0);
        let mut bounds = Vec::with_capacity(self.genome_len());
        for _ in 0..self.anchors * self.anchors {
            for _ in 0..self.shells {
                bounds.extend([(0.0, 1.0), (0.0, mu_max), (sigma_min, sigma_max)]);
            }
            for _ in 0..self.bumps {
                bounds.extend([(0.0, 1.0), (0.0, 3.0), (0.05, 1.5)]);
            }
            bounds.push((-100.0, 100.0));
        }
        bounds
    }

    pub fn default_genome(&self, radius: f32, extent: f32) -> Vec<f32> {
        let mut genome = vec![0.0; self.genome_len()];
        let stride = self.genes_per_interaction();
        for source in 0..self.anchors {
            for destination in 0..self.anchors {
                let base = self.index(source, destination) * stride;
                for shell in 0..self.shells {
                    genome[base + shell * 3] = if shell == 0 { 1.0 } else { 0.0 };
                    genome[base + shell * 3 + 1] = radius;
                    genome[base + shell * 3 + 2] = radius * 0.5;
                }
                let bump_base = base + self.shells * 3;
                for bump in 0..self.bumps {
                    genome[bump_base + bump * 3] = if bump == 0 { 1.0 } else { 0.0 };
                    genome[bump_base + bump * 3 + 1] = 1.5;
                    genome[bump_base + bump * 3 + 2] = 0.5;
                }
                genome[bump_base + self.bumps * 3] = if source == destination { 40.0 } else { 0.0 };
            }
        }
        clamp_to(&mut genome, &self.bounds(radius, extent));
        genome
    }

    pub fn clamp(&self, genome: &mut [f32], radius: f32, extent: f32) {
        clamp_to(genome, &self.bounds(radius, extent));
    }
}

/// Campaign-level configuration: the fixed parts of a world that no genome may vary, plus the
/// shape used to size freshly generated genomes.
#[derive(Clone, Debug, PartialEq)]
pub struct Caps {
    pub particles: usize,
    pub dimensions: usize,
    pub anchors: usize,
    pub radius: f32,
    pub rate: f32,
    pub seed: u64,
    pub shells: usize,
    pub bumps: usize,
}

impl Default for Caps {
    fn default() -> Self {
        Caps {
            particles: 1200,
            dimensions: 3,
            anchors: 4,
            radius: 12.0,
            rate: 10.0,
            seed: 1,
            shells: 3,
            bumps: 2,
        }
    }
}

impl Caps {
    /// Probe once at campaign creation and pass the result to every candidate evaluation.
    pub fn geometry_scale(&self) -> GeometryScale {
        GeometryScale::probe(self.particles, self.dimensions, self.radius)
    }

    /// Coordination plus one initial trait-density logit per anchor/bin.
    pub fn world_genes(&self) -> usize {
        1 + self.anchors
    }

    pub fn layout(&self) -> Layout {
        Layout {
            anchors: self.anchors,
            shells: self.shells,
            bumps: self.bumps,
        }
    }

    pub fn genome_len(&self) -> usize {
        self.world_genes() + self.layout().genome_len()
    }

    /// Decode a genome into runtime parameters. Anchor count comes from the genome's own length,
    /// so `Caps::anchors` only sizes freshly generated genomes and never constrains what the
    /// runtime will accept.
    pub fn resolve<'a>(&self, genome: &'a [f32]) -> (Params, &'a [f32]) {
        let layout = Layout::for_genome(genome.len(), self.shells, self.bumps).unwrap_or_else(|| {
            panic!(
                "genome of {} genes fits no anchor count at {} shells and {} bumps",
                genome.len(),
                self.shells,
                self.bumps
            )
        });
        let (world, interactions) = genome.split_at(1 + layout.anchors);
        (
            Params {
                particles: self.particles,
                dimensions: self.dimensions,
                coordination: finite(world[0], COORDINATION_BOUNDS.0)
                    .clamp(COORDINATION_BOUNDS.0, COORDINATION_BOUNDS.1),
                radius: self.radius,
                rate: self.rate,
                seed: self.seed,
                anchors: layout.anchors,
                shells: self.shells,
                bumps: self.bumps,
                trait_logits: world[1..]
                    .iter()
                    .map(|&gene| {
                        finite(gene, 0.0).clamp(TRAIT_LOGIT_BOUNDS.0, TRAIT_LOGIT_BOUNDS.1)
                    })
                    .collect(),
            },
            interactions,
        )
    }

    /// Pair-block bounds are widest at the largest box, so they are taken at maximum coordination
    /// and stay valid for every genome a search can produce.
    pub fn bounds(&self, scale: GeometryScale) -> Vec<(f32, f32)> {
        let extent = scale.geometry(COORDINATION_BOUNDS.1).extent;
        let mut bounds = Vec::with_capacity(self.genome_len());
        bounds.push(COORDINATION_BOUNDS);
        bounds.extend(std::iter::repeat_n(TRAIT_LOGIT_BOUNDS, self.anchors));
        bounds.extend(self.layout().bounds(self.radius, extent));
        bounds
    }

    pub fn default_genome(&self, scale: GeometryScale) -> Vec<f32> {
        let extent = scale.geometry(COORDINATION_BOUNDS.1).extent;
        let mut genome = vec![9.0];
        genome.extend(std::iter::repeat_n(0.0, self.anchors));
        genome.extend(self.layout().default_genome(self.radius, extent));
        genome
    }
}

pub fn random_genome(bounds: &[(f32, f32)], rng: &mut Rng) -> Vec<f32> {
    bounds
        .iter()
        .map(|&(low, high)| rng.range(low, high))
        .collect()
}
