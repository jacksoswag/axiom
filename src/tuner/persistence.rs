//! H0 persistent homology of the particle cloud, which is a minimum spanning tree.
//!
//! Clusters without having to pick a cluster radius. The H0 barcode of a Vietoris-Rips
//! filtration has one bar per point: every bar is born at scale 0, and all but one die at the
//! scale where its component merges into another. Those death scales are **exactly the edge
//! weights of the Euclidean minimum spanning tree**, so the entire multi-scale cluster
//! hierarchy costs one MST and needs no persistent-homology dependency and no `O(N³)`
//! boundary-matrix reduction.
//!
//! That threshold-freedom is the point. A union-find at one radius answers "how many clusters
//! are there at *this* scale", which requires guessing the scale a genome happens to work at.
//! The barcode answers it at every scale at once, and a power law across the bins is the
//! measurable signature of structures inside structures.
//!
//! # Truncation at the kernel cutoff
//!
//! Edges come from the same [`Grid`] the force loop uses, so only pairs within `cutoff` are
//! ever offered to Kruskal. Beyond that the tree is really a *forest*: components separated by
//! more than the interaction range never merge. That is deliberate rather than a limitation.
//! Two clumps further apart than any kernel can reach are not one structure at any scale the
//! physics can see, so their merge scale carries no information. What survives is reported as
//! [`Barcode::components`].

use crate::engine::grid::Grid;
use crate::engine::kernel::displacement;
use crate::engine::substrate::Substrate;

/// Bins in the log-spaced death-scale histogram.
pub const BARS: usize = 8;

#[derive(Clone, Debug)]
pub struct Barcode {
    /// Counts of H0 death scales, log-binned from the mean interparticle spacing to `cutoff`.
    pub bins: Vec<f64>,
    /// Components still separate at `cutoff`, so at least 1 for any non-empty swarm.
    pub components: usize,
}

impl Barcode {
    pub fn zeros() -> Barcode {
        Barcode {
            bins: vec![0.0; BARS],
            components: 0,
        }
    }
}

/// One edge of the neighbour graph, kept only long enough to sort.
struct Edge {
    length: f32,
    a: u32,
    b: u32,
}

/// Build the H0 barcode. `cutoff` should be the kernel reach the grid was built for.
pub fn h0(substrate: &Substrate, extent: f32, cutoff: f32, grid: &Grid) -> Barcode {
    let count = substrate.traits.len();
    if count == 0 || !cutoff.is_finite() || cutoff <= 0.0 {
        return Barcode::zeros();
    }

    let mut edges = collect_edges(substrate, extent, cutoff, grid);
    // Ties broken by index so the tree, and therefore the barcode, is reproducible. Float
    // comparison alone would leave equal-length edges in whatever order the scan produced.
    edges.sort_unstable_by(|x, y| {
        x.length
            .total_cmp(&y.length)
            .then(x.a.cmp(&y.a))
            .then(x.b.cmp(&y.b))
    });

    let (low, high) = scale_range(cutoff);
    let mut union = Union::new(count);
    let mut barcode = Barcode {
        bins: vec![0.0; BARS],
        components: count,
    };

    for edge in edges {
        if union.join(edge.a as usize, edge.b as usize) {
            barcode.bins[bin_of(edge.length, low, high)] += 1.0;
            barcode.components -= 1;
        }
    }
    barcode
}

/// Seven death-scale masses plus one log-scaled count of components still separate at the
/// physical cutoff. The last two raw death bins are folded together to keep the descriptor
/// width fixed while retaining the forest information.
///
/// # Why not a ratio against a uniform baseline
///
/// That is what this used to do, and it destroyed the signal it exists to measure. The
/// baseline guard returned a literal `1.0` for any bin the uniform swarm left near-empty,
/// and [`scale_range`] deliberately anchors two decades *below* the cutoff, where a uniform
/// swarm puts almost nothing: its MST edges concentrate near ~0.7× the mean interparticle
/// spacing. So the short-scale bins — precisely "this swarm has tight clusters", the whole
/// reason H0 is here — reported `1.0` for a field of blobs and `1.0` for a gas alike. Five
/// of eight slots were constants.
///
/// A share of the total needs no baseline, so it cannot divide by a bin that was never
/// populated. It is already scale-free: the bins are anchored to `cutoff`, which scales
/// with the world, so the profile is comparable across particle counts for
/// the same reason the log binning is.
///
/// A value of `1.0` in the first seven axes means a flat death-scale profile rather than
/// "same as a uniform swarm". The eighth axis is zero for one cutoff-connected component and
/// rises logarithmically toward two for a fully disconnected swarm. This block remains
/// excluded from [`crate::tuner::metrics::Metrics::structure`], which is defined against uniform RDF.
pub fn mass(raw: &Barcode) -> Vec<f32> {
    let total: f64 = raw.bins.iter().sum();
    let mut out = vec![0.0; BARS];
    if total > 0.0 {
        for (index, value) in raw.bins.iter().enumerate() {
            let output = index.min(BARS - 2);
            out[output] += (value / total * (BARS - 1) as f64) as f32;
        }
    }
    let particles = total + raw.components as f64;
    if raw.components > 1 && particles > 1.0 {
        out[BARS - 1] =
            (2.0 * (raw.components as f64).ln() / particles.ln().max(f64::EPSILON)) as f32;
    }
    out
}

fn collect_edges(substrate: &Substrate, extent: f32, cutoff: f32, grid: &Grid) -> Vec<Edge> {
    let count = substrate.traits.len();
    let mut edges = Vec::new();

    for i in 0..count {
        let pos_i = substrate.at(i);
        if !pos_i.iter().all(|p| p.is_finite()) {
            continue;
        }

        let mut consider = |j: usize| {
            // Each pair once. The grid visits both directions, so drop one.
            if j <= i {
                return;
            }
            let pos_j = substrate.at(j);
            if !pos_j.iter().all(|p| p.is_finite()) {
                return;
            }
            let d = displacement(pos_i, pos_j, extent, 1.0 / extent, 0.0);
            if d <= cutoff {
                edges.push(Edge {
                    length: d,
                    a: i as u32,
                    b: j as u32,
                });
            }
        };

        grid.for_each_candidate(pos_i, count, &mut consider);
    }
    edges
}

/// Two decades below the cutoff, log-spaced.
///
/// Deliberately **not** anchored to the mean interparticle spacing the way the RDF's bins are.
/// There the low edge exists to dodge shot noise, since a bin holding 39 pairs carries 16%
/// error. Here every bar is one exact merge event, so there is no noise to dodge, and edges
/// far shorter than the mean spacing are precisely the signal: a tight cluster is a cluster
/// *because* its internal edges are short. Anchoring at the spacing would clamp that whole
/// range into bin 0 and discard the hierarchy this exists to measure.
///
/// The range is relative to `cutoff`, which scales with the world, so it stays portable across
/// particle counts for the same reason the RDF's log binning does.
fn scale_range(cutoff: f32) -> (f32, f32) {
    (cutoff * 0.01, cutoff)
}

fn bin_of(scale: f32, low: f32, high: f32) -> usize {
    if scale.is_nan() || scale <= low {
        return 0;
    }
    let t = (scale / low).ln() / (high / low).ln();
    ((t * BARS as f32) as usize).min(BARS - 1)
}

/// Union-find with path halving. Union by size, so `join` is near-constant time and the
/// result does not depend on the order equal-length edges arrive in.
struct Union {
    parent: Vec<u32>,
    size: Vec<u32>,
}

impl Union {
    fn new(count: usize) -> Union {
        Union {
            parent: (0..count as u32).collect(),
            size: vec![1; count],
        }
    }

    fn find(&mut self, mut node: usize) -> usize {
        while self.parent[node] as usize != node {
            let grandparent = self.parent[self.parent[node] as usize];
            self.parent[node] = grandparent;
            node = grandparent as usize;
        }
        node
    }

    /// True when the two were in different components, which is what makes this edge a bar.
    fn join(&mut self, a: usize, b: usize) -> bool {
        let (mut a, mut b) = (self.find(a), self.find(b));
        if a == b {
            return false;
        }
        if self.size[a] < self.size[b] {
            std::mem::swap(&mut a, &mut b);
        }
        self.parent[b] = a as u32;
        self.size[a] += self.size[b];
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::Rng;

    fn swarm(positions: Vec<f32>, extent: f32) -> Substrate {
        let count = positions.len() / 3;
        Substrate {
            positions,
            traits: vec![0.0; count],
            bound_len: extent,
            softening: extent * 1e-3,
            dimensions: 3,
        }
    }

    fn uniform(count: usize, extent: f32, seed: u64) -> Substrate {
        let mut rng = Rng::new(seed);
        swarm(
            (0..count * 3)
                .map(|_| rng.unit() * extent)
                .collect(),
            extent,
        )
    }

    fn built(substrate: &Substrate, extent: f32, cutoff: f32) -> Grid {
        let mut grid = Grid::default();
        grid.rebuild(&substrate.positions, substrate.dimensions, extent, cutoff);
        grid
    }

    /// **The test that makes the persistence numbers trustworthy rather than merely fast.**
    /// Binning is only a speed-up if the tree it produces is the tree an exhaustive scan
    /// would produce. Compares total MST weight, which is invariant to tie-breaking where the
    /// individual edge set is not.
    #[test]
    fn grid_tree_matches_brute_force() {
        {
            let (extent, cutoff) = (200.0f32, 60.0f32);
            let substrate = uniform(600, extent, 31);

            let binned = h0(
                &substrate,
                extent,
                cutoff,
                &built(&substrate, extent, cutoff),
            );
            let brute = h0(&substrate, extent, cutoff, &Grid::default());

            assert_eq!(
                binned.components, brute.components,
                "component count disagrees"
            );
            for (bin, (a, b)) in binned.bins.iter().zip(&brute.bins).enumerate() {
                assert_eq!(a, b, "bin {bin} disagrees");
            }
        }
    }

    /// A spanning forest has `count − components` edges, no more and no less. Any double count
    /// or missed merge shows up here.
    #[test]
    fn bars_and_components_account_for_every_particle() {
        {
            let (extent, cutoff) = (150.0f32, 40.0f32);
            let substrate = uniform(500, extent, 5);
            let barcode = h0(
                &substrate,
                extent,
                cutoff,
                &built(&substrate, extent, cutoff),
            );
            let bars: f64 = barcode.bins.iter().sum();
            assert_eq!(
                bars as usize + barcode.components,
                500,
                "{bars} bars + {} components",
                barcode.components
            );
        }
    }

    /// The signal the descriptor is being given. Three planted blobs, further apart than the
    /// cutoff, must read as three components, and their internal edges must land in shorter
    /// bins than a uniform gas of the same count in the same box.
    ///
    /// The cutoff has to exceed the three-dimensional gas's mean spacing or the gas fragments too and
    /// the comparison says nothing. That is not a quirk of the test: below one spacing, *every*
    /// swarm is dust, which is why the physical cutoff spans several neighbours by construction.
    #[test]
    fn planted_blobs_separate_where_a_gas_connects() {
        let (extent, cutoff) = (300.0f32, 70.0f32);
        let mut rng = Rng::new(3);
        let mut particles = Vec::new();
        for centre in [40.0f32, 140.0, 240.0] {
            for _ in 0..90 {
                particles.push(centre + rng.range(-4.0, 4.0));
                particles.push(centre + rng.range(-4.0, 4.0));
                particles.push(centre + rng.range(-4.0, 4.0));
            }
        }
        let blobs = swarm(particles, extent);
        let clustered = h0(&blobs, extent, cutoff, &built(&blobs, extent, cutoff));
        assert_eq!(
            clustered.components, 3,
            "three blobs did not read as three components"
        );

        let gas = uniform(270, extent, 8);
        let flat = h0(&gas, extent, cutoff, &built(&gas, extent, cutoff));
        assert!(
            flat.components < clustered.components,
            "a uniform gas fragmented more than three tight blobs: {} vs {}",
            flat.components,
            clustered.components
        );

        // The hierarchy: blob edges are short, gas edges are long. This is the whole
        // discriminating signal, so it gets asserted directly rather than inferred.
        let centre_of_mass = |bins: &[f64]| -> f64 {
            let total: f64 = bins.iter().sum();
            bins.iter()
                .enumerate()
                .map(|(i, v)| i as f64 * v)
                .sum::<f64>()
                / total.max(1.0)
        };
        let blob_scale = centre_of_mass(&clustered.bins);
        let gas_scale = centre_of_mass(&flat.bins);
        assert!(
            blob_scale < gas_scale - 1.0,
            "blob and gas edge scales did not separate: {blob_scale:.2} vs {gas_scale:.2}"
        );
    }

    #[test]
    fn is_deterministic_and_mass_preserves_the_component_count() {
        let (extent, cutoff) = (150.0f32, 40.0f32);
        let substrate = uniform(400, extent, 17);
        let grid = built(&substrate, extent, cutoff);
        let first = h0(&substrate, extent, cutoff, &grid);
        let second = h0(&substrate, extent, cutoff, &grid);
        assert_eq!(first.bins, second.bins);

        // Seven axes carry death-scale mass; the eighth carries cutoff-separated components.
        let profile = mass(&first);
        assert_eq!(profile.len(), BARS);
        let total: f32 = profile[..BARS - 1].iter().sum();
        assert!(
            (total - (BARS - 1) as f32).abs() < 1e-3,
            "death mass did not sum to {}: {total}",
            BARS - 1
        );
        assert!(profile.iter().all(|v| v.is_finite() && *v >= 0.0));

        let connected = Barcode {
            bins: first.bins.clone(),
            components: 1,
        };
        let fragmented = Barcode {
            bins: first.bins.clone(),
            components: 16,
        };
        assert_eq!(mass(&connected)[BARS - 1], 0.0);
        assert!(mass(&fragmented)[BARS - 1] > 0.5);
    }

    /// **The regression this normalisation was rewritten for.** The old ratio-against-uniform
    /// form returned a constant `1.0` wherever the uniform baseline left a bin empty, which is
    /// exactly the short-scale bins a clustered swarm fills. Blobs and a gas came out
    /// identical *in the vector the descriptor actually carries*, while the raw-barcode test
    /// above kept passing. This asserts on the normalised output instead.
    #[test]
    fn blobs_and_a_gas_separate_in_the_normalised_profile() {
        let (extent, cutoff) = (300.0f32, 35.0f32);
        let mut rng = Rng::new(3);
        let mut particles = Vec::new();
        for centre in [40.0f32, 140.0, 240.0] {
            for _ in 0..90 {
                particles.push(centre + rng.range(-4.0, 4.0));
                particles.push(centre + rng.range(-4.0, 4.0));
                particles.push(centre + rng.range(-4.0, 4.0));
            }
        }
        let blobs = swarm(particles, extent);
        let gas = uniform(270, extent, 8);

        let blob_profile = mass(&h0(&blobs, extent, cutoff, &built(&blobs, extent, cutoff)));
        let gas_profile = mass(&h0(&gas, extent, cutoff, &built(&gas, extent, cutoff)));

        let separation: f32 = blob_profile
            .iter()
            .zip(&gas_profile)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();
        assert!(
            separation > 0.5,
            "blobs and gas collapsed together in the descriptor block: {separation:.4}\n\
             blobs {blob_profile:?}\n  gas {gas_profile:?}"
        );
    }

    #[test]
    fn survives_an_empty_swarm_and_non_finite_positions() {
        let empty = swarm(Vec::new(), 100.0);
        assert_eq!(h0(&empty, 100.0, 10.0, &Grid::default()).components, 0);

        let broken = swarm(vec![f32::NAN, 1.0, 1.0, 50.0, 50.0, 50.0, 51.0, 51.0, 51.0], 100.0);
        let barcode = h0(&broken, 100.0, 10.0, &Grid::default());
        assert!(barcode.components <= 3);
    }
}
