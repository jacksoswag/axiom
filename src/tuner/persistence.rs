//! H0 persistent homology of the particle cloud, which is a minimum spanning tree.
//! Every H0 bar is born at scale 0 and dies where its component merges, and those death scales
//! are exactly the edge weights of the Euclidean MST, so the whole multi-scale cluster
//! hierarchy costs one MST and no boundary-matrix reduction. Edges stop at the kernel cutoff,
//! so the tree is really a forest, deliberately: clumps further apart than any kernel can reach
//! are not one structure at any scale the physics can see. Survivors are Barcode::components.

use crate::engine::kernel::distance_sq;
use crate::engine::substrate::Substrate;

/// Bins in the log-spaced death-scale histogram.
pub const BARS: usize = 8;

#[derive(Clone, Debug)]
pub struct Barcode {
    pub bins: Vec<f64>, // H0 death-scale counts, log-binned up to cutoff
    pub components: usize, // still separate at cutoff, at least 1 for any non-empty swarm
}
impl Barcode {
    pub fn zeros() -> Barcode { Barcode { bins: vec![0.0; BARS], components: 0 } }
}

/// One edge of the neighbor graph, kept only long enough to sort.
struct Edge { length: f32, a: u32, b: u32 }

/// Build the H0 barcode at the kernel cutoff. Owns its grid rebuild, so a caller only hands
/// over the substrate; rebuild at a huge cutoff first to force the all-pairs path.
pub fn h0(substrate: &mut Substrate, cutoff: f32) -> Barcode {
    let count = substrate.traits.len();
    if count == 0 || !cutoff.is_finite() || cutoff <= 0.0 { return Barcode::zeros(); }
    substrate.rebuild_grid(cutoff);

    let mut edges = collect_edges(substrate, cutoff);
    // Ties broken by index so the tree, and therefore the barcode, is reproducible. Float
    // comparison alone would leave equal-length edges in whatever order the scan produced.
    edges.sort_unstable_by(|x, y| x.length.total_cmp(&y.length).then(x.a.cmp(&y.a)).then(x.b.cmp(&y.b)));

    let (low, high) = scale_range(cutoff);
    let mut union = Union::new(count);
    let mut barcode = Barcode { bins: vec![0.0; BARS], components: count };
    for edge in edges {
        if union.join(edge.a as usize, edge.b as usize) { // first join at this scale = one bar
            barcode.bins[bin_of(edge.length, low, high)] += 1.0;
            barcode.components -= 1;
        }
    }
    barcode
}

/// Seven death-scale mass shares plus one log-scaled component mass. A share of the total needs
/// no uniform baseline, so it cannot divide by a bin the baseline left empty (the old ratio form
/// returned a constant 1.0 exactly in the short-scale bins that ARE the clustering signal).
/// 1.0 in the first seven axes means a flat profile; the eighth is 0 for one cutoff-connected
/// component and rises toward 2 for dust. Excluded from Metrics::structure, which is RDF-based.
pub fn mass(raw: &Barcode) -> Vec<f32> {
    let total: f64 = raw.bins.iter().sum();
    let mut out = vec![0.0; BARS];
    if total > 0.0 {
        for (index, value) in raw.bins.iter().enumerate() { // last two raw bins fold together
            out[index.min(BARS - 2)] += (value / total * (BARS - 1) as f64) as f32;
        }
    }
    let particles = total + raw.components as f64;
    if raw.components > 1 && particles > 1.0 {
        out[BARS - 1] = (2.0 * (raw.components as f64).ln() / particles.ln().max(f64::EPSILON)) as f32;
    }
    out
}

fn collect_edges(substrate: &Substrate, cutoff: f32) -> Vec<Edge> {
    let mut edges = Vec::new();
    for i in 0..substrate.traits.len() {
        let pos_i = substrate.pos(i);
        if !pos_i.iter().all(|p| p.is_finite()) { continue; }
        let mut consider = |j: usize| {
            if j <= i { return; } // each pair once; the grid offers both directions
            let pos_j = substrate.pos(j);
            if !pos_j.iter().all(|p| p.is_finite()) { return; }
            let distance = distance_sq(pos_i, pos_j, substrate, &mut []).sqrt();
            if distance <= cutoff { edges.push(Edge { length: distance, a: i as u32, b: j as u32 }); }
        };
        substrate.visit_neighbors(pos_i, &mut consider);
    }
    edges
}

/// Two decades below the cutoff, log-spaced. Deliberately NOT anchored to the mean spacing the
/// way the RDF is: every bar is one exact merge event, so there is no shot noise to dodge, and
/// edges far shorter than the spacing are precisely the signal, a tight cluster is a cluster
/// BECAUSE its internal edges are short. Relative to cutoff, so it ports across particle counts.
fn scale_range(cutoff: f32) -> (f32, f32) { (cutoff * 0.01, cutoff) }

fn bin_of(scale: f32, low: f32, high: f32) -> usize {
    if scale.is_nan() || scale <= low { return 0; }
    let t = (scale / low).ln() / (high / low).ln();
    ((t * BARS as f32) as usize).min(BARS - 1)
}

/// Union-find with path halving, union by size, so join is near-constant time and the result
/// does not depend on the order equal-length edges arrive in.
struct Union { parent: Vec<u32>, size: Vec<u32> }
impl Union {
    fn new(count: usize) -> Union { Union { parent: (0..count as u32).collect(), size: vec![1; count] } }
    fn find(&mut self, mut node: usize) -> usize {
        while self.parent[node] as usize != node {
            let grandparent = self.parent[self.parent[node] as usize]; // path halving
            self.parent[node] = grandparent; node = grandparent as usize;
        }
        node
    }
    /// True when the two were in different components, which is what makes this edge a bar.
    fn join(&mut self, a: usize, b: usize) -> bool {
        let (mut a, mut b) = (self.find(a), self.find(b));
        if a == b { return false; }
        if self.size[a] < self.size[b] { std::mem::swap(&mut a, &mut b); }
        self.parent[b] = a as u32; self.size[a] += self.size[b];
        true
    }
}
