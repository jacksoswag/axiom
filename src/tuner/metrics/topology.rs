//! How many distinct bodies a sim holds, and at what scales they merge. Every bar is born at scale
//! 0 and dies where its component merges, and those death scales are exactly the edge weights of
//! the Euclidean MST, so the whole cluster hierarchy costs one spanning tree. Edges stop at the
//! kernel cutoff, deliberately: clumps no kernel can reach across are not one structure at any scale.

use crate::engine::kernel::distance_sq;
use crate::engine::substrate::Substrate;
use super::{alive, log_bin, Blocks, Metric, Spec};

/// Bins in the log-spaced death-scale histogram
const BARS: usize = 8;
pub const METRIC: Metric = &Spec { graph: true, ..Spec::of("topology", BARS, measure) };

pub fn measure(base: &Blocks, _: &[f32]) -> Vec<f32> {
    mass(base.barcode()).into_iter().map(|value| value.clamp(0.0, 1.0)).collect()
}

pub struct Barcode {
    pub bins: Vec<f64>, // H0 death-scale counts, log-binned up to cutoff
    pub components: usize, // still separate at cutoff, at least 1 for any non-empty swarm
}

/// One edge of the neighbor graph, kept only long enough to sort. Squared, because sorting on the
/// square agrees with sorting on the length, and only the edges that go on to join two components
/// (at most one per particle, out of every candidate pair) ever need the real distance.
struct Edge { length_sq: f32, a: u32, b: u32 }

/// Owns its grid rebuild, so a caller only hands over the substrate.
pub fn h0(substrate: &mut Substrate, cutoff: f32) -> Barcode {
    let count = substrate.traits.len();
    let empty = Barcode { bins: vec![0.0; BARS], components: 0 };
    if count == 0 || !cutoff.is_finite() || cutoff <= 0.0 { return empty; }
    substrate.rebuild_grid(cutoff);

    let mut edges = collect_edges(substrate, cutoff);
    // Ties broken by index so the tree, and therefore the barcode, is reproducible. Float
    // comparison alone would leave equal-length edges in whatever order the scan produced.
    edges.sort_unstable_by(|x, y| x.length_sq.total_cmp(&y.length_sq).then(x.a.cmp(&y.a)).then(x.b.cmp(&y.b)));

    let mut union = Union::new(count);
    let mut barcode = Barcode { bins: vec![0.0; BARS], components: count };
    for edge in edges {
        if union.join(edge.a as usize, edge.b as usize) { // first join at this scale = one bar
            barcode.bins[log_bin(edge.length_sq.sqrt(), cutoff * 0.01, cutoff, BARS)] += 1.0;
            barcode.components -= 1;
        }
    }
    barcode
}

/// Seven death-scale mass shares plus one log-scaled component mass. A share of the total needs no
/// uniform baseline, so it cannot divide by a bin the baseline left empty. 0.5 in the first seven
/// axes means a flat profile; the eighth is 0 for one cutoff-connected component, rising for dust.
fn mass(raw: &Barcode) -> Vec<f32> {
    let total: f64 = raw.bins.iter().sum();
    let mut out = vec![0.0; BARS];
    if total > 0.0 {
        for (index, value) in raw.bins.iter().enumerate() { // last two raw bins fold together
            out[index.min(BARS - 2)] += (value / total * (BARS - 1) as f64 * 0.5) as f32;
        }
    }
    let particles = total + raw.components as f64;
    if raw.components > 1 && particles > 1.0 {
        out[BARS - 1] = ((raw.components as f64).ln() / particles.ln().max(f64::EPSILON)) as f32;
    }
    out
}

fn collect_edges(substrate: &Substrate, cutoff: f32) -> Vec<Edge> {
    let count = substrate.traits.len();
    let dims = substrate.dimensions;
    // Liveness once per particle rather than once per candidate pair, which is the same question
    // asked a few dozen times over for every particle in reach.
    let living: Vec<bool> = substrate.positions.chunks_exact(dims).map(alive).collect();
    let cutoff_sq = cutoff * cutoff; // the pair test squared, so a candidate costs no square root
    let mut edges = Vec::with_capacity(count * 4); // a few neighbors each, past the first few doublings
    for i in 0..count {
        if !living[i] { continue; }
        let pos_i = substrate.pos(i);
        let mut consider = |j: usize| {
            if j <= i || !living[j] { return; } // each pair once; the grid offers both directions
            let length_sq = distance_sq(pos_i, substrate.pos(j), substrate, &mut []);
            if length_sq <= cutoff_sq { edges.push(Edge { length_sq, a: i as u32, b: j as u32 }); }
        };
        substrate.visit_neighbors(pos_i, &mut consider);
    }
    edges
}

/// Union-find with path halving, union by size, so join is near-constant time and the result does
/// not depend on the order equal-length edges arrive in.
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
    /// True when the two were in different components, which is what makes this edge a bar
    fn join(&mut self, a: usize, b: usize) -> bool {
        let (mut a, mut b) = (self.find(a), self.find(b));
        if a == b { return false; }
        if self.size[a] < self.size[b] { std::mem::swap(&mut a, &mut b); }
        self.parent[b] = a as u32; self.size[a] += self.size[b];
        true
    }
}
