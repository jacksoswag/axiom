//! Graph-mode CA demo (§3, §7.6) — the seam as a runnable path.
//!
//! The *same* `aggregate(neighbors) → update` law that drives grid Lenia runs
//! here over a small-world graph substrate. Output is a node×time spacetime
//! diagram plus a PageRank ranking of the nodes, showing one rule + graph
//! substrate + graph analysis composed as a single experiment rather than three.

use crate::analysis::pagerank;
use crate::growth::{Growth, GrowthKind};
use crate::substrate::{mp_step, GraphSubstrate, Substrate, Xorshift};

pub struct GraphRun {
    pub node_count: usize,
    pub steps: usize,
    /// Row-major `steps × node_count` spacetime matrix of node states.
    pub spacetime: Vec<f32>,
    pub pagerank: Vec<f32>,
    pub degree: Vec<usize>,
}

/// Run a graph-Lenia: node state relaxes toward a growth response of its
/// neighborhood mean. A small-world topology gives hubs the PageRank surfaces.
pub fn run_graph_lenia(
    n: usize,
    k: usize,
    rewire: f32,
    steps: usize,
    dt: f32,
    growth_mu: f32,
    growth_sigma: f32,
    seed: u64,
) -> GraphRun {
    let sub = GraphSubstrate::small_world(n, k, rewire, seed);
    let growth = Growth { kind: GrowthKind::Gauss, mu: growth_mu, sigma: growth_sigma };

    let mut rng = Xorshift::new(seed ^ 0xabcd);
    let mut state: Vec<f32> = (0..n).map(|_| rng.unit()).collect();
    let mut next = vec![0.0f32; n];

    let mut spacetime = Vec::with_capacity(steps * n);
    for _ in 0..steps {
        spacetime.extend_from_slice(&state);
        mp_step(&sub, &state, &mut next, |s, u| (s + dt * growth.apply(u)).clamp(0.0, 1.0));
        std::mem::swap(&mut state, &mut next);
    }

    let pr = pagerank(sub.adjacency(), 0.85, 100);
    let degree: Vec<usize> = (0..n).map(|i| sub.neighbors(i).len()).collect();

    GraphRun { node_count: n, steps, spacetime, pagerank: pr, degree }
}
