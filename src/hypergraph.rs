//! Hypergraph substrate (§2.1, §3) — edges generalized to n-ary relations.
//!
//! A hyperedge connects any number of nodes. This module carries two things the
//! design guide calls out as the "PageRank-hypergraph territory": a hypergraph
//! **CA** (nodes update from the aggregate of their incident hyperedges) and
//! **hypergraph PageRank** via the two-step random walk (node → incident
//! hyperedge → member node), power-iterated.

use crate::growth::{Growth, GrowthKind};
use crate::substrate::Xorshift;

pub struct Hypergraph {
    pub node_count: usize,
    /// Each hyperedge is a list of member node indices.
    pub edges: Vec<Vec<usize>>,
    /// Incidence: for each node, the hyperedges it belongs to.
    pub incident: Vec<Vec<usize>>,
}

impl Hypergraph {
    pub fn new(node_count: usize, edges: Vec<Vec<usize>>) -> Hypergraph {
        let mut incident = vec![Vec::new(); node_count];
        for (e, members) in edges.iter().enumerate() {
            for &m in members {
                incident[m].push(e);
            }
        }
        Hypergraph { node_count, edges, incident }
    }

    /// Random hypergraph: `m` hyperedges, each linking `k` distinct random nodes.
    pub fn random(n: usize, m: usize, k: usize, seed: u64) -> Hypergraph {
        let mut rng = Xorshift::new(seed);
        let edges = (0..m)
            .map(|_| {
                let mut members = Vec::with_capacity(k);
                while members.len() < k.min(n) {
                    let v = rng.below(n);
                    if !members.contains(&v) {
                        members.push(v);
                    }
                }
                members
            })
            .collect();
        Hypergraph::new(n, edges)
    }

    /// One hypergraph-CA step: each node relaxes toward a growth response of the
    /// mean state over the nodes it shares a hyperedge with. The same
    /// `aggregate → update` law, aggregated over n-ary relations.
    pub fn ca_step(&self, state: &[f32], out: &mut [f32], growth: &Growth, dt: f32) {
        for i in 0..self.node_count {
            let mut acc = 0.0f32;
            let mut cnt = 0.0f32;
            for &e in &self.incident[i] {
                for &m in &self.edges[e] {
                    if m != i {
                        acc += state[m];
                        cnt += 1.0;
                    }
                }
            }
            let u = if cnt > 0.0 { acc / cnt } else { 0.0 };
            out[i] = (state[i] + dt * growth.apply(u)).clamp(0.0, 1.0);
        }
    }

    /// Hypergraph PageRank via the two-step random walk: from a node, pick an
    /// incident hyperedge uniformly, then a member of that hyperedge uniformly.
    /// Power-iterated with damping.
    pub fn pagerank(&self, damping: f32, iters: usize) -> Vec<f32> {
        let n = self.node_count;
        if n == 0 {
            return Vec::new();
        }
        let inv_n = 1.0 / n as f32;
        let mut pr = vec![inv_n; n];
        for _ in 0..iters {
            let mut next = vec![(1.0 - damping) * inv_n; n];
            // Dangling nodes (no incident hyperedge) spread mass uniformly.
            let dangling: f32 = (0..n).filter(|&i| self.incident[i].is_empty()).map(|i| pr[i]).sum();
            let dshare = damping * dangling * inv_n;
            for v in next.iter_mut() {
                *v += dshare;
            }
            for u in 0..n {
                let inc = &self.incident[u];
                if inc.is_empty() {
                    continue;
                }
                let per_edge = damping * pr[u] / inc.len() as f32;
                for &e in inc {
                    let members = &self.edges[e];
                    // Walk to a member other than u (uniform); if the edge is a
                    // singleton, mass returns to u.
                    let others = members.iter().filter(|&&m| m != u).count();
                    if others == 0 {
                        next[u] += per_edge;
                    } else {
                        let share = per_edge / others as f32;
                        for &m in members {
                            if m != u {
                                next[m] += share;
                            }
                        }
                    }
                }
            }
            pr = next;
        }
        pr
    }
}

/// Run a hypergraph-CA and return the node×time spacetime plus PageRank + degree.
pub fn run_hyper_ca(n: usize, m: usize, k: usize, steps: usize, dt: f32, mu: f32, sigma: f32, seed: u64) -> HyperRun {
    let hg = Hypergraph::random(n, m, k, seed);
    let growth = Growth { kind: GrowthKind::Gauss, mu, sigma };
    let mut rng = Xorshift::new(seed ^ 0x55);
    let mut state: Vec<f32> = (0..n).map(|_| rng.unit()).collect();
    let mut next = vec![0.0f32; n];
    let mut spacetime = Vec::with_capacity(steps * n);
    for _ in 0..steps {
        spacetime.extend_from_slice(&state);
        hg.ca_step(&state, &mut next, &growth, dt);
        std::mem::swap(&mut state, &mut next);
    }
    let pr = hg.pagerank(0.85, 100);
    let degree: Vec<usize> = hg.incident.iter().map(|e| e.len()).collect();
    HyperRun { n, steps, spacetime, pagerank: pr, degree }
}

pub struct HyperRun {
    pub n: usize,
    pub steps: usize,
    pub spacetime: Vec<f32>,
    pub pagerank: Vec<f32>,
    pub degree: Vec<usize>,
}
