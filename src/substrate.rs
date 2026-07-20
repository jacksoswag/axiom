//! The substrate seam (§3) — *where things live*.
//!
//! A grid *is* a graph with lattice adjacency; a graph carries arbitrary edges.
//! Both implement one `Substrate` trait, so a message-passing rule and the graph
//! analysis layer run over either. The fast Lenia path in `rule.rs` stays
//! specialized (§1: don't over-abstract the hot loop); this trait is the generic
//! CPU side used for graph-mode dynamics and for treating any substrate as a
//! graph to analyze.

use rayon::prelude::*;

pub trait Substrate: Sync {
    fn node_count(&self) -> usize;
    /// Neighbors of node `i` as `(index, weight)` pairs.
    fn neighbors(&self, i: usize) -> &[(usize, f32)];
    fn adjacency(&self) -> &[Vec<(usize, f32)>];
}

/// Explicit graph substrate: adjacency lists with edge weights.
pub struct GraphSubstrate {
    pub adj: Vec<Vec<(usize, f32)>>,
}

impl GraphSubstrate {
    pub fn new(adj: Vec<Vec<(usize, f32)>>) -> Self {
        GraphSubstrate { adj }
    }

    /// Watts–Strogatz-style small-world ring: `n` nodes each linked to `k`
    /// nearest neighbors on a ring, each edge rewired with probability `p`.
    /// Rewired long-range edges create hubs the PageRank observer surfaces.
    pub fn small_world(n: usize, k: usize, p: f32, seed: u64) -> Self {
        let mut rng = Xorshift::new(seed);
        let mut set: Vec<Vec<usize>> = vec![Vec::new(); n];
        let half = (k / 2).max(1);
        for i in 0..n {
            for j in 1..=half {
                let mut t = (i + j) % n;
                if rng.unit() < p {
                    // rewire to a random node (avoid self / duplicate)
                    let mut r = rng.below(n);
                    let mut tries = 0;
                    while (r == i || set[i].contains(&r)) && tries < 8 {
                        r = rng.below(n);
                        tries += 1;
                    }
                    t = r;
                }
                if t != i && !set[i].contains(&t) {
                    set[i].push(t);
                    set[t].push(i);
                }
            }
        }
        let adj = set
            .into_iter()
            .map(|nbrs| nbrs.into_iter().map(|j| (j, 1.0)).collect())
            .collect();
        GraphSubstrate { adj }
    }
}

impl Substrate for GraphSubstrate {
    fn node_count(&self) -> usize {
        self.adj.len()
    }
    fn neighbors(&self, i: usize) -> &[(usize, f32)] {
        &self.adj[i]
    }
    fn adjacency(&self) -> &[Vec<(usize, f32)>] {
        &self.adj
    }
}

/// A grid exposed as a graph (4-neighborhood). Realizes "a grid is a graph".
/// Built eagerly, so intended for the small grids used in graph-mode demos and
/// analysis, not the large fields the specialized Lenia path runs on.
pub struct GridSubstrate {
    adj: Vec<Vec<(usize, f32)>>,
}

impl GridSubstrate {
    pub fn new(h: usize, w: usize, torus: bool) -> Self {
        let mut adj = vec![Vec::with_capacity(4); h * w];
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                let push = |yy: i32, xx: i32, adj: &mut Vec<Vec<(usize, f32)>>| {
                    let (yy, xx) = if torus {
                        (yy.rem_euclid(h as i32), xx.rem_euclid(w as i32))
                    } else if yy < 0 || yy >= h as i32 || xx < 0 || xx >= w as i32 {
                        return;
                    } else {
                        (yy, xx)
                    };
                    adj[i].push((yy as usize * w + xx as usize, 1.0));
                };
                push(y as i32 - 1, x as i32, &mut adj);
                push(y as i32 + 1, x as i32, &mut adj);
                push(y as i32, x as i32 - 1, &mut adj);
                push(y as i32, x as i32 + 1, &mut adj);
            }
        }
        GridSubstrate { adj }
    }
}

impl Substrate for GridSubstrate {
    fn node_count(&self) -> usize {
        self.adj.len()
    }
    fn neighbors(&self, i: usize) -> &[(usize, f32)] {
        &self.adj[i]
    }
    fn adjacency(&self) -> &[Vec<(usize, f32)>] {
        &self.adj
    }
}

/// One message-passing step over any substrate: `U_i = Σ w·state / Σ w`, then
/// `out_i = update(state_i, U_i)`. A convolutional CA and a graph-NCA are the
/// same `aggregate(neighbors) → update` on different neighborhoods (§3).
pub fn mp_step<F>(sub: &(dyn Substrate + '_), state: &[f32], out: &mut [f32], update: F)
where
    F: Fn(f32, f32) -> f32 + Sync,
{
    out.par_iter_mut().enumerate().for_each(|(i, o)| {
        let mut acc = 0.0f32;
        let mut wsum = 0.0f32;
        for &(j, w) in sub.neighbors(i) {
            acc += state[j] * w;
            wsum += w;
        }
        let u = if wsum > 0.0 { acc / wsum } else { 0.0 };
        *o = update(state[i], u);
    });
}

/// Minimal deterministic RNG (xorshift64*), so graph builders are seed-reproducible
/// without pulling in an RNG crate.
pub struct Xorshift {
    s: u64,
}

impl Xorshift {
    pub fn new(seed: u64) -> Self {
        Xorshift { s: seed.max(1) ^ 0x9e3779b97f4a7c15 }
    }
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.s;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.s = x;
        x.wrapping_mul(0x2545f4914f6cdd1d)
    }
    #[inline]
    pub fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
    #[inline]
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}
