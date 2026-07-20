//! The analysis / observability layer (§2.10, §5.5) — the instrument's payload.
//!
//! Most CA tools render; few *measure*. Observers run over live state and emit
//! named scalars and structured records: physical descriptors, connected-component
//! detection with persistent geometry, and PageRank centrality over the emergent
//! organism interaction graph — the graph×CA seam expressed as observation.

use crate::config::AnalysisConfig;
use crate::field::Field;
use std::collections::BTreeMap;

/// A named bag of scalar metrics from one observation.
#[derive(Debug, Clone, Default)]
pub struct Record {
    pub scalars: BTreeMap<String, f64>,
}

impl Record {
    pub fn put(&mut self, k: &str, v: f64) {
        self.scalars.insert(k.to_string(), v);
    }
}

pub trait Observer: Send + Sync {
    fn observe(&self, field: &Field, torus: bool) -> Record;
    fn name(&self) -> &'static str;
}

pub fn build_observers(cfgs: &[AnalysisConfig]) -> Vec<Box<dyn Observer>> {
    cfgs.iter()
        .map(|c| match c {
            AnalysisConfig::Descriptors => {
                Box::new(Descriptors) as Box<dyn Observer>
            }
            AnalysisConfig::Detect { threshold } => {
                Box::new(Detector { threshold: *threshold })
            }
            AnalysisConfig::PageRank { threshold, link_radius } => {
                Box::new(PageRankObserver { threshold: *threshold, link_radius: *link_radius })
            }
        })
        .collect()
}

// --- Physical descriptors -----------------------------------------------------

pub struct Descriptors;

/// Circular-mean centroid of a channel on a torus. Handles wrap-around, so a
/// glider crossing the boundary still reports a continuous position.
pub fn circular_centroid(field: &Field, ch: usize) -> (f64, f64) {
    use std::f64::consts::TAU;
    let (h, w) = (field.h, field.w);
    let (mut sx, mut cx, mut sy, mut cy, mut m) = (0.0, 0.0, 0.0, 0.0, 0.0f64);
    for y in 0..h {
        for x in 0..w {
            let v = field.get(ch, y, x) as f64;
            if v <= 0.0 {
                continue;
            }
            let ax = TAU * x as f64 / w as f64;
            let ay = TAU * y as f64 / h as f64;
            cx += v * ax.cos();
            sx += v * ax.sin();
            cy += v * ay.cos();
            sy += v * ay.sin();
            m += v;
        }
    }
    if m == 0.0 {
        return (0.0, 0.0);
    }
    let px = (sx.atan2(cx).rem_euclid(TAU)) / TAU * w as f64;
    let py = (sy.atan2(cy).rem_euclid(TAU)) / TAU * h as f64;
    (px, py)
}

impl Observer for Descriptors {
    fn observe(&self, field: &Field, _torus: bool) -> Record {
        let mut r = Record::default();
        for ch in 0..field.c {
            let mass = field.mass(ch);
            r.put(&format!("mass_c{ch}"), mass);
            let (cx, cy) = circular_centroid(field, ch);
            r.put(&format!("centroid_x_c{ch}"), cx);
            r.put(&format!("centroid_y_c{ch}"), cy);
        }
        r
    }
    fn name(&self) -> &'static str {
        "descriptors"
    }
}

// --- Connected-component detection --------------------------------------------

#[derive(Debug, Clone)]
pub struct Component {
    pub id: usize,
    pub size: usize,
    pub mass: f64,
    pub cx: f64,
    pub cy: f64,
    pub bbox: (usize, usize, usize, usize), // ymin, xmin, ymax, xmax
}

/// Label 4-connected components of `channel(ch)` above `threshold`.
/// Boundaries are treated as bounded here (organisms are local), which is fine
/// for detection even when the dynamics themselves wrap.
pub fn connected_components(field: &Field, ch: usize, threshold: f32) -> Vec<Component> {
    let (h, w) = (field.h, field.w);
    let data = field.channel(ch);
    let mut label = vec![0usize; h * w]; // 0 = unlabeled
    let mut comps: Vec<Component> = Vec::new();
    let mut stack: Vec<usize> = Vec::new();
    for start in 0..h * w {
        if data[start] < threshold || label[start] != 0 {
            continue;
        }
        let id = comps.len() + 1;
        stack.push(start);
        label[start] = id;
        let (mut size, mut mass, mut mx, mut my) = (0usize, 0.0f64, 0.0f64, 0.0f64);
        let (mut ymin, mut xmin, mut ymax, mut xmax) = (h, w, 0usize, 0usize);
        while let Some(p) = stack.pop() {
            let (y, x) = (p / w, p % w);
            let v = data[p] as f64;
            size += 1;
            mass += v;
            mx += v * x as f64;
            my += v * y as f64;
            ymin = ymin.min(y);
            xmin = xmin.min(x);
            ymax = ymax.max(y);
            xmax = xmax.max(x);
            let visit = |ny: i32, nx: i32, stack: &mut Vec<usize>, label: &mut Vec<usize>| {
                if ny < 0 || ny >= h as i32 || nx < 0 || nx >= w as i32 {
                    return;
                }
                let q = ny as usize * w + nx as usize;
                if label[q] == 0 && data[q] >= threshold {
                    label[q] = id;
                    stack.push(q);
                }
            };
            visit(y as i32 - 1, x as i32, &mut stack, &mut label);
            visit(y as i32 + 1, x as i32, &mut stack, &mut label);
            visit(y as i32, x as i32 - 1, &mut stack, &mut label);
            visit(y as i32, x as i32 + 1, &mut stack, &mut label);
        }
        let inv = if mass > 0.0 { 1.0 / mass } else { 0.0 };
        comps.push(Component {
            id,
            size,
            mass,
            cx: mx * inv,
            cy: my * inv,
            bbox: (ymin, xmin, ymax, xmax),
        });
    }
    comps
}

pub struct Detector {
    pub threshold: f32,
}

impl Observer for Detector {
    fn observe(&self, field: &Field, _torus: bool) -> Record {
        let comps = connected_components(field, 0, self.threshold);
        let mut r = Record::default();
        r.put("components", comps.len() as f64);
        r.put("largest_mass", comps.iter().map(|c| c.mass).fold(0.0, f64::max));
        r
    }
    fn name(&self) -> &'static str {
        "detect"
    }
}

// --- PageRank over the organism interaction graph -----------------------------

/// Power-iteration PageRank over a weighted, possibly-dangling graph. Exact
/// enough for centrality ranking; the design guide's "PageRank as an observable
/// over an evolving swarm" as a standard observer (§3, §2.10).
pub fn pagerank(adj: &[Vec<(usize, f32)>], damping: f32, iters: usize) -> Vec<f32> {
    let n = adj.len();
    if n == 0 {
        return Vec::new();
    }
    let inv_n = 1.0 / n as f32;
    let out_w: Vec<f32> = adj.iter().map(|nb| nb.iter().map(|(_, w)| *w).sum()).collect();
    let mut pr = vec![inv_n; n];
    for _ in 0..iters {
        let mut next = vec![(1.0 - damping) * inv_n; n];
        // Redistribute mass held by dangling nodes uniformly.
        let dangling: f32 = (0..n).filter(|&i| out_w[i] == 0.0).map(|i| pr[i]).sum();
        let dshare = damping * dangling * inv_n;
        for v in next.iter_mut() {
            *v += dshare;
        }
        for i in 0..n {
            if out_w[i] > 0.0 {
                let give = damping * pr[i] / out_w[i];
                for &(j, w) in &adj[i] {
                    next[j] += give * w;
                }
            }
        }
        pr = next;
    }
    pr
}

/// Build an undirected interaction graph over detected organisms: an edge for
/// every pair within `link_radius`, weighted by inverse distance.
pub fn interaction_graph(comps: &[Component], link_radius: f32) -> Vec<Vec<(usize, f32)>> {
    let n = comps.len();
    let mut adj = vec![Vec::new(); n];
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = comps[i].cx - comps[j].cx;
            let dy = comps[i].cy - comps[j].cy;
            let d = ((dx * dx + dy * dy) as f32).sqrt();
            if d <= link_radius && d > 0.0 {
                let wgt = 1.0 / d;
                adj[i].push((j, wgt));
                adj[j].push((i, wgt));
            }
        }
    }
    adj
}

pub struct PageRankObserver {
    pub threshold: f32,
    pub link_radius: f32,
}

/// Full detection → interaction-graph → PageRank pass, returning ranked organisms.
pub fn organism_ranking(
    field: &Field,
    threshold: f32,
    link_radius: f32,
) -> (Vec<Component>, Vec<f32>) {
    let comps = connected_components(field, 0, threshold);
    let adj = interaction_graph(&comps, link_radius);
    let pr = pagerank(&adj, 0.85, 100);
    (comps, pr)
}

impl Observer for PageRankObserver {
    fn observe(&self, field: &Field, _torus: bool) -> Record {
        let (comps, pr) = organism_ranking(field, self.threshold, self.link_radius);
        let mut r = Record::default();
        r.put("organisms", comps.len() as f64);
        let top = pr.iter().cloned().fold(0.0f32, f32::max);
        r.put("max_pagerank", top as f64);
        r
    }
    fn name(&self) -> &'static str {
        "pagerank"
    }
}
