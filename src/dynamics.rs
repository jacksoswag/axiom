//! Dynamical, spectral, and topological descriptors, organism tracking, and the
//! experiment log (§2.10, §5.5) — the deeper half of the instrument layer.
//!
//! All dependency-free: a hand DFT for periodicity, histogram entropy, an H0
//! persistent-homology computation via superlevel-set union-find, nearest-centroid
//! organism tracking across frames, and an append-only experiment log with
//! in-Rust nearest-neighbor similarity search.

use crate::analysis::Component;
use crate::field::Field;
use std::io::Write;
use std::path::Path;

// --- Spectral -----------------------------------------------------------------

/// Dominant oscillation period of a time-series via a direct DFT (detrended).
/// Returns `(period_in_steps, periodicity_strength)` where strength is the peak
/// power as a fraction of total spectral power.
pub fn dominant_period(series: &[f32]) -> (f32, f32) {
    let t = series.len();
    if t < 4 {
        return (0.0, 0.0);
    }
    let mean = series.iter().sum::<f32>() / t as f32;
    let tau = std::f32::consts::TAU;
    let mut best_k = 1usize;
    let mut best_p = 0.0f32;
    let mut total = 0.0f32;
    for k in 1..=t / 2 {
        let (mut re, mut im) = (0.0f32, 0.0f32);
        for (n, &s) in series.iter().enumerate() {
            let ang = tau * k as f32 * n as f32 / t as f32;
            let v = s - mean;
            re += v * ang.cos();
            im -= v * ang.sin();
        }
        let p = re * re + im * im;
        total += p;
        if p > best_p {
            best_p = p;
            best_k = k;
        }
    }
    let strength = if total > 0.0 { best_p / total } else { 0.0 };
    (t as f32 / best_k as f32, strength)
}

// --- Dynamical ----------------------------------------------------------------

/// Shannon entropy (bits) of a channel's value histogram.
pub fn value_entropy(field: &Field, ch: usize, bins: usize) -> f32 {
    let mut hist = vec![0u32; bins];
    let data = field.channel(ch);
    for &v in data {
        let b = ((v.clamp(0.0, 1.0) * bins as f32) as usize).min(bins - 1);
        hist[b] += 1;
    }
    let total = data.len() as f32;
    let mut h = 0.0f32;
    for &c in &hist {
        if c > 0 {
            let p = c as f32 / total;
            h -= p * p.log2();
        }
    }
    h
}

/// Mean absolute per-cell change between two frames (activity / instability).
pub fn activity(prev: &Field, cur: &Field) -> f32 {
    prev.data.iter().zip(cur.data.iter()).map(|(a, b)| (a - b).abs()).sum::<f32>() / prev.data.len() as f32
}

// --- H0 persistent homology ---------------------------------------------------

/// H0 persistence of a channel via a superlevel-set filtration. Cells are added
/// in descending value; a component is born at a local peak and dies at the value
/// where it merges into an older one. Returns `(birth, death)` pairs, most
/// persistent first. The number of high-persistence pairs is a robust count of
/// distinct structures independent of a single threshold.
pub fn h0_persistence(field: &Field, ch: usize) -> Vec<(f32, f32)> {
    let (h, w) = (field.h, field.w);
    let data = field.channel(ch);
    let n = h * w;
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| data[b].partial_cmp(&data[a]).unwrap());

    let mut parent: Vec<usize> = (0..n).collect();
    let mut peak = vec![0.0f32; n]; // birth value of a component's root
    let mut added = vec![false; n];
    let mut pairs: Vec<(f32, f32)> = Vec::new();

    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }

    for &i in &order {
        added[i] = true;
        peak[i] = data[i];
        let (y, x) = (i / w, i % w);
        let neigh = [
            (y as i32 - 1, x as i32),
            (y as i32 + 1, x as i32),
            (y as i32, x as i32 - 1),
            (y as i32, x as i32 + 1),
        ];
        for (ny, nx) in neigh {
            if ny < 0 || ny >= h as i32 || nx < 0 || nx >= w as i32 {
                continue;
            }
            let j = ny as usize * w + nx as usize;
            if !added[j] {
                continue;
            }
            let (ri, rj) = (find(&mut parent, i), find(&mut parent, j));
            if ri == rj {
                continue;
            }
            // Younger (lower peak) component dies at the current cell value.
            let (older, younger) = if peak[ri] >= peak[rj] { (ri, rj) } else { (rj, ri) };
            pairs.push((peak[younger], data[i]));
            parent[younger] = older;
        }
    }
    // The single surviving component never dies; record it against the global min.
    let gmin = data.iter().cloned().fold(f32::MAX, f32::min);
    let root = find(&mut parent, order[0]);
    pairs.push((peak[root], gmin));
    pairs.sort_by(|a, b| (b.0 - b.1).partial_cmp(&(a.0 - a.1)).unwrap());
    pairs
}

/// Count H0 features whose persistence exceeds `min_persistence`.
pub fn count_persistent(pairs: &[(f32, f32)], min_persistence: f32) -> usize {
    pairs.iter().filter(|(b, d)| b - d >= min_persistence).count()
}

// --- Organism tracking → phylogeny --------------------------------------------

pub struct TrackEvent {
    pub births: usize,
    pub deaths: usize,
    pub matched: usize,
}

/// Assigns persistent IDs to detected organisms across frames by nearest-centroid
/// matching, so lifetimes and a lineage can be read off a run.
pub struct Tracker {
    next_id: usize,
    prev: Vec<(usize, f64, f64)>, // (id, cx, cy)
    max_dist: f64,
    pub max_live_id: usize,
}

impl Tracker {
    pub fn new(max_dist: f64) -> Tracker {
        Tracker { next_id: 0, prev: Vec::new(), max_dist, max_live_id: 0 }
    }

    /// Match `comps` to the previous frame; returns the id per component and the
    /// birth/death/match counts for this step.
    pub fn update(&mut self, comps: &[Component]) -> (Vec<usize>, TrackEvent) {
        let mut ids = vec![usize::MAX; comps.len()];
        let mut prev_used = vec![false; self.prev.len()];
        let mut matched = 0;
        // Greedy nearest match.
        for (ci, c) in comps.iter().enumerate() {
            let mut best = None;
            let mut best_d = self.max_dist;
            for (pi, &(_, px, py)) in self.prev.iter().enumerate() {
                if prev_used[pi] {
                    continue;
                }
                let d = ((c.cx - px).powi(2) + (c.cy - py).powi(2)).sqrt();
                if d < best_d {
                    best_d = d;
                    best = Some(pi);
                }
            }
            if let Some(pi) = best {
                prev_used[pi] = true;
                ids[ci] = self.prev[pi].0;
                matched += 1;
            }
        }
        let deaths = prev_used.iter().filter(|u| !**u).count();
        let mut births = 0;
        for (ci, id) in ids.iter_mut().enumerate() {
            if *id == usize::MAX {
                *id = self.next_id;
                self.next_id += 1;
                births += 1;
            }
            let _ = ci;
        }
        self.max_live_id = self.next_id;
        self.prev = comps.iter().zip(ids.iter()).map(|(c, &id)| (id, c.cx, c.cy)).collect();
        (ids, TrackEvent { births, deaths, matched })
    }

    pub fn total_lineages(&self) -> usize {
        self.next_id
    }
}

// --- Experiment log + similarity ----------------------------------------------

/// One logged run: a provenance hash, a name, and a descriptor vector.
pub struct Experiment {
    pub hash: u64,
    pub name: String,
    pub descriptor: Vec<f32>,
}

pub fn log_experiment(path: &Path, exp: &Experiment) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
    let d: Vec<String> = exp.descriptor.iter().map(|v| format!("{v}")).collect();
    writeln!(f, "{{\"hash\":\"{:016x}\",\"name\":\"{}\",\"descriptor\":[{}]}}", exp.hash, exp.name, d.join(","))?;
    Ok(())
}

pub fn load_experiments(path: &Path) -> Vec<Experiment> {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(_) => return Vec::new(),
    };
    text.lines().filter_map(parse_experiment).collect()
}

fn parse_experiment(line: &str) -> Option<Experiment> {
    let name = between(line, "\"name\":\"", "\"")?.to_string();
    let hash = u64::from_str_radix(between(line, "\"hash\":\"", "\"")?, 16).ok()?;
    let arr = between(line, "\"descriptor\":[", "]")?;
    let descriptor = arr.split(',').filter_map(|s| s.trim().parse().ok()).collect();
    Some(Experiment { hash, name, descriptor })
}

fn between<'a>(s: &'a str, start: &str, end: &str) -> Option<&'a str> {
    let a = s.find(start)? + start.len();
    let rest = &s[a..];
    let b = rest.find(end)?;
    Some(&rest[..b])
}

/// Nearest logged runs to `query` by normalized Euclidean descriptor distance.
pub fn find_similar(experiments: &[Experiment], query: &[f32], k: usize) -> Vec<(f32, usize)> {
    // Per-dimension scale from the corpus, so no single descriptor dominates.
    let dims = query.len();
    let mut scale = vec![1e-6f32; dims];
    for e in experiments {
        for (i, &v) in e.descriptor.iter().enumerate().take(dims) {
            scale[i] = scale[i].max(v.abs());
        }
    }
    let mut scored: Vec<(f32, usize)> = experiments
        .iter()
        .enumerate()
        .map(|(idx, e)| {
            let mut d = 0.0f32;
            for i in 0..dims.min(e.descriptor.len()) {
                let diff = (query[i] - e.descriptor[i]) / scale[i];
                d += diff * diff;
            }
            (d.sqrt(), idx)
        })
        .collect();
    scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    scored.truncate(k);
    scored
}
