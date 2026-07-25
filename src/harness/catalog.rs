//! What a frontend may choose from, handed over as data so no list of metrics, criteria, or knobs is
//! written twice. The word() matches are exhaustive on purpose: a new variant stops the build until it
//! is named here, which is the same pressure criterion.rs already relies on.

use crate::engine::params::FixedGenome;
use crate::engine::resolve::COORDINATION_BOUNDS;
use crate::harness::protocol::{quote, quoted};
use crate::tuner::algorithms::Algorithm;
use crate::tuner::criterion::Criterion;
use crate::tuner::metrics::{field, Metric, Reduce, ALL};

/// One metric by the key its own file declares, so a criterion axis and a gate arrive from a frontend
/// as text and land on the same Spec a search would have used.
pub fn metric_of(key: &str) -> Option<Metric> { ALL.into_iter().find(|metric| metric.key == key) }

/// Every shape field a control can move: name, default, and the range offered. The shape parser reads
/// this table too, so a field cannot appear in a frontend and be ignored on the way back in. Seed is
/// absent deliberately: it is a whole number too large for this table's f32 defaults.
pub const SHAPE_FIELDS: [(&str, f32, f32, f32); 6] = [
    ("particles", 320.0, 64.0, 1000000.0),
    ("anchors", 3.0, 2.0, 8.0),
    ("shells", 2.0, 1.0, 4.0),
    ("bumps", 1.0, 1.0, 3.0),
    ("radius", 2.0, 0.5, 20.0),
    ("dt", 0.05, 0.001, 0.5),
];
/// Every evolve knob on the same terms. Zero both lane shares and the search collapses to random
/// draws, which is the baseline a real setting has to beat, so both ranges reach 0.
pub const EVOLVE_KNOBS: [(&str, f32, f32, f32); 7] = [
    ("generations", 20.0, 1.0, 2000.0),
    ("batch", 48.0, 1.0, 512.0),
    ("capacity", 96.0, 1.0, 512.0),
    ("tournament", 0.5, 0.0, 1.0),
    ("expedition", 0.25, 0.0, 1.0),
    ("iso", 0.05, 0.0, 0.5),
    ("line", 0.3, 0.0, 1.0),
];
/// What a table says a missing key means. A frontend may send any subset of a config, so every knob
/// needs an answer without one.
pub fn default_of(table: &[(&str, f32, f32, f32)], key: &str) -> f32 {
    table.iter().find(|(name, ..)| *name == key).map(|(_, default, ..)| *default).unwrap_or(0.0)
}

/// Which criteria exist and whether each takes an axis list. Beside criterion_word so the two move in
/// one edit; novelty wants a space to be far apart in, structure reads its own metrics.
const CRITERIA: [(&str, bool); 2] = [("novelty", true), ("structure", false)];

pub fn criterion_word(criterion: &Criterion) -> &'static str {
    match criterion {
        Criterion::Novelty(_) => "novelty",
        Criterion::Structure => "structure",
    }
}
pub fn algorithm_word(algorithm: &Algorithm) -> &'static str {
    match algorithm {
        Algorithm::Evolve(_) => "evolve",
    }
}
fn reduce_word(reduce: Reduce) -> &'static str {
    match reduce {
        Reduce::Mean => "mean",
        Reduce::Last => "last",
        Reduce::Once => "once",
    }
}

/// Everything a frontend needs to build its own controls: what can be measured and what that costs,
/// what can score, what can search, and the shape fields underneath all of it.
pub fn body() -> String {
    let metrics: Vec<String> = ALL.iter().map(|metric| {
        let depends: Vec<&str> = metric.depends.iter().map(|read| read.key).collect();
        let grids: Vec<String> = metric.sides.iter().map(|side| side.to_string()).collect();
        format!("{{\"key\":{},\"width\":{},\"reduce\":\"{}\",\"depends\":{},\"grids\":[{}],\"pairs\":{},\"graph\":{},\"motion\":{}}}",
            quote(metric.key), metric.width, reduce_word(metric.reduce), quoted(&depends),
            grids.join(","), metric.pairs, metric.graph, metric.motion)
    }).collect();
    let criteria: Vec<String> = CRITERIA.iter()
        .map(|(name, axes)| format!("{{\"name\":{},\"axes\":{axes}}}", quote(name))).collect();
    format!("\"protocol\":1,\"dimensions\":{},\"metrics\":[{}],\"criteria\":[{}],\"algorithms\":[{{\"name\":\"evolve\",\"knobs\":{}}}],\"shape\":{}",
        field::GRID_AXES, metrics.join(","), criteria.join(","), knobs(&EVOLVE_KNOBS), knobs(&SHAPE_FIELDS))
}
fn knobs(table: &[(&str, f32, f32, f32)]) -> String {
    let entries: Vec<String> = table.iter().map(|(name, default, low, high)| {
        format!("{{\"name\":{},\"default\":{default},\"low\":{low},\"high\":{high}}}", quote(name))
    }).collect();
    format!("[{}]", entries.join(","))
}

/// Every gene a frontend can move: what it means, what to call it, how far it may travel at the box
/// this genome resolved to, and where it stands now. Bounds come in already narrowed to that box,
/// since a pair gene's ceiling follows the box and a stale slider would offer reach the sim rejects.
pub fn layout(shape: &FixedGenome, bounds: &[(f32, f32)], genes: &[f32], box_len: f32) -> String {
    let slots = slot_names(shape);
    let entries: Vec<String> = bounds.iter().enumerate().map(|(index, &(low, high))| {
        let (kind, label) = if index == 0 { ("coordination", "neighbors sensed".to_string()) }
            else if index <= shape.anchor_count { ("logit", format!("anchor {} share", index - 1)) }
            else {
                let pair = (index - 1 - shape.anchor_count) / shape.pair_stride();
                let slot = (index - 1 - shape.anchor_count) % shape.pair_stride();
                return format!("{{\"index\":{index},\"kind\":\"pair\",\"source\":{},\"destination\":{},\"label\":{},\"low\":{low},\"high\":{high},\"value\":{}}}",
                    pair / shape.anchor_count, pair % shape.anchor_count, quote(&slots[slot]),
                    genes.get(index).copied().unwrap_or(low));
            };
        format!("{{\"index\":{index},\"kind\":\"{kind}\",\"label\":{},\"low\":{low},\"high\":{high},\"value\":{}}}",
            quote(&label), genes.get(index).copied().unwrap_or(low))
    }).collect();
    format!("\"box_len\":{box_len:.4},\"coordination\":[{},{}],\"genes\":[{}]",
        COORDINATION_BOUNDS.0, COORDINATION_BOUNDS.1, entries.join(","))
}

/// The name of every slot in one pair block, in the order Matrix::derive reads them
fn slot_names(shape: &FixedGenome) -> Vec<String> {
    let mut names = Vec::with_capacity(shape.pair_stride());
    for shell in 0..shape.shells {
        for part in ["amp", "peak", "width"] { names.push(format!("shell {shell} {part}")); }
    }
    for bump in 0..shape.bumps {
        for part in ["amp", "peak", "width"] { names.push(format!("bump {bump} {part}")); }
    }
    names.push("weight".to_string());
    names
}
