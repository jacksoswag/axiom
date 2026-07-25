//! Checks if genome simulation passes basic quality checks before evaluating and scoring.
//! Also prevent some algorithms accidentally optimizing for chaos.

use crate::tuner::metrics::{scalar, Metric, Reduce};

/// Valid range on one metric, read in that metric's own descriptor units.
pub struct Gate {
    pub metric: Metric,
    pub floor: f32,
    pub ceiling: f32,
}

/// If a genome clears every gate. The plan comes along because a descriptor carries no layout of
/// its own. A partial reading is one taken mid-sim, which skips the gates whose metric only lands
/// on the final sample. A dead sim reads 0 everywhere, so any gate with a floor rejects it.
pub fn valid(values: &[f32], plan: &[Metric], gates: &[Gate], partial: bool) -> bool {
    gates.iter().all(|gate| {
        if partial && gate.metric.reduce == Reduce::Once { return true; }
        let value = scalar(values, plan, gate.metric);
        value >= gate.floor && value <= gate.ceiling
    })
}
