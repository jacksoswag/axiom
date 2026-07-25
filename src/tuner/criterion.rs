//! How good is this genome: one number, from one function, chosen per search. A criterion ranks and
//! nothing else, and names the metrics it reads so a run measures them. Whether a genome is worth
//! ranking at all is the gate's job, so nothing here reads a threshold.

pub mod novelty;
pub mod structure;

use crate::tuner::metrics::Metric;
use crate::tuner::specimen::Specimen;

/// Which scoring function a search uses. A new criterion is a module and a variant, and both
/// matches below stop compiling until it is wired in
pub enum Criterion {
    /// Axes novelty spreads genomes across. Not a dependency list: novelty names no metric, it just
    /// wants a space to be far apart in. Gates widen that space too, since distance reads the whole plan.
    Novelty(Vec<Metric>),
    Structure,
}
impl Criterion {
    /// What scoring needs measured. A plan is this plus whatever the gates read, so nothing here
    /// has to anticipate a gate
    pub fn metrics(&self) -> &[Metric] {
        match self {
            Criterion::Novelty(axes) => axes,
            Criterion::Structure => &structure::METRICS,
        }
    }
    /// Score one genome against a population: population-relative for novelty, absolute for
    /// structure. Re-scoring a member passes a population holding it, biasing everyone alike
    pub fn score(&self, values: &[f32], plan: &[Metric], population: &[Specimen]) -> f32 {
        match self {
            Criterion::Novelty(_) => novelty::score(values, population),
            Criterion::Structure => structure::score(values, plan),
        }
    }
}
