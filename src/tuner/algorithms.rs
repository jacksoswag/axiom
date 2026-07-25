//! Routes which tuning algorithm runs. Fully configurable

pub mod evolve;

use crate::tuner::driver::{Search, Tally};
use crate::tuner::specimen::Specimen;
use crate::util::Rng;
use evolve::Evolve;

/// ponytail: one variant, so this match and the word catalog feeds a frontend both answer constantly.
/// Kept deliberately as the seam a second algorithm lands on. The ceiling is that catalog's algorithm
/// list is a hardcoded single entry sitting apart from this enum, so adding one is two edits and
/// nothing makes them move together. Collapse this to a bare Evolve on Search if the second never comes.
pub enum Algorithm { Evolve(Evolve) }
impl Algorithm {
    /// Search to the algorithm's own budget and hand back the population it kept, unranked. Every
    /// algorithm reports each round to watch and stops early when it answers false.
    pub fn explore(&self, search: &Search, rng: &mut Rng,
        watch: &mut impl FnMut(usize, &[Specimen], &Tally) -> bool) -> Vec<Specimen>
    {
        match self {
            Algorithm::Evolve(evolve) => evolve.explore(search, rng, watch),
        }
    }
}
