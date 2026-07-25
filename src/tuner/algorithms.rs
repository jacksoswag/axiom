//! Routes which tuning algorithm runs. Fully configurable

pub mod evolve;

use crate::tuner::driver::Search;
use crate::tuner::specimen::Specimen;
use crate::util::Rng;
use evolve::Evolve;

pub enum Algorithm { Evolve(Evolve) }
impl Algorithm {
    /// Search to the algorithm's own budget and hand back the population it kept, unranked.
    pub fn explore(&self, search: &Search, rng: &mut Rng) -> Vec<Specimen> {
        match self {
            Algorithm::Evolve(evolve) => evolve.explore(search, rng),
        }
    }
}
