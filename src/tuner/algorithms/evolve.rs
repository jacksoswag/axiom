//! Evolutionary search: mutate what is already working, aim part of every generation at unexplored
//! behavior rather than at good behavior, and keep a fresh-draw escape route open at any batch size.
//! Parents are picked by whatever the criterion scored them with, so this one algorithm hill-climbs
//! a fitness or chases novelty depending only on what it was routed with. It owns its whole loop and
//! reaches the sim only through the Search that owns it.

use crate::engine::params::{clamp, Genome};
use crate::tuner::driver::{Reject, Search, Tally};
use crate::tuner::specimen::Specimen;
use crate::util::{distance, Rng};
use rayon::prelude::*;

/// generations is search depth, batch is how many genomes each round draws, capacity is how many
/// survive to breed. mutation shapes the draw; zero its lanes and it collapses to random search.
pub struct Evolve {
    pub generations: usize,
    pub batch: usize,
    pub capacity: usize,
    pub mutation: Mutation,
}
impl Evolve {
    /// One run to completion. Generation zero proposes against an empty population, which comes back
    /// as fresh draws, so the opening sample needs no special case.
    pub fn explore(&self, search: &Search, rng: &mut Rng,
        watch: &mut impl FnMut(usize, &[Specimen], &Tally) -> bool) -> Vec<Specimen>
    {
        // Whatever the caller handed over gets measured first, so generation zero can improve on a world
        // someone already chose instead of starting again from noise. Anything that fails its gates on
        // the way in simply is not there, which is the same answer the search would give it later.
        let mut population: Vec<Specimen> = search.seeds.iter()
            .filter_map(|genes| search.run_genome(genes, &[]).ok()).collect();
        for generation in 0..=self.generations {
            rescore(search, &mut population);
            let genomes = propose(self.mutation, &population, &search.bounds, self.batch, rng);
            let judged: Vec<Result<Specimen, Reject>> = genomes.into_par_iter()
                .map(|genome| search.run_genome(&genome, &population))
                .collect(); // order-preserving, so a parallel batch judges identically to a serial one
            let mut tally = Tally { evaluated: judged.len(), ..Tally::default() };
            let evaluated: Vec<Specimen> = judged.into_iter().filter_map(|outcome| match outcome {
                Ok(specimen) => Some(specimen),
                Err(Reject::Died) => { tally.died += 1; None }
                Err(Reject::Gated) => { tally.gated += 1; None }
                Err(Reject::Unscorable) => { tally.unscorable += 1; None }
            }).collect();
            population = retain(population, evaluated, self.capacity);
            if !watch(generation, &population, &tally) { break; } // a caller that stops keeps what was found
        }
        population
    }
}

/// A population-relative criterion leaves every stored score stale the moment the population moves
/// around it, so the whole population is re-scored before anything is ranked or picked from. Each
/// member is scored against a set that contains it; the bias is identical for everyone, so it can
/// never change an ordering. An absolute criterion has nothing to redo and does not walk the list.
fn rescore(search: &Search, population: &mut [Specimen]) {
    if !search.criterion.relative() { return; }
    // Scores land in a list first so the population can be read whole while it is being scored
    // against, which is what the old copy of it bought.
    let reference: &[Specimen] = population;
    let scores: Vec<f32> = reference.iter()
        .map(|member| search.criterion.score(&member.metrics, &search.metrics, reference)).collect();
    for (member, score) in population.iter_mut().zip(scores) { member.score = score; }
}

/// Zero both lanes and every slot comes back a fresh uniform draw, which is the random-search
/// baseline a real setting has to beat: if random matches evolution under some criterion, that
/// criterion is not steering.
#[derive(Clone, Copy)]
pub struct Mutation {
    pub tournament: f32, // share of a generation mutated from a score tournament
    pub expedition: f32, // share mutated from whatever sits nearest a random behavior target
    pub iso: f32, // per-gene jitter, as a fraction of that gene's own range
    pub line: f32, // one shared step along the vector toward a second parent
}

/// Fresh draws take whatever the two named lanes do not claim, so the three always sum to batch
/// regardless of rounding and there is always a way out of a converged population.
///
/// ponytail: the old search widened mutation and shifted the lanes once novelty went flat. That
/// needed history across generations, which propose cannot see here. Restore it by giving Evolve a
/// mutation state field, not by handing this the whole run.
fn propose(mutation: Mutation, population: &[Specimen], bounds: &[(f32, f32)], batch: usize,
    rng: &mut Rng) -> Vec<Vec<f32>>
{
    let share = |fraction: f32| ((batch as f32 * fraction.clamp(0.0, 1.0)) as usize).min(batch);
    let tournament = share(mutation.tournament);
    let expedition = share(mutation.expedition).min(batch - tournament);
    (0..batch).map(|slot| {
        if population.len() < 2 { return fresh(bounds, rng); } // too small to mutate from
        if slot < tournament { mutate(mutation, best_of_two(population, rng), population, bounds, rng) }
        else if slot < tournament + expedition { mutate(mutation, nearest_target(population, rng), population, bounds, rng) }
        else { fresh(bounds, rng) }
    }).collect()
}

fn fresh(bounds: &[(f32, f32)], rng: &mut Rng) -> Vec<f32> {
    Genome::build_random(bounds, rng.next_u64())
}

/// Iso plus line: independent noise scaled to each gene's range, then one shared draw along the
/// difference vector toward a second parent, so a child inherits a direction the population
/// already contains instead of dissolving into undirected noise.
fn mutate(mutation: Mutation, first: usize, population: &[Specimen], bounds: &[(f32, f32)],
    rng: &mut Rng) -> Vec<f32>
{
    let second = &population[best_of_two(population, rng)];
    let first = &population[first];
    let line = rng.normal();
    let mut genome: Vec<f32> = first.genome.iter().zip(&second.genome).zip(bounds)
        .map(|((&a, &b), &(low, high))| a + mutation.iso * (high - low) * rng.normal() + mutation.line * line * (b - a))
        .collect();
    clamp(&mut genome, bounds);
    genome
}

/// Binary tournament on score, ties falling to the first draw so the pick stays deterministic.
fn best_of_two(population: &[Specimen], rng: &mut Rng) -> usize {
    let (left, right) = (rng.below(population.len()), rng.below(population.len()));
    if population[right].score > population[left].score { right } else { left }
}

/// Throw a dart at a uniform point in descriptor space and mutate whichever member landed nearest
/// it. In dozens of dimensions that is a random goal direction rather than coverage-aware
/// illumination, and that is the intent: distant intentions without a model inventing state.
fn nearest_target(population: &[Specimen], rng: &mut Rng) -> usize {
    let width = population[0].metrics.len();
    let target: Vec<f32> = (0..width).map(|_| rng.range(0.0, 1.0)).collect();
    // One distance per member. Comparing on the fly re-measures both sides of every comparison, which
    // is two passes over the descriptor space for an answer one pass already has.
    population.iter().enumerate().map(|(index, member)| (distance(&member.metrics, &target), index))
        .min_by(|left, right| left.0.total_cmp(&right.0)).map_or(0, |(_, index)| index)
}

/// Fold an evaluated generation into the population that survives it. Best score first, and a
/// member is only kept if it sits clear of everything already kept, so the top scorer in a cluster
/// survives and the rest of that cluster does not: a converged batch cannot fill the population
/// with copies of one genome.
fn retain(population: Vec<Specimen>, evaluated: Vec<Specimen>, capacity: usize) -> Vec<Specimen> {
    let capacity = capacity.max(1);
    let mut combined = population;
    combined.extend(evaluated);
    let cutoff = CROWDING * median_spacing(&combined);
    combined.sort_by(|left, right| right.score.total_cmp(&left.score));
    let mut kept: Vec<Specimen> = Vec::with_capacity(capacity);
    for member in combined {
        if kept.len() >= capacity { break; }
        let descriptor = &member.metrics;
        if kept.iter().all(|other| distance(descriptor, &other.metrics) >= cutoff) { kept.push(member); }
    }
    kept
}

/// How close counts as a duplicate, as a fraction of the population's own spacing rather than an
/// absolute distance, so it holds whatever units and however many axes a descriptor turns out to
/// have. A population of identical genomes has zero spacing and drops nothing, which is what lets an
/// opening generation through.
const CROWDING: f32 = 0.25;

/// Median distance from a member to its closest neighbor: the natural scale of this population's
/// descriptor space. Quadratic in the population and serial between two parallel batches, which is
/// invisible at the capacity ceiling of 512 and would not stay that way if the ceiling moved.
fn median_spacing(population: &[Specimen]) -> f32 {
    let mut spacing: Vec<f32> = population.iter().enumerate().map(|(index, member)| {
        population.iter().enumerate().filter(|(other, _)| *other != index)
            .map(|(_, other)| distance(&member.metrics, &other.metrics))
            .fold(f32::INFINITY, f32::min) // a lone member has no neighbor, so it stays infinite and drops out below
    }).filter(|value| value.is_finite()).collect();
    if spacing.is_empty() { return 0.0; }
    spacing.sort_by(f32::total_cmp);
    spacing[spacing.len() / 2]
}
