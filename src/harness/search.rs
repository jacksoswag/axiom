//! A search a frontend started: assembled from one command line, run on its own thread, and reported
//! one generation at a time. Cancellation is a flag the watch closure reads, so tuner needs no idea a
//! frontend exists and the session loop needs no channel of its own.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Instant;

use crate::engine::params::FixedGenome;
use crate::harness::catalog::{algorithm_word, criterion_word, default_of, metric_of, EVOLVE_KNOBS};
use crate::harness::protocol::{numbers, quote, quoted, Command};
use crate::tuner::algorithms::evolve::{Evolve, Mutation};
use crate::tuner::algorithms::Algorithm;
use crate::tuner::criterion::Criterion;
use crate::tuner::driver::{Search, Tally};
use crate::tuner::gate::Gate;
use crate::tuner::metrics::Metric;
use crate::tuner::specimen::Specimen;

/// A running search, seen from the session loop: lines to forward, a flag to raise, a thread to join.
pub struct Job {
    stop: Arc<AtomicBool>,
    events: Receiver<String>,
    handle: JoinHandle<Vec<Specimen>>,
}
impl Job {
    /// Every line the search produced since the last look. A search reports one generation at a time,
    /// so this is empty most turns and never long, and the caller reads it straight off the channel
    /// rather than through a list built for the occasion.
    pub fn drain(&self) -> std::sync::mpsc::TryIter<'_, String> { self.events.try_iter() }
    pub fn finished(&self) -> bool { self.handle.is_finished() }
    pub fn stop(&self) { self.stop.store(true, Ordering::Relaxed); }
    /// The ranked population, or whatever a panicked search left behind. A search that ends because
    /// its stop flag went up still returns everything it found.
    pub fn join(self) -> Result<Vec<Specimen>, String> {
        self.handle.join().map_err(|panic| {
            let message = panic.downcast_ref::<String>().map(|text| text.as_str())
                .or_else(|| panic.downcast_ref::<&str>().copied());
            format!("the search died: {}", message.unwrap_or("no message"))
        })
    }
}

/// One search from one command. Anything a frontend leaves out falls back to the knob table, so a
/// config may arrive as thin as `search start criterion=structure`. 'seeds' is what the caller wants
/// refined, which is how the world on screen becomes the place a search starts.
pub fn start(command: &Command, shape: &FixedGenome, gates: Vec<Gate>, seeds: Vec<Vec<f32>>)
    -> Result<Job, String>
{
    let knob = |name: &str| -> Result<f32, String> {
        Ok(command.number(name)?.unwrap_or(default_of(&EVOLVE_KNOBS, name)))
    };
    let evolve = Evolve {
        generations: knob("generations")? as usize,
        batch: (knob("batch")? as usize).max(1),
        capacity: (knob("capacity")? as usize).max(1),
        mutation: Mutation { tournament: knob("tournament")?, expedition: knob("expedition")?,
            iso: knob("iso")?, line: knob("line")? },
    };
    let criterion = match command.text("criterion").unwrap_or("novelty") {
        "novelty" => Criterion::Novelty(axes(command)?),
        "structure" => Criterion::Structure,
        other => return Err(format!("no criterion called '{other}'")),
    };
    let algorithm = match command.text("algorithm").unwrap_or("evolve") {
        "evolve" => Algorithm::Evolve(evolve),
        other => return Err(format!("no algorithm called '{other}'")),
    };
    let timesteps = command.count("timesteps")?.unwrap_or(400).max(1);
    let seed = command.seed("seed")?.unwrap_or(shape.seed);

    let stop = Arc::new(AtomicBool::new(false));
    let raised = Arc::clone(&stop);
    let (sender, events) = channel();
    let shape = shape.clone();
    // The Search is assembled inside the thread so the plan it settles on is reported from the one
    // place that knows it, and so a long probe measurement never blocks the session loop.
    let handle = std::thread::spawn(move || {
        let mut search = Search::new(shape, algorithm, criterion, gates, timesteps, seed);
        search.seeds = seeds;
        let opened = Instant::now();
        sender.send(started(&search)).ok();
        let mut watch = |generation: usize, population: &[Specimen], tally: &Tally| {
            sender.send(generation_line(generation, population, tally, opened)).is_ok()
                && !raised.load(Ordering::Relaxed) // a dropped receiver stops it too, since nobody is left to tell
        };
        search.run(&mut watch)
    });
    Ok(Job { stop, events, handle })
}

/// Novelty wants a space to be far apart in and names no metric of its own, so an empty axis list is
/// refused here rather than quietly scoring every genome zero.
fn axes(command: &Command) -> Result<Vec<Metric>, String> {
    let mut axes = Vec::new();
    for key in command.words("axes") {
        axes.push(metric_of(key).ok_or_else(|| format!("no metric called '{key}'"))?);
    }
    if axes.is_empty() { return Err("novelty wants axes=key,key to spread genomes across".into()); }
    Ok(axes)
}

/// What the search resolved to, which a frontend cannot work out for itself: the plan is the only
/// layout a descriptor has, and its widths say where one metric's slots end. The shape travels too,
/// because a survivor is only loadable back into a world that has the shape it was drawn for.
fn started(search: &Search) -> String {
    let keys: Vec<&str> = search.metrics.iter().map(|metric| metric.key).collect();
    let widths: Vec<String> = search.metrics.iter().map(|metric| metric.width.to_string()).collect();
    let gates: Vec<String> = search.gates.iter().map(|gate| {
        format!("{{\"metric\":{},\"floor\":{},\"ceiling\":{}}}", quote(gate.metric.key), gate.floor, gate.ceiling)
    }).collect();
    let shape = &search.fixed_genome;
    format!("{{\"type\":\"search\",\"state\":\"started\",\"criterion\":\"{}\",\"algorithm\":\"{}\",\"timesteps\":{},\"seed\":{},\"refining\":{},\"plan\":{},\"widths\":[{}],\"gates\":[{}],\"shape\":{{\"particles\":{},\"anchors\":{},\"shells\":{},\"bumps\":{},\"radius\":{},\"dt\":{},\"seed\":{}}}}}",
        criterion_word(&search.criterion), algorithm_word(&search.algorithm), search.timesteps,
        search.seed, search.seeds.len(), quoted(&keys), widths.join(","), gates.join(","),
        shape.particle_count, shape.anchor_count, shape.shells, shape.bumps, shape.radius,
        shape.dt, shape.seed)
}

/// One generation as it lands. The population is already best first, so the leader and the middle of
/// the pack come off the ends; the whole population waits until the search is over rather than
/// copying a hundred genomes a generation at a view nobody is watching yet. The tally rides along
/// because "nothing survived" and "everything was gated" call for opposite adjustments.
///
/// ponytail: browsing the population while the search runs is what this leaves out. Add it as a
/// command that answers with descriptors alone, not by widening this line: genomes here would put a
/// hundred copies a generation on the wire for a panel nobody has opened.
fn generation_line(generation: usize, population: &[Specimen], tally: &Tally, opened: Instant) -> String {
    let best = population.first();
    let median = population.get(population.len() / 2).map(|member| member.score).unwrap_or(0.0);
    format!("{{\"type\":\"generation\",\"index\":{generation},\"kept\":{},\"best\":{:.6},\"median\":{median:.6},\"elapsed_ms\":{},\"evaluated\":{},\"died\":{},\"gated\":{},\"unscorable\":{},\"genes\":{},\"descriptor\":{}}}",
        population.len(), best.map(|member| member.score).unwrap_or(0.0),
        opened.elapsed().as_millis(), tally.evaluated, tally.died, tally.gated, tally.unscorable,
        numbers(best.map(|member| member.genome.as_slice()).unwrap_or(&[]), 6),
        numbers(best.map(|member| member.metrics.as_slice()).unwrap_or(&[]), 6))
}

/// One survivor, complete enough that a frontend can store it, rank it, and load it back into a world
pub fn specimen_body(rank: usize, specimen: &Specimen) -> String {
    format!("\"rank\":{rank},\"score\":{:.6},\"genes\":{},\"descriptor\":{}",
        specimen.score, numbers(&specimen.genome, 6), numbers(&specimen.metrics, 6))
}
