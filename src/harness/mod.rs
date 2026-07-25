//! What a frontend talks to: commands in on stdin, JSON events out on stdout, one line each. The only
//! place this crate prints, spawns a thread, or holds a world that is still running, so engine and
//! tuner stay unaware that anything is watching them.

pub mod catalog;
pub mod playground;
pub mod protocol;
pub mod rollout;
pub mod search;

use std::sync::mpsc::{channel, Receiver, RecvTimeoutError, TryRecvError};
use std::time::{Duration, Instant};

use crate::engine::params::FixedGenome;
use crate::harness::catalog::{metric_of, SHAPE_FIELDS};
use crate::harness::playground::Playground;
use crate::harness::protocol::{Command, Emit};
use crate::harness::search::{specimen_body, Job};
use crate::tuner::gate::Gate;
use crate::tuner::metrics::field;

/// How often a moving world ships a frame, takes a live reading, and owes its next batch of ticks.
/// Faster than a screen refreshes spends wire on positions nobody sees, slower reads as a stutter.
const FRAME_INTERVAL: Duration = Duration::from_millis(40);
/// Steps a fast-forward takes per turn of the loop. A jump of a hundred thousand ticks runs in slices
/// so it still answers a command and still streams frames on the way.
const FORWARD_CHUNK: usize = 200;

struct Session {
    world: Playground,
    job: Option<Job>,
    out: Emit,
    forward: usize, // ticks still owed to a fast-forward
    reseed: bool, // raised by an edit, spent once per turn, so a slider drag costs one rebuild
    relaw: bool,
    fresh: bool, // the picture changed while paused, so one frame is owed
    frames: bool,
    quit: bool,
    closed: bool, // stdin ended, so no command can arrive and only owed work is left
    framed: Instant,
    sampled: Instant,
}

/// Read, step, report, repeat. An idle session blocks on stdin instead of spinning, so a paused
/// playground costs nothing.
pub fn run() {
    let commands = listen();
    let mut session = Session::new();
    session.greet();
    loop {
        if session.resting() {
            if session.job.is_none() {
                match commands.recv() {
                    Ok(line) => session.apply(&line),
                    Err(_) => return, // stdin closed and nothing is running, so there is nothing left to do
                }
            } else {
                // A search holds every core for minutes at a time, and this loop has nothing to do while
                // it runs but forward the lines it sends. Waiting on the frame interval gives that core
                // back to the batch and still answers a stop, or a generation line, within one frame.
                match commands.recv_timeout(FRAME_INTERVAL) {
                    Ok(line) => session.apply(&line),
                    Err(RecvTimeoutError::Disconnected) => session.closed = true,
                    Err(RecvTimeoutError::Timeout) => {}
                }
            }
        }
        loop {
            match commands.try_recv() {
                Ok(line) => session.apply(&line),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => { session.closed = true; break; }
            }
        }
        session.settle();
        // A closed stdin with the owed work done is a finished script. Nothing can arrive to stop a
        // running world after that, so it would otherwise step until the process was killed.
        if session.quit || (session.closed && session.forward == 0 && session.job.is_none()) { return; }
        session.turn();
    }
}

/// stdin on its own thread, so an idle session can wait on a command and a busy one can look for one
/// without waiting.
fn listen() -> Receiver<String> {
    let (sender, receiver) = channel();
    std::thread::spawn(move || {
        for line in std::io::stdin().lines() {
            let Ok(line) = line else { break };
            if sender.send(line).is_err() { break; }
        }
    });
    receiver
}

impl Session {
    fn new() -> Session {
        let shape = FixedGenome { particle_count: 320, dimensions: field::GRID_AXES, anchor_count: 3,
            shells: 2, bumps: 1, radius: 2.0, dt: 0.05, seed: 1 };
        Session { world: Playground::new(shape), job: None, out: Emit::new(), forward: 0, reseed: false,
            relaw: false, fresh: true, frames: true, quit: false, closed: false, framed: Instant::now(),
            sampled: Instant::now() }
    }
    /// Nothing to advance and nothing to draw, so the loop has no reason to hold a core. With no search
    /// running that means waiting on stdin outright; with one it means waiting on a short interval,
    /// since a job's lines still want forwarding.
    fn resting(&self) -> bool {
        !self.world.running && self.forward == 0 && !self.fresh
    }
    /// Everything a frontend needs before it can draw a single control, and a picture to put on the
    /// canvas: a page that connected after the world was paused has nothing else to show.
    fn greet(&mut self) {
        let catalog = catalog::body();
        self.out.event("catalog", &catalog);
        self.report_layout();
        self.report_state();
        self.fresh = true;
    }
    fn report_layout(&mut self) {
        let body = catalog::layout(&self.world.shape, &self.world.bounds(), &self.world.genes,
            self.world.box_len());
        self.out.event("layout", &body);
    }
    fn report_state(&mut self) {
        let body = format!("{},\"frames\":{},\"searching\":{}", self.world.state_body(), self.frames,
            self.job.is_some());
        self.out.event("state", &body);
    }
    fn apply(&mut self, line: &str) {
        let Some(command) = Command::parse(line) else { return };
        if let Err(problem) = self.dispatch(&command) { self.out.error(&problem); }
    }
    /// One command. An unknown verb is an error rather than silence: a frontend that thinks it asked
    /// for something deserves to hear that it did not.
    fn dispatch(&mut self, command: &Command) -> Result<(), String> {
        // A frontend can send a shape and a reading in one breath, and a batch of commands drains
        // before the next settle, so anything that reads the world, or throws away what has been read
        // of it, settles first. Edits do not: that is what lets a slider drag coalesce into one rebuild.
        if command.verb == "catalog" || command.verb == "search" || command.verb == "gates"
            || (command.verb == "measure" && command.flag("now")) { self.settle(); }
        match command.verb.as_str() {
            "catalog" => self.greet(),
            "shape" => { self.reshape(command)?; self.reseed = true; }
            // The whole batch is checked before any of it lands. A frontend that hears an error reads it
            // as "nothing happened", and a batch that applied its first half would make that a lie.
            "gene" => {
                let edits = command.indexed()?;
                if edits.iter().any(|(index, _)| *index >= self.world.gene_count()) {
                    return Err(format!("this genome has {} genes", self.world.gene_count()));
                }
                for (index, value) in edits {
                    self.world.edit(index, value);
                    if self.world.shapes_world(index) { self.reseed = true; } else { self.relaw = true; }
                }
            }
            // Genes land raw. A search stores what it drew and a rollout clamps at simulate time, so
            // clamping on the way in here would quietly make a loaded specimen a different genome.
            "genome" => {
                let genes = command.floats("genes")?;
                if genes.len() != self.world.shape.gene_len() {
                    return Err(format!("this shape wants {} genes, got {}", self.world.shape.gene_len(), genes.len()));
                }
                self.world.genes = genes;
                self.reseed = true;
            }
            "run" | "pause" => { self.world.running = command.verb == "run"; self.report_state(); }
            "step" => { self.forward += command.count("count")?.unwrap_or(1); }
            "speed" => {
                self.world.speed = command.count("steps")?.unwrap_or(1).clamp(1, 1024);
                self.report_state();
            }
            "reseed" => {
                if let Some(seed) = command.seed("seed")? { self.world.shape.seed = seed; }
                self.reseed = true;
            }
            "measure" => self.watch(command)?,
            "gates" => self.gate(command)?,
            // Turning frames back on owes a picture: a paused world has nothing else to send.
            "frames" => { self.frames = !command.flag("off"); self.fresh = self.frames; self.report_state(); }
            "search" => self.search(command)?,
            "quit" => self.quit = true,
            other => return Err(format!("no command called '{other}'")),
        }
        Ok(())
    }
    /// Any subset of the shape. Ranges are checked here because this is the trust boundary: past it a
    /// count of zero anchors reaches a decode that indexes a genome and panics.
    fn reshape(&mut self, command: &Command) -> Result<(), String> {
        let mut shape = self.world.shape.clone();
        for (name, _, low, high) in SHAPE_FIELDS {
            let Some(value) = command.number(name)? else { continue };
            if value < low || value > high { return Err(format!("{name} wants {low} to {high}, got {value}")); }
            match name {
                "particles" => shape.particle_count = value as usize,
                "anchors" => shape.anchor_count = value as usize,
                "shells" => shape.shells = value as usize,
                "bumps" => shape.bumps = value as usize,
                "radius" => shape.radius = value,
                "dt" => shape.dt = value,
                _ => {}
            }
        }
        if let Some(seed) = command.seed("seed")? { shape.seed = seed; }
        if let Some(dimensions) = command.count("dimensions")? {
            if dimensions != field::GRID_AXES {
                return Err(format!("the measurement grid is {}-axis", field::GRID_AXES));
            }
        }
        self.world.shape = shape;
        Ok(())
    }
    /// What to watch and how often. `now` takes one reading on the spot, Once metrics included, which
    /// is the only way a repair experiment ever runs in a live world.
    fn watch(&mut self, command: &Command) -> Result<(), String> {
        if command.text("keys").is_some() || command.count("every")?.is_some() {
            let mut wanted = Vec::new();
            for key in command.words("keys") {
                wanted.push(metric_of(key).ok_or_else(|| format!("no metric called '{key}'"))?);
            }
            let every = command.count("every")?.unwrap_or(30).max(1);
            self.world.watch(&wanted, if wanted.is_empty() { 0 } else { every });
            self.report_state();
        }
        if command.flag("now") {
            let values = self.world.measure(true);
            let body = self.world.sample_body(&values, true);
            self.out.event("sample", &body);
        }
        Ok(())
    }
    /// Every gate at once, as metric=floor:ceiling. Replacing the set re-settles the plan, so a gate's
    /// metric is always measured and never reads zero for want of being asked for.
    fn gate(&mut self, command: &Command) -> Result<(), String> {
        let mut gates = Vec::new();
        for (key, value) in command.entries() {
            let metric = metric_of(key).ok_or_else(|| format!("no metric called '{key}'"))?;
            let (floor, ceiling) = value.split_once(':')
                .ok_or_else(|| format!("{key} wants floor:ceiling, got '{value}'"))?;
            gates.push(Gate { metric, floor: reading(key, floor)?, ceiling: reading(key, ceiling)? });
        }
        self.world.set_gates(gates);
        self.report_state();
        Ok(())
    }
    /// Start or stop the one search a session runs at a time. Starting pauses the world: rayon takes
    /// every core for a batch, and a playground stuttering at two frames a second reads as a bug.
    ///
    /// ponytail: a search cannot be resumed. `from=world` seeds it with the one genome on screen, so
    /// carrying on from a recorded run means loading survivors back one at a time. Teach `from` to take
    /// a run name and hand its whole population to `seeds` if that ever gets in the way.
    fn search(&mut self, command: &Command) -> Result<(), String> {
        if command.flag("stop") {
            match &self.job {
                Some(job) => { job.stop(); self.out.event("search", "\"state\":\"stopping\""); }
                None => return Err("no search is running".into()),
            }
            return Ok(());
        }
        if !command.flag("start") { return Err("search wants start or stop".into()); }
        // Stopping finishes the generation it is in, since a batch of rollouts is already out on every
        // core. A big batch is therefore a long goodbye, which is worth saying out loud here.
        if self.job.is_some() { return Err("a search is already running, and a stop lands after its current generation".into()); }
        let gates = self.world.gates().iter().cloned().collect();
        // `from=world` hands the search the genome on screen to refine. Without it the search opens on
        // fresh draws, which is the honest baseline.
        let seeds = if command.flag("from") || command.text("from") == Some("world") {
            vec![self.world.genes.clone()]
        } else { Vec::new() };
        self.job = Some(search::start(command, &self.world.shape, gates, seeds)?);
        self.world.running = false;
        self.report_state();
        Ok(())
    }
    /// Spend the pending flags, once, before anything steps. A slider drag arrives as dozens of edits
    /// and one pair edit costs a fresh density calibration, so the rebuild happens here and not there.
    fn settle(&mut self) {
        if self.reseed { self.world.reseed(); self.report_layout(); }
        else if self.relaw { self.world.refresh_law(); }
        if self.reseed || self.relaw { self.fresh = true; self.report_state(); }
        self.reseed = false;
        self.relaw = false;
    }
    /// One turn: forward the search, advance the world, read it if a reading is due, draw it if a
    /// frame is due, and give back whatever is left of the interval a running world is paced to.
    fn turn(&mut self) {
        let opened = Instant::now();
        if let Some(job) = &self.job {
            for line in job.drain() { self.out.line(&line); }
            if job.finished() { self.harvest(); }
        }
        let steps = if self.forward > 0 { self.forward.min(FORWARD_CHUNK) }
            else if self.world.running { self.world.speed } else { 0 };
        if steps > 0 {
            self.forward = self.forward.saturating_sub(steps);
            if !self.world.advance(steps) {
                self.forward = 0;
                self.out.error("the world blew up, so it stopped");
                self.report_state();
            }
            // A cadence in ticks is a cadence in wall time too, or it isn't a cadence: a fast-forward
            // covers thousands of ticks a second, and a full evaluation is worth more than that.
            if self.world.measurement_due() && self.sampled.elapsed() >= FRAME_INTERVAL {
                let values = self.world.measure(false);
                let body = self.world.sample_body(&values, false);
                self.out.event("sample", &body);
                self.sampled = Instant::now();
            }
            if self.forward == 0 && !self.world.running { self.report_state(); } // a fast-forward ended
        }
        // A fresh picture draws at once, since an edit made while paused should land under the cursor;
        // a moving one draws on the interval, since nobody sees more than a screen refresh.
        if self.frames && (self.fresh || (steps > 0 && self.framed.elapsed() >= FRAME_INTERVAL)) {
            let body = self.world.frame_body();
            self.out.event("frame", &body);
            self.framed = Instant::now();
            self.fresh = false;
        }
        // Speed is a rate, not a batch size: a running world owes `speed` ticks every frame interval and
        // hands back what is left of one. A fast-forward is exempt, since a jump of a hundred thousand
        // ticks is asked for rather than watched, and an idle session is already asleep on stdin.
        if self.world.running && self.forward == 0 {
            if let Some(rest) = FRAME_INTERVAL.checked_sub(opened.elapsed()) { std::thread::sleep(rest); }
        }
    }
    /// Wind up a finished search: every survivor best first, then how many there were. A panic upstairs
    /// arrives here as a message rather than as silence.
    fn harvest(&mut self) {
        let Some(job) = self.job.take() else { return };
        for line in job.drain() { self.out.line(&line); } // whatever landed since the last look
        match job.join() {
            Ok(population) => {
                let bodies: Vec<String> = population.iter().enumerate()
                    .map(|(rank, specimen)| specimen_body(rank, specimen)).collect();
                self.out.events("specimen", &bodies); // one write for the burst, not one per survivor
                self.out.event("search", &format!("\"state\":\"done\",\"kept\":{}", population.len()));
            }
            Err(problem) => {
                let body = format!("\"state\":\"failed\",\"reason\":{}", protocol::quote(&problem));
                self.out.event("search", &body);
            }
        }
        self.report_state();
    }
}

/// One end of a gate range, named so the complaint says which gate was wrong
fn reading(key: &str, text: &str) -> Result<f32, String> {
    match text.parse::<f32>() {
        Ok(value) if value.is_finite() => Ok(value),
        _ => Err(format!("{key} wants finite bounds, got '{text}'")),
    }
}
