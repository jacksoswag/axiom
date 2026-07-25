//! The live world a frontend drives: one shape, one genome, the Sim they decode into, and the little
//! that live measurement needs to keep reading across ticks. Edits land here; the loop above only
//! decides when. Nothing here prints, and nothing here knows a search exists.

use std::fmt::Write;

use crate::engine::params::{FixedGenome, Genome};
use crate::engine::sim::Sim;
use crate::engine::resolve::Probe;
use crate::engine::substrate::Substrate;
use crate::engine::r#trait::init_particle_traits;
use crate::harness::protocol::{numbers, quote, quoted, write_numbers};
use crate::harness::rollout::{build, SAMPLES};
use crate::tuner::gate::Gate;
use crate::tuner::metrics::{alive, evaluate, plan, rdf, scalar, wants_baseline, Blocks, Metric, Motion, Rollout};

/// Positions one frame carries at most. Past this a frame decimates by an integer stride: a cloud
/// keeps its shape, the wire keeps its size, and density read off it keeps its meaning because the
/// frontend's threshold is in multiples of uniform density rather than an absolute count.
const FRAME_POINTS: usize = 8192;

pub struct Playground {
    pub shape: FixedGenome,
    pub genes: Vec<f32>,
    pub probe: Probe, // kept across reseeds, since only three shape fields can stale it
    pub running: bool,
    pub speed: usize, // ticks a running world owes per frame interval, which makes the knob a rate
    pub every: usize, // ticks between live readings, 0 is off
    sim: Sim,
    gates: Vec<Gate>,
    wanted: Vec<Metric>, // what a frontend asked to watch, kept apart from the plan that answers it
    plan: Vec<Metric>,
    baseline: Option<rdf::Histogram>, // this cloud's uniform start, built the first time a plan reads one
    history: Vec<Vec<f32>>, // last few readings, so anything measuring change has something behind it
    motion: Option<Motion>,
    measured: u64, // tick of the last reading, so a cadence survives a speed change
    frame: String, // the three buffers a frame is staged and written into, kept so a stream of them
    frame_positions: Vec<f32>, // costs no allocation at all: at 8,192 points and 25 Hz it was most of
    frame_traits: Vec<f32>, // what this thread did between steps
}
impl Playground {
    /// A world from a shape alone: one uniform draw inside the bounds that shape allows, seeded from
    /// the shape's own seed so the same shape always opens on the same world.
    pub fn new(shape: FixedGenome) -> Playground {
        let probe = Probe::new(&shape);
        let genes = Genome::build_random(&shape.bounds(&probe), shape.seed);
        let mut sim = build(&shape, &genes, &probe);
        sim.threaded = true; // one sim with every core to itself, the opposite of a search batch
        Playground { shape, genes, probe, sim, baseline: None, gates: Vec::new(), running: false, speed: 1,
            every: 0, wanted: Vec::new(), plan: Vec::new(), history: Vec::new(), motion: None, measured: 0,
            frame: String::new(), frame_positions: Vec::new(), frame_traits: Vec::new() }
    }
    /// Rebuild the world from the shape and genes as they now stand. The readings go with it: a
    /// descriptor's layout is its plan and its baseline is this cloud's own uniform start, so a
    /// history that outlived its world would index the wrong slots against the wrong reference.
    pub fn reseed(&mut self) {
        // The reference measurement depends on particle count, dimensions and radius and nothing else,
        // so a coordination or logit edit reseeds the world without paying for a fresh one.
        if !self.probe.fits(&self.shape) { self.probe = Probe::new(&self.shape); }
        if self.genes.len() != self.shape.gene_len() { // a shape change leaves stale slots behind
            self.genes = Genome::build_random(&self.shape.bounds(&self.probe), self.shape.seed);
        }
        self.sim = build(&self.shape, &self.genes, &self.probe);
        self.sim.threaded = true;
        self.baseline = None;
        self.forget();
    }
    /// The law changed and the world did not: rederive the matrix and its density norms, and keep the
    /// particles exactly where they are, so an edit is judged against the state it was made in.
    pub fn refresh_law(&mut self) {
        let mut fresh = build(&self.shape, &self.genes, &self.probe);
        fresh.substrate.positions = std::mem::take(&mut self.sim.substrate.positions);
        fresh.substrate.traits = std::mem::take(&mut self.sim.substrate.traits);
        fresh.substrate.memberships = std::mem::take(&mut self.sim.substrate.memberships);
        fresh.tick = self.sim.tick;
        fresh.threaded = true;
        self.sim = fresh;
    }
    /// True when a gene sets the world rather than the law, which is the difference between reseeding
    /// and rederiving: coordination sets the box, the logits set who gets seeded where.
    pub fn shapes_world(&self, index: usize) -> bool { index <= self.shape.anchor_count }
    /// One gene, clamped into what this box allows. A frontend's slider may still be showing the range
    /// from a wider box, and a pair gene out past the box's reach wraps a kernel into itself.
    pub fn edit(&mut self, index: usize, value: f32) {
        let (low, high) = self.bounds()[index];
        self.genes[index] = value.clamp(low, high);
    }
    /// Per-gene ranges at the box this genome resolved to. The world prefix keeps the widest-box
    /// bounds a search mutates under; only the pair blocks narrow.
    pub fn bounds(&self) -> Vec<(f32, f32)> {
        let mut bounds = self.shape.bounds(&self.probe);
        bounds.truncate(1 + self.shape.anchor_count);
        bounds.extend(self.shape.pair_bounds(self.box_len()));
        bounds
    }
    pub fn box_len(&self) -> f32 { self.sim.substrate.box_len }
    /// Whether this law can use the spatial index at all. A reach past a third of the box, or a matrix
    /// too inert to reach anywhere, leaves every step walking every pair, which is where a world that
    /// looks like every other one costs many times more to run than the last.
    pub fn gridded(&self) -> bool { self.sim.substrate.gridded(self.sim.matrix.max_reach) }
    pub fn tick(&self) -> u64 { self.sim.tick }
    pub fn gene_count(&self) -> usize { self.genes.len() }
    /// Step the world on. A blow-up stops the run instead of streaming NaN positions at a frontend,
    /// and reports which it was so the frontend can say so.
    pub fn advance(&mut self, steps: usize) -> bool {
        for _ in 0..steps {
            self.sim.step();
            if !alive(&self.sim.substrate.positions) { self.running = false; return false; }
        }
        true
    }
    pub fn watch(&mut self, wanted: &[Metric], every: usize) {
        self.wanted = wanted.to_vec();
        self.every = every;
        self.settle_plan();
    }
    pub fn gates(&self) -> &[Gate] { &self.gates }
    pub fn set_gates(&mut self, gates: Vec<Gate>) { self.gates = gates; self.settle_plan(); }
    /// What was asked for plus whatever the gates read, mirroring how a search settles its own plan: a
    /// gate whose metric went unmeasured would read 0 and look failed rather than unknown.
    fn settle_plan(&mut self) {
        let mut keys = self.wanted.clone();
        keys.extend(self.gates.iter().map(|gate| gate.metric));
        self.plan = plan(&keys);
        self.forget(); // a new layout means every earlier reading indexes the wrong slots
    }
    pub fn measurement_due(&self) -> bool {
        self.every > 0 && !self.plan.is_empty() && self.sim.tick - self.measured >= self.every as u64
    }
    /// One reading of the running world, through the same blocks and the same evaluation a rollout
    /// uses, so a live number and a searched one mean the same thing. 'full' pays for the Once
    /// metrics, which cost a few hundred extra steps, so the loop only asks when a frontend does.
    pub fn measure(&mut self, full: bool) -> Vec<f32> {
        if self.plan.is_empty() { return Vec::new(); }
        // The uniform start is what structure gets measured against, and reading one is an all-pairs
        // pass, so it waits here for the first plan that wants it. The cloud it measures is redrawn
        // from the shape and genome that seeded this world rather than kept alive beside the running
        // one: seeding is a few thousand rng draws against a histogram quadratic in the swarm.
        if self.baseline.is_none() && wants_baseline(&self.plan) {
            let mut start = Substrate::build(&self.shape, self.box_len());
            init_particle_traits(&mut start, &self.shape.decode(&self.genes).trait_distribution, self.shape.seed);
            self.baseline = Some(rdf::build(&start));
        }
        let unread = rdf::blank(); // what a plan with nothing to compare against is handed instead
        let previous = self.motion.take(); // out of the way, so the block below can hand back a new one
        let rollout = Rollout { fixed_genome: &self.shape, matrix: &self.sim.matrix,
            baseline: self.baseline.as_ref().unwrap_or(&unread) };
        let blocks = Blocks::build(&rollout, &mut self.sim.substrate, &self.plan, previous, &self.history);
        let values = evaluate(&blocks, full);
        self.motion = blocks.carry();
        self.history.push(values.clone());
        if self.history.len() > SAMPLES { self.history.remove(0); } // the count a rollout folds, so the two compare
        self.measured = self.sim.tick;
        values
    }
    /// Drop what only made sense against the world that just went away
    fn forget(&mut self) { self.history.clear(); self.motion = None; self.measured = self.sim.tick; }

    /// The cloud as a frontend draws it: positions and traits, decimated, at the precision a screen
    /// rounds to anyway.
    ///
    /// ponytail: decimal text, at about seven bytes a coordinate against four for the number itself.
    /// Base64 f32 is the escape hatch, worth taking only once the pipe or JSON.parse shows up in a
    /// profile; the stride and the true count already travel, so the wire's shape is negotiable.
    pub fn frame_body(&mut self) -> &str {
        let count = self.sim.substrate.traits.len();
        let stride = (count / FRAME_POINTS.max(1)).max(1);
        self.frame_positions.clear(); self.frame_traits.clear();
        for particle in (0..count).step_by(stride) {
            self.frame_positions.extend_from_slice(self.sim.substrate.pos(particle));
            self.frame_traits.push(self.sim.substrate.traits[particle]);
        }
        // The true count travels beside the decimated one: blend reach follows the real swarm's spacing,
        // so thinning the wire never inflates what a particle looks like.
        let frame = &mut self.frame;
        frame.clear();
        write!(frame, "\"tick\":{},\"box_len\":{:.4},\"stride\":{stride},\"particles\":{count},\"positions\":",
            self.sim.tick, self.sim.substrate.box_len).ok();
        write_numbers(frame, &self.frame_positions, 3);
        frame.push_str(",\"traits\":");
        write_numbers(frame, &self.frame_traits, 3);
        &self.frame
    }
    /// One reading, laid out by the plan that produced it, with every gate's verdict beside it. The
    /// widths travel too: a descriptor is a bare row of numbers and the plan is the only layout it has.
    pub fn sample_body(&self, values: &[f32], full: bool) -> String {
        let keys: Vec<&str> = self.plan.iter().map(|metric| metric.key).collect();
        let widths: Vec<String> = self.plan.iter().map(|metric| metric.width.to_string()).collect();
        let gates: Vec<String> = self.gates.iter().map(|gate| {
            let value = scalar(values, &self.plan, gate.metric);
            format!("{{\"metric\":{},\"floor\":{},\"ceiling\":{},\"value\":{value:.6},\"pass\":{}}}",
                quote(gate.metric.key), gate.floor, gate.ceiling,
                value >= gate.floor && value <= gate.ceiling)
        }).collect();
        format!("\"tick\":{},\"full\":{full},\"keys\":{},\"widths\":[{}],\"values\":{},\"gates\":[{}]",
            self.sim.tick, quoted(&keys), widths.join(","), numbers(values, 6), gates.join(","))
    }
    /// Everything a frontend would otherwise have to remember on its own
    pub fn state_body(&self) -> String {
        let keys: Vec<&str> = self.plan.iter().map(|metric| metric.key).collect();
        format!("\"running\":{},\"tick\":{},\"box_len\":{:.4},\"gridded\":{},\"speed\":{},\"every\":{},\"watching\":{},\"gene_count\":{},\"shape\":{{\"particles\":{},\"anchors\":{},\"shells\":{},\"bumps\":{},\"radius\":{},\"dt\":{},\"seed\":{}}}",
            self.running, self.sim.tick, self.box_len(), self.gridded(), self.speed, self.every, quoted(&keys),
            self.genes.len(), self.sim.substrate.traits.len(), self.shape.anchor_count,
            self.shape.shells, self.shape.bumps, self.shape.radius, self.shape.dt, self.shape.seed)
    }
}
