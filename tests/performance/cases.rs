//! What gets measured. Every case builds its own world from the same surviving genome, so a size
//! change is the only thing that differs between two rows of the same table.

use axiom::engine::gpu;
use axiom::engine::lenia::{step, walk};
use axiom::engine::matrix::Matrix;
use axiom::engine::params::{FixedGenome, Genome};
use axiom::engine::resolve::Probe;
use axiom::engine::sim::{calibration_substrate, Sim};
use axiom::engine::substrate::Substrate;
use axiom::engine::r#trait::init_particle_traits;
use axiom::harness::playground::Playground;
use axiom::harness::rollout;
use axiom::tuner::algorithms::evolve::{Evolve, Mutation};
use axiom::tuner::algorithms::Algorithm;
use axiom::tuner::criterion::structure::METRICS as STRUCTURE;
use axiom::tuner::criterion::Criterion;
use axiom::tuner::driver::Search;
use axiom::tuner::metrics::{self, connectivity, field, heterogeneity, rdf, robustness, topology};
use axiom::tuner::metrics::{Blocks, Rollout};

use crate::{rate, time, Timing};

/// The shape every case starts from, at whatever population it is asking about
fn shape(particle_count: usize) -> FixedGenome {
    FixedGenome { particle_count, dimensions: 3, anchor_count: 3, shells: 2, bumps: 1,
        radius: 2.0, dt: 0.05, seed: 1 }
}

/// A genome still alive and still on the grid after a long run. A cloud that has blown up folds into
/// one cell, where every step walks every pair, so timing one measures the fallback rather than the
/// engine. Found once at the smallest size and reused: raw genes carry across populations, which is
/// exactly what the harness does when someone drags the particle count.
fn living() -> Vec<f32> {
    let small = shape(320);
    let probe = Probe::new(&small);
    for seed in 1..500u64 {
        let genes = Genome::build_random(&small.bounds(&probe), seed);
        let mut sim = rollout::build(&small, &genes, &probe);
        sim.run(2_000);
        if metrics::alive(&sim.substrate.positions) && sim.substrate.gridded(sim.matrix.max_reach) {
            return genes;
        }
    }
    panic!("no seed under 500 drew a world that survives two thousand steps on the grid");
}

/// One sim of a given size, wound forward past the opening transient so a timing reads the settled
/// world rather than the first few ticks of a uniform cloud.
fn settled(particle_count: usize, genes: &[f32]) -> Sim {
    let form = shape(particle_count);
    let probe = Probe::new(&form);
    let mut sim = rollout::build(&form, genes, &probe);
    sim.run(200);
    assert!(metrics::alive(&sim.substrate.positions), "the fixture blew up at {particle_count} particles");
    sim
}

pub fn engine() -> Vec<Timing> {
    let genes = living();
    let mut out = Vec::new();
    for count in [320usize, 2_000, 8_000] {
        for threaded in [false, true] {
            let mut sim = settled(count, &genes);
            let dt = sim.fixed_genome.dt;
            let name = format!("step, {count} particles, {}", if threaded { "threaded" } else { "sequential" });
            out.push(time("engine", name, "", || {
                step(&mut sim.substrate, &sim.matrix, dt, threaded);
                sim.substrate.positions[0] as f64
            }));
            assert!(metrics::alive(&sim.substrate.positions), "the fixture blew up while being timed");
            let each = out[out.len() - 1].each;
            out.last_mut().map(|timing| timing.rate = rate(count as f64, each));
        }
    }
    // Where the card earns its keep. Both backends are asked for the same tick over the same index,
    // so the difference is the walk and nothing around it. A machine with no adapter reports the cores
    // twice, which is the honest reading of what it would actually run.
    for count in [40_000usize, 150_000] {
        let mut sim = settled(count, &genes);
        let dt = sim.fixed_genome.dt;
        let reach = sim.matrix.max_reach;
        for on_card in [false, true] {
            let name = format!("step, {count} particles, {}", if on_card { "card" } else { "cores" });
            out.push(time("engine", name, "", || {
                sim.substrate.rebuild_grid(reach);
                if on_card && gpu::worth_it(&sim.substrate) { gpu::step(&mut sim.substrate, &sim.matrix, dt); }
                else { walk(&mut sim.substrate, &sim.matrix, dt, true); }
                sim.substrate.positions[0] as f64
            }));
            let each = out[out.len() - 1].each;
            out.last_mut().map(|timing| timing.rate = rate(count as f64, each));
        }
        let mut sim = settled(count, &genes);
        out.push(time("engine", format!("rebuild_grid, {count} particles"), "", || {
            sim.substrate.rebuild_grid(reach);
            reach as f64
        }));
    }
    let mut sim = settled(8_000, &genes);
    let reach = sim.matrix.max_reach;
    out.push(time("engine", "rebuild_grid, 8,000 particles", "", || {
        sim.substrate.rebuild_grid(reach);
        reach as f64
    }));
    out
}

pub fn setup() -> Vec<Timing> {
    let genes = living();
    let mut out = Vec::new();
    let small = shape(320);
    let large = shape(20_000);
    out.push(time("setup", "Probe::new, 320 particles", "", || Probe::new(&small).box_len(8.0) as f64));
    out.push(time("setup", "Probe::new, 20,000 particles (capped at 1,500)", "",
        || Probe::new(&large).box_len(8.0) as f64));

    let probe = Probe::new(&small);
    let genome = small.decode(&genes);
    let box_len = probe.box_len(genome.coordination);
    out.push(time("setup", "Matrix::derive", "", || Matrix::derive(&small, &genome).max_reach as f64));
    out.push(time("setup", "Matrix::derive + norm_densities, 320", "", || {
        let mut matrix = Matrix::derive(&small, &genome);
        matrix.norm_densities(&mut calibration_substrate(small.clone(), box_len));
        matrix.interactions[0].norm as f64
    }));
    let big = shape(8_000);
    let big_genome = big.decode(&genes);
    let big_box = Probe::new(&big).box_len(big_genome.coordination);
    out.push(time("setup", "Matrix::derive + norm_densities, 8,000", "", || {
        let mut matrix = Matrix::derive(&big, &big_genome);
        matrix.norm_densities(&mut calibration_substrate(big.clone(), big_box));
        matrix.interactions[0].norm as f64
    }));
    out.push(time("setup", "Substrate::build + traits, 8,000", "", || {
        let mut substrate = Substrate::build(&big, big_box);
        init_particle_traits(&mut substrate, &big_genome.trait_distribution, big.seed);
        substrate.positions[0] as f64
    }));
    out.push(time("setup", "Sim::new, 320 (the whole per-candidate setup)", "",
        || rollout::build(&small, &genes, &probe).matrix.max_reach as f64));
    out
}

pub fn measurement() -> Vec<Timing> {
    let genes = living();
    let mut out = Vec::new();
    // The one block that was quadratic in the swarm. Past the sample ceiling it costs the same at
    // every size, which is the whole reason a campaign at a hundred and fifty thousand is possible.
    for count in [320usize, 8_000, 150_000] {
        let sim = settled(count, &genes);
        out.push(time("measurement", format!("rdf::build, {count} particles"), "",
            || rdf::build(&sim.substrate).bins[0]));
    }
    for count in [320usize, 8_000] {
        let mut sim = settled(count, &genes);
        let reach = sim.matrix.max_reach;
        out.push(time("measurement", format!("topology::h0, {count} particles"), "",
            || topology::h0(&mut sim.substrate, reach).components as f64));
    }
    let sim = settled(8_000, &genes);
    out.push(time("measurement", "field::build side 8, 8,000 particles", "",
        || field::build(&sim.substrate, 8).density[0] as f64));
    out.push(time("measurement", "field::build side 16, 8,000 particles", "",
        || field::build(&sim.substrate, 16).density[0] as f64));

    // The battery as a rollout actually runs it: one sample of the structure plan, blocks and all.
    let plan = metrics::plan(&STRUCTURE);
    let mut sim = settled(320, &genes);
    let baseline = rdf::build(&sim.substrate);
    let shape = sim.fixed_genome.clone();
    out.push(time("measurement", "one sample, structure plan, 320 (no Once metrics)", "", || {
        let rollout = Rollout { fixed_genome: &shape, matrix: &sim.matrix, baseline: &baseline };
        let blocks = Blocks::build(&rollout, &mut sim.substrate, &plan, None, &[]);
        metrics::evaluate(&blocks, false)[0] as f64
    }));
    out.push(time("measurement", "connectivity + heterogeneity alone, 320", "", || {
        let rollout = Rollout { fixed_genome: &shape, matrix: &sim.matrix, baseline: &baseline };
        let blocks = Blocks::build(&rollout, &mut sim.substrate, &plan, None, &[]);
        connectivity::measure(&blocks, &[])[0] as f64 + heterogeneity::measure(&blocks, &[])[0] as f64
    }));
    // The repair experiment, which is a rollout's worth of stepping hidden inside one metric.
    let robust = metrics::plan(&[robustness::METRIC]);
    out.push(time("measurement", "robustness::measure, 320 (400 steps + 8 signatures)", "", || {
        let rollout = Rollout { fixed_genome: &shape, matrix: &sim.matrix, baseline: &baseline };
        let blocks = Blocks::build(&rollout, &mut sim.substrate, &robust, None, &[]);
        robustness::measure(&blocks, &[])[0] as f64
    }));
    out
}

pub fn search() -> Vec<Timing> {
    let genes = living();
    let small = shape(320);
    let probe = Probe::new(&small);
    let mut out = Vec::new();
    let bare: Vec<metrics::Metric> = Vec::new();
    out.push(time("search", "rollout, 400 steps, nothing measured", "",
        || rollout::sim(&small, &bare, &[], &probe, &genes, 400).len() as f64));
    let plan = metrics::plan(&STRUCTURE);
    out.push(time("search", "rollout, 400 steps, structure plan", "",
        || rollout::sim(&small, &plan, &[], &probe, &genes, 400).len() as f64));
    let novelty = metrics::plan(&[rdf::METRIC]);
    out.push(time("search", "rollout, 400 steps, rdf only", "",
        || rollout::sim(&small, &novelty, &[], &probe, &genes, 400).len() as f64));

    out.push(time("search", "one generation, structure, batch 48, 400 steps", "", || {
        let evolve = Evolve { generations: 0, batch: 48, capacity: 96,
            mutation: Mutation { tournament: 0.5, expedition: 0.25, iso: 0.05, line: 0.3 } };
        let search = Search::new(shape(320), Algorithm::Evolve(evolve), Criterion::Structure,
            Vec::new(), 400, 1);
        search.run(&mut |_, _, _| true).len() as f64
    }));
    let each = out[out.len() - 1].each;
    out.last_mut().map(|timing| timing.rate = format!("{} candidates", rate(48.0, each)));
    out
}

pub fn harness() -> Vec<Timing> {
    let mut out = Vec::new();
    let mut world = Playground::new(shape(8_192));
    out.push(time("harness", "frame_body, 8,192 points", "", || world.frame_body().len() as f64));
    let each = out[out.len() - 1].each;
    out.last_mut().map(|timing| timing.rate = format!("{} chars", rate(world.frame_body().len() as f64, each)));
    let mut live = Playground::new(shape(320));
    out.push(time("harness", "playground advance, 320 particles", "", || {
        live.advance(1);
        live.tick() as f64
    }));
    out
}

/// Two runs of one genome, and two runs of one sim, compared bit for bit. Every optimization here
/// leans on replay staying exact within a binary, so the suite says so out loud rather than assuming.
pub fn determinism() -> String {
    let genes = living();
    let small = shape(320);
    let probe = Probe::new(&small);
    let plan = metrics::plan(&STRUCTURE);
    let first = rollout::sim(&small, &plan, &[], &probe, &genes, 400);
    let second = rollout::sim(&small, &plan, &[], &probe, &genes, 400);
    let descriptors = first.iter().map(|v| v.to_bits()).eq(second.iter().map(|v| v.to_bits()));

    let mut a = settled(320, &genes);
    let mut b = settled(320, &genes);
    a.run(200); b.run(200);
    let positions = a.substrate.positions.iter().map(|v| v.to_bits())
        .eq(b.substrate.positions.iter().map(|v| v.to_bits()));

    // The threaded and sequential paths accumulate per particle in the same fixed order, so they are
    // the same arithmetic on the same values and have to agree bit for bit.
    let mut threaded = settled(320, &genes);
    let dt = threaded.fixed_genome.dt;
    for _ in 0..200 { step(&mut threaded.substrate, &threaded.matrix, dt, true); }
    let mut plain = settled(320, &genes);
    for _ in 0..200 { step(&mut plain.substrate, &plain.matrix, dt, false); }
    let paths = threaded.substrate.positions.iter().map(|v| v.to_bits())
        .eq(plain.substrate.positions.iter().map(|v| v.to_bits()));

    // The card is the one thing here that cannot match bit for bit: its exp() is the hardware
    // approximation rather than the one libm ships. So it is asked the two questions it can answer,
    // whether a tick lands where the cores put it to the precision f32 has, and whether it repeats.
    let (card_apart, card_repeats) = against_the_cores(&genes);

    assert!(descriptors, "two rollouts of one genome disagreed");
    assert!(positions, "two runs of one sim disagreed");
    assert!(paths, "threaded and sequential stepping disagreed");
    assert!(card_repeats, "the card did not repeat itself over twenty ticks");
    format!("- two rollouts of one genome produce bit-identical descriptors: {descriptors}\n\
             - two runs of one sim produce bit-identical positions: {positions}\n\
             - threaded and sequential stepping agree bit for bit: {paths}\n\
             - the card repeats itself bit for bit over twenty ticks: {card_repeats}\n\
             - one tick on the card lands within {card_apart:.2e} of a box of where the cores put it\n")
}

/// One tick each way at a size the router really sends to the card, and twenty ticks on the card
/// twice. Reports the worst coordinate disagreement as a fraction of the box, and whether the card
/// repeated. With no card, or with AXIOM_GPU=0, both paths are the cores and the answer is exactly 0.
fn against_the_cores(genes: &[f32]) -> (f32, bool) {
    let form = shape(40_000);
    let probe = Probe::new(&form);
    let run = |ticks: usize, on_card: bool| {
        let mut sim = rollout::build(&form, genes, &probe);
        for _ in 0..ticks {
            let (dt, reach) = (sim.fixed_genome.dt, sim.matrix.max_reach);
            sim.substrate.rebuild_grid(reach);
            if on_card && gpu::worth_it(&sim.substrate) { gpu::step(&mut sim.substrate, &sim.matrix, dt); }
            else { walk(&mut sim.substrate, &sim.matrix, dt, true); }
        }
        sim.substrate.positions
    };
    let span = rollout::build(&form, genes, &probe).substrate.box_len;
    let worst = run(1, true).iter().zip(&run(1, false)).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let bits = |held: Vec<f32>| held.iter().map(|value| value.to_bits()).collect::<Vec<_>>();
    (worst / span, bits(run(20, true)) == bits(run(20, true)))
}
