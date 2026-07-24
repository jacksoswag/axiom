//! Step timing across particle counts in the fixed three-dimensional world.
//! cargo run --release --no-default-features --example bench
//! Reports milliseconds per simulation step and the frame rate that implies, so the
//! O(N^2) -> O(N*k) claim is a measurement rather than an estimate.

use axiom::engine::sim::Sim;
use axiom::tuner::genome::Caps;
use std::time::Instant;

fn main() {
    let counts: Vec<usize> = std::env::args().skip(1).filter_map(|a| a.parse().ok()).collect();
    let counts = if counts.is_empty() { vec![1_000, 10_000, 50_000, 100_000] } else { counts };

    println!("{:>9} {:>9} {:>8} {:>10} {:>9}", "particles", "box", "ms/step", "steps/sec", "setup sec");
    println!("{}", "-".repeat(60));

    for &total in &counts {
        let caps = Caps { particle_count: total, ..Caps::default() };
        let probe = caps.probe(); // campaign-level measurement, outside the per-world timing
        let genome = caps.default_genome(&probe);

        let build = Instant::now();
        let mut sim = Sim::new(&caps.params(&genome, &probe));
        let setup = build.elapsed();

        sim.step(); // warm the neighbor index and the allocator before timing

        let steps = if total > 60_000 { 12 } else { 40 };
        let start = Instant::now();
        for _ in 0..steps { sim.step(); }
        let per_step = start.elapsed().as_secs_f64() / steps as f64;

        println!("{:>9} {:>9.0} {:>8.2} {:>10.0} {:>9.2}",
            total, sim.params.box_len, per_step * 1e3, 1.0 / per_step, setup.as_secs_f64());
    }

    println!("\nsteps/sec is the simulation rate. A viewer running one step per frame");
    println!("renders at that rate; steps_per_frame above 1 divides it.");
}
