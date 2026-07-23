//! Step timing across particle counts in the fixed three-dimensional world.
//!
//! `cargo run --release --no-default-features --example bench`
//!
//! Reports milliseconds per simulation step and the frame rate that implies, so the
//! `O(N²)` → `O(N·k)` claim is a measurement rather than an estimate.

use axiom::engine::world::World;
use axiom::engine::genome::Params;
use std::time::Instant;

fn main() {
    let counts: Vec<usize> = std::env::args()
        .skip(1)
        .filter_map(|a| a.parse().ok())
        .collect();
    let counts = if counts.is_empty() {
        vec![1_000, 10_000, 50_000, 100_000]
    } else {
        counts
    };

    println!(
        "{:>9} {:>9} {:>8} {:>10} {:>9}",
        "particles", "extent", "ms/step", "steps/sec", "setup sec"
    );
    println!("{}", "-".repeat(60));

    for &total in &counts {
        let params = Params {
            particles: total,
            coordination: 9.0,
            ..Default::default()
        };
        let geometry = params.geometry();
        let genome = params
            .layout()
            .default_genome(params.radius, geometry.extent);

        let build = Instant::now();
        let mut world = World::with_geometry(&params, &geometry, &genome);
        let setup = build.elapsed();

        // Warm the neighbour index and the allocator before timing.
        world.step();

        let steps = if total > 60_000 { 12 } else { 40 };
        let start = Instant::now();
        for _ in 0..steps {
            world.step();
        }
        let per_step = start.elapsed().as_secs_f64() / steps as f64;

        println!(
            "{:>9} {:>9.0} {:>8.2} {:>10.0} {:>9.2}",
            total,
            world.geometry.extent,
            per_step * 1e3,
            1.0 / per_step,
            setup.as_secs_f64(),
        );
    }

    println!("\nsteps/sec is the simulation rate. A viewer running one step per frame");
    println!("renders at that rate; `steps_per_frame` above 1 divides it.");
}
