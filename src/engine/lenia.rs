//! One Particle-Lenia simulation step.
//! Each source trait contributes to 2 anchor channels and each receiver responds
//! through 2 destination channels. The pair-indexed control matrix therefore preserves
//! the discrete model exactly at anchor traits while keeping interpolation cheap.

use crate::engine::gpu;
use crate::engine::matrix::Matrix;
use crate::engine::kernel::{strength_and_slope, distance_sq};
use crate::engine::substrate::Substrate;
use rayon::prelude::*;

/// Advance positions by the closed-form local energy gradient of the pair-indexed anchor field.
/// Owns the per-step spatial index rebuild, so a caller only needs to hand over the substrate.
///
/// threaded spreads the particle loop over the cores. It is a large win for the one sim that owns the
/// machine and a loss for one of a batch: inside a search the batch already fills every core, so a
/// second level of splitting adds fork-join scheduling and a fresh set of scratch buffers per
/// work-stealing split on top of work that was saturating them already.
pub fn step(substrate: &mut Substrate, matrix: &Matrix, dt: f32, threaded: bool) {
    if !(matrix.max_reach > 0.0) { return; } // no shell reaches anything, so no pair can move a particle
    substrate.rebuild_grid(matrix.max_reach); // re-bucket for this step's widest interaction radius
    // The card runs the same walk over the index that was just built, and is worth the round trip only
    // once the swarm is large. Below that, or with no card, the cores do it, which is also the only
    // path that reproduces bit for bit from one machine to another.
    if gpu::worth_it(substrate) { gpu::step(substrate, matrix, dt); } else { walk(substrate, matrix, dt, threaded); }
}

/// The same tick on the cores, over an index the caller has already rebuilt. Reachable on its own
/// because the card's answer only means something measured against this one, at a size the router
/// would otherwise never send here.
pub fn walk(substrate: &mut Substrate, matrix: &Matrix, dt: f32, threaded: bool) {
    let dims = substrate.dimensions; // coordinates per particle
    let anchor_count = matrix.anchor_count;
    let interactions = &matrix.interactions; // flat [anchors x anchors], source-major
    let memberships = &substrate.memberships; // each particle's up-to-2 active anchors, settled at trait init

    let mut delta = std::mem::take(&mut substrate.delta); // this tick's position change, one slot per coordinate
    delta.clear(); delta.resize(substrate.positions.len(), 0.0); // last tick's buffer, zeroed rather than replaced
    let scratch = || (vec![0.0; interactions.len()], vec![0.0; interactions.len() * dims], vec![0.0; dims]);
    let particle = |work: &mut (Vec<f32>, Vec<f32>, Vec<f32>), (i, out): (usize, &mut [f32])| { // one chunk of dims floats per particle, indexed
        let (potential, gradient, direction) = work; // potential, gradient, direction scratch
        potential.fill(0.0); gradient.fill(0.0); // clear scratch left by the last particle this thread did
        let i_pos = &substrate.positions[i * dims..(i + 1) * dims]; // this particle's own coordinates
        let receiver = memberships[i]; // this particle's own up-to-2 active anchors

        // For every neighbor j within reach, accumulate sensed potential K(d) and its gradient
        // per (source, destination) anchor pair. Anchors with 0 weight don't contribute.
        let mut visit = |j: usize| {
            if i == j { return; } // skip self
            let source = memberships[j]; // neighbor's 2 active anchors
            let j_pos = &substrate.positions[j * dims..(j + 1) * dims]; // neighbor's coordinates
            let distance_sq = distance_sq(i_pos, j_pos, substrate, direction); // softened, fills per-axis buffer
            let mut distance = -1.0; // lazy: same value for every surviving index below, sqrt at most once

            for &(source_anchor, source_weight) in &source.entries { // neighbor's active anchors
                if source_weight <= 0.0 { continue; }
                for &(destination_anchor, destination_weight) in &receiver.entries { // this particle's active anchors
                    if destination_weight <= 0.0 { continue; }
                    let index = source_anchor as usize * anchor_count + destination_anchor as usize; // flatten to this pair's row
                    if distance_sq > interactions[index].reach_sq { continue; } // squared compare skips the sqrt
                    if distance < 0.0 { distance = distance_sq.sqrt(); }
                    let (kernel, kernel_prime) = strength_and_slope(distance, matrix.shells_of(index)); // K(d) and its slope
                    potential[index] += source_weight * kernel; // K(d), this pair's sensed density
                    let row = &mut gradient[index * dims..(index + 1) * dims]; // this pair's gradient accumulator
                    let scale = source_weight * kernel_prime / distance; // chain rule: dK/dx = K'(d) * d(distance)/dx
                    for axis in 0..dims { row[axis] += scale * direction[axis]; } // scale direction into the gradient
                }
            }
        };
        substrate.visit_neighbors(i_pos, &mut visit); // grid-accelerated neighbor walk

        // Turn accumulated potential into motion: G'(U/norm), each pair's pull strength at
        // the density i's neighbors produced, scaled into this tick's step.
        for source_anchor in 0..anchor_count { // every possible source, not just active ones
            for &(destination_anchor, destination_weight) in &receiver.entries {
                if destination_weight <= 0.0 { continue; }
                let index = source_anchor * anchor_count + destination_anchor as usize;
                let interaction = &interactions[index]; // this pair's behavior
                let (_, g_prime) = strength_and_slope(potential[index] / interaction.norm, matrix.bumps_of(index)); // B'(U/norm), bump mixture slope
                let scale = 2.0 * dt * destination_weight * interaction.weight * g_prime / interaction.norm; // 2x: G = 2B-1 maps to [-1,1], the -1 dies in G'
                let row = &gradient[index * dims..(index + 1) * dims]; // pair's accumulated direction
                for axis in 0..dims { out[axis] += scale * row[axis]; } // add to tick's pos delta
            }
        }
    };
    if threaded {
        delta.par_chunks_mut(dims).enumerate().for_each_init(scratch, particle);
    } else {
        let mut work = scratch(); // one set for the whole loop, since one thread walks every particle
        for item in delta.chunks_mut(dims).enumerate() { particle(&mut work, item); }
    }
    // Explicit Euler step, wrapped back onto the torus.
    for (coordinate, change) in substrate.positions.iter_mut().zip(&delta) {
        *coordinate = (*coordinate + *change).rem_euclid(substrate.box_len); // advance, then wrap
    }
    substrate.delta = delta; // buffer back where the next tick finds it
}
