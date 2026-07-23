//! One continuous-trait Particle-Lenia step.
//!
//! Each source trait contributes to at most two anchor channels and each receiver responds
//! through at most two destination channels. The pair-indexed control net therefore preserves
//! the discrete model exactly at anchor traits while keeping interpolation cheap.

use crate::engine::grid::Grid;
use crate::engine::interaction::Net;
use crate::engine::kernel::{bump_and_slope, displacement_squared, growth};
use crate::engine::r#trait::membership;
use crate::engine::substrate::Substrate;
use rayon::prelude::*;

/// Advance positions by the closed-form local energy gradient of the pair-indexed anchor field.
pub fn step(
    sub: &mut Substrate,
    net: &Net,
    dt: f32,
    extent: f32,
    softening: f32,
    grid: &Grid,
) {
    let count = sub.len();
    let dims = sub.dims();
    let positions = &sub.positions;
    let anchors = net.anchors();
    let pairs = net.pairs();
    let inverse_extent = 1.0 / extent;
    let softening_squared = softening * softening;
    let reach_squared: Vec<f32> = pairs
        .iter()
        .map(|interaction| interaction.reach.powi(2))
        .collect();
    let mut delta = vec![0.0; positions.len()];
    let memberships: Vec<[(usize, f32); 2]> = sub
        .traits
        .iter()
        .map(|&trait_value| membership(trait_value, anchors).entries)
        .collect();

    delta.par_chunks_mut(dims).enumerate().for_each_init(
        || Scratch::new(pairs.len(), dims),
        |scratch, (i, out)| {
            let Scratch {
                potential,
                gradient,
                displacement,
            } = scratch;
            potential.fill(0.0);
            gradient.fill(0.0);
            let position_i = &positions[i * dims..(i + 1) * dims];
            let receiver = memberships[i];

            let mut visit = |j: usize| {
                if i == j {
                    return;
                }
                let source = memberships[j];
                let position_j = &positions[j * dims..(j + 1) * dims];
                let squared = displacement_squared(
                    position_i,
                    position_j,
                    extent,
                    inverse_extent,
                    displacement,
                );
                for &(source_anchor, source_weight) in &source {
                    if source_weight <= 0.0 {
                        continue;
                    }
                    for &(destination_anchor, destination_weight) in &receiver {
                        if destination_weight <= 0.0 {
                            continue;
                        }
                        let index = source_anchor * anchors + destination_anchor;
                        if squared + softening_squared > reach_squared[index] {
                            continue;
                        }
                        let distance = (squared + softening_squared).sqrt();
                        let (kernel, kernel_prime) =
                            bump_and_slope(distance, &pairs[index].shells);
                        potential[index] += source_weight * kernel;
                        let row = &mut gradient[index * dims..(index + 1) * dims];
                        let scale = source_weight * kernel_prime / distance;
                        for axis in 0..dims {
                            row[axis] += scale * displacement[axis];
                        }
                    }
                }
            };

            grid.for_each_candidate(position_i, count, &mut visit);

            for source_anchor in 0..anchors {
                for &(destination_anchor, destination_weight) in &receiver {
                    if destination_weight <= 0.0 {
                        continue;
                    }
                    let index = source_anchor * anchors + destination_anchor;
                    let interaction = &pairs[index];
                    let (_, growth_prime) =
                        growth(potential[index] / interaction.norm, &interaction.bumps);
                    let scale = dt * destination_weight * interaction.weight * growth_prime
                        / interaction.norm;
                    let row = &gradient[index * dims..(index + 1) * dims];
                    for axis in 0..dims {
                        out[axis] += scale * row[axis];
                    }
                }
            }
        },
    );

    for (coordinate, change) in sub.positions.iter_mut().zip(delta) {
        *coordinate = (*coordinate + change).rem_euclid(extent);
    }
}

struct Scratch {
    potential: Vec<f32>,
    gradient: Vec<f32>,
    displacement: Vec<f32>,
}

impl Scratch {
    fn new(pairs: usize, dims: usize) -> Scratch {
        Scratch {
            potential: vec![0.0; pairs],
            gradient: vec![0.0; pairs * dims],
            displacement: vec![0.0; dims],
        }
    }
}
