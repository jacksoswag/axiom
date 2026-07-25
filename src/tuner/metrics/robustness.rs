//! Does a genome repair itself. The only metric that runs an experiment instead of an observation:
//! damage a copy, step it and an untouched control forward, and ask whether the damaged one caught
//! up to where the control ended. Costs a run's worth of stepping, so it reduces Once.

use crate::engine::kernel::distance_sq;
use crate::engine::lenia::step;
use crate::engine::substrate::Substrate;
use crate::util::{distance, Rng};
use rayon::prelude::*;
use super::{alive, connectivity, evaluate, heterogeneity, rdf, Blocks, Metric, Reduce, Spec};

const WIDTH: usize = 1;
pub const METRIC: Metric = &Spec { reduce: Reduce::Once, ..Spec::of("robustness", WIDTH, measure) };

/// Mean share of the damage its probes closed, which is already the axis
pub fn measure(base: &Blocks, _: &[f32]) -> Vec<f32> { vec![repaired(base)] }

/// Constants rather than run settings: changing any of them changes what a robustness reading
/// means, so two runs that disagree about them were never comparable in the first place.
const PROBES: usize = 3;
const SHAKE: f32 = 0.5; // displacement applied to the damaged region, in kernel cutoffs
const MIN_EFFECTIVE_DAMAGE: f32 = 0.01; // damage below this proves nothing, the probe is discarded
const STEPS: usize = 100; // relaxation window, fixed so a long search and a short one still compare

/// How much of the damage closes: shake a local region, step on, and ask how far the genome got
/// back toward where its own undamaged control went. Compared against the future control rather
/// than the present state, so ordinary drift is not billed as damage.
///
/// A share rather than a count of probes clearing a threshold. Counting gave a step function with
/// PROBES + 1 levels that sat on zero for most genomes, so anything multiplying by it lost its
/// gradient entirely.
fn repaired(base: &Blocks) -> f32 {
    let rollout = base.rollout;
    let shake_distance = rollout.matrix.max_reach * SHAKE;
    let box_len = base.substrate.box_len;
    let mut control = base.substrate.copy();
    let present = signature(base, &mut control); // control still is the present, so this reads it

    // Every probe's damage is applied here, in probe order, so the rng draws land in the sequence they
    // always did no matter how the relaxations below are scheduled.
    let mut damaged: Vec<Substrate> = Vec::with_capacity(PROBES);
    for probe in 0..PROBES {
        let mut copy = base.substrate.copy();
        if copy.traits.is_empty() { continue; }
        let mut rng = Rng::new(rollout.fixed_genome.seed.wrapping_mul(0x9E37_79B9).wrapping_add(probe as u64 + 1));
        let dims = copy.dimensions;
        let center = copy.pos(rng.below(copy.traits.len())).to_vec();
        let region = (2.0 * shake_distance).min(box_len * 0.25); // damage stays local to one neighborhood
        let hit: Vec<usize> = (0..copy.traits.len())
            .filter(|&i| distance_sq(copy.pos(i), &center, &copy, &mut []).sqrt() <= region)
            .collect();
        for i in hit { // same particle order as the scan, so rng draws stay reproducible
            for coordinate in &mut copy.positions[i * dims..(i + 1) * dims] {
                *coordinate = (*coordinate + shake_distance * rng.normal()).rem_euclid(box_len);
            }
        }
        damaged.push(copy);
    }

    // The control's next hundred steps and each probe's own are independent runs of one law, and every
    // comparison below wants all of them, so they go out at once. Inside a batch search the cores are
    // already full and rayon just runs them where they stand; in a live world it is the difference
    // between the picture freezing for a second and freezing for a quarter of one.
    let (future, probes) = rayon::join(
        || {
            for _ in 0..STEPS {
                step(&mut control, rollout.matrix, rollout.fixed_genome.dt, false);
                if !alive(&control.positions) { return None; }
            }
            Some(signature(base, &mut control))
        },
        || damaged.par_iter_mut().map(|copy| {
            let damage = distance(&present, &signature(base, copy));
            if damage < MIN_EFFECTIVE_DAMAGE { return None; } // the shake did nothing, so nothing is proven
            for _ in 0..STEPS {
                step(copy, rollout.matrix, rollout.fixed_genome.dt, false);
                if !alive(&copy.positions) { break; }
            }
            Some((damage, alive(&copy.positions).then(|| signature(base, copy))))
        }).collect::<Vec<_>>());

    let Some(future) = future else { return 0.0 }; // a control that blew up has no future to compare against
    let (mut closed, mut counted) = (0.0, 0);
    for outcome in probes.into_iter().flatten() { // probe order, so the sum reads the same every run
        counted += 1;
        let (damage, Some(after)) = outcome else { continue }; // blowing up closes nothing, and scores it
        closed += (1.0 - distance(&future, &after) / damage).clamp(0.0, 1.0);
    }
    if counted == 0 { 0.0 } else { closed / counted as f32 }
}

/// The spatial picture repair compares against, read through the same plan machinery the descriptor
/// uses, so damage and recovery land in the space the search already thinks in. Robustness is
/// deliberately absent, which is what keeps this from recursing into itself.
const SIGNATURE: [Metric; 3] = [rdf::METRIC, heterogeneity::METRIC, connectivity::METRIC];

fn signature(base: &Blocks, substrate: &mut Substrate) -> Vec<f32> {
    evaluate(&base.probe(substrate, &SIGNATURE), true)
}
