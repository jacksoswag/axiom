//! Every knob a run can turn, in one place.
//!
//! One `Tuning` value fully determines a campaign: the fixed world, discovery search, the
//! viability gates, the repair protocol, which expensive tiers run, and how the scheduling
//! model is fit. Nothing else in `tuner` holds a tunable constant.
//!
//! What is deliberately *not* here: descriptor geometry (`RDF_BINS`, `HETEROGENEITY_SIDES`,
//! `LAGS`, `BARS`, `SAMPLES`, `MOBILITY_CEILING`) and `NEIGHBOURS`. Those define what a
//! descriptor axis *means*, size fixed-width arrays, and keep `descriptor_len()` a `const fn`.
//! They change by editing code and bumping `DESCRIPTOR_VERSION`, never per run.
//!
//! Anything here that can change what counts as evidence feeds [`Tuning::digest`], which
//! [`crate::tuner::campaign::ExperimentIdentity`] carries. Two campaigns tuned differently
//! cannot merge their ledgers by accident.

use crate::engine::genome::Caps;
use crate::tuner::learning::{ContinuationQuotas, LogisticConfig};

/// Isotropic and line-directed mutation scale, as a fraction of each gene's bounded range.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mutation {
    pub iso: f32,
    pub line: f32,
    /// Spread multiplier applied while the search is stalled.
    pub stalled_spread: f32,
}

impl Default for Mutation {
    fn default() -> Self {
        Mutation {
            iso: 0.01,
            line: 0.2,
            stalled_spread: 1.8,
        }
    }
}

/// Share of each generation spent on novelty parents and goal-directed expeditions. Whatever
/// the two do not claim goes to random births, which preserves a random escape route at every
/// batch size.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lanes {
    pub novelty: f32,
    pub expedition: f32,
    pub stalled_novelty: f32,
    pub stalled_expedition: f32,
}

impl Default for Lanes {
    fn default() -> Self {
        Lanes {
            novelty: 0.7,
            expedition: 0.2,
            stalled_novelty: 0.5,
            stalled_expedition: 0.3,
        }
    }
}

impl Lanes {
    /// Lane counts for one generation. Random takes the remainder, so the three always sum to
    /// `batch` regardless of rounding.
    pub fn counts(&self, batch: usize, stalled: bool) -> LaneCounts {
        let (novelty_share, expedition_share) = if stalled {
            (self.stalled_novelty, self.stalled_expedition)
        } else {
            (self.novelty, self.expedition)
        };
        let novelty = share(batch, novelty_share);
        let expedition = share(batch, expedition_share).min(batch - novelty);
        LaneCounts {
            novelty,
            expedition,
            random: batch - novelty - expedition,
        }
    }
}

fn share(batch: usize, fraction: f32) -> usize {
    ((batch as f32 * fraction.clamp(0.0, 1.0)) as usize).min(batch)
}

#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct LaneCounts {
    pub novelty: usize,
    pub expedition: usize,
    pub random: usize,
}

/// Binary admission and promotion thresholds. These never combine into a scalar fitness; each
/// one is a named clause that a world either clears or fails.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gates {
    pub structure_floor: f32,
    pub structure_ceiling: f32,
    pub variance_floor: f32,
    pub robustness_floor: f32,
    pub heterogeneity_floor: f32,
    pub connected_fraction_floor: f32,
    pub mobility_floor: f32,
    /// Structure at or below which a rollout is watched for sustained dispersion.
    pub dispersed_structure: f32,
    /// Structure at or above which a rollout is watched for sustained collapse.
    pub collapsed_structure: f32,
    /// Consecutive watchdog windows required before a rollout stops early.
    pub sustained_windows: usize,
}

impl Default for Gates {
    fn default() -> Self {
        Gates {
            structure_floor: 0.05,
            structure_ceiling: 1.0,
            variance_floor: 1e-4,
            robustness_floor: 0.66,
            heterogeneity_floor: 0.01,
            connected_fraction_floor: 0.75,
            mobility_floor: 0.001,
            dispersed_structure: 0.01,
            collapsed_structure: 1.25,
            sustained_windows: 3,
        }
    }
}

/// The damage-and-recover protocol behind the `robustness` metric.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Repair {
    pub probes: usize,
    /// Displacement applied to the damaged region, as a fraction of the kernel cutoff.
    pub shake: f32,
    /// Fraction of the original damage a world must close to count as recovered.
    pub recovery: f32,
    /// Damage below this is too small to prove anything, so the probe is discarded.
    pub min_effective_damage: f32,
}

impl Default for Repair {
    fn default() -> Self {
        Repair {
            probes: 3,
            shake: 0.5,
            recovery: 0.5,
            min_effective_damage: 0.01,
        }
    }
}

/// Discovery search: the cheap tier that fills the archive.
#[derive(Clone, Debug, PartialEq)]
pub struct Discovery {
    pub generations: usize,
    pub steps: usize,
    pub batch: usize,
    pub initial: usize,
    pub capacity: usize,
    pub seed: u64,
    pub promotion_budget: usize,
    pub stall_window: usize,
    pub mutation: Mutation,
    pub lanes: Lanes,
    /// Archive indices a browser pinned. They may influence only the expedition share and are
    /// still evaluated and pruned like every other parent.
    pub curated_parent_indices: Vec<usize>,
}

impl Default for Discovery {
    fn default() -> Self {
        Discovery {
            generations: 30,
            steps: 1500,
            batch: 32,
            initial: 64,
            capacity: 2000,
            seed: 7,
            promotion_budget: 0,
            stall_window: 8,
            mutation: Mutation::default(),
            lanes: Lanes::default(),
            curated_parent_indices: Vec::new(),
        }
    }
}

/// Which expensive tiers run, and for how long. `None` means the tier is off. Seed counts and
/// pass requirements are properties of the tier itself, not of a run.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct Tiers {
    pub persistence_steps: Option<usize>,
    pub certification_steps: Option<usize>,
}

/// How the persistence scheduling model is fit. It allocates rollout effort and nothing else.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Learning {
    pub minimum_labeled_rows: usize,
    pub model_seed: u64,
    pub quotas: ContinuationQuotas,
    pub logistic: LogisticConfig,
}

impl Default for Learning {
    fn default() -> Self {
        Learning {
            minimum_labeled_rows: 30,
            model_seed: 0xA71_0A0,
            quotas: ContinuationQuotas {
                high_survival: 4,
                high_uncertainty: 3,
                uniform_neighborhood: 3,
            },
            logistic: LogisticConfig::default(),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Tuning {
    pub world: Caps,
    pub discovery: Discovery,
    pub gates: Gates,
    pub repair: Repair,
    pub tiers: Tiers,
    pub learning: Learning,
}

impl Tuning {
    /// Fold every setting that changes what an evaluation *means* into one value. Search
    /// scheduling knobs (seed, batch, generations, capacity, promotion budget, curated pins)
    /// are excluded on purpose: they change how much searching happens, never how a result is
    /// measured or judged, so batches that differ only there may share a ledger.
    pub fn digest(&self) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        let mut eat = |bytes: &[u8]| {
            for byte in bytes {
                hash ^= *byte as u64;
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
        };
        let Gates {
            structure_floor,
            structure_ceiling,
            variance_floor,
            robustness_floor,
            heterogeneity_floor,
            connected_fraction_floor,
            mobility_floor,
            dispersed_structure,
            collapsed_structure,
            sustained_windows,
        } = self.gates;
        for value in [
            structure_floor,
            structure_ceiling,
            variance_floor,
            robustness_floor,
            heterogeneity_floor,
            connected_fraction_floor,
            mobility_floor,
            dispersed_structure,
            collapsed_structure,
        ] {
            eat(&value.to_bits().to_le_bytes());
        }
        eat(&(sustained_windows as u64).to_le_bytes());

        let Repair {
            probes,
            shake,
            recovery,
            min_effective_damage,
        } = self.repair;
        eat(&(probes as u64).to_le_bytes());
        for value in [shake, recovery, min_effective_damage] {
            eat(&value.to_bits().to_le_bytes());
        }

        let Mutation {
            iso,
            line,
            stalled_spread,
        } = self.discovery.mutation;
        let Lanes {
            novelty,
            expedition,
            stalled_novelty,
            stalled_expedition,
        } = self.discovery.lanes;
        for value in [
            iso,
            line,
            stalled_spread,
            novelty,
            expedition,
            stalled_novelty,
            stalled_expedition,
        ] {
            eat(&value.to_bits().to_le_bytes());
        }
        eat(&(self.discovery.stall_window as u64).to_le_bytes());
        hash
    }
}

/// One named setting, addressable as a `key=value` pair.
///
/// The same table drives CLI parsing, generated help, and any future settings panel, so a knob
/// cannot exist without documentation or drift out of sync with its help text.
pub struct Knob {
    pub key: &'static str,
    pub help: &'static str,
    setter: fn(&mut Tuning, &str) -> Result<(), String>,
    shower: fn(&Tuning) -> String,
}

impl Knob {
    pub fn set(&self, tuning: &mut Tuning, value: &str) -> Result<(), String> {
        (self.setter)(tuning, value)
    }
    pub fn show(&self, tuning: &Tuning) -> String {
        (self.shower)(tuning)
    }
}

macro_rules! knob {
    ($key:literal, $help:literal, $($field:ident).+, $ty:ty) => {
        Knob {
            key: $key,
            help: $help,
            setter: |cfg: &mut Tuning, raw: &str| {
                cfg.$($field).+ = raw
                    .parse::<$ty>()
                    .map_err(|_| format!("{} expects {}", $key, stringify!($ty)))?;
                Ok(())
            },
            shower: |cfg: &Tuning| cfg.$($field).+.to_string(),
        }
    };
}

/// Steps for an optional tier: `0` turns it off, any other value turns it on with that budget.
macro_rules! tier_knob {
    ($key:literal, $help:literal, $field:ident) => {
        Knob {
            key: $key,
            help: $help,
            setter: |cfg: &mut Tuning, raw: &str| {
                let steps: usize = raw
                    .parse()
                    .map_err(|_| format!("{} expects a step count, or 0 to disable", $key))?;
                cfg.tiers.$field = (steps > 0).then_some(steps);
                Ok(())
            },
            shower: |cfg: &Tuning| cfg.tiers.$field.unwrap_or(0).to_string(),
        }
    };
}

pub const KNOBS: &[Knob] = &[
    // world
    knob!("particles", "particles per rollout", world.particles, usize),
    knob!("anchors", "continuous-trait anchors", world.anchors, usize),
    knob!("radius", "interaction scale", world.radius, f32),
    knob!("rate", "integration rate; dt is 1/rate", world.rate, f32),
    knob!("shells", "Gaussian kernel shells per pair", world.shells, usize),
    knob!("bumps", "Gaussian growth bumps per pair", world.bumps, usize),
    knob!("evaluation_seed", "rollout initialization seed", world.seed, u64),
    // discovery
    knob!("generations", "search generations", discovery.generations, usize),
    knob!("steps", "discovery steps per rollout", discovery.steps, usize),
    knob!("batch", "genomes per generation", discovery.batch, usize),
    knob!("initial", "random bootstrap genomes", discovery.initial, usize),
    knob!("capacity", "archive capacity", discovery.capacity, usize),
    knob!("seed", "search RNG seed", discovery.seed, u64),
    knob!("promotion_budget", "candidates queued for persistence", discovery.promotion_budget, usize),
    knob!("stall_window", "generations of flat novelty that count as a stall", discovery.stall_window, usize),
    knob!("mutation_iso", "isotropic mutation scale", discovery.mutation.iso, f32),
    knob!("mutation_line", "line-directed mutation scale", discovery.mutation.line, f32),
    knob!("stalled_spread", "mutation spread multiplier while stalled", discovery.mutation.stalled_spread, f32),
    knob!("lane_novelty", "novelty share of a generation", discovery.lanes.novelty, f32),
    knob!("lane_expedition", "expedition share of a generation", discovery.lanes.expedition, f32),
    knob!("stalled_lane_novelty", "novelty share while stalled", discovery.lanes.stalled_novelty, f32),
    knob!("stalled_lane_expedition", "expedition share while stalled", discovery.lanes.stalled_expedition, f32),
    // gates
    knob!("structure_floor", "reject below this structure", gates.structure_floor, f32),
    knob!("structure_ceiling", "reject above this structure", gates.structure_ceiling, f32),
    knob!("variance_floor", "reject below this temporal variance", gates.variance_floor, f32),
    knob!("robustness_floor", "reject below this repair fraction", gates.robustness_floor, f32),
    knob!("heterogeneity_floor", "world gate: local regime contrast", gates.heterogeneity_floor, f32),
    knob!("connected_fraction_floor", "world gate: connected material and void", gates.connected_fraction_floor, f32),
    knob!("mobility_floor", "reject below this mobility", gates.mobility_floor, f32),
    knob!("dispersed_structure", "watchdog dispersion threshold", gates.dispersed_structure, f32),
    knob!("collapsed_structure", "watchdog collapse threshold", gates.collapsed_structure, f32),
    knob!("sustained_windows", "watchdog windows before early stop", gates.sustained_windows, usize),
    // repair
    knob!("repair_probes", "damage probes per rollout", repair.probes, usize),
    knob!("repair_shake", "damage displacement, in kernel cutoffs", repair.shake, f32),
    knob!("repair_recovery", "damage fraction that must close", repair.recovery, f32),
    knob!("repair_min_damage", "smallest damage that proves anything", repair.min_effective_damage, f32),
    // tiers
    tier_knob!("persistence_steps", "persistence rollout steps, 0 to disable", persistence_steps),
    tier_knob!("certification_steps", "certification rollout steps, 0 to disable", certification_steps),
    // learning
    knob!("minimum_labeled_rows", "rows before the scheduler is fit", learning.minimum_labeled_rows, usize),
    knob!("model_seed", "scheduler split seed", learning.model_seed, u64),
    knob!("quota_high_survival", "scheduled slots for likely survivors", learning.quotas.high_survival, usize),
    knob!("quota_high_uncertainty", "scheduled slots for uncertain candidates", learning.quotas.high_uncertainty, usize),
    knob!("quota_uniform", "scheduled slots spread across neighborhoods", learning.quotas.uniform_neighborhood, usize),
];

pub fn knob(key: &str) -> Option<&'static Knob> {
    KNOBS.iter().find(|knob| knob.key == key)
}

/// Apply one `key=value` setting.
pub fn apply(tuning: &mut Tuning, key: &str, value: &str) -> Result<(), String> {
    knob(key)
        .ok_or_else(|| format!("unknown option {key:?}"))?
        .set(tuning, value)
}

/// Generated help: every knob, its documentation, and the value it currently holds.
pub fn help(tuning: &Tuning) -> String {
    let width = KNOBS.iter().map(|knob| knob.key.len()).max().unwrap_or(0);
    KNOBS
        .iter()
        .map(|knob| {
            format!(
                "  {:width$} = {:<10} {}\n",
                knob.key,
                knob.show(tuning),
                knob.help
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_knob_round_trips_through_its_own_text() {
        let mut tuning = Tuning::default();
        for knob in KNOBS {
            let shown = knob.show(&tuning);
            knob.set(&mut tuning, &shown)
                .unwrap_or_else(|problem| panic!("{}: {problem}", knob.key));
            assert_eq!(knob.show(&tuning), shown, "{} did not round trip", knob.key);
        }
    }

    #[test]
    fn knob_keys_are_unique() {
        let mut keys: Vec<&str> = KNOBS.iter().map(|knob| knob.key).collect();
        keys.sort_unstable();
        let count = keys.len();
        keys.dedup();
        assert_eq!(keys.len(), count, "duplicate knob key");
    }

    /// Measurement and judgment change the digest; how much searching happens does not.
    #[test]
    fn the_digest_separates_evidence_regimes_from_search_effort() {
        let base = Tuning::default();
        for key in ["batch", "generations", "seed", "capacity", "promotion_budget"] {
            let mut other = base.clone();
            apply(&mut other, key, "99").unwrap();
            assert_eq!(base.digest(), other.digest(), "{key} should not split evidence");
        }
        for key in ["robustness_floor", "mutation_iso", "repair_shake", "lane_novelty"] {
            let mut other = base.clone();
            apply(&mut other, key, "0.42").unwrap();
            assert_ne!(base.digest(), other.digest(), "{key} must split evidence");
        }
    }

    #[test]
    fn lane_counts_always_sum_to_the_batch() {
        let lanes = Lanes::default();
        for batch in 0..64 {
            for stalled in [false, true] {
                let counts = lanes.counts(batch, stalled);
                assert_eq!(
                    counts.novelty + counts.expedition + counts.random,
                    batch,
                    "batch {batch} stalled {stalled}"
                );
            }
        }
    }
}
