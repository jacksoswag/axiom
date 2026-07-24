//! Headless three-dimensional discovery search.
//!
//! Every simulation setting is a knob in `tuner::tuning::KNOBS`, so this file parses
//! `key=value` generically and never names one. Only output paths are handled here, because
//! they say where results go rather than what a run does.

use axiom::tuner::campaign::{run, validate, CampaignPersistence};
use axiom::tuner::state::CampaignState;
use axiom::tuner::tuning::{self, Tuning};
use axiom::tuner::viability::Rejection;
use std::process::ExitCode;

const PREAMBLE: &str = "\
usage: train [key=value ...]

AXIOM fixes the product boundary to three dimensions. Coordination and trait-distribution
logits are genome genes; everything below is an experiment setting. Values shown are the
defaults. Any setting that changes what an evaluation means also splits campaign evidence,
so a resumed campaign must keep them identical.

paths
  out=PATH           archive output                        (default archive.txt)
  state=PATH         durable campaign evidence             (default campaign.state)
  preset_dir=PATH    where certified recipes are written   (default presets)

settings
";

struct Paths {
    out: String,
    state: String,
    presets: String,
}

impl Default for Paths {
    fn default() -> Self {
        Paths {
            out: "archive.txt".into(),
            state: "campaign.state".into(),
            presets: "presets".into(),
        }
    }
}

fn usage() -> String {
    format!("{PREAMBLE}{}", tuning::help(&Tuning::default()))
}

fn main() -> ExitCode {
    if std::env::args()
        .skip(1)
        .any(|argument| matches!(argument.as_str(), "-h" | "--help" | "help"))
    {
        print!("{}", usage());
        return ExitCode::SUCCESS;
    }

    let mut tuning = Tuning::default();
    let mut paths = Paths::default();
    for argument in std::env::args().skip(1) {
        let Some((key, value)) = argument.split_once('=') else {
            eprintln!("{}", usage());
            return ExitCode::FAILURE;
        };
        let outcome = match key {
            "out" => {
                paths.out = value.to_owned();
                Ok(())
            }
            "state" => {
                paths.state = value.to_owned();
                Ok(())
            }
            "preset_dir" => {
                paths.presets = value.to_owned();
                Ok(())
            }
            _ => tuning::apply(&mut tuning, key, value),
        };
        if let Err(problem) = outcome {
            eprintln!("{problem}\n\n{}", usage());
            return ExitCode::FAILURE;
        }
    }
    if let Err(problem) = check(&tuning) {
        eprintln!("{problem}\n\n{}", usage());
        return ExitCode::FAILURE;
    }

    println!(
        "3-D, {} particles, {} anchors, {} genes, {} discovery rollouts of {} steps",
        tuning.world.particle_count,
        tuning.world.anchor_count,
        tuning.world.gene_len(),
        tuning.discovery.initial + 1 + tuning.discovery.generations * tuning.discovery.batch,
        tuning.discovery.steps
    );

    let mut on_generation = |generation,
                             archive: &axiom::tuner::archive::Archive,
                             report: &axiom::tuner::search::Report| {
        println!(
            "gen {:>4} archive {:>5} viable {:>5.1}% novelty {:.4} neighborhoods {:>4} promotions {:>4}{}\n           lanes n/e/r {}/{}/{} fallback {} rejected [{}]",
            generation + 1, archive.len(), report.viable_fraction() * 100.0,
            report.stall.median_current_novelty, report.stall.occupied_neighborhoods,
            report.promotions_queued, if report.stall.stalled { " STALLED" } else { "" },
            report.lanes.novelty, report.lanes.expedition, report.lanes.random, report.fallback_births,
            Rejection::ALL.iter().map(|reason| format!("{} {}", reason.label(), report.count(*reason))).collect::<Vec<_>>().join("  "),
        );
    };

    let state_path = std::path::Path::new(&paths.state);
    let mut state = if state_path.exists() {
        match CampaignState::load(state_path) {
            Ok(state) => state,
            Err(problem) => {
                eprintln!("could not load {}: {problem}", paths.state);
                return ExitCode::FAILURE;
            }
        }
    } else {
        CampaignState::default()
    };
    let persistence = CampaignPersistence {
        archive_path: Some(paths.out.clone().into()),
        state_path: Some(paths.state.clone().into()),
        preset_dir: Some(paths.presets.clone().into()),
    };

    let result = match run(&tuning, &persistence, &mut state, &mut on_generation) {
        Ok(result) => result,
        Err(problem) => {
            eprintln!("campaign failed: {problem}");
            return ExitCode::FAILURE;
        }
    };
    println!(
        "wrote {} archive entries and {} persistence candidates to {}",
        result.search.archive.len(),
        result.search.promotions.records().len(),
        paths.out
    );
    if !result.certifications.is_empty() {
        println!(
            "wrote {} certified checkpoint recipes to {}",
            result.certifications.len(),
            paths.presets
        );
    }
    ExitCode::SUCCESS
}

/// Reject settings that would produce no work at all, then hand the evidence contract to the
/// campaign, which owns what a tier is allowed to mean.
fn check(tuning: &Tuning) -> Result<(), String> {
    for (name, value) in [
        ("generations", tuning.discovery.generations),
        ("batch", tuning.discovery.batch),
        ("capacity", tuning.discovery.capacity),
        ("particles", tuning.world.particle_count),
        ("shells", tuning.world.shells),
        ("bumps", tuning.world.bumps),
        ("repair_probes", tuning.repair.probes),
    ] {
        if value == 0 {
            return Err(format!("{name} must be at least 1"));
        }
    }
    if tuning.world.anchor_count < 2 {
        return Err("anchors must be at least 2".into());
    }
    for (name, value) in [
        ("radius", tuning.world.radius),
        ("rate", tuning.world.rate),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(format!("{name} must be positive"));
        }
    }
    validate(tuning).map_err(|problem| problem.to_string())
}
