//! Render a deterministic CPU reference frame for visual inspection.

use axiom::engine::sim::Sim;
use axiom::tuner::archive::Archive;
use axiom::tuner::genome::Caps;
use axiom::viewer::{write_reference_snapshot, MaterialRecipe};
use std::path::Path;
use std::process::ExitCode;

fn main() -> ExitCode {
    match render() {
        Ok(path) => { println!("wrote {path}"); ExitCode::SUCCESS }
        Err(problem) => { eprintln!("{problem}"); ExitCode::FAILURE }
    }
}

fn render() -> Result<String, String> {
    let mut output = String::from("axiom-snapshot.ppm");
    let mut archive_path = None;
    let mut entry = 0usize;
    let mut steps = 1_500usize;
    let mut particle_override = None;
    let mut seed = 1u64;
    let mut width = 960usize;
    let mut height = 540usize;
    let mut resolution = None;
    let mut support = None;
    let mut iso = None;
    let mut absorption = None;
    for argument in std::env::args().skip(1) {
        let (key, value) = argument.split_once('=')
            .ok_or_else(|| format!("expected key=value, got {argument:?}"))?;
        match key {
            "out" => output = value.to_owned(),
            "archive" => archive_path = Some(value.to_owned()),
            "entry" => entry = parse(key, value)?,
            "steps" => steps = parse(key, value)?,
            "particles" => particle_override = Some(parse(key, value)?),
            "seed" => seed = parse(key, value)?,
            "width" => width = parse(key, value)?,
            "height" => height = parse(key, value)?,
            "resolution" => resolution = Some(parse(key, value)?),
            "support" => support = Some(parse(key, value)?),
            "iso" => iso = Some(parse(key, value)?),
            "absorption" => absorption = Some(parse(key, value)?),
            _ => return Err(format!("unknown option {key:?}")),
        }
    }

    let params = if let Some(path) = archive_path {
        let text = std::fs::read_to_string(&path).map_err(|problem| format!("could not read {path}: {problem}"))?;
        let (archive, mut tuning) = Archive::from_text(&text)?;
        if let Some(particle_count) = particle_override { tuning.world.particle_count = particle_count; }
        let selected = archive.entries().get(entry).ok_or_else(|| format!("archive has no entry {entry}"))?;
        tuning.world.params(&selected.genome, &tuning.world.probe())
    } else {
        let caps = Caps { particle_count: particle_override.unwrap_or(320), seed, ..Caps::default() };
        let probe = caps.probe();
        caps.params(&caps.default_genome(&probe), &probe)
    };
    let derived = MaterialRecipe::for_world(params.box_len, params.particle_count);
    let recipe = MaterialRecipe {
        resolution: resolution.unwrap_or(derived.resolution),
        support: support.unwrap_or(derived.support),
        iso: iso.unwrap_or(derived.iso),
        absorption: absorption.unwrap_or(derived.absorption),
    };
    let mut sim = Sim::new(&params);
    sim.run(steps as u64);
    write_reference_snapshot(Path::new(&output), &sim, &recipe, width, height)?;
    Ok(output)
}

fn parse<T: std::str::FromStr>(key: &str, value: &str) -> Result<T, String> {
    value.parse().map_err(|_| format!("bad value for {key}: {value:?}"))
}
