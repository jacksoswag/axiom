//! Render a deterministic CPU reference frame for visual inspection.

use axiom::tuner::archive::Archive;
use axiom::viewer::{write_reference_snapshot, MaterialRecipe};
use axiom::engine::world::World;
use axiom::engine::genome::Params;
use std::path::Path;
use std::process::ExitCode;

fn main() -> ExitCode {
    match render() {
        Ok(path) => {
            println!("wrote {path}");
            ExitCode::SUCCESS
        }
        Err(problem) => {
            eprintln!("{problem}");
            ExitCode::FAILURE
        }
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
        let (key, value) = argument
            .split_once('=')
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

    let (mut world, derived) = if let Some(path) = archive_path {
        let text = std::fs::read_to_string(&path)
            .map_err(|problem| format!("could not read {path}: {problem}"))?;
        let (archive, mut tuning) = Archive::from_text(&text)?;
        if let Some(particles) = particle_override {
            tuning.world.particles = particles;
        }
        let selected = archive
            .entries()
            .get(entry)
            .ok_or_else(|| format!("archive has no entry {entry}"))?;
        let (params, interactions) = tuning.world.resolve(&selected.genome);
        let geometry = params.geometry();
        (
            World::with_geometry(&params, &geometry, interactions),
            MaterialRecipe::for_world(geometry.extent, params.particles),
        )
    } else {
        let params = Params {
            particles: particle_override.unwrap_or(320),
            seed,
            ..Default::default()
        };
        let geometry = params.geometry();
        let genome = params
            .layout()
            .default_genome(params.radius, geometry.extent);
        (
            World::with_geometry(&params, &geometry, &genome),
            MaterialRecipe::for_world(geometry.extent, params.particles),
        )
    };
    let recipe = MaterialRecipe {
        resolution: resolution.unwrap_or(derived.resolution),
        support: support.unwrap_or(derived.support),
        iso: iso.unwrap_or(derived.iso),
        absorption: absorption.unwrap_or(derived.absorption),
    };
    for _ in 0..steps {
        world.step();
    }
    write_reference_snapshot(Path::new(&output), &world, &recipe, width, height)?;
    Ok(output)
}

fn parse<T: std::str::FromStr>(key: &str, value: &str) -> Result<T, String> {
    value
        .parse()
        .map_err(|_| format!("bad value for {key}: {value:?}"))
}
