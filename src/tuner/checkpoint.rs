//! Durable, cache-free world checkpoints.
//!
//! An archive entry records a discoverable rule. A checkpoint records one particular place in
//! that rule's history. Render fields, meshes, textures, and spatial indexes are intentionally
//! absent: they are deterministic caches rebuilt after restore.

use crate::util::Fnv;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use crate::engine::genome::Caps;
use crate::engine::world::World;
use crate::render_recipe::RenderRecipe;
use crate::tuner::density;

pub const SIMULATOR_VERSION: u32 = 2;
pub const GENOME_LAYOUT_VERSION: u32 = 2;
pub const MANIFEST_VERSION: u32 = 3;
pub const STATE_VERSION: u32 = 1;
const MANIFEST_MAGIC: [u8; 8] = *b"AXIOMWM1";
const STATE_MAGIC: [u8; 8] = *b"AXIOMWS1";
const MAX_STRING: usize = 1 << 20;
const MAX_VALUES: usize = 1 << 28;

/// Immutable metadata shared by every checkpoint branch of one world.
#[derive(Clone, Debug, PartialEq)]
pub struct WorldManifest {
    pub world_id: String,
    pub simulator_version: u32,
    pub descriptor_version: u32,
    pub genome_layout_version: u32,
    pub render_recipe_version: u32,
    pub render_recipe: RenderRecipe,
    pub seed: u64,
    pub tick: u64,
    pub dimensions: usize,
    pub particle_count: usize,
    pub coordination: f32,
    pub radius: f32,
    pub rate: f32,
    pub dt: f32,
    pub extent: f32,
    pub softening: f32,
    pub anchors: usize,
    pub shells: usize,
    pub bumps: usize,
    pub trait_logits: Vec<f32>,
    /// Measured norms are authoritative so restoring never depends on a calibration probe.
    pub interaction_norms: Vec<f32>,
    /// Full capped genome, including coordination and initial trait-density logits.
    pub genome: Vec<f32>,
    pub latest_checkpoint_id: String,
}

/// Authoritative dynamic state. Positions and traits retain their f32 bit patterns.
#[derive(Clone, Debug, PartialEq)]
pub struct WorldState {
    pub checkpoint_id: String,
    pub parent_checkpoint_id: Option<String>,
    pub world_id: String,
    pub tick: u64,
    pub dimensions: usize,
    pub positions: Vec<f32>,
    pub traits: Vec<f32>,
    pub genome: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SnapshotMetadata {
    pub world_id: String,
    pub checkpoint_id: String,
    pub parent_checkpoint_id: Option<String>,
    pub simulator_version: u32,
    pub descriptor_version: u32,
    pub genome_layout_version: u32,
    pub render_recipe_version: u32,
    pub render_recipe: RenderRecipe,
}

#[derive(Debug)]
pub enum Error {
    Io(io::Error),
    InvalidMagic { kind: &'static str },
    UnsupportedVersion { kind: &'static str, found: u32 },
    Truncated { field: &'static str },
    InvalidLength { field: &'static str, length: usize },
    NonFinite { field: &'static str },
    TraitOutOfRange,
    InvalidDimensions,
    InvalidIdentifier { field: &'static str },
    InvalidGenomeLayout { expected: usize, actual: usize },
    ParticleCountMismatch { positions: usize, traits: usize },
    ManifestStateMismatch { field: &'static str },
    CorruptChecksum { expected: u64, actual: u64 },
    ImmutableCheckpoint { path: PathBuf },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(problem) => write!(f, "checkpoint I/O: {problem}"),
            Error::InvalidMagic { kind } => write!(f, "not an AXIOM {kind}"),
            Error::UnsupportedVersion { kind, found } => {
                write!(f, "unsupported {kind} version {found}")
            }
            Error::Truncated { field } => write!(f, "truncated {field}"),
            Error::InvalidLength { field, length } => write!(f, "invalid {field} length {length}"),
            Error::NonFinite { field } => write!(f, "non-finite {field}"),
            Error::TraitOutOfRange => write!(f, "checkpoint trait lies outside 0..=1"),
            Error::InvalidDimensions => write!(f, "checkpoint dimensions must be three"),
            Error::InvalidIdentifier { field } => write!(f, "invalid checkpoint {field}"),
            Error::InvalidGenomeLayout { expected, actual } => {
                write!(
                    f,
                    "checkpoint genome has {actual} genes, expected {expected}"
                )
            }
            Error::ParticleCountMismatch { positions, traits } => {
                write!(
                    f,
                    "particle state has {positions} positions but {traits} traits"
                )
            }
            Error::ManifestStateMismatch { field } => {
                write!(f, "manifest and state disagree on {field}")
            }
            Error::CorruptChecksum { expected, actual } => {
                write!(
                    f,
                    "checkpoint checksum {actual:016x} does not match {expected:016x}"
                )
            }
            Error::ImmutableCheckpoint { path } => {
                write!(f, "checkpoint already exists: {}", path.display())
            }
        }
    }
}

impl std::error::Error for Error {}

impl From<io::Error> for Error {
    fn from(problem: io::Error) -> Self {
        Error::Io(problem)
    }
}

impl WorldManifest {
    pub fn validate(&self) -> Result<(), Error> {
        if self.dimensions == 0 {
            return Err(Error::InvalidDimensions);
        }
        for (kind, found, expected) in [
            ("simulator", self.simulator_version, SIMULATOR_VERSION),
            (
                "descriptor",
                self.descriptor_version,
                crate::tuner::metrics::DESCRIPTOR_VERSION,
            ),
            (
                "genome layout",
                self.genome_layout_version,
                GENOME_LAYOUT_VERSION,
            ),
        ] {
            if found != expected {
                return Err(Error::UnsupportedVersion { kind, found });
            }
        }
        if self.render_recipe_version != crate::render_recipe::VERSION {
            return Err(Error::UnsupportedVersion {
                kind: "render recipe",
                found: self.render_recipe_version,
            });
        }
        if !self.render_recipe.valid() {
            return Err(Error::InvalidLength {
                field: "render recipe",
                length: self.render_recipe.resolution,
            });
        }
        valid_identifier("world id", &self.world_id)?;
        valid_identifier("latest checkpoint id", &self.latest_checkpoint_id)?;
        if self.particle_count == 0 || self.anchors < 2 || self.shells == 0 || self.bumps == 0 {
            return Err(Error::InvalidLength {
                field: "particle count, anchors, shells, or bumps",
                length: self.particle_count,
            });
        }
        finite(
            "manifest scalar",
            &[
                self.coordination,
                self.radius,
                self.rate,
                self.dt,
                self.extent,
                self.softening,
            ],
        )?;
        if self.radius <= 0.0
            || self.rate <= 0.0
            || self.dt <= 0.0
            || self.extent <= 0.0
            || self.softening <= 0.0
        {
            return Err(Error::InvalidLength {
                field: "positive manifest scalar",
                length: 0,
            });
        }
        if self.render_recipe.support >= self.extent * 0.5 {
            return Err(Error::InvalidLength {
                field: "render support",
                length: self.render_recipe.resolution,
            });
        }
        if self.trait_logits.len() != self.anchors {
            return Err(Error::InvalidLength {
                field: "trait logits",
                length: self.trait_logits.len(),
            });
        }
        finite("manifest trait logits", &self.trait_logits)?;
        let interactions = self
            .anchors
            .checked_mul(self.anchors)
            .ok_or(Error::InvalidLength {
                field: "anchor interactions",
                length: self.anchors,
            })?;
        let genes_per_interaction = 3usize
            .checked_mul(
                self.shells
                    .checked_add(self.bumps)
                    .ok_or(Error::InvalidLength {
                        field: "interaction shape",
                        length: usize::MAX,
                    })?,
            )
            .and_then(|value| value.checked_add(1))
            .ok_or(Error::InvalidLength {
                field: "interaction shape",
                length: usize::MAX,
            })?;
        let expected = 1usize
            .checked_add(self.anchors)
            .and_then(|value| value.checked_add(interactions.checked_mul(genes_per_interaction)?))
            .ok_or(Error::InvalidLength {
                field: "genome layout",
                length: usize::MAX,
            })?;
        if self.genome.len() != expected {
            return Err(Error::InvalidGenomeLayout {
                expected,
                actual: self.genome.len(),
            });
        }
        if self.interaction_norms.len() != interactions {
            return Err(Error::InvalidLength {
                field: "interaction norms",
                length: self.interaction_norms.len(),
            });
        }
        finite("manifest genome", &self.genome)?;
        finite("interaction norms", &self.interaction_norms)?;
        if self.interaction_norms.iter().any(|norm| *norm <= 0.0) {
            return Err(Error::InvalidLength {
                field: "positive interaction norm",
                length: 0,
            });
        }
        if self.genome[0].to_bits() != self.coordination.to_bits()
            || self.genome[1..=self.anchors]
                .iter()
                .map(|value| value.to_bits())
                .ne(self.trait_logits.iter().map(|value| value.to_bits()))
        {
            return Err(Error::ManifestStateMismatch {
                field: "resolved world genes",
            });
        }
        Ok(())
    }
}

impl WorldState {
    pub fn validate(&self) -> Result<(), Error> {
        if self.dimensions == 0 {
            return Err(Error::InvalidDimensions);
        }
        valid_identifier("checkpoint id", &self.checkpoint_id)?;
        valid_identifier("world id", &self.world_id)?;
        if let Some(parent) = &self.parent_checkpoint_id {
            valid_identifier("parent checkpoint id", parent)?;
        }
        if !self.positions.len().is_multiple_of(self.dimensions) {
            return Err(Error::InvalidLength {
                field: "positions",
                length: self.positions.len(),
            });
        }
        let particles = self.positions.len() / self.dimensions;
        if self.traits.len() != particles {
            return Err(Error::ParticleCountMismatch {
                positions: particles,
                traits: self.traits.len(),
            });
        }
        if self.genome.is_empty() {
            return Err(Error::InvalidLength {
                field: "genome",
                length: 0,
            });
        }
        finite("positions", &self.positions)?;
        finite("traits", &self.traits)?;
        if self
            .traits
            .iter()
            .any(|trait_value| !(0.0..=1.0).contains(trait_value))
        {
            return Err(Error::TraitOutOfRange);
        }
        finite("genome", &self.genome)
    }
}

pub fn save_manifest(path: &Path, manifest: &WorldManifest) -> Result<(), Error> {
    manifest.validate()?;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&MANIFEST_MAGIC);
    write_u32(&mut bytes, MANIFEST_VERSION);
    write_string(&mut bytes, &manifest.world_id)?;
    for value in [
        manifest.simulator_version,
        manifest.descriptor_version,
        manifest.genome_layout_version,
        manifest.render_recipe_version,
    ] {
        write_u32(&mut bytes, value);
    }
    write_u64(&mut bytes, manifest.render_recipe.resolution as u64);
    for value in [
        manifest.render_recipe.support,
        manifest.render_recipe.iso,
        manifest.render_recipe.absorption,
    ] {
        write_f32(&mut bytes, value);
    }
    write_u64(&mut bytes, manifest.seed);
    write_u64(&mut bytes, manifest.tick);
    write_u64(&mut bytes, manifest.dimensions as u64);
    write_u64(&mut bytes, manifest.particle_count as u64);
    for value in [
        manifest.coordination,
        manifest.radius,
        manifest.rate,
        manifest.dt,
        manifest.extent,
        manifest.softening,
    ] {
        write_f32(&mut bytes, value);
    }
    write_u64(&mut bytes, manifest.anchors as u64);
    write_u64(&mut bytes, manifest.shells as u64);
    write_u64(&mut bytes, manifest.bumps as u64);
    write_f32s(&mut bytes, &manifest.trait_logits)?;
    write_f32s(&mut bytes, &manifest.interaction_norms)?;
    write_f32s(&mut bytes, &manifest.genome)?;
    write_string(&mut bytes, &manifest.latest_checkpoint_id)?;
    let manifest_checksum = checksum(&bytes);
    write_u64(&mut bytes, manifest_checksum);
    write_atomic(path, &bytes)
}

pub fn load_manifest(path: &Path) -> Result<WorldManifest, Error> {
    let mut bytes = Vec::new();
    File::open(path)?.read_to_end(&mut bytes)?;
    if bytes.len() < 8 {
        return Err(Error::Truncated {
            field: "world manifest",
        });
    }
    let checksum_at = bytes
        .len()
        .checked_sub(8)
        .ok_or(Error::Truncated { field: "checksum" })?;
    let expected = u64::from_le_bytes(bytes[checksum_at..].try_into().expect("eight bytes"));
    let actual = checksum(&bytes[..checksum_at]);
    if expected != actual {
        return Err(Error::CorruptChecksum { expected, actual });
    }
    let mut reader = Reader::new(&bytes[..checksum_at]);
    reader.magic(MANIFEST_MAGIC, "world manifest")?;
    reader.version(MANIFEST_VERSION, "world manifest")?;
    let world_id = reader.string("world id")?;
    let simulator_version = reader.u32("simulator version")?;
    let descriptor_version = reader.u32("descriptor version")?;
    let genome_layout_version = reader.u32("genome layout version")?;
    let render_recipe_version = reader.u32("render recipe version")?;
    let render_recipe = RenderRecipe {
        resolution: reader.usize("render resolution")?,
        support: reader.f32("render support")?,
        iso: reader.f32("render iso")?,
        absorption: reader.f32("render absorption")?,
    };
    let manifest = WorldManifest {
        world_id,
        simulator_version,
        descriptor_version,
        genome_layout_version,
        render_recipe_version,
        render_recipe,
        seed: reader.u64("seed")?,
        tick: reader.u64("tick")?,
        dimensions: reader.usize("dimensions")?,
        particle_count: reader.usize("particle count")?,
        coordination: reader.f32("coordination")?,
        radius: reader.f32("radius")?,
        rate: reader.f32("time scale")?,
        dt: reader.f32("dt")?,
        extent: reader.f32("extent")?,
        softening: reader.f32("softening")?,
        anchors: reader.usize("anchors")?,
        shells: reader.usize("shells")?,
        bumps: reader.usize("bumps")?,
        trait_logits: reader.f32s("trait logits")?,
        interaction_norms: reader.f32s("interaction norms")?,
        genome: reader.f32s("genome")?,
        latest_checkpoint_id: reader.string("latest checkpoint id")?,
    };
    reader.finished("world manifest")?;
    manifest.validate()?;
    Ok(manifest)
}

pub fn save_state(path: &Path, state: &WorldState) -> Result<(), Error> {
    state.validate()?;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&STATE_MAGIC);
    write_u32(&mut bytes, STATE_VERSION);
    write_string(&mut bytes, &state.checkpoint_id)?;
    match &state.parent_checkpoint_id {
        Some(id) => write_string(&mut bytes, id)?,
        None => write_u32(&mut bytes, u32::MAX),
    }
    write_string(&mut bytes, &state.world_id)?;
    write_u64(&mut bytes, state.tick);
    write_u64(&mut bytes, state.dimensions as u64);
    write_f32s(&mut bytes, &state.positions)?;
    write_f32s(&mut bytes, &state.traits)?;
    write_f32s(&mut bytes, &state.genome)?;
    let state_checksum = checksum(&bytes);
    write_u64(&mut bytes, state_checksum);
    write_immutable(path, &bytes)
}

pub fn load_state(path: &Path) -> Result<WorldState, Error> {
    let mut bytes = Vec::new();
    File::open(path)?.read_to_end(&mut bytes)?;
    if bytes.len() < 8 {
        return Err(Error::Truncated {
            field: "checkpoint",
        });
    }
    let checksum_at = bytes
        .len()
        .checked_sub(8)
        .ok_or(Error::Truncated { field: "checksum" })?;
    let expected = u64::from_le_bytes(bytes[checksum_at..].try_into().expect("eight bytes"));
    let actual = checksum(&bytes[..checksum_at]);
    if expected != actual {
        return Err(Error::CorruptChecksum { expected, actual });
    }
    let mut reader = Reader::new(&bytes[..checksum_at]);
    reader.magic(STATE_MAGIC, "world state")?;
    reader.version(STATE_VERSION, "world state")?;
    let checkpoint_id = reader.string("checkpoint id")?;
    let parent_checkpoint_id = reader.optional_string("parent checkpoint id")?;
    let state = WorldState {
        checkpoint_id,
        parent_checkpoint_id,
        world_id: reader.string("world id")?,
        tick: reader.u64("tick")?,
        dimensions: reader.usize("dimensions")?,
        positions: reader.f32s("positions")?,
        traits: reader.f32s("traits")?,
        genome: reader.f32s("genome")?,
    };
    reader.finished("world state")?;
    state.validate()?;
    Ok(state)
}

/// Stores both durable records under one world directory. Each file is independently atomic;
/// the manifest points at a fully written state only after the state save succeeds.
pub fn save_world(
    root: &Path,
    manifest: &WorldManifest,
    state: &WorldState,
) -> Result<(PathBuf, PathBuf), Error> {
    validate_pair(manifest, state)?;
    let world_root = root.join(&manifest.world_id);
    let state_path = world_root
        .join("checkpoints")
        .join(format!("{}.checkpoint", state.checkpoint_id));
    let manifest_path = world_root.join("world.manifest");
    match save_state(&state_path, state) {
        Ok(()) => {}
        Err(Error::ImmutableCheckpoint { .. }) => {
            let existing = load_state(&state_path)?;
            if state_checksum(&existing)? != state_checksum(state)? {
                return Err(Error::ImmutableCheckpoint { path: state_path });
            }
        }
        Err(problem) => return Err(problem),
    }
    save_manifest(&manifest_path, manifest)?;
    Ok((manifest_path, state_path))
}

pub fn load_world(
    manifest_path: &Path,
    state_path: &Path,
) -> Result<(WorldManifest, WorldState), Error> {
    let manifest = load_manifest(manifest_path)?;
    let state = load_state(state_path)?;
    validate_pair(&manifest, &state)?;
    Ok((manifest, state))
}

pub fn validate_pair(manifest: &WorldManifest, state: &WorldState) -> Result<(), Error> {
    manifest.validate()?;
    state.validate()?;
    if manifest.world_id != state.world_id || manifest.latest_checkpoint_id != state.checkpoint_id {
        return Err(Error::ManifestStateMismatch {
            field: "world or checkpoint id",
        });
    }
    if manifest.tick != state.tick {
        return Err(Error::ManifestStateMismatch { field: "tick" });
    }
    if manifest.particle_count != state.traits.len() {
        return Err(Error::ManifestStateMismatch {
            field: "particle count",
        });
    }
    if manifest
        .genome
        .iter()
        .map(|value| value.to_bits())
        .ne(state.genome.iter().map(|value| value.to_bits()))
    {
        return Err(Error::ManifestStateMismatch { field: "genome" });
    }
    Ok(())
}

/// Capture the exact dynamic state of a world alongside its complete resolved recipe.
pub fn snapshot_world(
    metadata: SnapshotMetadata,
    caps: &Caps,
    full_genome: Vec<f32>,
    world: &World,
) -> Result<(WorldManifest, WorldState), Error> {
    if full_genome.len() != caps.genome_len() {
        return Err(Error::InvalidGenomeLayout {
            expected: caps.genome_len(),
            actual: full_genome.len(),
        });
    }
    let (params, interaction_genome) = caps.resolve(&full_genome);
    let expected_extent = density::bound_len_for(&params);
    if !world.matches_recipe(&params, expected_extent, interaction_genome) {
        return Err(Error::ManifestStateMismatch {
            field: "world recipe",
        });
    }
    let manifest = WorldManifest {
        world_id: metadata.world_id.clone(),
        simulator_version: metadata.simulator_version,
        descriptor_version: metadata.descriptor_version,
        genome_layout_version: metadata.genome_layout_version,
        render_recipe_version: metadata.render_recipe_version,
        render_recipe: metadata.render_recipe,
        seed: params.seed,
        tick: world.tick,
        dimensions: params.dimensions,
        particle_count: world.substrate.traits.len(),
        coordination: params.coordination,
        radius: params.radius,
        rate: params.rate,
        dt: world.dt,
        extent: world.substrate.bound_len,
        softening: world.substrate.softening,
        anchors: params.anchors,
        shells: params.shells,
        bumps: params.bumps,
        trait_logits: params.trait_logits,
        interaction_norms: world.net.pairs.iter().map(|interaction| interaction.norm).collect(),
        genome: full_genome,
        latest_checkpoint_id: metadata.checkpoint_id.clone(),
    };
    debug_assert_eq!(
        manifest.genome.len() - caps.world_genes(),
        caps.layout().genome_len()
    );
    let state = WorldState {
        checkpoint_id: metadata.checkpoint_id,
        parent_checkpoint_id: metadata.parent_checkpoint_id,
        world_id: metadata.world_id,
        tick: world.tick,
        dimensions: params.dimensions,
        positions: world.substrate.positions.clone(),
        traits: world.substrate.traits.clone(),
        genome: manifest.genome.clone(),
    };
    validate_pair(&manifest, &state)?;
    Ok((manifest, state))
}

/// Recreate a cache-free three-dimensional world from saved data. The saved interaction norms
/// are restored after construction, avoiding any platform-dependent calibration drift.
pub fn restore_world(manifest: &WorldManifest, state: &WorldState) -> Result<World, Error> {
    validate_pair(manifest, state)?;
    let caps = Caps {
        particles: manifest.particle_count,
        dimensions: manifest.dimensions,
        anchors: manifest.anchors,
        radius: manifest.radius,
        rate: manifest.rate,
        seed: manifest.seed,
        shells: manifest.shells,
        bumps: manifest.bumps,
    };
    let (params, interaction_genome) = caps.resolve(&manifest.genome);
    if params.coordination.to_bits() != manifest.coordination.to_bits()
        || params
            .trait_logits
            .iter()
            .map(|value| value.to_bits())
            .ne(manifest.trait_logits.iter().map(|value| value.to_bits()))
    {
        return Err(Error::ManifestStateMismatch {
            field: "resolved parameters",
        });
    }
    let mut world = World::from_snapshot(
        &params,
        manifest.extent,
        interaction_genome,
        state.positions.clone(),
        state.traits.clone(),
        state.tick,
    );
    world.dt = manifest.dt;
    // Saved norms are untrusted deserialized data, validated here at the boundary they cross.
    assert_eq!(manifest.interaction_norms.len(), world.net.pairs.len());
    for (interaction, &norm) in world.net.pairs.iter_mut().zip(&manifest.interaction_norms) {
        assert!(norm.is_finite() && norm > 0.0);
        interaction.norm = norm;
    }
    Ok(world)
}

/// Stable checksum of a full checkpoint record. It includes identifiers and recipe state so
/// callers can bind an audit record to the exact serialized checkpoint, independent of files.
pub fn state_checksum(state: &WorldState) -> Result<u64, Error> {
    state.validate()?;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&STATE_MAGIC);
    write_u32(&mut bytes, STATE_VERSION);
    write_string(&mut bytes, &state.checkpoint_id)?;
    match &state.parent_checkpoint_id {
        Some(id) => write_string(&mut bytes, id)?,
        None => write_u32(&mut bytes, u32::MAX),
    }
    write_string(&mut bytes, &state.world_id)?;
    write_u64(&mut bytes, state.tick);
    write_u64(&mut bytes, state.dimensions as u64);
    write_f32s(&mut bytes, &state.positions)?;
    write_f32s(&mut bytes, &state.traits)?;
    write_f32s(&mut bytes, &state.genome)?;
    Ok(checksum(&bytes))
}

fn finite(field: &'static str, values: &[f32]) -> Result<(), Error> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(Error::NonFinite { field })
    }
}

fn valid_identifier(field: &'static str, value: &str) -> Result<(), Error> {
    if value.is_empty()
        || value.len() > MAX_STRING
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        return Err(Error::InvalidIdentifier { field });
    }
    Ok(())
}

/// Publish an immutable file only if its final name does not exist. A hard link provides the
/// no-replace publication step after the temporary sibling is fully flushed.
fn write_immutable(path: &Path, bytes: &[u8]) -> Result<(), Error> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    if path.exists() {
        return Err(Error::ImmutableCheckpoint {
            path: path.to_path_buf(),
        });
    }
    let stem = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("checkpoint");
    for attempt in 0..128u32 {
        let temporary = parent.join(format!(".{stem}.tmp-{}-{attempt}", std::process::id()));
        let mut file = match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
        {
            Ok(file) => file,
            Err(problem) if problem.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(problem) => return Err(Error::Io(problem)),
        };
        if let Err(problem) = file.write_all(bytes).and_then(|_| file.sync_all()) {
            let _ = fs::remove_file(&temporary);
            return Err(Error::Io(problem));
        }
        match fs::hard_link(&temporary, path) {
            Ok(()) => {
                fs::remove_file(&temporary)?;
                return Ok(());
            }
            Err(problem) if problem.kind() == io::ErrorKind::AlreadyExists => {
                let _ = fs::remove_file(&temporary);
                return Err(Error::ImmutableCheckpoint {
                    path: path.to_path_buf(),
                });
            }
            Err(problem) => {
                let _ = fs::remove_file(&temporary);
                return Err(Error::Io(problem));
            }
        }
    }
    Err(Error::Io(io::Error::new(
        io::ErrorKind::AlreadyExists,
        "checkpoint temp name exhausted",
    )))
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), Error> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let stem = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("checkpoint");
    let mut temporary = None;
    for attempt in 0..128u32 {
        let candidate = parent.join(format!(".{stem}.tmp-{}-{attempt}", std::process::id()));
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(mut file) => {
                file.write_all(bytes)?;
                file.sync_all()?;
                temporary = Some(candidate);
                break;
            }
            Err(problem) if problem.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(problem) => return Err(Error::Io(problem)),
        }
    }
    let temporary = temporary.ok_or_else(|| {
        Error::Io(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "checkpoint temp name exhausted",
        ))
    })?;
    if let Err(problem) = fs::rename(&temporary, path) {
        let _ = fs::remove_file(&temporary);
        return Err(Error::Io(problem));
    }
    Ok(())
}

fn write_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}
fn write_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}
fn write_f32(out: &mut Vec<u8>, value: f32) {
    write_u32(out, value.to_bits());
}

fn write_string(out: &mut Vec<u8>, value: &str) -> Result<(), Error> {
    if value.is_empty() || value.len() > MAX_STRING {
        return Err(Error::InvalidLength {
            field: "string",
            length: value.len(),
        });
    }
    write_u32(out, value.len() as u32);
    out.extend_from_slice(value.as_bytes());
    Ok(())
}

fn write_f32s(out: &mut Vec<u8>, values: &[f32]) -> Result<(), Error> {
    if values.is_empty() || values.len() > MAX_VALUES {
        return Err(Error::InvalidLength {
            field: "float vector",
            length: values.len(),
        });
    }
    write_u64(out, values.len() as u64);
    for &value in values {
        write_f32(out, value);
    }
    Ok(())
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Reader { bytes, at: 0 }
    }
    fn take(&mut self, len: usize, field: &'static str) -> Result<&'a [u8], Error> {
        let end = self
            .at
            .checked_add(len)
            .filter(|end| *end <= self.bytes.len())
            .ok_or(Error::Truncated { field })?;
        let slice = &self.bytes[self.at..end];
        self.at = end;
        Ok(slice)
    }
    fn magic(&mut self, expected: [u8; 8], kind: &'static str) -> Result<(), Error> {
        if self.take(8, kind)? != expected {
            return Err(Error::InvalidMagic { kind });
        }
        Ok(())
    }
    fn version(&mut self, expected: u32, kind: &'static str) -> Result<(), Error> {
        let found = self.u32("version")?;
        if found != expected {
            return Err(Error::UnsupportedVersion { kind, found });
        }
        Ok(())
    }
    fn u32(&mut self, field: &'static str) -> Result<u32, Error> {
        Ok(u32::from_le_bytes(
            self.take(4, field)?.try_into().expect("four bytes"),
        ))
    }
    fn u64(&mut self, field: &'static str) -> Result<u64, Error> {
        Ok(u64::from_le_bytes(
            self.take(8, field)?.try_into().expect("eight bytes"),
        ))
    }
    fn usize(&mut self, field: &'static str) -> Result<usize, Error> {
        usize::try_from(self.u64(field)?).map_err(|_| Error::InvalidLength {
            field,
            length: usize::MAX,
        })
    }
    fn f32(&mut self, field: &'static str) -> Result<f32, Error> {
        let value = f32::from_bits(self.u32(field)?);
        if value.is_finite() {
            Ok(value)
        } else {
            Err(Error::NonFinite { field })
        }
    }
    fn string(&mut self, field: &'static str) -> Result<String, Error> {
        let len = self.u32(field)? as usize;
        if len == 0 || len > MAX_STRING {
            return Err(Error::InvalidLength { field, length: len });
        }
        let value = std::str::from_utf8(self.take(len, field)?)
            .map_err(|_| Error::InvalidLength { field, length: len })?;
        Ok(value.to_owned())
    }
    fn optional_string(&mut self, field: &'static str) -> Result<Option<String>, Error> {
        let len = self.u32(field)?;
        if len == u32::MAX {
            return Ok(None);
        }
        let len = len as usize;
        if len == 0 || len > MAX_STRING {
            return Err(Error::InvalidLength { field, length: len });
        }
        let value = std::str::from_utf8(self.take(len, field)?)
            .map_err(|_| Error::InvalidLength { field, length: len })?;
        Ok(Some(value.to_owned()))
    }
    fn f32s(&mut self, field: &'static str) -> Result<Vec<f32>, Error> {
        let len = self.usize(field)?;
        if len == 0 || len > MAX_VALUES {
            return Err(Error::InvalidLength { field, length: len });
        }
        let mut values = Vec::with_capacity(len);
        for _ in 0..len {
            values.push(self.f32(field)?);
        }
        Ok(values)
    }
    fn finished(&self, field: &'static str) -> Result<(), Error> {
        if self.at == self.bytes.len() {
            Ok(())
        } else {
            Err(Error::InvalidLength {
                field,
                length: self.bytes.len() - self.at,
            })
        }
    }
}

fn checksum(bytes: &[u8]) -> u64 {
    let mut hash = Fnv::new();
    hash.bytes(bytes.iter().copied());
    hash.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn state() -> WorldState {
        WorldState {
            checkpoint_id: "tick-42".into(),
            parent_checkpoint_id: Some("tick-12".into()),
            world_id: "reef".into(),
            tick: 42,
            dimensions: 3,
            positions: vec![0.0, -0.0, 5.5, 7.25, 8.0, 9.0],
            traits: vec![0.125, 0.875],
            genome: vec![1.0, -0.0, 3.25],
        }
    }

    fn manifest() -> WorldManifest {
        let mut genome = vec![9.0, 0.0, 0.0];
        genome.extend(std::iter::repeat_n(0.0, 28));
        WorldManifest {
            world_id: "reef".into(),
            simulator_version: SIMULATOR_VERSION,
            descriptor_version: crate::tuner::metrics::DESCRIPTOR_VERSION,
            genome_layout_version: GENOME_LAYOUT_VERSION,
            render_recipe_version: 1,
            render_recipe: RenderRecipe::default(),
            seed: 9,
            tick: 42,
            dimensions: 3,
            particle_count: 2,
            coordination: 9.0,
            radius: 12.0,
            rate: 10.0,
            dt: 0.1,
            extent: 80.0,
            softening: 0.08,
            anchors: 2,
            shells: 1,
            bumps: 1,
            trait_logits: vec![0.0, 0.0],
            interaction_norms: vec![1.0; 4],
            genome,
            latest_checkpoint_id: "tick-42".into(),
        }
    }

    fn temporary(name: &str) -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("axiom-{name}-{}-{stamp}", std::process::id()))
    }

    #[test]
    fn state_round_trip_preserves_particle_trait_tick_and_genome_bits() {
        let path = temporary("round-trip");
        let before = state();
        save_state(&path, &before).unwrap();
        let after = load_state(&path).unwrap();
        let _ = fs::remove_file(path);
        assert_eq!(after.tick, before.tick);
        assert_eq!(
            after
                .positions
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            before
                .positions
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            after.traits.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            before
                .traits
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            after.genome.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            before
                .genome
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn manifest_and_state_write_atomically_as_a_world_pair() {
        let root = temporary("world");
        let manifest = manifest();
        let mut state = state();
        state.genome = manifest.genome.clone();
        let (manifest_path, state_path) = save_world(&root, &manifest, &state).unwrap();
        assert_eq!(load_manifest(&manifest_path).unwrap(), manifest);
        assert_eq!(load_state(&state_path).unwrap(), state);
        assert!(fs::read_dir(&root).unwrap().all(|item| !item
            .unwrap()
            .file_name()
            .to_string_lossy()
            .contains(".tmp-")));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn state_and_manifest_corruption_are_reported_by_checksum() {
        let state_path = temporary("corrupt-state");
        save_state(&state_path, &state()).unwrap();
        let mut bytes = fs::read(&state_path).unwrap();
        bytes[20] ^= 0x40;
        fs::write(&state_path, bytes).unwrap();
        assert!(matches!(
            load_state(&state_path),
            Err(Error::CorruptChecksum { .. })
        ));

        let manifest_path = temporary("corrupt-manifest");
        save_manifest(&manifest_path, &manifest()).unwrap();
        let mut bytes = fs::read(&manifest_path).unwrap();
        bytes[20] ^= 0x40;
        fs::write(&manifest_path, bytes).unwrap();
        assert!(matches!(
            load_manifest(&manifest_path),
            Err(Error::CorruptChecksum { .. })
        ));
        let _ = fs::remove_file(state_path);
        let _ = fs::remove_file(manifest_path);
    }

    fn rewrite_checksum(bytes: &mut [u8]) {
        let at = bytes.len() - 8;
        let value = checksum(&bytes[..at]).to_le_bytes();
        bytes[at..].copy_from_slice(&value);
    }

    fn after_state_header(bytes: &[u8]) -> usize {
        fn skip_string(bytes: &[u8], at: &mut usize) {
            let len = u32::from_le_bytes(bytes[*at..*at + 4].try_into().unwrap()) as usize;
            *at += 4 + len;
        }
        let mut at = 12;
        skip_string(bytes, &mut at);
        skip_string(bytes, &mut at);
        skip_string(bytes, &mut at);
        at + 16
    }

    #[test]
    fn named_format_corruption_errors_are_rejected() {
        let path = temporary("format-corruption");
        save_state(&path, &state()).unwrap();
        let original = fs::read(&path).unwrap();

        let mut bytes = original.clone();
        bytes[0] ^= 1;
        rewrite_checksum(&mut bytes);
        fs::write(&path, &bytes).unwrap();
        assert!(matches!(load_state(&path), Err(Error::InvalidMagic { .. })));

        let mut bytes = original.clone();
        bytes[8..12].copy_from_slice(&999u32.to_le_bytes());
        rewrite_checksum(&mut bytes);
        fs::write(&path, &bytes).unwrap();
        assert!(matches!(
            load_state(&path),
            Err(Error::UnsupportedVersion { .. })
        ));

        let mut bytes = original.clone();
        let positions_length = after_state_header(&bytes);
        bytes[positions_length..positions_length + 8]
            .copy_from_slice(&((MAX_VALUES as u64) + 1).to_le_bytes());
        rewrite_checksum(&mut bytes);
        fs::write(&path, &bytes).unwrap();
        assert!(matches!(
            load_state(&path),
            Err(Error::InvalidLength { .. })
        ));

        let mut bytes = original.clone();
        let first_position = after_state_header(&bytes) + 8;
        bytes[first_position..first_position + 4]
            .copy_from_slice(&f32::NAN.to_bits().to_le_bytes());
        rewrite_checksum(&mut bytes);
        fs::write(&path, &bytes).unwrap();
        assert!(matches!(
            load_state(&path),
            Err(Error::NonFinite { field: "positions" })
        ));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn checkpoint_ids_are_immutable_while_manifest_updates() {
        let root = temporary("immutable");
        let manifest = manifest();
        let mut state = state();
        state.genome = manifest.genome.clone();
        save_world(&root, &manifest, &state).unwrap();
        save_world(&root, &manifest, &state).unwrap();
        let mut conflicting = state.clone();
        conflicting.positions[0] += 1.0;
        assert!(matches!(
            save_world(&root, &manifest, &conflicting),
            Err(Error::ImmutableCheckpoint { .. })
        ));
        let mut newer = manifest.clone();
        newer.render_recipe.iso = 1.4;
        save_manifest(&root.join("reef").join("world.manifest"), &newer).unwrap();
        assert_eq!(
            load_manifest(&root.join("reef").join("world.manifest"))
                .unwrap()
                .render_recipe
                .iso,
            1.4
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn snapshot_restore_has_exact_continuation_for_ten_thousand_steps() {
        let caps = Caps {
            particles: 2,
            dimensions: 3,
            anchors: 2,
            radius: 2.0,
            rate: 10.0,
            seed: 41,
            shells: 1,
            bumps: 1,
        };
        let genome = caps.default_genome(density::widest_bound_len(&caps));
        let (params, interactions) = caps.resolve(&genome);
        let extent = density::bound_len_for(&params);
        let mut original = World::with_extent(&params, extent, interactions);
        for _ in 0..12 {
            original.step();
        }
        let (manifest, state) = snapshot_world(
            SnapshotMetadata {
                world_id: "tiny-reef".into(),
                checkpoint_id: "tick-12".into(),
                parent_checkpoint_id: None,
                simulator_version: SIMULATOR_VERSION,
                descriptor_version: crate::tuner::metrics::DESCRIPTOR_VERSION,
                genome_layout_version: GENOME_LAYOUT_VERSION,
                render_recipe_version: crate::render_recipe::VERSION,
                render_recipe: RenderRecipe {
                    support: extent * 0.1,
                    ..RenderRecipe::default()
                },
            },
            &caps,
            genome,
            &original,
        )
        .unwrap();
        let root = temporary("continuation");
        let (manifest_path, state_path) = save_world(&root, &manifest, &state).unwrap();
        let (loaded_manifest, loaded_state) = load_world(&manifest_path, &state_path).unwrap();
        let mut restored = restore_world(&loaded_manifest, &loaded_state).unwrap();
        assert_eq!(original.state_hash(), restored.state_hash());
        assert_eq!(
            state_checksum(&state).unwrap(),
            state_checksum(&loaded_state).unwrap()
        );
        for _ in 0..10_000 {
            original.step();
            restored.step();
        }
        assert_eq!(original.tick, restored.tick);
        assert_eq!(original.state_hash(), restored.state_hash());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn incompatible_versions_and_stale_snapshot_recipes_are_rejected() {
        let mut old = manifest();
        old.simulator_version += 1;
        assert!(matches!(
            old.validate(),
            Err(Error::UnsupportedVersion {
                kind: "simulator",
                ..
            })
        ));

        let caps = Caps {
            particles: 8,
            anchors: 2,
            shells: 1,
            bumps: 1,
            ..Default::default()
        };
        let genome = caps.default_genome(density::widest_bound_len(&caps));
        let (params, interactions) = caps.resolve(&genome);
        let extent = density::bound_len_for(&params);
        let world = World::with_extent(&params, extent, interactions);
        let mut stale = genome.clone();
        let last = stale.len() - 1;
        stale[last] = (stale[last] + 1.0).min(100.0);
        assert!(matches!(
            snapshot_world(
                SnapshotMetadata {
                    world_id: "stale".into(),
                    checkpoint_id: "tick-0".into(),
                    parent_checkpoint_id: None,
                    simulator_version: SIMULATOR_VERSION,
                    descriptor_version: crate::tuner::metrics::DESCRIPTOR_VERSION,
                    genome_layout_version: GENOME_LAYOUT_VERSION,
                    render_recipe_version: crate::render_recipe::VERSION,
                    render_recipe: RenderRecipe {
                        support: extent * 0.1,
                        ..RenderRecipe::default()
                    },
                },
                &caps,
                stale,
                &world,
            ),
            Err(Error::ManifestStateMismatch {
                field: "world recipe"
            })
        ));
    }

    #[test]
    fn non_finite_and_bad_lengths_have_named_errors() {
        let mut bad = state();
        bad.traits[0] = f32::NAN;
        assert!(matches!(
            save_state(&temporary("nan"), &bad),
            Err(Error::NonFinite { field: "traits" })
        ));
        bad = state();
        bad.traits.pop();
        assert!(matches!(
            bad.validate(),
            Err(Error::ParticleCountMismatch { .. })
        ));
        bad = state();
        bad.traits[0] = 1.1;
        assert!(matches!(bad.validate(), Err(Error::TraitOutOfRange)));
    }
}
