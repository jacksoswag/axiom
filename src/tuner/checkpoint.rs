//! Durable, cache-free world checkpoints. An archive entry records a discoverable rule; a
//! checkpoint records one particular place in that rule's history. A manifest is Caps plus the
//! full genome plus the resolved box and measured norms, everything Sim reconstruction needs
//! with no derived duplicates; render fields, meshes, and spatial indexes are caches rebuilt
//! after restore.

use crate::engine::sim::Sim;
use crate::render_recipe::RenderRecipe;
use crate::tuner::genome::Caps;
use crate::util::Fnv;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

pub const SIMULATOR_VERSION: u32 = 3;
pub const GENOME_LAYOUT_VERSION: u32 = 2;
pub const MANIFEST_VERSION: u32 = 4;
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
    pub caps: Caps,
    pub box_len: f32, // resolved once, stored so restore never re-probes
    pub tick: u64,
    pub interaction_norms: Vec<f32>, // measured norms are authoritative, restore never recalibrates
    pub genome: Vec<f32>, // full capped genome, coordination and logits included
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
            Error::UnsupportedVersion { kind, found } => write!(f, "unsupported {kind} version {found}"),
            Error::Truncated { field } => write!(f, "truncated {field}"),
            Error::InvalidLength { field, length } => write!(f, "invalid {field} length {length}"),
            Error::NonFinite { field } => write!(f, "non-finite {field}"),
            Error::TraitOutOfRange => write!(f, "checkpoint trait lies outside 0..=1"),
            Error::InvalidDimensions => write!(f, "checkpoint dimensions must be positive"),
            Error::InvalidIdentifier { field } => write!(f, "invalid checkpoint {field}"),
            Error::InvalidGenomeLayout { expected, actual } => write!(f, "checkpoint genome has {actual} genes, expected {expected}"),
            Error::ParticleCountMismatch { positions, traits } => write!(f, "particle state has {positions} positions but {traits} traits"),
            Error::ManifestStateMismatch { field } => write!(f, "manifest and state disagree on {field}"),
            Error::CorruptChecksum { expected, actual } => write!(f, "checkpoint checksum {actual:016x} does not match {expected:016x}"),
            Error::ImmutableCheckpoint { path } => write!(f, "checkpoint already exists: {}", path.display()),
        }
    }
}
impl std::error::Error for Error {}
impl From<io::Error> for Error {
    fn from(problem: io::Error) -> Self { Error::Io(problem) }
}

/// Genome length for possibly hostile counts, refusing on overflow rather than wrapping.
fn checked_gene_len(caps: &Caps) -> Option<usize> {
    let pairs = caps.anchor_count.checked_mul(caps.anchor_count)?;
    let stride = 3usize.checked_mul(caps.shells.checked_add(caps.bumps)?)?.checked_add(1)?;
    1usize.checked_add(caps.anchor_count)?.checked_add(pairs.checked_mul(stride)?)
}

impl WorldManifest {
    pub fn validate(&self) -> Result<(), Error> {
        for (kind, found, expected) in [
            ("simulator", self.simulator_version, SIMULATOR_VERSION),
            ("descriptor", self.descriptor_version, crate::tuner::metrics::DESCRIPTOR_VERSION),
            ("genome layout", self.genome_layout_version, GENOME_LAYOUT_VERSION),
        ] {
            if found != expected { return Err(Error::UnsupportedVersion { kind, found }); }
        }
        if self.render_recipe_version != crate::render_recipe::VERSION {
            return Err(Error::UnsupportedVersion { kind: "render recipe", found: self.render_recipe_version });
        }
        if !self.render_recipe.valid() {
            return Err(Error::InvalidLength { field: "render recipe", length: self.render_recipe.resolution });
        }
        valid_identifier("world id", &self.world_id)?;
        valid_identifier("latest checkpoint id", &self.latest_checkpoint_id)?;
        let caps = &self.caps;
        if caps.dimensions == 0 { return Err(Error::InvalidDimensions); }
        if caps.particle_count == 0 || caps.anchor_count < 2 || caps.shells == 0 || caps.bumps == 0 {
            return Err(Error::InvalidLength { field: "particle count, anchors, shells, or bumps", length: caps.particle_count });
        }
        finite("manifest scalar", &[caps.radius, caps.rate, self.box_len])?;
        if caps.radius <= 0.0 || caps.rate <= 0.0 || self.box_len <= 0.0 {
            return Err(Error::InvalidLength { field: "positive manifest scalar", length: 0 });
        }
        if self.render_recipe.support >= self.box_len * 0.5 {
            return Err(Error::InvalidLength { field: "render support", length: self.render_recipe.resolution });
        }
        let expected = checked_gene_len(caps).ok_or(Error::InvalidLength { field: "genome layout", length: usize::MAX })?;
        if self.genome.len() != expected {
            return Err(Error::InvalidGenomeLayout { expected, actual: self.genome.len() });
        }
        let pairs = caps.anchor_count * caps.anchor_count;
        if self.interaction_norms.len() != pairs {
            return Err(Error::InvalidLength { field: "interaction norms", length: self.interaction_norms.len() });
        }
        finite("manifest genome", &self.genome)?;
        finite("interaction norms", &self.interaction_norms)?;
        if self.interaction_norms.iter().any(|norm| *norm <= 0.0) {
            return Err(Error::InvalidLength { field: "positive interaction norm", length: 0 });
        }
        Ok(())
    }
}

impl WorldState {
    pub fn validate(&self) -> Result<(), Error> {
        if self.dimensions == 0 { return Err(Error::InvalidDimensions); }
        valid_identifier("checkpoint id", &self.checkpoint_id)?;
        valid_identifier("world id", &self.world_id)?;
        if let Some(parent) = &self.parent_checkpoint_id { valid_identifier("parent checkpoint id", parent)?; }
        if self.positions.len() % self.dimensions != 0 {
            return Err(Error::InvalidLength { field: "positions", length: self.positions.len() });
        }
        let particles = self.positions.len() / self.dimensions;
        if self.traits.len() != particles {
            return Err(Error::ParticleCountMismatch { positions: particles, traits: self.traits.len() });
        }
        if self.genome.is_empty() { return Err(Error::InvalidLength { field: "genome", length: 0 }); }
        finite("positions", &self.positions)?;
        finite("traits", &self.traits)?;
        if self.traits.iter().any(|trait_value| !(0.0..=1.0).contains(trait_value)) {
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
    for value in [manifest.simulator_version, manifest.descriptor_version,
        manifest.genome_layout_version, manifest.render_recipe_version]
    { write_u32(&mut bytes, value); }
    write_u64(&mut bytes, manifest.render_recipe.resolution as u64);
    for value in [manifest.render_recipe.support, manifest.render_recipe.iso, manifest.render_recipe.absorption] {
        write_f32(&mut bytes, value);
    }
    let caps = &manifest.caps;
    for value in [caps.particle_count as u64, caps.dimensions as u64, caps.anchor_count as u64] {
        write_u64(&mut bytes, value);
    }
    write_f32(&mut bytes, caps.radius);
    write_f32(&mut bytes, caps.rate);
    write_u64(&mut bytes, caps.seed);
    write_u64(&mut bytes, caps.shells as u64);
    write_u64(&mut bytes, caps.bumps as u64);
    write_f32(&mut bytes, manifest.box_len);
    write_u64(&mut bytes, manifest.tick);
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
    let checksum_at = bytes.len().checked_sub(8).ok_or(Error::Truncated { field: "world manifest" })?;
    let expected = u64::from_le_bytes(bytes[checksum_at..].try_into().expect("eight bytes"));
    let actual = checksum(&bytes[..checksum_at]);
    if expected != actual { return Err(Error::CorruptChecksum { expected, actual }); }
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
    let caps = Caps {
        particle_count: reader.usize("particle count")?,
        dimensions: reader.usize("dimensions")?,
        anchor_count: reader.usize("anchors")?,
        radius: reader.f32("radius")?,
        rate: reader.f32("rate")?,
        seed: reader.u64("seed")?,
        shells: reader.usize("shells")?,
        bumps: reader.usize("bumps")?,
    };
    let manifest = WorldManifest {
        world_id, simulator_version, descriptor_version, genome_layout_version,
        render_recipe_version, render_recipe, caps,
        box_len: reader.f32("box length")?,
        tick: reader.u64("tick")?,
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
    let bytes = state_bytes(state)?;
    write_immutable(path, &bytes)
}

pub fn load_state(path: &Path) -> Result<WorldState, Error> {
    let mut bytes = Vec::new();
    File::open(path)?.read_to_end(&mut bytes)?;
    let checksum_at = bytes.len().checked_sub(8).ok_or(Error::Truncated { field: "checkpoint" })?;
    let expected = u64::from_le_bytes(bytes[checksum_at..].try_into().expect("eight bytes"));
    let actual = checksum(&bytes[..checksum_at]);
    if expected != actual { return Err(Error::CorruptChecksum { expected, actual }); }
    let mut reader = Reader::new(&bytes[..checksum_at]);
    reader.magic(STATE_MAGIC, "world state")?;
    reader.version(STATE_VERSION, "world state")?;
    let checkpoint_id = reader.string("checkpoint id")?;
    let parent_checkpoint_id = reader.optional_string("parent checkpoint id")?;
    let state = WorldState {
        checkpoint_id, parent_checkpoint_id,
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
pub fn save_world(root: &Path, manifest: &WorldManifest, state: &WorldState) -> Result<(PathBuf, PathBuf), Error> {
    validate_pair(manifest, state)?;
    let world_root = root.join(&manifest.world_id);
    let state_path = world_root.join("checkpoints").join(format!("{}.checkpoint", state.checkpoint_id));
    let manifest_path = world_root.join("world.manifest");
    match save_state(&state_path, state) {
        Ok(()) => {}
        Err(Error::ImmutableCheckpoint { .. }) => { // same id twice is fine only for identical bytes
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

pub fn load_world(manifest_path: &Path, state_path: &Path) -> Result<(WorldManifest, WorldState), Error> {
    let manifest = load_manifest(manifest_path)?;
    let state = load_state(state_path)?;
    validate_pair(&manifest, &state)?;
    Ok((manifest, state))
}

pub fn validate_pair(manifest: &WorldManifest, state: &WorldState) -> Result<(), Error> {
    manifest.validate()?;
    state.validate()?;
    if manifest.world_id != state.world_id || manifest.latest_checkpoint_id != state.checkpoint_id {
        return Err(Error::ManifestStateMismatch { field: "world or checkpoint id" });
    }
    if manifest.tick != state.tick { return Err(Error::ManifestStateMismatch { field: "tick" }); }
    if manifest.caps.particle_count != state.traits.len() || manifest.caps.dimensions != state.dimensions {
        return Err(Error::ManifestStateMismatch { field: "particle count or dimensions" });
    }
    if manifest.genome.iter().map(|value| value.to_bits()).ne(state.genome.iter().map(|value| value.to_bits())) {
        return Err(Error::ManifestStateMismatch { field: "genome" });
    }
    Ok(())
}

/// Capture the exact dynamic state of a sim alongside its complete resolved recipe. The genome
/// must decode, at the sim's own box, to exactly the law the sim is running.
pub fn snapshot_world(metadata: SnapshotMetadata, caps: &Caps, full_genome: Vec<f32>, sim: &Sim)
    -> Result<(WorldManifest, WorldState), Error>
{
    let expected = checked_gene_len(caps).ok_or(Error::InvalidLength { field: "genome layout", length: usize::MAX })?;
    if full_genome.len() != expected {
        return Err(Error::InvalidGenomeLayout { expected, actual: full_genome.len() });
    }
    let decoded = caps.params_at(&full_genome, sim.params.box_len);
    let bits = |values: &[f32]| values.iter().map(|value| value.to_bits()).collect::<Vec<_>>();
    let matches = decoded.particle_count == sim.params.particle_count
        && decoded.dimensions == sim.params.dimensions
        && decoded.anchor_count == sim.params.anchor_count
        && decoded.shells == sim.params.shells && decoded.bumps == sim.params.bumps
        && decoded.seed == sim.params.seed
        && decoded.coordination.to_bits() == sim.params.coordination.to_bits()
        && decoded.dt.to_bits() == sim.params.dt.to_bits()
        && decoded.radius.to_bits() == sim.params.radius.to_bits()
        && decoded.box_len.to_bits() == sim.params.box_len.to_bits()
        && bits(&decoded.trait_distribution) == bits(&sim.params.trait_distribution)
        && bits(&decoded.interactions) == bits(&sim.params.interactions);
    if !matches { return Err(Error::ManifestStateMismatch { field: "world recipe" }); }
    let manifest = WorldManifest {
        world_id: metadata.world_id.clone(),
        simulator_version: metadata.simulator_version,
        descriptor_version: metadata.descriptor_version,
        genome_layout_version: metadata.genome_layout_version,
        render_recipe_version: metadata.render_recipe_version,
        render_recipe: metadata.render_recipe,
        caps: caps.clone(),
        box_len: sim.params.box_len,
        tick: sim.tick,
        interaction_norms: sim.matrix.interactions.iter().map(|interaction| interaction.norm).collect(),
        genome: full_genome,
        latest_checkpoint_id: metadata.checkpoint_id.clone(),
    };
    let state = WorldState {
        checkpoint_id: metadata.checkpoint_id,
        parent_checkpoint_id: metadata.parent_checkpoint_id,
        world_id: metadata.world_id,
        tick: sim.tick,
        dimensions: sim.params.dimensions,
        positions: sim.substrate.positions.clone(),
        traits: sim.substrate.traits.clone(),
        genome: manifest.genome.clone(),
    };
    validate_pair(&manifest, &state)?;
    Ok((manifest, state))
}

/// Recreate a cache-free sim from saved data. The saved interaction norms are installed after
/// construction, so restoring never depends on a calibration run.
pub fn restore_world(manifest: &WorldManifest, state: &WorldState) -> Result<Sim, Error> {
    validate_pair(manifest, state)?;
    let params = manifest.caps.params_at(&manifest.genome, manifest.box_len);
    let mut sim = Sim::new(&params);
    sim.substrate.positions = state.positions.clone();
    sim.substrate.traits = state.traits.clone();
    sim.tick = state.tick;
    for (interaction, &norm) in sim.matrix.interactions.iter_mut().zip(&manifest.interaction_norms) {
        interaction.norm = norm; // validated finite and positive by manifest.validate
    }
    Ok(sim)
}

/// Stable checksum of a full checkpoint record, binding audit records to the exact serialized
/// checkpoint independent of files.
pub fn state_checksum(state: &WorldState) -> Result<u64, Error> {
    state.validate()?;
    let bytes = state_bytes(state)?;
    Ok(checksum(&bytes[..bytes.len() - 8]))
}

/// The state's serialized form: header, payload, trailing checksum.
fn state_bytes(state: &WorldState) -> Result<Vec<u8>, Error> {
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
    Ok(bytes)
}

fn finite(field: &'static str, values: &[f32]) -> Result<(), Error> {
    if values.iter().all(|value| value.is_finite()) { Ok(()) } else { Err(Error::NonFinite { field }) }
}

fn valid_identifier(field: &'static str, value: &str) -> Result<(), Error> {
    if value.is_empty() || value.len() > MAX_STRING
        || !value.bytes().all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    { return Err(Error::InvalidIdentifier { field }); }
    Ok(())
}

/// Publish an immutable file only if its final name does not exist. A hard link provides the
/// no-replace publication step after the temporary sibling is fully flushed.
fn write_immutable(path: &Path, bytes: &[u8]) -> Result<(), Error> {
    let parent = path.parent().filter(|parent| !parent.as_os_str().is_empty()).unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    if path.exists() { return Err(Error::ImmutableCheckpoint { path: path.to_path_buf() }); }
    let stem = path.file_name().and_then(|name| name.to_str()).unwrap_or("checkpoint");
    for attempt in 0..128u32 {
        let temporary = parent.join(format!(".{stem}.tmp-{}-{attempt}", std::process::id()));
        let mut file = match OpenOptions::new().write(true).create_new(true).open(&temporary) {
            Ok(file) => file,
            Err(problem) if problem.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(problem) => return Err(Error::Io(problem)),
        };
        if let Err(problem) = file.write_all(bytes).and_then(|_| file.sync_all()) {
            let _ = fs::remove_file(&temporary);
            return Err(Error::Io(problem));
        }
        match fs::hard_link(&temporary, path) {
            Ok(()) => { fs::remove_file(&temporary)?; return Ok(()); }
            Err(problem) if problem.kind() == io::ErrorKind::AlreadyExists => {
                let _ = fs::remove_file(&temporary);
                return Err(Error::ImmutableCheckpoint { path: path.to_path_buf() });
            }
            Err(problem) => { let _ = fs::remove_file(&temporary); return Err(Error::Io(problem)); }
        }
    }
    Err(Error::Io(io::Error::new(io::ErrorKind::AlreadyExists, "checkpoint temp name exhausted")))
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), Error> {
    // A bare filename yields Some(""), which is not a directory anything can be created in.
    let parent = path.parent().filter(|parent| !parent.as_os_str().is_empty()).unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let stem = path.file_name().and_then(|name| name.to_str()).unwrap_or("checkpoint");
    let mut temporary = None;
    for attempt in 0..128u32 {
        let candidate = parent.join(format!(".{stem}.tmp-{}-{attempt}", std::process::id()));
        match OpenOptions::new().write(true).create_new(true).open(&candidate) {
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
    let temporary = temporary
        .ok_or_else(|| Error::Io(io::Error::new(io::ErrorKind::AlreadyExists, "checkpoint temp name exhausted")))?;
    if let Err(problem) = fs::rename(&temporary, path) {
        let _ = fs::remove_file(&temporary);
        return Err(Error::Io(problem));
    }
    Ok(())
}

fn write_u32(out: &mut Vec<u8>, value: u32) { out.extend_from_slice(&value.to_le_bytes()); }
fn write_u64(out: &mut Vec<u8>, value: u64) { out.extend_from_slice(&value.to_le_bytes()); }
fn write_f32(out: &mut Vec<u8>, value: f32) { write_u32(out, value.to_bits()); }

fn write_string(out: &mut Vec<u8>, value: &str) -> Result<(), Error> {
    if value.is_empty() || value.len() > MAX_STRING {
        return Err(Error::InvalidLength { field: "string", length: value.len() });
    }
    write_u32(out, value.len() as u32);
    out.extend_from_slice(value.as_bytes());
    Ok(())
}

fn write_f32s(out: &mut Vec<u8>, values: &[f32]) -> Result<(), Error> {
    if values.is_empty() || values.len() > MAX_VALUES {
        return Err(Error::InvalidLength { field: "float vector", length: values.len() });
    }
    write_u64(out, values.len() as u64);
    for &value in values { write_f32(out, value); }
    Ok(())
}

struct Reader<'a> { bytes: &'a [u8], at: usize }
impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self { Reader { bytes, at: 0 } }
    fn take(&mut self, len: usize, field: &'static str) -> Result<&'a [u8], Error> {
        let end = self.at.checked_add(len).filter(|end| *end <= self.bytes.len()).ok_or(Error::Truncated { field })?;
        let slice = &self.bytes[self.at..end];
        self.at = end;
        Ok(slice)
    }
    fn magic(&mut self, expected: [u8; 8], kind: &'static str) -> Result<(), Error> {
        if self.take(8, kind)? != expected { return Err(Error::InvalidMagic { kind }); }
        Ok(())
    }
    fn version(&mut self, expected: u32, kind: &'static str) -> Result<(), Error> {
        let found = self.u32("version")?;
        if found != expected { return Err(Error::UnsupportedVersion { kind, found }); }
        Ok(())
    }
    fn u32(&mut self, field: &'static str) -> Result<u32, Error> {
        Ok(u32::from_le_bytes(self.take(4, field)?.try_into().expect("four bytes")))
    }
    fn u64(&mut self, field: &'static str) -> Result<u64, Error> {
        Ok(u64::from_le_bytes(self.take(8, field)?.try_into().expect("eight bytes")))
    }
    fn usize(&mut self, field: &'static str) -> Result<usize, Error> {
        usize::try_from(self.u64(field)?).map_err(|_| Error::InvalidLength { field, length: usize::MAX })
    }
    fn f32(&mut self, field: &'static str) -> Result<f32, Error> {
        let value = f32::from_bits(self.u32(field)?);
        if value.is_finite() { Ok(value) } else { Err(Error::NonFinite { field }) }
    }
    fn string(&mut self, field: &'static str) -> Result<String, Error> {
        let len = self.u32(field)? as usize;
        if len == 0 || len > MAX_STRING { return Err(Error::InvalidLength { field, length: len }); }
        let value = std::str::from_utf8(self.take(len, field)?).map_err(|_| Error::InvalidLength { field, length: len })?;
        Ok(value.to_owned())
    }
    fn optional_string(&mut self, field: &'static str) -> Result<Option<String>, Error> {
        let len = self.u32(field)?;
        if len == u32::MAX { return Ok(None); }
        let len = len as usize;
        if len == 0 || len > MAX_STRING { return Err(Error::InvalidLength { field, length: len }); }
        let value = std::str::from_utf8(self.take(len, field)?).map_err(|_| Error::InvalidLength { field, length: len })?;
        Ok(Some(value.to_owned()))
    }
    fn f32s(&mut self, field: &'static str) -> Result<Vec<f32>, Error> {
        let len = self.usize(field)?;
        if len == 0 || len > MAX_VALUES { return Err(Error::InvalidLength { field, length: len }); }
        let mut values = Vec::with_capacity(len);
        for _ in 0..len { values.push(self.f32(field)?); }
        Ok(values)
    }
    fn finished(&self, field: &'static str) -> Result<(), Error> {
        if self.at == self.bytes.len() { Ok(()) } else {
            Err(Error::InvalidLength { field, length: self.bytes.len() - self.at })
        }
    }
}

fn checksum(bytes: &[u8]) -> u64 {
    let mut hash = Fnv::new();
    hash.bytes(bytes.iter().copied());
    hash.finish()
}
