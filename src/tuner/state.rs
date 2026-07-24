//! Durable campaign state: the append-only ledger and its experiment identity, as one
//! checksummed binary file. Floats are written as raw bits, preserving feature rows and
//! scheduling decisions exactly across restarts. No migration path: a mismatched version or
//! identity is rejected by name, never reinterpreted.

use crate::tuner::archive::{ParentSource, SourceTier};
use crate::tuner::ledger::{budget_valid_for_source, feature_schema, CampaignLedger, CampaignRecord,
    ExperimentIdentity, FEATURE_SCHEMA_VERSION};
use crate::tuner::metrics::{Connectivity, Metrics};
use crate::tuner::rollout::{EarlyStop, EvaluationBudget, EvaluationTier, WindowRecord};
use crate::tuner::tuning::Tuning;
use crate::tuner::viability::{GateMargins, Rejection, WorldRejection};
use crate::util::Fnv;
use std::fmt;
use std::fs;
use std::io::{self, Write};
use std::path::Path;

const STATE_MAGIC: &[u8; 8] = b"AXIOMCP1";
const STATE_VERSION: u32 = 4;

#[derive(Debug)]
pub enum CampaignStateError {
    Io(io::Error),
    InvalidFormat(&'static str),
    CorruptChecksum { expected: u64, actual: u64 },
}
impl fmt::Display for CampaignStateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "campaign state I/O: {error}"),
            Self::InvalidFormat(field) => write!(f, "invalid campaign state {field}"),
            Self::CorruptChecksum { expected, actual } =>
                write!(f, "campaign state checksum {actual:016x} does not match {expected:016x}"),
        }
    }
}
impl std::error::Error for CampaignStateError {}
impl From<io::Error> for CampaignStateError {
    fn from(error: io::Error) -> Self { Self::Io(error) }
}

/// Durable in-memory campaign state. Keep this object between search batches so persistence
/// labels eventually meet the lineage-held-out training threshold.
#[derive(Clone, Debug, Default)]
pub struct CampaignState {
    identity: Option<ExperimentIdentity>,
    pub ledger: CampaignLedger,
}
impl CampaignState {
    pub fn for_tuning(tuning: &Tuning) -> Self {
        Self { identity: Some(ExperimentIdentity::for_tuning(tuning)), ..Self::default() }
    }
    pub fn identity(&self) -> Option<&ExperimentIdentity> { self.identity.as_ref() }
    /// Bind to an identity on first use; refuse a different one ever after.
    pub fn bind(&mut self, identity: ExperimentIdentity) -> Result<(), ExperimentMismatch> {
        match &self.identity {
            Some(bound) if bound != &identity => Err(ExperimentMismatch),
            Some(_) => Ok(()),
            None => { self.identity = Some(identity); Ok(()) }
        }
    }
    pub fn save(&self, path: &Path) -> Result<(), CampaignStateError> {
        let identity = self.identity.as_ref().ok_or(CampaignStateError::InvalidFormat("experiment identity"))?;
        let mut out = Vec::new();
        out.extend_from_slice(STATE_MAGIC);
        put_u32(&mut out, STATE_VERSION);
        put_identity(&mut out, identity);
        put_u64(&mut out, next_id(&self.ledger));
        put_u64(&mut out, self.ledger.records().len() as u64);
        for record in self.ledger.records() { put_record(&mut out, record); }
        let checksum = campaign_checksum(&out);
        put_u64(&mut out, checksum);
        write_atomic(path, &out).map_err(CampaignStateError::Io)
    }
    pub fn load(path: &Path) -> Result<Self, CampaignStateError> {
        let bytes = fs::read(path)?;
        let checksum_at = bytes.len().checked_sub(8).ok_or(CampaignStateError::InvalidFormat("checksum"))?;
        let expected = u64::from_le_bytes(bytes[checksum_at..].try_into().map_err(|_| CampaignStateError::InvalidFormat("checksum"))?);
        let actual = campaign_checksum(&bytes[..checksum_at]);
        if expected != actual { return Err(CampaignStateError::CorruptChecksum { expected, actual }); }
        let mut reader = StateReader::new(&bytes[..checksum_at]);
        if reader.take(8)? != STATE_MAGIC { return Err(CampaignStateError::InvalidFormat("magic")); }
        if reader.u32()? != STATE_VERSION { return Err(CampaignStateError::InvalidFormat("version")); }
        let identity = reader.identity()?;
        let next_id = reader.u64()?;
        let records = (0..reader.len()?).map(|_| reader.record()).collect::<Result<Vec<_>, _>>()?;
        if !reader.finished() { return Err(CampaignStateError::InvalidFormat("trailing bytes")); }
        if records.iter().any(|record| {
            record.feature_schema_version != FEATURE_SCHEMA_VERSION
                || !feature_schema().accepts(&record.features)
                || !budget_valid_for_source(record.budget, record.source_tier)
        }) { return Err(CampaignStateError::InvalidFormat("feature schema or evaluation budget")); }
        Ok(Self { identity: Some(identity), ledger: CampaignLedger::restore(records, next_id) })
    }
}

/// The one non-IO failure state loading can hand a campaign: the evidence belongs elsewhere.
#[derive(Debug)]
pub struct ExperimentMismatch;

/// The ledger's next evaluation id, recovered from its records so the serialized form never
/// depends on private counters.
fn next_id(ledger: &CampaignLedger) -> u64 {
    ledger.records().iter().map(|record| record.evaluation_id).max().map_or(1, |id| id + 1)
}

/// Atomic replace with the durable-directory sync. A bare filename yields Some("") from
/// parent(), which is not a directory anything can be created in, so it maps to ".".
pub fn write_atomic(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let parent = path.parent().filter(|parent| !parent.as_os_str().is_empty()).unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let name = path.file_name().and_then(|name| name.to_str()).unwrap_or("campaign");
    let temporary = parent.join(format!(".{name}.tmp-{}", std::process::id()));
    let result = (|| {
        let mut file = fs::File::create(&temporary)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        fs::rename(&temporary, path)?;
        fs::File::open(parent)?.sync_all()?;
        Ok(())
    })();
    if result.is_err() { let _ = fs::remove_file(&temporary); }
    result
}

fn put_identity(out: &mut Vec<u8>, identity: &ExperimentIdentity) {
    for version in [identity.simulator_version, identity.descriptor_version,
        identity.genome_layout_version, identity.feature_schema_version]
    { put_u32(out, version); }
    put_u64(out, identity.particle_count as u64);
    put_u64(out, identity.anchor_count as u64);
    put_u32(out, identity.radius_bits);
    put_u32(out, identity.rate_bits);
    put_u64(out, identity.evaluation_seed);
    put_u64(out, identity.shells as u64);
    put_u64(out, identity.bumps as u64);
    put_u64(out, identity.discovery_steps as u64);
    put_u64(out, identity.tuning_digest);
}

fn put_record(out: &mut Vec<u8>, record: &CampaignRecord) {
    put_u64(out, record.evaluation_id);
    put_u64(out, record.genome_hash);
    put_u64(out, record.lineage_id);
    put_f32s(out, &record.genome);
    out.push(parent_source_code(record.source));
    out.push(source_tier_code(record.source_tier));
    out.push(evaluation_tier_code(record.budget.tier));
    put_u64(out, record.budget.steps as u64);
    put_u64(out, record.seed);
    put_metrics(out, &record.metrics);
    put_u32(out, record.feature_schema_version);
    put_f32s(out, &record.features);
    put_margins(out, record.margins);
    out.push(rejection_code(record.base_gate));
    out.push(world_rejection_code(record.world_gate));
    out.push(early_stop_code(record.early_stop));
    out.push(u8::from(record.tier_passed));
    put_u64(out, record.windows.len() as u64);
    for window in &record.windows {
        put_u64(out, window.tick as u64);
        put_f32s(out, &window.spatial);
        put_f32(out, window.structure);
        for value in window.heterogeneity { put_f32(out, value); }
        for value in window.connectivity.dense { put_f32(out, value); }
        for value in window.connectivity.void { put_f32(out, value); }
        put_f32(out, window.mobility);
        put_f32(out, window.turnover);
    }
}

fn put_metrics(out: &mut Vec<u8>, metrics: &Metrics) {
    for value in [metrics.mobility, metrics.temporal_variance, metrics.turnover, metrics.robustness, metrics.structure] {
        put_f32(out, value);
    }
    for value in metrics.autocorrelation { put_f32(out, value); }
    for value in metrics.heterogeneity { put_f32(out, value); }
    for value in metrics.connectivity.dense { put_f32(out, value); }
    for value in metrics.connectivity.void { put_f32(out, value); }
    put_f32s(out, &metrics.descriptor);
    out.push(u8::from(metrics.alive));
}

fn put_margins(out: &mut Vec<u8>, margins: GateMargins) {
    for value in [margins.structure_floor, margins.structure_ceiling, margins.mobility, margins.temporal,
        margins.repair, margins.heterogeneity, margins.material, margins.void]
    { put_f32(out, value); }
}

fn put_u32(out: &mut Vec<u8>, value: u32) { out.extend_from_slice(&value.to_le_bytes()); }
fn put_u64(out: &mut Vec<u8>, value: u64) { out.extend_from_slice(&value.to_le_bytes()); }
fn put_f32(out: &mut Vec<u8>, value: f32) { put_u32(out, value.to_bits()); }
fn put_f32s(out: &mut Vec<u8>, values: &[f32]) {
    put_u64(out, values.len() as u64);
    for value in values { put_f32(out, *value); }
}

fn campaign_checksum(bytes: &[u8]) -> u64 {
    let mut hash = Fnv::new();
    hash.bytes(bytes.iter().copied());
    hash.finish()
}

struct StateReader<'a> { bytes: &'a [u8], at: usize }
impl<'a> StateReader<'a> {
    fn new(bytes: &'a [u8]) -> Self { Self { bytes, at: 0 } }
    fn take(&mut self, length: usize) -> Result<&'a [u8], CampaignStateError> {
        let end = self.at.checked_add(length).filter(|end| *end <= self.bytes.len())
            .ok_or(CampaignStateError::InvalidFormat("truncated"))?;
        let slice = &self.bytes[self.at..end];
        self.at = end;
        Ok(slice)
    }
    fn byte(&mut self) -> Result<u8, CampaignStateError> { Ok(self.take(1)?[0]) }
    fn u32(&mut self) -> Result<u32, CampaignStateError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().expect("four bytes")))
    }
    fn u64(&mut self) -> Result<u64, CampaignStateError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().expect("eight bytes")))
    }
    fn usize(&mut self) -> Result<usize, CampaignStateError> {
        usize::try_from(self.u64()?).map_err(|_| CampaignStateError::InvalidFormat("length"))
    }
    fn len(&mut self) -> Result<usize, CampaignStateError> {
        let length = self.usize()?;
        if length > 1 << 28 { return Err(CampaignStateError::InvalidFormat("length")); }
        Ok(length)
    }
    fn f32(&mut self) -> Result<f32, CampaignStateError> { Ok(f32::from_bits(self.u32()?)) }
    fn f32s(&mut self) -> Result<Vec<f32>, CampaignStateError> {
        (0..self.len()?).map(|_| self.f32()).collect()
    }
    fn f32_array<const N: usize>(&mut self) -> Result<[f32; N], CampaignStateError> {
        let values = (0..N).map(|_| self.f32()).collect::<Result<Vec<_>, _>>()?;
        values.try_into().map_err(|_| CampaignStateError::InvalidFormat("fixed float array"))
    }
    fn bool(&mut self) -> Result<bool, CampaignStateError> {
        match self.byte()? {
            0 => Ok(false), 1 => Ok(true),
            _ => Err(CampaignStateError::InvalidFormat("boolean")),
        }
    }
    fn metrics(&mut self) -> Result<Metrics, CampaignStateError> {
        let mobility = self.f32()?;
        let temporal_variance = self.f32()?;
        let turnover = self.f32()?;
        let robustness = self.f32()?;
        let structure = self.f32()?;
        Ok(Metrics {
            mobility, temporal_variance, turnover, robustness, structure,
            autocorrelation: self.f32_array()?,
            heterogeneity: self.f32_array()?,
            connectivity: Connectivity { dense: self.f32_array()?, void: self.f32_array()? },
            descriptor: self.f32s()?,
            alive: self.bool()?,
        })
    }
    fn margins(&mut self) -> Result<GateMargins, CampaignStateError> {
        Ok(GateMargins {
            structure_floor: self.f32()?, structure_ceiling: self.f32()?, mobility: self.f32()?,
            temporal: self.f32()?, repair: self.f32()?, heterogeneity: self.f32()?,
            material: self.f32()?, void: self.f32()? })
    }
    fn identity(&mut self) -> Result<ExperimentIdentity, CampaignStateError> {
        Ok(ExperimentIdentity {
            simulator_version: self.u32()?, descriptor_version: self.u32()?,
            genome_layout_version: self.u32()?, feature_schema_version: self.u32()?,
            particle_count: self.usize()?, anchor_count: self.usize()?,
            radius_bits: self.u32()?, rate_bits: self.u32()?,
            evaluation_seed: self.u64()?, shells: self.usize()?, bumps: self.usize()?,
            discovery_steps: self.usize()?, tuning_digest: self.u64()? })
    }
    fn record(&mut self) -> Result<CampaignRecord, CampaignStateError> {
        let evaluation_id = self.u64()?;
        let genome_hash = self.u64()?;
        let lineage_id = self.u64()?;
        let genome = self.f32s()?;
        let source = parent_source_from_code(self.byte()?)?;
        let source_tier = source_tier_from_code(self.byte()?)?;
        let tier = evaluation_tier_from_code(self.byte()?)?;
        let budget = EvaluationBudget { tier, steps: self.usize()? };
        let seed = self.u64()?;
        let metrics = self.metrics()?;
        let feature_schema_version = self.u32()?;
        let features = self.f32s()?;
        let margins = self.margins()?;
        let base_gate = rejection_from_code(self.byte()?)?;
        let world_gate = world_rejection_from_code(self.byte()?)?;
        let early_stop = early_stop_from_code(self.byte()?)?;
        let tier_passed = self.bool()?;
        let windows = (0..self.len()?).map(|_| Ok(WindowRecord {
            tick: self.usize()?,
            spatial: self.f32s()?,
            structure: self.f32()?,
            heterogeneity: self.f32_array()?,
            connectivity: Connectivity { dense: self.f32_array()?, void: self.f32_array()? },
            mobility: self.f32()?,
            turnover: self.f32()?,
        })).collect::<Result<Vec<_>, CampaignStateError>>()?;
        Ok(CampaignRecord {
            evaluation_id, genome_hash, lineage_id, genome, source, source_tier, budget, seed,
            metrics, feature_schema_version, features, margins, base_gate, world_gate, early_stop,
            windows, tier_passed })
    }
    fn finished(&self) -> bool { self.at == self.bytes.len() }
}

fn parent_source_code(source: ParentSource) -> u8 {
    match source {
        ParentSource::Bootstrap => 0, ParentSource::Novelty => 1, ParentSource::Expedition => 2,
        ParentSource::Random => 3, ParentSource::Curated => 4,
    }
}
fn parent_source_from_code(code: u8) -> Result<ParentSource, CampaignStateError> {
    match code {
        0 => Ok(ParentSource::Bootstrap), 1 => Ok(ParentSource::Novelty), 2 => Ok(ParentSource::Expedition),
        3 => Ok(ParentSource::Random), 4 => Ok(ParentSource::Curated),
        _ => Err(CampaignStateError::InvalidFormat("parent source")),
    }
}
fn source_tier_code(tier: SourceTier) -> u8 {
    match tier { SourceTier::Discovery => 0, SourceTier::Persistence => 1, SourceTier::Certification => 2 }
}
fn source_tier_from_code(code: u8) -> Result<SourceTier, CampaignStateError> {
    match code {
        0 => Ok(SourceTier::Discovery), 1 => Ok(SourceTier::Persistence), 2 => Ok(SourceTier::Certification),
        _ => Err(CampaignStateError::InvalidFormat("source tier")),
    }
}
fn evaluation_tier_code(tier: EvaluationTier) -> u8 {
    match tier { EvaluationTier::Discovery => 0, EvaluationTier::Persistence => 1, EvaluationTier::Certification => 2 }
}
fn evaluation_tier_from_code(code: u8) -> Result<EvaluationTier, CampaignStateError> {
    match code {
        0 => Ok(EvaluationTier::Discovery), 1 => Ok(EvaluationTier::Persistence), 2 => Ok(EvaluationTier::Certification),
        _ => Err(CampaignStateError::InvalidFormat("evaluation tier")),
    }
}
fn rejection_code(value: Result<(), Rejection>) -> u8 {
    match value {
        Ok(()) => 0, Err(Rejection::Dead) => 1, Err(Rejection::Dispersed) => 2,
        Err(Rejection::Frozen) => 3, Err(Rejection::Collapsed) => 4, Err(Rejection::Fragile) => 5,
    }
}
fn rejection_from_code(code: u8) -> Result<Result<(), Rejection>, CampaignStateError> {
    match code {
        0 => Ok(Ok(())), 1 => Ok(Err(Rejection::Dead)), 2 => Ok(Err(Rejection::Dispersed)),
        3 => Ok(Err(Rejection::Frozen)), 4 => Ok(Err(Rejection::Collapsed)), 5 => Ok(Err(Rejection::Fragile)),
        _ => Err(CampaignStateError::InvalidFormat("base gate")),
    }
}
fn world_rejection_code(value: Result<(), WorldRejection>) -> u8 {
    match value {
        Ok(()) => 0, Err(WorldRejection::Homogeneous) => 1,
        Err(WorldRejection::MaterialDisconnected) => 2, Err(WorldRejection::VoidDisconnected) => 3,
    }
}
fn world_rejection_from_code(code: u8) -> Result<Result<(), WorldRejection>, CampaignStateError> {
    match code {
        0 => Ok(Ok(())), 1 => Ok(Err(WorldRejection::Homogeneous)),
        2 => Ok(Err(WorldRejection::MaterialDisconnected)), 3 => Ok(Err(WorldRejection::VoidDisconnected)),
        _ => Err(CampaignStateError::InvalidFormat("world gate")),
    }
}
fn early_stop_code(value: Option<EarlyStop>) -> u8 {
    match value {
        None => 0, Some(EarlyStop::NonFinite) => 1,
        Some(EarlyStop::SustainedDispersion) => 2, Some(EarlyStop::SustainedCollapse) => 3,
    }
}
fn early_stop_from_code(code: u8) -> Result<Option<EarlyStop>, CampaignStateError> {
    match code {
        0 => Ok(None), 1 => Ok(Some(EarlyStop::NonFinite)),
        2 => Ok(Some(EarlyStop::SustainedDispersion)), 3 => Ok(Some(EarlyStop::SustainedCollapse)),
        _ => Err(CampaignStateError::InvalidFormat("early stop")),
    }
}
