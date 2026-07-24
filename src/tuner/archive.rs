//! Versioned discovery archive with deterministic, batch-level capacity pruning.
//! The archive carries the exact Tuning that produced and judged it, so a file is
//! self-describing: whatever reads it back can reproduce the worlds and knows which gates
//! admitted them. The format has no migration path; layout changes force a rerun.

use crate::tuner::metrics::{descriptor_len, Connectivity, Metrics, HETEROGENEITY_SIDES, HETEROGENEITY_VALUES, LAGS};
use crate::tuner::novelty::{crowding, novelty, NEIGHBORS};
use crate::tuner::tuning::{self, Tuning};
use crate::tuner::viability::{viable, Rejection};
use crate::util::Fnv;
use std::fmt::Write as _;

const FORMAT_TAG: &str = "axiom-archive";
const FORMAT_VERSION: u32 = 9;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParentSource { Bootstrap, Novelty, Expedition, Random, Curated }
impl ParentSource {
    fn code(self) -> &'static str {
        match self {
            Self::Bootstrap => "bootstrap", Self::Novelty => "novelty", Self::Expedition => "expedition",
            Self::Random => "random", Self::Curated => "curated",
        }
    }
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "bootstrap" => Ok(Self::Bootstrap), "novelty" => Ok(Self::Novelty),
            "expedition" => Ok(Self::Expedition), "random" => Ok(Self::Random), "curated" => Ok(Self::Curated),
            _ => Err(format!("unknown parent source {value:?}")),
        }
    }
}

/// Evaluation source tiers are append-only ledger labels. They do not grant a learned model
/// authority over discovery admission or parent selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceTier { Discovery, Persistence, Certification }

#[derive(Clone, Debug)]
pub struct EvaluationRecord {
    pub genome_hash: u64,
    pub lineage_id: u64,
    pub seed: u64,
    pub parent_source: ParentSource,
    pub source_tier: SourceTier,
    pub metrics: Metrics,
    pub gate: Result<(), Rejection>,
}

/// In-memory append-only evaluation ledger. Persistence and certification workers append their
/// exact outcome here without changing archive admission semantics.
#[derive(Clone, Default, Debug)]
pub struct EvaluationLedger { records: Vec<EvaluationRecord> }
impl EvaluationLedger {
    pub fn records(&self) -> &[EvaluationRecord] { &self.records }
    pub fn append(&mut self, record: EvaluationRecord) { self.records.push(record); }
}

/// One viable discovered recipe. Admission novelty is immutable provenance; current novelty is
/// recalculated from the archive frozen at each generation boundary.
#[derive(Clone, Debug)]
pub struct Entry {
    pub genome: Vec<f32>,
    pub metrics: Metrics,
    pub admission_novelty: f32,
    pub current_novelty: f32,
    pub lineage_id: u64,
    pub parent_lineage_id: Option<u64>,
    pub birth_generation: u64,
    pub parent_source: ParentSource,
}
impl Entry {
    pub fn stable_key(&self) -> u64 { genome_hash(&self.genome) ^ self.lineage_id.rotate_left(17) }
}

/// Candidate admission data collected before the archive transaction. Search evaluates these
/// in parallel, then one deterministic merge decides the retained set.
#[derive(Clone, Debug)]
pub struct Admission {
    pub genome: Vec<f32>,
    pub metrics: Metrics,
    pub lineage_id: u64,
    pub parent_lineage_id: Option<u64>,
    pub birth_generation: u64,
    pub parent_source: ParentSource,
    pub admission_novelty: f32,
}

#[derive(Debug)]
pub struct Archive {
    tuning: Tuning,
    entries: Vec<Entry>,
}
impl Archive {
    pub fn new(tuning: Tuning) -> Self { Self { tuning, entries: Vec::new() } }
    pub fn tuning(&self) -> &Tuning { &self.tuning }
    pub fn capacity(&self) -> usize { self.tuning.discovery.capacity.max(1) }
    pub fn entries(&self) -> &[Entry] { &self.entries }
    pub fn len(&self) -> usize { self.entries.len() }
    pub fn is_empty(&self) -> bool { self.entries.is_empty() }
    pub fn descriptors(&self) -> Vec<Vec<f32>> {
        self.entries.iter().map(|entry| entry.metrics.descriptor.clone()).collect()
    }
    /// Recompute selection novelty against one frozen snapshot, excluding each entry itself so
    /// a small archive is not dominated by artificial zeros.
    pub fn refresh_current_novelty(&mut self) { refresh_novelty(&mut self.entries); }
    /// Merge every viable candidate in one transaction. Scoring and tie breaking see the
    /// complete combined population, then the retained archive is refreshed again so its
    /// current novelty describes the archive that actually remains.
    pub fn merge(&mut self, admissions: impl IntoIterator<Item = Admission>) -> Vec<Entry> {
        let mut admitted = Vec::new();
        for admission in admissions { // re-gate at the boundary: merge accepts data, not trust
            if viable(&admission.metrics, &self.tuning.gates).is_err()
                || admission.metrics.descriptor.len() != descriptor_len()
                || admission.metrics.descriptor.iter().any(|value| !value.is_finite())
                || admission.genome.iter().any(|value| !value.is_finite())
                || !admission.admission_novelty.is_finite()
            { continue; }
            admitted.push(Entry {
                genome: admission.genome, metrics: admission.metrics,
                admission_novelty: admission.admission_novelty, current_novelty: 0.0,
                lineage_id: admission.lineage_id, parent_lineage_id: admission.parent_lineage_id,
                birth_generation: admission.birth_generation, parent_source: admission.parent_source });
        }
        if admitted.is_empty() { self.refresh_current_novelty(); return admitted; }

        let mut combined = self.entries.clone();
        combined.append(&mut admitted);
        combined.sort_by(entry_order);
        refresh_novelty(&mut combined);
        let descriptors: Vec<Vec<f32>> = combined.iter().map(|entry| entry.metrics.descriptor.clone()).collect();
        let mut ranked: Vec<(Entry, f32)> = combined.into_iter().enumerate().map(|(index, entry)| {
            let others: Vec<Vec<f32>> = descriptors.iter().enumerate()
                .filter(|(other, _)| *other != index).map(|(_, descriptor)| descriptor.clone()).collect();
            (entry.clone(), crowding(&entry.metrics.descriptor, &others))
        }).collect();
        ranked.sort_by(|(left, left_crowding), (right, right_crowding)| {
            right.current_novelty.total_cmp(&left.current_novelty)
                .then_with(|| right_crowding.total_cmp(left_crowding))
                .then_with(|| entry_order(left, right))
        });
        ranked.truncate(self.capacity());
        self.entries = ranked.into_iter().map(|(entry, _)| entry).collect();
        self.entries.sort_by(entry_order);
        self.refresh_current_novelty();
        self.entries.clone()
    }
    pub fn to_text(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "{FORMAT_TAG} {FORMAT_VERSION}");
        for knob in tuning::KNOBS { // header serializes every knob, in registry order
            let _ = writeln!(out, "{} {}", knob.key, knob.show(&self.tuning));
        }
        let _ = writeln!(out, "entries {}", self.entries.len());
        for entry in &self.entries {
            let metrics = &entry.metrics;
            let _ = write!(out, "entry {} {} {} {} {} {} {} {} {} {} {} {} {}",
                entry.lineage_id, entry.parent_lineage_id.unwrap_or(0), entry.birth_generation,
                entry.parent_source.code(), entry.admission_novelty, entry.current_novelty,
                metrics.mobility, metrics.temporal_variance, metrics.robustness, metrics.structure,
                metrics.turnover, metrics.alive as u8, metrics.autocorrelation.len());
            for value in metrics.autocorrelation { let _ = write!(out, " {value}"); }
            for value in metrics.heterogeneity.iter()
                .chain(metrics.connectivity.dense.iter()).chain(metrics.connectivity.void.iter())
                .chain(metrics.descriptor.iter()).chain(entry.genome.iter())
            { let _ = write!(out, " {value}"); }
            let _ = writeln!(out);
        }
        out
    }
    /// Validate and read a complete archive. Deliberately no permissive migration path:
    /// descriptor and genome layout changes must force a rerun.
    pub fn from_text(text: &str) -> Result<(Self, Tuning), String> {
        let mut lines = text.lines();
        expect_line(&mut lines, FORMAT_TAG, Some(&FORMAT_VERSION.to_string()))?;
        let tuning = read_tuning(&mut lines)?;
        let count = parse_usize(required_value(&mut lines, "entries")?, "entries")?;
        validate_tuning(&tuning)?;
        if count > tuning.discovery.capacity {
            return Err(format!("entries {count} exceeds capacity {}", tuning.discovery.capacity));
        }
        let metrics_tail = HETEROGENEITY_SIDES.len() * HETEROGENEITY_VALUES + 4;
        let expected_values = 14 + LAGS.len() + metrics_tail + descriptor_len() + tuning.world.gene_len();
        let mut entries = Vec::with_capacity(count);
        for number in 0..count {
            let line = lines.next().ok_or_else(|| format!("missing entry {}", number + 1))?;
            let fields: Vec<&str> = line.split_whitespace().collect();
            if fields.len() != expected_values || fields.first() != Some(&"entry") {
                return Err(format!("entry {} has wrong field count", number + 1));
            }
            let lineage_id = parse_u64(fields[1], "lineage")?;
            let parent = parse_u64(fields[2], "parent lineage")?;
            let birth_generation = parse_u64(fields[3], "birth generation")?;
            let parent_source = ParentSource::parse(fields[4])?;
            let admission_novelty = parse_float(fields[5], "admission novelty")?;
            let current_novelty = parse_float(fields[6], "current novelty")?;
            let autocorrelation_count = parse_usize(fields[13], "autocorrelation count")?;
            if autocorrelation_count != LAGS.len() {
                return Err(format!("entry {} has incompatible autocorrelation count", number + 1));
            }
            let mut autocorrelation = [0.0; LAGS.len()];
            for (slot, token) in autocorrelation.iter_mut().zip(&fields[14..14 + LAGS.len()]) {
                *slot = parse_float(token, "autocorrelation")?;
            }
            let value_start = 14 + LAGS.len();
            let heterogeneity_end = value_start + HETEROGENEITY_SIDES.len() * HETEROGENEITY_VALUES;
            let connectivity_end = heterogeneity_end + 4;
            let mut heterogeneity = [0.0; HETEROGENEITY_SIDES.len() * HETEROGENEITY_VALUES];
            for (slot, token) in heterogeneity.iter_mut().zip(&fields[value_start..heterogeneity_end]) {
                *slot = parse_float(token, "heterogeneity")?;
            }
            let connectivity = Connectivity {
                dense: [parse_float(fields[heterogeneity_end], "dense connectivity")?,
                    parse_float(fields[heterogeneity_end + 1], "dense connectivity")?],
                void: [parse_float(fields[heterogeneity_end + 2], "void connectivity")?,
                    parse_float(fields[heterogeneity_end + 3], "void connectivity")?]};
            let descriptor_end = connectivity_end + descriptor_len();
            let descriptor = fields[connectivity_end..descriptor_end].iter()
                .map(|token| parse_float(token, "descriptor")).collect::<Result<Vec<_>, _>>()?;
            let genome = fields[descriptor_end..].iter()
                .map(|token| parse_float(token, "genome")).collect::<Result<Vec<_>, _>>()?;
            let alive = match fields[12] {
                "0" => false, "1" => true,
                _ => return Err("alive must be 0 or 1".into()),
            };
            let metrics = Metrics {
                mobility: parse_float(fields[7], "mobility")?,
                temporal_variance: parse_float(fields[8], "temporal variance")?,
                robustness: parse_float(fields[9], "robustness")?,
                structure: parse_float(fields[10], "structure")?,
                turnover: parse_float(fields[11], "turnover")?,
                autocorrelation, heterogeneity, connectivity, descriptor, alive };
            if viable(&metrics, &tuning.gates).is_err() {
                return Err(format!("entry {} fails the viability gate", number + 1));
            }
            entries.push(Entry { genome, metrics, admission_novelty, current_novelty, lineage_id,
                parent_lineage_id: (parent != 0).then_some(parent), birth_generation, parent_source });
        }
        if lines.any(|line| !line.trim().is_empty()) { return Err("trailing archive content".into()); }
        entries.sort_by(entry_order);
        let mut archive = Self { tuning: tuning.clone(), entries };
        archive.refresh_current_novelty();
        Ok((archive, tuning))
    }
    /// Read only the header, stopping before the entries. Used to list runs cheaply.
    pub fn header(text: &str) -> Result<Tuning, String> {
        let mut lines = text.lines();
        expect_line(&mut lines, FORMAT_TAG, Some(&FORMAT_VERSION.to_string()))?;
        let tuning = read_tuning(&mut lines)?;
        let count = parse_usize(required_value(&mut lines, "entries")?, "entries")?;
        validate_tuning(&tuning)?;
        if count > tuning.discovery.capacity {
            return Err(format!("entries {count} exceeds capacity {}", tuning.discovery.capacity));
        }
        Ok(tuning)
    }
}

/// Exclude-self novelty over one entry set, shared by the per-generation refresh and the merge.
fn refresh_novelty(entries: &mut [Entry]) {
    let descriptors: Vec<Vec<f32>> = entries.iter().map(|entry| entry.metrics.descriptor.clone()).collect();
    for (index, entry) in entries.iter_mut().enumerate() {
        let others: Vec<Vec<f32>> = descriptors.iter().enumerate()
            .filter(|(other, _)| *other != index).map(|(_, descriptor)| descriptor.clone()).collect();
        entry.current_novelty = novelty(&entry.metrics.descriptor, &others, NEIGHBORS);
    }
}

fn entry_order(left: &Entry, right: &Entry) -> std::cmp::Ordering {
    left.stable_key().cmp(&right.stable_key()).then_with(|| left.lineage_id.cmp(&right.lineage_id))
}

pub fn genome_hash(genome: &[f32]) -> u64 {
    let mut hash = Fnv::new();
    for value in genome { hash.float(*value); }
    hash.finish()
}

fn expect_line(lines: &mut std::str::Lines<'_>, key: &str, value: Option<&str>) -> Result<(), String> {
    let line = lines.next().ok_or_else(|| format!("missing {key} header"))?;
    let fields: Vec<&str> = line.split_whitespace().collect();
    if fields.first() != Some(&key) || value.is_some_and(|expected| fields.get(1) != Some(&expected)) || fields.len() != 2 {
        return Err(format!("expected {key} header"));
    }
    Ok(())
}

fn required_value<'a>(lines: &mut std::str::Lines<'a>, key: &str) -> Result<&'a str, String> {
    let line = lines.next().ok_or_else(|| format!("missing {key} header"))?;
    let fields: Vec<&str> = line.split_whitespace().collect();
    if fields.first() != Some(&key) || fields.len() != 2 { return Err(format!("expected {key} header")); }
    Ok(fields[1])
}

fn parse_float(value: &str, name: &str) -> Result<f32, String> {
    let number = value.parse::<f32>().map_err(|_| format!("invalid {name}"))?;
    number.is_finite().then_some(number).ok_or_else(|| format!("non-finite {name}"))
}
fn parse_usize(value: &str, name: &str) -> Result<usize, String> {
    value.parse().map_err(|_| format!("invalid {name}"))
}
fn parse_u64(value: &str, name: &str) -> Result<u64, String> {
    value.parse().map_err(|_| format!("invalid {name}"))
}

/// Read one key value line per knob, in registry order. A knob cannot be added to Tuning
/// without appearing here, so the header can never silently omit a setting.
fn read_tuning(lines: &mut std::str::Lines<'_>) -> Result<Tuning, String> {
    let mut tuning = Tuning::default();
    for knob in tuning::KNOBS {
        let value = required_value(lines, knob.key)?;
        knob.set(&mut tuning, value)?;
    }
    Ok(tuning)
}

fn validate_tuning(tuning: &Tuning) -> Result<(), String> {
    let world = &tuning.world;
    if world.particle_count == 0 || world.anchor_count < 2 || world.shells == 0 || world.bumps == 0
        || tuning.discovery.capacity == 0
        || !world.radius.is_finite() || world.radius <= 0.0
        || !world.rate.is_finite() || world.rate <= 0.0
    { return Err("archive header contains an invalid world".into()); }
    Ok(())
}
