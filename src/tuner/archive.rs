//! Versioned discovery archive with deterministic, batch-level capacity pruning.

use crate::tuner::metrics::{
    descriptor_len, Connectivity, Metrics, HETEROGENEITY_SIDES, HETEROGENEITY_VALUES, LAGS,
};
use crate::tuner::novelty::{crowding, novelty, NEIGHBOURS};
use crate::tuner::tuning::{self, Tuning};
use crate::tuner::viability::viable;
use crate::util::Fnv;
use std::fmt::Write as _;

const FORMAT_TAG: &str = "axiom-archive";
const FORMAT_VERSION: u32 = 9;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParentSource {
    Bootstrap,
    Novelty,
    Expedition,
    Random,
    Curated,
}

/// Evaluation source tiers are append-only ledger labels.  They do not grant a learned model
/// authority over discovery admission or parent selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceTier {
    Discovery,
    Persistence,
    Certification,
}

#[derive(Clone, Debug)]
pub struct EvaluationRecord {
    pub genome_hash: u64,
    pub lineage_id: u64,
    pub seed: u64,
    pub parent_source: ParentSource,
    pub source_tier: SourceTier,
    pub metrics: Metrics,
    pub gate: Result<(), crate::tuner::viability::Rejection>,
}

/// In-memory append-only evaluation ledger.  Persistence/certification workers can append
/// their exact outcome here without changing archive admission semantics.
#[derive(Clone, Default, Debug)]
pub struct EvaluationLedger {
    records: Vec<EvaluationRecord>,
}

impl EvaluationLedger {
    pub fn records(&self) -> &[EvaluationRecord] {
        &self.records
    }
    pub fn append(&mut self, record: EvaluationRecord) {
        self.records.push(record);
    }
}

impl ParentSource {
    fn code(self) -> &'static str {
        match self {
            Self::Bootstrap => "bootstrap",
            Self::Novelty => "novelty",
            Self::Expedition => "expedition",
            Self::Random => "random",
            Self::Curated => "curated",
        }
    }

    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "bootstrap" => Ok(Self::Bootstrap),
            "novelty" => Ok(Self::Novelty),
            "expedition" => Ok(Self::Expedition),
            "random" => Ok(Self::Random),
            "curated" => Ok(Self::Curated),
            _ => Err(format!("unknown parent source {value:?}")),
        }
    }
}

/// One viable discovered recipe.  Admission novelty is immutable provenance; current novelty
/// is recalculated from the archive frozen at each generation boundary.
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
    pub fn stable_key(&self) -> u64 {
        genome_hash(&self.genome) ^ self.lineage_id.rotate_left(17)
    }
}

/// Candidate admission data collected before the archive transaction.  Search evaluates these
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

/// A discovery archive plus the exact tuning that produced and judged it. Carrying the tuning
/// makes an archive file self-describing: whatever reads it back can reproduce the worlds and
/// knows which gates admitted them.
#[derive(Debug)]
pub struct Archive {
    tuning: Tuning,
    entries: Vec<Entry>,
}

impl Archive {
    pub fn new(tuning: Tuning) -> Self {
        Self {
            tuning,
            entries: Vec::new(),
        }
    }

    pub fn tuning(&self) -> &Tuning {
        &self.tuning
    }

    pub fn capacity(&self) -> usize {
        self.tuning.discovery.capacity.max(1)
    }

    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn descriptors(&self) -> Vec<Vec<f32>> {
        self.entries
            .iter()
            .map(|entry| entry.metrics.descriptor.clone())
            .collect()
    }

    /// Recompute selection novelty against one frozen archive snapshot.  The calculation
    /// excludes the entry itself, which avoids the artificial zero that otherwise dominates a
    /// small archive.
    pub fn refresh_current_novelty(&mut self) {
        let snapshot = self.descriptors();
        for (index, entry) in self.entries.iter_mut().enumerate() {
            let others: Vec<Vec<f32>> = snapshot
                .iter()
                .enumerate()
                .filter(|(other, _)| *other != index)
                .map(|(_, descriptor)| descriptor.clone())
                .collect();
            entry.current_novelty = novelty(&entry.metrics.descriptor, &others, NEIGHBOURS);
        }
    }

    /// Merge every viable candidate in one transaction.  Scoring and tie breaking are based on
    /// the complete combined population, then the retained archive is refreshed again so its
    /// current novelty describes the archive that actually remains.
    pub fn merge(&mut self, admissions: impl IntoIterator<Item = Admission>) -> Vec<Entry> {
        let mut admitted = Vec::new();
        for admission in admissions {
            if viable(&admission.metrics, &self.tuning.gates).is_err()
                || admission.metrics.descriptor.len() != descriptor_len()
                || admission
                    .metrics
                    .descriptor
                    .iter()
                    .any(|value| !value.is_finite())
                || admission.genome.iter().any(|value| !value.is_finite())
                || !admission.admission_novelty.is_finite()
            {
                continue;
            }
            admitted.push(Entry {
                genome: admission.genome,
                metrics: admission.metrics,
                admission_novelty: admission.admission_novelty,
                current_novelty: 0.0,
                lineage_id: admission.lineage_id,
                parent_lineage_id: admission.parent_lineage_id,
                birth_generation: admission.birth_generation,
                parent_source: admission.parent_source,
            });
        }
        if admitted.is_empty() {
            self.refresh_current_novelty();
            return admitted;
        }

        let mut combined = self.entries.clone();
        combined.append(&mut admitted);
        combined.sort_by(entry_order);
        score_combined(&mut combined);
        let descriptors: Vec<Vec<f32>> = combined
            .iter()
            .map(|entry| entry.metrics.descriptor.clone())
            .collect();
        let mut ranked: Vec<(Entry, f32)> = combined
            .into_iter()
            .enumerate()
            .map(|(index, entry)| {
                let others: Vec<Vec<f32>> = descriptors
                    .iter()
                    .enumerate()
                    .filter(|(other, _)| *other != index)
                    .map(|(_, descriptor)| descriptor.clone())
                    .collect();
                (entry.clone(), crowding(&entry.metrics.descriptor, &others))
            })
            .collect();
        ranked.sort_by(|(left, left_crowding), (right, right_crowding)| {
            right
                .current_novelty
                .total_cmp(&left.current_novelty)
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
        for knob in tuning::KNOBS {
            let _ = writeln!(out, "{} {}", knob.key, knob.show(&self.tuning));
        }
        let _ = writeln!(out, "entries {}", self.entries.len());
        for entry in &self.entries {
            let metrics = &entry.metrics;
            let parent = entry.parent_lineage_id.unwrap_or(0);
            let _ = write!(
                out,
                "entry {} {} {} {} {} {} {} {} {} {} {} {} {}",
                entry.lineage_id,
                parent,
                entry.birth_generation,
                entry.parent_source.code(),
                entry.admission_novelty,
                entry.current_novelty,
                metrics.mobility,
                metrics.temporal_variance,
                metrics.robustness,
                metrics.structure,
                metrics.turnover,
                metrics.alive as u8,
                metrics.autocorrelation.len(),
            );
            for value in metrics.autocorrelation {
                let _ = write!(out, " {value}");
            }
            for value in metrics
                .heterogeneity
                .iter()
                .chain(metrics.connectivity.dense.iter())
                .chain(metrics.connectivity.void.iter())
                .chain(metrics.descriptor.iter())
                .chain(entry.genome.iter())
            {
                let _ = write!(out, " {value}");
            }
            let _ = writeln!(out);
        }
        out
    }

    /// Validate and read a complete archive.  The format deliberately has no permissive
    /// migration path: descriptor and genome layout changes must force a rerun.
    pub fn from_text(text: &str) -> Result<(Self, Tuning), String> {
        let mut lines = text.lines();
        expect_line(&mut lines, FORMAT_TAG, Some(&FORMAT_VERSION.to_string()))?;
        let tuning = read_tuning(&mut lines)?;
        let count = parse_usize(required_value(&mut lines, "entries")?, "entries")?;
        validate_tuning(&tuning)?;
        if count > tuning.discovery.capacity {
            return Err(format!(
                "entries {count} exceeds capacity {}",
                tuning.discovery.capacity
            ));
        }
        let metrics_tail = HETEROGENEITY_SIDES.len() * HETEROGENEITY_VALUES + 4;
        let expected_values =
            14 + LAGS.len() + metrics_tail + descriptor_len() + tuning.world.genome_len();
        let mut entries = Vec::with_capacity(count);
        for number in 0..count {
            let line = lines
                .next()
                .ok_or_else(|| format!("missing entry {}", number + 1))?;
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
                return Err(format!(
                    "entry {} has incompatible autocorrelation count",
                    number + 1
                ));
            }
            let mut autocorrelation = [0.0; LAGS.len()];
            for (slot, token) in autocorrelation.iter_mut().zip(&fields[14..14 + LAGS.len()]) {
                *slot = parse_float(token, "autocorrelation")?;
            }
            let value_start = 14 + LAGS.len();
            let heterogeneity_end = value_start + HETEROGENEITY_SIDES.len() * HETEROGENEITY_VALUES;
            let connectivity_end = heterogeneity_end + 4;
            let mut heterogeneity = [0.0; HETEROGENEITY_SIDES.len() * HETEROGENEITY_VALUES];
            for (slot, token) in heterogeneity
                .iter_mut()
                .zip(&fields[value_start..heterogeneity_end])
            {
                *slot = parse_float(token, "heterogeneity")?;
            }
            let connectivity = Connectivity {
                dense: [
                    parse_float(fields[heterogeneity_end], "dense connectivity")?,
                    parse_float(fields[heterogeneity_end + 1], "dense connectivity")?,
                ],
                void: [
                    parse_float(fields[heterogeneity_end + 2], "void connectivity")?,
                    parse_float(fields[heterogeneity_end + 3], "void connectivity")?,
                ],
            };
            let descriptor_end = connectivity_end + descriptor_len();
            let genome = fields[descriptor_end..]
                .iter()
                .map(|token| parse_float(token, "genome"))
                .collect::<Result<Vec<_>, _>>()?;
            let descriptor = fields[connectivity_end..descriptor_end]
                .iter()
                .map(|token| parse_float(token, "descriptor"))
                .collect::<Result<Vec<_>, _>>()?;
            let alive = match fields[12] {
                "0" => false,
                "1" => true,
                _ => return Err("alive must be 0 or 1".into()),
            };
            let metrics = Metrics {
                mobility: parse_float(fields[7], "mobility")?,
                temporal_variance: parse_float(fields[8], "temporal variance")?,
                robustness: parse_float(fields[9], "robustness")?,
                structure: parse_float(fields[10], "structure")?,
                turnover: parse_float(fields[11], "turnover")?,
                autocorrelation,
                heterogeneity,
                connectivity,
                descriptor,
                alive,
            };
            if viable(&metrics, &tuning.gates).is_err() {
                return Err(format!("entry {} fails the viability gate", number + 1));
            }
            entries.push(Entry {
                genome,
                metrics,
                admission_novelty,
                current_novelty,
                lineage_id,
                parent_lineage_id: (parent != 0).then_some(parent),
                birth_generation,
                parent_source,
            });
        }
        if lines.any(|line| !line.trim().is_empty()) {
            return Err("trailing archive content".into());
        }
        entries.sort_by(entry_order);
        let mut archive = Self {
            tuning: tuning.clone(),
            entries,
        };
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
            return Err(format!(
                "entries {count} exceeds capacity {}",
                tuning.discovery.capacity
            ));
        }
        Ok(tuning)
    }
}

fn score_combined(entries: &mut [Entry]) {
    let descriptors: Vec<Vec<f32>> = entries
        .iter()
        .map(|entry| entry.metrics.descriptor.clone())
        .collect();
    for (index, entry) in entries.iter_mut().enumerate() {
        let others: Vec<Vec<f32>> = descriptors
            .iter()
            .enumerate()
            .filter(|(other, _)| *other != index)
            .map(|(_, descriptor)| descriptor.clone())
            .collect();
        entry.current_novelty = novelty(&entry.metrics.descriptor, &others, NEIGHBOURS);
    }
}

fn entry_order(left: &Entry, right: &Entry) -> std::cmp::Ordering {
    left.stable_key()
        .cmp(&right.stable_key())
        .then_with(|| left.lineage_id.cmp(&right.lineage_id))
}

pub fn genome_hash(genome: &[f32]) -> u64 {
    let mut hash = Fnv::new();
    for value in genome {
        hash.float(*value);
    }
    hash.finish()
}

fn expect_line<'a>(
    lines: &mut std::str::Lines<'a>,
    key: &str,
    value: Option<&str>,
) -> Result<(), String> {
    let line = lines
        .next()
        .ok_or_else(|| format!("missing {key} header"))?;
    let fields: Vec<&str> = line.split_whitespace().collect();
    if fields.first() != Some(&key)
        || value.is_some_and(|expected| fields.get(1) != Some(&expected))
        || fields.len() != 2
    {
        return Err(format!("expected {key} header"));
    }
    Ok(())
}

fn required_value<'a>(lines: &mut std::str::Lines<'a>, key: &str) -> Result<&'a str, String> {
    let line = lines
        .next()
        .ok_or_else(|| format!("missing {key} header"))?;
    let fields: Vec<&str> = line.split_whitespace().collect();
    if fields.first() != Some(&key) || fields.len() != 2 {
        return Err(format!("expected {key} header"));
    }
    Ok(fields[1])
}

fn parse_float(value: &str, name: &str) -> Result<f32, String> {
    let number = value
        .parse::<f32>()
        .map_err(|_| format!("invalid {name}"))?;
    number
        .is_finite()
        .then_some(number)
        .ok_or_else(|| format!("non-finite {name}"))
}
fn parse_usize(value: &str, name: &str) -> Result<usize, String> {
    value.parse().map_err(|_| format!("invalid {name}"))
}
fn parse_u64(value: &str, name: &str) -> Result<u64, String> {
    value.parse().map_err(|_| format!("invalid {name}"))
}
/// Read one `key value` line per knob, in registry order. A knob cannot be added to `Tuning`
/// without appearing here, so the header can never silently omit a setting.
fn read_tuning<'a>(lines: &mut std::str::Lines<'a>) -> Result<Tuning, String> {
    let mut tuning = Tuning::default();
    for knob in tuning::KNOBS {
        let value = required_value(lines, knob.key)?;
        knob.set(&mut tuning, value)?;
    }
    Ok(tuning)
}

fn validate_tuning(tuning: &Tuning) -> Result<(), String> {
    let world = &tuning.world;
    if world.particles == 0
        || world.anchors < 2
        || world.shells == 0
        || world.bumps == 0
        || tuning.discovery.capacity == 0
        || !world.radius.is_finite()
        || world.radius <= 0.0
        || !world.rate.is_finite()
        || world.rate <= 0.0
    {
        return Err("archive header contains an invalid world".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_tuning() -> Tuning {
        Tuning {
            world: crate::engine::genome::Caps {
                particles: 20,
                anchors: 2,
                shells: 1,
                bumps: 1,
                ..Default::default()
            },
            discovery: crate::tuner::tuning::Discovery {
                capacity: 4,
                ..Default::default()
            },
            ..Tuning::default()
        }
    }

    fn metrics(descriptor: Vec<f32>) -> Metrics {
        Metrics {
            mobility: 0.02,
            temporal_variance: 0.05,
            autocorrelation: [0.9, 0.7, 0.4],
            turnover: 0.1,
            robustness: 1.0,
            structure: 0.5,
            heterogeneity: [0.2; HETEROGENEITY_SIDES.len() * HETEROGENEITY_VALUES],
            connectivity: Connectivity {
                dense: [1.0; 2],
                void: [1.0; 2],
            },
            descriptor,
            alive: true,
        }
    }
    fn admission(id: u64, x: f32) -> Admission {
        let mut descriptor = vec![1.0; descriptor_len()];
        descriptor[0] = x;
        Admission {
            genome: vec![x],
            metrics: metrics(descriptor),
            lineage_id: id,
            parent_lineage_id: None,
            birth_generation: 1,
            parent_source: ParentSource::Novelty,
            admission_novelty: x.abs(),
        }
    }

    #[test]
    fn round_trip_preserves_metadata_and_refreshes_current_novelty() {
        let tuning = small_tuning();
        let caps = tuning.world.clone();
        let mut archive = Archive::new(tuning);
        let mut first = admission(7, 0.0);
        first.genome = caps.default_genome(crate::tuner::density::widest_bound_len(&caps));
        let mut second = admission(9, 3.0);
        second.genome = caps.default_genome(crate::tuner::density::widest_bound_len(&caps));
        second.genome[0] = 10.0;
        archive.merge([first, second]);
        let text = archive.to_text();
        let (restored, read_tuning) = Archive::from_text(&text).unwrap();
        assert_eq!(read_tuning.world.anchors, 2);
        assert!(restored.entries().iter().any(|entry| entry.lineage_id == 7));
        assert!(restored
            .entries()
            .iter()
            .all(|entry| entry.current_novelty.is_finite()));
    }

    #[test]
    fn admission_novelty_never_changes_when_current_novelty_refreshes() {
        let mut archive = Archive::new(small_tuning());
        archive.merge([admission(1, 0.0), admission(2, 1.0)]);
        let original_admission = archive
            .entries()
            .iter()
            .find(|entry| entry.lineage_id == 1)
            .unwrap()
            .admission_novelty;
        let previous_current = archive
            .entries()
            .iter()
            .find(|entry| entry.lineage_id == 1)
            .unwrap()
            .current_novelty;
        archive.merge([admission(3, 0.1)]);
        let entry = archive
            .entries()
            .iter()
            .find(|entry| entry.lineage_id == 1)
            .unwrap();
        assert_eq!(entry.admission_novelty, original_admission);
        assert_ne!(entry.current_novelty, previous_current);
    }

    #[test]
    fn old_format_is_rejected() {
        assert!(Archive::from_text("axiom-archive 5\n").is_err());
    }

    #[test]
    fn header_does_not_parse_entry_bodies() {
        let tuning = small_tuning();
        let expected = tuning.world.clone();
        let archive = Archive::new(tuning);
        let text = format!("{}entry deliberately malformed\n", archive.to_text());
        let restored = Archive::header(&text).unwrap();
        assert_eq!(restored.world.particles, expected.particles);
        assert_eq!(restored.world.anchors, expected.anchors);
    }
}
