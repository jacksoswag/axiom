//! Deterministic numeric learning used only to allocate expensive evaluations and rank
//! human-qualified worlds.  Neither model admits archive entries or defines viability.
//!
//! `ponytail:` this deliberately stops at standardized linear models. Add a richer model
//! only after a lineage-held-out report shows that these features leave useful signal behind.

use crate::util::Rng;
use std::cmp::Ordering;

/// Names and order for one versioned numeric feature vector.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FeatureSchema {
    names: Vec<String>,
}

impl FeatureSchema {
    pub fn new(names: impl IntoIterator<Item = impl AsRef<str>>) -> Self {
        FeatureSchema {
            names: names
                .into_iter()
                .map(|name| name.as_ref().to_owned())
                .collect(),
        }
    }

    pub fn width(&self) -> usize {
        self.names.len()
    }

    pub fn names(&self) -> &[String] {
        &self.names
    }

    pub fn accepts(&self, features: &[f32]) -> bool {
        features.len() == self.width() && features.iter().all(|value| value.is_finite())
    }
}

/// Training-set normalization. Constant features remain zero after transformation.
#[derive(Clone, Debug)]
pub struct Standardizer {
    mean: Vec<f32>,
    scale: Vec<f32>,
}

impl Standardizer {
    pub fn fit(
        schema: &FeatureSchema,
        rows: impl IntoIterator<Item = Vec<f32>>,
    ) -> Result<Self, LearnError> {
        let rows: Vec<Vec<f32>> = rows.into_iter().collect();
        if rows.is_empty() {
            return Err(LearnError::Empty);
        }
        if rows.iter().any(|row| !schema.accepts(row)) {
            return Err(LearnError::InvalidFeatures);
        }
        let width = schema.width();
        let mut mean = vec![0.0; width];
        for row in &rows {
            for (slot, value) in mean.iter_mut().zip(row) {
                *slot += *value;
            }
        }
        for value in &mut mean {
            *value /= rows.len() as f32;
        }
        let mut scale = vec![0.0; width];
        for row in &rows {
            for ((slot, value), average) in scale.iter_mut().zip(row).zip(&mean) {
                *slot += (*value - *average).powi(2);
            }
        }
        for value in &mut scale {
            *value = (*value / rows.len() as f32).sqrt().max(1e-6);
        }
        Ok(Standardizer { mean, scale })
    }

    pub fn transform(&self, features: &[f32]) -> Vec<f32> {
        features
            .iter()
            .zip(&self.mean)
            .zip(&self.scale)
            .map(|((value, mean), scale)| (*value - *mean) / *scale)
            .collect()
    }
}

#[derive(Clone, Debug)]
pub struct FeatureRow {
    pub id: u64,
    pub lineage_id: u64,
    pub neighborhood: u64,
    pub features: Vec<f32>,
    pub survives: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Partition {
    Train,
    Calibration,
    Test,
}

/// Rows belong to exactly one deterministic evolutionary-family partition. A lineage is a
/// related mutation family, while `FeatureRow::id` identifies one observation row.
#[derive(Clone, Debug, Default)]
pub struct GroupedSplit {
    pub train: Vec<usize>,
    pub calibration: Vec<usize>,
    pub test: Vec<usize>,
}

impl GroupedSplit {
    pub fn partition(&self, index: usize) -> Option<Partition> {
        if self.train.contains(&index) {
            Some(Partition::Train)
        } else if self.calibration.contains(&index) {
            Some(Partition::Calibration)
        } else if self.test.contains(&index) {
            Some(Partition::Test)
        } else {
            None
        }
    }
}

/// A stable 70/15/15 evolutionary-family split. It never rebalances based on labels.
pub fn lineage_split(rows: &[FeatureRow], seed: u64) -> GroupedSplit {
    let mut split = GroupedSplit::default();
    for (index, row) in rows.iter().enumerate() {
        match partition_for(row.lineage_id, seed) {
            Partition::Train => split.train.push(index),
            Partition::Calibration => split.calibration.push(index),
            Partition::Test => split.test.push(index),
        }
    }
    split
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LogisticConfig {
    pub members: usize,
    pub epochs: usize,
    pub learning_rate: f32,
    pub l2: f32,
    pub min_class_lineages: usize,
    pub min_brier_improvement: f32,
    pub continuation_fraction: f32,
}

impl Default for LogisticConfig {
    fn default() -> Self {
        LogisticConfig {
            members: 7,
            epochs: 250,
            learning_rate: 0.08,
            l2: 0.001,
            min_class_lineages: 5,
            min_brier_improvement: 0.005,
            continuation_fraction: 0.2,
        }
    }
}

#[derive(Clone, Debug)]
struct LogisticModel {
    weights: Vec<f32>,
    bias: f32,
}

impl LogisticModel {
    fn score(&self, features: &[f32]) -> f32 {
        self.bias
            + self
                .weights
                .iter()
                .zip(features)
                .map(|(weight, value)| weight * value)
                .sum::<f32>()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PersistencePrediction {
    pub probability: f32,
    pub uncertainty: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct SchedulingAuthority {
    pub active: bool,
    pub brier: f32,
    pub uniform_brier: f32,
    pub pr_auc: f32,
    pub calibration_error: f32,
    pub precision_at_budget: f32,
    pub recall_at_budget: f32,
    pub positive_lineages: usize,
    pub negative_lineages: usize,
}

#[derive(Clone, Debug)]
pub struct PersistenceEnsemble {
    schema: FeatureSchema,
    scaler: Standardizer,
    members: Vec<LogisticModel>,
    temperature: f32,
    pub authority: SchedulingAuthority,
}

impl PersistenceEnsemble {
    pub fn predict(&self, features: &[f32]) -> Result<PersistencePrediction, LearnError> {
        if !self.schema.accepts(features) {
            return Err(LearnError::InvalidFeatures);
        }
        let features = self.scaler.transform(features);
        let probabilities: Vec<f32> = self
            .members
            .iter()
            .map(|member| sigmoid(member.score(&features) / self.temperature))
            .collect();
        let probability = probabilities.iter().sum::<f32>() / probabilities.len().max(1) as f32;
        let uncertainty = probabilities
            .iter()
            .map(|value| (*value - probability).powi(2))
            .sum::<f32>()
            / probabilities.len().max(1) as f32;
        Ok(PersistencePrediction {
            probability,
            uncertainty: uncertainty.sqrt(),
        })
    }

    pub fn temperature(&self) -> f32 {
        self.temperature
    }
}

/// Fits only on the training partition, calibrates temperature on calibration, and decides
/// scheduling authority using untouched test rows.
pub fn fit_persistence(
    schema: FeatureSchema,
    rows: &[FeatureRow],
    split: &GroupedSplit,
    config: LogisticConfig,
    seed: u64,
) -> Result<PersistenceEnsemble, LearnError> {
    validate_rows(&schema, rows)?;
    let training = take(rows, &split.train);
    if training.is_empty() {
        return Err(LearnError::EmptyTraining);
    }
    let scaler = Standardizer::fit(&schema, training.iter().map(|row| row.features.clone()))?;
    let train: Vec<PreparedRow> = training.iter().map(|row| prepare(row, &scaler)).collect();
    let members = train_ensemble(&train, schema.width(), config, seed)?;
    let calibration: Vec<PreparedRow> = take(rows, &split.calibration)
        .iter()
        .map(|row| prepare(row, &scaler))
        .collect();
    let temperature = choose_temperature(&members, &calibration);
    let test: Vec<PreparedRow> = take(rows, &split.test)
        .iter()
        .map(|row| prepare(row, &scaler))
        .collect();
    let authority = authority_report(&members, temperature, &train, &test, config);
    Ok(PersistenceEnsemble {
        schema,
        scaler,
        members,
        temperature,
        authority,
    })
}

#[derive(Clone)]
struct PreparedRow {
    lineage_id: u64,
    features: Vec<f32>,
    label: f32,
}

fn prepare(row: &FeatureRow, scaler: &Standardizer) -> PreparedRow {
    PreparedRow {
        lineage_id: row.lineage_id,
        features: scaler.transform(&row.features),
        label: if row.survives { 1.0 } else { 0.0 },
    }
}

fn train_ensemble(
    rows: &[PreparedRow],
    width: usize,
    config: LogisticConfig,
    seed: u64,
) -> Result<Vec<LogisticModel>, LearnError> {
    if rows.iter().all(|row| row.label == 0.0) || rows.iter().all(|row| row.label == 1.0) {
        return Err(LearnError::MissingClasses);
    }
    let mut lineages: Vec<u64> = rows.iter().map(|row| row.lineage_id).collect();
    lineages.sort_unstable();
    lineages.dedup();
    let mut models = Vec::with_capacity(config.members);
    for member in 0..config.members.max(1) {
        let mut rng = Rng::new(seed.wrapping_add(member as u64 + 1));
        let sampled: Vec<u64> = (0..lineages.len())
            .map(|_| lineages[rng.below(lineages.len())])
            .collect();
        let mut sample = Vec::new();
        for lineage in sampled {
            sample.extend(rows.iter().filter(|row| row.lineage_id == lineage).cloned());
        }
        if sample.iter().all(|row| row.label == 0.0) || sample.iter().all(|row| row.label == 1.0) {
            sample = rows.to_vec();
        }
        models.push(train_logistic(&sample, width, config));
    }
    Ok(models)
}

fn train_logistic(rows: &[PreparedRow], width: usize, config: LogisticConfig) -> LogisticModel {
    let mut model = LogisticModel {
        weights: vec![0.0; width],
        bias: 0.0,
    };
    for _ in 0..config.epochs {
        let mut gradient = vec![0.0; width];
        let mut bias_gradient = 0.0;
        for row in rows {
            let error = sigmoid(model.score(&row.features)) - row.label;
            for (gradient, feature) in gradient.iter_mut().zip(&row.features) {
                *gradient += error * feature;
            }
            bias_gradient += error;
        }
        let scale = 1.0 / rows.len().max(1) as f32;
        for (weight, gradient) in model.weights.iter_mut().zip(gradient) {
            *weight -= config.learning_rate * (gradient * scale + config.l2 * *weight);
        }
        model.bias -= config.learning_rate * bias_gradient * scale;
    }
    model
}

fn choose_temperature(models: &[LogisticModel], calibration: &[PreparedRow]) -> f32 {
    if calibration.is_empty() || models.is_empty() {
        return 1.0;
    }
    let mut best = (f32::INFINITY, 1.0);
    for step in 1..=32 {
        let temperature = 0.25 + step as f32 * 0.125;
        let loss = calibration
            .iter()
            .map(|row| {
                let logit = mean_logit(models, &row.features) / temperature;
                cross_entropy(sigmoid(logit), row.label)
            })
            .sum::<f32>()
            / calibration.len() as f32;
        if loss < best.0 {
            best = (loss, temperature);
        }
    }
    best.1
}

fn authority_report(
    models: &[LogisticModel],
    temperature: f32,
    train: &[PreparedRow],
    test: &[PreparedRow],
    config: LogisticConfig,
) -> SchedulingAuthority {
    let mut positives = Vec::new();
    let mut negatives = Vec::new();
    for row in test {
        if row.label > 0.5 {
            positives.push(row.lineage_id);
        } else {
            negatives.push(row.lineage_id);
        }
    }
    positives.sort_unstable();
    positives.dedup();
    negatives.sort_unstable();
    negatives.dedup();
    let base = train.iter().map(|row| row.label).sum::<f32>() / train.len().max(1) as f32;
    let uniform_brier = if test.is_empty() {
        f32::INFINITY
    } else {
        test.iter()
            .map(|row| (base - row.label).powi(2))
            .sum::<f32>()
            / test.len() as f32
    };
    let brier = if test.is_empty() {
        f32::INFINITY
    } else {
        test.iter()
            .map(|row| {
                let probability = sigmoid(mean_logit(models, &row.features) / temperature);
                (probability - row.label).powi(2)
            })
            .sum::<f32>()
            / test.len() as f32
    };
    let predictions: Vec<(f32, f32)> = test
        .iter()
        .map(|row| {
            (
                sigmoid(mean_logit(models, &row.features) / temperature),
                row.label,
            )
        })
        .collect();
    let pr_auc = average_precision(&predictions);
    let calibration_error = calibration_error(&predictions);
    let (precision_at_budget, recall_at_budget) =
        recall_at_budget(&predictions, config.continuation_fraction);
    let active = positives.len() >= config.min_class_lineages
        && negatives.len() >= config.min_class_lineages
        && brier + config.min_brier_improvement < uniform_brier
        && recall_at_budget > config.continuation_fraction;
    SchedulingAuthority {
        active,
        brier,
        uniform_brier,
        pr_auc,
        calibration_error,
        precision_at_budget,
        recall_at_budget,
        positive_lineages: positives.len(),
        negative_lineages: negatives.len(),
    }
}

fn average_precision(predictions: &[(f32, f32)]) -> f32 {
    let positives = predictions.iter().filter(|(_, label)| *label > 0.5).count();
    if positives == 0 {
        return 0.0;
    }
    let mut sorted = predictions.to_vec();
    sorted.sort_by(|a, b| descending(a.0, b.0));
    let mut found = 0usize;
    let mut area = 0.0;
    for (rank, (_, label)) in sorted.iter().enumerate() {
        if *label > 0.5 {
            found += 1;
            area += found as f32 / (rank + 1) as f32;
        }
    }
    area / positives as f32
}

fn calibration_error(predictions: &[(f32, f32)]) -> f32 {
    if predictions.is_empty() {
        return 0.0;
    }
    let mut total = 0.0;
    for bin in 0..10 {
        let low = bin as f32 / 10.0;
        let high = (bin + 1) as f32 / 10.0;
        let bucket: Vec<(f32, f32)> = predictions
            .iter()
            .copied()
            .filter(|(probability, _)| *probability >= low && (bin == 9 || *probability < high))
            .collect();
        if !bucket.is_empty() {
            let confidence = bucket
                .iter()
                .map(|(probability, _)| probability)
                .sum::<f32>()
                / bucket.len() as f32;
            let frequency =
                bucket.iter().map(|(_, label)| label).sum::<f32>() / bucket.len() as f32;
            total +=
                (confidence - frequency).abs() * bucket.len() as f32 / predictions.len() as f32;
        }
    }
    total
}

fn recall_at_budget(predictions: &[(f32, f32)], fraction: f32) -> (f32, f32) {
    let positives = predictions.iter().filter(|(_, label)| *label > 0.5).count();
    if predictions.is_empty() || positives == 0 {
        return (0.0, 0.0);
    }
    let mut sorted = predictions.to_vec();
    sorted.sort_by(|a, b| descending(a.0, b.0));
    let budget = ((sorted.len() as f32 * fraction.clamp(0.0, 1.0)).ceil() as usize).max(1);
    let found = sorted
        .iter()
        .take(budget)
        .filter(|(_, label)| *label > 0.5)
        .count();
    (
        found as f32 / budget as f32,
        found as f32 / positives as f32,
    )
}

fn mean_logit(models: &[LogisticModel], features: &[f32]) -> f32 {
    models
        .iter()
        .map(|model| model.score(features))
        .sum::<f32>()
        / models.len().max(1) as f32
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContinuationQuotas {
    pub high_survival: usize,
    pub high_uncertainty: usize,
    pub uniform_neighborhood: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct ContinuationCandidate {
    pub id: u64,
    pub neighborhood: u64,
    pub prediction: PersistencePrediction,
}

/// Fixed quotas make model use auditable. Without authority, every slot is neighborhood-uniform.
pub fn select_continuations(
    candidates: &[ContinuationCandidate],
    quotas: ContinuationQuotas,
    authority_active: bool,
) -> Vec<u64> {
    let slots = quotas.high_survival + quotas.high_uncertainty + quotas.uniform_neighborhood;
    let mut chosen = Vec::new();
    if authority_active {
        let mut by_probability = candidates.to_vec();
        by_probability.sort_by(|a, b| {
            descending(a.prediction.probability, b.prediction.probability).then(a.id.cmp(&b.id))
        });
        take_unique(&mut chosen, &by_probability, quotas.high_survival);
        let mut by_uncertainty = candidates.to_vec();
        by_uncertainty.sort_by(|a, b| {
            descending(a.prediction.uncertainty, b.prediction.uncertainty).then(a.id.cmp(&b.id))
        });
        take_unique(&mut chosen, &by_uncertainty, quotas.high_uncertainty);
        take_uniform_neighborhoods(&mut chosen, candidates, quotas.uniform_neighborhood);
    } else {
        take_uniform_neighborhoods(&mut chosen, candidates, slots);
    }
    if chosen.len() < slots {
        let mut remainder = candidates.to_vec();
        remainder.sort_by_key(|candidate| candidate.id);
        let remaining = slots - chosen.len();
        take_unique(&mut chosen, &remainder, remaining);
    }
    chosen
}

fn take_unique(chosen: &mut Vec<u64>, candidates: &[ContinuationCandidate], count: usize) {
    let target = chosen.len() + count;
    for candidate in candidates {
        if !chosen.contains(&candidate.id) {
            chosen.push(candidate.id);
            if chosen.len() >= target {
                break;
            }
        }
    }
}

fn take_uniform_neighborhoods(
    chosen: &mut Vec<u64>,
    candidates: &[ContinuationCandidate],
    count: usize,
) {
    let target = chosen.len() + count;
    let mut ordered = candidates.to_vec();
    ordered.sort_by_key(|candidate| (candidate.neighborhood, candidate.id));
    loop {
        let before = chosen.len();
        let mut last = None;
        for candidate in &ordered {
            if Some(candidate.neighborhood) == last {
                continue;
            }
            last = Some(candidate.neighborhood);
            if !chosen.contains(&candidate.id) {
                chosen.push(candidate.id);
                if chosen.len() == target {
                    return;
                }
            }
        }
        if chosen.len() == before {
            return;
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LearnError {
    Empty,
    EmptyTraining,
    InvalidFeatures,
    MissingClasses,
}

fn validate_rows(schema: &FeatureSchema, rows: &[FeatureRow]) -> Result<(), LearnError> {
    if rows.iter().all(|row| schema.accepts(&row.features)) {
        Ok(())
    } else {
        Err(LearnError::InvalidFeatures)
    }
}

fn take<'a>(rows: &'a [FeatureRow], indexes: &[usize]) -> Vec<&'a FeatureRow> {
    indexes
        .iter()
        .filter_map(|index| rows.get(*index))
        .collect()
}

fn partition_for(lineage_id: u64, seed: u64) -> Partition {
    match mix(lineage_id ^ seed) % 20 {
        0..=13 => Partition::Train,
        14..=16 => Partition::Calibration,
        _ => Partition::Test,
    }
}

fn mix(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponent = value.exp();
        exponent / (1.0 + exponent)
    }
}

fn cross_entropy(probability: f32, target: f32) -> f32 {
    let probability = probability.clamp(1e-6, 1.0 - 1e-6);
    -target * probability.ln() - (1.0 - target) * (1.0 - probability).ln()
}

fn descending(left: f32, right: f32) -> Ordering {
    right.partial_cmp(&left).unwrap_or(Ordering::Equal)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schema() -> FeatureSchema {
        FeatureSchema::new(["signal", "noise"])
    }

    fn signal_rows() -> Vec<FeatureRow> {
        (0..160u64)
            .map(|lineage_id| {
                let signal = (lineage_id % 16) as f32 - 7.5;
                FeatureRow {
                    id: lineage_id,
                    lineage_id,
                    neighborhood: lineage_id % 5,
                    features: vec![signal, (mix(lineage_id) % 7) as f32],
                    survives: signal > 0.0,
                }
            })
            .collect()
    }

    #[test]
    fn grouped_split_keeps_lineages_whole() {
        let rows = (0..80u64)
            .flat_map(|lineage_id| {
                (0..3).map(move |copy| FeatureRow {
                    id: lineage_id * 10 + copy,
                    lineage_id,
                    neighborhood: 0,
                    features: vec![copy as f32],
                    survives: copy % 2 == 0,
                })
            })
            .collect::<Vec<_>>();
        let split = lineage_split(&rows, 9);
        for lineage in 0..80u64 {
            let memberships: Vec<Partition> = rows
                .iter()
                .enumerate()
                .filter(|(_, row)| row.lineage_id == lineage)
                .filter_map(|(index, _)| split.partition(index))
                .collect();
            assert_eq!(memberships.len(), 3);
            assert!(memberships
                .iter()
                .all(|partition| *partition == memberships[0]));
        }
    }

    #[test]
    fn persistence_signal_beats_uniform_and_gains_authority() {
        let rows = signal_rows();
        let split = lineage_split(&rows, 31);
        let model = fit_persistence(schema(), &rows, &split, LogisticConfig::default(), 4).unwrap();
        assert!(model.authority.active, "{:#?}", model.authority);
        assert!(model.authority.brier < model.authority.uniform_brier);
        let low = model.predict(&[-6.0, 2.0]).unwrap();
        let high = model.predict(&[6.0, 2.0]).unwrap();
        assert!(high.probability > low.probability + 0.5);
    }

    #[test]
    fn useless_persistence_model_loses_authority() {
        let rows: Vec<FeatureRow> = (0..240u64)
            .map(|lineage_id| FeatureRow {
                id: lineage_id,
                lineage_id,
                neighborhood: lineage_id % 3,
                features: vec![1.0, 1.0],
                survives: mix(lineage_id) & 1 == 0,
            })
            .collect();
        let split = lineage_split(&rows, 11);
        let model = fit_persistence(schema(), &rows, &split, LogisticConfig::default(), 2).unwrap();
        assert!(!model.authority.active, "{:#?}", model.authority);
    }

    #[test]
    fn continuation_selection_keeps_fixed_quotas_and_uniform_fallback() {
        let candidates = vec![
            ContinuationCandidate {
                id: 1,
                neighborhood: 0,
                prediction: PersistencePrediction {
                    probability: 0.9,
                    uncertainty: 0.1,
                },
            },
            ContinuationCandidate {
                id: 2,
                neighborhood: 0,
                prediction: PersistencePrediction {
                    probability: 0.8,
                    uncertainty: 0.9,
                },
            },
            ContinuationCandidate {
                id: 3,
                neighborhood: 1,
                prediction: PersistencePrediction {
                    probability: 0.2,
                    uncertainty: 0.2,
                },
            },
            ContinuationCandidate {
                id: 4,
                neighborhood: 2,
                prediction: PersistencePrediction {
                    probability: 0.1,
                    uncertainty: 0.3,
                },
            },
        ];
        let quotas = ContinuationQuotas {
            high_survival: 1,
            high_uncertainty: 1,
            uniform_neighborhood: 1,
        };
        let learned = select_continuations(&candidates, quotas, true);
        assert_eq!(learned.len(), 3);
        assert!(learned.contains(&1));
        assert!(learned.contains(&2));
        let fallback = select_continuations(&candidates, quotas, false);
        assert_eq!(fallback.len(), 3);
        assert_eq!(fallback, vec![1, 3, 4]);
    }

}
