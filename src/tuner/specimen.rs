//! Produced by a passed genome run. Carries all metric data and scores needed for tuning

#[derive(Clone)]
pub struct Specimen {
    pub genome: Vec<f32>,
    pub metrics: Vec<f32>, // the descriptor, laid out by the Search's plan; nothing here knows that layout
    pub score: f32,
}
