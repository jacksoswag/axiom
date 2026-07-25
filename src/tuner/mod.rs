//! Measures, gates, scores, and searches the sims the engine runs. Pure logic: nothing here opens a
//! file, prints, or builds a sim, which is run's job. A criterion says what a search wants and a
//! gate says what it demands; together they settle the metric plan, and nothing else can move it.

pub mod algorithms;
pub mod criterion;
pub mod driver;
pub mod gate;
pub mod metrics;
pub mod specimen;

