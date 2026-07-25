//! The fast suite: one promise per case, no child processes, no sleeps, nothing that has to be read
//! out of a report to be understood. A case here earns its place by failing for a reason that
//! matters, and the whole target runs in seconds.
//!
//! cargo test --test smoke

mod fixture;

mod engine;
mod harness;
mod tuner;
