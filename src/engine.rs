//! The engine (§1, §6) — a thin driver that wires configured modules together.
//!
//! `Engine::from_config` is the "registry": it reads the tagged config enums and
//! instantiates the matching trait objects. Stepping is ping-pong (never read the
//! buffer being written, §5.2). Observers run on demand.

use crate::analysis::{build_observers, Observer, Record};
use crate::config::{Config, RuleConfig};
use crate::field::Field;
use crate::presets::apply_init;
use crate::nca::Nca;
use crate::rule::{AsymptoticLeniaRule, FlowLeniaRule, GrayScottRule, LeniaRule, Rule};

pub struct Engine {
    pub field: Field,
    scratch: Field,
    rule: Box<dyn Rule>,
    observers: Vec<Box<dyn Observer>>,
    pub torus: bool,
    pub step_count: u64,
    pub config: Config,
}

impl Engine {
    pub fn from_config(config: Config) -> Engine {
        let s = &config.substrate;
        let mut field = Field::zeros(s.channels, s.height, s.width);
        apply_init(&mut field, &config.init, config.seed);
        let scratch = field.clone();
        let torus = s.torus();
        let rule = build_rule(&config);
        let observers = build_observers(&config.analysis);
        Engine { field, scratch, rule, observers, torus, step_count: 0, config }
    }

    /// Swap in a rule rebuilt from a modified config while keeping the current
    /// field and step count — live parameter tuning (§7.4). Assumes unchanged
    /// substrate dimensions.
    pub fn rebuild_rule(&mut self, config: Config) {
        self.rule = build_rule(&config);
        self.observers = build_observers(&config.analysis);
        self.torus = config.substrate.torus();
        self.config = config;
    }

    pub fn rule_name(&self) -> &'static str {
        self.rule.name()
    }

    pub fn step(&mut self) {
        self.rule.step(&self.field, &mut self.scratch, self.torus);
        std::mem::swap(&mut self.field, &mut self.scratch);
        self.step_count += 1;
    }

    /// Run every configured observer against the current state.
    pub fn observe(&self) -> Vec<(&'static str, Record)> {
        self.observers
            .iter()
            .map(|o| (o.name(), o.observe(&self.field, self.torus)))
            .collect()
    }

    /// Re-seed the field from a (possibly new) config. Used by the live window's
    /// reset / preset keys.
    pub fn reset_from(&mut self, config: Config) {
        *self = Engine::from_config(config);
    }
}

fn build_rule(config: &Config) -> Box<dyn Rule> {
    let s = &config.substrate;
    match &config.rule {
        RuleConfig::Lenia(l) => Box::new(LeniaRule::from_config(l, s.channels)),
        RuleConfig::AsymptoticLenia(l) => Box::new(AsymptoticLeniaRule::from_config(l, s.channels)),
        RuleConfig::FlowLenia(f) => Box::new(FlowLeniaRule::from_config(f)),
        RuleConfig::GrayScott(g) => Box::new(GrayScottRule::from_config(g)),
        RuleConfig::Nca(n) => match &n.weights {
            Some(w) => Box::new(Nca::from_theta(s.channels, n.hidden, n.update_rate, w)),
            None => Box::new(Nca::random(s.channels, n.hidden, n.update_rate, n.weight_seed)),
        },
    }
}
