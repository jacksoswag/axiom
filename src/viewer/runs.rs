//! The run library: pick a training run from disk, browse its best results, load one.
//!
//! Replaces a hand-typed path and a slider over two thousand entries. Two things make it
//! useful rather than merely prettier.
//!
//! **Scanning reads headers only.** A 2000-entry archive is several megabytes of floats, so
//! parsing every file in a directory to build a list would stall the frame. The scan uses
//! [`Archive::header`], which stops at the first entry; the full parse happens when a run is
//! actually selected.
//!
//! **Novelty is not offered as a ranking key.** It is the quantity the search optimises, so
//! ordering results by it ranks them by how hard they were looked for rather than by what they
//! are. The three keys here are ones nothing optimises toward. Novelty is still shown on every
//! card, because how isolated a result was when found is worth knowing; it just does not get
//! to decide the order.

use eframe::egui::{RichText, Sense, Ui};
use std::path::PathBuf;

use super::theme;
use crate::tuner::archive::{Archive, Entry};
use crate::tuner::genome::Caps;
use crate::tuner::tuning::Tuning;

/// Results offered per run. Enough to see the range a search found, few enough to judge.
const SHORTLIST: usize = 12;

/// How the shortlist is ordered.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Rank {
    /// Fraction of perturbation shakes survived. The closest thing to "most alive".
    Robustness,
    /// Distance from a uniform gas.
    Structure,
    /// How much the pair statistics move across the tail. Separates churning from settled.
    Variance,
}

impl Rank {
    pub const ALL: [Rank; 3] = [Rank::Robustness, Rank::Structure, Rank::Variance];

    pub fn label(self) -> &'static str {
        match self {
            Rank::Robustness => "robustness",
            Rank::Structure => "structure",
            Rank::Variance => "variance",
        }
    }

    fn of(self, entry: &Entry) -> f32 {
        match self {
            Rank::Robustness => entry.metrics.robustness,
            Rank::Structure => entry.metrics.structure,
            Rank::Variance => entry.metrics.temporal_variance,
        }
    }
}

/// One archive file on disk, described from its header alone.
pub struct Run {
    pub path: PathBuf,
    pub name: String,
    pub tuning: Tuning,
    pub bytes: u64,
}

pub struct Library {
    pub directory: String,
    runs: Vec<Run>,
    selected: Option<usize>,
    shortlist: Vec<Entry>,
    chosen: Option<usize>,
    rank: Rank,
    status: String,
}

impl Default for Library {
    fn default() -> Self {
        Library {
            directory: ".".into(),
            runs: Vec::new(),
            selected: None,
            shortlist: Vec::new(),
            chosen: None,
            rank: Rank::Robustness,
            status: String::new(),
        }
    }
}

impl Library {
    /// List every readable archive in `directory`, cheapest-first.
    ///
    /// A file that is not an archive is skipped in silence, since a directory full of
    /// unrelated text is the normal case. A file that *is* an archive but of the wrong format
    /// is reported, because that is a thing the user did and wants to know about.
    pub fn scan(&mut self) {
        self.runs.clear();
        self.selected = None;
        self.shortlist.clear();
        self.chosen = None;

        let entries = match std::fs::read_dir(&self.directory) {
            Ok(entries) => entries,
            Err(problem) => {
                self.status = format!("{}: {problem}", self.directory);
                return;
            }
        };

        let mut stale = 0usize;
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_none_or(|e| e != "txt") {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&path) else {
                continue;
            };
            if !text.starts_with("axiom-archive") {
                continue;
            }
            match Archive::header(&text) {
                Ok(tuning) => self.runs.push(Run {
                    name: path
                        .file_stem()
                        .map(|s| s.to_string_lossy().into_owned())
                        .unwrap_or_default(),
                    bytes: entry.metadata().map(|m| m.len()).unwrap_or(0),
                    path,
                    tuning,
                }),
                Err(_) => stale += 1,
            }
        }

        self.runs.sort_by(|a, b| a.name.cmp(&b.name));
        self.status = match (self.runs.len(), stale) {
            (0, 0) => format!("no archives in {}", self.directory),
            (found, 0) => format!("{found} runs"),
            (found, old) => format!("{found} runs, {old} from an older format"),
        };
    }

    /// Fully parse one run and build its shortlist.
    fn select(&mut self, index: usize) {
        self.selected = Some(index);
        self.chosen = None;
        self.shortlist.clear();

        let Some(run) = self.runs.get(index) else {
            return;
        };
        let text = match std::fs::read_to_string(&run.path) {
            Ok(text) => text,
            Err(problem) => {
                self.status = problem.to_string();
                return;
            }
        };
        match Archive::from_text(&text) {
            Ok((archive, _)) => {
                self.shortlist = archive.entries().to_vec();
                self.status = format!("{} entries in {}", self.shortlist.len(), run.name);
                self.reorder();
            }
            Err(problem) => self.status = problem,
        }
    }

    fn reorder(&mut self) {
        let rank = self.rank;
        self.shortlist
            .sort_by(|a, b| rank.of(b).total_cmp(&rank.of(a)));
        self.shortlist.truncate(SHORTLIST);
        self.chosen = None;
    }

    /// The world the chosen card asked for, ready to adopt. None until a card is clicked.
    pub fn chosen_world(&self) -> Option<(Caps, Vec<f32>)> {
        let caps = &self.runs.get(self.selected?)?.tuning.world;
        let entry = self.shortlist.get(self.chosen?)?;
        Some((caps.clone(), entry.genome.clone()))
    }

    /// Returns true when a new card was clicked and the app should adopt it.
    pub fn ui(&mut self, ui: &mut Ui) -> bool {
        super::controls::header(ui, "runs");

        ui.horizontal(|ui| {
            ui.add(
                eframe::egui::TextEdit::singleline(&mut self.directory)
                    .desired_width(150.0)
                    .hint_text("directory"),
            );
            if ui.button("scan").clicked() {
                self.scan();
            }
        });

        let mut open = None;
        for (index, run) in self.runs.iter().enumerate() {
            let active = self.selected == Some(index);
            let text = RichText::new(format!(
                "{}   {} particles, {} trait anchors, {} KB",
                run.name,
                run.tuning.world.particle_count,
                run.tuning.world.anchor_count,
                run.bytes / 1024
            ))
            .size(10.5)
            .color(if active {
                theme::ACCENT
            } else {
                theme::TEXT_MUTED
            });
            if ui.selectable_label(active, text).clicked() {
                open = Some(index);
            }
        }
        if let Some(index) = open {
            self.select(index);
        }

        if !self.status.is_empty() {
            ui.label(
                RichText::new(&self.status)
                    .color(theme::TEXT_FAINT)
                    .size(10.0),
            );
        }
        if self.shortlist.is_empty() {
            return false;
        }

        ui.add_space(6.0);
        super::controls::header(ui, "best results");
        ui.horizontal(|ui| {
            ui.label(RichText::new("rank by").color(theme::TEXT_FAINT).size(10.0));
            for option in Rank::ALL {
                if ui
                    .selectable_label(self.rank == option, option.label())
                    .clicked()
                    && self.rank != option
                {
                    self.rank = option;
                    self.reorder();
                }
            }
        });

        self.cards(ui)
    }

    fn cards(&mut self, ui: &mut Ui) -> bool {
        let Some(caps) = self
            .selected
            .and_then(|i| self.runs.get(i))
            .map(|r| r.tuning.world.clone())
        else {
            return false;
        };

        let mut clicked = None;
        for (index, entry) in self.shortlist.iter().enumerate() {
            let coordination = entry.genome.first().copied().unwrap_or(0.0); // archived genomes are bounds-clamped
            let selected = self.chosen == Some(index);

            let response = ui
                .scope(|ui| {
                    let fill = if selected {
                        theme::ACCENT_DIM
                    } else {
                        theme::PANEL_FILL
                    };
                    eframe::egui::Frame::new()
                        .fill(fill)
                        .inner_margin(6.0)
                        .corner_radius(theme::RADIUS_CONTROL)
                        .show(ui, |ui| {
                            ui.set_width(ui.available_width());
                            ui.label(
                                RichText::new(format!(
                                    "#{index}   {} {:.3}",
                                    self.rank.label(),
                                    self.rank.of(entry)
                                ))
                                .color(theme::TEXT)
                                .size(11.0)
                                .strong(),
                            );
                            // Every number that justifies the ranking, on the card. The point
                            // is to judge a result, not to trust an ordering.
                            ui.label(
                                RichText::new(format!(
                                    "robust {:.2}   structure {:.2}   variance {:.4}\n\
                                     novelty {:.2}   coordination {:.1}   anchors {}",
                                    entry.metrics.robustness,
                                    entry.metrics.structure,
                                    entry.metrics.temporal_variance,
                                    entry.current_novelty,
                                    coordination,
                                    caps.anchor_count,
                                ))
                                .color(theme::TEXT_MUTED)
                                .size(10.0),
                            );
                        });
                })
                .response;

            if ui
                .interact(response.rect, ui.next_auto_id(), Sense::click())
                .clicked()
            {
                clicked = Some(index);
            }
            ui.add_space(3.0);
        }

        if let Some(index) = clicked {
            self.chosen = Some(index);
            return true;
        }
        false
    }
}
