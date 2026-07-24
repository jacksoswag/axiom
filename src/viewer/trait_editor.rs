//! Editing the pair-indexed continuous-trait control net. Works on the pair region of the full
//! genome; the world prefix (coordination, logits) belongs to the world controls.

use eframe::egui::{Color32, Pos2, Rect, Sense, Stroke, Ui, Vec2};
use eframe::epaint::{PathStroke, Shape};

use super::particles::Style;
use super::theme;
use crate::engine::kernel::{strength_and_slope, Shell};
use crate::tuner::genome::Caps;

#[derive(Default)]
pub struct Editor {
    selected: (usize, usize),
}

impl Editor {
    pub fn ui(&mut self, ui: &mut Ui, caps: &Caps, genome: &mut [f32], style: &mut Style, box_len: f32) -> bool {
        if genome.len() != caps.gene_len() { return false; }
        let genes = &mut genome[1 + caps.anchor_count..]; // the pair region; prefix edits live in controls
        let anchor_count = caps.anchor_count;
        self.selected.0 = self.selected.0.min(anchor_count - 1);
        self.selected.1 = self.selected.1.min(anchor_count - 1);
        self.matrix(ui, caps, genes, box_len);
        ui.add_space(5.0);
        let changed = self.interaction(ui, caps, genes, box_len);
        ui.add_space(5.0);
        self.appearance(ui, style);
        changed
    }
    fn matrix(&mut self, ui: &mut Ui, caps: &Caps, genes: &[f32], box_len: f32) {
        let anchor_count = caps.anchor_count;
        let bounds = caps.pair_bounds(box_len);
        let stride = caps.pair_stride();
        ui.label(eframe::egui::RichText::new("trait control net   source -> receiver")
            .color(theme::TEXT_MUTED).size(11.0));
        let cell = 24.0;
        let gap = 3.0;
        let side = anchor_count as f32 * cell + (anchor_count as f32 - 1.0) * gap;
        let (response, painter) = ui.allocate_painter(Vec2::splat(side), Sense::click());
        for source in 0..anchor_count {
            for destination in 0..anchor_count {
                let rect = Rect::from_min_size(
                    response.rect.min + Vec2::new(destination as f32 * (cell + gap), source as f32 * (cell + gap)),
                    Vec2::splat(cell));
                let index = (source * anchor_count + destination) * stride + stride - 1; // the pair's weight gene
                let limit = bounds[index].1.max(1e-3);
                let weight = (genes[index] / limit).clamp(-1.0, 1.0);
                let color = if weight >= 0.0 { theme::ACCENT } else { theme::NEGATIVE };
                painter.rect_filled(rect, theme::RADIUS_CONTROL,
                    Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), (18.0 + 190.0 * weight.abs()) as u8));
                if (source, destination) == self.selected {
                    painter.rect_stroke(rect, theme::RADIUS_CONTROL, Stroke::new(1.5, theme::TEXT),
                        eframe::egui::StrokeKind::Middle);
                }
                if response.clicked()
                    && response.interact_pointer_pos().is_some_and(|point| rect.contains(point))
                { self.selected = (source, destination); }
            }
        }
    }
    fn interaction(&self, ui: &mut Ui, caps: &Caps, genes: &mut [f32], box_len: f32) -> bool {
        let (source, destination) = self.selected;
        let stride = caps.pair_stride();
        let base = (source * caps.anchor_count + destination) * stride;
        let bump_base = base + caps.shells * 3;
        let weight_index = bump_base + caps.bumps * 3;
        let bounds = caps.pair_bounds(box_len);
        let mut changed = false;
        ui.label(eframe::egui::RichText::new(format!("trait anchor {source} -> {destination}"))
            .color(theme::TEXT).size(12.0));
        let (low, high) = bounds[weight_index];
        changed |= ui.add(eframe::egui::Slider::new(&mut genes[weight_index], low..=high)
            .text("weight").fixed_decimals(1)).changed();
        theme::well().show(ui, |ui| {
            let shells = read_mixture(genes, base, caps.shells);
            curve(ui, 56.0, theme::ACCENT, (0.0, box_len * 0.5), |distance| strength_and_slope(distance, &shells).0);
            for shell in 0..caps.shells {
                changed |= triple(ui, genes, &bounds, base + shell * 3, &format!("shell {shell}"));
            }
        });
        theme::well().show(ui, |ui| {
            let bumps = read_mixture(genes, bump_base, caps.bumps);
            // G = 2B - 1: the plot shows the same growth lenia integrates, whose scale line
            // carries the 2x while the -1 dies in its derivative.
            curve(ui, 56.0, theme::TRAIT_STOPS[3], (0.0, 3.0), |density| 2.0 * strength_and_slope(density, &bumps).0 - 1.0);
            for bump in 0..caps.bumps {
                changed |= triple(ui, genes, &bounds, bump_base + bump * 3, &format!("growth {bump}"));
            }
        });
        changed
    }
    fn appearance(&self, ui: &mut Ui, style: &mut Style) {
        ui.add(eframe::egui::Slider::new(&mut style.bead_size, 1.0..=8.0).text("bead size"));
        ui.add(eframe::egui::Slider::new(&mut style.glow, 0.0..=4.0).text("bead glow"));
        ui.checkbox(&mut style.show_frame, "torus frame");
    }
}

fn triple(ui: &mut Ui, genes: &mut [f32], bounds: &[(f32, f32)], at: usize, label: &str) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(eframe::egui::RichText::new(label).color(theme::TEXT_FAINT).size(10.0));
        for (offset, name) in [(0usize, "amp"), (1, "peak"), (2, "width")] {
            let (low, high) = bounds[at + offset];
            changed |= ui.add(eframe::egui::DragValue::new(&mut genes[at + offset])
                .speed((high - low) * 0.004).range(low..=high)
                .prefix(format!("{name} ")).fixed_decimals(2)).changed();
        }
    });
    changed
}

fn curve(ui: &mut Ui, height: f32, color: Color32, domain: (f32, f32), function: impl Fn(f32) -> f32) {
    let (rect, _) = ui.allocate_exact_size(Vec2::new(ui.available_width(), height), Sense::hover());
    const STEPS: usize = 96;
    let samples: Vec<f32> = (0..=STEPS)
        .map(|i| function(domain.0 + (domain.1 - domain.0) * i as f32 / STEPS as f32)).collect();
    let low = samples.iter().copied().fold(f32::MAX, f32::min).min(0.0);
    let high = samples.iter().copied().fold(f32::MIN, f32::max).max(1e-6);
    let span = (high - low).max(1e-6);
    let at_y = |value: f32| rect.bottom() - (value - low) / span * rect.height();
    if low < 0.0 { // zero line, so negative growth reads as below-the-line
        ui.painter().line_segment(
            [Pos2::new(rect.left(), at_y(0.0)), Pos2::new(rect.right(), at_y(0.0))],
            Stroke::new(1.0, Color32::from_white_alpha(0x18)));
    }
    let points = samples.iter().enumerate()
        .map(|(i, &value)| Pos2::new(rect.left() + rect.width() * i as f32 / STEPS as f32, at_y(value)))
        .collect();
    ui.painter().add(Shape::line(points, PathStroke::new(1.5, color)));
}

fn read_mixture(genes: &[f32], base: usize, count: usize) -> Vec<Shell> {
    (0..count).map(|slot| Shell {
        amp: genes[base + slot * 3],
        peak: genes[base + slot * 3 + 1],
        width: genes[base + slot * 3 + 2].max(1e-4),
    }).collect()
}
