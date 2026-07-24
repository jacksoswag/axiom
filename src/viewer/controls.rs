//! Compact controls for the fixed-3-D product world.

use eframe::egui::{RichText, Slider, Ui};

use super::camera::Camera;
use super::material::{Recipe, Renderer as MaterialRenderer};
use super::theme;
use crate::tuner::genome::Caps;

pub fn header(ui: &mut Ui, text: &str) {
    ui.add_space(2.0);
    ui.label(RichText::new(text.to_uppercase()).color(theme::TEXT_FAINT).size(9.5));
    ui.add_space(2.0);
}

/// World settings: caps fields plus the genome's world prefix (coordination, trait logits).
/// Returns true only when the authoritative world must be reseeded.
pub fn world_section(ui: &mut Ui, caps: &mut Caps, genome: &mut [f32], box_len: f32) -> bool {
    let mut rebuild = false;
    header(ui, "world");
    ui.label(RichText::new("3-D periodic world").color(theme::ACCENT).size(10.0));
    rebuild |= ui.add(Slider::new(&mut caps.particle_count, 64..=20_000).text("particles")).changed();
    if !genome.is_empty() {
        rebuild |= ui.add(Slider::new(&mut genome[0], 3.0..=20.0).text("coordination").fixed_decimals(1)).changed();
    }
    ui.label(RichText::new(format!("box {box_len:.1}  derived from density")).color(theme::TEXT_FAINT).size(10.0));
    rebuild |= ui.add(Slider::new(&mut caps.radius, 1.0..=100.0).text("radius")).changed();
    rebuild |= ui.add(Slider::new(&mut caps.rate, 1.0..=200.0).text("T   dt = 1/T")).changed();
    rebuild |= ui.add(Slider::new(&mut caps.anchor_count, 2..=8).text("trait anchors")).changed();
    rebuild |= ui.add(Slider::new(&mut caps.shells, 1..=4).text("kernel shells")).changed();
    rebuild |= ui.add(Slider::new(&mut caps.bumps, 1..=3).text("growth bumps")).changed();
    rebuild |= ui.add(eframe::egui::DragValue::new(&mut caps.seed).prefix("seed ")).changed();

    if genome.len() == caps.gene_len() { // a shape change leaves stale logit slots; rebuild resets them
        header(ui, "initial trait density");
        for (index, logit) in genome[1..=caps.anchor_count].iter_mut().enumerate() {
            rebuild |= ui.add(Slider::new(logit, -4.0..=4.0).text(format!("bin {index}"))).changed();
        }
    }
    rebuild
}

pub fn projection_section(ui: &mut Ui, camera: &mut Camera) {
    header(ui, "flight");
    ui.label(RichText::new(if camera.captured {
            "captured: WASD move, Q/E down/up, shift faster, esc to release"
        } else {
            "drag to orbit, scroll to dolly, double-click to capture"
        })
        .color(if camera.captured { theme::ACCENT } else { theme::TEXT_FAINT })
        .size(10.0));
    ui.add(Slider::new(&mut camera.focal, 200.0..=2400.0).text("focal"));
    ui.add(Slider::new(&mut camera.move_speed, 0.05..=4.0).text("fly speed"));
}

pub fn material_section(ui: &mut Ui, enabled: &mut bool, beads: &mut bool, renderer: &mut MaterialRenderer) {
    header(ui, "render");
    ui.checkbox(enabled, "causal material");
    ui.checkbox(beads, "particle grain");
    let Recipe { resolution, support, iso, absorption } = renderer.recipe_mut();
    ui.add(Slider::new(resolution, 24..=64).text("field resolution"));
    ui.add(Slider::new(support, 1.0..=40.0).text("visual support"));
    ui.add(Slider::new(iso, 0.02..=2.0).text("material threshold"));
    ui.add(Slider::new(absorption, 0.02..=0.8).text("absorption"));
}
