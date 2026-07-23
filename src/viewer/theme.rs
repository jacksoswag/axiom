//! Palette and surface treatments.
//!
//! One accent, one radius scale, one type of surface. Every saturated hue on screen belongs
//! to trait data, so the chrome stays neutral and never competes with the thing being
//! studied. The accent is reserved for selection and active state alone.

use eframe::egui::{Color32, CornerRadius, Frame, Margin, Stroke};
use eframe::epaint::Shadow;

/// Near-black with a blue cast. Not pure `#000000`: pure black flattens the glow falloff
/// and kills the sense of depth the halos are there to create.
pub const CANVAS: Color32 = Color32::from_rgb(0x07, 0x08, 0x0A);

pub const TEXT: Color32 = Color32::from_rgb(0xE8, 0xEA, 0xED);
pub const TEXT_MUTED: Color32 = Color32::from_rgb(0x8A, 0x90, 0x99);
pub const TEXT_FAINT: Color32 = Color32::from_rgb(0x5A, 0x60, 0x69);

/// The single accent. Selection and active state only.
pub const ACCENT: Color32 = Color32::from_rgb(0x5E, 0xE0, 0xC0);
pub const ACCENT_DIM: Color32 = Color32::from_rgba_premultiplied(0x1C, 0x43, 0x3A, 0x8C);

/// Predation reads red, attraction reads teal, in the interaction matrix.
pub const NEGATIVE: Color32 = Color32::from_rgb(0xE0, 0x5E, 0x6E);

pub const PANEL_FILL: Color32 = Color32::from_rgba_premultiplied(0x12, 0x14, 0x18, 0xB8);
pub const PANEL_STROKE: Color32 = Color32::from_rgba_premultiplied(0x1A, 0x1A, 0x1A, 0x1A);
pub const HAIRLINE: Color32 = Color32::from_rgba_premultiplied(0x14, 0x14, 0x14, 0x14);

/// Radius scale. Two values, used consistently: surfaces and controls.
pub const RADIUS_SURFACE: u8 = 10;
pub const RADIUS_CONTROL: u8 = 6;

/// Trait ramp used by the optional true-particle grain and curve editor.
pub const TRAIT_STOPS: [Color32; 8] = [
    Color32::from_rgb(0xFF, 0x5C, 0x5C),
    Color32::from_rgb(0x4F, 0x9C, 0xFF),
    Color32::from_rgb(0x9B, 0xE5, 0x4F),
    Color32::from_rgb(0xFF, 0xC4, 0x4F),
    Color32::from_rgb(0xC4, 0x7C, 0xFF),
    Color32::from_rgb(0xFF, 0x8A, 0x3D),
    Color32::from_rgb(0xFF, 0x6F, 0xC4),
    Color32::from_rgb(0x7A, 0xE0, 0xFF),
];

pub fn trait_color(value: f32) -> Color32 {
    let scaled = value.clamp(0.0, 1.0) * (TRAIT_STOPS.len() - 1) as f32;
    let left = (scaled.floor() as usize).min(TRAIT_STOPS.len() - 1);
    let right = (left + 1).min(TRAIT_STOPS.len() - 1);
    let mix = scaled - left as f32;
    let a = TRAIT_STOPS[left];
    let b = TRAIT_STOPS[right];
    Color32::from_rgb(
        (a.r() as f32 + (b.r() as f32 - a.r() as f32) * mix) as u8,
        (a.g() as f32 + (b.g() as f32 - a.g() as f32) * mix) as u8,
        (a.b() as f32 + (b.b() as f32 - a.b() as f32) * mix) as u8,
    )
}

/// The glass surface.
///
/// egui has no backdrop blur, so this is a translucent fill plus an edge treatment. The
/// blur that makes it read as glass rather than tinted plastic is painted separately by
/// `particles::frost`, which redraws the swarm behind this rect at large radius and low
/// alpha. Labelled an approximation on purpose: this is not Apple's Liquid Glass, which is
/// a platform material with no web or egui equivalent.
pub fn glass() -> Frame {
    Frame::NONE
        .fill(PANEL_FILL)
        .stroke(Stroke::new(1.0, PANEL_STROKE))
        .corner_radius(CornerRadius::same(RADIUS_SURFACE))
        .inner_margin(Margin::symmetric(14, 12))
        .shadow(Shadow {
            offset: [0, 8],
            blur: 28,
            spread: 0,
            color: Color32::from_black_alpha(0x66),
        })
}

/// Inset surface for grouping controls inside the panel.
pub fn well() -> Frame {
    Frame::NONE
        .fill(Color32::from_rgba_premultiplied(0x08, 0x09, 0x0C, 0x9C))
        .stroke(Stroke::new(1.0, HAIRLINE))
        .corner_radius(CornerRadius::same(RADIUS_CONTROL))
        .inner_margin(Margin::symmetric(10, 8))
}

/// Apply the palette to egui's own widget styling so built-in controls match.
pub fn install(ctx: &eframe::egui::Context) {
    // 0.35 keeps a style per theme, so pin the preference and then mutate both.
    ctx.set_theme(eframe::egui::ThemePreference::Dark);
    ctx.all_styles_mut(|style| {
        let v = &mut style.visuals;

        v.dark_mode = true;
        v.panel_fill = CANVAS;
        v.window_fill = PANEL_FILL;
        v.extreme_bg_color = Color32::from_rgb(0x04, 0x05, 0x07);
        v.override_text_color = Some(TEXT);
        v.selection.bg_fill = ACCENT_DIM;
        v.selection.stroke = Stroke::new(1.0, ACCENT);
        v.hyperlink_color = ACCENT;

        for widget in [
            &mut v.widgets.noninteractive,
            &mut v.widgets.inactive,
            &mut v.widgets.hovered,
            &mut v.widgets.active,
            &mut v.widgets.open,
        ] {
            widget.corner_radius = CornerRadius::same(RADIUS_CONTROL);
        }
        v.widgets.noninteractive.bg_stroke = Stroke::new(1.0, HAIRLINE);
        v.widgets.inactive.bg_fill = Color32::from_rgba_premultiplied(0x1E, 0x21, 0x27, 0xC8);
        v.widgets.inactive.weak_bg_fill = Color32::from_rgba_premultiplied(0x18, 0x1B, 0x20, 0xC8);
        v.widgets.hovered.bg_fill = Color32::from_rgba_premultiplied(0x2A, 0x2E, 0x36, 0xD8);
        v.widgets.hovered.bg_stroke = Stroke::new(1.0, Color32::from_white_alpha(0x22));
        v.widgets.active.bg_fill = ACCENT_DIM;
        v.widgets.active.bg_stroke = Stroke::new(1.0, ACCENT);

        style.spacing.item_spacing = eframe::egui::vec2(8.0, 7.0);
        style.spacing.slider_width = 150.0;
        style.spacing.interact_size.y = 20.0;
    });
}
