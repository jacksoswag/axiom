//! The fixed-three-dimensional explorer. App state is one Caps, one full genome, and the Sim
//! they decode into; everything drawn derives from the sim.

mod camera;
mod controls;
pub mod material; // the reference field semantics, the crate's testable render surface
mod particles;
mod runs;
mod theme;
mod trait_editor;

use eframe::egui::{self, Align2, Key, Order, Rect, RichText, Sense, Vec2};

use crate::engine::resolve::Probe;
use crate::engine::sim::Sim;
use crate::tuner::genome::Caps;
use camera::Camera;
use material::Renderer as MaterialRenderer;
use particles::{Renderer, Style};
use runs::Library;

pub use material::Recipe as MaterialRecipe;

const PANEL_WIDTH: f32 = 320.0;
const RAIL_WIDTH: f32 = 34.0;

pub struct App {
    caps: Caps,
    probe: Probe, // re-measured when caps change; every decode below reuses it
    genome: Vec<f32>,
    sim: Sim,
    camera: Camera,
    style: Style,
    renderer: Renderer,
    material: MaterialRenderer,
    editor: trait_editor::Editor,
    library: Library,
    material_enabled: bool,
    beads_enabled: bool,
    running: bool,
    collapsed: bool,
    steps_per_frame: usize,
    status: String,
}

impl App {
    fn new(ctx: &egui::Context) -> App {
        theme::install(ctx);
        let caps = Caps { particle_count: 320, ..Caps::default() };
        let probe = caps.probe();
        let genome = caps.default_genome(&probe);
        let sim = Sim::new(&caps.params(&genome, &probe));
        let mut camera = Camera::default();
        camera.frame_world(sim.params.box_len);
        App {
            style: Style::default(),
            renderer: Renderer::new(ctx),
            material: MaterialRenderer::new(ctx),
            editor: trait_editor::Editor::default(),
            library: Library::default(),
            camera, caps, probe, genome, sim,
            material_enabled: true,
            beads_enabled: false,
            running: true,
            collapsed: false,
            steps_per_frame: 1,
            status: String::new(),
        }
    }
    /// Reseed the world from the current caps and genome. A shape change resets the genome.
    fn rebuild(&mut self) {
        self.probe = self.caps.probe();
        if self.genome.len() != self.caps.gene_len() {
            self.genome = self.caps.default_genome(&self.probe);
            self.status = "control-net shape changed, genome reset".into();
        }
        self.sim = Sim::new(&self.caps.params(&self.genome, &self.probe));
        self.camera.frame_world(self.sim.params.box_len);
        self.material.reset_history();
    }
    /// The editor changed pair genes: rederive the law and its norms, keep the particles.
    fn refresh_law(&mut self) {
        let mut sim = Sim::new(&self.caps.params(&self.genome, &self.probe));
        sim.substrate.positions = std::mem::take(&mut self.sim.substrate.positions);
        sim.substrate.traits = std::mem::take(&mut self.sim.substrate.traits);
        sim.tick = self.sim.tick;
        self.sim = sim;
    }
    fn adopt_selected(&mut self) {
        let Some((caps, genome)) = self.library.chosen_world() else { return; };
        if genome.len() != caps.gene_len() {
            self.status = "archive entry does not fit its own shape".into();
            return;
        }
        self.caps = caps;
        self.genome = genome;
        self.rebuild();
    }
    fn canvas(&mut self, ui: &mut egui::Ui, dt: f32) -> Rect {
        let viewport = ui.available_rect_before_wrap();
        let response = ui.allocate_rect(viewport, Sense::click_and_drag());
        let painter = ui.painter_at(viewport);
        painter.rect_filled(viewport, 0.0, theme::CANVAS);
        self.navigate(ui, &response, dt);
        if self.running {
            for _ in 0..self.steps_per_frame { self.sim.step(); }
            ui.ctx().request_repaint();
        }
        let material_visible = self.material_enabled
            && self.material.draw(&painter, viewport, material::Scene {
                positions: &self.sim.substrate.positions,
                traits: &self.sim.substrate.traits,
                extent: self.sim.params.box_len,
                tick: self.sim.tick,
                camera: &self.camera,
            }).is_ok();
        if material_visible {
            if self.beads_enabled { self.renderer.draw_beads(&painter, &self.scene(viewport)); }
        } else {
            self.renderer.draw(&painter, &self.scene(viewport));
        }
        viewport
    }
    fn scene(&self, viewport: Rect) -> particles::Scene<'_> {
        particles::Scene { substrate: &self.sim.substrate, camera: &self.camera, viewport, style: &self.style }
    }
    fn navigate(&mut self, ui: &egui::Ui, response: &egui::Response, dt: f32) {
        let box_len = self.sim.params.box_len;
        let ctx = ui.ctx();
        if response.double_clicked() { self.set_capture(ctx, true); }
        if self.camera.captured && ctx.input(|input| input.key_pressed(Key::Escape)) {
            self.set_capture(ctx, false);
        }
        if self.camera.captured {
            let (look, movement, fast) = ctx.input(|input| {
                let axis = |positive: Key, negative: Key| {
                    (input.key_down(positive) as i32 - input.key_down(negative) as i32) as f32
                };
                (input.pointer.delta(),
                    [axis(Key::D, Key::A), axis(Key::E, Key::Q), axis(Key::W, Key::S)],
                    input.modifiers.shift)
            });
            self.camera.fly(look, movement, box_len, dt * if fast { 4.0 } else { 1.0 });
            ctx.request_repaint();
        } else {
            if response.dragged() { self.camera.orbit(response.drag_delta()); }
            if response.hovered() {
                let scroll = ctx.input(|input| input.smooth_scroll_delta.y);
                if scroll != 0.0 { self.camera.dolly(scroll, box_len); }
            }
        }
    }
    fn set_capture(&mut self, ctx: &egui::Context, capture: bool) {
        self.camera.captured = capture;
        ctx.send_viewport_cmd(egui::ViewportCommand::CursorGrab(
            if capture { egui::CursorGrab::Locked } else { egui::CursorGrab::None }));
        ctx.send_viewport_cmd(egui::ViewportCommand::CursorVisible(!capture));
    }
    fn panel(&mut self, ctx: &egui::Context, viewport: Rect) {
        let open = ctx.animate_bool_with_time(egui::Id::new("panel"), !self.collapsed, 0.18);
        let width = RAIL_WIDTH + (PANEL_WIDTH - RAIL_WIDTH) * open;
        let margin = 12.0;
        let rect = Rect::from_min_size(viewport.min + Vec2::splat(margin),
            Vec2::new(width, viewport.height() - margin * 2.0));
        self.renderer.frost(
            &ctx.layer_painter(egui::LayerId::new(Order::Middle, egui::Id::new("frost"))),
            &self.scene(viewport), rect);
        egui::Area::new(egui::Id::new("controls"))
            .order(Order::Foreground)
            .anchor(Align2::LEFT_TOP, Vec2::splat(margin))
            .show(ctx, |ui| {
                ui.set_width(width);
                ui.set_max_height(viewport.height() - margin * 2.0);
                theme::glass().show(ui, |ui| self.panel_contents(ui, open));
            });
    }
    fn panel_contents(&mut self, ui: &mut egui::Ui, open: f32) {
        ui.horizontal(|ui| {
            if ui.button(if self.collapsed { "»" } else { "«" }).clicked() {
                self.collapsed = !self.collapsed;
            }
            if open > 0.5 {
                ui.label(RichText::new("axiom").color(theme::TEXT).size(15.0).strong());
            }
        });
        if open < 0.5 { return; }
        ui.separator();
        ui.horizontal(|ui| {
            if ui.button(if self.running { "pause" } else { "run" }).clicked() {
                self.running = !self.running;
            }
            if ui.button("reseed").clicked() { self.rebuild(); }
            ui.add(egui::DragValue::new(&mut self.steps_per_frame).range(1..=16).prefix("×"));
        });
        egui::ScrollArea::vertical().show(ui, |ui| {
            if controls::world_section(ui, &mut self.caps, &mut self.genome, self.sim.params.box_len) {
                self.rebuild();
                return;
            }
            ui.add_space(6.0);
            controls::projection_section(ui, &mut self.camera);
            ui.add_space(6.0);
            controls::material_section(ui, &mut self.material_enabled, &mut self.beads_enabled, &mut self.material);
            ui.add_space(6.0);
            controls::header(ui, "trait interactions");
            if self.editor.ui(ui, &self.caps, &mut self.genome, &mut self.style, self.sim.params.box_len) {
                self.refresh_law();
            }
            ui.add_space(6.0);
            if self.library.ui(ui) { self.adopt_selected(); }
            if !self.status.is_empty() {
                ui.add_space(6.0);
                ui.label(RichText::new(&self.status).color(theme::TEXT_FAINT).size(10.0));
            }
        });
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let dt = ui.ctx().input(|input| input.stable_dt).clamp(1.0 / 240.0, 1.0 / 15.0);
        let ctx = ui.ctx().clone();
        let viewport = egui::CentralPanel::default()
            .frame(egui::Frame::NONE.fill(theme::CANVAS))
            .show(ui, |ui| self.canvas(ui, dt))
            .inner;
        self.panel(&ctx, viewport);
    }
}

pub fn run() -> eframe::Result<()> {
    eframe::run_native(
        "axiom",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default().with_inner_size([1440.0, 900.0]),
            ..Default::default()
        },
        Box::new(|cc| Ok(Box::new(App::new(&cc.egui_ctx)))),
    )
}

/// Write the exact CPU reference material view as a binary PPM image. This gives search runs
/// a dependency-free visual inspection artifact without introducing a second renderer.
pub fn write_reference_snapshot(path: &std::path::Path, sim: &Sim, recipe: &MaterialRecipe,
    width: usize, height: usize) -> Result<(), String>
{
    if width == 0 || height == 0 { return Err("snapshot dimensions must be positive".into()); }
    let mut camera = Camera::default();
    camera.frame_world(sim.params.box_len);
    camera.focal = height as f32;
    let image = material::reference_image(&sim.substrate.positions, &sim.substrate.traits,
        sim.params.box_len, recipe, &camera, width, height)
        .map_err(|problem| format!("could not render snapshot: {problem:?}"))?;
    let background = theme::CANVAS;
    let mut bytes = format!("P6\n{width} {height}\n255\n").into_bytes();
    bytes.reserve(width * height * 3);
    for pixel in image.pixels {
        let alpha = pixel.a() as u16;
        for (foreground, background) in
            [(pixel.r(), background.r()), (pixel.g(), background.g()), (pixel.b(), background.b())]
        {
            bytes.push(((foreground as u16 * alpha + background as u16 * (255 - alpha) + 127) / 255) as u8);
        }
    }
    std::fs::write(path, bytes).map_err(|problem| problem.to_string())
}
