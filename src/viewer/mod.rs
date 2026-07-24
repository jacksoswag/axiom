//! The fixed-three-dimensional explorer.

mod camera;
mod controls;
mod material;
mod particles;
mod runs;
mod theme;
mod trait_editor;

use eframe::egui::{self, Align2, Key, Order, Rect, RichText, Sense, Vec2};

use crate::engine::world::World;
use crate::engine::genome::Params;
use crate::tuner::density;
use camera::Camera;
use material::Renderer as MaterialRenderer;
use particles::{Renderer, Style};
use runs::Library;

pub use material::Recipe as MaterialRecipe;

const PANEL_WIDTH: f32 = 320.0;
const RAIL_WIDTH: f32 = 34.0;

pub struct App {
    params: Params,
    extent: f32,
    genome: Vec<f32>,
    world: World,
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
        let params = Params {
            particles: 320,
            dimensions: 3,
            coordination: 9.0,
            radius: 12.0,
            rate: 10.0,
            seed: 1,
            anchors: 4,
            shells: 3,
            bumps: 2,
            trait_logits: vec![0.0; 4],
        };
        let extent = density::bound_len_for(&params);
        let genome = params
            .layout()
            .default_genome(params.radius, extent);
        let world = World::with_extent(&params, extent, &genome);
        let mut camera = Camera::default();
        camera.frame_world(extent);
        App {
            style: Style::default(),
            renderer: Renderer::new(ctx),
            material: MaterialRenderer::new(ctx),
            editor: trait_editor::Editor::default(),
            library: Library::default(),
            camera,
            params,
            extent,
            genome,
            world,
            material_enabled: true,
            beads_enabled: false,
            running: true,
            collapsed: false,
            steps_per_frame: 1,
            status: String::new(),
        }
    }

    fn rebuild(&mut self) {
        self.params.trait_logits.resize(self.params.anchors, 0.0);
        self.extent = density::bound_len_for(&self.params);
        let layout = self.params.layout();
        if self.genome.len() != layout.genome_len() {
            self.genome = layout.default_genome(self.params.radius, self.extent);
            self.status = "control-net shape changed, genome reset".into();
        } else {
            layout.clamp(&mut self.genome, self.params.radius, self.extent);
        }
        self.camera.frame_world(self.extent);
        self.world = World::with_extent(&self.params, self.extent, &self.genome);
        self.material.reset_history();
    }

    fn adopt_selected(&mut self) {
        let Some((params, mut genome)) = self.library.chosen_world() else {
            return;
        };
        self.params = params;
        self.extent = density::bound_len_for(&self.params);
        self.params
            .layout()
            .clamp(&mut genome, self.params.radius, self.extent);
        self.genome = genome;
        self.camera.frame_world(self.extent);
        self.world = World::with_extent(&self.params, self.extent, &self.genome);
        self.material.reset_history();
    }

    fn canvas(&mut self, ui: &mut egui::Ui, dt: f32) -> Rect {
        let viewport = ui.available_rect_before_wrap();
        let response = ui.allocate_rect(viewport, Sense::click_and_drag());
        let painter = ui.painter_at(viewport);
        painter.rect_filled(viewport, 0.0, theme::CANVAS);
        self.navigate(ui, &response, dt);
        if self.running {
            for _ in 0..self.steps_per_frame {
                self.world.step();
            }
            ui.ctx().request_repaint();
        }

        let material_visible = self.material_enabled
            && self
                .material
                .draw(
                    &painter,
                    viewport,
                    material::Scene {
                        positions: &self.world.substrate.positions,
                        traits: &self.world.substrate.traits,
                        extent: self.world.substrate.bound_len,
                        tick: self.world.tick,
                        camera: &self.camera,
                    },
                )
                .is_ok();
        if material_visible {
            if self.beads_enabled {
                self.renderer.draw_beads(&painter, &self.scene(viewport));
            }
        } else {
            self.renderer.draw(&painter, &self.scene(viewport));
        }
        viewport
    }

    fn scene(&self, viewport: Rect) -> particles::Scene<'_> {
        particles::Scene {
            substrate: &self.world.substrate,
            camera: &self.camera,
            extent: self.world.substrate.bound_len,
            viewport,
            style: &self.style,
        }
    }

    fn navigate(&mut self, ui: &egui::Ui, response: &egui::Response, dt: f32) {
        let extent = self.world.substrate.bound_len;
        let ctx = ui.ctx();
        if response.double_clicked() {
            self.set_capture(ctx, true);
        }
        if self.camera.captured && ctx.input(|input| input.key_pressed(Key::Escape)) {
            self.set_capture(ctx, false);
        }
        if self.camera.captured {
            let (look, movement, fast) = ctx.input(|input| {
                let axis = |positive: Key, negative: Key| {
                    (input.key_down(positive) as i32 - input.key_down(negative) as i32) as f32
                };
                (
                    input.pointer.delta(),
                    [
                        axis(Key::D, Key::A),
                        axis(Key::E, Key::Q),
                        axis(Key::W, Key::S),
                    ],
                    input.modifiers.shift,
                )
            });
            self.camera
                .fly(look, movement, extent, dt * if fast { 4.0 } else { 1.0 });
            ctx.request_repaint();
        } else {
            if response.dragged() {
                self.camera.orbit(response.drag_delta());
            }
            if response.hovered() {
                let scroll = ctx.input(|input| input.smooth_scroll_delta.y);
                if scroll != 0.0 {
                    self.camera.dolly(scroll, extent);
                }
            }
        }
    }

    fn set_capture(&mut self, ctx: &egui::Context, capture: bool) {
        self.camera.captured = capture;
        ctx.send_viewport_cmd(egui::ViewportCommand::CursorGrab(if capture {
            egui::CursorGrab::Locked
        } else {
            egui::CursorGrab::None
        }));
        ctx.send_viewport_cmd(egui::ViewportCommand::CursorVisible(!capture));
    }

    fn panel(&mut self, ctx: &egui::Context, viewport: Rect) {
        let open = ctx.animate_bool_with_time(egui::Id::new("panel"), !self.collapsed, 0.18);
        let width = RAIL_WIDTH + (PANEL_WIDTH - RAIL_WIDTH) * open;
        let margin = 12.0;
        let rect = Rect::from_min_size(
            viewport.min + Vec2::splat(margin),
            Vec2::new(width, viewport.height() - margin * 2.0),
        );
        self.renderer.frost(
            &ctx.layer_painter(egui::LayerId::new(Order::Middle, egui::Id::new("frost"))),
            &self.scene(viewport),
            rect,
        );
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
                ui.label(
                    RichText::new("axiom")
                        .color(theme::TEXT)
                        .size(15.0)
                        .strong(),
                );
            }
        });
        if open < 0.5 {
            return;
        }
        ui.separator();
        ui.horizontal(|ui| {
            if ui
                .button(if self.running { "pause" } else { "run" })
                .clicked()
            {
                self.running = !self.running;
            }
            if ui.button("reseed").clicked() {
                self.rebuild();
            }
            ui.add(
                egui::DragValue::new(&mut self.steps_per_frame)
                    .range(1..=16)
                    .prefix("×"),
            );
        });
        egui::ScrollArea::vertical().show(ui, |ui| {
            if controls::world_section(ui, &mut self.params, self.extent) {
                self.rebuild();
                return;
            }
            ui.add_space(6.0);
            controls::projection_section(ui, &mut self.camera);
            ui.add_space(6.0);
            controls::material_section(
                ui,
                &mut self.material_enabled,
                &mut self.beads_enabled,
                &mut self.material,
            );
            ui.add_space(6.0);
            controls::header(ui, "trait interactions");
            if self.editor.ui(
                ui,
                &self.params,
                &mut self.genome,
                &mut self.style,
                self.extent,
            ) {
                self.world.set_genome(&self.params, &self.genome);
            }
            ui.add_space(6.0);
            if self.library.ui(ui) {
                self.adopt_selected();
            }
            if !self.status.is_empty() {
                ui.add_space(6.0);
                ui.label(
                    RichText::new(&self.status)
                        .color(theme::TEXT_FAINT)
                        .size(10.0),
                );
            }
        });
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let dt = ui
            .ctx()
            .input(|input| input.stable_dt)
            .clamp(1.0 / 240.0, 1.0 / 15.0);
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
pub fn write_reference_snapshot(
    path: &std::path::Path,
    world: &World,
    recipe: &MaterialRecipe,
    width: usize,
    height: usize,
) -> Result<(), String> {
    if width == 0 || height == 0 {
        return Err("snapshot dimensions must be positive".into());
    }
    let mut camera = Camera::default();
    camera.frame_world(world.substrate.bound_len);
    camera.focal = height as f32;
    let image = material::reference_image(
        &world.substrate.positions,
        &world.substrate.traits,
        world.substrate.bound_len,
        recipe,
        &camera,
        width,
        height,
    )
    .map_err(|problem| format!("could not render snapshot: {problem:?}"))?;
    let background = theme::CANVAS;
    let mut bytes = format!("P6\n{width} {height}\n255\n").into_bytes();
    bytes.reserve(width * height * 3);
    for pixel in image.pixels {
        let alpha = pixel.a() as u16;
        for (foreground, background) in [
            (pixel.r(), background.r()),
            (pixel.g(), background.g()),
            (pixel.b(), background.b()),
        ] {
            bytes.push(
                ((foreground as u16 * alpha + background as u16 * (255 - alpha) + 127) / 255) as u8,
            );
        }
    }
    std::fs::write(path, bytes).map_err(|problem| problem.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn product_defaults_to_three_dimensional_material() {
        let app = App::new(&egui::Context::default());
        assert!(app.material_enabled);
        assert_eq!(app.world.substrate.traits.len(), app.params.particles);
    }

    #[test]
    fn changing_anchor_count_rebuilds_a_matching_control_net() {
        let mut app = App::new(&egui::Context::default());
        app.params.anchors = 3;
        app.rebuild();
        assert_eq!(app.params.trait_logits.len(), 3);
        assert_eq!(app.genome.len(), app.params.layout().genome_len());
    }
}
