//! Live window (§7.4) — turn a knob and watch. This is where the instrument
//! becomes alive: real-time stepping, false-color display, and interactive
//! controls over presets, colormap, speed, and reseeding.

use crate::config::{Config, RuleConfig};
use crate::engine::Engine;
use crate::presets;
use crate::render;
use anyhow::Result;
use minifb::{Key, KeyRepeat, Scale, Window, WindowOptions};

const CMAPS: [&str; 4] = ["viridis", "turbo", "magma", "gray"];

pub fn run(config: Config) -> Result<()> {
    let mut engine = Engine::from_config(config);
    let (h, w) = (engine.field.h, engine.field.w);
    let up = if h.max(w) <= 160 { 3 } else { 2 };

    let mut window = Window::new(
        "AXIOM",
        w * up,
        h * up,
        WindowOptions { scale: Scale::X1, ..WindowOptions::default() },
    )?;
    window.set_target_fps(60);

    let mut cmap_idx = CMAPS
        .iter()
        .position(|c| *c == engine.config.render.colormap)
        .unwrap_or(0);
    let mut channel = engine.config.render.channel;
    let mut scale = engine.config.render.scale;
    let mut paused = false;
    let mut steps_per_frame: usize = 1;
    let mut buf: Vec<u32> = Vec::new();

    print_controls();

    while window.is_open() && !window.is_key_down(Key::Q) && !window.is_key_down(Key::Escape) {
        for key in window.get_keys_pressed(KeyRepeat::No) {
            match key {
                Key::Space => paused = !paused,
                Key::R => engine.reset_from(engine.config.clone()),
                Key::N => cmap_idx = (cmap_idx + 1) % CMAPS.len(),
                Key::Up => steps_per_frame = (steps_per_frame + 1).min(50),
                Key::Down => steps_per_frame = steps_per_frame.saturating_sub(1).max(1),
                Key::Comma => tune(&mut engine, -0.002, 0.0),
                Key::Period => tune(&mut engine, 0.002, 0.0),
                Key::Semicolon => tune(&mut engine, 0.0, -0.001),
                Key::Apostrophe => tune(&mut engine, 0.0, 0.001),
                Key::LeftBracket => tune_dt(&mut engine, 0.8),
                Key::RightBracket => tune_dt(&mut engine, 1.25),
                Key::Key1 => switch(&mut engine, presets::orbium(), &mut channel, &mut scale, &mut cmap_idx),
                Key::Key2 => switch(&mut engine, presets::soup(), &mut channel, &mut scale, &mut cmap_idx),
                Key::Key3 => switch(&mut engine, presets::life(), &mut channel, &mut scale, &mut cmap_idx),
                Key::Key4 => switch(&mut engine, presets::gray_scott(), &mut channel, &mut scale, &mut cmap_idx),
                Key::S => {
                    let path = format!("axiom_snapshot_{}_{}.png", engine.config.name, engine.step_count);
                    match render::save_field_png(
                        std::path::Path::new(&path),
                        &engine.field,
                        channel,
                        CMAPS[cmap_idx],
                        scale,
                    ) {
                        Ok(_) => println!("saved {path}"),
                        Err(e) => eprintln!("save failed: {e}"),
                    }
                }
                _ => {}
            }
        }

        if !paused {
            for _ in 0..steps_per_frame {
                engine.step();
            }
        }

        render::field_to_argb(&engine.field, channel, CMAPS[cmap_idx], scale, up, &mut buf);
        window.update_with_buffer(&buf, w * up, h * up)?;

        let mass = engine.field.mass(channel);
        let growth = current_growth(&engine.config)
            .map(|(mu, sg)| format!(" · μ {mu:.3} σ {sg:.4}"))
            .unwrap_or_default();
        window.set_title(&format!(
            "AXIOM · {} [{}] · step {} · {}x · mass {:.0}{} · cmap {}{}",
            engine.config.name,
            engine.rule_name(),
            engine.step_count,
            steps_per_frame,
            mass,
            growth,
            CMAPS[cmap_idx],
            if paused { " · PAUSED" } else { "" },
        ));
    }
    Ok(())
}

fn switch(engine: &mut Engine, cfg: Config, channel: &mut usize, scale: &mut f32, cmap_idx: &mut usize) {
    *channel = cfg.render.channel;
    *scale = cfg.render.scale;
    *cmap_idx = CMAPS.iter().position(|c| *c == cfg.render.colormap).unwrap_or(*cmap_idx);
    engine.reset_from(cfg);
}

/// Read the growth (μ, σ) of the first kernel, for Lenia-family rules.
fn current_growth(cfg: &Config) -> Option<(f32, f32)> {
    let k = match &cfg.rule {
        RuleConfig::Lenia(l) | RuleConfig::AsymptoticLenia(l) => l.kernels.first(),
        RuleConfig::FlowLenia(f) => f.base.kernels.first(),
        _ => None,
    }?;
    Some((k.growth.mu, k.growth.sigma))
}

/// Nudge growth (μ, σ) live and rebuild the rule without reseeding — watch the
/// pattern respond to a parameter change in real time.
fn tune(engine: &mut Engine, dmu: f32, dsigma: f32) {
    let Some((mu, sg)) = current_growth(&engine.config) else { return };
    let (mu, sg) = ((mu + dmu).clamp(0.01, 0.5), (sg + dsigma).clamp(0.001, 0.1));
    let mut cfg = engine.config.clone();
    match &mut cfg.rule {
        RuleConfig::Lenia(l) | RuleConfig::AsymptoticLenia(l) => {
            for k in &mut l.kernels {
                k.growth.mu = mu;
                k.growth.sigma = sg;
            }
        }
        RuleConfig::FlowLenia(f) => {
            for k in &mut f.base.kernels {
                k.growth.mu = mu;
                k.growth.sigma = sg;
            }
        }
        _ => {}
    }
    engine.rebuild_rule(cfg);
}

/// Scale the integration timestep live (rebuild the rule, keep the field).
fn tune_dt(engine: &mut Engine, factor: f32) {
    let mut cfg = engine.config.clone();
    let apply = |dt: &mut f32| *dt = (*dt * factor).clamp(0.001, 2.0);
    match &mut cfg.rule {
        RuleConfig::Lenia(l) | RuleConfig::AsymptoticLenia(l) => apply(&mut l.dt),
        RuleConfig::FlowLenia(f) => apply(&mut f.base.dt),
        RuleConfig::GrayScott(g) => apply(&mut g.dt),
        RuleConfig::Nca(_) => return,
    }
    engine.rebuild_rule(cfg);
}

fn print_controls() {
    println!(
        "\nControls:\n  \
         Space  pause/resume        R  reset/reseed\n  \
         N      cycle colormap      S  save PNG snapshot\n  \
         Up/Dn  steps per frame     1/2/3/4  load orbium/soup/life/gray_scott\n  \
         , / .  growth μ down/up    ; / '  growth σ down/up  (live, no reseed)\n  \
         [ / ]  timestep dt down/up  (live, no reseed)\n  \
         Q/Esc  quit\n"
    );
}
