//! AXIOM CLI.
//!
//!   axiom run   <preset|config.yaml>     live window (turn a knob and watch)
//!   axiom headless <preset|config.yaml>  step headless, dump PNG frames + metrics
//!   axiom validate                       reproduce the Orbium oracle + smoke-test rules
//!   axiom graph                          graph×CA×ML seam demo (spacetime + PageRank)
//!   axiom list                           list bundled presets
//!   axiom dump   <preset>                print a preset's config as YAML (the schema)

use anyhow::{bail, Context, Result};
use axiom::analysis::{circular_centroid, organism_ranking};
use axiom::config::Config;
use axiom::engine::Engine;
use axiom::graph_ca::run_graph_lenia;
use axiom::{presets, render};
use std::io::Write;
use std::path::{Path, PathBuf};

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(|s| s.as_str()).unwrap_or("help");
    match cmd {
        "run" => cmd_run(&args[1..]),
        "headless" => cmd_headless(&args[1..]),
        "validate" => cmd_validate(),
        "learn" => cmd_learn(&args[1..]),
        "qd" => cmd_qd(&args[1..]),
        "particle" => cmd_particle(&args[1..]),
        "hyper" => cmd_hyper(&args[1..]),
        "loaf" => cmd_loaf(&args[1..]),
        "analyze" => cmd_analyze(&args[1..]),
        "similar" => cmd_similar(&args[1..]),
        "gpu" => cmd_gpu(&args[1..]),
        "graph" => cmd_graph(&args[1..]),
        "list" => cmd_list(),
        "dump" => cmd_dump(&args[1..]),
        "help" | "-h" | "--help" => {
            print_usage();
            Ok(())
        }
        other => {
            eprintln!("unknown command '{other}'\n");
            print_usage();
            std::process::exit(2);
        }
    }
}

fn print_usage() {
    println!(
        "AXIOM — configurable CA × graph × ML research engine\n\n\
         USAGE:\n  \
         axiom run <preset|config>        live window (turn a knob and watch)\n  \
         axiom headless <preset|config>   headless run, dump PNGs + metrics\n  \
         axiom validate                   reproduce Orbium oracle + smoke-test every subsystem\n  \
         axiom learn                      train an NCA to imitate Gray-Scott (ES) + world-model rollout\n  \
         axiom qd                         MAP-Elites: illuminate Lenia behavior space\n  \
         axiom gpu                        GPU compute benchmark vs CPU oracle (wgpu)\n  \
         axiom graph                      graph×CA seam (small-world spacetime + PageRank)\n  \
         axiom hyper                      hypergraph CA + hypergraph PageRank\n  \
         axiom particle                   particle swarm + proximity-graph PageRank\n  \
         axiom loaf                       spacetime relaxation: infer occluded time from endpoints\n  \
         axiom list | dump <preset>       list presets / print one as YAML\n\n\
         PRESETS: {}\n\n\
         Rules: lenia, asymptotic_lenia, flow_lenia (mass-conserving), gray_scott, nca (learned)\n",
        presets::NAMES.join(", ")
    );
}

/// Resolve an argument to a Config: a preset name, or a path to a YAML/JSON file.
fn load_cfg(arg: &str) -> Result<Config> {
    if let Some(cfg) = presets::by_name(arg) {
        return Ok(cfg);
    }
    let path = Path::new(arg);
    if path.exists() {
        return Config::load(path);
    }
    bail!("'{arg}' is not a known preset ({}) or an existing config file", presets::NAMES.join(", "));
}

fn flag_val<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).map(|s| s.as_str())
}

fn cmd_run(args: &[String]) -> Result<()> {
    let target = args.first().map(|s| s.as_str()).unwrap_or("orbium");
    let cfg = load_cfg(target)?;
    println!("running '{}' (rule {:?}, {}x{}, provenance {:#018x})",
        cfg.name, rule_tag(&cfg), cfg.substrate.width, cfg.substrate.height, cfg.provenance_hash());
    #[cfg(feature = "window")]
    {
        axiom::viz::run(cfg)
    }
    #[cfg(not(feature = "window"))]
    {
        let _ = cfg;
        bail!("built without the `window` feature — use `axiom headless` instead");
    }
}

fn rule_tag(cfg: &Config) -> &'static str {
    use axiom::config::RuleConfig::*;
    match cfg.rule {
        Lenia(_) => "lenia",
        AsymptoticLenia(_) => "asymptotic_lenia",
        FlowLenia(_) => "flow_lenia",
        GrayScott(_) => "gray_scott",
        Nca(_) => "nca",
    }
}

fn cmd_headless(args: &[String]) -> Result<()> {
    let target = args.first().map(|s| s.as_str()).unwrap_or("orbium");
    let mut cfg = load_cfg(target)?;
    let steps: u64 = flag_val(args, "--steps")
        .and_then(|s| s.parse().ok())
        .or(cfg.steps)
        .unwrap_or(400);
    let every: u64 = flag_val(args, "--every").and_then(|s| s.parse().ok()).unwrap_or((steps / 12).max(1));
    let out = PathBuf::from(flag_val(args, "--out").unwrap_or("out").to_string()).join(&cfg.name);
    if let Some(s) = flag_val(args, "--scale").and_then(|s| s.parse().ok()) {
        cfg.render.scale = s;
    }
    std::fs::create_dir_all(&out).with_context(|| format!("creating {}", out.display()))?;

    let force_cpu = args.iter().any(|a| a == "--cpu");
    let render_cfg = cfg.render.clone();
    let mut engine = Engine::from_config(cfg.clone());
    let mut metrics = std::fs::File::create(out.join("metrics.jsonl"))?;
    let mut frames = 0;

    // GPU is the default execution path for eligible configs; CPU otherwise.
    #[cfg(feature = "gpu")]
    let gpu = if force_cpu { None } else { try_gpu(&cfg, engine.field.h, engine.field.w) };
    #[cfg(feature = "gpu")]
    if let Some(g) = gpu.as_ref() {
        g.upload(engine.field.channel(0));
    }
    let _ = force_cpu;
    #[cfg(feature = "gpu")]
    let backend = if gpu.is_some() { "gpu" } else { "cpu" };
    #[cfg(not(feature = "gpu"))]
    let backend = "cpu";

    println!("headless '{}' [{}] — {} steps, frame every {}, out {}", cfg.name, backend, steps, every, out.display());
    let mut cur: u64 = 0;
    loop {
        #[cfg(feature = "gpu")]
        if let Some(g) = gpu.as_ref() {
            let d = g.read();
            engine.field.channel_mut(0).copy_from_slice(&d);
        }
        let path = out.join(format!("frame_{cur:05}.png"));
        render::save_field_png(&path, &engine.field, render_cfg.channel, &render_cfg.colormap, render_cfg.scale)?;
        frames += 1;
        for (name, rec) in engine.observe() {
            write_metric(&mut metrics, cur, name, &rec)?;
        }
        if cur >= steps {
            break;
        }
        let adv = every.min(steps - cur);
        #[allow(unused_mut)]
        let mut advanced = false;
        #[cfg(feature = "gpu")]
        if let Some(g) = gpu.as_ref() {
            g.advance(adv as usize);
            advanced = true;
        }
        if !advanced {
            for _ in 0..adv {
                engine.step();
            }
        }
        cur += adv;
    }
    std::fs::write(out.join("config.yaml"), cfg.to_yaml()?)?;
    let (cx, cy) = circular_centroid(&engine.field, render_cfg.channel);
    println!(
        "done — {} frames, final mass {:.1}, centroid ({:.1},{:.1})\n  frames + metrics.jsonl + config.yaml in {}",
        frames, engine.field.mass(render_cfg.channel), cx, cy, out.display()
    );
    Ok(())
}

fn write_metric(f: &mut std::fs::File, step: u64, observer: &str, rec: &axiom::analysis::Record) -> Result<()> {
    let mut line = format!("{{\"step\":{step},\"observer\":\"{observer}\"");
    for (k, v) in &rec.scalars {
        line.push_str(&format!(",\"{k}\":{v}"));
    }
    line.push('}');
    writeln!(f, "{line}")?;
    Ok(())
}

fn cmd_validate() -> Result<()> {
    println!("=== AXIOM validation ===\n");
    let out = PathBuf::from("out/validate");
    std::fs::create_dir_all(&out)?;
    let mut all_pass = true;

    // 1. Orbium oracle: alive (bounded mass) + moving (nonzero path length).
    {
        let cfg = presets::orbium();
        let mut engine = Engine::from_config(cfg);
        let init_mass = engine.field.mass(0);
        let (mut px, mut py) = circular_centroid(&engine.field, 0);
        let (mut path, mut min_m, mut max_m) = (0.0f64, init_mass, init_mass);
        let steps = 300u64;
        for _ in 0..steps {
            engine.step();
            let m = engine.field.mass(0);
            min_m = min_m.min(m);
            max_m = max_m.max(m);
            let (cx, cy) = circular_centroid(&engine.field, 0);
            path += toroidal_dist(px, cx, 128.0) + toroidal_dist(py, cy, 128.0);
            px = cx;
            py = cy;
        }
        let final_mass = engine.field.mass(0);
        let alive = final_mass > 0.2 * init_mass && max_m < 5.0 * init_mass;
        let moving = path > 8.0;
        let pass = alive && moving;
        all_pass &= pass;
        render::save_field_png(&out.join("orbium_final.png"), &engine.field, 0, "viridis", 1.0)?;
        println!(
            "[{}] Orbium oracle: init_mass={:.1} final_mass={:.1} mass_band=[{:.1},{:.1}] path={:.1} cells",
            mark(pass), init_mass, final_mass, min_m, max_m, path
        );
        println!("       alive={alive} (bounded, not dead)  moving={moving} (glider translates)");
    }

    // 2. Gray-Scott: a structurally different rule forms patterns via the same engine.
    {
        let cfg = presets::gray_scott();
        let mut engine = Engine::from_config(cfg);
        for _ in 0..4000 {
            engine.step();
        }
        let v = engine.field.mass(1) / (200.0 * 200.0);
        let pass = v > 0.005 && v < 0.9;
        all_pass &= pass;
        render::save_field_png(&out.join("gray_scott_final.png"), &engine.field, 1, "magma", 2.5)?;
        println!("[{}] Gray-Scott rule: mean(v)={:.5} (patterns formed, not blank/saturated)", mark(pass), v);
    }

    // 3. Detection + PageRank observer produces a ranked organism graph.
    {
        let cfg = presets::life();
        let mut engine = Engine::from_config(cfg);
        for _ in 0..40 {
            engine.step();
        }
        let (comps, pr) = organism_ranking(&engine.field, 0.15, 55.0);
        let pass = comps.len() >= 2 && pr.iter().cloned().fold(0.0f32, f32::max) > 0.0;
        all_pass &= pass;
        render::save_field_png(&out.join("life_step40.png"), &engine.field, 0, "turbo", 1.0)?;
        println!("[{}] Detection+PageRank: {} organisms, PR sum={:.3}", mark(pass), comps.len(), pr.iter().sum::<f32>());
    }

    // 4. Graph-mode seam: message-passing CA + PageRank over a small-world graph.
    {
        let run = run_graph_lenia(120, 6, 0.15, 128, 0.2, 0.3, 0.06, 42);
        let pr_sum: f32 = run.pagerank.iter().sum();
        let pr_max = run.pagerank.iter().cloned().fold(0.0f32, f32::max);
        let pr_min = run.pagerank.iter().cloned().fold(1.0f32, f32::min);
        let pass = (pr_sum - 1.0).abs() < 0.05 && pr_max > pr_min;
        all_pass &= pass;
        render::save_matrix_png(&out.join("graph_spacetime.png"), &run.spacetime, run.steps, run.node_count, "turbo")?;
        println!("[{}] Graph seam: PR sum={:.3}, hub/leaf ratio={:.1}x", mark(pass), pr_sum, (pr_max / pr_min.max(1e-6)));
    }

    // 5. Flow Lenia: mass is conserved by construction (its defining property).
    {
        let cfg = presets::flow();
        let mut engine = Engine::from_config(cfg);
        let init_mass = engine.field.mass(0);
        for _ in 0..200 {
            engine.step();
        }
        let final_mass = engine.field.mass(0);
        let drift = (final_mass - init_mass).abs() / init_mass.max(1e-9);
        let pass = drift < 1e-3;
        all_pass &= pass;
        render::save_field_png(&out.join("flow_final.png"), &engine.field, 0, "turbo", 1.0)?;
        println!("[{}] Flow Lenia mass conservation: {:.2} → {:.2} (drift {:.2e})", mark(pass), init_mass, final_mass, drift);
    }

    // 6. Asymptotic Lenia: relaxation stays alive and bounded.
    {
        let cfg = presets::asymptotic();
        let mut engine = Engine::from_config(cfg);
        for _ in 0..200 {
            engine.step();
        }
        let m = engine.field.mass(0) / (160.0 * 160.0);
        let pass = m > 0.001 && m < 0.95;
        all_pass &= pass;
        render::save_field_png(&out.join("asymptotic_final.png"), &engine.field, 0, "magma", 1.0)?;
        println!("[{}] Asymptotic Lenia: mean activation {:.3} (alive, bounded)", mark(pass), m);
    }

    // 7. GPU compute matches the CPU oracle for one Lenia step.
    #[cfg(feature = "gpu")]
    {
        use axiom::config::RuleConfig;
        use axiom::gpu::GpuLenia;
        let cfg = presets::orbium();
        if let RuleConfig::Lenia(l) = &cfg.rule {
            match GpuLenia::from_lenia(128, 128, l, true) {
                Ok(gpu) => {
                    let mut engine = Engine::from_config(cfg.clone());
                    let init = engine.field.channel(0).to_vec();
                    let g1 = gpu.run(&init, 1);
                    engine.step();
                    let diff = g1.iter().zip(engine.field.channel(0)).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
                    let pass = diff < 1e-3;
                    all_pass &= pass;
                    println!("[{}] GPU vs CPU (1 Lenia step): max abs diff {:.2e} on {}", mark(pass), diff, gpu.adapter_name);
                }
                Err(e) => println!("[SKIP] GPU unavailable: {e}"),
            }
        }
    }

    println!("\n{} — artifacts in {}", if all_pass { "ALL PASS" } else { "FAILURES PRESENT" }, out.display());
    if !all_pass {
        std::process::exit(1);
    }
    Ok(())
}

fn toroidal_dist(a: f64, b: f64, n: f64) -> f64 {
    let d = (a - b).abs();
    d.min(n - d)
}

fn mark(pass: bool) -> &'static str {
    if pass { "PASS" } else { "FAIL" }
}

fn cmd_learn(args: &[String]) -> Result<()> {
    use axiom::config::{Config, NcaConfig, RuleConfig, SubstrateConfig};
    use axiom::nca::{rollout_error, train_imitate, train_imitate_grad, Nca};
    use axiom::rule::GrayScottRule;

    let use_es = args.iter().any(|a| a == "--es");
    let gens: usize = flag_val(args, "--gens").and_then(|s| s.parse().ok()).unwrap_or(if use_es { 80 } else { 300 });
    let hidden: usize = flag_val(args, "--hidden").and_then(|s| s.parse().ok()).unwrap_or(24);
    let lr: f32 = flag_val(args, "--lr").and_then(|s| s.parse().ok()).unwrap_or(0.02);
    let size = 48usize;

    // Target: Gray-Scott. Training data: on-distribution states from its own rollout.
    let gs_cfg = presets::gray_scott();
    let gsc = match &gs_cfg.rule {
        RuleConfig::GrayScott(g) => g.clone(),
        _ => bail!("expected gray_scott preset"),
    };
    let target = GrayScottRule::from_config(&gsc);
    let torus = true;

    let mut small = gs_cfg.clone();
    small.substrate.width = size;
    small.substrate.height = size;
    let mut eng = Engine::from_config(small);
    let (n_train, n_val, stride) = (14, 6, 12);
    let mut batch = Vec::new();
    for i in 0..(n_train + n_val) * stride {
        eng.step();
        if i % stride == 0 {
            batch.push(eng.field.clone());
        }
    }
    let (train, val) = batch.split_at(n_train);

    let method = if use_es { "evolution strategies" } else { "gradient descent (Adam, analytic backprop)" };
    println!("Learning a Gray-Scott step with an NCA ({method})");
    println!("  {} params, {} train / {} val states {}x{}, {} {}\n",
        Nca::param_count(2, hidden), n_train, n_val, size, size, gens, if use_es { "generations" } else { "epochs" });

    let report = if use_es {
        train_imitate(&target, torus, train, val, 2, hidden, gens, 40, 0.03, 0.02, 1234)
    } else {
        train_imitate_grad(&target, torus, train, val, 2, hidden, gens, lr, 1234)
    };

    let first = report.loss_history.first().copied().unwrap_or(0.0);
    let last = report.loss_history.last().copied().unwrap_or(0.0);
    println!("  loss {first:.5} → {last:.5}  ({:.1}x lower)   val {:.5}", first / last.max(1e-9), report.val_loss);
    println!("  {}", sparkline(&report.loss_history));

    // World-model evaluation: rollout error vs horizon.
    let nca = Nca::from_theta(2, hidden, 1.0, &report.theta);
    let seed_field = eng.field.clone();
    let errs = rollout_error(&nca, &target, torus, &seed_field, 40);
    println!("\n  rollout prediction MSE vs horizon (learned rule as world model):");
    for h in [1usize, 5, 10, 20, 40] {
        println!("    step {:>3}: {:.5}", h, errs[h - 1]);
    }

    // Persist the learned rule as a runnable config.
    let learned = Config {
        name: "nca_learned".into(),
        schema_version: 1,
        substrate: SubstrateConfig { kind: "grid".into(), width: 200, height: 200, channels: 2, topology: "torus".into() },
        rule: RuleConfig::Nca(NcaConfig { hidden, update_rate: 1.0, weight_seed: 0, weights: Some(report.theta) }),
        init: axiom::config::InitConfig::GrayScottSeed,
        render: axiom::config::RenderConfig { colormap: "magma".into(), channel: 1, scale: 2.5 },
        analysis: vec![],
        steps: Some(2000),
        seed: 11,
    };
    std::fs::create_dir_all("configs")?;
    std::fs::write("configs/nca_learned.yaml", learned.to_yaml()?)?;
    println!("\n  learned rule saved → configs/nca_learned.yaml  (run: axiom run configs/nca_learned.yaml)");
    Ok(())
}

fn cmd_qd(args: &[String]) -> Result<()> {
    let iterations: usize = flag_val(args, "--iters").and_then(|s| s.parse().ok()).unwrap_or(60);
    let bins: usize = flag_val(args, "--bins").and_then(|s| s.parse().ok()).unwrap_or(16);
    let grid = 64usize;
    let steps = 80usize;

    println!("MAP-Elites over Lenia (μ, σ) — behavior space = final mass × mobility");
    println!("  {bins}x{bins} archive, {grid}x{grid} sims, {iterations} iterations × 32 batch\n");
    let report = axiom::qd::run(grid, steps, iterations, 32, bins, 7);

    std::fs::create_dir_all("out/qd")?;
    // Montage: each archive cell → its elite's final field, laid out over behavior space.
    let tile = grid;
    let side = bins * tile;
    let mut rgb = vec![18u8; side * side * 3];
    for by in 0..bins {
        for bx in 0..bins {
            if let Some(e) = &report.archive[by * bins + bx] {
                for y in 0..tile {
                    for x in 0..tile {
                        let [r, g, b] = render::colormap("viridis", e.field[y * tile + x]);
                        // mass axis → rows (top = low), mobility → cols
                        let py = by * tile + y;
                        let px = bx * tile + x;
                        let o = (py * side + px) * 3;
                        rgb[o] = r;
                        rgb[o + 1] = g;
                        rgb[o + 2] = b;
                    }
                }
            }
        }
    }
    render::save_png(Path::new("out/qd/archive.png"), &rgb, side, side)?;

    println!("  coverage: {:.0}% ({} / {} bins filled)", report.coverage() * 100.0, report.archive.iter().flatten().count(), bins * bins);
    if let Some(b) = report.best() {
        println!("  most-structured elite: μ={:.3} σ={:.4}  (mass {:.3}, mobility {:.3})", b.mu, b.sigma, b.mass, b.mobility);
    }
    println!("  archive montage (rows=mass, cols=mobility) → out/qd/archive.png");
    Ok(())
}

struct Descriptors {
    final_mass: f32,
    mean_activity: f32,
    period: f32,
    periodicity: f32,
    entropy: f32,
    h0: usize,
    lineages: usize,
    births: usize,
    deaths: usize,
    vector: Vec<f32>,
}

fn compute_descriptors(cfg: &Config) -> Descriptors {
    use axiom::analysis::connected_components;
    use axiom::dynamics::{activity, count_persistent, dominant_period, h0_persistence, value_entropy, Tracker};

    let mut engine = Engine::from_config(cfg.clone());
    let steps = cfg.steps.unwrap_or(300).min(800) as usize;
    let ch = cfg.render.channel;
    let mut series = Vec::with_capacity(steps);
    let (mut act_sum, mut act_n) = (0.0f32, 0usize);
    let mut prev = engine.field.clone();
    let mut tracker = Tracker::new(20.0);
    let (mut births, mut deaths) = (0usize, 0usize);
    for s in 0..steps {
        engine.step();
        series.push(engine.field.mass(ch) as f32);
        act_sum += activity(&prev, &engine.field);
        act_n += 1;
        prev.data.copy_from_slice(&engine.field.data);
        if s % 3 == 0 {
            let comps = connected_components(&engine.field, ch, 0.15);
            let (_ids, ev) = tracker.update(&comps);
            births += ev.births;
            deaths += ev.deaths;
        }
    }
    let (period, periodicity) = dominant_period(&series);
    let entropy = value_entropy(&engine.field, ch, 32);
    let h0 = count_persistent(&h0_persistence(&engine.field, ch), 0.1);
    let mean_activity = if act_n > 0 { act_sum / act_n as f32 } else { 0.0 };
    let final_mass = engine.field.mass(ch) as f32 / (engine.field.h * engine.field.w) as f32;
    let lineages = tracker.total_lineages();
    let vector = vec![final_mass, mean_activity, period, periodicity, entropy, h0 as f32, lineages as f32];
    Descriptors { final_mass, mean_activity, period, periodicity, entropy, h0, lineages, births, deaths, vector }
}

fn print_descriptors(d: &Descriptors) {
    println!("  final mass (mean activation): {:.4}", d.final_mass);
    println!("  mean activity (per-step change): {:.5}", d.mean_activity);
    println!("  dominant period: {:.1} steps  (periodicity strength {:.2})", d.period, d.periodicity);
    println!("  value entropy: {:.2} bits", d.entropy);
    println!("  H0 persistent features (persistence > 0.1): {}", d.h0);
    println!("  organism lineages: {} total ({} births, {} deaths tracked)", d.lineages, d.births, d.deaths);
}

fn cmd_analyze(args: &[String]) -> Result<()> {
    use axiom::dynamics::{log_experiment, Experiment};
    let target = args.first().map(|s| s.as_str()).unwrap_or("orbium");
    let cfg = load_cfg(target)?;
    println!("Analyzing '{}' ({} steps): spectral · dynamical · topological · tracking\n", cfg.name, cfg.steps.unwrap_or(300));
    let d = compute_descriptors(&cfg);
    print_descriptors(&d);
    let log = PathBuf::from("out/experiments.jsonl");
    log_experiment(&log, &Experiment { hash: cfg.provenance_hash(), name: cfg.name.clone(), descriptor: d.vector })?;
    println!("\n  logged to {} (provenance {:#018x})", log.display(), cfg.provenance_hash());
    Ok(())
}

fn cmd_similar(args: &[String]) -> Result<()> {
    use axiom::dynamics::{find_similar, load_experiments};
    let target = args.first().map(|s| s.as_str()).unwrap_or("orbium");
    let cfg = load_cfg(target)?;
    let d = compute_descriptors(&cfg);
    let exps = load_experiments(Path::new("out/experiments.jsonl"));
    if exps.is_empty() {
        bail!("no experiment log yet — run `axiom analyze <preset>` on a few presets first");
    }
    println!("Descriptor for '{}':", cfg.name);
    print_descriptors(&d);
    println!("\n  nearest logged runs (normalized descriptor distance):");
    for (dist, idx) in find_similar(&exps, &d.vector, 6) {
        println!("    {:>7.3}  {}", dist, exps[idx].name);
    }
    Ok(())
}

fn cmd_loaf(args: &[String]) -> Result<()> {
    use axiom::loaf::Loaf;

    let grid = 48usize;
    let horizon: usize = flag_val(args, "--horizon").and_then(|s| s.parse().ok()).unwrap_or(24);
    let iters: usize = flag_val(args, "--iters").and_then(|s| s.parse().ok()).unwrap_or(1200);
    let react: f32 = flag_val(args, "--reaction").and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let lr: f32 = flag_val(args, "--lr").and_then(|s| s.parse().ok()).unwrap_or(if react > 0.0 { 0.08 } else { 0.15 });
    std::fs::create_dir_all("out/loaf")?;

    // A patterned initial slice (a few gaussian bumps) that diffuses over time.
    let mut init = vec![0.0f32; grid * grid];
    for &(cy, cx, a) in &[(16i32, 16i32, 1.0f32), (32, 30, 0.8), (20, 34, 0.9)] {
        for y in 0..grid as i32 {
            for x in 0..grid as i32 {
                let d2 = ((x - cx).pow(2) + (y - cy).pow(2)) as f32;
                init[(y * grid as i32 + x) as usize] += a * (-d2 / 20.0).exp();
            }
        }
    }

    let truth = Loaf::ground_truth(grid, horizon, 1.0, 0.15, react, init);
    let mut loaf = Loaf { grid, t: horizon, d: 1.0, dt: 0.15, react, vol: truth.vol.clone() };
    loaf.occlude_interior();
    let err_before = loaf.interior_error(&truth);

    let hist = loaf.relax(iters, lr);
    let err_after = loaf.interior_error(&truth);

    let row = grid / 2;
    render::save_matrix_png(Path::new("out/loaf/truth_xt.png"), &truth.xt_slice(row), horizon, grid, "viridis")?;
    let mut occ = Loaf { grid, t: horizon, d: 1.0, dt: 0.15, react, vol: truth.vol.clone() };
    occ.occlude_interior();
    render::save_matrix_png(Path::new("out/loaf/occluded_xt.png"), &occ.xt_slice(row), horizon, grid, "viridis")?;
    render::save_matrix_png(Path::new("out/loaf/recovered_xt.png"), &loaf.xt_slice(row), horizon, grid, "viridis")?;

    let kind = if react > 0.0 { format!("nonlinear (Fisher-KPP reaction r={react})") } else { "linear diffusion".into() };
    println!("Spacetime-loaf [{kind}]: relax a {grid}x{grid}x{horizon} volume to global consistency");
    println!("  fixed endpoints (t=0, t={}), interior inferred by relaxation\n", horizon - 1);
    println!("  energy (residual): {:.2e} → {:.2e}   {}", hist.first().unwrap(), hist.last().unwrap(), sparkline(&hist));
    println!("  interior reconstruction error: {:.4} (occluded) → {:.4} (recovered)", err_before, err_after);
    println!("  {:.0}x closer to the true trajectory — inferred the middle from the ends", err_before / err_after.max(1e-9));
    println!("\n  (x,t) slices → out/loaf/{{truth,occluded,recovered}}_xt.png");
    Ok(())
}

fn cmd_particle(args: &[String]) -> Result<()> {
    use axiom::analysis::pagerank;
    use axiom::particle::Particles;

    let n: usize = flag_val(args, "--n").and_then(|s| s.parse().ok()).unwrap_or(700);
    let steps: usize = flag_val(args, "--steps").and_then(|s| s.parse().ok()).unwrap_or(300);
    let mode = flag_val(args, "--mode").unwrap_or("lenia");
    let world = 128.0f32;
    let grid = 128usize;
    std::fs::create_dir_all("out/particle")?;

    let hidden = 8usize;
    let nca_weights: Vec<f32> = {
        let mut rng = axiom::substrate::Xorshift::new(5);
        (0..Particles::nca_weight_count(hidden)).map(|_| rng.unit() * 2.0 - 1.0).collect()
    };
    let mut p = Particles::new(n, world, 3);
    let every = (steps / 6).max(1);
    for s in 0..=steps {
        if s % every == 0 {
            let f = p.density_field(grid);
            let field = axiom::field::Field { c: 1, h: grid, w: grid, data: f };
            render::save_field_png(&PathBuf::from(format!("out/particle/frame_{s:05}.png")), &field, 0, "turbo", 1.0)?;
        }
        if s < steps {
            match mode {
                "swarm" => p.step(0.5, 6.0, 2.0, 1.0, 3.0, 2.5),
                "nca" => p.step_nca(0.5, 12.0, 5.0, 1.5, &nca_weights, hidden),
                _ => p.step_lenia(0.3, 5.0, 1.5, 4.0, 1.2, 1.0, 2.0),
            }
        }
    }
    let adj = p.proximity_graph(7.0);
    let pr = pagerank(&adj, 0.85, 100);
    let mean_deg = adj.iter().map(|a| a.len()).sum::<usize>() as f32 / n as f32;
    let top = (0..n).max_by(|&a, &b| pr[a].partial_cmp(&pr[b]).unwrap()).unwrap();
    println!("Particle substrate [{mode}]: {n} particles, {steps} steps, torus world {world}");
    println!("  self-organized; proximity graph mean degree {:.1}", mean_deg);
    println!("  most-central particle: #{top} (PageRank {:.4}, {} neighbors)", pr[top], adj[top].len());
    println!("  density frames → out/particle/");
    Ok(())
}

fn cmd_hyper(args: &[String]) -> Result<()> {
    use axiom::hypergraph::run_hyper_ca;

    let n: usize = flag_val(args, "--nodes").and_then(|s| s.parse().ok()).unwrap_or(140);
    let m: usize = flag_val(args, "--edges").and_then(|s| s.parse().ok()).unwrap_or(90);
    let k: usize = flag_val(args, "--arity").and_then(|s| s.parse().ok()).unwrap_or(4);
    let steps: usize = flag_val(args, "--steps").and_then(|s| s.parse().ok()).unwrap_or(160);
    std::fs::create_dir_all("out/hyper")?;

    let run = run_hyper_ca(n, m, k, steps, 0.15, 0.28, 0.09, 17);
    render::save_matrix_png(Path::new("out/hyper/spacetime.png"), &run.spacetime, run.steps, run.n, "turbo")?;

    let mut ranked: Vec<usize> = (0..run.n).collect();
    ranked.sort_by(|&a, &b| run.pagerank[b].partial_cmp(&run.pagerank[a]).unwrap());
    println!("Hypergraph seam: {n} nodes, {m} hyperedges of arity {k}, {steps} CA steps");
    println!("  hypergraph PageRank via two-step random walk (node→hyperedge→node)\n");
    println!("    {:>5}  {:>10}  {:>12}", "node", "pagerank", "hyperedges");
    for &i in ranked.iter().take(8) {
        println!("    {:>5}  {:>10.5}  {:>12}", i, run.pagerank[i], run.degree[i]);
    }
    println!("\n  spacetime diagram → out/hyper/spacetime.png");
    Ok(())
}

fn sparkline(v: &[f32]) -> String {
    let bars = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    // Subsample long series to ~72 columns, and use log scale so orders-of-magnitude
    // decay reads clearly.
    const MAX: usize = 72;
    let sampled: Vec<f32> = if v.len() <= MAX {
        v.to_vec()
    } else {
        (0..MAX).map(|i| v[i * (v.len() - 1) / (MAX - 1)]).collect()
    };
    let logs: Vec<f32> = sampled.iter().map(|&x| (x.max(1e-30)).ln()).collect();
    let (lo, hi) = logs.iter().fold((f32::MAX, f32::MIN), |(a, b), &x| (a.min(x), b.max(x)));
    let range = (hi - lo).max(1e-9);
    logs.iter().map(|&x| bars[(((x - lo) / range) * 7.0).round() as usize]).collect()
}

/// Build a GPU stepper if the config is single-channel, torus, single-kernel
/// gauss-growth Lenia (the case the shader implements).
#[cfg(feature = "gpu")]
fn try_gpu(cfg: &Config, h: usize, w: usize) -> Option<axiom::gpu::GpuLenia> {
    use axiom::config::RuleConfig;
    if cfg.substrate.channels != 1 || !cfg.substrate.torus() {
        return None;
    }
    match &cfg.rule {
        RuleConfig::Lenia(l) => axiom::gpu::GpuLenia::from_lenia(h, w, l, true).ok(),
        _ => None,
    }
}

#[cfg(feature = "gpu")]
fn cmd_gpu(args: &[String]) -> Result<()> {
    use axiom::config::RuleConfig;
    use axiom::gpu::GpuLenia;
    use std::time::Instant;

    let size: usize = flag_val(args, "--size").and_then(|s| s.parse().ok()).unwrap_or(256);
    let steps: usize = flag_val(args, "--steps").and_then(|s| s.parse().ok()).unwrap_or(300);

    let mut cfg = presets::soup();
    cfg.substrate.width = size;
    cfg.substrate.height = size;
    let lenia = match &cfg.rule {
        RuleConfig::Lenia(l) => l.clone(),
        _ => bail!("gpu benchmark expects a single-kernel Lenia preset"),
    };

    let engine = Engine::from_config(cfg.clone());
    let init = engine.field.channel(0).to_vec();
    let gpu = GpuLenia::from_lenia(size, size, &lenia, true)?;

    // Correctness: one GPU step vs one CPU step (same input).
    let g1 = gpu.run(&init, 1);
    let mut cpu = Engine::from_config(cfg.clone());
    cpu.step();
    let max_diff = g1.iter().zip(cpu.field.channel(0)).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);

    gpu.run(&init, 2); // warmup
    let t0 = Instant::now();
    let out = gpu.run(&init, steps);
    let gpu_s = t0.elapsed().as_secs_f64();

    let mut cpu2 = Engine::from_config(cfg.clone());
    let t1 = Instant::now();
    for _ in 0..steps {
        cpu2.step();
    }
    let cpu_s = t1.elapsed().as_secs_f64();

    let cells = (size * size * steps) as f64;
    std::fs::create_dir_all("out/gpu")?;
    let field = axiom::field::Field { c: 1, h: size, w: size, data: out };
    render::save_field_png(Path::new("out/gpu/gpu_final.png"), &field, 0, &cfg.render.colormap, 1.0)?;

    println!("GPU compute — adapter: {}", gpu.adapter_name);
    println!("  grid {size}x{size}, {steps} steps");
    println!("  1-step vs CPU max abs diff: {max_diff:.2e}  ({})", if max_diff < 1e-3 { "match" } else { "MISMATCH" });
    println!("  GPU: {:.3}s  ({:.1} Mcell/s)", gpu_s, cells / gpu_s / 1e6);
    println!("  CPU: {:.3}s  ({:.1} Mcell/s)", cpu_s, cells / cpu_s / 1e6);
    println!("  speedup: {:.1}x", cpu_s / gpu_s);
    println!("  frame saved to out/gpu/gpu_final.png");
    Ok(())
}

#[cfg(not(feature = "gpu"))]
fn cmd_gpu(_args: &[String]) -> Result<()> {
    bail!("built without the `gpu` feature — rebuild with --features gpu");
}

fn cmd_graph(args: &[String]) -> Result<()> {
    let nodes: usize = flag_val(args, "--nodes").and_then(|s| s.parse().ok()).unwrap_or(160);
    let steps: usize = flag_val(args, "--steps").and_then(|s| s.parse().ok()).unwrap_or(200);
    let rewire: f32 = flag_val(args, "--rewire").and_then(|s| s.parse().ok()).unwrap_or(0.08);
    let dt: f32 = flag_val(args, "--dt").and_then(|s| s.parse().ok()).unwrap_or(0.12);
    let mu: f32 = flag_val(args, "--mu").and_then(|s| s.parse().ok()).unwrap_or(0.30);
    let sigma: f32 = flag_val(args, "--sigma").and_then(|s| s.parse().ok()).unwrap_or(0.085);
    let out = PathBuf::from(flag_val(args, "--out").unwrap_or("out/graph"));
    std::fs::create_dir_all(&out)?;

    let run = run_graph_lenia(nodes, 6, rewire, steps, dt, mu, sigma, 42);
    render::save_matrix_png(&out.join("spacetime.png"), &run.spacetime, run.steps, run.node_count, "turbo")?;

    let mut ranked: Vec<usize> = (0..run.node_count).collect();
    ranked.sort_by(|&a, &b| run.pagerank[b].partial_cmp(&run.pagerank[a]).unwrap());

    println!(
        "graph-CA seam: {} nodes, small-world (k=6, rewire={}), {} steps of graph-Lenia\n",
        nodes, rewire, steps
    );
    println!("  spacetime diagram (time↓ × node→) saved to {}", out.join("spacetime.png").display());
    println!("\n  top-10 nodes by PageRank centrality (hubs the dynamics flow through):");
    println!("    {:>5}  {:>10}  {:>6}", "node", "pagerank", "degree");
    for &i in ranked.iter().take(10) {
        println!("    {:>5}  {:>10.5}  {:>6}", i, run.pagerank[i], run.degree[i]);
    }
    Ok(())
}

fn cmd_list() -> Result<()> {
    println!("bundled presets:");
    for name in presets::NAMES {
        let cfg = presets::by_name(name).unwrap();
        println!("  {:<12} {:>4}x{:<4} ch{}  rule={:<10} {}",
            name, cfg.substrate.width, cfg.substrate.height, cfg.substrate.channels, rule_tag(&cfg),
            cfg.analysis.iter().map(desc_analysis).collect::<Vec<_>>().join("+"));
    }
    println!("\nrun one with:  axiom run <name>   |   axiom headless <name>");
    Ok(())
}

fn desc_analysis(a: &axiom::config::AnalysisConfig) -> &'static str {
    use axiom::config::AnalysisConfig::*;
    match a {
        Descriptors => "descriptors",
        Detect { .. } => "detect",
        PageRank { .. } => "pagerank",
    }
}

fn cmd_dump(args: &[String]) -> Result<()> {
    let name = args.first().map(|s| s.as_str()).unwrap_or("orbium");
    let cfg = load_cfg(name)?;
    print!("{}", cfg.to_yaml()?);
    Ok(())
}
