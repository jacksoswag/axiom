//! What this machine costs to run: engine stepping, the measurement battery, per-candidate setup,
//! and one real search generation, written into tests/reports/DATE_perf.md. Ignored by default,
//! because it is minutes of real simulation rather than a correctness check.
//!
//! tests/run.sh --perf, or cargo test --release --test performance -- --ignored --nocapture

mod cases;

use std::fmt::Write;
use std::process::Command;
use std::time::{Duration, Instant};

/// One measured case. runs is how many times the body went round before the floor was reached, so a
/// microsecond case and a ten-second one both report a mean worth reading.
pub struct Timing {
    pub group: &'static str,
    pub name: String,
    pub runs: usize,
    pub each: Duration,
    pub rate: String, // whatever unit makes this case comparable across sizes, or empty
}

/// Run body until it has spent at least this long, then report the mean. One run of a microsecond
/// case measures the clock; a fixed count measures nothing at all when the case is a whole search.
const FLOOR: Duration = Duration::from_millis(400);

/// Time a case. The body hands back a number that lands in a black box, so nothing measured here can
/// be optimized away for having no reader.
pub fn time(group: &'static str, name: impl Into<String>, rate: impl Into<String>,
    mut body: impl FnMut() -> f64) -> Timing
{
    let opened = Instant::now();
    let (mut runs, mut sink) = (0usize, 0.0);
    while opened.elapsed() < FLOOR {
        sink += body();
        runs += 1;
    }
    let each = opened.elapsed() / runs as u32;
    std::hint::black_box(sink);
    let timing = Timing { group, name: name.into(), runs, each, rate: rate.into() };
    println!("  {:<44} {:>12}  x{}  {}", timing.name, span(timing.each), timing.runs, timing.rate);
    timing
}

/// A duration at a readable scale. Nanoseconds for a pair, seconds for a generation, and never a
/// column of zeroes or a column of nine-digit integers.
pub fn span(each: Duration) -> String {
    let nanos = each.as_secs_f64() * 1e9;
    if nanos < 1_000.0 { format!("{nanos:.0} ns") }
    else if nanos < 1_000_000.0 { format!("{:.2} us", nanos / 1e3) }
    else if nanos < 1_000_000_000.0 { format!("{:.2} ms", nanos / 1e6) }
    else { format!("{:.3} s", nanos / 1e9) }
}

/// Work per second, for the cases whose whole point is a rate: pairs, steps, particle-steps.
pub fn rate(count: f64, each: Duration) -> String {
    let per_second = count / each.as_secs_f64();
    if per_second > 1e9 { format!("{:.2}B/s", per_second / 1e9) }
    else if per_second > 1e6 { format!("{:.1}M/s", per_second / 1e6) }
    else if per_second > 1e3 { format!("{:.0}K/s", per_second / 1e3) }
    else { format!("{per_second:.2}/s") } // a generation a minute is a rate too, and rounded to K it read zero
}

/// The whole suite, as one test, because the cases share a fixture and the report wants all of them.
/// Splitting it into a test apiece would let cargo run them side by side, and every number here is a
/// wall-clock reading that assumes it has the machine.
#[test]
#[ignore = "minutes of real simulation; run it deliberately"]
fn performance() {
    let mut timings = Vec::new();
    println!("\nengine");
    timings.extend(cases::engine());
    println!("measurement");
    timings.extend(cases::measurement());
    println!("setup");
    timings.extend(cases::setup());
    println!("search");
    timings.extend(cases::search());
    println!("harness");
    timings.extend(cases::harness());

    let path = format!("{}/tests/reports/{}_perf.md", env!("CARGO_MANIFEST_DIR"), today());
    std::fs::create_dir_all(format!("{}/tests/reports", env!("CARGO_MANIFEST_DIR"))).ok();
    match std::fs::write(&path, report(&timings, &cases::determinism())) {
        Ok(()) => println!("\nwrote {path}"),
        Err(problem) => println!("\ncould not write {path}: {problem}"),
    }
}

/// The report, one table per group, in the order the cases ran.
fn report(timings: &[Timing], determinism: &str) -> String {
    let mut out = String::new();
    writeln!(out, "# Performance report, {}\n", today()).ok();
    writeln!(out, "{}\n", machine()).ok();
    writeln!(out, "Default shape unless a case says otherwise: 320 particles, 3 anchors, 2 shells,").ok();
    writeln!(out, "1 bump, radius 2, dt 0.05. Every case runs until it has spent {} ms and reports the", FLOOR.as_millis()).ok();
    writeln!(out, "mean, so the run count in each row is the sample size behind it.\n").ok();
    let mut group = "";
    for timing in timings {
        if timing.group != group {
            group = timing.group;
            writeln!(out, "\n## {group}\n").ok();
            writeln!(out, "| case | each | runs | rate |").ok();
            writeln!(out, "|---|---:|---:|---:|").ok();
        }
        writeln!(out, "| {} | {} | {} | {} |", timing.name, span(timing.each), timing.runs, timing.rate).ok();
    }
    writeln!(out, "\n## determinism\n\n{determinism}").ok();
    out
}

/// What the numbers were taken on, since a report without it is a number without a unit. The state of
/// the machine travels too, and it is not decoration: a run taken while something else had the cores,
/// or while a hot or nearly flat laptop was being throttled, comes out several times slower across
/// every row at once, and the only way to tell that from a real regression is to have written it down.
fn machine() -> String {
    let ask = |args: &[&str]| Command::new(args[0]).args(&args[1..]).output().ok()
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .map(|text| text.trim().to_string()).unwrap_or_default();
    let profile = if cfg!(debug_assertions) { "debug (run with --release for numbers worth keeping)" }
        else { "release" };
    let load = ask(&["sysctl", "-n", "vm.loadavg"]);
    let power = ask(&["pmset", "-g", "batt"]).lines().last().unwrap_or("").trim().to_string();
    format!("Machine: {} , {} logical cores. Profile: {}.\n\nAt the time of the run: load {}, power {}.",
        ask(&["sysctl", "-n", "machdep.cpu.brand_string"]), ask(&["sysctl", "-n", "hw.ncpu"]),
        profile, load, power)
}

/// The date the numbers were taken, in the reader's own timezone. std has a clock and no calendar,
/// and the system has both.
fn today() -> String {
    Command::new("date").arg("+%Y-%m-%d").output().ok()
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .map(|text| text.trim().to_string())
        .filter(|text| !text.is_empty())
        .unwrap_or_else(|| "undated".to_string())
}
