//! Behavior end to end, driven the way a frontend drives it: one process, command lines in, JSON event
//! lines out, and the Python relay in front of it over a real socket. Every case here is a promise the
//! protocol makes to a page that cannot see inside. Each one records what it observed into
//! tests/reports/DATE_behave.md, so a run says what the harness did rather than only that it passed.
//!
//! cargo test --test behavior -- --nocapture

mod areas;
mod relay;
mod wire;

use std::panic::AssertUnwindSafe;
use std::path::PathBuf;
use std::process::Command;

/// One promise, whether it held, and what was actually seen, grouped by the area that asked.
pub struct Report { area: String, pub rows: Vec<(String, String, bool, String)> }
impl Report {
    fn new() -> Report { Report { area: String::new(), rows: Vec::new() } }
    fn area(&mut self, name: &str) { self.area = name.to_string(); }
    /// The observation is the point: a report of nothing but "ok" says nothing a green test run did
    /// not already say.
    pub fn check(&mut self, promise: &str, held: bool, seen: impl Into<String>) {
        self.rows.push((self.area.clone(), promise.to_string(), held, seen.into()));
    }
    fn failed(&self) -> usize { self.rows.iter().filter(|(_, _, held, _)| !held).count() }
    fn write(&self, path: &PathBuf) {
        let mut out = String::new();
        out.push_str(&format!("# Behavior, {}\n\n", today()));
        out.push_str(&format!("{} promises checked, {} broken. Driven end to end: the harness over its own\n\
            command protocol, the relay over a socket. Every number below was measured on this run.\n\n",
            self.rows.len(), self.failed()));
        let mut area = String::new();
        for (row_area, promise, held, seen) in &self.rows {
            if *row_area != area {
                area = row_area.clone();
                out.push_str(&format!("\n## {area}\n\n| held | promise | observed |\n|---|---|---|\n"));
            }
            let mark = if *held { "yes" } else { "NO" };
            out.push_str(&format!("| {mark} | {promise} | {} |\n", seen.replace('|', "\\|")));
        }
        std::fs::create_dir_all(path.parent().expect("a parent directory")).expect("the reports directory");
        std::fs::write(path, out).expect("the report is writable");
    }
}
/// Today, from the one clock a test can reach without taking a dependency for it
pub fn today() -> String {
    Command::new("date").arg("+%Y-%m-%d").output().ok()
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
        .filter(|stamp| stamp.len() == 10).unwrap_or_else(|| "undated".to_string())
}

/// Every area, in order, with a report of what each one saw. An area that panics costs its own row and
/// nothing else: the report is worth more than the first failure's backtrace.
#[test]
fn behavior() {
    let areas: [(&str, fn(&mut Report)); 9] = [
        ("the greeting", areas::greeting),
        ("the command boundary", areas::boundary),
        ("editing a world", areas::editing),
        ("running a world", areas::running),
        ("measuring a world", areas::measuring),
        ("searching", areas::searching),
        ("stopping a search", areas::stopping),
        ("cost regimes", areas::regimes),
        ("leaving", areas::leaving),
    ];
    let mut report = Report::new();
    for (name, run) in areas {
        report.area(name);
        if std::panic::catch_unwind(AssertUnwindSafe(|| run(&mut report))).is_err() {
            report.check("the area ran to its end", false, "it panicked".to_string());
        }
    }
    report.area("the relay");
    if std::panic::catch_unwind(AssertUnwindSafe(|| relay::relay(&mut report))).is_err() {
        report.check("the area ran to its end", false, "it panicked".to_string());
    }

    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/reports")
        .join(format!("{}_behave.md", today()));
    report.write(&path);
    let broken: Vec<&str> = report.rows.iter().filter(|(_, _, held, _)| !held)
        .map(|(_, promise, _, _)| promise.as_str()).collect();
    println!("{} promises checked, report at {}", report.rows.len(), path.display());
    assert!(broken.is_empty(), "{} broken: {broken:#?}", broken.len());
}
