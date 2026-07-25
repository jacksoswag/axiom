//! Reading what comes back off the wire and driving what goes in: one field out of an event line, one
//! burst of lines out of a running harness. Every line read here was written by this repo, which is
//! what lets a reader this small stand in for a JSON parser.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{channel, Receiver};
use std::time::{Duration, Instant};

/// How long any one line is worth waiting for. A search generation on a loaded machine is the slowest
/// thing here; a harness that has stopped answering costs this much once and then the case fails.
pub const PATIENCE: Duration = Duration::from_secs(20);
/// A burst is over once nothing has arrived for this long. Frames pace at 40 ms, so this clears one.
const LULL: Duration = Duration::from_millis(300);

/// One field out of an event line. It finds the key and takes the value whole, following quotes and
/// brackets so a message with a comma in it and an array of positions both come back in one piece.
pub fn field<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let mark = format!("\"{key}\":");
    let rest = &line[line.find(&mark)? + mark.len()..];
    if rest.starts_with('"') {
        let mut escaped = false;
        for (at, character) in rest.char_indices().skip(1) {
            if escaped { escaped = false; }
            else if character == '\\' { escaped = true; }
            else if character == '"' { return Some(&rest[..=at]); }
        }
        return Some(rest);
    }
    let mut depth = 0i32;
    for (at, character) in rest.char_indices() {
        match character {
            '[' | '{' => depth += 1,
            ']' | '}' if depth > 0 => depth -= 1,
            ']' | '}' | ',' if depth == 0 => return Some(&rest[..at]),
            _ => {}
        }
    }
    Some(rest)
}
pub fn text(line: &str, key: &str) -> String { field(line, key).unwrap_or("").trim_matches('"').to_string() }
pub fn number(line: &str, key: &str) -> f64 { field(line, key).unwrap_or("").parse().unwrap_or(f64::NAN) }
pub fn kind(line: &str) -> String { text(line, "type") }
/// An array field as numbers. A null slot reads as NaN, which is what the wire means by it.
pub fn list(line: &str, key: &str) -> Vec<f64> {
    field(line, key).unwrap_or("[]").trim_matches(|c| c == '[' || c == ']')
        .split(',').filter(|slot| !slot.is_empty()).map(|slot| slot.parse().unwrap_or(f64::NAN)).collect()
}
/// The layout's gene entries, one per gene, in the order the genome holds them
pub fn genes_of(layout: &str) -> Vec<String> {
    layout.split("{\"index\":").skip(1).map(|entry| format!("{{\"index\":{}", entry.trim_end_matches(['}', ']'])))
        .collect()
}

pub struct Harness { child: Child, input: ChildStdin, lines: Receiver<String>, fences: usize }
impl Harness {
    pub fn open() -> Harness {
        let mut child = Command::new(env!("CARGO_BIN_EXE_axiom"))
            .stdin(Stdio::piped()).stdout(Stdio::piped()).spawn().expect("the harness binary runs");
        let input = child.stdin.take().expect("stdin is piped");
        let output = child.stdout.take().expect("stdout is piped");
        let (sender, lines) = channel();
        std::thread::spawn(move || {
            for line in BufReader::new(output).lines() {
                let Ok(line) = line else { break };
                if sender.send(line).is_err() { break; }
            }
        });
        Harness { child, input, lines, fences: 0 }
    }
    pub fn tell(&mut self, line: &str) {
        let _ = writeln!(self.input, "{line}");
        let _ = self.input.flush();
    }
    pub fn next(&mut self, patience: Duration) -> Option<String> { self.lines.recv_timeout(patience).ok() }
    /// Everything said since the last look, then the greeting this one asks for. The fence is an
    /// unknown verb, and its complaint comes back in the order it was sent: that is what tells the two
    /// apart, since several commands report a state of their own and a greeting is not the only one.
    pub fn look(&mut self) -> Look {
        self.fences += 1;
        let fence = format!("fence{}", self.fences);
        self.tell(&fence);
        self.tell("catalog");
        let mut lines = Vec::new();
        while let Some(line) = self.next(PATIENCE) {
            if kind(&line) == "error" && line.contains(&fence) { break; } // everything before it is the answer
            lines.push(line);
        }
        while let Some(line) = self.next(PATIENCE) {
            let done = kind(&line) == "state";
            lines.push(line);
            if done { break; }
        }
        // A greeting owes a frame, which lands just after its state. Taking the tail leaves the channel
        // empty, so nothing left over can be mistaken for the answer to the next look.
        let deadline = Instant::now() + Duration::from_secs(2);
        while Instant::now() < deadline {
            match self.next(LULL) {
                Some(line) => lines.push(line),
                None => break,
            }
        }
        Look { lines }
    }
    /// Everything up to and including the line this answers true on, for a burst a fence cannot be
    /// sent through: a running search reports on its own schedule and a look would swallow it.
    pub fn until(&mut self, done: impl Fn(&str) -> bool) -> Look {
        let mut lines = Vec::new();
        while let Some(line) = self.next(PATIENCE) {
            let last = done(&line);
            lines.push(line);
            if last { break; }
        }
        Look { lines }
    }
    pub fn close(mut self) -> bool {
        self.tell("quit");
        for _ in 0..100 {
            match self.child.try_wait() {
                Ok(Some(_)) => return true,
                _ => std::thread::sleep(Duration::from_millis(50)),
            }
        }
        let _ = self.child.kill();
        false
    }
}

/// One burst of event lines, asked about by type rather than by position
pub struct Look { pub lines: Vec<String> }
impl Look {
    pub fn of(&self, wanted: &str) -> Vec<&str> {
        self.lines.iter().filter(|line| kind(line) == wanted).map(|line| line.as_str()).collect()
    }
    pub fn last(&self, wanted: &str) -> String { self.of(wanted).last().copied().unwrap_or("").to_string() }
    pub fn first(&self, wanted: &str) -> String { self.of(wanted).first().copied().unwrap_or("").to_string() }
    pub fn state(&self) -> String { self.last("state") }
    pub fn layout(&self) -> String { self.last("layout") }
    pub fn errors(&self) -> Vec<&str> { self.of("error") }
    pub fn complaint(&self) -> String {
        self.errors().first().map(|line| text(line, "message")).unwrap_or_default()
    }
}
