//! The frontend boundary in both directions: one command per line in, one JSON event per line out.
//! Commands are key=value so nothing here parses JSON, and events are JSON so a browser does not have
//! to parse anything either. The only place an outside caller's numbers are checked.

use std::fmt::Write as _; // for writing numbers into a String buffer; Emit below wants the io one
use std::io::Write;

/// One command line: a verb and its key=value arguments, values kept as text until a handler asks for
/// a number. One parser serves every verb, so a new command never touches this.
pub struct Command {
    pub verb: String,
    args: Vec<(String, String)>,
}
impl Command {
    /// None for a blank line, which is what a frontend sends when it sends nothing. A bare word
    /// becomes a valueless key, so `search stop` and `stop=` arrive the same way.
    pub fn parse(line: &str) -> Option<Command> {
        let mut words = line.split_whitespace();
        let verb = words.next()?.to_lowercase();
        let args = words.map(|word| match word.split_once('=') {
            Some((key, value)) => (key.to_lowercase(), value.to_owned()),
            None => (word.to_lowercase(), String::new()),
        }).collect();
        Some(Command { verb, args })
    }
    pub fn flag(&self, key: &str) -> bool { self.args.iter().any(|(name, _)| name == key) }
    pub fn text(&self, key: &str) -> Option<&str> {
        self.args.iter().find(|(name, _)| name == key).map(|(_, value)| value.as_str())
    }
    /// Absent reads as None so every caller keeps what it already had; present and unreadable is the
    /// caller's mistake and says so. Non-finite is rejected here, once, so nothing below sees a NaN.
    pub fn number(&self, key: &str) -> Result<Option<f32>, String> {
        match self.text(key) {
            None => Ok(None),
            Some(value) => match value.parse::<f32>() {
                Ok(number) if number.is_finite() => Ok(Some(number)),
                _ => Err(format!("{key} wants a finite number, got '{value}'")),
            },
        }
    }
    pub fn count(&self, key: &str) -> Result<Option<usize>, String> {
        Ok(self.number(key)?.map(|value| value.max(0.0) as usize))
    }
    /// Seeds get their own reader because a u64 seed loses its low bits through f32.
    pub fn seed(&self, key: &str) -> Result<Option<u64>, String> {
        match self.text(key) {
            None => Ok(None),
            Some(value) => value.parse::<u64>().map(Some).map_err(|_| format!("{key} wants a whole seed, got '{value}'")),
        }
    }
    /// A comma list, empty when the key is absent or holds nothing
    pub fn words(&self, key: &str) -> Vec<&str> {
        self.text(key).unwrap_or_default().split(',').filter(|word| !word.is_empty()).collect()
    }
    pub fn floats(&self, key: &str) -> Result<Vec<f32>, String> {
        self.words(key).into_iter().map(|word| match word.parse::<f32>() {
            Ok(number) if number.is_finite() => Ok(number),
            _ => Err(format!("{key} holds '{word}', which is not a finite number")),
        }).collect()
    }
    /// Every argument as it arrived, for a verb whose keys are names it cannot know ahead of time
    pub fn entries(&self) -> Vec<(&str, &str)> {
        self.args.iter().map(|(key, value)| (key.as_str(), value.as_str())).collect()
    }
    /// Every argument whose key is a number, for a frontend that names genes by position. Anything
    /// named stays out of the way, so `gene 12=0.4 quiet` still reads one edit.
    pub fn indexed(&self) -> Result<Vec<(usize, f32)>, String> {
        let mut edits = Vec::new();
        for (key, value) in &self.args {
            let Ok(index) = key.parse::<usize>() else { continue };
            match value.parse::<f32>() {
                Ok(number) if number.is_finite() => edits.push((index, number)),
                _ => return Err(format!("gene {index} wants a finite number, got '{value}'")),
            }
        }
        Ok(edits)
    }
}

/// Events leave one line at a time and flushed, because a frontend is blocked waiting on the next
/// one. Bodies are built by whoever owns the data; this wraps and ships them.
pub struct Emit { out: std::io::Stdout }
impl Emit {
    pub fn new() -> Emit { Emit { out: std::io::stdout() } }
    pub fn event(&mut self, kind: &str, body: &str) {
        let separator = if body.is_empty() { "" } else { "," };
        // A closed pipe means the frontend is gone, and there is nothing left for this process to do.
        if writeln!(self.out, "{{\"type\":\"{kind}\"{separator}{body}}}").is_err() { std::process::exit(0); }
        if self.out.flush().is_err() { std::process::exit(0); }
    }
    /// A run of events of one kind, written and flushed together. A harvest ships up to 512 specimens
    /// in one burst, and a write and a flush apiece is 512 trips down the pipe for a single list.
    pub fn events(&mut self, kind: &str, bodies: &[String]) {
        let mut batch = String::new();
        for body in bodies {
            let separator = if body.is_empty() { "" } else { "," };
            writeln!(batch, "{{\"type\":\"{kind}\"{separator}{body}}}").ok();
        }
        if self.out.write_all(batch.as_bytes()).is_err() { std::process::exit(0); }
        if self.out.flush().is_err() { std::process::exit(0); }
    }
    /// A whole line built elsewhere, which is how a worker thread reports without holding stdout
    pub fn line(&mut self, line: &str) {
        if writeln!(self.out, "{line}").is_err() { std::process::exit(0); }
        if self.out.flush().is_err() { std::process::exit(0); }
    }
    pub fn error(&mut self, message: &str) { self.event("error", &format!("\"message\":{}", quote(message))); }
}

/// A JSON array of numbers at the precision its reader needs: three decimals for positions a screen
/// will round anyway, six for a descriptor a frontend compares against a recorded one.
pub fn numbers(values: &[f32], decimals: usize) -> String {
    let mut out = String::with_capacity(values.len() * (decimals + 4) + 2);
    write_numbers(&mut out, values, decimals);
    out
}

/// The same array, straight into a buffer the caller keeps. A frame carries up to 32,768 of these
/// twenty-five times a second, and formatting each one through a String of its own was an allocation
/// per value on the same thread that steps the sim.
pub fn write_numbers(out: &mut String, values: &[f32], decimals: usize) {
    out.push('[');
    for (slot, value) in values.iter().enumerate() {
        if slot > 0 { out.push(','); }
        if value.is_finite() { write!(out, "{value:.decimals$}").ok(); } else { out.push_str("null"); }
    }
    out.push(']');
}

/// A JSON array of quoted strings, for the metric keys that give a descriptor its layout
pub fn quoted(items: &[&str]) -> String {
    let inside: Vec<String> = items.iter().map(|item| quote(item)).collect();
    format!("[{}]", inside.join(","))
}

/// One JSON string. Metric keys are plain ASCII, but an error message carries whatever a frontend
/// typed, so both escapes and the control range are handled here rather than trusted.
pub fn quote(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for character in text.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            control if control < ' ' || control == '\u{7f}' => out.push_str(&format!("\\u{:04x}", control as u32)),
            plain => out.push(plain),
        }
    }
    out.push('"');
    out
}
