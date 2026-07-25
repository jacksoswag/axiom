//! What the relay promises: the one trust boundary Python owns, the only thing in the stack that
//! touches disk, and the only place a browser's own connection can outlive the harness behind it.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use crate::Report;

pub fn relay(report: &mut Report) {
    let port = match TcpListener::bind("127.0.0.1:0").and_then(|held| held.local_addr()) {
        Ok(address) => address.port(),
        Err(problem) => return report.check("a port to serve on", false, problem.to_string()),
    };
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let server = Command::new("python3").arg(root.join("ui/server.py")).arg(port.to_string())
        .env("AXIOM_BIN", env!("CARGO_BIN_EXE_axiom")).stdout(Stdio::null()).stderr(Stdio::null()).spawn();
    let Ok(mut server) = server else {
        return report.check("python3 is here to run the relay", false, "no python3, the relay went unchecked".to_string());
    };
    let mut up = false;
    for _ in 0..100 {
        if get(port, "/index.html").is_some() { up = true; break; }
        std::thread::sleep(Duration::from_millis(100));
    }
    report.check("the relay serves the page it is there to serve",
        up && get(port, "/index.html").map(|(code, body)| code == 200 && body.contains("canvas")).unwrap_or(false),
        format!("port {port}"));
    if !up {
        let _ = server.kill();
        return;
    }
    // A stream first, so the greeting the harness sends on startup is still ahead of us.
    let stream = listen(port);

    let missing = get(port, "/nothing.js").map(|(code, _)| code).unwrap_or(0);
    let traversal = get(port, "/runs/..").map(|(code, _)| code).unwrap_or(0);
    let escape = get(port, "/runs/../server.py").map(|(code, _)| code).unwrap_or(0);
    report.check("an unknown path is a 404", missing == 404, format!("{missing}"));
    report.check("a run name that is a directory is a 404, not a traceback", traversal == 404,
        format!("/runs/.. answered {traversal}"));
    report.check("a run name cannot climb out of the runs directory", escape == 404, format!("{escape}"));

    let ok = post(port, "/command", "catalog").map(|(code, _)| code).unwrap_or(0);
    let split = post(port, "/command", "run\nsearch start criterion=structure").map(|(code, _)| code).unwrap_or(0);
    report.check("a command posts through to the harness", ok == 200, format!("{ok}"));
    report.check("a body carrying a second line is refused rather than framed into two commands",
        split == 400, format!("{split}"));

    let greeted = stream.map(|mut open| sip(&mut open, "\"type\":\"catalog\"", Duration::from_secs(5)));
    report.check("a listener gets the harness's own lines",
        greeted.unwrap_or(false), "catalog over the event stream".to_string());

    // The relay outliving the harness is the case a browser cannot see for itself.
    let watch = listen(port);
    let _ = post(port, "/command", "quit");
    let told = watch.map(|mut open| sip(&mut open, "\"type\":\"harness\"", Duration::from_secs(8)));
    report.check("a browser is told when the harness dies", told.unwrap_or(false),
        "harness gone over the event stream".to_string());
    let after = post(port, "/command", "run").map(|(code, _)| code).unwrap_or(0);
    report.check("a command sent to a dead harness fails loudly", after == 503, format!("{after}"));
    let _ = server.kill();
    let _ = server.wait();

    // The one case a socket cannot provoke, since the kernel buffers on our behalf: what a listener
    // that stopped reading gets when it starts again.
    let lagging = Command::new("python3").arg("-c").arg(LAGGING).arg(root.join("ui/server.py")).output();
    let said = lagging.map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string()).unwrap_or_default();
    report.check("a browser that fell behind catches up rather than replaying the past",
        said.starts_with("the specimen"), said);
}

/// Drives the relay's own queue where it lives: five hundred frames and one specimen offered to a
/// listener that is not reading. A frame is worth dropping because another is already on its way; a
/// specimen is the only copy of itself and has to survive the pile.
const LAGGING: &str = r#"
import importlib.util, sys
spec = importlib.util.spec_from_file_location("relay", sys.argv[1])
relay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(relay)
listener = relay.Listener()
for tick in range(500):
    listener.offer('{"type":"frame","tick":%d}' % tick, True)
listener.offer('{"type":"specimen"}', False)
first, second, third = listener.take(0.1), listener.take(0.1), listener.take(0.1)
crowded = relay.Listener()
for index in range(relay.BACKLOG + 50):
    crowded.offer('{"type":"generation","index":%d}' % index, False)
crowded.offer('{"type":"frame"}', True)
kept = 0
while crowded.take(0.01) is not None:
    kept += 1
wrong = []
if '"specimen"' not in (first or ''): wrong.append('news did not come first: %s' % first)
if '"tick":499' not in (second or ''): wrong.append('a stale frame survived: %s' % second)
if third is not None: wrong.append('something waited behind the newest frame')
if kept != relay.BACKLOG + 1: wrong.append('a crowded listener held %d lines' % kept)
print('; '.join(wrong) or 'the specimen, then frame 499, then nothing, with %d news kept under pressure' % (kept - 1))
"#;

/// One request, one answer, over its own connection. Enough HTTP to ask the relay a question: the
/// status line is what most of these cases turn on, and the body follows the blank line.
fn ask(port: u16, request: &str) -> Option<(u16, String)> {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).ok()?;
    stream.set_read_timeout(Some(Duration::from_secs(8))).ok()?;
    stream.write_all(request.as_bytes()).ok()?;
    let mut answer = Vec::new();
    stream.read_to_end(&mut answer).ok()?;
    let answer = String::from_utf8_lossy(&answer).to_string();
    let code = answer.split_whitespace().nth(1)?.parse().ok()?;
    Some((code, answer.split_once("\r\n\r\n").map(|(_, body)| body.to_string()).unwrap_or_default()))
}
fn get(port: u16, path: &str) -> Option<(u16, String)> {
    ask(port, &format!("GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"))
}
fn post(port: u16, path: &str, body: &str) -> Option<(u16, String)> {
    ask(port, &format!("POST {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()))
}
/// An event stream left open, since what it is for is what arrives later
fn listen(port: u16) -> Option<TcpStream> {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).ok()?;
    stream.set_read_timeout(Some(Duration::from_millis(400))).ok()?;
    stream.write_all(b"GET /events HTTP/1.1\r\nHost: localhost\r\n\r\n").ok()?;
    Some(stream)
}
/// Read an open stream until the text shows up or the patience runs out
fn sip(stream: &mut TcpStream, wanted: &str, patience: Duration) -> bool {
    let opened = Instant::now();
    let mut seen = String::new();
    let mut buffer = [0u8; 8192];
    while opened.elapsed() < patience {
        match stream.read(&mut buffer) {
            Ok(0) => return seen.contains(wanted),
            Ok(read) => seen.push_str(&String::from_utf8_lossy(&buffer[..read])),
            Err(_) => {} // a read timeout, which is only this listener having nothing to say yet
        }
        if seen.contains(wanted) { return true; }
    }
    false
}
