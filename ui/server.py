#!/usr/bin/env python3
"""Serves the page and relays it to the harness. The browser posts one command line, the harness answers
with JSON event lines, and every listener gets a copy. This is also the only thing in the stack that
touches disk: a run is exactly the event lines a search produced, so recording one is appending them."""

import json
import os
import subprocess
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNS = HERE / "runs"
PAGE = {"/": "index.html", "/index.html": "index.html", "/app.js": "app.js",
        "/world.js": "world.js", "/style.css": "style.css"}
TYPES = {".html": "text/html", ".js": "text/javascript", ".css": "text/css"}
# A stalled browser must not back up the harness. News waits its turn up to this many lines; a frame
# never waits, because the one behind it is already a better answer to the same question.
BACKLOG = 400
# The one event Python speaks for itself. Everything else here came from the harness, so a browser has
# no other way to learn that the thing it is driving stopped existing.
GONE = '{"type":"harness","state":"gone"}'
# Survivor counts already read off disk, keyed by run name and held against how the file stood then.
COUNTED = {}


def binary():
    """The built harness, or cargo as a fallback so a fresh clone still runs."""
    if os.environ.get("AXIOM_BIN"):
        return [os.environ["AXIOM_BIN"]]
    for candidate in (".cache/target/release/axiom", "target/release/axiom",
                      ".cache/target/debug/axiom", "target/debug/axiom"):
        path = ROOT / candidate
        if path.exists():
            return [str(path)]
    return ["cargo", "run", "--quiet", "--release"]


class Listener:
    """One open event stream. A frame is a picture of now, so a late one replaces the picture already
    waiting instead of queueing behind it: a browser that fell behind catches up rather than replaying
    the past. Everything else is news, and news is kept."""

    def __init__(self):
        self.news = deque()
        self.frame = None
        self.ready = threading.Condition()

    def offer(self, line, frame):
        with self.ready:
            if frame:
                self.frame = line
            elif len(self.news) < BACKLOG:
                self.news.append(line)
            else:
                return  # this browser is so far behind that nothing more can be said to it
            self.ready.notify()

    def take(self, timeout):
        """The next line for this browser, or None once the wait runs out. News before pictures: a
        specimen is the only copy of itself and the picture is about to be replaced anyway."""
        with self.ready:
            if not self.news and self.frame is None:
                self.ready.wait(timeout)
            if self.news:
                return self.news.popleft()
            line, self.frame = self.frame, None
            return line


class Harness:
    """One child process, one reader thread, any number of listeners."""

    def __init__(self):
        command = binary()
        print(f"harness: {' '.join(command)}", file=sys.stderr)
        try:
            self.child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.PIPE,
                                          stdout=subprocess.PIPE, text=True, bufsize=1)
        except OSError as problem:
            sys.exit(f"cannot start the harness ({' '.join(command)}): {problem}")
        self.listeners = []
        self.lock = threading.Lock()
        self.recording = None
        self.alive = True
        threading.Thread(target=self.pump, daemon=True).start()

    def pump(self):
        """Every line the harness says, to everyone listening. This thread ending is the only signal
        that the harness is gone, so it has to end that way however it ends: a reader that died quietly
        would leave the browsers watching a live-looking page and the child blocked on a full pipe."""
        try:
            for line in self.child.stdout:
                line = line.rstrip("\n")
                if not line:
                    continue
                # The type is the first key of every line, and a frame is a fifth of a megabyte of
                # positions. Asking what kind of line this is by scanning all of it, twenty-five times a
                # second, costs more than everything else this thread does.
                frame = line.startswith('{"type":"frame"')
                if not frame:
                    self.record(line)
                self.broadcast(line, frame)
        except Exception as problem:
            print(f"harness reader stopped: {problem}", file=sys.stderr)
        self.alive = False
        self.close_recording()
        self.child.terminate()  # if the reader died first, nothing will ever drain the child again
        self.broadcast(GONE, False)

    def broadcast(self, line, frame):
        with self.lock:
            listeners = list(self.listeners)
        for listener in listeners:
            listener.offer(line, frame)

    def record(self, line):
        """A search writes itself to disk as it goes, so a run survives the process that made it."""
        if '"type":"search"' in line:
            event = json.loads(line)
            if event.get("state") == "started":
                self.close_recording()  # a search that died mid-run left this one open
                RUNS.mkdir(exist_ok=True)
                stamp = time.strftime("%Y%m%d-%H%M%S")
                self.recording = open(RUNS / f"{stamp}-{event.get('criterion', 'search')}.jsonl", "w")
                self.recording.write(line + "\n")
                self.recording.flush()
                return
            if event.get("state") in ("done", "failed") and self.recording:
                self.recording.write(line + "\n")
                self.close_recording()
                return
        if self.recording and ('"type":"specimen"' in line or '"type":"generation"' in line):
            self.recording.write(line + "\n")
            self.recording.flush()

    def close_recording(self):
        if self.recording:
            self.recording.close()
            self.recording = None

    def send(self, line):
        """False when there is nothing left to send to, so the browser hears about it on the same
        request rather than waiting for an answer that is never coming."""
        if not self.alive:
            return False
        try:
            self.child.stdin.write(line + "\n")
            self.child.stdin.flush()
            return True
        except (OSError, ValueError):
            self.alive = False
            self.broadcast(GONE, False)
            return False

    def subscribe(self):
        listener = Listener()
        with self.lock:
            self.listeners.append(listener)
        if not self.alive:
            listener.offer(GONE, False)  # a page that arrived after the death still deserves to know
        return listener

    def forget(self, listener):
        with self.lock:
            if listener in self.listeners:
                self.listeners.remove(listener)


class Relay(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):
        pass  # the harness is the interesting output, not one line per fetch

    def do_GET(self):
        if self.path in PAGE:
            return self.file(HERE / PAGE[self.path])
        if self.path == "/events":
            return self.events()
        if self.path == "/runs":
            return self.runs()
        if self.path.startswith("/runs/"):
            return self.run(self.path[len("/runs/"):])
        self.send_error(404)

    def do_POST(self):
        if self.path != "/command":
            return self.send_error(404)
        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            return self.send_error(400, "a length that is not a number")
        line = self.rfile.read(length).decode("utf-8", "replace").strip()
        # One post is one command. The harness frames on newlines and takes no defense of its own, so a
        # body carrying an interior one would arrive there as several commands nobody typed.
        if "\n" in line or "\r" in line:
            return self.send_error(400, "one command per post")
        if not HARNESS.send(line):
            return self.send_error(503, "the harness is gone")
        self.answer(b"ok", "text/plain")

    def file(self, path):
        try:
            body = path.read_bytes()
        except OSError:
            return self.send_error(404)
        self.answer(body, TYPES.get(path.suffix, "application/octet-stream"))

    def answer(self, body, kind):
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def events(self):
        listener = HARNESS.subscribe()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            while True:
                line = listener.take(15)
                if line is None:
                    self.wfile.write(b": still here\n\n")  # a comment keeps a quiet stream open
                else:
                    self.wfile.write(f"data: {line}\n\n".encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            HARNESS.forget(listener)

    def runs(self):
        """A run file never changes once its search ended, so a count taken off one keeps until the file
        does. Rereading every run in the directory on every listing is the same answer bought again."""
        RUNS.mkdir(exist_ok=True)
        listing = []
        for path in sorted(RUNS.glob("*.jsonl"), reverse=True):
            stat = path.stat()
            token = (stat.st_mtime_ns, stat.st_size)
            if COUNTED.get(path.name, (None,))[0] != token:
                COUNTED[path.name] = (token, path.read_text(errors="replace").count('"type":"specimen"'))
            listing.append({"name": path.name, "specimens": COUNTED[path.name][1]})
        self.answer(json.dumps(listing).encode(), "application/json")

    def run(self, name):
        path = RUNS / name
        if "/" in name or not path.is_file(): # a name like '..' exists and is not a run
            return self.send_error(404)
        self.answer(path.read_bytes(), "text/plain")


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8731
    HARNESS = Harness()
    server = ThreadingHTTPServer(("127.0.0.1", port), Relay)
    server.daemon_threads = True
    print(f"axiom on http://127.0.0.1:{port}", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        HARNESS.send("quit")
