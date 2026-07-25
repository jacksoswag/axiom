// The page holds no model of the world. Every control sends a command line, every readout came from an
// event the harness sent back, and what survives a reload is taste (panel state, render settings) rather
// than anything the simulation owns.

import { createWorld } from "./world.js";

const pick = (id) => document.getElementById(id);
const world = createWorld(pick("canvas"));
const state = {
  catalog: null, layout: null, live: null, search: null, run: null,
  genes: [], pair: 0, anchors: 0,
  watching: new Set(), axes: new Set(), gates: [],
  knobs: {}, shapeKnobs: {}, renderKnobs: {},
  specimens: [], chosen: -1,
  dragging: false, deferred: null, deferredShape: null,
  criterion: "novelty", timesteps: 400, follow: false, refine: false, cadence: 30,
};

// what a reload keeps

const KEY = "axiom.ui.1";
function keep() {
  const settings = {};
  for (const knob of RENDER_FIELDS) settings[knob[1]] = knob[5] ? world.camera[knob[1]] : world[knob[1]];
  localStorage.setItem(KEY, JSON.stringify({
    settings, material: world.material, beads: world.beads, contained: world.contained,
    watching: [...state.watching], axes: [...state.axes], gates: state.gates,
    criterion: state.criterion, timesteps: state.timesteps, cadence: state.cadence,
    follow: state.follow, refine: state.refine,
    searchSeed: state.searchSeed ? state.searchSeed.value : 1,
    knobs: Object.fromEntries(Object.entries(state.knobs).map(([name, control]) => [name, control.get()])),
    folds: [...document.querySelectorAll(".fold")].map((fold) => fold.open),
    drawer: pick("drawer").classList.contains("away"), rail: pick("rail").classList.contains("away"),
  }));
}
const memory = (() => {
  try { return JSON.parse(localStorage.getItem(KEY)) || {}; } catch { return {}; }
})();

// transport

let refused = false; // a dead harness would otherwise toast once per slider tick
function send(line) {
  fetch("/command", { method: "POST", body: line }).then((answer) => {
    if (answer.ok) { refused = false; return; }
    if (!refused) toast(answer.status === 503 ? "the harness is gone" : "that command was refused", true);
    refused = true;
  }).catch(() => { if (!refused) toast("the harness is not answering", true); refused = true; });
}
function listen() {
  const stream = new EventSource("/events");
  stream.onmessage = (message) => handle(JSON.parse(message.data));
  // The harness greeted whoever was listening when it started, which was nobody. Asking again is how a
  // page that arrives late, or reloads, gets the catalog, the layout, and the state it needs.
  stream.onopen = () => { pick("link").classList.add("live"); pick("link").lastChild.textContent = "live"; send("catalog"); };
  stream.onerror = () => { pick("link").classList.remove("live"); pick("link").lastChild.textContent = "reconnecting"; };
}
function toast(text, bad) {
  const note = document.createElement("div");
  note.className = bad ? "toast bad" : "toast";
  note.textContent = text;
  pick("toasts").appendChild(note);
  setTimeout(() => note.remove(), 4200);
}

let latest = null; // the newest frame that has arrived, waiting for the draw loop to want it
function handle(event) {
  switch (event.type) {
    case "catalog": state.catalog = event; build(); break;
    case "layout": applyLayout(event); break;
    case "state": showState(event); break;
    // A frame is a picture of now, and the renderer draws at most one picture a refresh, so the newest
    // one waits here and the draw loop takes it. Unpacking each on arrival paid for pictures nobody saw:
    // after any stall the backlog uploads in full, and a hidden tab keeps receiving frames long after
    // rAF stopped asking for any.
    case "frame": latest = event; break;
    case "sample": showSample(event); break;
    case "search": showSearch(event); break;
    case "generation": showGeneration(event); break;
    case "specimen": state.specimens.push(event); listSpecimens(); break;
    case "error": toast(event.message, true); break;
    // The relay speaks for itself exactly once, to say the process it was relaying stopped existing.
    case "harness":
      pick("link").classList.remove("live");
      pick("link").lastChild.textContent = "harness gone";
      toast("the harness stopped. restart the server to get one back", true);
      break;
  }
}

// control builders

function make(tag, className, inside) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (inside !== undefined) node.innerHTML = inside;
  return node;
}
function fold(host, title, extra) {
  const block = make("details", "fold");
  block.innerHTML = `<summary>${title}${extra ? `<span class="tail">${extra}</span>` : ""}</summary>`;
  const body = make("div", "body");
  block.appendChild(body);
  host.appendChild(block);
  const index = document.querySelectorAll(".fold").length - 1;
  block.open = memory.folds ? memory.folds[index] !== false : index < 2;
  block.ontoggle = keep;
  return body;
}
/// Label, slider, and a box the exact value can be typed into. A slider is for feeling out a range and
/// the box is for saying what you meant, so every setting here has both.
function ctl(host, spec) {
  const digits = spec.step >= 1 ? 0 : Math.min(4, String(spec.step).split(".")[1].length);
  const line = make("div", "ctl");
  line.appendChild(make("label", null, spec.label));
  const slide = make("input");
  slide.type = "range"; slide.min = spec.low; slide.max = spec.high; slide.step = spec.step;
  const box = make("input", "num");
  box.type = "number"; box.min = spec.low; box.max = spec.high; box.step = spec.step;
  const put = (value, tell) => {
    const held = Math.min(spec.high, Math.max(spec.low, Number(value)));
    slide.value = held;
    box.value = held.toFixed(digits);
    if (tell) spec.change(held);
  };
  slide.oninput = () => put(slide.value, true);
  slide.onpointerdown = () => { state.dragging = true; };
  box.onchange = () => put(box.value, true);
  box.onkeydown = (event) => { if (event.key === "Enter") box.blur(); };
  put(spec.value, false);
  line.append(slide, box);
  host.appendChild(line);
  return { set: (value) => put(value, false), get: () => parseFloat(slide.value), line };
}
function press(host, text, click, className) {
  const made = make("button", className, text);
  made.onclick = () => click(made);
  host.appendChild(made);
  return made;
}
function row(host, className) {
  const made = make("div", className ? `row ${className}` : "row");
  host.appendChild(made);
  return made;
}
function seg(host, options, value, change) {
  const group = make("div", "seg");
  for (const option of options) {
    const button = make("button", option === value ? "on" : null, option);
    button.onclick = () => {
      for (const other of group.children) other.classList.toggle("on", other === button);
      change(option);
    };
    group.appendChild(button);
  }
  host.appendChild(group);
  return group;
}
/// A toggle set that reports the whole set, since that is how the harness takes both a watch list and a
/// novelty axis list: name everything you want, and what you leave out is off.
function chips(host, items, active, change) {
  const strip = make("div", "chips");
  for (const item of items) {
    const button = make("button", active.has(item.key) ? "on" : null,
      `${item.key}${item.cost ? `<span class="cost">${item.cost}</span>` : ""}`);
    button.onclick = () => {
      if (active.has(item.key)) active.delete(item.key); else active.add(item.key);
      button.classList.toggle("on", active.has(item.key));
      change();
      keep();
    };
    strip.appendChild(button);
  }
  host.appendChild(strip);
  return strip;
}

// the sections

const STEP_OF = { particles: 1, anchors: 1, shells: 1, bumps: 1, radius: 0.05, dt: 0.001 };
const RENDER_FIELDS = [
  ["field side", "side", 24, 128, 4], // the renderer's own ceiling: past it the slider would move nothing
  ["blend reach", "blend", 0.5, 3, 0.05],
  ["threshold", "iso", 0.2, 8, 0.05],
  ["absorption", "absorption", 0.02, 6, 0.02],
  ["depth haze", "haze", 0, 6, 0.05],
  ["glow", "emission", 0, 0.5, 0.005],
  ["exposure", "exposure", 0.2, 3, 0.05],
  ["march steps", "steps", 24, 320, 8],
  ["march reach", "reach", 0.5, 6, 0.1],
  ["bead size", "grain", 0.05, 1.5, 0.05],
  ["render scale", "scale", 0.35, 1, 0.05],
  ["field of view", "focal", 0.4, 1.8, 0.02, true],
  ["fly speed", "speed", 0.1, 3, 0.05, true],
];

function build() {
  const host = pick("sections");
  host.textContent = "";
  buildWorld(fold(host, "world"));
  state.genomeHost = fold(host, "genome");
  buildSearch(fold(host, "search"));
  buildMeasure(fold(host, "measurement"));
  buildRender(fold(host, "render"));
  if (state.layout) buildGenome();
  applyMemory();
}

function buildWorld(host) {
  state.shapeKnobs = {};
  for (const field of state.catalog.shape) {
    state.shapeKnobs[field.name] = ctl(host, { label: field.name, low: field.low, high: field.high,
      step: STEP_OF[field.name] || 0.01, value: field.default,
      change: (value) => send(`shape ${field.name}=${value}`) });
  }
  const seeds = make("div", "pair-row");
  const seed = make("input", "num");
  seed.type = "number"; seed.min = 0; seed.value = 1;
  seed.onchange = () => send(`shape seed=${Math.max(0, parseInt(seed.value) || 0)}`);
  state.seedBox = seed;
  seeds.append(make("label", null, "seed"), seed);
  host.appendChild(seeds);
  press(seeds, "roll", () => {
    seed.value = Math.floor(Math.random() * 1e9);
    send(`shape seed=${seed.value}`);
  });
  host.appendChild(make("p", "note", "the box follows coordination and density"));
}

/// The genome as the genome is shaped: the world prefix, then one cell per ordered anchor pair. The grid
/// carries each pair's directed weight as color, so where the law is strong is visible before anything is
/// opened, and one click brings that pair's own genes up underneath.
function buildGenome() {
  const host = state.genomeHost;
  if (!host) return;
  host.textContent = "";
  state.geneKnobs = {};
  for (const gene of state.layout.genes.filter((gene) => gene.kind !== "pair")) {
    state.geneKnobs[gene.index] = ctl(host, { label: gene.label, low: gene.low, high: gene.high,
      value: gene.value, step: (gene.high - gene.low) / 400, change: (value) => editGene(gene.index, value) });
  }
  const pairs = state.layout.genes.filter((gene) => gene.kind === "pair");
  if (!pairs.length) return;
  state.anchors = 1 + Math.max(...pairs.map((gene) => gene.source));
  host.appendChild(make("div", "axis", "<span>source anchor →</span><span>receiver ↓</span>"));
  const grid = make("div", "matrix");
  grid.style.gridTemplateColumns = `repeat(${state.anchors}, 1fr)`;
  state.matrix = [];
  for (let destination = 0; destination < state.anchors; destination++) {
    for (let source = 0; source < state.anchors; source++) {
      const slot = source * state.anchors + destination;
      const cell = make("button", null, `${source}→${destination}`);
      cell.onclick = () => { state.pair = slot; buildPair(); paintMatrix(); keep(); };
      state.matrix[slot] = cell;
      grid.appendChild(cell);
    }
  }
  host.appendChild(grid);
  state.pair = Math.min(state.pair, state.anchors * state.anchors - 1);
  state.pairHost = make("div");
  host.appendChild(state.pairHost);
  const share = row(host, "tight");
  press(share, "copy genome", async () => {
    await navigator.clipboard.writeText(state.genes.join(","));
    toast(`${state.genes.length} genes copied`);
  });
  press(share, "paste", async () => {
    const text = await navigator.clipboard.readText();
    send(`genome genes=${text.trim().replace(/\s+/g, "")}`);
  });
  paintMatrix();
  buildPair();
}
function buildPair() {
  const host = state.pairHost;
  host.textContent = "";
  const stride = state.layout.genes.filter((gene) => gene.kind === "pair").length / (state.anchors * state.anchors);
  const start = 1 + state.anchors + state.pair * stride;
  const source = Math.floor(state.pair / state.anchors);
  host.appendChild(make("div", "axis",
    `<span>anchor ${source} → ${state.pair % state.anchors}</span><span>${stride} genes</span>`));
  for (const gene of state.layout.genes.slice(start, start + stride)) {
    state.geneKnobs[gene.index] = ctl(host, { label: gene.label, low: gene.low, high: gene.high,
      value: gene.value, step: (gene.high - gene.low) / 400, change: (value) => editGene(gene.index, value) });
  }
}
/// Weight decides a cell's color: teal pulls, red pushes, and how far from grey says how hard.
function paintMatrix() {
  if (!state.matrix) return;
  const stride = state.layout.genes.filter((gene) => gene.kind === "pair").length / (state.anchors * state.anchors);
  for (let slot = 0; slot < state.matrix.length; slot++) {
    const weight = state.genes[1 + state.anchors + slot * stride + stride - 1] || 0;
    const strength = Math.min(1, Math.abs(weight) / 40);
    const hue = weight >= 0 ? "89, 224, 200" : "232, 121, 95";
    state.matrix[slot].style.background = `rgba(${hue}, ${(0.06 + strength * 0.5).toFixed(3)})`;
    state.matrix[slot].classList.toggle("on", slot === state.pair);
    state.matrix[slot].title = `anchor ${Math.floor(slot / state.anchors)} to ${slot % state.anchors}, weight ${weight.toFixed(1)}`;
  }
}

// A drag fires dozens of edits a second and one pair edit costs a fresh density calibration, so edits
// pile up here and leave as a single line once a frame.
const pending = new Map();
let scheduled = 0;
function editGene(index, value) {
  state.genes[index] = value;
  pending.set(index, value);
  if (scheduled) return;
  // The cell colors go with the edits rather than with the input events: a drag fires dozens a second
  // and repainting A^2 cells for each of them is work nobody sees before the next one lands.
  scheduled = requestAnimationFrame(() => { scheduled = 0; paintMatrix(); flushGenes(false); });
}
function flushGenes(leaving) {
  const edits = [...pending].map(([slot, held]) => `${slot}=${held}`);
  pending.clear();
  if (!edits.length) return;
  const line = `gene ${edits.join(" ")}`;
  // A page on its way out is not allowed to wait for a fetch, and a beacon is the one post it can leave
  // behind. Otherwise the last edits of a drag would die with the tab.
  if (leaving) navigator.sendBeacon("/command", line); else send(line);
}
window.addEventListener("pagehide", () => flushGenes(true));
document.addEventListener("visibilitychange", () => { if (document.visibilityState === "hidden") flushGenes(true); });

function buildSearch(host) {
  seg(host, state.catalog.criteria.map((entry) => entry.name), state.criterion, (name) => {
    state.criterion = name;
    axisStrip.style.display = name === "novelty" ? "flex" : "none";
    keep();
  });
  const axisStrip = chips(host, state.catalog.metrics.map((metric) => ({ key: metric.key })), state.axes, () => {});
  axisStrip.style.display = state.criterion === "novelty" ? "flex" : "none";
  state.knobs = {};
  for (const knob of state.catalog.algorithms[0].knobs) {
    state.knobs[knob.name] = ctl(host, { label: knob.name, low: knob.low, high: knob.high,
      step: knob.high <= 1 ? 0.01 : 1, value: knob.default, change: keep });
  }
  state.knobs.timesteps = ctl(host, { label: "timesteps", low: 40, high: 4000, step: 20,
    value: state.timesteps, change: (value) => { state.timesteps = value; keep(); } });
  // The search's own seed, apart from the world's: the same config explores differently from a different
  // draw, and comparing two searches means holding one of the two seeds still.
  const seeds = make("div", "pair-row");
  const seed = make("input", "num");
  seed.type = "number"; seed.min = 0; seed.value = memory.searchSeed || 1;
  seed.onchange = keep;
  state.searchSeed = seed;
  seeds.append(make("label", null, "search seed"), seed);
  host.appendChild(seeds);
  press(seeds, "roll", () => { seed.value = Math.floor(Math.random() * 1e9); keep(); });
  const options = row(host, "tight");
  state.refineButton = press(options, "refine this world", (made) => {
    state.refine = !state.refine;
    made.classList.toggle("on", state.refine);
    keep();
  }, state.refine ? "on" : null);
  state.followButton = press(options, "follow the best", (made) => {
    state.follow = !state.follow;
    made.classList.toggle("on", state.follow);
    keep();
  }, state.follow ? "on" : null);
  host.appendChild(make("p", "note",
    "a search runs the world above, with the gates below. refine starts it from the genome on screen;"
    + " follow replays each new best into the view as it lands."));
}

function buildMeasure(host) {
  chips(host, state.catalog.metrics.map((metric) => ({
    key: metric.key,
    cost: metric.reduce === "once" ? "once" : metric.pairs ? "pairs" : metric.graph ? "graph" : "",
  })), state.watching, pushWatch);
  ctl(host, { label: "read every", low: 5, high: 400, step: 5, value: state.cadence,
    change: (value) => { state.cadence = value; pushWatch(); keep(); } });
  const acts = row(host, "tight");
  press(acts, "read once, repair included", () => send("measure now"));
  state.sampleHost = make("div");
  host.appendChild(state.sampleHost);

  host.appendChild(make("div", "axis", "<span>gates</span><span>floor and ceiling</span>"));
  const chooser = row(host, "tight");
  const choose = make("select", "grow");
  for (const metric of state.catalog.metrics) choose.appendChild(make("option", null, metric.key));
  chooser.appendChild(choose);
  press(chooser, "add gate", () => {
    if (state.gates.some((gate) => gate.metric === choose.value)) return;
    state.gates.push({ metric: choose.value, floor: 0.05, ceiling: 1 });
    drawGates();
    pushGates();
  });
  state.gateHost = make("div");
  host.appendChild(state.gateHost);
  drawGates();
}
function drawGates() {
  const host = state.gateHost;
  host.textContent = "";
  for (const gate of state.gates) {
    const line = make("div", "gate");
    line.appendChild(make("span", "name", gate.metric));
    for (const end of ["floor", "ceiling"]) {
      const box = make("input");
      box.type = "number"; box.step = 0.01; box.value = gate[end];
      box.onchange = () => { gate[end] = parseFloat(box.value) || 0; pushGates(); };
      line.appendChild(box);
    }
    const verdict = make("span", "verdict", "");
    gate.readout = verdict;
    line.appendChild(verdict);
    press(line, "×", () => {
      state.gates = state.gates.filter((other) => other !== gate);
      drawGates();
      pushGates();
    }, "tiny");
    host.appendChild(line);
  }
}
const pushWatch = () => send(`measure keys=${[...state.watching].join(",")} every=${state.cadence}`);
const pushGates = () => {
  send(`gates ${state.gates.map((gate) => `${gate.metric}=${gate.floor}:${gate.ceiling}`).join(" ")}`);
  keep();
};

function buildRender(host) {
  const modes = row(host, "tight");
  press(modes, "material", (made) => {
    world.material = !world.material;
    made.classList.toggle("on", world.material);
    world.touch();
    keep();
  }, world.material ? "on" : null);
  press(modes, "beads", (made) => {
    world.beads = !world.beads;
    made.classList.toggle("on", world.beads);
    world.mark();
    keep();
  }, world.beads ? "on" : null);
  // Contained shows the world as one object with its faces as the edge of the picture. Endless carries the
  // ray on through the copies a torus implies, which is what flying inside a world looks like.
  press(modes, "contained", (made) => {
    world.contained = !world.contained;
    made.classList.toggle("on", world.contained);
    world.mark();
    keep();
  }, world.contained ? "on" : null);
  press(modes, "reframe", () => world.frameWorld());
  state.renderKnobs = {};
  for (const [label, field, low, high, step, camera] of RENDER_FIELDS) {
    state.renderKnobs[field] = ctl(host, { label, low, high, step,
      value: camera ? world.camera[field] : world[field],
      change: (value) => {
        if (camera) world.camera[field] = value; else world[field] = value;
        if (field === "side" || field === "blend") world.touch(); else world.mark();
        keep();
      } });
  }
  host.appendChild(make("p", "note", "each particle is deposited as a sphere of blend reach x mean spacing."
    + " under about 0.8 every particle stands alone as its own grain; over 1.5 neighbors merge into one"
    + " continuous material. beads draw the particles themselves on top."));
  if (world.message) toast(world.message, true);
}

/// Taste comes back on reload; nothing the simulation owns does. The harness is asked for the rest.
function applyMemory() {
  for (const [name, control] of Object.entries(state.knobs)) {
    if (memory.knobs && memory.knobs[name] !== undefined) control.set(memory.knobs[name]);
  }
  pushWatch();
  if (state.gates.length) pushGates();
}

// readouts

function showState(event) {
  state.live = event;
  pick("clock").textContent = `tick ${event.tick.toLocaleString()}`;
  pick("census").textContent = `${event.shape.particles.toLocaleString()} particles`;
  // A law that outreaches its own grid, or one too inert to reach at all, steps every pair instead of
  // the neighbors. Same picture, many times the cost, so it says so rather than reading as a slow machine.
  pick("extent").textContent = `box ${event.box_len.toFixed(1)}${event.gridded === false ? " · all pairs" : ""}`;
  pick("play").textContent = event.running ? "pause" : "run";
  pick("play").classList.toggle("on", event.running);
  pick("speed").value = event.speed;
  pick("speed-read").textContent = event.speed;
  pick("start").disabled = event.searching;
  pick("stop").disabled = !event.searching;
  // Moving the knobs under a held pointer would fight the drag, so a shape that lands then waits for
  // release rather than being thrown away: a search finishing mid-drag still has a shape to hand over.
  if (state.dragging) state.deferredShape = event.shape; else syncShape(event.shape);
}
/// A loaded survivor brings a shape no slider was dragged to, so the knobs follow the world.
function syncShape(shape) {
  for (const [name, control] of Object.entries(state.shapeKnobs)) control.set(shape[name]);
  if (state.seedBox) state.seedBox.value = shape.seed;
}

/// A layout arrives on every reseed, which is also mid-drag on a coordination slider. Rebuilding the
/// controls under a held pointer would drop the drag, so a layout that lands then waits for release.
function applyLayout(event) {
  if (state.dragging) { state.deferred = event; return; }
  const same = state.layout && state.layout.genes.length === event.genes.length;
  state.layout = event;
  state.genes = event.genes.map((gene) => gene.value);
  if (same && state.geneKnobs) {
    for (const gene of event.genes) {
      const control = state.geneKnobs[gene.index];
      if (control) control.set(gene.value);
    }
    paintMatrix();
  } else if (state.catalog) {
    buildGenome();
  }
}

function showSample(event) {
  const host = state.sampleHost;
  if (!host) return;
  host.textContent = "";
  let start = 0;
  event.keys.forEach((key, slot) => {
    const width = event.widths[slot];
    const values = event.values.slice(start, start + width);
    start += width;
    const reading = make("div", "reading");
    reading.innerHTML = `<span>${key}<b>${width === 1 ? values[0].toFixed(4) : `${width} slots`}</b></span>`;
    if (width > 1) {
      const bars = make("div", "bars");
      for (const value of values) {
        const bar = make("i");
        bar.style.height = `${Math.max(1, Math.min(1, value) * 100)}%`;
        bars.appendChild(bar);
      }
      reading.appendChild(bars);
    }
    host.appendChild(reading);
  });
  for (const gate of event.gates) {
    const shown = state.gates.find((other) => other.metric === gate.metric);
    if (!shown || !shown.readout) continue;
    shown.readout.textContent = gate.pass ? "pass" : "fail";
    shown.readout.className = `verdict ${gate.pass ? "pass" : "fail"}`;
    shown.readout.title = `${gate.value.toFixed(4)} against ${gate.floor} to ${gate.ceiling}`;
  }
}

// the search

function startSearch() {
  const parts = [`search start criterion=${state.criterion}`];
  if (state.criterion === "novelty") {
    if (!state.axes.size) return toast("novelty needs axes to spread genomes across", true);
    parts.push(`axes=${[...state.axes].join(",")}`);
  }
  for (const [name, control] of Object.entries(state.knobs)) parts.push(`${name}=${control.get()}`);
  if (state.searchSeed) parts.push(`seed=${Math.max(0, parseInt(state.searchSeed.value) || 0)}`);
  if (state.refine) parts.push("from=world");
  state.specimens = [];
  state.chosen = -1;
  pick("log").querySelector("tbody").textContent = "";
  showSpecimens();
  send(parts.join(" "));
}

function showSearch(event) {
  const tag = pick("search-state");
  if (event.state === "started") {
    state.search = event;
    state.viewing = "live";
    tag.textContent = `${event.criterion} on ${event.shape.particles} particles`;
    pick("plan").innerHTML = `reading <b>${event.plan.join(", ")}</b> over <b>${event.timesteps}</b> timesteps`
      + `${event.refining ? `, refining ${event.refining} from the view` : ""}`;
    pick("tally").textContent = "";
  }
  if (event.state === "stopping") tag.textContent = "stopping after this generation";
  if (event.state === "done") {
    tag.textContent = `done, ${event.kept} kept`;
    track(1, `done, ${event.kept} survivors`);
    loadRuns(); // the run just written belongs in the list beside the older ones
  }
  if (event.state === "failed") { tag.textContent = "failed"; toast(event.reason, true); }
}

function showGeneration(event) {
  const body = pick("log").querySelector("tbody");
  const line = make("tr");
  line.innerHTML = `<td>${event.index}</td><td>${event.kept}</td><td class="best">${event.best.toFixed(5)}</td>`
    + `<td>${event.median.toFixed(5)}</td><td>${(event.elapsed_ms / 1000).toFixed(0)}s</td>`;
  body.prepend(line);
  while (body.children.length > 12) body.lastChild.remove();
  if (event.evaluated !== undefined) {
    pick("tally").innerHTML = `judged <b>${event.evaluated}</b> · died <b>${event.died}</b>`
      + ` · gated <b>${event.gated}</b> · unscorable <b>${event.unscorable}</b>`;
  }
  const total = (state.knobs.generations ? state.knobs.generations.get() : 20) + 1;
  track((event.index + 1) / total, `generation ${event.index + 1} of ${total} · best ${event.best.toFixed(5)}`);
  // Following means the view shows what the search just found, replayed to the tick it was measured at.
  if (state.follow && event.genes.length) {
    send(`genome genes=${event.genes.join(",")}`);
    send(`step count=${state.search ? state.search.timesteps : 400}`);
  }
}
function track(fraction, text) {
  pick("track").querySelector("i").style.width = `${Math.min(100, fraction * 100)}%`;
  pick("track-text").textContent = text;
}

/// Which search the survivors on screen belong to. A run read off disk and a search running now both
/// produce specimens, and each carries its own shape, so mixing them would load a genome into the wrong world.
function source() { return state.viewing === "disk" ? state.run : state.search; }

// A harvest lands as one event per survivor, and each one used to rebuild every card on screen: a run
// that kept a hundred paid for a hundred full rebuilds of a list that settles once. Nobody watching can
// see more than one of those a refresh.
let listing = 0;
function listSpecimens() {
  if (listing) return;
  listing = requestAnimationFrame(() => { listing = 0; showSpecimens(); });
}
function showSpecimens() {
  const host = pick("specimens");
  host.textContent = "";
  state.specimens.slice(0, 120).forEach((specimen, slot) => {
    const card = make("div", slot === state.chosen ? "card on" : "card");
    card.innerHTML = `<div class="top"><span>#${specimen.rank}</span><span class="score">${specimen.score.toFixed(5)}</span></div>`;
    const strip = make("div", "strip");
    for (const value of specimen.descriptor) {
      const bar = make("i");
      bar.style.height = `${Math.max(1, Math.min(1, value) * 100)}%`;
      strip.appendChild(bar);
    }
    card.appendChild(strip);
    const acts = make("div", "acts");
    press(acts, "load", () => load(specimen, slot), "tiny");
    press(acts, "replay", () => {
      load(specimen, slot);
      send(`step count=${(source() || {}).timesteps || 400}`);
    }, "tiny");
    card.appendChild(acts);
    card.onclick = (event) => { if (event.target === card || event.target.className === "strip") load(specimen, slot); };
    host.appendChild(card);
  });
  if (!state.specimens.length) host.appendChild(make("p", "note", "nothing yet. run a search, or open a run."));
  pick("survivor-count").textContent = state.specimens.length ? `${state.specimens.length} kept` : "";
}

/// Loading a survivor is a shape, a genome, and however long you want it stepped. Nothing was stored of
/// its state: the same shape and the same genes replay the same world exactly. The shape comes from
/// whichever search these survivors are from, since a genome only means anything inside its own shape.
function load(specimen, slot) {
  state.chosen = slot;
  showSpecimens();
  const shape = (source() || {}).shape;
  if (shape) {
    send(`shape particles=${shape.particles} anchors=${shape.anchors} shells=${shape.shells}`
      + ` bumps=${shape.bumps} radius=${shape.radius} dt=${shape.dt} seed=${shape.seed}`);
  }
  send(`genome genes=${specimen.genes.join(",")}`);
}

async function loadRuns() {
  const chooser = pick("run-pick");
  const chosen = chooser.value;
  const runs = await fetch("/runs").then((answer) => answer.json()).catch(() => []);
  chooser.textContent = "";
  chooser.appendChild(make("option", null, "this session"));
  for (const run of runs) {
    const option = make("option", null, `${run.name.replace(".jsonl", "")} (${run.specimens})`);
    option.value = run.name;
    chooser.appendChild(option);
  }
  if (chosen) chooser.value = chosen; // a refreshed list should not change what someone was reading
  chooser.onchange = async () => {
    if (!chooser.value || chooser.value === "this session") { state.viewing = "live"; return; }
    const lines = await fetch(`/runs/${chooser.value}`).then((answer) => answer.text());
    state.specimens = [];
    state.chosen = -1;
    for (const line of lines.split("\n").filter(Boolean)) {
      const event = JSON.parse(line);
      if (event.type === "search" && event.state === "started") state.run = event;
      if (event.type === "specimen") state.specimens.push(event);
    }
    state.viewing = "disk";
    pick("search-state").textContent = `${state.run ? state.run.criterion : "run"} from disk`;
    if (state.run) {
      pick("plan").innerHTML = `read <b>${state.run.plan.join(", ")}</b> over <b>${state.run.timesteps}</b>`
        + ` timesteps on <b>${state.run.shape.particles}</b> particles`;
      pick("tally").textContent = "";
    }
    showSpecimens();
  };
}

// the dock and the canvas

pick("play").onclick = () => send(state.live && state.live.running ? "pause" : "run");
pick("step").onclick = () => send("step count=200");
pick("reseed").onclick = () => send("reseed");
pick("speed").oninput = () => {
  pick("speed-read").textContent = pick("speed").value;
  send(`speed steps=${pick("speed").value}`);
};
pick("start").onclick = startSearch;
pick("stop").onclick = () => send("search stop");
/// What the panels cover is not stage, so the renderer slides the picture into the band that is left.
function measurePanels() {
  const drawer = pick("drawer");
  const rail = pick("rail");
  world.margin.left = drawer.classList.contains("away") ? 0 : drawer.getBoundingClientRect().right;
  world.margin.right = rail.classList.contains("away") ? 0
    : window.innerWidth - rail.getBoundingClientRect().left;
}
for (const [handle, panel] of [["drawer-handle", "drawer"], ["rail-handle", "rail"]]) {
  const shift = () => {
    const away = pick(panel).classList.toggle("away");
    pick(handle).classList.toggle("shown", away);
    setTimeout(measurePanels, 240); // after the panel has finished sliding
    keep();
  };
  pick(handle).onclick = shift;
  pick(panel).ondblclick = (event) => { if (event.target === pick(panel)) shift(); };
}
window.addEventListener("resize", measurePanels);
measurePanels();

const canvas = pick("canvas");
let dragging = false;
canvas.onpointerdown = (event) => { dragging = true; canvas.setPointerCapture(event.pointerId); };
canvas.onpointerup = () => { dragging = false; };
canvas.onpointermove = (event) => {
  if (dragging || world.camera.flying) world.orbit(event.movementX, event.movementY);
};
canvas.onwheel = (event) => { world.dolly(-event.deltaY); event.preventDefault(); };
canvas.ondblclick = () => canvas.requestPointerLock();
document.addEventListener("pointerlockchange", () => {
  world.camera.flying = document.pointerLockElement === canvas;
  pick("hint").textContent = world.camera.flying
    ? "wasd to move, q and e down and up, shift to hurry, esc to release"
    : "double click the world to fly";
});
window.addEventListener("pointerup", () => {
  state.dragging = false;
  if (state.deferred) { const held = state.deferred; state.deferred = null; applyLayout(held); }
  if (state.deferredShape) { const held = state.deferredShape; state.deferredShape = null; syncShape(held); }
});

const held = new Set();
window.addEventListener("keydown", (event) => {
  if (event.target.tagName === "INPUT" || event.target.tagName === "SELECT") return;
  held.add(event.key.toLowerCase());
  if (event.key === " ") { send(state.live && state.live.running ? "pause" : "run"); event.preventDefault(); }
  if (event.key === "r") send("reseed");
  if (event.key === "f") world.frameWorld();
  if (event.key === "[") pick("drawer-handle").onclick();
  if (event.key === "]") pick("rail-handle").onclick();
});
window.addEventListener("keyup", (event) => held.delete(event.key.toLowerCase()));

let last = performance.now();
let counted = 0;
let sampled = last;
function draw(now) {
  const seconds = Math.min(0.1, (now - last) / 1000);
  last = now;
  if (world.camera.flying) {
    const axis = (positive, negative) => (held.has(positive) ? 1 : 0) - (held.has(negative) ? 1 : 0);
    const move = [axis("d", "a"), axis("e", "q"), axis("w", "s")];
    if (move[0] || move[1] || move[2]) world.fly(move, seconds * (held.has("shift") ? 4 : 1));
  }
  // The newest frame, once, here rather than on the socket. A hidden tab gets no refreshes, so it also
  // does no unpacking and no uploading; whatever piled up while it was away collapses to one picture.
  if (latest && world.ready && document.visibilityState !== "hidden") {
    const event = latest;
    latest = null;
    world.setFrame(event.positions, event.traits, event.box_len);
    pick("clock").textContent = `tick ${event.tick.toLocaleString()}`;
    // Past a few thousand particles a frame is thinned, and a viewer judging structure deserves to know
    // the picture is drawn from a sample of the swarm rather than all of it.
    if (event.stride > 1 && event.particles) {
      pick("census").textContent = `${event.particles.toLocaleString()} particles, ${event.traits.length.toLocaleString()} drawn`;
    }
  }
  if (world.ready && world.render()) counted++;
  if (now - sampled > 900) {
    const drawn = Math.round(counted / ((now - sampled) / 1000));
    pick("rate").textContent = drawn ? `${drawn} fps` : "still";
    counted = 0;
    sampled = now;
  }
  requestAnimationFrame(draw);
}

// what the last session left behind, before the first event arrives

if (memory.settings) {
  for (const knob of RENDER_FIELDS) {
    const value = memory.settings[knob[1]];
    if (value === undefined) continue;
    if (knob[5]) world.camera[knob[1]] = value; else world[knob[1]] = value;
  }
}
if (memory.material !== undefined) world.material = memory.material;
if (memory.beads !== undefined) world.beads = memory.beads;
if (memory.contained !== undefined) world.contained = memory.contained;
for (const key of memory.watching || ["heterogeneity", "connectivity", "turnover"]) state.watching.add(key);
for (const key of memory.axes || ["rdf", "heterogeneity", "connectivity", "mobility", "turnover"]) state.axes.add(key);
state.gates = (memory.gates || []).map((gate) => ({ metric: gate.metric, floor: gate.floor, ceiling: gate.ceiling }));
state.criterion = memory.criterion || "novelty";
state.cadence = memory.cadence || 30;
state.follow = !!memory.follow;
state.refine = !!memory.refine;
if (memory.drawer) { pick("drawer").classList.add("away"); pick("drawer-handle").classList.add("shown"); }
if (memory.rail) { pick("rail").classList.add("away"); pick("rail-handle").classList.add("shown"); }

requestAnimationFrame(draw);
loadRuns();
listen();

// The console is part of the instrument: anything a control can do, `axiom.send("...")` can do too, and
// the renderer's own numbers are readable while it runs.
window.axiom = { world, state, send, keep };
