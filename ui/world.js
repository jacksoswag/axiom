// The view: particles in, material out. Each frame of positions is deposited into a periodic density
// field on the GPU, and the picture is raymarched out of that field, so nothing on screen was invented
// by the renderer. Density is read in multiples of a uniform swarm, which is what lets one threshold
// mean the same thing at any particle count, box, or decimation stride.

const DEPOSIT_VERTEX = `#version 300 es
precision highp float;
uniform sampler2D particles; // one texel per particle: xyz position, w trait
uniform int count;
uniform float boxLen;
uniform float support;   // splat reach in world units
uniform float layerZ;    // world depth this layer samples
out vec2 local;          // offset from the splat center, in support units
out float depth;         // that same offset along z
out float trait;
void main() {
  int particle = gl_InstanceID >> 2;             // four periodic images per particle, three usually culled
  int image = gl_InstanceID & 3;
  if (particle >= count) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
  ivec2 texel = ivec2(particle % 1024, particle / 1024);
  vec4 read = texelFetch(particles, texel, 0);
  vec3 position = read.xyz;

  // A splat within reach of a face also lands on the far side. The image offsets are picked per axis
  // and a copy that would not reach this layer at all is thrown away before it costs a fragment.
  vec2 wrap = vec2(0.0);
  if (position.x < support) wrap.x = boxLen; else if (position.x > boxLen - support) wrap.x = -boxLen;
  if (position.y < support) wrap.y = boxLen; else if (position.y > boxLen - support) wrap.y = -boxLen;
  vec2 shift = vec2(float(image & 1) * wrap.x, float((image >> 1) & 1) * wrap.y);
  if ((image & 1) == 1 && wrap.x == 0.0) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
  if (((image >> 1) & 1) == 1 && wrap.y == 0.0) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }

  float gap = position.z - layerZ;
  gap -= boxLen * floor(gap / boxLen + 0.5);     // minimum image, so the field wraps along z too
  float slice = abs(gap) / support;
  if (slice >= 1.0) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
  float radius = support * sqrt(1.0 - slice * slice); // a sphere cut at this depth reaches this far across

  vec2 corner = vec2(float(gl_VertexID & 1) * 2.0 - 1.0, float((gl_VertexID >> 1) & 1) * 2.0 - 1.0);
  vec2 center = position.xy + shift;
  local = corner * radius / support;
  depth = gap / support;
  trait = read.w;
  gl_Position = vec4((center + corner * radius) / boxLen * 2.0 - 1.0, 0.0, 1.0);
}`;

const DEPOSIT_FRAGMENT = `#version 300 es
precision highp float;
in vec2 local;
in float depth;
in float trait;
uniform float amplitude; // scaled so one particle deposits one unit of mass whatever the support is
out vec4 deposit;
void main() {
  float squared = dot(local, local) + depth * depth;
  if (squared >= 1.0) discard;                   // outside the sphere, so the splat has a clean edge
  float shape = 1.0 - squared;
  float weight = amplitude * shape * shape * shape; // smooth to its own rim, so no facet appears there
  // Traits sit on a circle and wrap at the 0/1 seam, so the average of two neighbors is a vector one:
  // carry cos and sin and the seam takes care of itself.
  float angle = 6.28318530718 * trait;
  deposit = vec4(weight, weight * cos(angle), weight * sin(angle), 0.0);
}`;

const SCREEN_VERTEX = `#version 300 es
void main() {
  vec2 corner = vec2(float(gl_VertexID & 1) * 4.0 - 1.0, float((gl_VertexID >> 1) & 1) * 4.0 - 1.0);
  gl_Position = vec4(corner, 0.0, 1.0);          // one oversized triangle covers the viewport
}`;

const MARCH_FRAGMENT = `#version 300 es
precision highp float;
precision highp sampler3D;
uniform sampler3D volume;
uniform vec3 eye;
uniform vec3 right;
uniform vec3 up;
uniform vec3 ahead;
uniform vec2 viewport;
uniform vec2 shift; // how far the picture slides so panels covering the canvas do not push it off center
uniform float tanHalf;
uniform float boxLen;
uniform float uniform_density; // mean field value of an evenly spread swarm, the reference for every read
uniform float iso;
uniform float absorption;
uniform float haze;            // extinction per box travelled, whether there is material there or not
uniform float emission;
uniform float exposure;
uniform float reach;           // how far a ray travels, in boxes, before it gives up
uniform int steps;
uniform int contained;         // stop at the box, or carry on through its copies
uniform vec3 backdrop;
out vec4 color;

/// Where a ray enters and leaves the box. A periodic world has no far wall, so seeing one as a single
/// object means choosing to stop at its faces; flying inside one means choosing not to.
vec2 faces(vec3 origin, vec3 ray, float size) {
  vec3 slope = 1.0 / (ray + 1e-9);
  vec3 near = (vec3(0.0) - origin) * slope;
  vec3 far = (vec3(size) - origin) * slope;
  vec3 low = min(near, far);
  vec3 high = max(near, far);
  return vec2(max(max(low.x, low.y), low.z), min(min(high.x, high.y), high.z));
}

vec4 field(vec3 position) { return texture(volume, position / boxLen); } // the sampler wraps, so space does

vec3 hue(float turn) {
  vec3 wheel = abs(mod(turn * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0;
  return clamp(wheel, 0.0, 1.0);
}
/// Color says what the neighborhood is made of: an angle where the traits agree, grey where they do not,
/// so a mixed region never reads as a confident color it does not have.
vec3 tint(vec4 read) {
  float coherence = length(read.gb) / max(read.r, 1e-6);
  return mix(vec3(0.72, 0.74, 0.78), hue(atan(read.b, read.g) / 6.28318530718 + 0.5), clamp(coherence, 0.0, 1.0));
}
void main() {
  vec2 screen = (gl_FragCoord.xy / viewport) * 2.0 - 1.0;
  screen.x *= viewport.x / viewport.y;
  screen -= shift;
  vec3 ray = normalize(right * screen.x * tanHalf + up * screen.y * tanHalf + ahead);
  float entry = 0.0;
  float span = reach * boxLen;
  if (contained == 1) {
    vec2 hit = faces(eye, ray, boxLen);
    entry = max(hit.x, 0.0);
    span = min(hit.y - entry, span);
    if (span <= 0.0) { color = vec4(backdrop, 1.0); return; } // this ray never meets the world
  }
  float stride = span / float(steps);
  vec3 position = eye + ray * (entry + stride * 0.5);
  vec3 key = normalize(vec3(0.4, 0.85, -0.3));
  vec3 gathered = vec3(0.0);
  float clarity = 1.0; // what is left of the ray after everything it has already passed through
  // A torus has no far wall, so a ray would otherwise pile up copies of the world until every pixel read
  // as fog. Distance alone dims what is behind, which is what gives the picture depth.
  float dimming = exp(-haze * stride / boxLen);
  for (int taken = 0; taken < steps; taken++) {
    clarity *= dimming;
    if (clarity < 0.004) break;
    vec4 read = field(position);
    float density = read.r / uniform_density;
    if (density > iso) {
      float excess = density - iso;
      float opacity = 1.0 - exp(-absorption * excess * stride);
      float step = boxLen / 128.0; // one gradient tap, wide enough to outrun the field's own sampling noise
      vec3 slope = vec3(
        field(position + vec3(step, 0.0, 0.0)).r - field(position - vec3(step, 0.0, 0.0)).r,
        field(position + vec3(0.0, step, 0.0)).r - field(position - vec3(0.0, step, 0.0)).r,
        field(position + vec3(0.0, 0.0, step)).r - field(position - vec3(0.0, 0.0, step)).r);
      vec3 normal = -normalize(slope + vec3(1e-9));    // density falls outward, so the surface faces up the slope
      vec3 shade = tint(read);
      float lambert = max(dot(normal, key), 0.0);
      float sheen = pow(max(dot(normal, -ray), 0.0), 24.0);
      vec3 lit = shade * (0.22 + 0.78 * lambert) + vec3(sheen) * 0.16;
      lit += shade * excess * excess * emission;        // interior glow, so thick material reads as material
      gathered += clarity * opacity * lit;
      clarity *= 1.0 - opacity;
      if (clarity < 0.01) break;                       // nothing behind this will be seen
    }
    position += ray * stride;
  }
  color = vec4(gathered * exposure + backdrop * clarity, 1.0);
}`;

const POINT_VERTEX = `#version 300 es
precision highp float;
uniform sampler2D particles;
uniform vec3 eye;
uniform vec3 right;
uniform vec3 up;
uniform vec3 ahead;
uniform vec2 viewport;
uniform vec2 shift; // how far the picture slides so panels covering the canvas do not push it off center
uniform float tanHalf;
uniform float grain; // world radius one bead stands for
out float trait;
void main() {
  ivec2 texel = ivec2(gl_VertexID % 1024, gl_VertexID / 1024);
  vec4 read = texelFetch(particles, texel, 0);
  vec3 offset = read.xyz - eye;
  float away = dot(offset, ahead);
  if (away <= 0.001) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; } // behind the eye
  trait = read.w;
  float aspect = viewport.x / viewport.y;
  gl_Position = vec4(dot(offset, right) / (tanHalf * aspect) + shift.x * away / aspect,
    dot(offset, up) / tanHalf + shift.y * away, 0.0, away);
  gl_PointSize = clamp(grain / (away * tanHalf) * viewport.y, 1.5, 40.0);
}`;

const POINT_FRAGMENT = `#version 300 es
precision highp float;
in float trait;
out vec4 color;
vec3 hue(float turn) {
  vec3 wheel = abs(mod(turn * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0;
  return clamp(wheel, 0.0, 1.0);
}
void main() {
  vec2 offset = gl_PointCoord * 2.0 - 1.0;
  float squared = dot(offset, offset);
  if (squared > 1.0) discard;
  float fall = 1.0 - squared;
  // Beads are grain under the material, not the subject: pale, soft, and dim enough that a thousand of
  // them read as a cloud with depth instead of as confetti.
  vec3 tone = mix(vec3(0.82, 0.86, 0.92), hue(trait), 0.45);
  color = vec4(tone * fall * 0.34, fall * 0.34);
}`;

/// Mass of the compact kernel above over a sphere of radius one, which is what turns a deposit into a
/// density: 4 pi times the integral of q^2 (1 - q^2)^3.
const KERNEL_MASS = 4.0 * Math.PI * 16.0 / 315.0;

export function createWorld(canvas) {
  const gl = canvas.getContext("webgl2", { antialias: false, alpha: false, depth: false });
  const view = {
    // Beads stay on beside the material: a world that has not organized yet has no material to show, and a
    // blank canvas reads as a broken renderer rather than as a uniform gas.
    ready: false, message: "", material: true, beads: true, contained: true,
    side: 96, support: 0, spacing: 2.0, blend: 1.1, iso: 1.6, absorption: 1.4, haze: 1.6,
    emission: 0.05, exposure: 1.4, reach: 2.0, steps: 112, scale: 0.9, grain: 0.16,
    count: 0, boxLen: 1, margin: { left: 0, right: 0 },
    camera: { yaw: 0.6, pitch: -0.35, radius: 1.0, position: [0, 0, 0], focal: 1.0, flying: false, speed: 0.6 },
  };
  // Without a context there is nothing to draw, but the page goes on sending frames and pointer moves.
  // Every entry point answers, doing nothing, so the toast is what a viewer sees instead of a TypeError
  // on every event that arrives afterwards.
  if (!gl) {
    view.message = "this browser has no WebGL2";
    for (const name of ["setFrame", "render", "frameWorld", "orbit", "dolly", "fly", "touch", "mark"]) {
      view[name] = () => {};
    }
    return view;
  }
  const float = gl.getExtension("EXT_color_buffer_float");
  if (!float) { view.message = "no float render targets, so material view is off"; view.material = false; }

  const deposit = program(DEPOSIT_VERTEX, DEPOSIT_FRAGMENT);
  const march = program(SCREEN_VERTEX, MARCH_FRAGMENT);
  const beads = program(POINT_VERTEX, POINT_FRAGMENT);
  const empty = gl.createVertexArray(); // every pass reads from a texture, so no pass owns a buffer
  let particles = gl.createTexture();
  let particleWidth = 0, particleHeight = 0; // what that texture was given room for, so it is only remade on a change
  let room = null; // the swarm as the texture wants it, refilled in place frame after frame
  const frame = gl.createFramebuffer();
  let volume = null;
  let volumeSide = 0;
  let stale = true; // positions moved, so the field owes a rebuild
  let dirty = true; // something on screen would look different, so the picture owes a redraw
  view.ready = true;
  const mark = () => { dirty = true; };

  function program(vertexSource, fragmentSource) {
    const built = gl.createProgram();
    for (const [kind, source] of [[gl.VERTEX_SHADER, vertexSource], [gl.FRAGMENT_SHADER, fragmentSource]]) {
      const shader = gl.createShader(kind);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        view.message = gl.getShaderInfoLog(shader).split("\n")[0];
        console.error(gl.getShaderInfoLog(shader), source);
      }
      gl.attachShader(built, shader);
    }
    gl.linkProgram(built);
    if (!gl.getProgramParameter(built, gl.LINK_STATUS)) view.message = gl.getProgramInfoLog(built);
    const at = {};
    for (let slot = 0; slot < gl.getProgramParameter(built, gl.ACTIVE_UNIFORMS); slot++) {
      const name = gl.getActiveUniform(built, slot).name;
      at[name] = gl.getUniformLocation(built, name);
    }
    return { built, at };
  }

  /// One frame from the harness. Everything scales off the count that actually arrived: blend reach from
  /// its spacing, brightness from its density. A thinned world therefore reads as the same material at a
  /// slightly smoother grain instead of as a different one.
  function setFrame(positions, traits, boxLen) {
    const count = Math.min(traits.length, 1024 * 1024);
    // The harness decimates far below this, so reaching it means the two ends disagree about the cap
    // rather than that a viewer wanted a million beads.
    if (traits.length > count) console.warn(`frame carried ${traits.length} points, drawing ${count}`);
    const width = Math.min(1024, Math.max(1, count));
    const height = Math.ceil(Math.max(1, count) / 1024);
    // One buffer and one texture for as long as the count holds, refilled in place. Staging a frame
    // through a pack buffer and then a texture-sized one, and respecifying the texture on top, was four
    // allocations and four copies of the swarm twenty-five times a second.
    if (room === null || room.length < width * height * 4) room = new Float32Array(width * height * 4);
    if (width !== particleWidth || height !== particleHeight) {
      gl.deleteTexture(particles);
      particles = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, particles);
      gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA32F, width, height);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      particleWidth = width; particleHeight = height;
    }
    for (let particle = 0; particle < count; particle++) {
      room[particle * 4] = positions[particle * 3];
      room[particle * 4 + 1] = positions[particle * 3 + 1];
      room[particle * 4 + 2] = positions[particle * 3 + 2];
      room[particle * 4 + 3] = traits[particle];
    }
    gl.bindTexture(gl.TEXTURE_2D, particles);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, width, height, gl.RGBA, gl.FLOAT, room);
    view.count = count;
    if (Math.abs(boxLen - view.boxLen) > 1e-6) {
      // The box moves whenever coordination does. Riding it keeps the framing a viewer chose instead of
      // dropping them inside the material or leaving them a speck away from it. A box that changed by a
      // quarter is a different world, so that gets framed again, unless someone is flying through it.
      const ratio = boxLen / view.boxLen;
      const fresh = view.camera.radius <= 1.0001 || (!view.camera.flying && Math.abs(ratio - 1) > 0.25);
      for (let axis = 0; axis < 3; axis++) {
        view.camera.position[axis] = boxLen * 0.5 + (view.camera.position[axis] - view.boxLen * 0.5) * ratio;
      }
      view.camera.radius *= ratio;
      view.boxLen = boxLen;
      if (fresh) frameWorld();
    }
    view.spacing = boxLen / Math.cbrt(Math.max(count, 1));
    stale = true;
    mark();
  }

  /// Put the box in the middle of the picture: back off along whatever direction the camera is already
  /// looking, so framing never changes where the world appears to be.
  function frameWorld() {
    mark();
    const middle = view.boxLen * 0.5;
    const radius = view.boxLen * 1.7; // far enough out that the whole box fits, corners included
    const { ahead } = basis();
    view.camera.radius = radius;
    view.camera.position = ahead.map((axis) => middle - axis * radius);
  }

  /// Rebuild the field one layer at a time. A layer is one draw of every particle's four periodic
  /// images, and the vertex shader throws away everything that cannot reach it.
  function rebuild() {
    if (!view.material || view.count === 0) return;
    // Blend reach follows the spacing of what actually arrived rather than an absolute size, so the
    // picture keeps its shape at any particle count. Two spacings holds about thirty neighbors.
    view.support = view.spacing * view.blend;
    const side = Math.max(16, Math.min(128, view.side | 0));
    if (side !== volumeSide) {
      if (volume) gl.deleteTexture(volume);
      volume = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_3D, volume);
      gl.texStorage3D(gl.TEXTURE_3D, 1, gl.RGBA16F, side, side, side);
      for (const axis of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T, gl.TEXTURE_WRAP_R]) {
        gl.texParameteri(gl.TEXTURE_3D, axis, gl.REPEAT); // the world is a torus, so the field is too
      }
      gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      volumeSide = side;
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, frame);
    gl.viewport(0, 0, side, side);
    gl.useProgram(deposit.built);
    gl.bindVertexArray(empty);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, particles);
    gl.uniform1i(deposit.at.particles, 0);
    gl.uniform1i(deposit.at.count, view.count);
    gl.uniform1f(deposit.at.boxLen, view.boxLen);
    gl.uniform1f(deposit.at.support, view.support);
    gl.uniform1f(deposit.at.amplitude, 1.0 / (KERNEL_MASS * Math.pow(view.support, 3)));
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE); // mass adds up, which is the whole point of a density
    gl.clearColor(0, 0, 0, 0); // one setting for every layer, not one per layer
    for (let layer = 0; layer < side; layer++) {
      gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, volume, 0, layer);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.uniform1f(deposit.at.layerZ, (layer + 0.5) / side * view.boxLen);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, view.count * 4);
    }
    gl.disable(gl.BLEND);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    stale = false;
  }

  // Reading a layout box every refresh makes the browser settle layout every refresh. An observer says
  // when the canvas actually changed, which is the only time the answer is a different one.
  let resized = true;
  let grained = 0; // the scale the drawing buffer was last sized for, since a slider moves it too
  new ResizeObserver(() => { resized = true; mark(); }).observe(canvas);
  function resize() {
    const grain = view.scale * window.devicePixelRatio;
    if (!resized && grain === grained) return;
    resized = false; grained = grain;
    const width = Math.max(1, Math.round(canvas.clientWidth * grain));
    const height = Math.max(1, Math.round(canvas.clientHeight * grain));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width; canvas.height = height;
      mark();
    }
  }

  /// Camera basis from two angles. Orbit keeps a point in front at a fixed distance; flying keeps the
  /// angles and moves the eye, so the two share one representation and never disagree.
  function basis() {
    const { yaw, pitch } = view.camera;
    const ahead = [Math.sin(yaw) * Math.cos(pitch), Math.sin(pitch), Math.cos(yaw) * Math.cos(pitch)];
    const right = [Math.cos(yaw), 0, -Math.sin(yaw)];
    const up = [right[1] * ahead[2] - right[2] * ahead[1], right[2] * ahead[0] - right[0] * ahead[2],
      right[0] * ahead[1] - right[1] * ahead[0]];
    return { ahead, right, up };
  }
  function center() {
    const { ahead } = basis();
    const { position, radius } = view.camera;
    return [position[0] + ahead[0] * radius, position[1] + ahead[1] * radius, position[2] + ahead[2] * radius];
  }
  function orbit(dx, dy) {
    mark();
    const pivot = center();
    view.camera.yaw -= dx * 0.005;
    view.camera.pitch = Math.max(-1.5, Math.min(1.5, view.camera.pitch - dy * 0.005));
    const { ahead } = basis();
    view.camera.position = [pivot[0] - ahead[0] * view.camera.radius, pivot[1] - ahead[1] * view.camera.radius,
      pivot[2] - ahead[2] * view.camera.radius];
  }
  function dolly(amount) {
    mark();
    const pivot = center();
    view.camera.radius = Math.max(view.boxLen * 0.05, Math.min(view.boxLen * 6, view.camera.radius * (1 - amount * 0.0015)));
    const { ahead } = basis();
    view.camera.position = [pivot[0] - ahead[0] * view.camera.radius, pivot[1] - ahead[1] * view.camera.radius,
      pivot[2] - ahead[2] * view.camera.radius];
  }
  function fly(move, seconds) {
    mark();
    const { ahead, right, up } = basis();
    const pace = view.boxLen * view.camera.speed * seconds;
    for (let axis = 0; axis < 3; axis++) {
      view.camera.position[axis] += (ahead[axis] * move[2] + right[axis] * move[0] + up[axis] * move[1]) * pace;
    }
  }

  /// Redraws only when something would look different. A paused world under a still camera is already on
  /// screen, and marching it again every refresh spends a GPU to arrive at the same pixels.
  function render() {
    resize();
    if (!dirty && !stale) return false;
    dirty = false;
    if (stale) rebuild();
    const { ahead, right, up } = basis();
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.bindVertexArray(empty);
    const tanHalf = Math.tan(view.camera.focal * 0.5);
    // The canvas runs under the panels, so the free band between them is where the world belongs.
    const aspect = canvas.width / canvas.height;
    const slide = (view.margin.left - view.margin.right) / canvas.clientWidth * aspect;
    if (view.material && volume) {
      gl.useProgram(march.built);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_3D, volume);
      gl.uniform1i(march.at.volume, 0);
      gl.uniform3fv(march.at.eye, view.camera.position);
      gl.uniform3fv(march.at.right, right);
      gl.uniform3fv(march.at.up, up);
      gl.uniform3fv(march.at.ahead, ahead);
      gl.uniform2f(march.at.viewport, canvas.width, canvas.height);
      gl.uniform2f(march.at.shift, slide, 0);
      gl.uniform1f(march.at.tanHalf, tanHalf);
      gl.uniform1f(march.at.boxLen, view.boxLen);
      gl.uniform1f(march.at.uniform_density, Math.max(view.count, 1) / Math.pow(view.boxLen, 3));
      gl.uniform1f(march.at.iso, view.iso);
      gl.uniform1f(march.at.absorption, view.absorption);
      gl.uniform1f(march.at.haze, view.haze);
      gl.uniform1f(march.at.emission, view.emission);
      gl.uniform1f(march.at.exposure, view.exposure);
      gl.uniform1f(march.at.reach, view.reach);
      gl.uniform1i(march.at.steps, Math.max(16, Math.min(320, view.steps | 0)));
      gl.uniform1i(march.at.contained, view.contained ? 1 : 0);
      gl.uniform3f(march.at.backdrop, 0.027, 0.031, 0.039);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    } else {
      gl.clearColor(0.027, 0.031, 0.039, 1.0);
      gl.clear(gl.COLOR_BUFFER_BIT);
    }
    if (view.beads || !view.material) {
      gl.useProgram(beads.built);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, particles);
      gl.uniform1i(beads.at.particles, 0);
      gl.uniform3fv(beads.at.eye, view.camera.position);
      gl.uniform3fv(beads.at.right, right);
      gl.uniform3fv(beads.at.up, up);
      gl.uniform3fv(beads.at.ahead, ahead);
      gl.uniform2f(beads.at.viewport, canvas.width, canvas.height);
      gl.uniform2f(beads.at.shift, slide, 0);
      gl.uniform1f(beads.at.tanHalf, tanHalf);
      gl.uniform1f(beads.at.grain, view.spacing * view.grain);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
      gl.drawArrays(gl.POINTS, 0, view.count);
      gl.disable(gl.BLEND);
    }
    return true;
  }

  view.setFrame = setFrame;
  view.render = render;
  view.frameWorld = frameWorld;
  view.orbit = orbit;
  view.dolly = dolly;
  view.fly = fly;
  view.touch = () => { stale = true; mark(); };
  view.mark = mark; // a setting changed that only the shader will notice
  return view;
}
