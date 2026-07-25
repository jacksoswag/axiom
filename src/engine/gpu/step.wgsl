// One Particle-Lenia step, one thread per particle. The same walk lenia.rs does on the CPU: the
// grid's 3^dims stencil, the pair-indexed anchor field, and an explicit Euler step wrapped onto the
// torus. The grid itself arrives already built, since a prefix sum over the cells is a poor trade
// against the microseconds the CPU spends laying one out.
//
// Accumulation order matches the CPU walk exactly, because the cell map and its sorted particle list
// are the ones the CPU built. The arithmetic still is not bit-identical: exp() here is the hardware
// approximation rather than the one libm ships, so trajectories drift from the CPU's over a run.

struct Params {
  count: u32,
  anchor_count: u32,
  shell_count: u32,
  bump_count: u32,
  grid_len: u32,
  pad_a: u32,
  pad_b: u32,
  pad_c: u32,
  cell_len: f32,
  box_len: f32,
  softening_sq: f32,
  dt: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> positions: array<f32>;    // three per particle
@group(0) @binding(2) var<storage, read> memberships: array<f32>;  // four per particle: anchor, weight, anchor, weight
@group(0) @binding(3) var<storage, read> cell_map: array<u32>;     // cell edges in list-space
@group(0) @binding(4) var<storage, read> sorted: array<u32>;       // particle indices grouped by cell
@group(0) @binding(5) var<storage, read> pairs: array<vec4<f32>>;  // per interaction: weight, norm, reach_sq, spare
@group(0) @binding(6) var<storage, read> shells: array<vec4<f32>>; // amp, peak, 1/(2w^2), 1/w^2
@group(0) @binding(7) var<storage, read> bumps: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> next: array<f32>;   // where this tick's positions land

// A receiver has at most two active anchors and a source may be any of them, so a particle only ever
// touches anchor_count x 2 of the matrix. Sized for the largest shape the harness allows.
const MAX_SLOTS: u32 = 16u;

/// Rust's f32::rem_euclid: a truncated remainder lifted back above zero. WGSL's own % truncates the
/// same way, so this is the same two steps rather than the floor form, which disagrees on the seam.
fn rem_euclid(value: f32, span: f32) -> f32 {
  let rest = value % span;
  return select(rest, rest + span, rest < 0.0);
}
/// Rust's f32::round, which goes half away from zero. WGSL's round goes half to even, and the minimum
/// image below is the one place that difference would put a pair on the wrong side of the box.
fn away_from_zero(value: f32) -> f32 {
  return sign(value) * floor(abs(value) + 0.5);
}
/// Which cell index a coordinate falls into along one axis
fn axis_cell(coordinate: f32) -> i32 {
  let span = params.cell_len * f32(params.grid_len);
  let wrapped = rem_euclid(coordinate, span);
  return min(i32(wrapped / params.cell_len), i32(params.grid_len) - 1);
}

/// K(d) and its slope from one pair's shell mixture
fn sensed(x: f32, pair: u32) -> vec2<f32> {
  var value = 0.0;
  var slope = 0.0;
  let start = pair * params.shell_count;
  for (var k = 0u; k < params.shell_count; k = k + 1u) {
    let shell = shells[start + k];
    let gap = x - shell.y;
    let term = exp(-(gap * gap * shell.z)) * shell.x;
    value = value + term;
    slope = slope + (-gap * shell.w * term);
  }
  return vec2<f32>(value, slope);
}
/// G(u) and its slope from one pair's bump mixture. Same shape as sensed, over density rather than
/// distance; two arrays rather than one pointer because a storage pointer parameter is more machinery
/// than eight duplicated lines are worth.
fn grown(x: f32, pair: u32) -> vec2<f32> {
  var value = 0.0;
  var slope = 0.0;
  let start = pair * params.bump_count;
  for (var k = 0u; k < params.bump_count; k = k + 1u) {
    let bump = bumps[start + k];
    let gap = x - bump.y;
    let term = exp(-(gap * gap * bump.z)) * bump.x;
    value = value + term;
    slope = slope + (-gap * bump.w * term);
  }
  return vec2<f32>(value, slope);
}

@compute @workgroup_size(64)
fn advance(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= params.count) { return; }
  // Threads take particles in grid order, not index order. Neighbouring lanes then stand in the same
  // cell and walk the same twenty-seven cells, so the positions they read arrive together instead of
  // each lane pulling its own line out of a swarm that no longer fits in any cache.
  let i = sorted[id.x];
  let anchors = params.anchor_count;

  // Sensed potential and its gradient, per (source anchor, which of this particle's two). The matrix
  // is anchors squared but only this slice of it can ever be touched from here. All four numbers ride
  // in one row: an accumulator this size lives in thread memory rather than registers, and four
  // separately indexed arrays meant four trips out to it where one row means one.
  var sensed_field: array<vec4<f32>, MAX_SLOTS>; // potential, then the gradient's three axes
  for (var slot = 0u; slot < MAX_SLOTS; slot = slot + 1u) { sensed_field[slot] = vec4<f32>(0.0); }

  let pos_i = vec3<f32>(positions[i * 3u], positions[i * 3u + 1u], positions[i * 3u + 2u]);
  var home: array<i32, 3>;
  home[0] = axis_cell(pos_i.x); home[1] = axis_cell(pos_i.y); home[2] = axis_cell(pos_i.z);
  let span = i32(params.grid_len);

  // Home cell plus every neighbor cell, in the order the CPU visits them: axis 0 takes the least
  // significant trit of the combo and is the most significant digit of the flat cell index.
  for (var combo = 0u; combo < 27u; combo = combo + 1u) {
    var rest = combo;
    var cell = 0u;
    for (var axis = 0u; axis < 3u; axis = axis + 1u) {
      let offset = i32(rest % 3u) - 1;
      rest = rest / 3u;
      let folded = ((home[axis] + offset) % span + span) % span;
      cell = cell * params.grid_len + u32(folded);
    }
    let first = cell_map[cell];
    let last = cell_map[cell + 1u];
    for (var at = first; at < last; at = at + 1u) {
      let j = sorted[at];
      if (j == i) { continue; }
      let pos_j = vec3<f32>(positions[j * 3u], positions[j * 3u + 1u], positions[j * 3u + 2u]);
      // Softened distance under the minimum image, per axis in the order the CPU sums them
      var apart = pos_i - pos_j;
      let inverse = 1.0 / params.box_len;
      apart = apart - params.box_len * vec3<f32>(
        away_from_zero(apart.x * inverse), away_from_zero(apart.y * inverse), away_from_zero(apart.z * inverse));
      let apart_sq = apart.x * apart.x + apart.y * apart.y + apart.z * apart.z + params.softening_sq;

      var distance = -1.0; // lazy: the same value for every surviving pair below, sqrt at most once
      for (var s = 0u; s < 2u; s = s + 1u) {
        let source_weight = memberships[j * 4u + s * 2u + 1u];
        if (source_weight <= 0.0) { continue; }
        let source_anchor = u32(memberships[j * 4u + s * 2u]);
        for (var d = 0u; d < 2u; d = d + 1u) {
          let receiver_weight = memberships[i * 4u + d * 2u + 1u];
          if (receiver_weight <= 0.0) { continue; }
          let receiver_anchor = u32(memberships[i * 4u + d * 2u]);
          let pair = source_anchor * anchors + receiver_anchor;
          if (apart_sq > pairs[pair].z) { continue; } // squared compare skips the sqrt
          if (distance < 0.0) { distance = sqrt(apart_sq); }
          let reading = sensed(distance, pair);
          let slot = source_anchor * 2u + d;
          let scale = source_weight * reading.y / distance; // chain rule onto each axis
          sensed_field[slot] = sensed_field[slot]
            + vec4<f32>(source_weight * reading.x, scale * apart.x, scale * apart.y, scale * apart.z);
        }
      }
    }
  }

  // Turn accumulated potential into motion, over every possible source rather than the active ones
  var move_x = 0.0; var move_y = 0.0; var move_z = 0.0;
  for (var s = 0u; s < anchors; s = s + 1u) {
    for (var d = 0u; d < 2u; d = d + 1u) {
      let receiver_weight = memberships[i * 4u + d * 2u + 1u];
      if (receiver_weight <= 0.0) { continue; }
      let receiver_anchor = u32(memberships[i * 4u + d * 2u]);
      let pair = s * anchors + receiver_anchor;
      let law = pairs[pair]; // weight, norm, reach_sq, spare
      let slot = s * 2u + d;
      let held = sensed_field[slot];
      let response = grown(held.x / law.y, pair);
      // 2x: G = 2B-1 maps to [-1,1], and the -1 dies in G'
      let scale = 2.0 * params.dt * receiver_weight * law.x * response.y / law.y;
      move_x = move_x + scale * held.y;
      move_y = move_y + scale * held.z;
      move_z = move_z + scale * held.w;
    }
  }

  next[i * 3u] = rem_euclid(pos_i.x + move_x, params.box_len);
  next[i * 3u + 1u] = rem_euclid(pos_i.y + move_y, params.box_len);
  next[i * 3u + 2u] = rem_euclid(pos_i.z + move_z, params.box_len);
}
