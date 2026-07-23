//! Deterministic uniform spatial hash over particle positions in `dims`-D. A derived cache:
//! rebuilt from positions each step, never part of authoritative state.

/// Particles binned into a uniform grid, stored CSR-style: `order` lists particle indices
/// grouped by cell, and `start[c]..start[c + 1]` is cell `c`'s slice of `order`.
#[derive(Default)]
pub struct Grid {
    dims: usize,     // spatial dimensionality this index was built for
    grid_len: usize, // cells per axis
    cell_len: f32,   // edge length of one cell
    start: Vec<u32>,
    order: Vec<u32>,
    active: bool,
}

impl Grid {
    /// Bin `positions` (`dims` floats each) for a torus of side `bound_len`, one cell per
    /// `cutoff`. Stays inactive below three cells per axis, where the 3^dims stencil would wrap
    /// onto itself and visit a particle twice.
    pub fn rebuild(&mut self, positions: &[f32], dims: usize, bound_len: f32, cutoff: f32) {
        self.active = false;
        self.dims = dims;
        let particle_count = positions.len().checked_div(dims).unwrap_or(0);
        // A non-positive cutoff or bound length casts to 0 or a saturated max; both fail the guard.
        let grid_len = (bound_len / cutoff) as usize;
        if particle_count == 0
            || grid_len < 3
            || grid_len.saturating_pow(dims as u32) > 1 << 21
        {
            return;
        }
        self.grid_len = grid_len;
        self.cell_len = bound_len / grid_len as f32;
        let total = grid_len.pow(dims as u32);

        self.start.clear();
        self.start.resize(total + 1, 0);
        for i in 0..particle_count {
            let cell = self.cell_of(positions, i);
            self.start[cell + 1] += 1;
        }
        for c in 0..total {
            self.start[c + 1] += self.start[c];
        }

        self.order.resize(particle_count, 0);
        let mut cursor = self.start.clone();
        for i in 0..particle_count {
            let cell = self.cell_of(positions, i);
            self.order[cursor[cell] as usize] = i as u32;
            cursor[cell] += 1;
        }
        self.active = true;
    }

    /// Visit every particle that could lie within reach of `position`: the 3^dims toroidally
    /// adjacent cells when active, every particle otherwise. Callers never branch on `active()`.
    pub fn for_each_candidate(&self, position: &[f32], count: usize, mut visit: impl FnMut(usize)) {
        if !self.active {
            for particle in 0..count {
                visit(particle);
            }
            return;
        }
        let side = self.grid_len as i32;
        // Walk every {-1,0,1}^dims offset as a mixed-radix (base 3) counter, folding each
        // combination straight into a cell index so no per-particle scratch is allocated.
        for combo in 0..3usize.pow(self.dims as u32) {
            let mut cell = 0;
            let mut rest = combo;
            for &coordinate in &position[..self.dims] {
                let offset = (rest % 3) as i32 - 1;
                rest /= 3;
                let home = self.axis_cell(coordinate) as i32;
                cell = cell * self.grid_len + (home + offset).rem_euclid(side) as usize;
            }
            for &particle in &self.order[self.start[cell] as usize..self.start[cell + 1] as usize] {
                visit(particle as usize);
            }
        }
    }

    fn axis_cell(&self, coordinate: f32) -> usize {
        // A non-finite coordinate casts to 0; the clamp keeps one landing exactly on the bound
        // inside the last cell. Positions arrive already wrapped into [0, bound_len).
        let bound_len = self.cell_len * self.grid_len as f32;
        ((coordinate.rem_euclid(bound_len) / self.cell_len) as usize).min(self.grid_len - 1)
    }

    fn cell_of(&self, positions: &[f32], i: usize) -> usize {
        let p = &positions[i * self.dims..];
        let mut cell = 0;
        for &coordinate in &p[..self.dims] {
            cell = cell * self.grid_len + self.axis_cell(coordinate);
        }
        cell
    }
}
