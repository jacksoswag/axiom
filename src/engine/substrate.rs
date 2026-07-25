//! Substrate is where the periodic space is defined and particles are randomly generated and
//! stored, together with the spatial index derived from their positions. The index is a cache,
//! not authoritative state, and reflects positions as of the last rebuild_grid call

use std::sync::atomic::{AtomicU64, Ordering};

use crate::engine::params::FixedGenome;
use crate::engine::r#trait::Membership;
use crate::util::Rng;

/// A number no other substrate in this process has held. Nothing here reads it: it exists so a cache
/// downstream, on a card or anywhere else, can ask whether the population it holds is still the one
/// in hand without comparing every trait in it.
fn fresh_id() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

pub struct Substrate {
    pub id: u64, // this population's identity, so a cache elsewhere can tell one from another
    pub positions: Vec<f32>, // particle positions, dimensions floats each
    pub traits: Vec<f32>, // species phenotype blend
    pub memberships: Vec<Membership>, // each particle's anchors, a pure function of its trait
    pub box_len: f32,
    pub softening_sq: f32, // kept squared: every caller adds it to a squared distance
    pub dimensions: usize,
    pub delta: Vec<f32>, // a step's position change, held so a tick refills a buffer instead of allocating one
    grid: Grid,
}
/// Spatial hash grid. Cache derived from Substrate's own positions
#[derive(Default)]
struct Grid {
    grid_len: usize, // cells per axis
    cell_len: f32, // edge length of one cell
    sort: Vec<u32>, // particle indices grouped by cell
    map: Vec<u32>,  // cell edges in list-space (CSR optimization)
    cursor: Vec<u32>, // scatter position per cell, a refilled copy of map rather than a fresh clone
    cells: Vec<u32>, // each particle's home cell, found once and read by both passes below
}

impl Substrate {
    /// A substrate from fixed_genome: positions seeded at random, traits stay zero until init_particle_traits runs
    pub fn build(fg: &FixedGenome, box_len: f32) -> Substrate {
        let (particles, dims, seed) = (fg.particle_count, fg.dimensions, fg.seed);
        let mut substrate = Substrate {
            id: fresh_id(),
            positions: vec![0.0; particles * dims],
            traits: vec![0.0; particles],
            memberships: Vec::new(),
            box_len, softening_sq: (box_len * 1e-3).powi(2),
            dimensions: dims,
            delta: Vec::new(),
            grid: Grid::default()};
        let mut rng = Rng::new(seed); // seed particles at random positions
        for position in substrate.positions.iter_mut() { *position = rng.unit() * box_len; }
        substrate
    }
    /// Another substrate holding this one's particles: same box, same state, an empty index. Skips the
    /// random position fill build pays for, which a caller copying over it would only throw away.
    pub fn copy(&self) -> Substrate {
        Substrate {
            id: fresh_id(),
            positions: self.positions.clone(), traits: self.traits.clone(),
            memberships: self.memberships.clone(),
            box_len: self.box_len, softening_sq: self.softening_sq,
            dimensions: self.dimensions,
            delta: Vec::new(), grid: Grid::default()}
    }
    /// Cache current positions for a torus of side box_len, one cell per cutoff. Stays inactive
    /// below three cells per axis, where the 3^dims stencil wraps onto itself and affects particles twice.
    pub fn rebuild_grid(&mut self, cutoff: f32) {
        let dims = self.dimensions;
        let box_len = self.box_len;
        let grid_len = self.grid_length(cutoff);

        if grid_len == 0 { self.grid.grid_len = 0; return; }
        self.grid.grid_len = grid_len;
        self.grid.cell_len = box_len / grid_len as f32;
        let total_cells = grid_len.pow(dims as u32); // grid_length already turned away anything that overflows here
        let particle_count = self.positions.len().checked_div(dims).unwrap_or(0);

        // Home cells, once. Both passes below want them and finding one costs a remainder and a
        // divide per axis, so the second pass reads this instead of walking the positions again.
        let mut cells = std::mem::take(&mut self.grid.cells); // out of the way, so the fill can read positions
        cells.clear();
        for i in 0..particle_count { cells.push(self.cell_of(i) as u32); }

        // Build map (cell -> particle index)
        self.grid.map.clear();
        self.grid.map.resize(total_cells + 1, 0);
        for &cell in &cells { // count particles in each cell
            self.grid.map[cell as usize + 1] += 1;
        }
        for c in 0..total_cells { // make cumulative
            self.grid.map[c + 1] += self.grid.map[c];
        }
        // Build sort (particles sorted by the above cell-map)
        self.grid.sort.resize(particle_count, 0);
        self.grid.cursor.resize(self.grid.map.len(), 0);
        self.grid.cursor.copy_from_slice(&self.grid.map); // where the next particle of each cell lands
        for (i, &cell) in cells.iter().enumerate() {
            self.grid.sort[self.grid.cursor[cell as usize] as usize] = i as u32;
            self.grid.cursor[cell as usize] += 1;
        }
        self.grid.cells = cells; // buffer back where the next rebuild finds it
    }

    /// Visit every particle in position's cell and the 3^dims adjacent cells, callers still distance-check
    pub fn visit_neighbors(&self, position: &[f32], mut visit: impl FnMut(usize)) {
        let grid_len = self.grid.grid_len; let dims = self.dimensions;
        if grid_len == 0 { // O(n^2) fallback path
            for particle in 0..self.traits.len() {
                visit(particle); // run the desired function on each candidate
            } return;
        }
        // The home cell is the same for all 3^dims combos, so it is found once per axis here rather
        // than 3^dims times inside the loop: 81 remainders and divides in 3D where 3 do the job.
        let mut home = [0i32; Self::MAX_AXES];
        for axis in 0..dims { home[axis] = self.axis_cell(position[axis]) as i32; }
        // Main path (Grid): walk position's home cell plus all 3^dims neighbors (9 in 2D, 27 in 3D, etc)
        for combo_index in 0..3usize.pow(dims as u32) {
            let mut cell = 0; let mut rest = combo_index;
            for axis in 0..dims {
                let offset = (rest % 3) as i32 - 1; // this axis's neighbor offset: -1, 0, or +1
                rest /= 3; // read next dimension axis next
                cell = cell * grid_len + (home[axis] + offset).rem_euclid(grid_len as i32) as usize; // fold into flat cell index
            }
            // Compile particle list in this cell
            let cell_start = self.grid.map[cell] as usize;
            let cell_end = self.grid.map[cell + 1] as usize;
            for &particle in &self.grid.sort[cell_start..cell_end] {
                visit(particle as usize); // visit particles in this neighbor cell
            }
        }
    }
    /// Axes one stack-held home cell has room for. A genome past this indexes off the end, which is
    /// as far past what the measurement grid accepts (3) as a shape can travel.
    const MAX_AXES: usize = 8;
    /// Which cell index coordinate falls into, along one axis.
    fn axis_cell(&self, coordinate: f32) -> usize {
        let box_len = self.grid.cell_len * self.grid.grid_len as f32; // full box length, this axis
        ((coordinate.rem_euclid(box_len) / self.grid.cell_len) as usize).min(self.grid.grid_len - 1)
    }
    /// Which flat cell a particle lives in, across all axes.
    fn cell_of(&self, i: usize) -> usize {
        let dims = self.dimensions;
        let p = &self.positions[i * dims..]; // this particle's coordinates start here
        let mut cell = 0;
        for &coordinate in &p[..dims] { // only this particle's dims coordinates, not the rest
            cell = cell * self.grid.grid_len + self.axis_cell(coordinate); } // fold this axis in
        cell
    }
    /// One particle's coordinates, the slice arithmetic lives here so callers never repeat it.
    pub fn pos(&self, particle: usize) -> &[f32] {
        &self.positions[particle * self.dimensions..(particle + 1) * self.dimensions]
    }
    /// The index as numbers, for a caller that has to ship it somewhere this crate does not reach.
    /// Cells per axis of zero means the last rebuild found no index worth having and every pair is
    /// being walked, which is the one shape a caller must not hand to something expecting a grid.
    pub fn grid_shape(&self) -> (usize, f32, usize) {
        (self.grid.grid_len, self.grid.cell_len, self.grid.map.len().saturating_sub(1))
    }
    pub fn grid_map(&self) -> &[u32] { &self.grid.map }
    pub fn grid_sorted(&self) -> &[u32] { &self.grid.sort }
    /// Whether a walk at this cutoff takes the grid or every pair. The grid is rebuilt inside a step;
    /// this answers ahead of one, so a caller can say which regime a law is running in. A matrix whose
    /// shells are all inert reaches zero, which lands here as all-pairs, though step bails on such a
    /// law before it walks anything.
    pub fn gridded(&self, cutoff: f32) -> bool { self.grid_length(cutoff) != 0 }
    /// Cells per axis at this cutoff, or 0 for the O(n^2) path. Three floors meet here. Below three
    /// cells per axis the stencil wraps onto itself and touches a particle twice, and params.rs lets a
    /// kernel reach 0.375 of the box, so that is legally reachable. Past CELL_CEILING the map is too
    /// big to hold. And a cell count far above the particle count is its own trap: a legal small-reach
    /// genome asks for a million cells to hold three hundred particles, where clearing and scanning
    /// the map costs more per tick than every pair it prunes. Capping at roughly four cells per
    /// particle stays correct because a cell wider than the cutoff is still covered by the same
    /// 3-wide stencil; it just prunes a little less.
    fn grid_length(&self, cutoff: f32) -> usize {
        if !(cutoff > 0.0) { return 0; } // an inert law reaches nothing, and infinity casts to a legal length
        let dims = self.dimensions;
        let particle_count = self.positions.len().checked_div(dims).unwrap_or(0);
        let by_particles = ((4 * particle_count.max(1)) as f64).powf(1.0 / dims as f64).ceil() as usize;
        let len = ((self.box_len / cutoff) as usize).min(by_particles.max(3));
        if len < 3 || len.saturating_pow(dims as u32) > Self::CELL_CEILING { 0 } else { len }
    }
    /// Most cells the index will hold, and the only one of the three floors above that is about memory
    /// rather than correctness. The four-per-particle cap is what actually binds at every size the
    /// shape allows; this sits above it so that a swarm of a million still indexes rather than falling
    /// back to walking every pair, which at that size is a hundred billion of them per step.
    const CELL_CEILING: usize = 1 << 24;
}
