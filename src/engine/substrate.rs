//! Substrate is where the periodic space is defined and particles are randomly generated and
//! stored, together with the spatial index derived from their positions. The index is a cache,
//! not authoritative state, and reflects positions as of the last rebuild_grid call

use crate::engine::params::Params;
use crate::util::Rng;

pub struct Substrate {
    pub positions: Vec<f32>, // particle positions, dimensions floats each
    pub traits: Vec<f32>, // species phenotype blend
    pub box_len: f32,
    pub softening: f32,
    pub dimensions: usize,
    grid: Grid,
}
/// Spatial hash grid. Cache derived from Substrate's own positions
#[derive(Default)]
struct Grid {
    grid_len: usize, // cells per axis
    cell_len: f32, // edge length of one cell
    sort: Vec<u32>, // particle indices grouped by cell
    map: Vec<u32>,  // cell edges in list-space (CSR optimization)
}

impl Substrate {
    /// A substrate from params: positions seeded at random, traits stay zero until init_particle_traits runs
    pub fn build(params: &Params) -> Substrate {
        let (particles, dims, box_len) = (params.particle_count, params.dimensions, params.box_len);
        let mut substrate = Substrate {
            positions: vec![0.0; particles * dims],
            traits: vec![0.0; particles],
            box_len, softening: box_len * 1e-3,
            dimensions: dims,
            grid: Grid::default()};
        let mut rng = Rng::new(params.seed); // seed particles at random positions
        for position in substrate.positions.iter_mut() { *position = rng.unit() * box_len; }
        substrate
    }
    /// Cache current positions for a torus of side box_len, one cell per cutoff. Stays inactive 
    /// below three cells per axis, where the 3^dims stencil wraps onto itself and affects particles twice.
    pub fn rebuild_grid(&mut self, cutoff: f32) {
        let dims = self.dimensions;
        let box_len = self.box_len;
        let grid_len = (box_len / cutoff) as usize;

        if !Self::grid_valid(dims, grid_len) { self.grid.grid_len = 0; return; }
        self.grid.grid_len = grid_len;
        self.grid.cell_len = box_len / grid_len as f32;
        let total_cells = grid_len.pow(dims as u32);
        let particle_count = self.positions.len().checked_div(dims).unwrap_or(0);

        // Build map (cell -> particle index)
        self.grid.map.clear();
        self.grid.map.resize(total_cells + 1, 0);
        for i in 0..particle_count { // count particles in each cell
            let cell = self.cell_of(i);
            self.grid.map[cell + 1] += 1;
        }
        for c in 0..total_cells { // make cumulative
            self.grid.map[c + 1] += self.grid.map[c];
        }
        // Build sort (particles sorted by the above cell-map)
        self.grid.sort.resize(particle_count, 0);
        let mut temp_map = self.grid.map.clone();
        for i in 0..particle_count {
            let cell = self.cell_of(i);
            self.grid.sort[temp_map[cell] as usize] = i as u32;
            temp_map[cell] += 1;
        }
    }

    /// Visit every particle in position's cell and the 3^dims adjacent cells, callers still distance-check
    pub fn visit_neighbors(&self, position: &[f32], mut visit: impl FnMut(usize)) {
        let grid_len = self.grid.grid_len; let dims = self.dimensions;
        if !Self::grid_valid(dims, grid_len) { // O(n^2) fallback path
            for particle in 0..self.traits.len() {
                visit(particle); // run the desired function on each candidate
            } return;
        }
        // Main path (Grid): walk position's home cell plus all 3^dims neighbors (9 in 2D, 27 in 3D, etc)
        for combo_index in 0..3usize.pow(dims as u32) {
            let mut cell = 0; let mut rest = combo_index;
            for &coordinate in &position[..dims] {
                let offset = (rest % 3) as i32 - 1; // this axis's neighbor offset: -1, 0, or +1
                rest /= 3; // read next dimension axis next
                let home = self.axis_cell(coordinate) as i32;
                cell = cell * grid_len + (home + offset).rem_euclid(grid_len as i32) as usize; // fold into flat cell index
            }
            // Compile particle list in this cell
            let cell_start = self.grid.map[cell] as usize;
            let cell_end = self.grid.map[cell + 1] as usize;
            for &particle in &self.grid.sort[cell_start..cell_end] {
                visit(particle as usize); // visit particles in this neighbor cell
            }
        }
    }
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
    /// Grid path needs at least 3 cells per axis and few enough cells to allocate, otherwise O(n^2)
    fn grid_valid(dims: usize, len: usize) -> bool {
        !(len < 3 || len.saturating_pow(dims as u32) > 1 << 21)
    }
}
