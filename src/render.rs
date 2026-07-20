//! Visualization (§5.3) — false-color mapping and PNG output.
//!
//! 2-D false color is the primary debugger. Colormaps are anchor-interpolated
//! LUTs (compact, recognizable) rather than full 256-entry tables. The same
//! mapping feeds both the headless PNG writer and the live window, so views are
//! reproducible from config.

use crate::field::Field;
use anyhow::Result;
use std::path::Path;

/// Named perceptual colormaps, sampled at anchor stops and linearly interpolated.
pub fn colormap(name: &str, t: f32) -> [u8; 3] {
    let t = t.clamp(0.0, 1.0);
    let anchors: &[[f32; 3]] = match name {
        "turbo" => &TURBO,
        "magma" => &MAGMA,
        "gray" | "grey" => &GRAY,
        _ => &VIRIDIS,
    };
    interp(anchors, t)
}

fn interp(anchors: &[[f32; 3]], t: f32) -> [u8; 3] {
    let n = anchors.len();
    let scaled = t * (n - 1) as f32;
    let i = (scaled.floor() as usize).min(n - 2);
    let f = scaled - i as f32;
    let a = anchors[i];
    let b = anchors[i + 1];
    [
        ((a[0] + (b[0] - a[0]) * f) * 255.0).round() as u8,
        ((a[1] + (b[1] - a[1]) * f) * 255.0).round() as u8,
        ((a[2] + (b[2] - a[2]) * f) * 255.0).round() as u8,
    ]
}

/// Render one channel of a field into an RGB8 buffer (row-major, 3 bytes/pixel).
pub fn field_to_rgb(field: &Field, channel: usize, cmap: &str, scale: f32) -> Vec<u8> {
    let (h, w) = (field.h, field.w);
    let src = field.channel(channel);
    let mut rgb = vec![0u8; h * w * 3];
    for (i, &v) in src.iter().enumerate() {
        let [r, g, b] = colormap(cmap, v * scale);
        rgb[i * 3] = r;
        rgb[i * 3 + 1] = g;
        rgb[i * 3 + 2] = b;
    }
    rgb
}

/// Render one channel into a `0xAARRGGBB` buffer for the live window, with an
/// integer nearest-neighbor upscale.
#[cfg(feature = "window")]
pub fn field_to_argb(field: &Field, channel: usize, cmap: &str, scale: f32, up: usize, out: &mut Vec<u32>) {
    let (h, w) = (field.h, field.w);
    let src = field.channel(channel);
    out.resize(h * up * w * up, 0);
    for y in 0..h {
        for x in 0..w {
            let [r, g, b] = colormap(cmap, src[y * w + x] * scale);
            let px = 0xFF00_0000 | ((r as u32) << 16) | ((g as u32) << 8) | b as u32;
            for dy in 0..up {
                let row = (y * up + dy) * (w * up);
                for dx in 0..up {
                    out[row + x * up + dx] = px;
                }
            }
        }
    }
}

/// Save an RGB8 buffer as PNG.
pub fn save_png(path: &Path, rgb: &[u8], w: usize, h: usize) -> Result<()> {
    image::save_buffer(path, rgb, w as u32, h as u32, image::ColorType::Rgb8)?;
    Ok(())
}

/// Save a field channel directly as PNG.
pub fn save_field_png(path: &Path, field: &Field, channel: usize, cmap: &str, scale: f32) -> Result<()> {
    let rgb = field_to_rgb(field, channel, cmap, scale);
    save_png(path, &rgb, field.w, field.h)
}

/// Render a `rows × cols` scalar matrix (e.g. a graph spacetime diagram) to PNG.
pub fn save_matrix_png(path: &Path, mat: &[f32], rows: usize, cols: usize, cmap: &str) -> Result<()> {
    let mut rgb = vec![0u8; rows * cols * 3];
    for (i, &v) in mat.iter().enumerate() {
        let [r, g, b] = colormap(cmap, v);
        rgb[i * 3] = r;
        rgb[i * 3 + 1] = g;
        rgb[i * 3 + 2] = b;
    }
    save_png(path, &rgb, cols, rows)
}

// Anchor stops (normalized RGB). Recognizable approximations of the matplotlib maps.
const VIRIDIS: [[f32; 3]; 9] = [
    [0.267, 0.005, 0.329],
    [0.283, 0.141, 0.458],
    [0.254, 0.265, 0.530],
    [0.207, 0.372, 0.553],
    [0.164, 0.471, 0.558],
    [0.128, 0.567, 0.551],
    [0.135, 0.659, 0.518],
    [0.267, 0.749, 0.441],
    [0.993, 0.906, 0.144],
];

const MAGMA: [[f32; 3]; 6] = [
    [0.001, 0.000, 0.014],
    [0.232, 0.059, 0.438],
    [0.550, 0.161, 0.506],
    [0.868, 0.288, 0.409],
    [0.987, 0.588, 0.398],
    [0.987, 0.991, 0.749],
];

const TURBO: [[f32; 3]; 8] = [
    [0.190, 0.072, 0.232],
    [0.275, 0.520, 0.984],
    [0.149, 0.804, 0.863],
    [0.353, 0.941, 0.471],
    [0.745, 0.941, 0.196],
    [0.980, 0.706, 0.157],
    [0.941, 0.353, 0.118],
    [0.478, 0.016, 0.012],
];

const GRAY: [[f32; 3]; 2] = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]];
