//! Mandelbrot perturbation: one reference orbit at `c_ref`, per-pixel δ with
//! `δ_{n+1} = 2 Z_n δ_n + δ_n² + δc` (same f64 precision as direct for now — infrastructure for a
//! future high-precision reference).

use crate::palette_f64;
use crate::smooth_iter_f64;

const BAILOUT: f64 = 4.0;

/// Use perturbation when `half_w` is below this (auto mode). Coarse views rarely benefit.
pub const PERTURB_AUTO_HALF_W: f64 = 0.02;

/// Pixel side length of each perturbation tile. Each tile uses its own reference at the tile's
/// complex center so `|δc|` stays smaller than one global reference at the image center.
const PERTURB_TILE_PX: u32 = 160;

/// If `|δ|²` exceeds this factor times `max(1, |Z|²)`, fall back to direct iteration for the pixel.
const GLITCH_REL_EPS: f64 = 1e-8;

/// Build `Z_0 … Z_{max_iter}` at `c_ref` (length `max_iter + 1`). `None` if the reference escapes before completing.
fn mandelbrot_reference_orbit(cr: f64, ci: f64, max_iter: u32) -> Option<(Vec<f64>, Vec<f64>)> {
    let cap = max_iter as usize + 1;
    let mut zr = 0.0_f64;
    let mut zi = 0.0_f64;
    let mut ref_zr = Vec::with_capacity(cap);
    let mut ref_zi = Vec::with_capacity(cap);

    for k in 0..=max_iter {
        ref_zr.push(zr);
        ref_zi.push(zi);
        if k == max_iter {
            break;
        }
        if zr * zr + zi * zi >= BAILOUT {
            return None;
        }
        let nzr = zr * zr - zi * zi + cr;
        let nzi = 2.0 * zr * zi + ci;
        zr = nzr;
        zi = nzi;
    }

    Some((ref_zr, ref_zi))
}

#[inline]
fn mandelbrot_escape_direct(
    re: f64,
    im: f64,
    max_iter: u32,
    palette_id: u32,
) -> (u8, u8, u8, u8) {
    crate::escape_scalar_f64(re, im, max_iter, 0, 0.0, 0.0, palette_id)
}

/// Perturbed escape for Mandelbrot with `c = c_ref + (dcr, dci)`.
fn escape_mandelbrot_perturb_one(
    re: f64,
    im: f64,
    dcr: f64,
    dci: f64,
    max_iter: u32,
    ref_zr: &[f64],
    ref_zi: &[f64],
    palette_id: u32,
) -> (u8, u8, u8, u8) {
    let mut dr = 0.0_f64;
    let mut di = 0.0_f64;

    for n in 0..max_iter as usize {
        let zr = ref_zr[n];
        let zi = ref_zi[n];
        let zpr = zr + dr;
        let zpi = zi + di;
        let r2 = zpr * zpr + zpi * zpi;
        if r2 >= BAILOUT {
            let zmag = r2.sqrt().max(BAILOUT * 1.000_000_1);
            return palette_f64(smooth_iter_f64(n as u32, zmag), palette_id);
        }

        let zmag2 = zr * zr + zi * zi;
        let dmag2 = dr * dr + di * di;
        let thresh = GLITCH_REL_EPS * zmag2.max(1.0);
        if dmag2 > thresh {
            return mandelbrot_escape_direct(re, im, max_iter, palette_id);
        }

        let d2r = dr * dr - di * di;
        let d2i = 2.0 * dr * di;
        let t1r = 2.0 * (zr * dr - zi * di);
        let t1i = 2.0 * (zr * di + zi * dr);
        dr = t1r + d2r + dcr;
        di = t1i + d2i + dci;
    }

    let zr = ref_zr[max_iter as usize];
    let zi = ref_zi[max_iter as usize];
    crate::interior_by_id(zr + dr, zi + di, palette_id)
}

#[inline]
fn complex_at_pixel(
    px: f64,
    py: f64,
    center_x: f64,
    center_y: f64,
    half_w: f64,
    half_h: f64,
    bw_f: f64,
    bh_f: f64,
    two_hw: f64,
    two_hh: f64,
) -> (f64, f64) {
    let re = center_x - half_w + (px / bw_f) * two_hw;
    let im0 = center_y - half_h + (py / bh_f) * two_hh;
    (re, im0)
}

fn write_tile_direct(
    out: &mut [u8],
    bw: usize,
    tx: usize,
    ty: usize,
    x1: usize,
    y1: usize,
    center_x: f64,
    center_y: f64,
    half_w: f64,
    half_h: f64,
    bw_f: f64,
    bh_f: f64,
    two_hw: f64,
    two_hh: f64,
    max_iter: u32,
    palette_id: u32,
) {
    for y in ty..y1 {
        let py = y as f64 + 0.5;
        let im0 = center_y - half_h + (py / bh_f) * two_hh;
        for x in tx..x1 {
            let px = x as f64 + 0.5;
            let re = center_x - half_w + (px / bw_f) * two_hw;
            let (r, g, b, a) = mandelbrot_escape_direct(re, im0, max_iter, palette_id);
            let base = (y * bw + x) * 4;
            out[base] = r;
            out[base + 1] = g;
            out[base + 2] = b;
            out[base + 3] = a;
        }
    }
}

fn write_tile_perturb(
    out: &mut [u8],
    bw: usize,
    tx: usize,
    ty: usize,
    x1: usize,
    y1: usize,
    cref_r: f64,
    cref_i: f64,
    center_x: f64,
    center_y: f64,
    half_w: f64,
    half_h: f64,
    bw_f: f64,
    bh_f: f64,
    two_hw: f64,
    two_hh: f64,
    max_iter: u32,
    ref_zr: &[f64],
    ref_zi: &[f64],
    palette_id: u32,
) {
    for y in ty..y1 {
        let py = y as f64 + 0.5;
        let im0 = center_y - half_h + (py / bh_f) * two_hh;
        for x in tx..x1 {
            let px = x as f64 + 0.5;
            let re = center_x - half_w + (px / bw_f) * two_hw;
            let dcr = re - cref_r;
            let dci = im0 - cref_i;
            let (r, g, b, a) = escape_mandelbrot_perturb_one(
                re,
                im0,
                dcr,
                dci,
                max_iter,
                ref_zr,
                ref_zi,
                palette_id,
            );
            let base = (y * bw + x) * 4;
            out[base] = r;
            out[base + 1] = g;
            out[base + 2] = b;
            out[base + 3] = a;
        }
    }
}

/// Renders the view with tiled Mandelbrot perturbation (reference at each tile's complex center).
/// Falls back to direct iteration for a tile when the reference orbit escapes early.
pub fn render_mandelbrot_perturb(
    out: &mut [u8],
    buf_width: u32,
    buf_height: u32,
    center_x: f64,
    center_y: f64,
    half_w: f64,
    half_h: f64,
    max_iter: u32,
    palette_id: u32,
) {
    let bw = buf_width as usize;
    let bh = buf_height as usize;
    let two_hw = 2.0 * half_w;
    let two_hh = 2.0 * half_h;
    let bw_f = buf_width as f64;
    let bh_f = buf_height as f64;
    let tw = PERTURB_TILE_PX as usize;

    let mut ty = 0usize;
    while ty < bh {
        let y1 = (ty + tw).min(bh);
        let mut tx = 0usize;
        while tx < bw {
            let x1 = (tx + tw).min(bw);
            let last_x = x1 - 1;
            let last_y = y1 - 1;
            let mid_px = (tx + last_x) as f64 * 0.5 + 0.5;
            let mid_py = (ty + last_y) as f64 * 0.5 + 0.5;
            let (cref_r, cref_i) = complex_at_pixel(
                mid_px, mid_py, center_x, center_y, half_w, half_h, bw_f, bh_f, two_hw, two_hh,
            );

            if let Some((ref_zr, ref_zi)) = mandelbrot_reference_orbit(cref_r, cref_i, max_iter) {
                write_tile_perturb(
                    out,
                    bw,
                    tx,
                    ty,
                    x1,
                    y1,
                    cref_r,
                    cref_i,
                    center_x,
                    center_y,
                    half_w,
                    half_h,
                    bw_f,
                    bh_f,
                    two_hw,
                    two_hh,
                    max_iter,
                    &ref_zr,
                    &ref_zi,
                    palette_id,
                );
            } else {
                write_tile_direct(
                    out,
                    bw,
                    tx,
                    ty,
                    x1,
                    y1,
                    center_x,
                    center_y,
                    half_w,
                    half_h,
                    bw_f,
                    bh_f,
                    two_hw,
                    two_hh,
                    max_iter,
                    palette_id,
                );
            }

            tx = x1;
        }
        ty = y1;
    }
}
