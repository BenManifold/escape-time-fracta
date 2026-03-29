//! Escape-time fractal RGBA renderer (WASM).
//! Pixel mapping and iteration use **f64** so deep zoom keeps sub-pixel c resolution.

use wasm_bindgen::prelude::*;

const BAILOUT: f64 = 4.0;

#[wasm_bindgen]
pub fn alloc(len: usize) -> *mut u8 {
    let mut v = vec![0u8; len];
    let p = v.as_mut_ptr();
    core::mem::forget(v);
    p
}

#[wasm_bindgen]
pub fn dealloc(ptr: *mut u8, len: usize) {
    unsafe {
        drop(Vec::from_raw_parts(ptr, len, len));
    }
}

/// `fractal_kind`: 0 Mandelbrot, 1 Julia, 2 Burning Ship, 3 Tricorn.
#[wasm_bindgen]
pub fn render_rgba(
    out_ptr: *mut u8,
    out_len: usize,
    buf_width: u32,
    buf_height: u32,
    center_x: f64,
    center_y: f64,
    half_w: f64,
    aspect_h_over_w: f64,
    max_iter: u32,
    fractal_kind: u32,
    julia_re: f64,
    julia_im: f64,
) {
    let need = (buf_width as usize)
        .saturating_mul(buf_height as usize)
        .saturating_mul(4);
    if out_len < need || buf_width == 0 || buf_height == 0 {
        return;
    }
    let out = unsafe { core::slice::from_raw_parts_mut(out_ptr, need) };
    let half_h = half_w * aspect_h_over_w;

    render_all_f64(
        out,
        buf_width,
        buf_height,
        center_x,
        center_y,
        half_w,
        half_h,
        max_iter,
        fractal_kind,
        julia_re,
        julia_im,
    );
}

fn render_all_f64(
    out: &mut [u8],
    buf_width: u32,
    buf_height: u32,
    center_x: f64,
    center_y: f64,
    half_w: f64,
    half_h: f64,
    max_iter: u32,
    fractal_kind: u32,
    julia_re: f64,
    julia_im: f64,
) {
    let bw = buf_width as usize;
    let bh = buf_height as usize;
    let two_hw = 2.0 * half_w;
    let two_hh = 2.0 * half_h;
    let bw_f = buf_width as f64;
    let bh_f = buf_height as f64;

    for y in 0..bh {
        let py = y as f64 + 0.5;
        let im0 = center_y - half_h + (py / bh_f) * two_hh;
        for x in 0..bw {
            let px = x as f64 + 0.5;
            let re = center_x - half_w + (px / bw_f) * two_hw;
            let (r, g, b, a) = escape_scalar_f64(
                re,
                im0,
                max_iter,
                fractal_kind,
                julia_re,
                julia_im,
            );
            let base = (y * bw + x) * 4;
            out[base] = r;
            out[base + 1] = g;
            out[base + 2] = b;
            out[base + 3] = a;
        }
    }
}

fn escape_scalar_f64(
    re: f64,
    im: f64,
    max_iter: u32,
    fractal_kind: u32,
    jre: f64,
    jim: f64,
) -> (u8, u8, u8, u8) {
    let (mut zr, mut zi, cr, ci) = if fractal_kind == 1 {
        (re, im, jre, jim)
    } else {
        (0.0, 0.0, re, im)
    };

    for n in 0..max_iter {
        let r2 = zr * zr + zi * zi;
        if r2 >= BAILOUT {
            let zmag = r2.sqrt().max(BAILOUT * 1.000_000_1);
            let smooth = smooth_iter_f64(n, zmag);
            return palette_f64(smooth);
        }

        let (zr_n, zi_n) = match fractal_kind {
            2 => {
                let azr = zr.abs();
                let azi = zi.abs();
                (azr * azr - azi * azi + cr, 2.0 * azr * azi + ci)
            }
            3 => {
                let zrc = -zi;
                (zr * zr - zrc * zrc + cr, 2.0 * zr * zrc + ci)
            }
            _ => (zr * zr - zi * zi + cr, 2.0 * zr * zi + ci),
        };
        zr = zr_n;
        zi = zi_n;
    }

    (0, 0, 0, 255)
}

#[inline]
fn smooth_iter_f64(n: u32, zmag: f64) -> f64 {
    n as f64 + 1.0 - zmag.log2().log2()
}

/// Cosine-based palette (smooth, saturated exterior).
#[inline]
fn palette_f64(t: f64) -> (u8, u8, u8, u8) {
    let t = t * 0.15 + 0.1;
    let c = |off: f64| -> u8 {
        let v = 0.5 + 0.5 * (t * 6.283_185_307_179_586 + off).cos();
        (v.clamp(0.0, 1.0) * 255.0) as u8
    };
    (c(0.0), c(2.094_395_102_393_195_3), c(4.188_790_204_786_390_5), 255)
}
