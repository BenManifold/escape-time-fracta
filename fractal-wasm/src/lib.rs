//! Escape-time fractal RGBA renderer (WASM SIMD128).

use wasm_bindgen::prelude::*;

const BAILOUT: f32 = 4.0;

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

    #[cfg(target_arch = "wasm32")]
    unsafe {
        render_all_simd(
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

    #[cfg(not(target_arch = "wasm32"))]
    render_all_scalar(
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

#[cfg(not(target_arch = "wasm32"))]
fn render_all_scalar(
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
    let two_hw = (2.0 * half_w) as f32;
    let two_hh = (2.0 * half_h) as f32;
    let cx = center_x as f32;
    let cy = center_y as f32;
    let hw = half_w as f32;
    let hh = half_h as f32;
    let jre = julia_re as f32;
    let jim = julia_im as f32;
    let bw_f = buf_width as f32;
    let bh_f = buf_height as f32;

    for y in 0..bh {
        let py = y as f32 + 0.5;
        let im0 = cy - hh + (py / bh_f) * two_hh;
        for x in 0..bw {
            let px = x as f32 + 0.5;
            let re = cx - hw + (px / bw_f) * two_hw;
            let (r, g, b, a) = escape_scalar(re, im0, max_iter, fractal_kind, jre, jim);
            let base = (y * bw + x) * 4;
            out[base] = r;
            out[base + 1] = g;
            out[base + 2] = b;
            out[base + 3] = a;
        }
    }
}

#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn render_all_simd(
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
    use std::arch::wasm32::*;

    let bw = buf_width as usize;
    let bh = buf_height as usize;
    let two_hw = (2.0 * half_w) as f32;
    let two_hh = (2.0 * half_h) as f32;
    let cx = center_x as f32;
    let cy = center_y as f32;
    let hw = half_w as f32;
    let hh = half_h as f32;
    let jre = julia_re as f32;
    let jim = julia_im as f32;
    let bw_f = buf_width as f32;
    let bh_f = buf_height as f32;

    for y in 0..bh {
        let py = y as f32 + 0.5;
        let im0 = cy - hh + (py / bh_f) * two_hh;

        let mut x = 0usize;
        while x + 4 <= bw {
            let px0 = x as f32 + 0.5;
            let re = f32x4(
                cx - hw + (px0 / bw_f) * two_hw,
                cx - hw + ((px0 + 1.0) / bw_f) * two_hw,
                cx - hw + ((px0 + 2.0) / bw_f) * two_hw,
                cx - hw + ((px0 + 3.0) / bw_f) * two_hw,
            );

            let (zr, zi, cr, ci) = if fractal_kind == 1 {
                (
                    re,
                    f32x4_splat(im0),
                    f32x4_splat(jre),
                    f32x4_splat(jim),
                )
            } else {
                (
                    f32x4_splat(0.0),
                    f32x4_splat(0.0),
                    re,
                    f32x4_splat(im0),
                )
            };

            let colors = escape_block_colors_simd(zr, zi, cr, ci, max_iter, fractal_kind);
            let base = (y * bw + x) * 4;
            out[base..base + 16].copy_from_slice(&colors);
            x += 4;
        }

        while x < bw {
            let px = x as f32 + 0.5;
            let re = cx - hw + (px / bw_f) * two_hw;
            let (r, g, b, a) = escape_scalar(re, im0, max_iter, fractal_kind, jre, jim);
            let base = (y * bw + x) * 4;
            out[base] = r;
            out[base + 1] = g;
            out[base + 2] = b;
            out[base + 3] = a;
            x += 1;
        }
    }
}

#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
unsafe fn escape_block_colors_simd(
    mut zr: std::arch::wasm32::v128,
    mut zi: std::arch::wasm32::v128,
    cr: std::arch::wasm32::v128,
    ci: std::arch::wasm32::v128,
    max_iter: u32,
    fractal_kind: u32,
) -> [u8; 16] {
    use std::arch::wasm32::*;

    let mut it = [-1i32; 4];
    let mut r2_esc = [0.0f32; 4];

    for n in 0..max_iter {
        let zr2 = f32x4_mul(zr, zr);
        let zi2 = f32x4_mul(zi, zi);
        let r2 = f32x4_add(zr2, zi2);

        let mut all_done = true;
        for lane in 0..4 {
            if it[lane] >= 0 {
                continue;
            }
            let r2l = extract_r2_lane(r2, lane);
            if r2l >= BAILOUT {
                it[lane] = n as i32;
                r2_esc[lane] = r2l;
            } else {
                all_done = false;
            }
        }

        if all_done {
            break;
        }

        let (zr_n, zi_n) = match fractal_kind {
            2 => {
                let azr = f32x4_pmax(zr, f32x4_neg(zr));
                let azi = f32x4_pmax(zi, f32x4_neg(zi));
                step_mandel_like(azr, azi, cr, ci)
            }
            3 => {
                let zrc = f32x4_neg(zi);
                step_mandel_like(zr, zrc, cr, ci)
            }
            _ => step_mandel_like(zr, zi, cr, ci),
        };
        zr = zr_n;
        zi = zi_n;
    }

    let mut out = [0u8; 16];
    for lane in 0..4 {
        let (r, g, b, a) = if it[lane] < 0 {
            (0u8, 0u8, 0u8, 255u8)
        } else {
            let n = it[lane] as u32;
            let r2 = r2_esc[lane].max(BAILOUT * 1.000_001);
            let zmag = r2.sqrt();
            let smooth = smooth_iter(n, zmag);
            palette(smooth)
        };
        let o = lane * 4;
        out[o] = r;
        out[o + 1] = g;
        out[o + 2] = b;
        out[o + 3] = a;
    }
    out
}

#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
#[inline]
unsafe fn extract_r2_lane(r2: std::arch::wasm32::v128, lane: usize) -> f32 {
    use std::arch::wasm32::*;
    match lane {
        0 => f32x4_extract_lane::<0>(r2),
        1 => f32x4_extract_lane::<1>(r2),
        2 => f32x4_extract_lane::<2>(r2),
        _ => f32x4_extract_lane::<3>(r2),
    }
}

#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
#[inline]
unsafe fn step_mandel_like(
    zr: std::arch::wasm32::v128,
    zi: std::arch::wasm32::v128,
    cr: std::arch::wasm32::v128,
    ci: std::arch::wasm32::v128,
) -> (std::arch::wasm32::v128, std::arch::wasm32::v128) {
    use std::arch::wasm32::*;
    let zr2 = f32x4_mul(zr, zr);
    let zi2 = f32x4_mul(zi, zi);
    let zri = f32x4_mul(zr, zi);
    let new_zr = f32x4_add(f32x4_sub(zr2, zi2), cr);
    let new_zi = f32x4_add(f32x4_add(zri, zri), ci);
    (new_zr, new_zi)
}

fn escape_scalar(
    re: f32,
    im: f32,
    max_iter: u32,
    fractal_kind: u32,
    jre: f32,
    jim: f32,
) -> (u8, u8, u8, u8) {
    let (mut zr, mut zi, cr, ci) = if fractal_kind == 1 {
        (re, im, jre, jim)
    } else {
        (0.0, 0.0, re, im)
    };

    for n in 0..max_iter {
        let r2 = zr * zr + zi * zi;
        if r2 >= BAILOUT {
            let zmag = r2.sqrt().max(BAILOUT * 1.000_001);
            let smooth = smooth_iter(n, zmag);
            return palette(smooth);
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
fn smooth_iter(n: u32, zmag: f32) -> f32 {
    n as f32 + 1.0 - zmag.log2().log2()
}

/// Cosine-based palette (smooth, saturated exterior).
#[inline]
fn palette(t: f32) -> (u8, u8, u8, u8) {
    let t = t * 0.15 + 0.1;
    let c = |off: f32| -> u8 {
        let v = 0.5 + 0.5 * (t * 6.283_185_5 + off).cos();
        (v.clamp(0.0, 1.0) * 255.0) as u8
    };
    (c(0.0), c(2.094), c(4.189), 255)
}
