//! Escape-time fractal RGBA renderer (WASM).
//! Pixel mapping and iteration use **f64** so deep zoom keeps sub-pixel c resolution.

mod perturb;

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

/// Fills `out` with RGBA8 samples of [`palette_f64`] along smooth-iteration axis: `t` runs linearly
/// from `0` to `t_max` over `n_entries` rows. Used by the WebGPU path to match CPU exterior colors.
/// Requires `out_len >= n_entries * 4` and `n_entries >= 2`.
#[wasm_bindgen]
pub fn fill_smooth_palette_lut(
    out_ptr: *mut u8,
    out_len: usize,
    n_entries: u32,
    palette_id: u32,
    t_max: f64,
) {
    let n = n_entries as usize;
    let need = n.saturating_mul(4);
    if out_len < need || n < 2 {
        return;
    }
    let out = unsafe { core::slice::from_raw_parts_mut(out_ptr, need) };
    let palette_id = palette_id.min(3);
    let denom = (n - 1) as f64;
    for i in 0..n {
        let t = (i as f64 / denom) * t_max;
        let (r, g, b, a) = palette_f64(t, palette_id);
        let base = i * 4;
        out[base] = r;
        out[base + 1] = g;
        out[base + 2] = b;
        out[base + 3] = a;
    }
}

/// `fractal_kind`: 0 Mandelbrot, 1 Julia, 2 Burning Ship, 3 Tricorn.
/// `palette_id`: 0 Nebula, 1 Flotilla (burning ships / sea), 2 Classic cosine, 3 Grayscale.
/// `perturb_mode`: 0 off, 1 on (Mandelbrot only), 2 auto (`half_w` &lt; threshold).
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
    palette_id: u32,
    perturb_mode: u32,
) {
    let need = (buf_width as usize)
        .saturating_mul(buf_height as usize)
        .saturating_mul(4);
    if out_len < need || buf_width == 0 || buf_height == 0 {
        return;
    }
    let out = unsafe { core::slice::from_raw_parts_mut(out_ptr, need) };
    let half_h = half_w * aspect_h_over_w;
    let palette_id = palette_id.min(3);
    let perturb_mode = perturb_mode.min(2);

    let use_perturb = fractal_kind == 0
        && (perturb_mode == 1
            || (perturb_mode == 2 && half_w < perturb::PERTURB_AUTO_HALF_W));

    if use_perturb {
        perturb::render_mandelbrot_perturb(
            out,
            buf_width,
            buf_height,
            center_x,
            center_y,
            half_w,
            half_h,
            max_iter,
            palette_id,
        );
        return;
    }

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
        palette_id,
    );
}

/// Escape iteration at `re`+`im` (same dynamics as `render_rgba`).
/// Returns `n` in `0..max_iter` when the orbit escaped on step `n`, or `max_iter` if still bounded.
#[wasm_bindgen]
pub fn probe_escape_iter(
    re: f64,
    im: f64,
    max_iter: u32,
    fractal_kind: u32,
    julia_re: f64,
    julia_im: f64,
) -> u32 {
    let (mut zr, mut zi, cr, ci) = if fractal_kind == 1 {
        (re, im, julia_re, julia_im)
    } else {
        (0.0, 0.0, re, im)
    };

    for n in 0..max_iter {
        let r2 = zr * zr + zi * zi;
        if r2 >= BAILOUT {
            return n;
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

    max_iter
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
    palette_id: u32,
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

pub(crate) fn escape_scalar_f64(
    re: f64,
    im: f64,
    max_iter: u32,
    fractal_kind: u32,
    jre: f64,
    jim: f64,
    palette_id: u32,
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
            return palette_f64(smooth, palette_id);
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

    interior_by_id(zr, zi, palette_id)
}

#[inline]
pub(crate) fn smooth_iter_f64(n: u32, zmag: f64) -> f64 {
    n as f64 + 1.0 - zmag.log2().log2()
}

/// Page background `#0c0c0f` (matches canvas); mixing toward it softens bands like a translucent veil.
const BG_R: f64 = 12.0 / 255.0;
const BG_G: f64 = 12.0 / 255.0;
const BG_B: f64 = 15.0 / 255.0;

#[inline]
fn interior_nebula_rgba(zr: f64, zi: f64) -> (u8, u8, u8, u8) {
    let ang = zi.atan2(zr);
    let w = 0.5 + 0.5 * ang.sin();
    let r2 = zr * zr + zi * zi;
    let rho = r2.sqrt().clamp(0.0, 2.0) * 0.5;

    let mut r = 0.052 + 0.035 * w + 0.018 * rho;
    let mut g = 0.038 + 0.022 * (1.0 - w) + 0.014 * rho;
    let mut b = 0.11 + 0.05 * w + 0.028 * rho;

    const VEIL_IN: f64 = 0.35;
    r = r * (1.0 - VEIL_IN) + BG_R * VEIL_IN;
    g = g * (1.0 - VEIL_IN) + BG_G * VEIL_IN;
    b = b * (1.0 - VEIL_IN) + BG_B * VEIL_IN;

    rgba_from_f64(r, g, b)
}

/// Ghost-Ship–style interior: deep teal water with a little reflected gold (not flat black).
#[inline]
fn interior_flotilla_rgba(zr: f64, zi: f64) -> (u8, u8, u8, u8) {
    let ang = zi.atan2(zr);
    let w = 0.5 + 0.5 * ang.sin();
    let r2 = zr * zr + zi * zi;
    let rho = r2.sqrt().clamp(0.0, 2.0) * 0.5;

    let mut r = 0.045 + 0.04 * w + 0.04 * rho;
    let mut g = 0.075 + 0.055 * (1.0 - w) + 0.05 * rho;
    let mut b = 0.11 + 0.07 * w + 0.055 * rho;
    r += 0.025 * rho * (ang * 2.1).cos().max(0.0);
    g += 0.02 * rho * (ang * 2.1).sin().max(0.0);

    const VEIL_IN: f64 = 0.26;
    r = r * (1.0 - VEIL_IN) + BG_R * VEIL_IN;
    g = g * (1.0 - VEIL_IN) + BG_G * VEIL_IN;
    b = b * (1.0 - VEIL_IN) + BG_B * VEIL_IN;

    rgba_from_f64(r, g, b)
}

#[inline]
fn interior_classic_rgba(zr: f64, zi: f64) -> (u8, u8, u8, u8) {
    let _ = (zr, zi);
    (0, 0, 0, 255)
}

#[inline]
fn interior_gray_rgba(zr: f64, zi: f64) -> (u8, u8, u8, u8) {
    let ang = zi.atan2(zr);
    let v = 0.04 + 0.06 * (0.5 + 0.5 * ang.sin());
    let x = v * (1.0 - VEIL_GRAY) + 0.05 * VEIL_GRAY;
    rgba_from_f64(x, x, x)
}

const VEIL_GRAY: f64 = 0.35;

#[inline]
fn rgba_from_f64(r: f64, g: f64, b: f64) -> (u8, u8, u8, u8) {
    (
        (r.clamp(0.0, 1.0) * 255.0) as u8,
        (g.clamp(0.0, 1.0) * 255.0) as u8,
        (b.clamp(0.0, 1.0) * 255.0) as u8,
        255,
    )
}

#[inline]
pub(crate) fn interior_by_id(zr: f64, zi: f64, palette_id: u32) -> (u8, u8, u8, u8) {
    match palette_id {
        1 => interior_flotilla_rgba(zr, zi),
        2 => interior_classic_rgba(zr, zi),
        3 => interior_gray_rgba(zr, zi),
        _ => interior_nebula_rgba(zr, zi),
    }
}

/// Triangular blend in smooth-iteration space (reduces band shearing).
#[inline]
fn blend_smoothed_palette(t: f64, inner: impl Fn(f64) -> (f64, f64, f64)) -> (f64, f64, f64) {
    const DT: f64 = 0.55;
    const W0: f64 = 0.22;
    const W1: f64 = 0.56;
    const W2: f64 = 0.22;
    let a = inner(t - DT);
    let b = inner(t);
    let c = inner(t + DT);
    (
        W0 * a.0 + W1 * b.0 + W2 * c.0,
        W0 * a.1 + W1 * b.1 + W2 * c.1,
        W0 * a.2 + W1 * b.2 + W2 * c.2,
    )
}

#[inline]
fn palette_channels_nebula(t: f64) -> (f64, f64, f64) {
    let u = t * 0.09 + 0.14;
    let r = 0.34 + 0.13 * (u + 0.9).sin();
    let g = 0.40 + 0.16 * (u + 2.45).sin();
    let b = 0.48 + 0.19 * (u + 4.0).sin();
    const LR: f64 = 0.42;
    const LG: f64 = 0.38;
    const LB: f64 = 0.52;
    const MIST: f64 = 0.24;
    let r = r * (1.0 - MIST) + LR * MIST;
    let g = g * (1.0 - MIST) + LG * MIST;
    let b = b * (1.0 - MIST) + LB * MIST;
    (r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0))
}

/// Ghost Ship–inspired: brighter teal/violet sky & water, strong **gold–yellow** hulls, **lime–seafoam** accents.
#[inline]
fn palette_channels_flotilla(t: f64) -> (f64, f64, f64) {
    let u = t * 0.084 + 0.12;
    // Lifted base — readable night sea / airglow (not near-black)
    let mut r = 0.055 + 0.09 * (u + 0.88).sin();
    let mut g = 0.075 + 0.1 * (u + 2.02).sin();
    let mut b = 0.12 + 0.11 * (u + 0.32).sin();

    let v = t * 0.26 + 0.5;
    r += 0.028 * (v + 0.08).sin() * (u * 1.95 + 0.8).sin();
    g += 0.034 * (v + 1.25).sin() * (u * 2.05 + 0.45).cos();
    b += 0.038 * (v + 2.4).cos() * (u * 1.75 + 2.0).sin();

    // Distant warm haze (reflection / lantern spill on water)
    let glow_far = (1.0 - (t / 24.0).min(1.0)).powf(1.15);
    r += 0.055 * glow_far;
    g += 0.048 * glow_far;
    b += 0.022 * glow_far;

    // Structure band: gold + yellow (high R,G) and green waves (high G)
    let rig = (t / 28.0).clamp(0.0, 1.0);
    let rig = rig * rig * (3.0 - 2.0 * rig);
    let w1 = 0.5 + 0.5 * (u * 1.06 + 2.5).sin();
    let w2 = 0.5 + 0.5 * (u * 0.9 + 3.85).cos();
    let w3 = 0.5 + 0.5 * (u * 1.4 + 5.1).sin();

    let fire = rig * (0.32 + 0.68 * w1);
    // Gold / straw (Ghost Ship hulls)
    let gold = fire * (0.45 + 0.55 * w2);
    r += gold * (0.95 + 0.2 * w3);
    g += gold * (0.82 + 0.22 * w3);
    b += gold * 0.28;

    // Yellow highlight (push toward sunflower without clipping)
    r += fire * 0.22 * w2 * w2;
    g += fire * 0.2 * w2 * w2;
    b += fire * 0.06 * w2;

    // Seafoam / chartreuse rigging
    let green_lick = fire * (0.35 + 0.65 * (1.0 - w2)) * (0.5 + 0.5 * w3);
    g += green_lick * 0.62;
    r += green_lick * 0.22;
    b += green_lick * 0.28;

    // Mist: greenish moonlit haze (matches reference water tone)
    const LR: f64 = 0.09;
    const LG: f64 = 0.14;
    const LB: f64 = 0.2;
    const MIST: f64 = 0.13;
    let r = r * (1.0 - MIST) + LR * MIST;
    let g = g * (1.0 - MIST) + LG * MIST;
    let b = b * (1.0 - MIST) + LB * MIST;
    (r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0))
}

#[inline]
fn palette_channels_gray(t: f64) -> (f64, f64, f64) {
    let u = t * 0.1 + 0.16;
    let v = (0.06 + 0.88 * (0.5 + 0.5 * (u + 1.05).sin())).clamp(0.0, 1.0);
    (v, v, v)
}

#[inline]
fn palette_classic_cosine(t: f64) -> (u8, u8, u8, u8) {
    let u = t * 0.15 + 0.1;
    let c = |off: f64| -> f64 { (0.5 + 0.5 * (u * std::f64::consts::TAU + off).cos()).clamp(0.0, 1.0) };
    let r = c(0.0);
    let g = c(2.0943951023931953);
    let b = c(4.1887902047863909);
    ((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8, 255)
}

#[inline]
pub(crate) fn palette_f64(t: f64, palette_id: u32) -> (u8, u8, u8, u8) {
    if palette_id == 2 {
        return palette_classic_cosine(t);
    }

    let (r, g, b) = match palette_id {
        1 => blend_smoothed_palette(t, palette_channels_flotilla),
        3 => blend_smoothed_palette(t, palette_channels_gray),
        _ => blend_smoothed_palette(t, palette_channels_nebula),
    };

    // Flotilla: lighter veil so gold/greens aren’t dragged to page black
    let veil = if palette_id == 1 { 0.07 } else { 0.20 };
    let r = r * (1.0 - veil) + BG_R * veil;
    let g = g * (1.0 - veil) + BG_G * veil;
    let b = b * (1.0 - veil) + BG_B * veil;
    rgba_from_f64(r, g, b)
}
