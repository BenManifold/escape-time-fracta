# Perturbation rendering (notes for this project)

Direct **f64** iteration (what `render_rgba` does today) is fine until the view is so deep that **ULP spacing in `c` is larger than pixel spacing on the complex plane**. Beyond that, tools like **Kalles Fraktaler**, **Fractalshades**, and **Mandelbrot Metal** switch to **perturbation**: one high-precision **reference orbit** \(Z_n\) at a reference \(c_\mathrm{ref}\), and nearby pixels use a **delta** \(\delta_n = z_n - Z_n\), \(\delta c = c - c_\mathrm{ref}\).

## Mandelbrot / quadratic family

Iterating \(z \mapsto z^2 + c\):

\[
\delta_{n+1} = 2 Z_n \delta_n + \delta_n^2 + \delta c
\]

The reference \(Z_n\) is computed in extended precision (or double-double / arbitrary precision). Neighbor pixels use **f64** (or lower) for \(\delta\) because \(\delta\) stays small relative to \(Z_n\). **Series approximation** goes further: treat \(\delta_n\) as a polynomial in \(\delta c\) and advance coefficients—see Fractalshades / “Perturbator” writeups.

## Burning Ship

\(z \mapsto (|\Re z| + i|\Im z|)^2 + c\) is **not holomorphic** because of `abs`. A practical approach (used in some deep-zoom Ship renderers) is **piecewise linearization**: on each step the Jacobian depends on the quadrant signs of \(\Re z\) and \(\Im z\); perturbation uses the appropriate **real** linearization. Glitches can occur near **sign switches**—mitigations include smaller tiles, re-reference \(c_\mathrm{ref}\), or falling back to full precision for problematic pixels.

## Implemented in this crate (Mandelbrot, f64)

- **`src/perturb.rs`**: reference orbit at **view center** \(c_\mathrm{ref} =\) `(center_x, center_y)`, stored as `Z_0 … Z_{max_iter}` (`max_iter + 1` values). Per pixel: \(\delta_0 = 0\), \(\delta c = c - c_\mathrm{ref}\), then \(\delta_{n+1} = 2 Z_n \delta_n + \delta_n^2 + \delta c\) with the usual escape test on \(Z_n + \delta_n\).
- **Glitch fallback**: if \(|\delta|^2\) is large relative to \(\max(1, |Z|^2)\), that pixel is recomputed with **direct** iteration (`escape_scalar_f64`).
- **UI / Web**: `perturb_mode` — **0** Off, **1** On (Mandelbrot when reference builds), **2** Auto when `half_w < PERTURB_AUTO_HALF_W` (0.02). The live **WebGPU** page calls `render_rgba` with mode **2** whenever `half_w < 0.02`, so Mandelbrot uses this path automatically; other families get **f64 direct** there. If a tile’s reference escapes before `max_iter`, that tile uses direct iteration.
- **Precision**: both reference and \(\delta\) are **f64**; extreme zoom still wants a **high-precision** reference orbit later.

## Roadmap

1. **Tiles**: per-tile \(c_\mathrm{ref}\) so \(|\delta c|\) stays small on wide views.
2. **High-precision `Z_n`** (double-double / arbitrary precision) with f64 \(\delta\).
3. **Burning Ship** piecewise perturbation; **series approximation** for inner tiles.
4. **GPU** — see `GPU.md`.

## References

- [mathr — Mandelbrot perturbation (PDF)](https://mathr.co.uk/mandelbrot/perturbation.pdf)  
- [Fractalshades — mathematical background](https://gbillotey.github.io/Fractalshades-doc/math.html)  
- [Mandelbrot Metal — perturbation rendering](https://mandelbrot-metal.com/perturbation-rendering)  
- [How perturbation theory and Taylor series make extreme fractal zooms possible (Medium)](https://medium.com/@michaelstebel/how-perturbation-theory-and-the-taylor-series-make-extreme-fractal-zooms-possible-2dfa9515b8cc)

The web app still uses `DEEP_ZOOM_HALF_W_MIN` as a pragmatic stop before **f64 \(c\)** spacing collapses relative to pixels; perturbation here does not yet extend precision beyond plain **f64** for the reference orbit.
