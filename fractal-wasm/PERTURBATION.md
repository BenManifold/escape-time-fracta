# Perturbation rendering (notes for this project)

Direct **f64** iteration (what `render_rgba` does in WASM) is fine until the view is so deep that **ULP spacing in `c` is larger than pixel spacing on the complex plane**. Beyond that, tools like **Kalles Fraktaler**, **Fractalshades**, and **Mandelbrot Metal** switch to **perturbation**: one high-precision **reference orbit** \(Z_n\) at a reference \(c_\mathrm{ref}\), and nearby pixels use a **delta** \(\delta_n = z_n - Z_n\), \(\delta c = c - c_\mathrm{ref}\).

## Mandelbrot / quadratic family

Iterating \(z \mapsto z^2 + c\):

\[
\delta_{n+1} = 2 Z_n \delta_n + \delta_n^2 + \delta c
\]

The reference \(Z_n\) is computed in extended precision (or double-double / arbitrary precision). Neighbor pixels use **f64** (or lower) for \(\delta\) because \(\delta\) stays small relative to \(Z_n\). **Series approximation** goes further: treat \(\delta_n\) as a polynomial in \(\delta c\) and advance coefficients—see Fractalshades / “Perturbator” writeups.

## Burning Ship

\(z \mapsto (|\Re z| + i|\Im z|)^2 + c\) is **not holomorphic** because of `abs`. A practical approach (used in some deep-zoom Ship renderers) is **piecewise linearization**: on each step the Jacobian depends on the quadrant signs of \(\Re z\) and \(\Im z\); perturbation uses the appropriate **real** linearization. Glitches can occur near **sign switches**—mitigations include smaller tiles, re-reference \(c_\mathrm{ref}\), or falling back to full precision for problematic pixels.

## Implemented in this crate (current)

- **`perturb.rs`**: tiled **f64** Mandelbrot perturbation (reference orbit per tile, δ recurrence, glitch fallback to direct escape).
- **`render_rgba`**: **Mandelbrot** with `perturb_mode` 1 (on) or 2 (auto when `half_w < PERTURB_AUTO_HALF_W`) uses **`render_mandelbrot_perturb`**; otherwise **`render_all_f64`** (Julia, Burning Ship, coarse Mandelbrot).
- The **interactive app** renders only via **WebGPU** (double-single `c` + orbit); **`render_rgba` / perturb** remain in this crate for tooling or future use. See [GPU.md](GPU.md).

## Roadmap

1. ~~**Tiles** for GPU path~~ (done for WebGPU).
2. ~~**Double-single reference orbit** in WGSL~~ (done); **double-single \(\delta\)** in Pass B for parity with f64 where it matters.
3. **High-precision `Z_n`** for Rust reference (double-double / arbitrary precision).
4. **Burning Ship** piecewise perturbation; **series approximation** for inner tiles.

## References

- [mathr — Mandelbrot perturbation (PDF)](https://mathr.co.uk/mandelbrot/perturbation.pdf)
- [Fractalshades — mathematical background](https://gbillotey.github.io/Fractalshades-doc/math.html)
- [Mandelbrot Metal — perturbation rendering](https://mandelbrot-metal.com/perturbation-rendering)
- [How perturbation theory and Taylor series make extreme fractal zooms possible (Medium)](https://medium.com/@michaelstebel/how-perturbation-theory-and-the-taylor-series-make-extreme-fractal-zooms-possible-2dfa9515b8cc)
