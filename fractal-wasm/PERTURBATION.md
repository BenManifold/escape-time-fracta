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

## Suggested roadmap for `escape-time-fracta`

1. **Tile the image**: each tile has its own \(c_\mathrm{ref}\) (e.g. tile center) and max \(|\delta c|\) within the tile.
2. **Reference orbit**: store `Vec<(Zr, Zi)>` for `n = 0..max_iter` at `c_ref` (may need `num-bigfloat` / `rug` in native tooling first; WASM may use a compact format or streaming).
3. **Perturb pass**: for each pixel, \(\delta_0 = 0\), \(\delta c = c - c_\mathrm{ref}\), loop with the recurrence until escape or `n == ref_len`.
4. **Re-reference** when \(|\delta_n|\) grows too large (Kalles-style “glitch detection”).
5. Optional: **bilinear approximation** (series) for speed inside a tile.

## References

- [mathr — Mandelbrot perturbation (PDF)](https://mathr.co.uk/mandelbrot/perturbation.pdf)  
- [Fractalshades — mathematical background](https://gbillotey.github.io/Fractalshades-doc/math.html)  
- [Mandelbrot Metal — perturbation rendering](https://mandelbrot-metal.com/perturbation-rendering)  
- [How perturbation theory and Taylor series make extreme fractal zooms possible (Medium)](https://medium.com/@michaelstebel/how-perturbation-theory-and-the-taylor-series-make-extreme-fractal-zooms-possible-2dfa9515b8cc)

This crate does **not** implement perturbation yet; the `DEEP_ZOOM_HALF_W_MIN` guard in the web app is a pragmatic stop before f64 **\(c\)** resolution collapses relative to the viewport.
