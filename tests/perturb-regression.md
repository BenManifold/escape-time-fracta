# Manual regression: WebGPU double-single escape

1. Load the app over HTTP; pick **Mandelbrot**, **Julia**, and **Burning Ship** in turn.
2. Zoom deeply (box-zoom and keyboard). The image should stay coherent without the old “f32 block” look from a pure single-precision orbit; very extreme depths may still differ from offline f64+perturb tools.
3. **Julia**: move Re/Im sliders; the view should update smoothly in real time.
