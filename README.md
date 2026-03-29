# escape-time-fracta

A fractal renderer using **WebAssembly** and JavaScript. The Rust core maps pixels and runs the escape-time loop in **64-bit float** so deep zoom keeps correct **c** spacing (the previous **f32** path loses detail when `halfW` is very small). GPU acceleration is not used.

## Prerequisites (development)

1. **Rust** — Install with [rustup](https://rustup.rs/) (includes `cargo` and `rustc`). On Windows, the default `x86_64-pc-windows-msvc` toolchain needs the **Visual Studio C++ build tools** (or full VS with the “Desktop development with C++” workload) so the linker is available.

2. **WebAssembly target** — After installing Rust:

   ```bash
   rustup target add wasm32-unknown-unknown
   ```

3. **wasm-pack** — Builds the crate and generates JS glue:

   ```bash
   cargo install wasm-pack
   ```

4. **PATH** — If `cargo` or `wasm-pack` are not found in a new terminal, ensure `%USERPROFILE%\.cargo\bin` is on your PATH (rustup adds this for new shells; restart Cursor/VS Code or open a new terminal after install).

## Build the WASM package

From the `fractal-wasm` directory:

```bash
wasm-pack build --target web --release
```

From the **repository root** (positional crate path — same effect as `cd fractal-wasm`):

```bash
wasm-pack build fractal-wasm --target web --release
```

**Do not** pass Cargo’s `--manifest-path` to `wasm-pack build`. `wasm-pack` does not treat it as a first-class flag; it ends up in extra options forwarded to `cargo`. The build logic also scans those options for `--target` and treats the next token as a **Rust target triple**. If `--target web` (wasm-pack’s *output* mode for ES modules) lands in that list, it is misread as the triple `web`, which leads to `rustup target add web` and the error *toolchain does not support target 'web'*.

The renderer does not require WASM SIMD or extra `RUSTFLAGS`.

Output is written to `fractal-wasm/pkg/`.

## Serve the static site

Browsers require correct MIME types for `.wasm` files. Use any static server from the repo root (where `index.html` lives), for example:

```bash
npx --yes serve .
```

or Python:

```bash
python -m http.server 8080
```

Then open the URL shown in the terminal.

Do not open `index.html` directly from the file system (`file://`): ES module imports and WASM fetch require a local HTTP server.

## Browser note

Use a current browser with WebAssembly support. If the module fails to load, check the console for fetch/MIME or instantiation errors.
