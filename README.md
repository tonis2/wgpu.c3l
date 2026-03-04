# wgpu.c3l

WebGPU bindings for the [C3 programming language](https://c3-lang.org), generated from the official [webgpu-native](https://github.com/webgpu-native/webgpu-headers) JSON spec.

The bindings wrap [wgpu-native](https://github.com/gfx-rs/wgpu-native) and work on Linux, macOS, and Windows.


## Building the bindings

The bindings in `lib/` are already generated and committed. You only need to rebuild them if you want to update to a newer spec version.

```bash
./build.sh
```

This downloads the latest `webgpu.json` spec and regenerates `lib/webgpu.c3` and `lib/commands.c3`.

## Running the cube example

### 1. Install wgpu-native

Download the native library for your platform:

```bash
./install.sh
```

This detects your OS and architecture and places the library in `./libs/wgpu-native/`.

### 2. Build and run

```bash
c3c run cube
```

Press `Q` or `Escape` to exit.

## Using the bindings in your own project

Add this library as a dependency in your `project.json`:

```json
"dependencies": ["wgpu"],
"dependency-search-paths": ["path/to/webgpu.c3l/.."]
```

Then import it:

```c3
import wgpu;
```
