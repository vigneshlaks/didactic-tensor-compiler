# Didactic-Tensor-Compiler

Didactic tensor compiler is an educational resource and reference implementation for understanding tensor compiler optimization. While materials exist on backpropagation and computing gradients, fewer resources explain the compilation and optimization processes that make tensor operations efficient on modern hardware. This project bridges that gap by providing a minimal, extensible codebase demonstrating their implementation with supplementary writings for those curious.

## Didactic IR
**Didactic IR & Tooling**

Simple and easy to understand tooling.

**Multi-Level Internal Representation**

The higher level is straight forward and implements.

**Kernel Implementations**

Baseline CUDA and METAL kernels have been implemented.

## Architecture

```
JSON IR → Frontend → Linked List → Optimizer
```

Frontend parses the json into a doubly-linked list. Pass manager runs a number of optimizations and lowering passes over the graph.

## IR Format

Networks are defined in JSON. A program has two top-level fields: `metadata` (pass configuration) and `input` (instruction list).


| `op`            | Fields                          | Description                        |
|-----------------|---------------------------------|------------------------------------|
| `const`         | `dim`, `init`, `trainable`      | Tensor constant / weight           |
| `matmul`        | `args: [lhs, rhs]`              | Matrix multiplication              |
| `relu`          | `args: [input]`                 | ReLU activation                    |
| `softmax`       | `args: [input]`                 | Softmax activation                 |
| `mse_loss`      | `args: [input]`, `dim`          | Mean squared error loss            |
| `cross_entropy` | `args: [input]`, `dim`          | Cross-entropy loss                 |


| Value     | Description                                      |
|-----------|--------------------------------------------------|
| `zeros`   | Zero-initialize (default)                        |
| `xavier`  | Xavier uniform initialization                    |
| `import`  | Load from binary file (requires `path` field)    |

### Example: 2-layer MNIST classifier

```json
{
  "metadata": {
    "passes": [
      {"type": "backend",      "config": {"backend": "cpu"}},
      {"type": "fusion",       "config": {"enabled": true}},
      {"type": "quantization", "config": {"precision": "int8"}}
    ]
  },
  "input": [
    {"id": "input", "op": "const", "dim": [1, 784]},
    {"id": "w1",    "op": "const", "dim": [784, 128], "init": "xavier", "trainable": true},
    {"id": "z1",    "op": "matmul", "args": ["input", "w1"]},
    {"id": "h1",    "op": "relu",   "args": ["z1"]},
    {"id": "w2",    "op": "const", "dim": [128, 10], "init": "xavier", "trainable": true},
    {"id": "logits","op": "matmul", "args": ["h1", "w2"]},
    {"id": "probs", "op": "softmax","args": ["logits"]},
    {"id": "loss",  "op": "cross_entropy", "args": ["probs"], "dim": [1, 10]}
  ]
}
```

## Passes

Passes are declared in the `metadata.passes` array and run in order before execution.

| Pass               | Config key    | Description                                               |
|--------------------|---------------|-----------------------------------------------------------|
| `BackendPass`      | `backend`     | Sets `cpu`, `gpu`, or `metal` on all ops                  |
| `FusionPass`       | `enabled`     | Fuses adjacent `matmul`+`relu` into a single kernel       |
| `QuantizationPass` | `precision`   | Inserts quantize/dequantize nodes (`fp16`, `int8`)        |
| `ShapeInferencePass` | —           | Fills in output shapes for ops that don't declare `dim`   |

## Backends

The build system auto-detects the available backend:

| Backend | Requirement              | Activated by          |
|---------|--------------------------|-----------------------|
| CPU     | none                     | default               |
| CUDA    | NVIDIA GPU + CUDA toolkit| `find_package(CUDA)`  |
| Metal   | Apple Silicon / macOS    | `APPLE` + metal-cpp   |

### Metal shader compilation

Metal kernels live in `src/shaders.metal`. The build system supports two loading strategies depending on what tools are installed:

| Strategy | Requirement | How it works |
|---|---|---|
| Build-time | Full Xcode app | `xcrun metal` compiles `.metal` → `.metallib` during `cmake --build`. Errors surface at build time. |
| Runtime fallback | Xcode Command Line Tools only | `shaders.metal` is read from disk and compiled by the GPU driver on first run. |

CMake detects which strategy to use automatically — no manual configuration needed. Install Xcode from the App Store to get the build-time path.

## Building

**Dependencies**
- CMake ≥ 3.18
- C++20 compiler
- [nlohmann/json](https://github.com/nlohmann/json)
- *(optional)* CUDA toolkit for GPU support
- *(optional)* [metal-cpp](https://developer.apple.com/metal/cpp/) under `metal-cpp/` for Apple Metal
- *(optional)* Xcode (full app) for build-time Metal shader compilation — Command Line Tools alone use the runtime fallback

```bash
cmake -B build
cmake --build build
```

The build will print which backend was detected:
```
Building with Metal support (M1)
```

## Running

```bash
./build/main
```

The binary loads MNIST from `data/MNIST/raw/`, parses `irs/mnist/mnist.json`, runs the pass pipeline, and trains for 3 epochs printing average loss per epoch.

To test a different IR file, modify the `filename` variable in `src/main.cpp`.

## Project Layout

```
didactic-tensor-compiler/
├── include/
│   ├── frontend.h      # IR node / linked list types
│   ├── ops.h           # Op classes
│   ├── types.h         # Tensors
│   ├── passes.h        # Pass / PassManager
│   ├── optimizers.h    # SGD
│   ├── gpu_exec.h      # CUDA dispatch
│   └── metal_exec.h    # Metal dispatch
├── src/
│   ├── frontend.cpp    # JSON → computation graph parser
│   ├── ops.cpp         # Forward / backward implementations
│   ├── types.cpp       # Tensor methods, stride, device transfer
│   ├── passes.cpp      # Pass implementations
│   ├── optimizers.cpp  # Training loop, SGD
│   ├── gpu_exec.cu     # CUDA kernels
│   ├── metal_exec.cpp  # Metal dispatch (metal-cpp)
│   └── shaders.metal   # MSL kernel source (compiled to shaders.metallib if Xcode present)
├── irs/
│   ├── mnist/          # MNIST 2-layer MLP IR + pretrained weights
│   └── two_dimensional/# Toy IR examples
├── data/MNIST/         # MNIST binary dataset
└── CMakeLists.txt
```

## Next Steps
- GPU Autotuning
- Some systems jazz
