# Neural-Network-Compiler

Neural Network Compiler is an educational resource and reference implementation for understanding tensor compiler optimization. While materials exist on backpropagation and computing gradients, fewer resources explain the compilation and optimization processes that make tensor operations efficient on modern hardware. This project bridges that gap by providing a minimal, extensible codebase demonstrating their implementation.

## Architecture

```
JSON IR → Frontend → Graph → Pass Manager → Backend Execution
```

The frontend parses JSON into a computation graph (doubly-linked list of nodes). The pass manager then runs optimization and lowering passes over the graph before dispatching to the selected backend (CPU, CUDA, or Metal).

## IR Format

Networks are defined in JSON. A program has two top-level fields: `metadata` (pass configuration) and `input` (instruction list).

### Ops

| `op`            | Fields                          | Description                        |
|-----------------|---------------------------------|------------------------------------|
| `const`         | `dim`, `init`, `trainable`      | Tensor constant / weight           |
| `matmul`        | `args: [lhs, rhs]`              | Matrix multiplication              |
| `relu`          | `args: [input]`                 | ReLU activation                    |
| `softmax`       | `args: [input]`                 | Softmax activation                 |
| `mse_loss`      | `args: [input]`, `dim`          | Mean squared error loss            |
| `cross_entropy` | `args: [input]`, `dim`          | Cross-entropy loss                 |

### Weight initialization

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
neural-network-compiler/
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

The following are natural extensions, roughly ordered by difficulty.

**New ops**
- `conv2d` — the step from MNIST to real vision models; requires im2col or a direct kernel
- `layer_norm` / `attention` — gets the IR to transformer-level networks
- `dropout` — training-only op that requires a forward/backward mode distinction in the graph

**New passes**
- `ConstantFoldingPass` — evaluate const→const subgraphs at compile time
- `DeadCodeEliminationPass` — prune nodes whose outputs are never consumed
- `MemoryPlanningPass` — assign shared buffer slots to activations so allocations are not per-node

**Backend improvements**
- Tiled CUDA matmul with shared memory — the current unoptimized kernel is one thread per output element
- Call vendor routines when available and fall back to the hand-written kernel otherwise
- Fuse more patterns

**Training**
- Adam optimizer (`optimizers.h` / `optimizers.cpp`) — SGD with momentum and adaptive learning rates
- Gradient clipping — prevents exploding gradients without changing the optimizer interface
- Symbolic autograd — derive gradients from the graph structure instead of hardcoding them per op

**IR / infrastructure**
- Dynamic shapes — re-run `ShapeInferencePass` when input dimensions change rather than requiring static dims in the JSON
- Multi-output nodes — the current `Node` holds a single output tensor; some ops produce multiple
