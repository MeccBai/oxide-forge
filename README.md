# OxideForge

English | [简体中文](README.cn.md)

OxideForge is a purpose-built Rust/CUDA neural-network runtime powered by
[CUDA-Oxide](https://nvlabs.github.io/cuda-oxide/index.html). It combines GPU
compute primitives, contiguous-memory containers, and neural-network execution
layers into a lightweight foundation for models with known shapes and controlled
data layouts.

OxideForge is not intended to reproduce a general-purpose tensor framework. It
uses handwritten CUDA kernels and explicit ownership to keep data flow
predictable, runtime overhead low, and the programming model close to the
hardware.

> Maximize zero-overhead abstraction, not abstraction itself.

## Status

OxideForge is experimental and its API is still evolving. The core forward and
backward paths are implemented, together with a versioned TOML + binary model
checkpoint format. Dataset integration and end-to-end training orchestration
remain under development.

Implemented capabilities include:

- CUDA context, module, stream, buffer allocation, and synchronization;
- immutable and mutable contiguous device spans, plus borrowed vector views;
- device-owning `Vector` and row-major `Matrix` containers;
- element-wise arithmetic, mapping, scaling, reduction, and row broadcasting;
- tiled matrix multiplication and shared-memory matrix transpose;
- row-wise Softmax and LayerNorm, including backward kernels;
- Linear, GELU, MLP, residual connections, and their backward paths;
- inference and training executors for a single-head Post-LN Transformer;
- parameter checkpoint save/load for MLP and Transformer executors;
- asynchronous submission on the primary stream and explicit fork/join for
  additional streams.

The remaining integration work is primarily:

- input and label pipelines;
- model-level forward/backward orchestration and training loops;
- small-shape numerical gradient tests and optimizer state persistence.

## Design Principles

### Contiguous memory first

`Matrix` always uses a contiguous row-major layout. Spans represent contiguous
device-memory regions only and deliberately do not support strides. Column-wise
access and disconnected regions must be handled through an explicit transpose
or physical rearrangement. This prevents the cost of irregular layouts from
propagating into every downstream kernel.

### Ownership defines data lifetime

`Matrix` and `Vector` own device memory; spans and views borrow it. Operations
that allocate a new container live on `CudaRuntime`, while in-place operations
live on the container itself. Training executors retain only the values needed
by backward. A layer's newly allocated output is moved directly into the next
layer's cache without an additional device copy. Final outputs are returned by
value so the parent model controls whether they remain alive.

### Synchronization is explicit

Dependent operations are queued on the same CUDA stream without synchronizing
after every kernel launch. Host-valued reductions and model boundaries form the
main synchronization points. Additional streams are reserved for genuinely
independent work and are explicitly rejoined.

### Specialized implementations over illusory generality

The runtime currently uses `f32` and targets known model shapes. Generality is
added only when it does not impose significant complexity or performance cost.
Optimization follows profiler evidence instead of speculative abstraction.

## Execution Model

The current Transformer consumes a `[sequence, hidden]` matrix:

```text
X = input + position
    ├── Q ──┐
    ├── K ──┴── QKᵀ / √hidden ── row softmax ──┐
    └── V ─────────────────────────────────────┴── attention value
                                                       │
X ───────────────── residual ── LayerNorm ── FFN ── residual ── LayerNorm
                                                                       │
                                                              output projection
```

Inference executors do not retain activations. Training executors keep only the
data required by backward, while the MLP owns scheduling for Linear parameter
updates. Individual Linear layers do not own a tape or workspace.

## Requirements

- an NVIDIA GPU with CUDA support;
- a working NVIDIA driver and CUDA development environment;
- the Rust nightly toolchain pinned in `rust-toolchain.toml`;
- `cargo oxide` installed and configured.

Check the CUDA-Oxide environment first:

```bash
cargo oxide doctor
```

Build and run the project:

```bash
cargo oxide run
```

A plain `cargo build` does not replace this workflow because CUDA-Oxide must
compile and link the device artifact separately.

Build with device debugging enabled:

```bash
cargo oxide run --device-debug
```

Keep optimization while emitting source line information for Compute Sanitizer
or profilers:

```bash
cargo oxide run --lineinfo
```

## Minimal Example

The following example maps a `[batch, input_features]` matrix through a Linear
layer:

```rust
let runtime = CudaRuntime::new()?;

let input = runtime.new_matrix(InitType::Random, 256, 128);
let projection = Linear::new(
    runtime.new_matrix(InitType::Random, 128, 64),
    None,
    Activation::Identity,
);

let output = projection.forward(&input, None, &runtime);
runtime.sync();

assert_eq!((output.rows(), output.cols()), (256, 64));
```

The project is currently a binary crate, so this demonstrates internal API
usage. A stable public crate interface is not a present goal.

## Repository Layout

```text
src/
├── cuda.rs                    CUDA kernels and module entry point
├── cuda/
│   ├── runtime.rs            context, streams, buffers, synchronization
│   ├── span.rs               contiguous device-memory borrows
│   └── container/
│       ├── matrix.rs         Matrix lifecycle, row operations, conversions
│       ├── matrix_compute.rs Matrix computation
│       ├── vector.rs         Vector construction and binary operations
│       ├── vector_compute.rs in-place Vector operations and reductions
│       └── vector_view.rs    contiguous borrowed-view operations
└── net/
    ├── checkpoint.rs         versioned TOML metadata and binary parameters
    ├── linear.rs             Linear, activation, and parameter updates
    ├── mlp.rs                inference/training MLP executors
    └── transformer.rs        single-head Transformer executors
```

See the [CUDA Runtime API](docs/api.md) for the complete container, span,
synchronization, and network-layer reference.

## Current Constraints

- `f32` only;
- contiguous row-major matrices only; no stride support;
- matrix M/K/N dimensions must currently be multiples of 16;
- row Softmax and LayerNorm backward currently support at most 1024 elements per
  row;
- the current Transformer is single-head and Post-LN;
- parameter updates use direct SGD rather than a general optimizer abstraction;
- checkpoint format version 1 stores little-endian `f32` parameters; unsupported
  versions are rejected explicitly.

These are explicit implementation boundaries, not emulations of a generic
framework API. OxideForge will expand when concrete model requirements and
profiling results justify the cost.
