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
checkpoint format.

Implemented capabilities include:

- CUDA context, module, stream, buffer allocation, and synchronization;
- immutable and mutable contiguous device spans, plus borrowed vector views;
- device-owning `Vector` and row-major `Matrix` containers;
- element-wise arithmetic, mapping, scaling, reduction, and row broadcasting;
- tiled matrix multiplication and shared-memory matrix transpose;
- row-wise Softmax, LayerNorm, and RMSNorm, all with backward kernels;
- Linear, reusable GELU/ReLU/SiLU/Sigmoid activations, MLP, BCE-with-logits,
  residual connections, and their backward paths;
- single-head Post-Norm Transformer executors with selectable LayerNorm/RMSNorm
  inference and training;
- parameter checkpoint save/load for MLP and Transformer executors;
- asynchronous submission on the primary stream and explicit fork/join for
  additional streams.

The remaining runtime work is primarily:

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
after every kernel launch. Device-returning model APIs are asynchronous; the
caller chooses model boundaries with `runtime.sync()`. Host-valued reductions
remain synchronization points. Additional streams are reserved for genuinely
independent work and are explicitly rejoined.

### Specialized implementations over illusory generality

The runtime currently uses `f32` and targets known model shapes. Generality is
added only when it does not impose significant complexity or performance cost.
Optimization follows profiler evidence instead of speculative abstraction.

## Execution Model

The current Transformer consumes a `[sequence, hidden]` matrix:

```text
X = position_encoding(input)
    ├── Q ──┐
    ├── K ──┴── QKᵀ / √hidden ── row softmax ──┐
    └── V ─────────────────────────────────────┴── attention value
                                                       │
X ───────────────── residual ── Norm ── FFN ── residual ── Norm
                                                               │
                                                      output projection
```

Inference and training select `NormType::Layer` or `NormType::Rms` when they are
constructed. Positional encoding is an owned `Matrix -> Matrix` closure, so it can
capture its own device-side state without coupling that state to Transformer.
Both normalization types provide forward and backward paths. Q/K/V
projections, their reusable streams, scaled
attention, Softmax, residual normalization, and the attention training cache are
owned by one shared Attention module. Inference executors do not retain
activations. Individual Linear layers do not own a tape or workspace.

## Requirements

- an NVIDIA GPU with CUDA support;
- a working NVIDIA driver and CUDA development environment;
- the Rust nightly toolchain pinned in `rust-toolchain.toml`;
- `cargo oxide` installed and configured.

Check the CUDA-Oxide environment first:

```bash
cargo oxide doctor
```

Check the base library from the workspace root:

```bash
cargo check -p oxide-forge
```

Executable consumers must use the CUDA-Oxide build workflow so the device
artifact is compiled and linked. The workspace's
[OCR example](../examples/ocr/README.md) is one complete consumer.

## Minimal Example

The following example maps a `[batch, input_features]` matrix through a Linear
layer:

```rust
let mut runtime = CudaRuntime::new()?;

let input = runtime.new_matrix(InitType::Random, 256, 128);
let projection = Linear::new(
    runtime.new_matrix(InitType::Random, 128, 64),
    None,
    Activation::Identity,
);

let output = projection.forward(&input, None, &mut runtime, None);
runtime.sync();

assert_eq!((output.rows(), output.cols()), (256, 64));
```

## Repository Layout

```text
src/
├── lib.rs                 base library entry point
├── cuda.rs                CUDA types and module routing entry point
├── cuda/
│   ├── device.rs          device-side module routing
│   ├── device/            reusable device implementations and kernel entries
│   ├── runtime.rs         context, streams, buffers, synchronization
│   ├── span.rs            contiguous device-memory borrows
│   └── container/         Matrix, Vector, conversion, row, and norm APIs
└── net/
    ├── checkpoint/        metadata, binary I/O, and model assembly
    ├── linear.rs          Linear, activation, and parameter updates
    ├── metadata.rs        public parameter metadata and host data
    ├── mlp.rs             inference/training MLP executors
    └── transformer/       attention, encoder, decoder, and position encoding
docs/
└── api.md                 complete runtime API reference
```

See the [CUDA Runtime API](docs/api.md) for the complete container, span,
synchronization, and network-layer reference.

## Current Constraints

- `f32` only;
- contiguous row-major matrices only; no stride support;
- matrix multiplication uses Tensor Core TF32 products with `f32` accumulation
  and output, requires SM80+, and currently requires M/K/N dimensions to be
  multiples of 16;
- row Softmax, LayerNorm, and RMSNorm backward currently support at most 1024
  elements per row;
- the current Transformer is single-head and Post-Norm; inference and training
  support LayerNorm or RMSNorm;
- parameter updates use direct SGD rather than a general optimizer abstraction;
- checkpoint format version 1 stores little-endian `f32` parameters; unsupported
  versions are rejected explicitly.

These are explicit implementation boundaries, not emulations of a generic
framework API. OxideForge will expand when concrete model requirements and
profiling results justify the cost.
