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
- explicit differentiable compute nodes for element-wise arithmetic, MatMul,
  activations, normalization, transpose, row reduction, and contiguous
  concat/split/copy/reduce;
- an explicit `GraphBuilder`/`BranchBuilder` DSL with static shape validation,
  forward scheduling, reverse-order backward scheduling, graph-level loss,
  optimizer steps, and learning-rate schedules;
- inference and training SwiGLU blocks assembled from reusable Linear,
  activation, product, and optimizer components;
- const-generic multi-head Post-Norm Transformer executors with selectable
  LayerNorm/RMSNorm inference and training;
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
live on the container itself. `Graph<true>` retains only the values required by
backward; layers and reusable blocks do not define separate training variants.
A newly allocated output moves into graph state without an additional device
copy. Final outputs are returned by value so the graph controls their lifetime.

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

### Explicit static graphs instead of tracing

OxideForge does not trace arbitrary Rust execution or construct a hidden dynamic
autograd graph. `GraphBuilder` records the topology explicitly: forward follows
the declared order and backward walks the same steps in reverse. Branching,
copying, concatenation, gradient routing, loss, and optimizer scheduling remain
visible in model code.

Drafts contain only configuration. Device parameters are allocated when
`GraphDraft::init` receives a `CudaRuntime`; `Graph<true>` enables training and
`Graph<false>` builds the inference path from the same draft.

## Execution Model

The current Transformer consumes a `[sequence, hidden]` matrix:

```text
X = position_encoding(input)
    ├── Q ──┐
    ├── K ──┴── QₕKₕᵀ / √head_dim ── row softmax ──┐
    └── V ─────────────────────────────────────┴── attention value
                                                       │
X ───────────────── residual ── Norm ── FFN ── residual ── Norm
                                                               │
                                                      output projection
```

`Encoder` and `Decoder` select `NormType::Layer` or `NormType::Rms` when they are
constructed. Positional encoding is an owned `Matrix -> Matrix` closure, so it
can capture device-side state without coupling that state to Transformer. Q/K/V
projections, reusable streams, scaled attention, Softmax, and residual
normalization are owned by one shared Attention module. Training state belongs
to `Graph<true>`; individual blocks do not own a tape or optimizer workspace.

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
artifact is compiled and linked. The workspace `example` package contains the
runnable consumers.

## Graph Example

This is the primary model API. The draft describes two Linear blocks without
allocating device memory. Initialization creates random weights and zero biases;
the graph owns forward, backward, optimizer, and learning-rate scheduling.

```rust
use oxide_forge::cuda::{CudaRuntime, InitType};
use oxide_forge::graph::{
    Graph, GraphBuilder, InitConfig, LearningRateScheduler, MatrixConfig,
};
use oxide_forge::net::linear::{Activation, LinearConfig};
use oxide_forge::net::mlp::Loss;

const ROWS: usize = 16;
const WIDTH: usize = 16;

let mut runtime = CudaRuntime::new()?;
let draft = GraphBuilder::start(MatrixConfig::new(ROWS, WIDTH))
    .then(LinearConfig::new(
        MatrixConfig::new(WIDTH, WIDTH),
        true,
        Activation::Gelu,
    ))
    .then(LinearConfig::new(
        MatrixConfig::new(WIDTH, WIDTH),
        true,
        Activation::Identity,
    ))
    .end();

let mut graph: Graph<true> = draft.init(
    &mut runtime,
    InitConfig::Random {
        loss: Loss::MeanSquaredError,
        learning_rate: LearningRateScheduler::new(1.0e-3)
            .linear(100, 1.0e-4),
    },
)?;

let target = runtime.new_matrix(InitType::Zero, ROWS, WIDTH, None);
for _ in 0..2 {
    let input = runtime.new_matrix(InitType::Random, ROWS, WIDTH, None);
    let loss = graph.train_step(input, &target, 1.0, 0.0, 0.9, &mut runtime);
    println!("loss = {}", loss.sum(&mut runtime, None) / ROWS as f32);
    runtime.recycle_vector(loss);
}
runtime.sync();
```

The same topology can be initialized for inference:

```rust
let inference_draft = build_model_draft();
let mut graph: Graph<false> = inference_draft.init(&mut runtime, init_config)?;
let output = graph.forward(input, &mut runtime);
```

## Repository Layout

```text
src/
├── lib.rs                 base library entry point
├── graph.rs               executable Graph, Branch, and node contract
├── graph/                 builders, drafts, training state, LR schedules
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
    ├── mlp.rs             shared MLP executor and graph-facing block
    ├── node/              explicit differentiable composition operations
    ├── swiglu.rs          SwiGLU feed-forward block
    └── transformer/       attention, encoder, decoder, and position encoding
docs/
└── api.md                 complete runtime API reference
```

See the [CUDA Runtime API](docs/api.md) for the complete container, span,
synchronization, and network-layer reference.

## Current Constraints

- `f32` only;
- contiguous row-major matrices only; no stride support;
- explicit static graph construction only; no runtime tracing or dynamic autograd;
- matrix multiplication uses Tensor Core TF32 products with `f32` accumulation
  and output, requires SM80+, and currently requires M/K/N dimensions to be
  multiples of 16;
- row Softmax, LayerNorm, and RMSNorm backward currently support at most 1024
  elements per row;
- the Transformer is Post-Norm and uses a compile-time head count; the Q/K/V
  buffers stay contiguous and are addressed as per-head ranges without a
  physical split;
- parameter updates currently use classical Momentum SGD rather than a general
  optimizer abstraction;
- checkpoint format version 1 stores little-endian `f32` parameters; unsupported
  versions are rejected explicitly.

These are explicit implementation boundaries, not emulations of a generic
framework API. OxideForge will expand when concrete model requirements and
profiling results justify the cost.
