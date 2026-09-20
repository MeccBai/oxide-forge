# OxideForge CUDA Runtime API

[Runtime README](../README.md) | English | [简体中文](api.cn.md)

This document describes the current implementation. The runtime targets
shape-controlled neural networks, `f32`, and contiguous row-major matrices. It
does not aim for unrestricted generality.

## Conventions

- Operations that return a new `Matrix` or `Vector` belong to `CudaRuntime`.
- Operations that mutate an existing value belong to `Matrix`, `Vector`, or
  `VectorView`.
- `Matrix` and `Vector` own device memory; views and spans borrow it.
- Views and spans represent contiguous regions only and do not support strides.
- Column-oriented data must be transposed or physically rearranged first.
- `net::node` composes explicit forward/backward paths; it does not record or
  discover an automatic computation graph.
- Device work on the primary stream is submitted asynchronously whenever
  practical.
- Reductions that return a host scalar must synchronize.

| Result or effect | API owner |
| --- | --- |
| Returns a new `Matrix` or `Vector` | `CudaRuntime` |
| Mutates an existing `Matrix` | `Matrix` |
| Mutates an existing `Vector` | `Vector` |
| Mutates a borrowed row/span | `VectorView` |
| Returns a scalar or host copy | The source container |
| Recycles owned device storage | `CudaRuntime` |

## Graph

`GraphBuilder` records an explicit static topology. It does not trace runtime
operations. `then` appends a device-independent block configuration; `copy`,
`map`, and `concat` describe branches. Shapes are propagated and validated while
the draft is built.

```rust
let draft = GraphBuilder::start(MatrixConfig::new(rows, width))
    .then(LinearConfig::new(
        MatrixConfig::new(width, hidden),
        true,
        Activation::Gelu,
    ))
    .then(LinearConfig::new(
        MatrixConfig::new(hidden, output_width),
        true,
        Activation::Identity,
    ))
    .end();

let mut graph: Graph<true> = draft.init(
    &mut runtime,
    InitConfig::Random {
        loss: Loss::MeanSquaredError,
        learning_rate: LearningRateScheduler::new(1.0e-3),
    },
)?;

let loss = graph.train_step(input, &target, 1.0, 0.0, 0.9, &mut runtime);
```

`LinearConfig` contains only the weight shape, whether a bias exists, and the
activation kind. It never allocates device memory. Under `InitConfig::Random`,
`GraphDraft::init` creates random weights and zero biases using the supplied
runtime. Training is selected by `Graph<true>`, not by a separate Linear type.

| Builder API | Effect |
| --- | --- |
| `GraphBuilder::start(config)` | Start a graph with one Matrix value |
| `then(linear_config)` | Append a configured Linear block |
| `then_node(node)` | Append an already initialized low-level `GraphNode` |
| `copy(count)` | Create independently owned values for branches |
| `map(branches)` | Run one branch for each current value |
| `concat(axis)` | Join branch outputs into one contiguous Matrix |
| `end()` | Validate one final output and create `GraphDraft` |

`BranchBuilder::start` creates a shape-independent branch. Its input and output
shapes are resolved when the branch is attached by `map`.

Forward executes steps in declaration order; backward walks precisely the same
topology in reverse. `Graph::step` updates every trainable node and advances the
graph learning-rate schedule exactly once. `Graph<false>` exposes forward only;
backward and optimizer execution fail immediately.

| Executable API | Description |
| --- | --- |
| `Graph<TRAINING>::forward(input, runtime)` | Consume one input and return the owned output |
| `Graph<true>::backward(gradient, runtime)` | Backpropagate and accumulate parameter gradients |
| `Graph<true>::step(momentum, batch_len, runtime)` | Apply accumulated gradients and advance LR once |
| `Graph<true>::train_step(...)` | Forward, loss, backward, and one optimizer step |
| `learning_rate()` | Inspect the scheduler and current rate |
| `get_input_config()` / `get_output_config()` | Inspect statically validated boundary shapes |

Branches are constructed independently and attached by `map`:

```rust
let left = BranchBuilder::start().then(left_linear_config).end();
let right = BranchBuilder::start().then(right_linear_config).end();

let draft = GraphBuilder::start(input_config)
    .copy(2)
    .map([left, right])
    .concat(MatrixAxis::Columns)
    .end();
```

## CudaRuntime

### Lifecycle and synchronization

```rust
let mut runtime = CudaRuntime::new()?;

runtime.stream();
runtime.sync();
```

| API | Description |
| --- | --- |
| `new()` | Create a context for GPU 0, a primary stream, and the loaded kernel module |
| `stream()` | Borrow the primary stream |
| `module()` | Borrow the loaded CUDA-Oxide module for internal wrappers |
| `sync()` | Wait for the primary stream |
| `create_extra_streams(n)` | Fork `n` non-blocking streams from the primary stream |
| `fork_streams(streams)` | Refresh the fork point before reusing existing streams |
| `join_streams(streams)` | Make the primary stream wait for extra streams without immediately blocking the CPU |
| `sync_streams(streams)` | Join the streams, then synchronize the primary stream |

```rust
let streams = runtime.create_extra_streams(task_count);
// Submit mutually independent work to the streams.
runtime.sync_streams(&streams);
```

A forked stream waits for work submitted to the primary stream before the fork.
Call `fork_streams` before submitting another reusable batch so it observes the
new primary-stream inputs. `join_streams` only inserts event dependencies; it
does not synchronize the CPU.
Calling `runtime.sync()` alone does not wait for extra streams that have not been
joined.

### DeviceBuffer

```rust
runtime.get_uninit_buffer(len);
runtime.get_zerod_buffer(len);
runtime.clone_buffer(&buffer);
runtime.concat_buffers(&[&a, &b]);
runtime.span_to_buffer_async(&span);
```

- `get_uninit_buffer` returns uninitialized memory; every element must be
  written before it is read.
- Clone, concatenation, and span-to-buffer conversion create independent
  ownership.
- Device-to-device clone, concatenation, and span-copy APIs only submit work;
  they do not synchronize the CPU.

## InitType

```rust
pub enum InitType {
    Sequence,
    Reverse,
    Random,
    Zero,
}
```

| Variant | Contents |
| --- | --- |
| `Sequence` | `0, 1, 2, ...` |
| `Reverse` | `len, len - 1, ...` |
| `Random` | Pseudorandom values in `[0, 1]` |
| `Zero` | All zeroes |

## Matrix

### Construction and properties

```rust
let matrix = runtime.new_matrix(InitType::Random, rows, cols, None);

matrix.rows();
matrix.cols();
let host = matrix.to_host(&runtime);
```

The linear row-major index is:

```text
index = row * cols + col
```

### CudaRuntime: operations that create containers

```rust
let product = runtime.matrix_multiply(&a, &b, None);
let sum = runtime.matrix_add(&a, &b, None);
let transposed = runtime.matrix_transpose(&a, None);
let row_sums = runtime.matrix_sum_rows(&a, None);
```

Element-wise container arithmetic accepts a device-compilable closure:

```rust
let c = runtime.matrix_binary(&a, &b, move |lhs, rhs| lhs * rhs, None);
matrix.binary_assign(&rhs, move |lhs, rhs| lhs + rhs, &runtime);
matrix.binary_assign_by_rows(&bias, move |lhs, rhs| lhs + rhs, &runtime);
```

Operand order for subtraction and division is always `lhs - rhs` and
`lhs / rhs`. `matrix_add`, `matrix_sub`, `matrix_mul`, and `matrix_div` are thin
wrappers over the generic entry point. `matrix_mul` is element-wise and must not
be confused with `matrix_multiply`.

The device layer uses one generic `slice_binary` kernel and one
`slice_binary_assign` kernel. The copied `move` closure is compiled as device
code and introduces no dynamic dispatch or additional kernel launch.

| API | Constraint | Output shape |
| --- | --- | --- |
| `matrix_multiply(a, b, stream)` | `a.cols == b.rows`; SM80+; M/K/N must currently be multiples of 16 | `[a.rows, b.cols]` |
| `matrix_add(a, b, stream)` | Identical shapes | Input shape |
| `matrix_transpose(a, stream)` | No additional shape constraint | `[a.cols, a.rows]` |
| `matrix_sum_rows(a, stream)` | Non-empty rows | Vector with `a.rows` elements |
| `softmax_rows_backward(p, dy, stream)` | Identical shapes; at most 1024 columns | `dScores`, same shape |
| `layer_norm_backward(x, dy, stream)` | Identical shapes; at most 1024 columns | `dX`, same shape |

`matrix_multiply` uses Tensor Core TF32 products with `f32` accumulation and
output. This trades a small amount of input-mantissa precision for substantially
higher throughput; it is not a strict IEEE FP32 GEMM.

Every allocating Matrix/Vector runtime API accepts a final `Option<&CudaStream>`.
Pass `None` for the primary stream or `Some(stream)` for an explicit stream.
Both forms submit asynchronously; synchronize or join streams only at an actual
dependency boundary.

### Scalar reductions

`Matrix`, `Vector`, and `VectorView` expose the same generic scalar reduction:

```rust
let squared_sum = matrix.map_reduce(
    &mut runtime,
    0.0,
    move |value| value * value,
    move |lhs, rhs| lhs + rhs,
);

let dot = lhs.zip_map_reduce(
    &rhs,
    &mut runtime,
    0.0,
    move |lhs, rhs| lhs * rhs,
    move |lhs, rhs| lhs + rhs,
);
```

`map_reduce` accepts any input length. One 1024-thread block processes the full
input, with each thread visiting as many coalesced positions as necessary, and
returns one `f32` directly. `sum`, `max`, and `map_sum` are thin wrappers over
this entry point. Both closures must use `move`, implement `Copy`, and be
compilable as device code.

`zip_map_reduce` applies a binary map to two equal-length containers, then uses
the same single-launch reduction path. It avoids materializing an element-wise
temporary; `Vector::dot` and the Dice intersection use this path directly.

The reduction closure must be associative and `identity` must be its identity
value. Floating-point results can differ slightly from a sequential CPU fold
because the parallel reduction order is different. Returning the final host
scalar synchronizes the primary stream. Empty input returns `identity` without
launching a kernel.

### In-place operations

```rust
matrix.scale(value, &runtime);
matrix.add_scalar(value, &runtime);
matrix.sigmoid(&runtime);
matrix.threshold(0.8, &runtime);
matrix.for_each(&runtime, move |x| x * 2.0);
matrix.softmax_rows(&runtime);
matrix.layer_norm(&runtime);
matrix.rms_norm(&runtime);
matrix.binary_assign_by_rows(&bias, move |lhs, rhs| lhs + rhs, &runtime);
```

| API | Synchronization behavior |
| --- | --- |
| `scale`, `add_scalar`, `sigmoid`, `threshold`, `for_each` | Asynchronous submission |
| `softmax_rows` | One block per row in one asynchronous launch |
| `layer_norm` | One block per row in one asynchronous launch |
| `rms_norm` | One block per row in one asynchronous launch |
| `binary_assign` | In-place element-wise operation on equal-shaped matrices; asynchronous |
| `binary_assign_by_rows` | One asynchronous broadcast kernel over the complete Matrix |

The closure passed to `for_each` must implement `Fn(f32) -> f32 + Copy` and must
be compilable as device code.

`matrix_sum_rows`, `softmax_rows`, and `layer_norm` launch one block for every
row. Blocks from the same launch are distributed across SMs by CUDA; no Matrix
row is materialized as a `VectorView`, and no per-row stream or host-valued
reduction is involved.

### Matrix and Vector conversion

| API | Result | Copies device data |
| --- | --- | --- |
| `vector_zip(vectors)` | Equal-length Vectors become Matrix rows | Yes |
| `matrix.row_views()` | A `Vec<VectorView>` over Matrix rows | No |
| `matrix_split(matrix, stream)` | Independent Vector for every row | Yes |
| `broadcast(vector, copies)` | `[copies, vector.len]` Matrix | Yes |
| `extract_vector(matrix)` | Transfers a single-row buffer; copies the first row otherwise | Depends |
| `matrix_slice(matrix, cols, rows)` | Physically rearranged contiguous matrix blocks | Yes |
| `matrix_concat_rows(matrices, stream)` | Concatenate complete row ranges | Yes |
| `matrix_concat_cols(matrices, stream)` | Physically interleave row segments by columns | Yes |
| `matrix_split_rows(matrix, sizes, stream)` | Independently owned row partitions | Yes |
| `matrix_split_cols(matrix, sizes, stream)` | Physically rearranged column partitions | Yes |
| `matrix_into_vector(matrix)` | Consumes the Matrix and transfers its entire buffer | No |
| `vector_into_matrix(vector)` | Consumes the Vector and transfers its buffer as one column | No |
| `clone_matrix(matrix, stream)` | Independent Matrix with the same shape | Yes |

While values returned by `row_views` are alive, the source Matrix remains
exclusively borrowed. Transpose before column-wise processing.

## Vector

### Construction and properties

```rust
let vector = runtime.new_vector(InitType::Random, len, None);
let cloned = runtime.clone_vector(&vector, None);

vector.len();
vector.to_host(&runtime);
```

Allocation and recycling APIs take `&mut CudaRuntime` because they mutate the
exact-size buffer pool directly. Read-only access and in-place kernels that do
not allocate continue to take `&CudaRuntime`.

### Computation

```rust
vector.add_scalar(value, &runtime);
vector.scale(value, &runtime);
vector.sigmoid(&runtime);
vector.exp_shifted(offset, &runtime); // exp(x - offset)

let sum = vector.sum(&mut runtime);
let max = vector.max(&mut runtime);
let squared_sum = vector.map_sum(&mut runtime, move |x| x * x);
let custom = vector.map_reduce(&mut runtime, 0.0, move |x| x, move |a, b| a + b);
vector.softmax(&mut runtime);

let c = runtime.vector_add(&a, &b, None);
let product = runtime.vector_binary(&a, &b, move |lhs, rhs| lhs * rhs, None);
vector.binary_assign(&rhs, &runtime, move |lhs, rhs| lhs + rhs);
let dot = a.dot(&b, &mut runtime);
```

`vector_add`, `vector_sub`, `vector_mul`, and `vector_div` are thin wrappers over
`vector_binary`. `dot` belongs to the source Vector because it returns a scalar
and is implemented as `zip_map_reduce` without a temporary product Vector.

`vector_binary` and its convenience wrappers submit asynchronously. `sum`,
`max`, and `dot` still form synchronization boundaries because they return host
`f32` values. For an empty input, `sum` returns `0.0` and `max` returns
`f32::NEG_INFINITY`.

### Contiguous spans

```rust
let full = vector.as_span();
let part = vector.span(offset, len);
```

Ranges use `[offset, offset + len)`. Construction checks both bounds and integer
overflow.

## VectorView

`VectorView` is an exclusive mutable borrow of a contiguous device region. It is
usually created by splitting a Matrix into rows:

```rust
let mut rows = matrix.row_views();
```

Available operations are:

```rust
view.len();
view.add_scalar(value, &runtime);
view.scale(value, &runtime);
view.for_each(&runtime, f);
view.sum(&mut runtime);
view.max(&mut runtime);
view.map_sum(&mut runtime, f);
view.map_reduce(&mut runtime, identity, map, reduce);
view.zip_map_reduce(&rhs, &mut runtime, identity, map, reduce);
view.softmax(&mut runtime);
```

A view mutates its source Matrix directly and neither owns nor frees memory.

## DeviceSpan (internal API)

Spans are currently crate-internal abstractions:

```text
DeviceSpan       contiguous immutable borrow
DeviceSpanMut    contiguous exclusive mutable borrow
```

Internal capabilities include:

```rust
DeviceSpan::from_buffer(buffer, offset, len);
DeviceSpan::chunks(buffer, chunk_size);
span.to_buffer(runtime);
span.to_buffer_async(runtime);

DeviceSpanMut::from_buffer(buffer, offset, len);
DeviceSpanMut::chunks(buffer, chunk_size);
mut_span.into_span();

runtime.concat_buffers_from_span(&spans);
```

- `chunks` splits by a fixed length; the final chunk may be shorter.
- `to_buffer` copies into an independently owned DeviceBuffer.
- `into_span` consumes a mutable span and downgrades it to an immutable span
  without copying.
- Spans do not support strides.

## Explicit Compute Nodes

`net::node` contains the public primitive operations used to compose a graph.
All node values operate on contiguous `Matrix` objects. Training mode is not a
node type: `Graph<true>` owns the tape, gradients, optimizer state, and reverse
execution, while `Graph<false>` keeps only forward execution state.

### BinaryNode

```rust
use oxide_forge::net::node::{BinaryNode, BinaryOp};

let add = BinaryNode::new(BinaryOp::Add);
let output = add.forward(&[&left, &right], &mut runtime);
```

| `BinaryOp` | Forward | Backward state |
| --- | --- | --- |
| `Add` | Element-wise sum of two or more equal-shaped matrices | Input count only |
| `Sub` | `lhs - rhs` | None |
| `Mul` | Element-wise product | Both inputs |
| `Div` | Element-wise `lhs / rhs` | Both inputs |
| `MatMul` | Matrix product | Both inputs |

`forward_owned` consumes inputs. Element-wise operations reuse the
first allocation as output; MatMul allocates its differently shaped result.
`forward_borrowed` is available only for Add and Sub. The training graph retains
Mul/Div/MatMul operands internally because their derivatives depend on them.

MatMul backward computes `dA = dY @ B^T` and `dB = A^T @ dY`; therefore the
same Tensor Core shape constraints as `CudaRuntime::matrix_multiply` apply to
its forward and backward products.

### SingleNode

```rust
use oxide_forge::net::{
    linear::Activation,
    node::{SingleNode, SingleType},
};

let relu = SingleNode::new(SingleType::Activation(Activation::Relu));
let output = relu.forward(input, &mut runtime);
```

`SingleType` supports `Activation(Identity/Gelu/Relu/Silu/Sigmoid)`, `Scale`,
`Softmax`, `LayerNorm`, `RmsNorm`, and `Transpose`. Scale, Identity, and
Transpose need no numerical forward cache. Activations and normalization retain
their input; Softmax retains its probability output. Transpose backward is
another physical transpose.

### Concat and Split

```rust
use oxide_forge::net::node::{ConcatNode, MatrixAxis, SplitNode};

let concat = ConcatNode::new(MatrixAxis::Columns);
let joined = concat.forward(&[&left, &right], &mut runtime);

let split = SplitNode::new(MatrixAxis::Rows, vec![16, 32]);
let branches = split.forward(&input, &mut runtime);
```

`ConcatNode` and `SplitNode` support rows and columns. Sizes describe row
counts for `Rows` and column counts for
`Columns`, must be non-zero, and must cover the input exactly. Every output owns
contiguous row-major storage. Column operations physically rearrange data rather
than creating a strided view.

### CopyNode and ReduceNode

```rust
use oxide_forge::net::node::{CopyNode, ReduceNode, ReduceOp};

let copy = CopyNode::new(3);
let sum = ReduceNode::new(ReduceOp::Sum, 3);
let mean = ReduceNode::new(ReduceOp::Mean, 3);
```

`CopyNode` creates `n` independently owned copies of one input; its backward
sums all branch gradients. `ReduceNode` consumes `n` equally shaped inputs and
reduces them element-wise with Sum or Mean; its backward copies the upstream
gradient to every input and applies the Mean scale when required.

### RowReduceNode

```rust
use oxide_forge::net::node::{RowReduceNode, RowReduction};

let mean = RowReduceNode::new(RowReduction::Mean);
let row_means = mean.forward(&input, &mut runtime); // [rows, 1]
```

`RowReduction::Sum` and `Mean` reduce `[rows, cols]` to `[rows, 1]`. During
training, the graph retains only the input shape. Backward physically broadcasts
each row gradient to a contiguous `[rows, cols]` Matrix; Mean divides by `cols`.

## Network Layers

### Linear

```rust
let linear = Linear::new(weights, bias, Activation::Gelu);
let output = linear.forward(&input, None, &mut runtime, None);
let output = linear.forward(&input, Some(&residual), &mut runtime, Some(stream));
```

Every Linear compute entry ends in `Option<&CudaStream>`. `None` selects the
runtime's primary stream; `Some(stream)` selects that stream. The public surface
contains only result-returning operations such as `forward`, `affine`, and
`backward`; preallocated `*_into` variants are internal implementation details.
All device-returning Linear operations submit asynchronously.

`weights` has shape `[input_features, output_features]`. If present, the bias
length must equal the output column count. A residual Matrix must have exactly
the same shape as the Linear output. Execution order is:

```text
matrix multiply → optional bias → optional residual → activation
```

Linear owns only its weights, optional bias, and activation. It does not cache
forward values or own a training tape, workspace, or gradient lifetime. The MLP
layer owns scheduling because it sees the complete local data flow.

Available activations are `Identity`, `Gelu`, `Relu`, `Silu`, and `Sigmoid`.
Sigmoid uses separate positive/negative branches to avoid overflow.

### MlpExecutor and Mlp

```rust
let mlp = Mlp::new(vec![layer1, layer2], None);
let output = mlp.forward(&input, &mut runtime);
```

`Loss::MeanSquaredError` consumes ordinary model outputs.
`Loss::BinaryCrossEntropyWithLogits` keeps the final output as logits for stable
loss and gradient computation. Call `Mlp::predict` (or
`Loss::activate_output`) to apply Sigmoid and obtain probabilities. The shared
`loss_rows` and `output_gradient` paths accept a positive-class weight for
imbalanced binary classification and segmentation.

`loss_and_output_gradient` optionally adds soft Dice loss through a non-negative
Dice weight and returns both row losses and the combined logit gradient. A zero
Dice weight is exactly the weighted-BCE path. Dice uses whole-matrix reductions,
so it introduces synchronization and is intended for segmentation objectives.

`MlpExecutor` owns the Linear layers and residual configuration. `Mlp` is the
public forward block. Neither it nor `Linear` owns a training tape or optimizer;
those lifetimes belong to `Graph<true>`.

A typical two-layer FFN is:

```text
features → hidden (GELU) → features (Identity)
```

A residual range `(start, end)` represents a skip from the input of layer
`start` to the output of layer `end - 1`.

When used in `Graph<true>`, the graph compiler creates the required backward
state, accumulates parameter gradients, and applies updates at the graph-level
`step` boundary. Moving a Matrix into graph state transfers only its owning
handle and does not copy device data.

### SwiGLU

`Swiglu` implements the standard three-projection SwiGLU block:

```text
gate   = input @ W_gate
up     = input @ W_up
hidden = SiLU(gate) * up
output = hidden @ W_down
```

`W_gate` and `W_up` have shape `[input_features, hidden_features]`; `W_down`
has shape `[hidden_features, output_features]`. The constructors accept these
three weight matrices and create bias-free Identity Linear projections.

Training state for these operations is generated and owned by `Graph<true>`.

### Transformer

```text
input                 [sequence, hidden]
position_encoding     [sequence, hidden]
Q/K/V                 [sequence, hidden]
QₕKₕᵀ                 [heads × sequence, sequence]
softmax(QₕKₕᵀ/√head)  [heads × sequence, sequence]
attention             [sequence, hidden]
residual + norm       [sequence, hidden]
MLP + residual + norm [sequence, hidden]
output projection     [sequence, output]
```

`Encoder` and `Decoder` select `NormType::Layer` or `NormType::Rms` at
construction. It also accepts an optional set of three reusable Q/K/V streams;
the supplied vector must contain exactly three streams. Passing `None` creates
them lazily on the first forward call. Positional encoding is an owned,
serializable enum:

```rust
let position_encoding = PositionEncoding::additive(position);

let transformer = Encoder::<8>::new(
    query,
    key,
    value,
    position_encoding,
    feed_forward,
    output,
    None,
    NormType::Layer,
);
```

Both executors use a Post-Norm layout. LayerNorm and RMSNorm both provide forward
and backward paths. `QkvProjection` owns the three
Linear projections and reusable streams. It accepts separate query and key/value
inputs, so decoder cross-attention can reuse it. `Attention` owns scaled score
calculation, row Softmax, the value GEMM, and residual normalization.

The head count is carried by `Encoder<const HEADS: usize>` and
`Decoder<const HEADS: usize>`.
Construction asserts that the projection width is divisible by
`HEADS`. Q/K/V remain single contiguous `[sequence, hidden]` matrices: no head
buffers are materialized. A batched-strided GEMM kernel derives the head index
from its global block index, so all head tiles are submitted by one launch while
each block addresses the appropriate Q/K/V range. The primary stream joins Q
and K before scores, while V can overlap until the value GEMM. These are CUDA
event dependencies rather than host synchronizations.

```text
output projection
→ second normalization backward
→ FFN + residual
→ first normalization backward
→ attention matrix products
→ row Softmax
→ scaled QKᵀ
→ Q/K/V projections
→ position encoding
```

When a Transformer is compiled into `Graph<true>`, backward execution and
parameter updates are graph responsibilities. Identity and additive positional
encoding both have an identity derivative with respect to the input.

## Model Checkpoints

MLP and Transformer executors use the same two-file checkpoint convention:

```text
model.toml   versioned model metadata and parameter byte ranges
model.bin    contiguous little-endian f32 parameter data
```

Passing a path without an extension adds `.toml`; other explicit extensions are
rejected. The corresponding `.bin` name is written into the metadata. Available
entry points are:

```rust
use crate::net::checkpoint;

checkpoint::dump_linear(&linear, "linear.toml", &runtime)?;
let linear = checkpoint::load_linear("linear.toml", &runtime)?;

checkpoint::dump_mlp(&mlp, "mlp.toml", &runtime)?;
let mlp = checkpoint::load_mlp("mlp.toml", &runtime)?;

checkpoint::dump_transformer(&model, "model.toml", &runtime)?;
let model = checkpoint::load_transformer("model.toml", &runtime)?;
```

All file operations live in `net::checkpoint`; `Linear`, MLP, and Transformer do
not expose file methods. Graph training state and Q/K/V streams are runtime state
and are not saved. Positional encoding metadata and additive values are stored
with the Transformer; streams are recreated lazily after loading.

The association-layer entry points are `dump_linear/load_linear`,
`dump_mlp/load_mlp`, and `dump_transformer/load_transformer`.

The TOML document records:

- format version, model type, scalar encoding, binary file name and size;
- loss function, Transformer block count, compile-time attention head count,
  and normalization type;
- MLP layer count and optional residual range `[start, end)`;
- the input/output neuron count and activation of each Linear layer;
- Matrix/Vector shapes and each parameter's `[byte_start, byte_end)` range;
- the Transformer's fixed attention and feed-forward residual connections.

`MlpExecutor::new` and `Mlp::new` default to
`Loss::MeanSquaredError`. Use `with_loss` when selecting the persisted loss
explicitly. Version 1 currently defines only mean squared error.

Each model exposes `get_meta_data` and `get_data`; its execution fields remain
private. The checkpoint I/O layer handles generic TOML and BIN conversion, while
the model-association layer validates metadata and constructs concrete models
through their public constructors. D2H uses `Matrix::to_host`/`Vector::to_host`;
H2D uses the public `matrix_from_host`/`vector_from_host` runtime APIs. Loading
validates the format, binary size, contiguous ranges, tensor shapes, Linear
connectivity, residual dimensions, and Transformer dimensions. No checkpoint
kernel, Span-specific transfer path, or privileged access exists.

## Synchronization Summary

| Operation | Behavior |
| --- | --- |
| Matrix multiply/add/transpose | Asynchronous submission |
| Matrix/View in-place element operation | Asynchronous submission |
| Device clone/concatenation and `vector_binary` | Asynchronous submission |
| `sum`, `max`, `map_sum` | Synchronize and return a host scalar |
| `matrix_sum_rows`, `softmax_rows`, `layer_norm`, `rms_norm` | One row-parallel kernel; asynchronous submission |
| `softmax_rows_backward`, `layer_norm_backward` | One row-wise kernel; asynchronous submission |
| `binary_assign_by_rows` | One Matrix-wide broadcast kernel; asynchronous submission |
| Linear/MLP/Transformer device-returning operations | Asynchronous submission |
| `to_host` | Host-read synchronization boundary |

Do not add a manual synchronization merely to connect two kernels submitted to
the same stream. CUDA stream ordering already makes the first kernel's output
visible to the second. Synchronize at a caller-owned host observation or timing
boundary.
