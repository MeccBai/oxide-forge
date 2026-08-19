# OxideForge CUDA Runtime API

English | [简体中文](api.cn.md)

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
- An `_async` operation only submits the copy and does not synchronize.

## InitType

```rust
pub enum InitType {
    Sequence,
    Reserve,
    Random,
    Zero,
}
```

| Variant | Contents |
| --- | --- |
| `Sequence` | `0, 1, 2, ...` |
| `Reserve` | `len, len - 1, ...` |
| `Random` | Pseudorandom values in `[0, 1]` |
| `Zero` | All zeroes |

## Matrix

### Construction and properties

```rust
let matrix = runtime.new_matrix(InitType::Random, rows, cols);

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
let product = runtime.matrix_multiply(&a, &b);
let sum = runtime.matrix_add(&a, &b);
let transposed = runtime.matrix_transpose(&a);
let row_sums = runtime.matrix_sum_rows(&a);
```

Element-wise arithmetic is selected through `BinaryOp`:

```rust
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

let c = runtime.matrix_binary(&a, &b, BinaryOp::Mul);
matrix.binary_assign(&rhs, BinaryOp::Add, &runtime);
matrix.binary_assign_by_rows(&bias, BinaryOp::Add, &runtime);
```

Operand order for subtraction and division is always `lhs - rhs` and
`lhs / rhs`. `matrix_add`, `matrix_sub`, `matrix_mul`, and `matrix_div` are thin
wrappers over the generic entry point. `matrix_mul` is element-wise and must not
be confused with `matrix_multiply`.

The device layer uses one `slice_binary` kernel and one `slice_binary_assign`
kernel. `BinaryOp` is a host-selected value shared by the entire launch, so this
abstraction introduces no dynamic dispatch, temporary buffer, or additional
kernel launch.

| API | Constraint | Output shape |
| --- | --- | --- |
| `matrix_multiply(a, b)` | `a.cols == b.rows`; SM80+; M/K/N must currently be multiples of 16 | `[a.rows, b.cols]` |
| `matrix_add(a, b)` | Identical shapes | Input shape |
| `matrix_transpose(a)` | No additional shape constraint | `[a.cols, a.rows]` |
| `matrix_sum_rows(a)` | At most 1024 columns | Vector with `a.rows` elements |
| `softmax_rows_backward(p, dy)` | Identical shapes; at most 1024 columns | `dScores`, same shape |
| `layer_norm_backward(x, dy)` | Identical shapes; at most 1024 columns | `dX`, same shape |

`matrix_multiply` uses Tensor Core TF32 products with `f32` accumulation and
output. This trades a small amount of input-mantissa precision for substantially
higher throughput; it is not a strict IEEE FP32 GEMM.

The allocating `matrix_multiply` entry point is a thin wrapper around the
internal `matrix_multiply_into_on`. The latter accepts a preallocated output and
an explicit stream, allowing model executors to schedule independent GEMMs
without exposing stream selection through the ordinary container API.

These operations submit work to the primary stream asynchronously. A later
kernel on the same stream may consume the returned Matrix without an explicit
synchronization.

### In-place operations

```rust
matrix.scale(value, &runtime);
matrix.add_scalar(value, &runtime);
matrix.for_each(&runtime, move |x| x * 2.0);
matrix.softmax_rows(&runtime);
matrix.layer_norm(&runtime);
matrix.rms_norm(&runtime);
matrix.binary_assign_by_rows(&bias, BinaryOp::Add, &runtime);
```

| API | Synchronization behavior |
| --- | --- |
| `scale`, `add_scalar`, `for_each` | Asynchronous submission |
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
| `matrix_split(matrix)` | Independent Vector for every row | Yes |
| `broadcast(vector, copies)` | `[copies, vector.len]` Matrix | Yes |
| `extract_vector(matrix)` | Transfers a single-row buffer; copies the first row otherwise | Depends |
| `matrix_slice(matrix, cols, rows)` | Physically rearranged contiguous matrix blocks | Yes |
| `matrix_into_vector(matrix)` | Consumes the Matrix and transfers its entire buffer | No |
| `vector_into_matrix(vector)` | Consumes the Vector and transfers its buffer as one column | No |
| `clone_matrix(matrix)` | Independent Matrix with the same shape | Yes |

While values returned by `row_views` are alive, the source Matrix remains
exclusively borrowed. Transpose before column-wise processing.

## Vector

### Construction and properties

```rust
let vector = runtime.new_vector(InitType::Random, len);
let cloned = runtime.clone_vector(&vector);

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
vector.exp_shifted(offset, &runtime); // exp(x - offset)

let sum = vector.sum(&mut runtime);
let max = vector.max(&mut runtime);
vector.softmax(&mut runtime);

let c = runtime.vector_add(&a, &b);
let product = runtime.vector_binary(&a, &b, BinaryOp::Mul);
let dot = a.dot(&b, &mut runtime);
```

`vector_add`, `vector_sub`, `vector_mul`, and `vector_div` are thin wrappers over
`vector_binary`. `dot` belongs to the source Vector because it returns a scalar;
its temporary product is still allocated and recycled through `CudaRuntime`.

`vector_binary` and its convenience wrappers currently synchronize before
returning. `sum`, `max`, and `dot` also synchronize because they
return host `f32` values. For an empty input, `sum` returns `0.0` and `max`
returns `f32::MIN`.

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
view.map_sum(&mut runtime, f);
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

## Network Layers

### Linear

```rust
let linear = Linear::new(weights, bias, Activation::Gelu);
let output = linear.forward(&input, None, &mut runtime);
let output = linear.forward(&input, Some(&residual), &mut runtime);
```

`weights` has shape `[input_features, output_features]`. If present, the bias
length must equal the output column count. A residual Matrix must have exactly
the same shape as the Linear output. Execution order is:

```text
matrix multiply → optional bias → optional residual → activation
```

Linear owns only its weights, optional bias, and activation. It does not cache
forward values or own a training tape, workspace, or gradient lifetime. The MLP
layer owns scheduling because it sees the complete local data flow.

### MlpExecutor, InferenceMLP, and TrainingMlp

```rust
let mlp = InferenceMLP::new(vec![layer1, layer2], None);
let output = mlp.forward(&input, &mut runtime);
```

`MlpExecutor` owns the Linear layers and residual configuration and provides the
shared forward/backward execution logic. `InferenceMLP` delegates forward only;
`TrainingMlp` additionally owns the forward activation tape and delegates
backward. Linear itself remains cache-free.

A typical two-layer FFN is:

```text
features → hidden (GELU) → features (Identity)
```

A residual range `(start, end)` represents a skip from the input of layer
`start` to the output of layer `end - 1`.

`TrainingMlp` stores only `layer_inputs[i]`, the input needed to run backward for
layer `i`. Each newly allocated Matrix moves directly into the next layer's
input slot; moving it into a `Vec` transfers only the owning handle and does not
copy device data. The last layer's output is returned by value instead of being
cached, allowing the parent model to decide whether to retain it.

Training forward consumes its input Matrix and moves it into the tape without
an implicit device copy. A caller that still needs the input must explicitly
copy it before the call or borrow the first tape entry afterward through
`input()`.

```rust
let output: Matrix = training_mlp.forward(input, runtime);
let input_gradient = training_mlp.backward(
    &output_gradient,
    learning_rate,
    runtime,
);
```

Backward recomputes pre-activation only for GELU layers; Identity layers do not
repeat the affine GEMM. Each layer computes `dX` using the old weights before
applying its SGD update. Residual gradients are accumulated at the source
activation.

### Transformer

```text
input                 [sequence, hidden]
+ position            [sequence, hidden]
Q/K/V                 [sequence, hidden]
QKᵀ                   [sequence, sequence]
softmax(QKᵀ/√hidden)  [sequence, sequence]
attention             [sequence, hidden]
residual + LayerNorm  [sequence, hidden]
MLP + residual + norm [sequence, hidden]
output projection     [sequence, output]
```

`InferenceTransformer` and `TrainingTransformer` both implement a Post-LN
layout. The training executor caches Q/K/V, Softmax probabilities, both
LayerNorm inputs, and the final encoded value. Its backward path covers:

Both forward executors preallocate Q/K/V outputs, fork three reusable streams,
and launch the projection Linear layers independently. The primary stream joins
Q and K before `QK^T`, while V remains eligible to overlap the score path and is
joined only before the attention GEMM. The joins are CUDA event dependencies,
not host synchronization.

```text
output projection
→ second LayerNorm
→ FFN + residual
→ first LayerNorm
→ attention matrix products
→ row Softmax
→ scaled QKᵀ
→ Q/K/V projections
→ input + position
```

`TrainingTransformer::backward` returns the input gradient and updates Linear
parameters and the positional Matrix with the supplied learning rate.

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
model.dump_to_file("model.toml", &runtime)?;

let linear = Linear::load_from_file("linear.toml", &mut runtime)?;

let mlp = MlpExecutor::load_from_file("model.toml", &mut runtime)?;
let mlp = InferenceMLP::load_from_file("model.toml", &mut runtime)?;
let mlp = TrainingMlp::load_from_file("model.toml", &mut runtime)?;

let model = InferenceTransformer::load_from_file("model.toml", &mut runtime)?;
let model = TrainingTransformer::load_from_file("model.toml", &mut runtime)?;
```

`Linear`, `MlpExecutor`, both MLP forms, and both Transformer forms provide
`dump_to_file` and `load_from_file`. The inference/training forms share identical
persistent parameters. Training activation caches are intentionally excluded
and are empty after loading.

The TOML document records:

- format version, model type, scalar encoding, binary file name and size;
- loss function and Transformer block count;
- MLP layer count and optional residual range `[start, end)`;
- the input/output neuron count and activation of each Linear layer;
- Matrix/Vector shapes and each parameter's `[byte_start, byte_end)` range;
- the Transformer's fixed attention and feed-forward residual connections.

`MlpExecutor::new`, `InferenceMLP::new`, and `TrainingMlp::new` default to
`Loss::MeanSquaredError`. Use `with_loss` when selecting the persisted loss
explicitly. Version 1 currently defines only mean squared error.

Dump uses the existing `Matrix::to_host` and `Vector::to_host` paths. Load seeks
directly to each declared range, reads only that parameter, and creates its
device allocation with `DeviceBuffer::from_host`. It validates the format,
binary size, ranges, tensor shapes, Linear connectivity, residual dimensions,
and Transformer dimensions before constructing the executor. No checkpoint
kernel or Span-specific transfer path exists.

## Synchronization Summary

| Operation | Behavior |
| --- | --- |
| Matrix multiply/add/transpose | Asynchronous submission |
| Matrix/View in-place element operation | Asynchronous submission |
| `vector_binary` and convenience wrappers | Synchronize before returning |
| `sum`, `max`, `map_sum` | Synchronize and return a host scalar |
| `matrix_sum_rows`, `softmax_rows`, `layer_norm`, `rms_norm` | One row-parallel kernel; asynchronous submission |
| `softmax_rows_backward`, `layer_norm_backward` | One row-wise kernel; asynchronous submission |
| `binary_assign_by_rows` | One Matrix-wide broadcast kernel; asynchronous submission |
| Transformer `forward` | Synchronizes before returning |
| `to_host` | Host-read synchronization boundary |

Do not add a manual synchronization merely to connect two kernels submitted to
the same primary stream. CUDA stream ordering already makes the first kernel's
output visible to the second.
