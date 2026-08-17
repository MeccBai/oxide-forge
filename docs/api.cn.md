# OxideForge CUDA Runtime API

[English](api.md) | 中文

本文档对应当前代码。运行时面向形状明确的神经网络、`f32` 和 row-major Matrix，
不追求绝对通用性。

## 设计约定

- 创建新 buffer/容器的操作由 `CudaRuntime` 提供；
- 修改自身的操作由 `Matrix`、`Vector`、`VectorView` 提供；
- `Matrix` 和 `Vector` 拥有显存；View/Span 只借用显存；
- View/Span 只表示连续区域，不支持 stride；
- 列方向数据必须先转置或物理重排；
- 主 stream 上的设备操作尽量异步排队；
- 返回 host 标量的归约操作必须同步。

## CudaRuntime

### 生命周期与同步

```rust
let runtime = CudaRuntime::new()?;

runtime.stream();
runtime.sync();
```

| API | 说明 |
| --- | --- |
| `new()` | 使用 GPU 0 创建 context、主 stream 并加载 kernel module |
| `stream()` | 获取主 stream |
| `module()` | 获取已加载的 CUDA-Oxide module，供内部封装使用 |
| `sync()` | 等待主 stream |
| `create_extra_streams(n)` | 从主 stream fork `n` 个非阻塞 stream |
| `join_streams(streams)` | 让主 stream 等待额外 stream，不立即阻塞 CPU |
| `sync_streams(streams)` | join 后同步主 stream |

```rust
let streams = runtime.create_extra_streams(task_count);
// 向各 stream 提交互不依赖的工作
runtime.sync_streams(&streams);
```

fork 出来的 stream 会等待 fork 前主 stream 上的工作。只调用 `runtime.sync()` 不会
自动等待尚未 join 的额外 stream。

### DeviceBuffer

```rust
runtime.get_uninit_buffer(len);
runtime.get_zerod_buffer(len);
runtime.clone_buffer(&buffer);
runtime.concat_buffers(&[&a, &b]);
runtime.span_to_buffer_async(&span);
```

- `get_uninit_buffer` 的内容未初始化，读取前必须完全写入；
- clone、concat 和 Span 转 buffer 都产生独立所有权；
- `_async` 接口只提交复制，不主动同步。

## InitType

```rust
pub enum InitType {
    Sequence,
    Reserve,
    Random,
    Zero,
}
```

| 类型 | 内容 |
| --- | --- |
| `Sequence` | `0, 1, 2, ...` |
| `Reserve` | `len, len-1, ...` |
| `Random` | `[0,1]` 伪随机值 |
| `Zero` | 全零 |

## Matrix

### 创建与查询

```rust
let matrix = runtime.new_matrix(InitType::Random, rows, cols);

matrix.rows();
matrix.cols();
let host = matrix.to_host(&runtime);
```

线性布局：

```text
index = row * cols + col
```

### 创建新 Matrix

```rust
let c = runtime.matrix_multiply(&a, &b);
let sum = runtime.matrix_add(&a, &b);
let transposed = runtime.matrix_transpose(&a);
```

逐元素四则运算统一由 `BinaryOp` 控制：

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

`Sub` 和 `Div` 的顺序固定为 `lhs - rhs`、`lhs / rhs`。`matrix_add/sub/mul/div`
是通用入口的薄封装，其中 `mul/div` 是逐元素运算，不是矩阵乘法。

底层只保留 `slice_binary` 和 `slice_binary_assign` 两个 kernel。`BinaryOp` 是整个
launch 共享的 host 枚举参数，warp 内不会因不同元素选择不同操作；该抽象不引入
动态分派、临时 buffer 或额外 kernel launch。

| API | 约束 | 输出 |
| --- | --- | --- |
| `matrix_multiply(a,b)` | `a.cols == b.rows`，当前 M/K/N 均须为 16 的倍数 | `[a.rows,b.cols]` |
| `matrix_add(a,b)` | 形状完全一致 | 输入形状 |
| `matrix_transpose(a)` | 无额外尺寸约束 | `[a.cols,a.rows]` |
| `softmax_rows_backward(p,dy)` | 两者同形且列数不超过 1024 | `dScores` |
| `layer_norm_backward(x,dy)` | 两者同形且列数不超过 1024 | `dX` |

三者向主 stream 异步提交。后续同 stream kernel 可以立即使用返回 Matrix，无需手动
插入同步。

### 修改自身

```rust
matrix.scale(value, &runtime);
matrix.add_val(value, &runtime);
matrix.for_each(&runtime, move |x| x * 2.0);
matrix.softmax_rows(&runtime);
matrix.layer_norm(&runtime);
matrix.binary_assign_by_rows(&bias, BinaryOp::Add, &runtime);
```

| API | 同步行为 |
| --- | --- |
| `scale/add_val/for_each` | 异步提交 |
| `softmax_rows` | `max/sum` 返回 host 标量，内部同步 |
| `layer_norm` | `sum/map_sum` 返回 host 标量，内部同步 |
| `binary_assign` | 与同形 Matrix 原地逐元素计算，异步提交 |
| `binary_assign_by_rows` | 将 Vector 广播到每行；多 stream 提交后统一汇合 |

传入 `for_each` 的闭包必须满足 `Fn(f32) -> f32 + Copy`，并能编译为设备代码。

### Matrix/Vector 转换

| API | 结果 | 复制 |
| --- | --- | --- |
| `vector_zip(vectors)` | 将等长 Vector 作为 Matrix 各行 | 是 |
| `split_view(matrix)` | 按行生成 `Vec<VectorView>` | 否 |
| `matrix_split(matrix)` | 按行生成独立 Vector | 是 |
| `broadcast(vector,copies)` | 重复为 `[copies,vector.len]` | 是 |
| `extract_vector(matrix)` | 单行转移 buffer，多行复制第一行 | 视情况 |
| `matrix_slice(matrix,cols,rows)` | 二维分块并重排为连续 Matrix | 是 |
| `to_vector(matrix)` | 消耗 Matrix，把完整 buffer 作为 Vector | 否 |
| `matrix_copy(matrix)` | 深复制为相同形状的独立 Matrix | 是 |

`split_view` 存活期间 Matrix 保持独占可变借用。按列操作应先转置。

## Vector

### 创建与属性

```rust
let vector = runtime.new_vector(InitType::Random, len);
let cloned = runtime.clone_vector(&vector);

vector.len();
vector.to_host(&runtime);
```

`new_vector` 只需要 `&CudaRuntime`，不要求可变 Runtime。

### 计算

```rust
vector.add(value, &runtime);
vector.scale(value, &runtime);
vector.exp(offset, &runtime); // exp(x - offset)

let sum = vector.sum(&runtime);
let max = vector.max(&runtime);
vector.softmax(&runtime);

let c = runtime.vector_add(&a, &b);
let product = runtime.vector_binary(&a, &b, BinaryOp::Mul);
let dot = runtime.vector_dot_product(&a, &b);
```

`vector_add/sub/mul/div` 同样是 `vector_binary` 的薄封装。点积内部先使用
`BinaryOp::Mul` 逐元素相乘，再执行 sum 归约。

当前 `vector_binary` 及其四个便利封装在返回前同步。`sum/max/vector_dot_product`
返回 host `f32`，同样是同步边界。空输入的 `sum` 返回 `0.0`，`max` 返回
`f32::MIN`。

### 连续 Span

```rust
let full = vector.as_span();
let part = vector.span(offset, len);
```

范围使用 `[offset, offset + len)`，创建时检查越界和整数溢出。

## VectorView

`VectorView` 是对连续设备区域的独占可变借用，通常来自：

```rust
let mut rows = runtime.split_view(&mut matrix);
```

可用操作：

```rust
view.len();
view.add(value, &runtime);
view.scale(value, &runtime);
view.for_each(&runtime, f);
view.sum(&runtime);
view.map_sum(&runtime, f);
view.softmax(&runtime);
```

View 直接修改原 Matrix，不拥有或释放内存。

## DeviceSpan（内部 API）

Span 当前为 crate 内部抽象：

```text
DeviceSpan       连续只读借用
DeviceSpanMut    连续独占可变借用
```

内部能力包括：

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

- `chunks` 按固定长度切分，尾段允许较短；
- `to_buffer` 复制为独立 DeviceBuffer；
- `into_span` 消耗可变 Span 并降级为只读 Span，不复制；
- Span 不支持 stride。

## 网络层

### Linear

```rust
let linear = Linear::new(weights, bias, Activation::Gelu);
let output = linear.forward(&input, None, &runtime);
let output = linear.forward(&input, Some(&residual), &runtime);
```

`weights` 形状为 `[input_features,output_features]`，bias 长度必须等于输出列数。
`residual` 必须与 Linear 输出形状一致。执行顺序为：

```text
matmul → optional bias → optional residual → activation
```

Linear 只保存权重、bias 和 activation，不缓存 forward 中间数据，不负责训练 tape、
workspace 或梯度生命周期；这些由拥有完整局部数据流的 MLP 层调度。

### MlpExecutor、InferenceMLP 与 TrainingMlp

```rust
let mlp = InferenceMLP::new(vec![layer1, layer2], None);
let output = mlp.forward(&input, &runtime);
```

`MlpExecutor` 统一保存 Linear 和 residual 配置，并提供 forward/backward 执行逻辑。
`InferenceMLP` 只委托 forward；`TrainingMlp` 额外持有 forward activation tape 并委托
backward。Linear 本身不缓存运行数据。

典型的两层 FFN 可以表示为：

```text
features → hidden（GELU）→ features（Identity）
```

Residual range `(start,end)` 表示从第 `start` 层输入到第 `end-1` 层输出的 skip。
TrainingMlp 只保存 `layer_inputs[i]`，即第 `i` 层 backward 所需的输入。每层产生的
新 Matrix 直接移动为下一层输入，移动进 Vec 只转移所有权句柄，不复制设备数据。
最后一层输出不由 MLP 缓存，而是按所有权返回给上级模型，由模型决定是否保留。
训练 forward 消费输入 Matrix 并直接移入 tape，不执行隐式设备复制。如果上级还需要
保留该输入，应由上级在调用前明确复制，或在 forward 后通过 `input()` 借用 tape
中的第一个输入。

```rust
let output: Matrix = training_mlp.forward(input, runtime);
let input_gradient = training_mlp.backward(
    &output_gradient,
    learning_rate,
    runtime,
);
```

Backward 对 GELU 层按需重算 pre-activation；Identity 层不会重跑 affine GEMM。每层先
用旧权重计算 `dX`，再执行 SGD 参数更新。Residual 梯度在 source activation 处累加。

### Transformer

```text
input                 [sequence,hidden]
+ position            [sequence,hidden]
Q/K/V                 [sequence,hidden]
QKᵀ                   [sequence,sequence]
softmax(QKᵀ/√hidden)  [sequence,sequence]
attention             [sequence,hidden]
residual + LayerNorm  [sequence,hidden]
MLP + residual + norm [sequence,hidden]
output projection     [sequence,output]
```

`InferenceTransformer` 和 `TrainingTransformer` 都采用 Post-LN。训练版本缓存
Q/K/V、Softmax probability、两次 LayerNorm 输入以及最终编码结果，并实现完整反传：

```text
output projection
→ second LayerNorm
→ FFN + residual
→ first LayerNorm
→ attention matmul
→ row Softmax
→ scaled QKᵀ
→ Q/K/V projection
→ input + position
```

`TrainingTransformer::backward` 返回输入梯度，并使用传入 learning rate 更新 Linear
参数和 position matrix。

## 同步速查

| 操作 | 行为 |
| --- | --- |
| Matrix multiply/add/transpose | 异步提交 |
| Matrix/View 原地元素操作 | 异步提交 |
| `vector_binary` 及便利封装 | 返回前同步 |
| `sum/max/map_sum` | 同步，返回 host 标量 |
| `softmax_rows/layer_norm` | 内部归约同步 |
| `softmax_rows_backward/layer_norm_backward` | 单次按行 kernel，异步提交 |
| `binary_assign_by_rows` | extra streams 并行并统一同步 |
| `InferenceTransformer::forward` / `TrainingTransformer::forward` | 返回前同步 |
| `to_host` | host 读取边界 |

不要为了连接同一主 stream 上的两个 kernel 手动同步；CUDA stream 顺序已经保证前一个
kernel 的输出对后一个可见。
