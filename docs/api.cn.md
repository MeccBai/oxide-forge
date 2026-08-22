# OxideForge CUDA Runtime API

[English](api.md) | 中文

本文档对应当前代码。运行时面向形状明确的神经网络、`f32` 和 row-major Matrix，
不追求绝对通用性。

## 设计约定

- 返回新 `Matrix` 或 `Vector` 的操作由 `CudaRuntime` 提供；
- 修改自身的操作由 `Matrix`、`Vector`、`VectorView` 提供；
- `Matrix` 和 `Vector` 拥有显存；View/Span 只借用显存；
- View/Span 只表示连续区域，不支持 stride；
- 列方向数据必须先转置或物理重排；
- 主 stream 上的设备操作尽量异步排队；
- 返回 host 标量的归约操作必须同步。

| 结果或影响 | API 所属 |
| --- | --- |
| 返回新的 `Matrix` 或 `Vector` | `CudaRuntime` |
| 修改已有 `Matrix` | `Matrix` |
| 修改已有 `Vector` | `Vector` |
| 修改借用的行或 Span | `VectorView` |
| 返回标量或 host 副本 | 源容器自身 |
| 回收拥有所有权的显存 | `CudaRuntime` |

## CudaRuntime

### 生命周期与同步

```rust
let mut runtime = CudaRuntime::new()?;

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
| `fork_streams(streams)` | 复用已有 stream 前刷新 fork 依赖点 |
| `join_streams(streams)` | 让主 stream 等待额外 stream，不立即阻塞 CPU |
| `sync_streams(streams)` | join 后同步主 stream |

```rust
let streams = runtime.create_extra_streams(task_count);
// 向各 stream 提交互不依赖的工作
runtime.sync_streams(&streams);
```

fork 出来的 stream 会等待 fork 前主 stream 上的工作。只调用 `runtime.sync()` 不会
自动等待尚未 join 的额外 stream。复用已有 stream 提交下一批任务前，需要调用
`fork_streams`，使其等待主 stream 上的新输入。`join_streams` 只插入 CUDA event
依赖，不同步 CPU。

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
- Device-to-device clone、concat 和 Span copy 都只提交工作，不主动同步 CPU。

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

### CudaRuntime：创建新容器

```rust
let c = runtime.matrix_multiply(&a, &b);
let sum = runtime.matrix_add(&a, &b);
let transposed = runtime.matrix_transpose(&a);
let row_sums = runtime.matrix_sum_rows(&a);
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
| `matrix_multiply(a,b)` | `a.cols == b.rows`，要求 SM80+，当前 M/K/N 均须为 16 的倍数 | `[a.rows,b.cols]` |
| `matrix_add(a,b)` | 形状完全一致 | 输入形状 |
| `matrix_transpose(a)` | 无额外尺寸约束 | `[a.cols,a.rows]` |
| `matrix_sum_rows(a)` | 列数不超过 1024 | 长度为 `a.rows` 的 Vector |
| `softmax_rows_backward(p,dy)` | 两者同形且列数不超过 1024 | `dScores` |
| `layer_norm_backward(x,dy)` | 两者同形且列数不超过 1024 | `dX` |

`matrix_multiply` 使用 Tensor Core TF32 乘法，并以 `f32` 累加和输出。它用少量输入尾数
精度换取显著吞吐提升，因此不是严格的 IEEE FP32 GEMM。

会分配结果的 `matrix_multiply` 是内部 `matrix_multiply_into_on` 的薄壳。后者接收
预分配输出和明确的 stream，让模型执行器可以调度互不依赖的 GEMM，而普通 container
API 不需要暴露 stream 选择。

三者向主 stream 异步提交。后续同 stream kernel 可以立即使用返回 Matrix，无需手动
插入同步。

### 修改自身

```rust
matrix.scale(value, &runtime);
matrix.add_scalar(value, &runtime);
matrix.for_each(&runtime, move |x| x * 2.0);
matrix.softmax_rows(&runtime);
matrix.layer_norm(&runtime);
matrix.rms_norm(&runtime);
matrix.binary_assign_by_rows(&bias, BinaryOp::Add, &runtime);
```

| API | 同步行为 |
| --- | --- |
| `scale/add_scalar/for_each` | 异步提交 |
| `softmax_rows` | 单次异步 launch，每行一个 block |
| `layer_norm` | 单次异步 launch，每行一个 block |
| `rms_norm` | 单次异步 launch，每行一个 block |
| `binary_assign` | 与同形 Matrix 原地逐元素计算，异步提交 |
| `binary_assign_by_rows` | 对完整 Matrix 启动一次异步广播 kernel |

传入 `for_each` 的闭包必须满足 `Fn(f32) -> f32 + Copy`，并能编译为设备代码。

`matrix_sum_rows`、`softmax_rows` 和 `layer_norm` 在一次 launch 中为每一行启动一个
block，由 CUDA 自动把这些 block 分配到不同 SM。它们不再把 Matrix 行物化为
`VectorView`，也不再执行逐行 stream 调度或返回 host 标量的归约。

### Matrix/Vector 转换

| API | 结果 | 复制 |
| --- | --- | --- |
| `vector_zip(vectors)` | 将等长 Vector 作为 Matrix 各行 | 是 |
| `matrix.row_views()` | 按行生成 `Vec<VectorView>` | 否 |
| `matrix_split(matrix)` | 按行生成独立 Vector | 是 |
| `broadcast(vector,copies)` | 重复为 `[copies,vector.len]` | 是 |
| `extract_vector(matrix)` | 单行转移 buffer，多行复制第一行 | 视情况 |
| `matrix_slice(matrix,cols,rows)` | 二维分块并重排为连续 Matrix | 是 |
| `matrix_into_vector(matrix)` | 消耗 Matrix，把完整 buffer 作为 Vector | 否 |
| `vector_into_matrix(vector)` | 消耗 Vector，把 buffer 作为单列 Matrix | 否 |
| `clone_matrix(matrix)` | 深复制为相同形状的独立 Matrix | 是 |

`row_views` 存活期间 Matrix 保持独占可变借用。按列操作应先转置。

## Vector

### 创建与属性

```rust
let vector = runtime.new_vector(InitType::Random, len);
let cloned = runtime.clone_vector(&vector);

vector.len();
vector.to_host(&runtime);
```

分配和回收 API 直接修改按精确长度分类的 buffer pool，因此接收
`&mut CudaRuntime`。只读访问以及不分配内存的原位 kernel 仍接收
`&CudaRuntime`。

### 计算

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

`vector_add/sub/mul/div` 同样是 `vector_binary` 的薄封装。`dot` 返回标量，因此属于
源 Vector；其临时乘积仍由 `CudaRuntime` 分配和回收。

`vector_binary` 及其四个便利封装异步提交。`sum/max/dot` 因为返回 host `f32`，
仍然是同步边界。空输入的 `sum` 返回 `0.0`，`max` 返回 `f32::MIN`。

### 连续 Span

```rust
let full = vector.as_span();
let part = vector.span(offset, len);
```

范围使用 `[offset, offset + len)`，创建时检查越界和整数溢出。

## VectorView

`VectorView` 是对连续设备区域的独占可变借用，通常来自：

```rust
let mut rows = matrix.row_views();
```

可用操作：

```rust
view.len();
view.add_scalar(value, &runtime);
view.scale(value, &runtime);
view.for_each(&runtime, f);
view.sum(&mut runtime);
view.map_sum(&mut runtime, f);
view.softmax(&mut runtime);
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
let output = linear.forward(&input, None, &mut runtime, None);
let output = linear.forward(&input, Some(&residual), &mut runtime, Some(stream));
```

所有 Linear 计算入口的最后一个参数统一为 `Option<&CudaStream>`：`None` 使用 runtime
主 stream，`Some(stream)` 使用指定 stream。对外只提供 `forward`、`affine`、
`backward` 这类直接返回结果的接口，不暴露预分配的 `*_into` 变体。所有返回设备对象的
Linear 运算都只异步提交。

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
let output = mlp.forward(&input, &mut runtime);
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

Backward 对非 Identity activation 层按需重算 pre-activation；Identity 层不会重跑
affine GEMM。每层先
用旧权重计算 `dX`，再执行 SGD 参数更新。Residual 梯度在 source activation 处累加。

### Transformer

```text
input                 [sequence,hidden]
+ position            [sequence,hidden]
Q/K/V                 [sequence,hidden]
QKᵀ                   [sequence,sequence]
softmax(QKᵀ/√hidden)  [sequence,sequence]
attention             [sequence,hidden]
residual + norm       [sequence,hidden]
MLP + residual + norm [sequence,hidden]
output projection     [sequence,output]
```

`InferenceTransformer` 在构造时选择 `NormType::Layer` 或 `NormType::Rms`，同时可以传入
三条可复用的 Q/K/V stream；传入的 Vec 必须正好包含三条 stream。传入 `None` 时会在
第一次 forward 中延迟创建：

```rust
let transformer = InferenceTransformer::new(
    query,
    key,
    value,
    position,
    feed_forward,
    output,
    None,
    NormType::Layer,
);
```

两种执行器都采用 Post-Norm。由于 RMSNorm backward 尚未实现，`TrainingTransformer`
目前仍固定使用 LayerNorm。训练版本缓存 Q/K/V、Softmax probability、两次 LayerNorm
输入以及最终编码结果，并实现完整反传：

两个 forward 执行器都会 fork 三条可复用 stream，并通过独立的 Linear `forward`
直接取得 Q/K/V。主 stream 在 `QK^T` 前只 join Q/K，让 V 可以继续与 score 路径重叠，
并在 attention GEMM 前才 join V。所有 join 都是 CUDA event 依赖，不会同步 CPU。

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

## 模型存档

MLP 和 Transformer executor 统一使用两个文件：

```text
model.toml   版本化模型元数据和参数字节范围
model.bin    连续 little-endian f32 参数
```

传入不带扩展名的路径时自动补 `.toml`，显式传入其他扩展名会报错；对应 `.bin` 文件名
会写进元数据。接口如下：

```rust
model.dump_to_file("model.toml", &runtime)?;

let linear = Linear::load_from_file("linear.toml", &mut runtime)?;

let mlp = MlpExecutor::load_from_file("model.toml", &mut runtime)?;
let mlp = InferenceMLP::load_from_file("model.toml", &mut runtime)?;
let mlp = TrainingMlp::load_from_file("model.toml", &mut runtime)?;

let model = InferenceTransformer::load_from_file("model.toml", &mut runtime)?;
let model = TrainingTransformer::load_from_file("model.toml", &mut runtime)?;
```

`Linear`、`MlpExecutor`、两种 MLP 和两种 Transformer 都提供 `dump_to_file` 与
`load_from_file`。推理版和训练版使用相同的持久参数表示；Transformer 元数据还会保存
normalization type，旧元数据缺少该字段时默认使用 LayerNorm。forward activation cache
和 Q/K/V stream 都属于运行时状态，不会保存；推理模型加载后延迟重建 stream，训练模型
加载后的 cache 为空。在 RMSNorm backward 完成前，把 RMSNorm Transformer 作为训练执行器
加载会返回错误。

TOML 保存以下信息：

- 格式版本、模型类型、标量编码、BIN 文件名和大小；
- 损失函数、Transformer block 数量和 normalization type；
- MLP 层数和可选 residual 范围 `[start,end)`；
- 每个 Linear 的输入/输出神经元数、activation；
- Matrix/Vector 形状和参数的 `[byte_start,byte_end)`；
- Transformer 固定存在的 attention 与 feed-forward residual 连接。

`MlpExecutor::new`、`InferenceMLP::new` 和 `TrainingMlp::new` 默认使用
`Loss::MeanSquaredError`；需要明确指定要持久化的损失函数时使用 `with_loss`。格式版本
1 当前只定义均方误差。

dump 直接复用现有 `Matrix::to_host` / `Vector::to_host`。load 根据 TOML 的 byte range
逐个 `seek + read_exact`，只读取当前参数，再通过 `DeviceBuffer::from_host` 创建显存。
加载器会先检查格式版本、BIN 大小、范围、tensor 形状、Linear 连接、residual 尺寸和
Transformer 尺寸。该功能不需要专用 kernel，也没有 Span 传输特例。

## 同步速查

| 操作 | 行为 |
| --- | --- |
| Matrix multiply/add/transpose | 异步提交 |
| Matrix/View 原地元素操作 | 异步提交 |
| Device clone/concat 与 `vector_binary` | 异步提交 |
| `sum/max/map_sum` | 同步，返回 host 标量 |
| `matrix_sum_rows/softmax_rows/layer_norm/rms_norm` | 单次行并行 kernel，异步提交 |
| `softmax_rows_backward/layer_norm_backward` | 单次按行 kernel，异步提交 |
| `binary_assign_by_rows` | 单次全矩阵广播 kernel，异步提交 |
| Linear/MLP/Transformer 返回设备对象的运算 | 异步提交 |
| `to_host` | host 读取边界 |

不要为了连接同一 stream 上的两个 kernel 手动同步；CUDA stream 顺序已经保证前一个
kernel 的输出对后一个可见。只在调用方需要观察 host 结果或计时时同步。
