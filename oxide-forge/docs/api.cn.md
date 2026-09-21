# OxideForge CUDA Runtime API

[运行时 README](../README.cn.md) | [English](api.md) | 中文

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

## Graph

`GraphBuilder` 记录显式的静态拓扑，不追踪运行时操作。`then` 追加与设备无关的
配置，`copy`、`map` 和 `concat` 描述分支；Draft 构建期间会传播并验证形状。

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
        initializer: RandomInit::XavierUniform,
        loss: Loss::MeanSquaredError,
        learning_rate: LearningRateScheduler::new(1.0e-3),
    },
)?;

let loss = graph.train_step(input, &target, 1.0, 0.0, 0.9, &mut runtime);
```

`LinearConfig` 只包含权重形状、是否包含 bias 和 activation，不分配显存。
`InitConfig` 分为 `Random`、`Regular` 和 `Load`。随机初始化使用指定的 `RandomInit`
生成权重，并将 bias 置零；规律初始化将指定的 `RegularInit` 应用于所有参数。
训练能力由 `Graph<true>` 决定，而不是由另一套 Linear 或 Block 类型决定。

| Builder API | 作用 |
| --- | --- |
| `GraphBuilder::start(config)` | 以一个 Matrix 值开始建图 |
| `then(linear_config)` | 追加一个 Linear 配置 |
| `then_node(node)` | 追加已初始化的底层 `GraphNode` |
| `copy(count)` | 为分支创建独立拥有所有权的值 |
| `map(branches)` | 为每个当前值运行一条 Branch |
| `concat(axis)` | 将分支结果拼成一个连续 Matrix |
| `end()` | 验证唯一输出并生成 `GraphDraft` |

Forward 按声明顺序执行，backward 沿同一拓扑反向执行。`Graph::step` 更新所有可训练
节点，并且每次只推进一次图级学习率调度器。`Graph<false>` 只允许 forward；训练入口
会立即失败。

| 执行 API | 说明 |
| --- | --- |
| `Graph<TRAINING>::forward(input, runtime)` | 消费输入并返回拥有所有权的输出 |
| `Graph<true>::backward(gradient, runtime)` | 反向传播并累积参数梯度 |
| `Graph<true>::step(momentum, batch_len, runtime)` | 应用梯度并推进一次学习率 |
| `Graph<true>::train_step(...)` | Forward、loss、backward 和一次更新 |
| `get_input_config()` / `get_output_config()` | 查询静态验证后的边界形状 |

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
    Random(RandomInit),
    Regular(RegularInit),
}

pub enum RandomInit {
    XavierUniform,
    XavierNormal,
    KaimingUniform,
    KaimingNormal,
    Uniform { min: f32, max: f32 },
    Normal { mean: f32, std_dev: f32 },
}

pub enum RegularInit {
    Constant(f32),
    Sequence,
    Reverse,
}
```

| 类型 | 内容 |
| --- | --- |
| `XavierUniform` | `U(-sqrt(6/(fan_in+fan_out)), +sqrt(...))` |
| `XavierNormal` | `N(0, sqrt(2/(fan_in+fan_out)))` |
| `KaimingUniform` | `U(-sqrt(6/fan_in), +sqrt(...))` |
| `KaimingNormal` | `N(0, sqrt(2/fan_in))` |
| `Uniform` / `Normal` | 使用明确参数的随机分布 |
| `Constant` | 全部设为同一个值，包括零 |
| `Sequence` / `Reverse` | 按索引生成的规律值 |

正态分布通过 GPU 上的 Box-Muller 变换生成。Matrix 的行列分别作为 fan-in/fan-out；
Vector 使用 `(len, 1)`。

## Matrix

### 创建与查询

```rust
let matrix = runtime.new_matrix(
    InitType::Random(RandomInit::KaimingNormal),
    rows,
    cols,
    None,
);

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

### 标量归约

`Matrix`、`Vector` 和 `VectorView` 提供相同的通用标量归约入口：

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

`map_reduce` 支持任意长度输入。一个 1024 线程的 block 处理完整输入，每个线程按需
读取多个保持合并访存的位置，并直接返回一个 `f32`。`sum`、`max` 和 `map_sum` 都只是
该入口的薄封装。两个闭包都必须写成 `move`、满足 `Copy`，并且能够编译为设备代码。

`zip_map_reduce` 先对两个等长容器执行二元 map，再复用同一条单次 launch 归约路径。它不会物化
逐元素运算的临时容器；`Vector::dot` 和 Dice intersection 已直接使用该入口。

reduce 闭包必须满足结合律，`identity` 必须是它的单位元。浮点归约的并行计算顺序与
CPU 顺序 fold 不同，因此末位可能略有差异。返回最终 host 标量时会同步主 stream。
空输入不启动 kernel，直接返回 `identity`。

### 修改自身

```rust
matrix.scale(value, &runtime);
matrix.add_scalar(value, &runtime);
matrix.sigmoid(&runtime);
matrix.threshold(0.8, &runtime);
matrix.for_each(&runtime, move |x| x * 2.0);
matrix.softmax_rows(&runtime);
matrix.layer_norm(&runtime);
matrix.rms_norm(&runtime);
matrix.binary_assign_by_rows(&bias, BinaryOp::Add, &runtime);
```

| API | 同步行为 |
| --- | --- |
| `scale/add_scalar/sigmoid/threshold/for_each` | 异步提交 |
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
let vector = runtime.new_vector(
    InitType::Random(RandomInit::Uniform { min: 0.0, max: 1.0 }),
    len,
    None,
);
let cloned = runtime.clone_vector(&vector, None);

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
vector.sigmoid(&runtime);
vector.exp_shifted(offset, &runtime); // exp(x - offset)

let sum = vector.sum(&mut runtime);
let max = vector.max(&mut runtime);
let squared_sum = vector.map_sum(&mut runtime, move |x| x * x);
let custom = vector.map_reduce(&mut runtime, 0.0, move |x| x, move |a, b| a + b);
vector.softmax(&mut runtime);

let c = runtime.vector_add(&a, &b);
let product = runtime.vector_binary(&a, &b, BinaryOp::Mul);
let dot = a.dot(&b, &mut runtime);
```

`vector_add/sub/mul/div` 同样是 `vector_binary` 的薄封装。`dot` 返回标量，因此属于
源 Vector，并通过 `zip_map_reduce` 实现，不再分配临时乘积 Vector。

`vector_binary` 及其四个便利封装异步提交。`sum/max/dot` 因为返回 host `f32`，
仍然是同步边界。空输入的 `sum` 返回 `0.0`，`max` 返回 `f32::NEG_INFINITY`。

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
view.max(&mut runtime);
view.map_sum(&mut runtime, f);
view.map_reduce(&mut runtime, identity, map, reduce);
view.zip_map_reduce(&rhs, &mut runtime, identity, map, reduce);
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

当前 activation 包括 `Identity`、`Gelu`、`Relu`、`Silu` 和 `Sigmoid`。Sigmoid 对正负
输入使用不同的数值分支，避免指数溢出。

### MlpExecutor 与 Mlp

```rust
let mlp = Mlp::new(vec![layer1, layer2], None);
let output = mlp.forward(&input, &mut runtime);
```

`Loss::MeanSquaredError` 直接使用模型输出。`Loss::BinaryCrossEntropyWithLogits` 保持
最后一层输出为 logits，以稳定地计算 loss 和梯度；推理时调用
`Mlp::predict`（或 `Loss::activate_output`）应用 Sigmoid 得到概率。共享的
`loss_rows` 与 `output_gradient` 接受正类别权重，可复用于类别不平衡的二分类和分割。

`loss_and_output_gradient` 可通过非负 Dice 权重加入 soft Dice loss，并同时返回逐行 loss
和组合后的 logit 梯度。Dice 权重为零时就是原加权 BCE 路径。Dice 需要对整张 Matrix
执行归约并产生同步，因此主要用于分割目标。

`MlpExecutor` 统一保存 Linear 和 residual 配置，`Mlp` 是公开的 forward block。
它和 Linear 都不拥有训练 tape 或 optimizer；这些生命周期统一由 `Graph<true>` 管理。

典型的两层 FFN 可以表示为：

```text
features → hidden（GELU）→ features（Identity）
```

Residual range `(start,end)` 表示从第 `start` 层输入到第 `end-1` 层输出的 skip。
当 MLP 被编译进 `Graph<true>` 时，图负责生成反向状态、累积参数梯度，并在图级
`step` 边界统一更新。Matrix 移入图状态只转移拥有所有权的句柄，不复制设备数据。

### Transformer

```text
input                 [sequence,hidden]
position_encoding     [sequence,hidden]
Q/K/V                 [sequence,hidden]
QₕKₕᵀ                 [heads × sequence,sequence]
softmax(QₕKₕᵀ/√head)  [heads × sequence,sequence]
attention             [sequence,hidden]
residual + norm       [sequence,hidden]
MLP + residual + norm [sequence,hidden]
output projection     [sequence,output]
```

`Encoder` 和 `Decoder` 在构造时选择 `NormType::Layer` 或 `NormType::Rms`，同时可以传入
三条可复用的 Q/K/V stream；传入的 Vec 必须正好包含三条 stream。传入 `None` 时会在
第一次 forward 中延迟创建。位置编码是由 Transformer 持有并可序列化的枚举：

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

两种执行器都采用 Post-Norm。LayerNorm 与 RMSNorm 均已提供 forward 和 backward。
`QkvProjection` 统一持有三个 Linear 投影和可复用 stream，并支持 query 与
key/value 使用不同输入，因此后续 decoder cross-attention 可直接复用。`Attention`
负责 scaled score、row Softmax、value GEMM 和 residual norm。

头数由 `Encoder<const HEADS: usize>` 和 `Decoder<const HEADS: usize>` 在编译期指定；构造时
断言投影宽度可以被 `HEADS` 整除。Q/K/V 始终保持为单个连续的
`[sequence, hidden]` 矩阵，不做物理拆分。batched-strided GEMM kernel 从全局 block
索引推导 head，各 head 的所有 tile 由一次 launch 提交，而每个 block 通过 stride
访问对应的 Q/K/V 范围。主 stream 在 score 计算前只等待 Q/K，V 仍可并行，直到
value GEMM 前才等待。这些依赖都是 CUDA event，不同步 CPU。

```text
output projection
→ second normalization backward
→ FFN + residual
→ first normalization backward
→ attention matmul
→ row Softmax
→ scaled QKᵀ
→ Q/K/V projection
→ position encoding
```

Transformer 编译进 `Graph<true>` 后，反向执行和参数更新均由图负责。Identity 与
加法式位置编码对输入的导数均为恒等映射。

## 模型存档

MLP 和 Transformer executor 统一使用两个文件：

```text
model.toml   版本化模型元数据和参数字节范围
model.bin    连续 little-endian f32 参数
```

传入不带扩展名的路径时自动补 `.toml`，显式传入其他扩展名会报错；对应 `.bin` 文件名
会写进元数据。接口如下：

```rust
use crate::net::checkpoint;

checkpoint::dump_linear(&linear, "linear.toml", &runtime)?;
let linear = checkpoint::load_linear("linear.toml", &runtime)?;

checkpoint::dump_mlp(&mlp, "mlp.toml", &runtime)?;
let mlp = checkpoint::load_mlp("mlp.toml", &runtime)?;

checkpoint::dump_transformer(&model, "model.toml", &runtime)?;
let model = checkpoint::load_transformer("model.toml", &runtime)?;
```

所有文件操作都只存在于 `net::checkpoint`；`Linear`、MLP 和 Transformer 不提供
文件方法。Graph 训练状态与 Q/K/V stream 都属于运行时状态，不会保存。位置编码的
类型和加法参数会随 Transformer 一起保存；加载后 stream 会延迟重建。

模型关联层入口为 `dump_linear/load_linear`、`dump_mlp/load_mlp` 和
`dump_transformer/load_transformer`。

TOML 保存以下信息：

- 格式版本、模型类型、标量编码、BIN 文件名和大小；
- 损失函数、Transformer block 数量、编译期 attention head 数和 normalization type；
- MLP 层数和可选 residual 范围 `[start,end)`；
- 每个 Linear 的输入/输出神经元数、activation；
- Matrix/Vector 形状和参数的 `[byte_start,byte_end)`；
- Transformer 固定存在的 attention 与 feed-forward residual 连接。

`MlpExecutor::new` 和 `Mlp::new` 默认使用
`Loss::MeanSquaredError`；需要明确指定要持久化的损失函数时使用 `with_loss`。格式版本
1 当前只定义均方误差。

每个模型只公开 `get_meta_data` 和 `get_data`，执行字段保持私有。checkpoint 的 I/O 层
负责通用 TOML/BIN 转换，模型关联层负责验证元数据，并通过公开构造器组装具体
模型。D2H 复用 `Matrix::to_host` / `Vector::to_host`，H2D 使用公开的
`matrix_from_host` / `vector_from_host` Runtime API。加载器会检查格式、BIN 大小、连续
byte range、tensor 形状、Linear 连接、residual 尺寸和 Transformer 尺寸。整个流程
不需要 checkpoint kernel、Span 传输特例或特权访问。

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
