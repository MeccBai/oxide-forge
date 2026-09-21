# OxideForge

[English](README.md) | 中文

OxideForge 是一个基于
[CUDA-Oxide](https://nvlabs.github.io/cuda-oxide/index.html) 编写的 Rust/CUDA
神经网络运行时。项目由 GPU 计算、连续内存容器和神经网络执行层组成，为形状明确、
数据布局可控的模型提供轻量基座。

项目不试图复刻通用张量框架。它面向形状明确、数据布局可控的模型，以手写 CUDA
kernel 和显式所有权换取可预测的数据流、较低的运行时开销，以及足够贴近硬件的编程
体验。

> 追求最大程度的无额外成本抽象，而不是最大程度的抽象。

## 项目状态

OxideForge 目前处于实验性开发阶段，API 会继续调整。核心前向与反向路径已经完成，
同时提供版本化 TOML + BIN 模型存档。

当前实现包括：

- CUDA context、module、stream、buffer 分配和同步；
- 连续只读/可变 Device Span，以及基于借用的 VectorView；
- 拥有显存的 `Vector` 和 row-major `Matrix`；
- 元素级四则运算、映射、缩放、归约和广播；
- tiled matrix multiplication 和 shared-memory transpose；
- row Softmax、LayerNorm 和 RMSNorm，三者均提供 backward kernel；
- Linear、可复用的 GELU/ReLU/SiLU/Sigmoid、MLP、BCE-with-logits、残差连接和对应反向传播；
- 单头 Post-Norm Transformer 执行器，推理和训练均可选 LayerNorm/RMSNorm；
- MLP 与 Transformer 参数的 checkpoint 保存和加载；
- 主 stream 异步提交，以及额外 stream 的 fork/join。

运行时尚未完成的主要工程闭环：

- 小尺寸数值梯度测试与 optimizer 状态持久化。

## 设计原则

### 连续内存优先

`Matrix` 固定使用 row-major 连续布局，Span 只表示一段连续设备内存，不支持 stride。
需要按列访问或把不连续区域变为独立对象时，先执行转置或显式重排。项目不会为了表面
上的通用性，把不连续布局的成本扩散到后续所有 kernel。

### 所有权表达数据生命周期

`Matrix` 和 `Vector` 拥有显存，Span 和 View 只借用显存。创建新容器的操作由
`CudaRuntime` 提供，原地修改由容器自身提供。训练执行器拥有 backward 真正需要的
中间结果；每层产生的新矩阵直接 move 到下一层缓存，不为缓存额外执行 device copy。
最终输出按所有权返回，由上级模型决定是否保留。

### 显式同步

能够连续执行的操作进入同一个 CUDA stream，不在每次 kernel launch 后等待。返回设备
对象的模型接口保持异步，模型边界由调用方通过 `runtime.sync()` 决定；返回 host 标量的
归约仍是同步点。额外 stream 只用于确实彼此独立的工作，并通过 fork/join 明确汇合。

### 专用实现胜过无成本假象

当前运行时固定使用 `f32`，并针对已知模型尺寸提供实现。只有在不会显著增加复杂度或
损害性能时才提升通用性。热点优化以 profiler 结果为依据，而不是预先堆叠抽象层。

## 计算模型

当前 Transformer 接收 `[sequence, hidden]` 矩阵：

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

推理和训练执行器在构造时选择 `NormType::Layer` 或 `NormType::Rms`，两种归一化均已
提供 forward 和 backward。位置编码是 Transformer 持有的 `Matrix -> Matrix` 闭包，
可以通过 `move` 捕获自己的 device 侧状态，而不再耦合到 Transformer 参数结构。Q/K/V
投影、可复用 stream、scaled
attention、Softmax、residual norm 以及 attention 训练 cache 统一属于共享
Attention 模块。推理层不保存 activation；Linear 自身不持有 tape 或 workspace。

## 环境要求

- 支持 CUDA 的 NVIDIA GPU；
- 可用的 NVIDIA Driver 和 CUDA 开发环境；
- `rust-toolchain.toml` 指定的 Rust nightly toolchain；
- 已安装 `cargo oxide`。

首先检查 CUDA-Oxide 环境：

```bash
cargo oxide doctor
```

在 workspace 根目录检查基础库：

```bash
cargo check -p oxide-forge
```

可执行程序必须通过 CUDA-Oxide 工作流构建，以便编译并链接 device artifact。workspace
中的 [OCR 示例](../examples/ocr/README.cn.md) 是一个完整的使用方。

## 最小示例

下面的示例构造一个 `[batch, input_features]` 输入，并通过 Linear 完成特征映射：

```rust
let mut runtime = CudaRuntime::new()?;

let input = runtime.new_matrix(
    InitType::Random(RandomInit::Uniform { min: 0.0, max: 1.0 }),
    256,
    128,
    None,
);
let projection = Linear::new(
    runtime.new_matrix(
        InitType::Random(RandomInit::KaimingNormal),
        128,
        64,
        None,
    ),
    None,
    Activation::Identity,
);

let output = projection.forward(&input, None, &mut runtime, None);
runtime.sync();

assert_eq!((output.rows(), output.cols()), (256, 64));
```

## 代码结构

```text
src/
├── lib.rs                 基础库入口
├── cuda.rs                CUDA 类型与模块路由入口
├── cuda/
│   ├── device.rs          device 侧模块路由
│   ├── device/            可复用 device 实现与 kernel 入口
│   ├── runtime.rs         context、stream、buffer 与同步
│   ├── span.rs            连续设备内存借用
│   └── container/         Matrix、Vector、转换、逐行及归一化 API
└── net/
    ├── checkpoint/        元数据、二进制 I/O 与模型组装
    ├── linear.rs          Linear、activation 与参数更新
    ├── metadata.rs        公开参数元数据与 host data
    ├── mlp.rs             通用 MLP executor 与图节点
    └── transformer/       attention、encoder、decoder 与位置编码
docs/
└── api.md                 完整运行时 API
```

更完整的容器、Span、同步及网络接口说明见
[CUDA Runtime API](docs/api.md)。

## 当前约束

- 仅支持 `f32`；
- 仅支持连续 row-major Matrix，不支持 stride；
- 当前矩阵乘法使用 Tensor Core TF32 乘法、`f32` 累加和输出，要求 SM80+，且
  M/K/N 均须为 16 的倍数；
- 当前 row Softmax、LayerNorm 和 RMSNorm backward 每行最多 1024 个元素；
- 当前 Transformer 是单头 Post-Norm 结构；推理和训练均支持 LayerNorm 或 RMSNorm；
- 当前参数更新为直接 SGD，不包含通用 optimizer；
- checkpoint 格式版本 1 固定保存 little-endian `f32` 参数；加载器会明确拒绝不支持的版本。

这些约束是当前实现边界，不是对通用框架接口的模拟。随着实际模型需要和 profiling
结果出现，项目会在明确成本的前提下扩展。
