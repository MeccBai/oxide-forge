# OxideForge

OxideForge 是一个基于
[CUDA-Oxide](https://nvlabs.github.io/cuda-oxide/index.html) 编写的 Rust/CUDA
神经网络运行时。项目由 GPU 计算、连续内存容器和神经网络执行层组成，为形状明确、
数据布局可控的模型提供轻量基座。

项目不试图复刻通用张量框架。它面向形状明确、数据布局可控的模型，以手写 CUDA
kernel 和显式所有权换取可预测的数据流、较低的运行时开销，以及足够贴近硬件的编程
体验。

> 追求最大程度的无额外成本抽象，而不是最大程度的抽象。

## 项目状态

OxideForge 目前处于实验性开发阶段，API 会继续调整。核心前向计算路径已经可以构建，
训练反向路径已经完成基础封装；数据集接入、checkpoint 格式和端到端训练程序仍在开发。

当前实现包括：

- CUDA context、module、stream、buffer 分配和同步；
- 连续只读/可变 Device Span，以及基于借用的 VectorView；
- 拥有显存的 `Vector` 和 row-major `Matrix`；
- 元素级四则运算、映射、缩放、归约和广播；
- tiled matrix multiplication 和 shared-memory transpose；
- row Softmax、LayerNorm 及其 backward kernel；
- Linear、GELU、MLP、残差连接和对应反向传播；
- 单头 Post-LN Transformer 的推理与训练执行器；
- 主 stream 异步提交，以及额外 stream 的 fork/join。

尚未完成的主要工程闭环：

- 输入和 label 的数据管线；
- 模型级 forward/backward 与训练循环；
- 权重、bias、位置编码及训练状态的 save/load；
- checkpoint round-trip 和小尺寸数值梯度测试。

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

能够连续执行的操作进入同一个 CUDA stream，不在每次 kernel launch 后等待。返回 host
标量的归约和模型边界才形成同步点。额外 stream 只用于确实彼此独立的工作，并通过
fork/join 明确汇合。

### 专用实现胜过无成本假象

当前运行时固定使用 `f32`，并针对已知模型尺寸提供实现。只有在不会显著增加复杂度或
损害性能时才提升通用性。热点优化以 profiler 结果为依据，而不是预先堆叠抽象层。

## 计算模型

当前 Transformer 接收 `[sequence, hidden]` 矩阵：

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

推理层不保存 activation。训练层只保存反向计算所需的数据，并由 MLP 层统一调度 Linear
参数更新；Linear 自身不持有 tape 或 workspace。

## 环境要求

- 支持 CUDA 的 NVIDIA GPU；
- 可用的 NVIDIA Driver 和 CUDA 开发环境；
- `rust-toolchain.toml` 指定的 Rust nightly toolchain；
- 已安装 `cargo oxide`。

首先检查 CUDA-Oxide 环境：

```bash
cargo oxide doctor
```

构建并运行：

```bash
cargo oxide run
```

普通 `cargo build` 不能替代这个流程，因为设备端代码需要由 CUDA-Oxide 单独生成并链接。

设备调试构建：

```bash
cargo oxide run --device-debug
```

保留优化并为 Compute Sanitizer 或 profiler 生成行号：

```bash
cargo oxide run --lineinfo
```

## 最小示例

下面的示例构造一个 `[batch, input_features]` 输入，并通过 Linear 完成特征映射：

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

项目当前是 binary crate，示例展示的是内部 API 的使用方式；稳定公共 crate 接口不是
现阶段的目标。

## 代码结构

```text
src/
├── cuda.rs                    CUDA kernel 与 module 入口
├── cuda/
│   ├── runtime.rs            context、stream、buffer 与同步
│   ├── span.rs               连续设备内存借用
│   └── container/
│       ├── matrix.rs         Matrix 生命周期、按行操作与转换
│       ├── matrix_compute.rs Matrix 计算
│       ├── vector.rs         Vector 创建和 Vector 间运算
│       ├── vector_compute.rs Vector 原地计算与归约
│       └── vector_view.rs    连续 View 接口与计算
└── net/
    ├── linear.rs             Linear、activation 与参数更新
    ├── mlp.rs                inference/training MLP executor
    └── transformer.rs        single-head Transformer
```

更完整的容器、Span、同步及网络接口说明见
[CUDA Runtime API](docs/api.md)。

## 当前约束

- 仅支持 `f32`；
- 仅支持连续 row-major Matrix，不支持 stride；
- 当前矩阵乘法要求 M/K/N 为 16 的倍数；
- 当前 row Softmax 和 LayerNorm backward 每行最多 1024 个元素；
- 当前 Transformer 是单头 Post-LN 结构；
- 当前参数更新为直接 SGD，不包含通用 optimizer；
- 当前尚无稳定 checkpoint 格式或兼容性承诺。

这些约束是当前实现边界，不是对通用框架接口的模拟。随着实际模型需要和 profiling
结果出现，项目会在明确成本的前提下扩展。
