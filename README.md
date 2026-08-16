# OCR

一个基于 [CUDA-Oxide](https://nvlabs.github.io/cuda-oxide/index.html) 的 Rust/CUDA
OCR 实验项目。目前目标是固定 `512×512` 输入的两级模型，并优先完成第一级单头
Transformer 掩码推理。项目用于练习 GPU 编程和模型实现，不以通用张量框架为目标。

## 第一级模型

```text
Image [512,512,3]
  → Patchify（16×16×3）
Tokens [1024,768]
  → Position Embedding
  → Single-Head Transformer
  → FFN（768→3072→768）
  → Output Projection（768→256）
Mask Patches [1024,256]
  → 按切分顺序拼回
Mask [512,512]
```

每个 token 输出一个 `16×16=256` 的局部掩码块。1024 个块直接重排回原图尺寸，
不做插值放大。

- [第一级架构说明](docs/STAGE1_ARCHITECTURE.md)
- [CUDA Runtime API](docs/impl/api.md)

## 当前能力

- DeviceBuffer 分配、复制与拼接；
- 连续只读/可变 Device Span；
- Vector、VectorView 和 row-major Matrix；
- 矩阵加法、tiled 矩阵乘法与优化转置；
- `for_each`、`scale`、`sum`、`max` 和 `map_sum`；
- 按行 Softmax 和 LayerNorm；
- Linear、GELU、MLP；
- 单头 Self-Attention、残差连接和 `768→256` 输出投影；
- 主 stream 异步提交及额外 stream 的 fork/join。

## 环境准备

```bash
cargo oxide doctor
```

项目必须使用 CUDA-Oxide 工作流生成设备 artifact。普通 `cargo build` 不能替代：

```bash
cargo oxide run
```

设备调试：

```bash
cargo oxide run --device-debug
```

保留优化并生成 profiler/Compute Sanitizer 可用的行号：

```bash
cargo oxide run --lineinfo
```

## 代码结构

```text
src/
├── cuda.rs                    CUDA kernel 与 module 入口
├── cuda/
│   ├── runtime.rs            context、stream、buffer 与同步
│   ├── span.rs               连续设备内存借用
│   └── container/
│       ├── matrix.rs         Matrix 属性、按行操作与转换
│       ├── matrix_compute.rs Matrix 计算
│       ├── vector.rs         Vector 创建与 Vector 间运算
│       ├── vector_compute.rs Vector 原地计算与归约
│       └── vector_view.rs    Span/View 接口与计算
└── net/
    ├── linear.rs
    ├── mlp.rs
    └── transformer.rs
```

## 执行与同步

矩阵乘法、矩阵加法、转置和元素级操作向主 stream 排队，不在每一步自动等待。
同一 stream 天然保证顺序，`Transformer::forward` 在返回前统一同步。

`sum/max/map_sum` 返回 CPU `f32`，因此必须等待归约结果。当前按行 Softmax 和
LayerNorm 保留这种直接实现；在 profiler 证明它们成为瓶颈前，不引入设备标量或
更复杂的调度。

`create_extra_streams()` 使用 `fork()` 创建真正的非阻塞 stream。额外 stream 完成后
必须用 `join_streams()` 或 `sync_streams()` 汇合。
