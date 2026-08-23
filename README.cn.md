# OxideForge Workspace

[English](README.md) | 简体中文

workspace 包含两个职责明确、彼此分离的成员：

- [`oxide-forge`](oxide-forge/README.cn.md)：可复用的 Rust/CUDA 运行时、容器、kernel
  和神经网络执行层；
- [`examples/ocr`](examples/ocr/README.cn.md)：OCR 专用的数据管线、模型组装、训练程序
  和推理程序。

```text
oxide-forge/       基础运行时库
examples/ocr/      OCR 使用方及可执行示例
```

OCR package 是默认 workspace member，因此可以在仓库根目录直接启动训练：

```bash
cargo oxide run
```

API、约束和具体命令分别记录在各成员自己的 README 中。
