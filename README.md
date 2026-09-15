# OxideForge Workspace

English | [简体中文](README.cn.md)

This workspace contains two deliberately separated members:

- [`oxide-forge`](oxide-forge/README.md): the reusable Rust/CUDA runtime,
  containers, kernels, and neural-network execution layers;
- [`examples/ocr`](examples/ocr/README.md): the OCR-specific dataset pipeline,
  model assembly, training program, and inference program.

```text
oxide-forge/       base runtime library
examples/ocr/      OCR consumer and executable examples
```

The OCR package is the default workspace member, so its training entry point can
be launched from the repository root with:

```bash
cargo oxide run
```
