# OxideForge Workspace

English | [简体中文](README.cn.md)

This workspace contains the reusable runtime and its focused benchmark:

- [`oxide-forge`](oxide-forge/README.md): the reusable Rust/CUDA runtime,
  containers, kernels, and neural-network execution layers;
- `bench`: the fixed-shape GPU benchmark executable.

```text
oxide-forge/       base runtime library
bench/             runtime benchmark
```

The benchmark is the default workspace member and can be launched through the
CUDA-Oxide workflow:

```bash
cargo oxide run
```
