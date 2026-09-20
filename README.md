# OxideForge Workspace

English | [简体中文](README.cn.md)

This workspace contains the reusable runtime and its executable examples:

- [`oxide-forge`](oxide-forge/README.md): the reusable Rust/CUDA runtime,
  containers, kernels, and neural-network execution layers;
- `example`: runnable Graph and CUDA API examples.

```text
oxide-forge/       base runtime library
example/           executable examples
```

The example package is the default workspace member and can be launched through the
CUDA-Oxide workflow:

```bash
cargo oxide run
```
