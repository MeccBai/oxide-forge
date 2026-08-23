# OCR Example

English | [简体中文](README.cn.md)

This package is an OCR-specific consumer of
[`oxide-forge`](../../oxide-forge/README.md). It owns the fixed 512×512 dataset
layout, OCR ViT and mask-MLP assembly, training loop, and inference entry point.
None of these application decisions are part of the base runtime API.

## Configuration

Training and inference read [`config.toml`](config.toml) by default. Paths are
resolved from the process working directory. Set `OCR_CONFIG` to select another
file.

```toml
[model]
load_dir = "model"
save_dir = "model"

[dataset]
source_dir = "dataset/image"
target_dir = "dataset/mask"

[training]
epochs = 30
learning_rate = 0.0001
save_every = 5
```

The current dataset loader expects 256 pairs named `0..255`: JPEG source images
and PNG target masks, all 512×512 pixels.

If both checkpoint files exist in `load_dir`, training resumes from them. If
neither exists, the model is randomly initialized. A partial `.toml`/`.bin`
pair is rejected. Training saves `ocr_vit.{toml,bin}` and
`ocr_mask_mlp.{toml,bin}` into `save_dir` every `save_every` epochs and at the
final epoch when necessary. It does not generate preview images.

## Training

From the workspace root:

```bash
cargo oxide run
```

With another configuration:

```bash
OCR_CONFIG=path/to/config.toml cargo oxide run
```

## Inference

```bash
cargo oxide run --bin ocr-inference
```

`OCR_SAMPLE` selects the dataset index. `OCR_OUTPUT` selects the output PNG:

```bash
OCR_SAMPLE=7 OCR_OUTPUT=model/sample-7.png \
  cargo oxide run --bin ocr-inference
```

Device debugging and line information remain available through
`--device-debug` and `--lineinfo`.
