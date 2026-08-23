# OCR 示例

[English](README.md) | 简体中文

该 package 是 [`oxide-forge`](../../oxide-forge/README.cn.md) 的 OCR 专用使用方，
负责固定的 512×512 数据布局、OCR ViT 与 mask MLP 组装、训练循环和推理入口。这些应用
决策不属于基础运行时 API。

## 配置

训练和推理默认读取 [`config.toml`](config.toml)。其中的路径相对于程序启动目录解析，
也可以通过 `OCR_CONFIG` 指定其他配置文件。

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

当前数据集固定为 `0..255` 共 256 对文件：源数据为 JPEG，目标掩码为 PNG，尺寸均为
512×512。

如果 `load_dir` 中的两份 checkpoint 完整存在，训练会从中继续；两份都不存在时随机
初始化；`.toml`/`.bin` 只存在一半时明确报错。训练每隔 `save_every` 轮覆盖写入
`save_dir` 中的 `ocr_vit.{toml,bin}` 与 `ocr_mask_mlp.{toml,bin}`，最后一轮不在保存
间隔上时额外保存一次。训练不会生成 preview 图片。

## 训练

在 workspace 根目录运行：

```bash
cargo oxide run
```

使用其他配置：

```bash
OCR_CONFIG=path/to/config.toml cargo oxide run
```

## 推理

```bash
cargo oxide run --bin ocr-inference
```

`OCR_SAMPLE` 指定数据集序号，`OCR_OUTPUT` 指定输出 PNG：

```bash
OCR_SAMPLE=7 OCR_OUTPUT=model/sample-7.png \
  cargo oxide run --bin ocr-inference
```

设备调试和 profiler 行号仍可分别通过 `--device-debug` 与 `--lineinfo` 开启。
