# 第一级掩码模型架构

## 1. 模型目标

第一级模型接收固定尺寸的 RGB 图像：

```text
Input: [512, 512, 3]
```

模型输出同尺寸的单通道掩码：

```text
Mask: [512, 512]
```

第一级只负责在整张图像上生成掩码、定位有效区域。第二级模型根据掩码裁剪
局部区域，再执行更高精度的识别。因此，第一级不需要分类 token，也不需要将
低分辨率掩码插值放大。

掩码通过每个 patch 独立输出一个局部 `16×16` mask patch，再按照输入切分
顺序原样拼回 `512×512`。

## 2. 它是 MLP 还是 Transformer

第一级整体是 **Transformer 模型**，不是纯 MLP 模型。

两者在模型中的职责不同：

| 组件 | 类型 | 作用 |
| --- | --- | --- |
| Patchify | 数据重排 | 将整张图像拆成 patch token |
| Transformer Encoder | Transformer | 建模不同 patch 之间的全局关系 |
| Encoder 内部 FFN | MLP | 对每个 token 的特征做非线性变换 |
| Mask Head | Linear/MLP | 将每个 token 映射成一个局部 mask patch |
| Unpatchify | 数据重排 | 将局部 mask patch 拼回完整掩码 |

MLP 只能独立处理每个 token，无法主动交换不同 patch 的信息。第一级需要根据
整张图像的上下文判断某个局部区域是否属于目标，因此主干使用 Transformer；
MLP 仅作为 Transformer 内部的 FFN 和最终输出头。

## 3. 尺寸约束

输入图像是 `512×512×3`，输入 patch 是 `16×16×3`：

```text
patch_width  = 16
patch_height = 16
patch_dim    = 16 × 16 × 3 = 768
```

每个方向有 32 个 patch：

```text
patch_cols  = 512 / 16 = 32
patch_rows  = 512 / 16 = 32
patch_count = 32 × 32 = 1024
```

因此正确的 patch 矩阵形状是：

```text
[1024, 768]
```

每个 token 最终输出一个 `16×16` 掩码块：

```text
mask_patch_dim = 16 × 16 = 256
MaskPatches    = [1024, 256]
```

总输出元素数：

```text
1024 × 256 = 262144 = 512 × 512
```

所以 `[1024,256]` 可以在不插值、不复制像素的情况下直接重排为
`[512,512]`。

> 注意：`[256,768]` 不能表示一张 `512×512×3` 图像的全部 `16×16×3`
> patch。它只包含 256 个 patch，对应 `256×256×3` 的覆盖范围。若坚持使用
> 256 个 token，则输入 patch 必须改成 `32×32×3`，mask head 也必须输出
> `32×32=1024` 个值。

## 4. 总体数据流

```text
Image [512,512,3]
    │
    │ Patchify: 16×16×3
    ▼
Patch Tokens [1024,768]
    │
    │ + Position Embedding [1024,768]
    ▼
Transformer Input [1024,768]
    │
    │ N × Single-Head Transformer Encoder
    ▼
Context Features [1024,768]
    │
    │ Per-token Mask Head: 768 → 256
    ▼
Mask Patches [1024,256]
    │
    │ Unpatchify: 32×32 个 16×16 patch
    ▼
Mask Logits [512,512]
    │
    │ Sigmoid / Threshold
    ▼
Mask [512,512]
```

## 5. Patchify

### 5.1 语义

输入按照从上到下、从左到右的 row-major patch 顺序切分：

```text
patch_index = patch_row * 32 + patch_col
```

每个 patch 内部包含 `16×16×3=768` 个 `f32`。由于 patch 原始维度恰好等于
Transformer hidden size 768，可以直接把展平后的 patch 作为 token。

### 5.2 实现方案

实现一个专用 GPU kernel，将图像布局重排为连续 token：

```text
[512,512,3] → [1024,768]
```

推荐接口：

```rust
pub fn patchify_16_rgb(&self, image: &Image) -> Matrix
```

该操作创建新的 `Matrix`，因此按照当前接口规则放在 `CudaRuntime` 上。

如果原图本身的连续布局已经以 patch 为单位排列，则可通过 Span 拼接实现；
常规像素 row-major 图像中的二维 patch 并不连续，使用单次重排 kernel 更合适。

## 6. 位置编码

每个 patch 必须保留空间位置：

```text
Tokens:   [1024,768]
Position: [1024,768]
X = Tokens + Position
```

位置编码是模型权重的一部分。由于输入尺寸和 patch 网格固定，不需要运行时插值。

实现直接复用：

```rust
let x = runtime.matrix_add(&tokens, &position_embedding);
```

## 7. 单头 Transformer Encoder

### 7.1 Attention

对于输入 `X: [1024,768]`：

```text
Q = X × Wq + bq   [1024,768]
K = X × Wk + bk   [1024,768]
V = X × Wv + bv   [1024,768]

Scores = Q × Kᵀ                    [1024,1024]
Scores = Scores / sqrt(768)
Scores = softmax_rows(Scores)

Attention = Scores × V             [1024,768]
Output = Attention × Wo + bo        [1024,768]
```

这是单头 self-attention，因此缩放维度是完整 hidden size 768。

`Wo` 属于标准 Attention 输出投影。如果模型从一开始就按无 `Wo` 结构训练，可以
省略；如果加载标准结构权重，则必须保留。

### 7.2 FFN/MLP

Transformer 内部仍包含逐 token MLP：

```text
Hidden = GELU(X × W1 + b1)   [1024,3072]
Output = Hidden × W2 + b2     [1024,768]
```

这里的 MLP 不负责 patch 之间的信息交换；跨 patch 信息已经由 Attention 完成。

### 7.3 残差与 LayerNorm

实现时必须固定为训练时采用的结构。

推荐采用 ViT 常见的 Pre-LN：

```text
N1 = LayerNorm(X)
X  = X + Attention(N1)

N2 = LayerNorm(X)
X  = X + MLP(N2)
```

如果训练端采用 Post-LN，则推理端必须保持：

```text
X = LayerNorm(X + Attention(X))
X = LayerNorm(X + MLP(X))
```

LayerNorm 的完整形式为：

```text
normalized = (x - mean) / sqrt(variance + epsilon)
output = normalized * gamma + beta
```

当前非破坏性 `map_sum` 可以正确计算方差，不需要复制临时 Vector。后续为了性能，
可将每行的 mean、variance 和 normalize 融合为一个 kernel。

## 8. Mask Head

Transformer 最终输出的每个 token 包含全局上下文特征，但还不是掩码像素。需要
一个逐 token 输出头将 768 维特征转换成该 patch 的 256 个 mask logits：

```text
[1024,768] × [768,256] + [256]
    ↓
[1024,256]
```

### 推荐方案：单层 Linear

```text
mask_patches = features × W_mask + b_mask
```

优点：

- 实现简单，直接复用矩阵乘法；
- 参数量和计算量较小；
- Transformer 输出已经包含全局上下文，head 只负责局部像素解码。

### 可选方案：两层 MLP

如果单层输出能力不足，可以使用：

```text
768 → hidden → GELU → 256
```

建议先实现单层 Linear，依据训练效果再决定是否增加 MLP。无论采用一层还是两层，
第一级主干仍然是 Transformer，Mask Head 才是 MLP/Linear。

## 9. Unpatchify

Unpatchify 只改变数据布局，不进行插值或数值计算。

对于输出像素 `(output_row, output_col)`：

```text
patch_row = output_row / 16
patch_col = output_col / 16
local_row = output_row % 16
local_col = output_col % 16

patch_index = patch_row * 32 + patch_col
local_index = local_row * 16 + local_col

mask[output_row, output_col] = mask_patches[patch_index, local_index]
```

推荐接口：

```rust
pub fn unpatchify_mask_16(&self, patches: &Matrix) -> Matrix
```

输入和输出：

```text
Input:  Matrix [1024,256]
Output: Matrix [512,512]
```

该操作创建新的 Matrix，因此放在 `CudaRuntime` 上。实现使用单次一维 kernel，
每个线程写一个最终 mask 像素；不需要 shared memory。

## 10. 掩码激活与阈值

训练通常直接使用 mask logits。推理时可执行：

```text
probability = sigmoid(logit)
mask = probability >= threshold
```

如果下游只需要二值区域，可以将 sigmoid 与阈值融合。由于：

```text
sigmoid(logit) >= 0.5  ⇔  logit >= 0
```

阈值固定为 0.5 时甚至可以直接判断 logit 是否大于等于 0，省略 sigmoid。

## 11. 运行时实现清单

### 已具备的基础能力

- 连续 `DeviceBuffer`、Span 和 View；
- Matrix 加法、缩放、乘法和转置；
- row softmax；
- 非破坏性 map-sum 归约；
- LayerNorm 所需的 sum、variance 和逐元素变换；
- GELU；
- 残差连接所需的 Matrix 加法。

### 第一级仍需封装

1. `patchify_16_rgb`：`[512,512,3] → [1024,768]`；
2. `SingleHeadAttention`：组织 Q/K/V、缩放、softmax、输出投影；
3. `TransformerBlock`：组织 Attention、FFN、残差和 LayerNorm；
4. `MaskHead`：`[1024,768] → [1024,256]`；
5. `unpatchify_mask_16`：`[1024,256] → [512,512]`；
6. 模型权重和 bias 的加载；
7. 多个 Encoder block 的顺序执行。

## 12. 最终结论

第一级应定义为：

> 以单头 Transformer Encoder 为主干、以逐 token Linear/MLP 为 mask head、
> 通过 unpatchify 生成原分辨率掩码的全局定位模型。

它不是纯 MLP。Transformer 负责让每个局部 patch 获得整张图像的上下文；MLP
负责 Transformer 内部的逐 token 非线性变换以及最终局部 mask 像素输出。

整个模型不存在低分辨率 mask 放大过程，最终 `512×512` 掩码由 1024 个
`16×16` mask patch 按原始空间顺序直接拼接得到。
