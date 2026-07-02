# LoRA 微调实验规格说明

本文档定义项目中 LoRA 微调实验的代码规格，用于指导后续代码生成。微调实验代码保存在 `PEFT_engine/` 文件夹中。

---

## 1. 设计原则

1. **独立性与低耦合**：模型加载、数据处理、训练循环、评估逻辑各自独立，通过配置文件组合
2. **可扩展**：新增数据集或模型只需实现对应接口，无需修改训练框架
3. **可复现**：固定随机种子，保存完整训练配置与 checkpoint
4. **断点续训**：支持从任意 epoch 恢复训练
5. **策略可切换**：通过配置文件切换微调策略（LoRA / Full Fine-tuning / Frozen Backbone），代码统一

---

## 2. 环境依赖

### 2.1 运行环境

本项目使用两个独立 conda 环境，分别对应两个模型：

| 环境名 | Python | PyTorch | CUDA | 用途 |
|--------|--------|---------|------|------|
| `cbramod` | 3.11 | 2.0.1+cu118 | 11.8 | CBraMod 相关实验 |
| `labram` | 3.11 | 2.0.1+cu118 | 11.8 | LaBraM 相关实验 |

**关键约束**：
- 启动前**必须** `unset LD_LIBRARY_PATH`，否则系统 CUDA 12.0 库会覆盖 PyTorch 自带的 CUDA 11.8 库
- `numpy < 2`（PyTorch 2.0.1 用 NumPy 1.x 编译）

### 2.2 额外依赖

在对应环境中安装 PEFT 库（从本地源码安装）：

```bash
unset LD_LIBRARY_PATH
source activate cbramod  # 或 labram
pip install -e external/Pefts/peft-main/
pip install pyyaml tensorboard
```

---

## 3. 代码结构

```
PEFT_engine/
├── configs/                        # YAML 配置文件（见 §13 实验矩阵）
│   ├── cbramod_chbmit_schemeA.yaml   # P0: 时间增强自研
│   ├── cbramod_siena_schemeA.yaml    # P0: 时间增强自研
│   ├── cbramod_chbmit_schemeC.yaml   # P1: FFN-only 消融
│   ├── cbramod_chbmit_schemeB.yaml   # P2: QKV 拆分 PEFT
│   ├── labram_chbmit_schemeB.yaml   # P0: 深层增强分层
│   ├── labram_siena_schemeC.yaml     # P0: 注意力+LayerScale
│   ├── labram_chbmit_schemeA.yaml     # P1: 标准 PEFT 消融
│   ├── labram_siena_schemeA.yaml      # P1: 标准 PEFT 消融
│   ├── cbramod_chbmit_full.yaml       # 对照: 全量微调
│   └── cbramod_chbmit_frozen.yaml     # 对照: Linear Probing
├── lora/                           # 自研 LoRA 实现（方案 A / B 用）
│   ├── __init__.py
│   ├── lora_layer.py              # LoRALayer / LoRALinear 基础类
│   ├── lora_mha.py                # LoRAMultiheadAttention 包装器
│   └── inject.py                  # inject_lora() 注入逻辑（含 per-module/per-layer r）
├── models/                         # 模型适配层
│   ├── __init__.py
│   ├── base_model.py              # 抽象模型接口
│   ├── cbramod_adapter.py         # CBraMod + LoRA 适配
│   └── labram_adapter.py          # LaBraM + LoRA 适配
├── datasets/                       # 数据集加载器
│   ├── __init__.py
│   ├── base_dataset.py            # 抽象数据集接口
│   ├── chbmit_dataset.py          # CHB-MIT 数据集
│   └── siena_dataset.py           # Siena 数据集
├── losses.py                       # 损失函数（FocalLoss / WeightedBCE）
├── augmentation.py                  # EEG 数据增强（时间平移/通道丢弃/噪声）
├── preprocessing/                  # 离线预处理脚本（独立于训练）
│   ├── preprocess_chbmit.py       # CHB-MIT EDF → pickle 分段
│   └── preprocess_siena.py        # Siena EDF → pickle 分段
├── trainer.py                      # 训练器（训练循环 + checkpoint + resume + 阈值优化）
├── evaluator.py                    # 评估器（segment-level 指标）
├── utils.py                        # 工具函数（日志、seed、IO）
└── main.py                         # 统一入口
```

---

## 4. 数据预处理 Pipeline

### 4.1 设计思路

预处理**离线执行**，独立于训练代码。产出统一中间格式，训练时的 Dataset 只负责加载 + 模型适配 reshape。

```
原始 EDF → [离线预处理] → 统一中间格式 pickle → [Dataset __getitem__] → 模型适配 reshape → 模型输入
```

### 4.2 统一中间格式

每个 segment 保存为一个 pickle 文件：

```python
{
    'X': ndarray,    # shape: [n_channels, n_samples]，如 [16, 2560]
    'y': int,        # 标签（0: 正常，1: 发作）
    'patient': str,  # 患者 ID（用于划分数据集）
}
```

- 采样率统一为 256 Hz
- 窗口长度：10 秒（即 `n_samples = 2560`）
- 通道数：16（选取标准双极导联）

### 4.3 CHB-MIT 预处理

复用 CBraMod 的预处理逻辑（`external/models/CBraMod-main/preprocessing/CHB-MIT/`），流程：

1. **process1.py 逻辑**：读取 EDF → 提取 16 导联 → 解析发作标注 → 保存为通道级 pickle
2. **process2.py 逻辑**：按 10 秒窗口切分 → 判断发作标签 → 正类过采样 → 按患者划分 train/val/test

**数据划分**（按患者）：
- Train: chb01–chb20
- Val: chb21–chb22
- Test: chb23–chb24

**输出目录**：`datas/CHB-MIT/processed/{train,val,test}/`

### 4.4 Siena 预处理

类似流程，输出到 `datas/Siena/processed/{train,val,test}/`。

### 4.5 模型适配 Reshape（在 Dataset 内完成）

不同模型对同一中间格式做不同的 reshape：

| 模型 | 原始 shape | 目标 shape | 操作 |
|------|-----------|------------|------|
| CBraMod | `[16, 2560]` | `[16, 10, 200]` | `resample(2000)` → `reshape(16, 10, 200)` → `/100` |
| LaBraM | `[16, 2560]` | `[16, 10, 200]` | `resample(2000)` → `reshape(16, 10, 200)` |

> **注意**：LaBraM 的 `n_electrodes` 维度即为 16 个双极导联通道，`n_patches=10`，`patch_size=200`。

---

## 5. 模型适配规格

### 5.1 抽象接口

`models/base_model.py` 定义所有模型适配器的基类：

```python
class BaseModelAdapter(ABC):
    @abstractmethod
    def build_model(self, config) -> nn.Module:
        """构建完整模型（backbone + 分类头），加载预训练权重"""

    @abstractmethod
    def apply_lora(self, model: nn.Module, lora_config: dict) -> nn.Module:
        """
        对模型应用 LoRA，返回注入后的模型。若 lora_config 为 None 则跳过（全量微调）。
        lora_config['type'] 决定注入方式：
          - 'peft':   使用 HuggingFace PEFT 库（CBraMod 方案 B/C、LaBraM 方案 A/C）
          - 'custom': 使用自研 LoRA 包装器（CBraMod 方案 A、LaBraM 方案 B）
        """

    @abstractmethod
    def get_trainable_param_info(self, model: nn.Module) -> dict:
        """返回 {'total': int, 'trainable': int, 'ratio': float}"""

    def save_adapter(self, model, path: str):
        """保存 LoRA 权重"""
        if hasattr(model, 'save_pretrained'):
            model.save_pretrained(path)          # PEFT 模型
        else:
            torch.save(model.state_dict(), path)  # 自研 LoRA 模型

    def load_adapter(self, model, path: str):
        """加载 LoRA 权重"""
        if hasattr(model, 'load_adapter'):
            model.load_adapter(path, adapter_name="default")  # PEFT 模型
        else:
            model.load_state_dict(torch.load(path))            # 自研 LoRA 模型
```

### 5.2 CBraMod 适配

**源码位置**：`external/models/CBraMod-main/`

**架构概要**：

| 组件 | 类 | 说明 |
|------|----|------|
| 主干 | `CBraMod` | `PatchEmbedding` → `TransformerEncoder`（12 层）→ `proj_out` |
| 编码层 | `TransformerEncoderLayer` | 双路注意力（空间 `self_attn_s` + 时间 `self_attn_t`）+ FFN |
| 注意力 | `nn.MultiheadAttention` | `self_attn_s`（d_model//2=100, nhead=4）、`self_attn_t`（d_model//2=100, nhead=4）|
| FFN | `nn.Linear` | `linear1`（200→800）、`linear2`（800→200）|
| 分类头 | 自定义 `classifier` | backbone + MLP classifier（训练时不冻结）|

**默认参数**：`d_model=200, dim_feedforward=800, n_layer=12, nhead=8`

**输入形状**：`[batch, 16, 10, 200]`（通道数×patch数×patch大小）

**预训练权重**：`external/models/CBraMod-main/pretrained_weights/pretrained_weights.pth`

#### CBraMod LoRA 挂载方案（癫痫二分类优化）

**问题**：CBraMod 注意力使用 `nn.MultiheadAttention`，QKV 为融合参数 `in_proj_weight`（shape `[3×100, 100]`），PEFT 库无法直接匹配。癫痫检测中时间路（`self_attn_t`）对发作时序模式判别最关键，需分配更大 r。

提供三种方案（对应 `docs/report/LoRA-Schemes.md` 实验矩阵）：

**方案 A（P0 主实验）— 时间路径增强自研 MHA 包装器**

自研 `LoRAMultiheadAttention` 包装类，直接在融合 `in_proj_weight` 上加 LoRA。时间路 r=16，空间路 r=8，FFN r=8。

| 目标模块 | r | 每层参数量 |
|---------|---|----------|
| `self_attn_t.in_proj` + `out_proj` | 16 | 9,600 |
| `self_attn_s.in_proj` + `out_proj` | 8 | 4,800 |
| `linear1` + `linear2` | 8 | 16,000 |
| **合计/层** | | **30,400** |

配置方式（`lora.type: custom`）：
```yaml
lora:
  type: custom                    # 自研注入，非 PEFT 库
  scheme: A
  r_temporal: 16                  # self_attn_t 的 r
  r_spatial: 8                    # self_attn_s 的 r
  r_ffn: 8                        # linear1/linear2 的 r
  lora_alpha_ratio: 2             # alpha = r × 2
  lora_dropout: 0.1
  modules_to_save: ["classifier"]
```

实现路径：`lora/lora_mha.py`（`LoRAMultiheadAttention`）+ `lora/inject.py`（`inject_lora`，支持 per-module r）

**方案 B（P2）— QKV 拆分 + 时间增强（PEFT 库）**

修改模型将 MHA 拆为独立 `q_proj`/`k_proj`/`v_proj`，PEFT 库挂载。时间路 r=16，空间路 r=8。需编写 `remap_pretrained_weights()` 将 `in_proj_weight` 拆分为 `q/k/v_proj.weight`。两次 `get_peft_model` 实现 per-module r。

配置方式（`lora.type: peft`）：
```yaml
lora:
  type: peft
  scheme: B
  r_temporal: 16
  r_spatial: 8
  r_ffn: 8
  target_modules_t: ["q_proj", "k_proj", "v_proj", "out_proj"]
  target_modules_s: ["q_proj", "k_proj", "v_proj", "out_proj"]
  target_modules_ffn: ["linear1", "linear2"]
  bias: "none"
  modules_to_save: ["classifier"]
```

**方案 C（P1 消融）— FFN-only + 增强训练**

仅对 FFN 挂载 LoRA（r=16），注意力完全冻结。

配置方式（`lora.type: peft`）：
```yaml
lora:
  type: peft
  scheme: C
  r: 16
  lora_alpha: 32
  lora_dropout: 0.1
  target_modules: ["linear1", "linear2"]
  bias: "none"
  modules_to_save: ["classifier"]
```

**方案选择**：CHB-MIT 和 Siena 均使用**方案 A**（主实验）；方案 C 作为消融对照（CHB-MIT）；方案 B 作为 PEFT 生态扩展备选。

### 5.3 LaBraM 适配

**源码位置**：`external/models/LaBraM-main/`

**架构概要**：

| 组件 | 类 | 说明 |
|------|----|------|
| 主干 | `NeuralTransformer` | `TemporalConv` → `cls_token` + `pos_embed` → `blocks`（12 层）→ `head` |
| 编码块 | `Block` | `norm1` → `Attention` → `norm2` → `Mlp` |
| 注意力 | `Attention` | `qkv`（nn.Linear, dim→3*dim, 无 bias）、`proj`（nn.Linear, dim→dim）|
| FFN | `Mlp` | `fc1`（nn.Linear, dim→4*dim）、`fc2`（nn.Linear, 4*dim→dim）|
| 分类头 | `head` | `nn.Linear`（训练时不冻结）|

**默认参数**（base）：`embed_dim=200, depth=12, num_heads=10, mlp_ratio=4, patch_size=200`

**模型注册**：通过 `timm.models.registry` 注册，名称为 `labram_base_patch200_200`

**输入形状**：`[batch, n_electrodes, n_patches, patch_size]`，如 `[B, 16, 10, 200]`

#### LaBraM 通道对齐策略

LaBraM 预训练使用 128 个标准电极位置（`pos_embed` shape `[1, 129, 200]`，含 cls_token）。微调时 CHB-MIT/Siena 只有 16 个双极导联。

**处理方式**：通过 `input_chans` 参数传入通道索引，LaBraM 的 `forward_features` 会自动选择对应的 `pos_embed` 子集。对于双极导联（非标准 10-20 电极），采用以下策略：

1. 将 16 个双极导联映射到最接近的标准电极位置索引
2. 在模型初始化时计算并缓存 `input_chans` 列表
3. 若无法精确映射，可**不使用 pos_embed**（设置 `use_abs_pos_emb=False`），退化为纯序列建模

#### LaBraM LoRA 挂载方案（癫痫二分类优化）

LaBraM 使用自研 `Attention` 类，QKV 为独立 `nn.Linear`（`[600, 200]`），PEFT 库可直接挂载。提供三种方案：

**方案 A（P1 消融）— PEFT 标准全量**

全部 4 模块统一 r=8，与 EEG-FM-Bench default 对标。

```yaml
lora:
  type: peft
  scheme: A
  r: 8
  lora_alpha: 16
  lora_dropout: 0.1
  target_modules: ["qkv", "proj", "fc1", "fc2"]
  bias: "none"
  modules_to_save: ["head"]
```

**方案 B（P0 CHB-MIT 主实验）— 深层增强分层 LoRA**

浅层 r=4，中层 r=8，深层 r=16，强化发作模式判别核心层。

```yaml
lora:
  type: custom                  # 需自研 inject 支持 per-layer r
  scheme: B
  layer_r_config:
    shallow: {layers: [0,1,2,3], r: 4}
    middle:  {layers: [4,5,6,7], r: 8}
    deep:    {layers: [8,9,10,11], r: 16}
  lora_alpha_ratio: 2            # alpha = r × 2
  lora_dropout: 0.1
  target_modules: ["qkv", "proj", "fc1", "fc2"]
  bias: "none"
  modules_to_save: ["head"]
```

实现：`lora/inject.py` → `inject_lora_layerwise(model, layer_r_config)`

**方案 C（P0 Siena 主实验）— 注意力 + LayerScale**

仅注意力（`qkv` + `proj`）r=16 + 解冻 LayerScale（`gamma_1`/`gamma_2`），适合小数据集。

```yaml
lora:
  type: peft
  scheme: C
  r: 16
  lora_alpha: 32
  lora_dropout: 0.1
  target_modules: ["qkv", "proj"]
  bias: "none"
  modules_to_save: ["head", "gamma_1", "gamma_2"]  # 解冻 LayerScale
```

**方案选择**：CHB-MIT 使用**方案 B**（深层增强）；Siena 使用**方案 C**（注意力+LayerScale）；方案 A 作为消融对照。

---

## 6. LoRA 超参数与可训练参数量

### 6.1 超参数表

以下为统一超参数，可通过配置文件覆盖：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `r` | 8 | LoRA 秩（CBraMod 时间路可设 16，LaBraM 深层可设 16） |
| `lora_alpha` | 16 | 缩放系数，实际缩放 = alpha / r |
| `lora_dropout` | 0.1 | LoRA 层 dropout（癫痫任务提高至 0.1 防过拟合） |
| `bias` | `"none"` | 不训练 bias |
| `learning_rate` | 1e-4 | LoRA 参数学习率 |
| `head_learning_rate` | 1e-3 | 分类头学习率（更高） |
| `weight_decay` | 0.05 | 权重衰减 |
| `epochs` | 50 | 最大训练轮数 |
| `batch_size` | 64 | 批大小 |
| `warmup_epochs` | 5 | 学习率预热 |
| `label_smoothing` | 0.1 | 标签平滑（仅多分类时生效，二分类自动忽略）|
| `grad_clip` | 1.0 | 梯度裁剪 |
| `early_stopping_patience` | 10 | 验证指标连续 N 个 epoch 不提升则停止训练 |
| `loss_type` | `"focal"` | 损失函数：`"focal"` / `"bce"` / `"bce_weighted"` |
| `focal_alpha` | 0.25 | Focal Loss 正类权重 |
| `focal_gamma` | 2.0 | Focal Loss 聚焦参数 |
| `sampler` | `"weighted"` | 采样策略：`"weighted"` / `"random"` / `"oversample"` |
| `oversample_factor` | 1.0 | 过采样倍率（sampler=oversample 时生效） |
| `threshold_optimization` | true | 训练后是否在验证集上搜索最优阈值 |
| `augment_time_shift` | 1.0 | 时间平移增强范围（秒） |
| `augment_channel_dropout` | 0.1 | 通道随机丢弃概率 |
| `augment_noise_std` | 0.01 | 高斯噪声标准差 |

### 6.2 可训练参数量估算

| 模型 | 方案 | r 配置 | LoRA 参数 | 分类头 | 总可训练 | 比例 |
|------|------|--------|----------|--------|---------|------|
| CBraMod | A (P0) | t=16, s=8, ffn=8 | 365K | ~32K | ~395K | 3.8% |
| CBraMod | B (P2) | t=16, s=8, ffn=8 | 422K | ~32K | ~452K | 4.3% |
| CBraMod | C (P1) | ffn=16 | 384K | ~32K | ~414K | 3.9% |
| LaBraM | A (P1) | 8 | 307K | ~200 | ~307K | 3.6% |
| LaBraM | B (P0) | 4/8/16 | 358K | ~200 | ~359K | 4.2% |
| LaBraM | C (P0) | attn=16 | 233K | ~200 | ~235K | 2.7% |

> 总参数量：CBraMod ≈ 10.5M，LaBraM ≈ 8.6M。参数量在 `build_model` 后通过 `get_trainable_param_info()` 精确计算并记入日志。

---

## 7. 数据集接口规格

### 7.1 抽象接口

`datasets/base_dataset.py` 定义统一接口：

```python
class BaseDataset(ABC):
    def __init__(self, config, model_name: str):
        """
        config:       配置中的 dataset 部分
        model_name:   'cbramod' 或 'labram'，控制 reshape 逻辑
        """

    @abstractmethod
    def get_data_loader(self) -> dict:
        """返回 {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}"""

    @property
    @abstractmethod
    def num_classes(self) -> int:
        """分类数（二分类任务返回 1，使用 BCEWithLogitsLoss）"""

    @property
    @abstractmethod
    def task_type(self) -> str:
        """'binary' 或 'multiclass'"""
```

### 7.2 CHB-MIT 数据集

**预处理后数据**：`datas/CHB-MIT/processed/{train,val,test}/`

**Dataset.__getitem__ 流程**：
1. 加载 pickle → 获取 `X: [16, 2560]`, `y: int`
2. 重采样 `[16, 2560]` → `[16, 2000]`（256Hz → 200Hz×10秒）
3. 根据 `model_name` 做适配：
   - CBraMod: `reshape(16, 10, 200)` → `/ 100`（归一化）
   - LaBraM: `reshape(16, 10, 200)`（无需额外归一化）
4. 返回 `(tensor, label)`

### 7.3 Siena 数据集

**预处理后数据**：`datas/Siena/processed/{train,val,test}/`

流程与 CHB-MIT 统一。

---

## 8. 训练流程规格

### 8.1 Trainer 接口

```python
class Trainer:
    def __init__(self, config, model, data_loader, evaluator):
        """
        config:       完整 YAML 配置
        model:        PeftModel（已应用 LoRA）或普通 nn.Module（全量微调时）
        data_loader:  {'train': ..., 'val': ..., 'test': ...}
        evaluator:    Evaluator 实例
        """

    def train(self):
        """完整训练流程：循环 epoch → 训练 → 验证 → early stopping → 断点"""

    def resume(self, checkpoint_path: str):
        """从 checkpoint 恢复训练状态"""
```

### 8.2 训练循环

每个 epoch 执行：

1. `model.train()` → 遍历 train loader（使用 WeightedRandomSampler 平衡类别）
2. 前向传播 → 计算 loss（+ 数据增强：时间平移 / 通道丢弃 / 噪声）
   - 二分类默认：`FocalLoss(alpha=0.25, gamma=2.0)`
   - 二分类备选：`BCEWithLogitsLoss(pos_weight=N_neg/N_pos)`
   - 多分类：`CrossEntropyLoss(label_smoothing=0.1)`
3. 反向传播 → `clip_grad_norm_` → `optimizer.step()` → `scheduler.step()`（**step-level 调度**）
4. 验证集评估 → 记录指标（**PR AUC** 为主选择指标）
5. Early Stopping 判断：若验证指标连续 `patience` 个 epoch 不提升，停止训练
6. 若验证指标提升 → 保存最佳 checkpoint
7. 每个 `save_ckpt_freq` epoch → 保存定期 checkpoint
8. 覆盖保存 `latest.pt`（用于断点续训）
9. 记录日志（控制台 + TensorBoard + JSON）

**训练后步骤**（`threshold_optimization=true` 时）：
10. 加载 best checkpoint → 在验证集上计算各阈值下的 F1
11. 选择 F1 最优阈值（或满足 sensitivity ≥ 0.8 的阈值）
12. 使用最优阈值在测试集上评估 → 保存最终结果

### 8.3 优化器策略

采用分组学习率：
- **LoRA 参数**：`learning_rate`（默认 1e-4）
- **分类头参数（modules_to_save）**：`head_learning_rate`（默认 1e-3）

优化器：`AdamW`（weight_decay=0.05）

调度器：`CosineAnnealingLR` with Warmup（step-level，总步数 = epochs × steps_per_epoch）

### 8.4 微调策略切换

通过配置文件中 `lora` 字段控制：

| 配置 | 行为 |
|------|------|
| `lora:` 正常配置 | 冻结 backbone，应用 LoRA + 分类头训练 |
| `lora: null` | 全量微调（所有参数均可训练）|
| `lora: {frozen: true}` | Frozen backbone（只训练分类头，Linear Probing）|

---

## 9. 评估流程规格

### 9.1 开发阶段：Segment-level 评估

训练循环内使用 sklearn 指标做模型选择（对 10 秒 segment 做分类判断）：

| 指标 | 函数 | 说明 | 选择指标 |
|------|------|------|----------|
| PR AUC | `auc(recall, precision)` | 正类精确率-召回率曲线下面积 | **主选择指标** |
| ROC AUC | `roc_auc_score` | 阈值无关判别力 | 次要 |
| Sensitivity | `recall_score` | 发作段召回率（假阴性代价） | 关键 |
| Specificity | `tn / (tn + fp)` | 正常段特异度（假阳性代价） | 关键 |
| Balanced Accuracy | `balanced_accuracy_score` | 各类召回率均值 | 次要 |
| F1 (binary) | `f1_score` | 发作类 F1 | 次要 |
| Cohen's Kappa | `cohen_kappa_score` | 一致性系数 | 次要 |
| False Alarm Rate | `fp / (fp + tn)` | 假阳性率 | 报告用 |

> **主选择指标改为 PR AUC**：在极端类别不平衡（发作段 <5%）场景下，ROC AUC 会高估性能（负类基数大导致假阳性率偏低）。PR AUC 更能反映模型对正类（发作）的实际判别能力。

### 9.2 最终阶段：Event-level 评估（SzCORE）

训练完成后，将最佳模型接入 SzCORE 评估流程：

1. 将 segment-level 模型包装为 EDF 推理接口（滑动窗口推理 → 发作事件聚合）
2. 按 SzCORE 数据流规范输出事件预测（`{onset, offset, confidence}`）
3. 使用 SzCORE 指标评估（event-based F1, OVLP, epoch-based sensitivity/specificity 等）

> 此阶段为训练后的独立评估步骤，不影响训练循环代码。

---

## 10. Checkpoint 与断点续训

### 10.1 保存内容

```python
checkpoint = {
    'epoch': int,                    # 当前完成的 epoch
    'best_metric': float,            # 最佳验证 PR AUC
    'best_epoch': int,               # 最佳 epoch
    'patience_counter': int,         # early stopping 计数器
    'optimizer_state': dict,         # optimizer.state_dict()
    'scheduler_state': dict,         # scheduler.state_dict()
    'config': dict,                  # 完整训练配置（便于复现）
}
# LoRA 权重单独保存
# - PEFT 模型：save_pretrained() → 目录格式（adapter_model.bin + adapter_config.json）
# - 自研 LoRA：torch.save(state_dict) → 单文件格式
```

> **关键**：LoRA 权重单独保存。PEFT 模型保存为目录格式（`adapter_model.bin` + `adapter_config.json`），自研 LoRA 模型保存为 `state_dict` 单文件。`save_adapter()` / `load_adapter()` 自动判断格式。

### 10.2 保存策略

- **最佳模型**：`{output_dir}/best_adapter/` — 验证指标提升时覆盖保存
- **最新 adapter**：`{output_dir}/latest_adapter/` — 每 epoch 覆盖
- **训练状态**：`{output_dir}/latest.pt` — 每 epoch 覆盖（含 optimizer/scheduler/epoch）
- **定期 checkpoint**：`{output_dir}/checkpoint_epoch{N}/` — 每 `save_ckpt_freq` epoch

### 10.3 恢复流程

1. 构建模型 → 应用 LoRA
2. 加载 adapter 权重：`model.load_adapter(f"{output_dir}/latest_adapter/")`
3. 加载 `latest.pt` → 恢复 optimizer/scheduler/epoch/patience
4. 从 `epoch + 1` 继续训练

---

## 11. 日志规格

### 11.1 控制台日志

每个 epoch 输出：

```
[Epoch 3/50] Loss: 0.23412 | Val: pr_auc=0.835, sens=0.82, spec=0.85 | LR: 9.0e-05 | Time: 12.3 min | EarlyStopping: 0/10
```

### 11.2 TensorBoard

记录以下标量（per epoch）：
- `train/loss`
- `val/pr_auc`（主指标）, `val/roc_auc`, `val/balanced_accuracy`, `val/f1`, `val/kappa`
- `val/sensitivity`, `val/specificity`, `val/false_alarm_rate`
- `test/pr_auc`, `test/roc_auc`, `test/balanced_accuracy`, `test/f1`, `test/kappa`
- `test/sensitivity`, `test/specificity`, `test/false_alarm_rate`
- `train/learning_rate`
- `model/trainable_params`（仅第 0 epoch）

日志目录：`{output_dir}/tensorboard/`

### 11.3 JSON 日志

每个 epoch 追加一行到 `{output_dir}/log.jsonl`：

```json
{"epoch": 3, "train_loss": 0.234, "val_pr_auc": 0.835, "val_roc_auc": 0.901, "val_sensitivity": 0.82, "val_specificity": 0.85, "test_pr_auc": 0.79, "test_roc_auc": 0.878, "lr": 9e-05, "time_min": 12.3, "patience": 0}
```

### 11.4 训练启动日志

训练开始时记录：
- 完整配置（YAML dump）
- 模型结构摘要（`print(model)`）
- 可训练参数统计（`total / trainable / ratio`）
- 数据集大小（train/val/test 样本数）

---

## 12. 配置文件格式

YAML 配置文件统一管理一次实验的全部参数：

```yaml
# configs/cbramod_chbmit_schemeA.yaml（P0 主实验：时间路径增强自研 MHA 包装器）
model:
  name: cbramod                     # 'cbramod' | 'labram'
  pretrained_weights: external/models/CBraMod-main/pretrained_weights/pretrained_weights.pth
  classifier: all_patch_reps        # CBraMod 分类头类型
  d_model: 200
  dim_feedforward: 800
  n_layer: 12
  nhead: 8

lora:                               # 设为 null 则全量微调；设 frozen: true 则 Linear Probing
  type: custom                      # 'custom'（自研注入）| 'peft'（PEFT 库）
  scheme: A                         # A / B / C（见 §5.2 / §5.3）
  r_temporal: 16                    # self_attn_t 的 r
  r_spatial: 8                      # self_attn_s 的 r
  r_ffn: 8                          # linear1/linear2 的 r
  lora_alpha_ratio: 2               # alpha = r × ratio
  lora_dropout: 0.1
  modules_to_save: ["classifier"]

dataset:
  name: chbmit                      # 'chbmit' | 'siena'
  processed_data_dir: datas/CHB-MIT/processed
  num_classes: 1                    # 1=二分类(Focal Loss), >1=多分类(CE)
  task_type: binary

train:
  epochs: 50
  batch_size: 64
  learning_rate: 0.0001
  head_learning_rate: 0.001
  weight_decay: 0.05
  warmup_epochs: 5
  label_smoothing: 0.1              # 仅多分类生效
  grad_clip: 1.0
  save_ckpt_freq: 10
  early_stopping_patience: 10
  seed: 3407
  num_workers: 8
  # 癫痫二分类优化
  loss_type: "focal"                # 'focal' | 'bce' | 'bce_weighted'
  focal_alpha: 0.25
  focal_gamma: 2.0
  sampler: "weighted"               # 'weighted' | 'random' | 'oversample'
  oversample_factor: 1.0
  threshold_optimization: true
  augment_time_shift: 1.0           # 时间平移增强范围（秒）
  augment_channel_dropout: 0.1      # 通道随机丢弃概率
  augment_noise_std: 0.01           # 高斯噪声标准差

output:
  dir: results/cbramod_chbmit_schemeA
```

> `output.dir` 下自动创建 `tensorboard/`、`best_adapter/`、`latest_adapter/` 等子目录。

### 12.1 启动命令

```bash
# 激活环境
unset LD_LIBRARY_PATH
source activate cbramod

# 训练（CBraMod 方案 A）
python PEFT_engine/main.py --config PEFT_engine/configs/cbramod_chbmit_schemeA.yaml

# 训练（LaBraM 方案 B，需激活 labram 环境）
python PEFT_engine/main.py --config PEFT_engine/configs/labram_chbmit_schemeB.yaml

# 断点续训
python PEFT_engine/main.py --config PEFT_engine/configs/cbramod_chbmit_schemeA.yaml \
    --resume results/cbramod_chbmit_schemeA/latest.pt
```

---

## 13. 实验矩阵

> 对应 `docs/report/LoRA-Schemes.md` §6 的实验设计

### 13.1 主实验 + 消融

| 优先级 | 模型 | 数据集 | 方案 | r 配置 | 配置文件 | 预估参数比 | 角色 |
|--------|------|--------|------|--------|----------|-----------|------|
| P0 | CBraMod | CHB-MIT | A | t=16, s=8, ffn=8 | `cbramod_chbmit_schemeA.yaml` | 3.8% | 主实验 |
| P0 | CBraMod | Siena | A | t=16, s=8, ffn=8 | `cbramod_siena_schemeA.yaml` | 3.8% | 主实验 |
| P0 | LaBraM | CHB-MIT | B | 0-3:4, 4-7:8, 8-11:16 | `labram_chbmit_schemeB.yaml` | 4.2% | 主实验 |
| P0 | LaBraM | Siena | C | attn=16 + LS | `labram_siena_schemeC.yaml` | 2.7% | 主实验 |
| P1 | CBraMod | CHB-MIT | C | ffn=16 | `cbramod_chbmit_schemeC.yaml` | 3.9% | 消融：FFN-only 下限 |
| P1 | LaBraM | CHB-MIT | A | 8 | `labram_chbmit_schemeA.yaml` | 3.6% | 消融：标准 vs 分层 |
| P1 | LaBraM | Siena | A | 8 | `labram_siena_schemeA.yaml` | 3.6% | 消融：标准 vs Attn+LS |
| P2 | CBraMod | CHB-MIT | B | t=16, s=8, ffn=8 | `cbramod_chbmit_schemeB.yaml` | 4.3% | PEFT 生态扩展 |
| P2 | LaBraM | CHB-MIT | C | attn=16 + LS | `labram_chbmit_schemeC.yaml` | 2.7% | 消融：小数据策略 |

### 13.2 超参数消融（基于 P0 主实验）

| 消融变量 | 实验组 | 对照组 |
|---------|--------|--------|
| r 值 | CBraMod A: t=16, s=8 | t=8, s=8 / t=32, s=16 |
| r 值 | LaBraM B: deep=16 | deep=8 / deep=32 |
| Focal gamma | 2.0 | 1.0 / 3.0 |
| Focal alpha | 0.25 | 0.5 / 0.1 |
| 采样策略 | WeightedRandomSampler | 随机采样 / 过采样 2× |
| 数据增强 | 全部开启 | 关闭 / 仅时间平移 |

### 13.3 对照组实验（使用同一代码框架）

| 策略 | 配置差异 | 说明 |
|------|----------|------|
| Full Fine-tuning | `lora: null` | 所有参数可训练，上界参考 |
| Linear Probing | `lora: {frozen: true}` | 仅训练分类头，下界参考 |
| LoRA（本方案） | 各模型 P0 方案 | 冻结 backbone + LoRA + 分类头 |

---

## 14. 未覆盖项（后续扩展）

以下功能不在本次实现范围内，但代码架构应预留接口：

1. **TUSZ 数据集**：待数据获取后实现新的 `Dataset` 子类
2. **其他 PEFT 方法**（AdaLoRA、Prefix Tuning、BitFit）：通过扩展 `apply_lora` 方法支持
3. **多 GPU 分布式训练**：LaBraM 原生支持 DDP，后续可集成
4. **SzCORE Docker 容器化**：最终评估阶段封装
