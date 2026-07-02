# EEG-FM-Bench LoRA 微调实现分析

> 分析对象：`external/Frames/EEG-FM-Bench-main/`
> 分析范围：CBraMod 与 LaBraM 的 LoRA 微调实现机制

---

## 1. 整体架构

EEG-FM-Bench 采用**工厂模式 + 抽象基类**的统一框架，所有模型共享同一套训练管线和 LoRA 注入逻辑。

```
baseline_main.py  (入口)
    │
    ├── ModelRegistry (工厂注册表)
    │     ├── CBraModConfig / CBraModTrainer / CBraModDataLoaderFactory
    │     └── LabramConfig / LabramTrainer / LabramDataLoaderFactory
    │
    ├── AbstractTrainer (训练基类)
    │     ├── apply_lora()          ← LoRA 注入入口
    │     ├── setup_optim_params() ← 分组学习率 + 冻结
    │     ├── train_epoch()        ← 训练循环
    │     ├── eval_epoch()         ← 评估循环
    │     └── save_lora_checkpoint() ← LoRA 权重保存
    │
    └── baseline/utils/lora.py (自研 LoRA 核心)
          ├── LoRALayer             ← 基础低秩层
          ├── LoRALinear            ← Linear + LoRA 包装
          ├── LoRAMultiheadAttention ← MHA + LoRA 包装
          ├── inject_lora()         ← 模块替换引擎
          └── get/set_lora_state_dict() ← 权重序列化
```

### 关键设计特征

| 特征 | 说明 |
|------|------|
| LoRA 实现方式 | **自研实现**，不依赖 HuggingFace PEFT 库 |
| 配置管理 | Pydantic BaseModel + OmegaConf YAML 合并 |
| 模型注册 | `ModelRegistry` 工厂模式，`__init__.py` 中统一注册 |
| 分布式 | 原生 PyTorch DDP 支持 |
| 混合精度 | `torch.amp.autocast` + `GradScaler` |

---

## 2. 自研 LoRA 核心实现

### 2.1 核心类

文件：`baseline/utils/lora.py`

**LoRALayer**（基础层）：
```python
class LoRALayer(nn.Module):
    # W' = W + B @ A * (alpha / r)
    self.lora_A = nn.Parameter(torch.zeros(r, in_features))    # Kaiming 初始化
    self.lora_B = nn.Parameter(torch.zeros(out_features, r))   # 零初始化
    self.scaling = lora_alpha / r
```

**LoRALinear**（包装 `nn.Linear`）：
- 冻结原始权重（`requires_grad = False`）
- 前向：`base_layer(x) + lora(x)`
- 支持 `merge()` / `unmerge()` 权重合并

**LoRAMultiheadAttention**（包装 `nn.MultiheadAttention`）：
- 冻结原始 `in_proj_weight`、`out_proj.weight` 等
- 为 `in_proj` 添加 LoRA（操作 shape `[3*embed_dim, embed_dim]` 的融合权重）
- 为 `out_proj` 添加 LoRA
- 前向时手动调用 `F.multi_head_attention_forward()` 并叠加 LoRA 贡献

> **关键区别**：不拆分 QKV 为独立 Linear，而是直接在融合的 `in_proj_weight` 上添加 LoRA 低秩矩阵。

### 2.2 注入引擎

`inject_lora()` 函数执行以下步骤：
1. 通过正则表达式匹配目标模块路径
2. 区分 `nn.Linear` 和 `nn.MultiheadAttention` 两种类型
3. 用 `LoRALinear` / `LoRAMultiheadAttention` 原地替换
4. 返回注入的模块路径列表

### 2.3 Scope 过滤

支持两种注入范围：
- `"transformer"`：仅注入 Transformer block 内的模块（通过正则匹配 `blocks.\d+.*`、`layers.\d+.*` 等模式）
- `"full"`：注入所有匹配模块

### 2.4 目标模块注册表

文件中维护 `MODEL_LORA_TARGETS` 字典，为每个模型预定义了不同粒度的目标模块：

```python
MODEL_LORA_TARGETS = {
    "cbramod": {
        "default":   ["self_attn_s", "self_attn_t", "linear1", "linear2"],
        "attention": ["self_attn_s", "self_attn_t"],
        "ffn":       ["linear1", "linear2"],
        "full":      ["self_attn_s", "self_attn_t", "linear1", "linear2"],
    },
    "labram": {
        "default":   ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"],
        "attention": ["attn.qkv", "attn.proj"],
        "ffn":       ["mlp.fc1", "mlp.fc2"],
        "full":      ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"],
    },
}
```

---

## 3. CBraMod 的 LoRA 实现

### 3.1 模型架构

文件：`baseline/cbramod/model.py`

```
CBraModUnifiedModel
├── encoder: CBraMod
│   ├── patch_embedding: PatchEmbedding (Conv2d + FFT spectral + positional encoding)
│   ├── encoder: TransformerEncoder (12 层 TransformerEncoderLayer)
│   │   └── 每层:
│   │       ├── self_attn_s: nn.MultiheadAttention (d_model//2=100, nhead//2=4)
│   │       ├── self_attn_t: nn.MultiheadAttention (d_model//2=100, nhead//2=4)
│   │       ├── linear1: nn.Linear (200→800)
│   │       └── linear2: nn.Linear (800→200)
│   └── proj_out: nn.Linear (200→out_dim)
└── classifier: MultiHeadClassifier
```

### 3.2 LoRA 挂载方式

CBraMod 的注意力使用 `nn.MultiheadAttention`，其 QKV 投影为融合参数 `in_proj_weight`（shape `[3*100, 100]`）。

EEG-FM-Bench 的处理方式：

1. **目标模块名**：`self_attn_s`、`self_attn_t`（直接匹配 MHA 模块本身，而非其子模块）
2. **注入类型**：`LoRAMultiheadAttention`
3. **LoRA 作用位置**：
   - `in_proj`：在融合的 `in_proj_weight [3*embed_dim, embed_dim]` 上添加 LoRA（`lora_A: [r, embed_dim]`，`lora_B: [3*embed_dim, r]`）
   - `out_proj`：在 `out_proj.weight [embed_dim, embed_dim]` 上添加 LoRA
4. **默认还挂载 FFN**：`linear1`、`linear2` 作为 `LoRALinear` 注入

每层注入数量：
| 模块 | 类型 | 每层数量 |
|------|------|----------|
| `self_attn_s` | LoRAMultiheadAttention | 1 |
| `self_attn_t` | LoRAMultiheadAttention | 1 |
| `linear1` | LoRALinear | 1 |
| `linear2` | LoRALinear | 1 |

12 层共注入 48 个 LoRA 模块。

### 3.3 数据适配

文件：`baseline/cbramod/cbramod_adapter.py`

- `scale = 0.01`（数据缩放，等价于 `/100`）
- `patch_size = 200`，`freq = 200`
- 不使用通道映射（`get_supported_channels()` 返回 None，跳过 montage 通道选择）
- 数据在 `CBraModUnifiedModel.forward()` 中 reshape：`[B, C, T]` → `[B, C, T//patch_size, patch_size]`

### 3.4 配置默认值

文件：`baseline/cbramod/cbramod_config.py`

| 参数 | 默认值 |
|------|--------|
| `d_model` | 200 |
| `dim_ffn` | 800 |
| `n_layer` | 12 |
| `n_head` | 8 |
| `max_epochs` | 50 |
| `max_lr` | 1e-4 |
| `lr_schedule` | cosine |
| `warmup_epochs` | 5 |
| `freeze_encoder` | False |
| `lora_r` | 16 (继承自 BaseLoRAArgs) |
| `lora_alpha` | 16 |
| `lora_scope` | transformer |

---

## 4. LaBraM 的 LoRA 实现

### 4.1 模型架构

文件：`baseline/labram/model.py`

```
LabramUnifiedModel
├── encoder: NeuralTransformer
│   ├── patch_embed: TemporalConv (3层 Conv2d + GroupNorm + GELU)
│   ├── cls_token: nn.Parameter [1, 1, 200]
│   ├── pos_embed: nn.Parameter [1, 129, 200] (128 electrodes + 1 cls)
│   ├── time_embed: nn.Parameter [1, 16, 200]
│   ├── blocks: 12 × Block
│   │   └── 每层:
│   │       ├── norm1: LayerNorm
│   │       ├── attn: Attention
│   │       │   ├── qkv: nn.Linear (200→600, bias=False)
│   │       │   └── proj: nn.Linear (200→200)
│   │       ├── norm2: LayerNorm
│   │       ├── mlp: Mlp
│   │       │   ├── fc1: nn.Linear (200→800)
│   │       │   └── fc2: nn.Linear (800→200)
│   │       ├── gamma_1: LayerScale
│   │       └── gamma_2: LayerScale
│   ├── norm: Identity
│   └── fc_norm: LayerNorm
└── classifier: MultiHeadClassifier
```

### 4.2 LoRA 挂载方式

LaBraM 的注意力 QKV 和 FFN 均为独立 `nn.Linear`，直接用 `LoRALinear` 包装：

| 目标模块 | 正则匹配模式 | 说明 |
|----------|-------------|------|
| `attn.qkv` | `.*\.attn\.qkv$` | QKV 联合投影 |
| `attn.proj` | `.*\.attn\.proj$` | 注意力输出投影 |
| `mlp.fc1` | `.*\.mlp\.fc1$` | FFN 升维 |
| `mlp.fc2` | `.*\.mlp\.fc2$` | FFN 降维 |

12 层共注入 48 个 `LoRALinear` 模块。

### 4.3 通道对齐

文件：`baseline/labram/labram_adapter.py`

LaBraM 支持通道映射：
- `get_supported_channels()` 返回 128+ 个标准电极位置（含双极导联）
- 通过 `AbstractDatasetAdapter._build_montage_mappings()` 自动匹配数据集通道到模型支持的通道
- `LabramUnifiedModel.forward()` 中使用 `input_chans` 参数索引 `pos_embed`：
  ```python
  chans_id = nn.functional.pad(chans_id + 1, (1, 0), value=0)
  features = self.encoder.forward_features(data, input_chans=chans_id, ...)
  ```

### 4.4 预训练权重加载

```python
# labram_trainer.py load_checkpoint()
checkpoint = torch.load(checkpoint_path, ...)
for k, v in checkpoint['model'].items():
    if k.startswith('student.'):
        encoder_state_dict[k[len('student.'):]] = v
self.encoder.load_state_dict(encoder_state_dict, strict=False)
```

### 4.5 配置默认值

文件：`baseline/labram/labram_config.py`

| 参数 | 默认值 |
|------|--------|
| `embed_dim` | 200 |
| `depth` | 12 |
| `num_heads` | 10 |
| `mlp_ratio` | 4.0 |
| `patch_size` | 200 |
| `eeg_size` | 2000 |
| `max_epochs` | 30 |
| `max_lr` | 8e-4 |
| `lr_schedule` | cosine |
| `freeze_encoder` | True（默认冻结） |
| `label_smoothing` | 0.1 |
| `layer_decay` | 0.9 |

---

## 5. LoRA 配置系统

### 5.1 配置结构

文件：`baseline/abstract/config.py`

```python
class BaseLoRAArgs(BaseModel):
    use_lora: bool = False
    lora_r: int = 16              # 默认 r=16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: List[str] = ["default"]  # "default" 触发模型特定默认值
    lora_exclude_modules: Optional[List[str]] = None
    lora_target_type: str = "default"  # default / full / attention / ffn
    lora_scope: str = "transformer"   # transformer / full
    lora_lr_scale: float = 1.0        # LoRA lr 相对于 head lr 的缩放
```

### 5.2 目标模块解析流程

```
lora_target_modules = ["default"]
    ↓
get_lora_target_modules()
    ↓ (lora_target_modules == ["default"])
get_model_lora_targets(model_type, lora_target_type)
    ↓
MODEL_LORA_TARGETS["cbramod"]["default"]
    → ["self_attn_s", "self_attn_t", "linear1", "linear2"]
```

如果用户在配置中显式指定 `lora_target_modules`（非 `["default"]`），则直接使用用户值。

### 5.3 YAML 配置示例

```yaml
# cbramod_unified.yaml
training:
  lora:
    use_lora: false              # 默认关闭
    lora_r: 8
    lora_alpha: 16
    lora_dropout: 0.0
    lora_target_modules: ['default']
    lora_exclude_modules: null
    lora_target_type: 'default'
    lora_scope: 'transformer'
    lora_lr_scale: 1.0
```

---

## 6. 训练流程分析

### 6.1 训练入口

```
trainer.run()
    ├── seed_torch()
    ├── setup_distributed()
    ├── setup_logging() + init_cloud_logging()
    ├── collect_dataset_info()
    ├── setup_model()
    │     ├── 构建 encoder + classifier
    │     ├── load_checkpoint() (预训练权重)
    │     ├── apply_lora() (注入 LoRA)
    │     ├── model.to(device)
    │     └── maybe_wrap_ddp()
    ├── setup_optimizer_and_scheduler()
    │     └── setup_optim_params() (分组学习率)
    └── run_unified_training() / run_separate_training()
          └── for epoch in range(max_epochs):
                ├── train_epoch()     (训练)
                ├── eval_epoch('eval') (验证)
                ├── eval_epoch('test') (测试)
                └── save_checkpoint() (定期)
```

### 6.2 优化器分组策略

文件：`baseline/abstract/trainer.py` → `setup_optim_params()`

| 参数组 | 识别规则 | 学习率 |
|--------|----------|--------|
| LoRA 参数 | 参数名含 `lora_A` 或 `lora_B` | `max_lr * lora_lr_scale` |
| 分类头参数 | 参数名含 `classifier` 或 `conv_router` | `max_lr` |
| Encoder 参数 | 其余参数 | LoRA 模式下冻结；非 LoRA 模式下 `max_lr * encoder_lr_scale` |

### 6.3 学习率调度

支持两种调度器：
- `onecycle`：OneCycleLR（step-level）
- `cosine`：LinearLR warmup → CosineAnnealingLR（step-level，SequentialLR 组合）

### 6.4 评估指标

`eval_epoch()` 计算的指标：

| 指标 | 适用场景 | 函数 |
|------|----------|------|
| Accuracy | 所有任务 | `(pred == label).mean()` |
| Balanced Accuracy | 所有任务 | `balanced_accuracy_score` |
| ROC AUC | 二分类 | `roc_auc_score` |
| PR AUC | 二分类 | `average_precision_score` |
| Cohen's Kappa | 多分类 | `cohen_kappa_score` |
| F1 (weighted) | 多分类 | `f1_score(average='weighted')` |

### 6.5 Checkpoint 策略

每次保存包含两部分：

**完整 checkpoint**（`{model_type}_{ds_name}_{suffix}.pt`）：
```python
{
    'epoch': int,
    'step': int,
    'model_state_dict': model.state_dict(),  # 含 LoRA 权重
    'optimizer_state_dict': ...,
    'scaler_state_dict': ...,
    'config': cfg.model_dump(),
}
```

**LoRA-only checkpoint**（`{model_type}_{ds_name}_{suffix}_lora.pt`）：
```python
# 仅含 lora_A 和 lora_B 参数
{"encoder.encoder.layers.0.self_attn_s.lora_in_proj.lora_A": tensor, ...}
```

---

## 7. 与本项目 LoRA Spec 的对比

| 维度 | EEG-FM-Bench 实现 | 本项目 LoRA Spec |
|------|-------------------|-----------------|
| **LoRA 库** | 自研实现（`baseline/utils/lora.py`） | HuggingFace PEFT 库 |
| **CBraMod QKV 处理** | `LoRAMultiheadAttention` 包装，在融合 `in_proj_weight` 上加 LoRA | 方案 B：拆分为独立 `q_proj/k_proj/v_proj` |
| **默认 r** | 16 | 8 |
| **默认 alpha** | 16 | 16 |
| **LoRA dropout** | 0.0 | 0.05 |
| **损失函数** | `CrossEntropyLoss`（统一，二分类也用 2 类） | 二分类 `BCEWithLogitsLoss`，多分类 `CrossEntropyLoss` |
| **Early Stopping** | 无 | patience=10 |
| **Best model 选择** | 无（仅定期 + 最后一次 checkpoint） | 按 ROC AUC 保存最佳 |
| **分类头** | `MultiHeadClassifier`（多数据集多头部，支持 5 种 head 类型） | 单数据集单头部 |
| **分组学习率** | LoRA: `max_lr * lora_lr_scale`；Head: `max_lr` | LoRA: `1e-4`；Head: `1e-3` |
| **调度器** | OneCycleLR / Cosine+Warmup（step-level） | CosineAnnealing+Warmup（step-level） |
| **Checkpoint 格式** | 自定义（full + LoRA-only 双保存） | PEFT 原生格式（`adapter_model.bin` + `adapter_config.json`） |
| **分布式** | 原生 DDP | 未涉及（预留接口） |
| **混合精度** | `torch.amp` + bf16 | 未指定 |
| **Scope 过滤** | 支持 transformer/full 两种范围 | 未涉及（全部注入） |
| **通道对齐** | LaBraM 支持完整通道映射 + `input_chans` 索引 | 提到但未详细定义 |
| **数据集** | TUAB/SEED/HMC 等（HuggingFace datasets 格式） | CHB-MIT/Siena（EDF + 自定义 pickle） |

---

## 8. 可借鉴的关键设计

### 8.1 自研 LoRA 对 MHA 的处理

EEG-FM-Bench 的 `LoRAMultiheadAttention` 直接在 `nn.MultiheadAttention` 的融合 `in_proj_weight` 上添加 LoRA，无需拆分模块结构。这比我们 spec 中的"方案 B（拆分 QKV）"更轻量，且前向计算保持与原始 MHA 一致。

**建议**：如果自研 LoRA，可参考此方案；如果使用 PEFT 库，仍需方案 B 或降级方案 A。

### 8.2 Scope 过滤机制

通过正则匹配模块路径，只注入 Transformer block 内的模块，避免在 embedding 或 head 层误注入。

### 8.3 双重 Checkpoint 保存

同时保存完整模型状态和 LoRA-only 状态，灵活性高：
- 完整状态用于断点续训（含 optimizer/scheduler）
- LoRA-only 状态用于模型部署或跨实验迁移

### 8.4 多目标模块预设

`MODEL_LORA_TARGETS` 为每个模型预定义了 `default/attention/ffn/full` 四种粒度，通过配置切换，便于消融实验。

### 8.5 参数分组自动化

通过参数名中的 `lora_A`/`lora_B` 关键字自动识别 LoRA 参数，无需手动维护参数列表。

---

## 9. 潜在问题与注意事项

1. **LoRA 默认关闭**：YAML 配置中 `use_lora: false`，需手动开启
2. **r=16 偏大**：EEG-FM-Bench 默认 r=16，实际实验可能需要调小
3. **无 Early Stopping**：固定训练 max_epochs 轮，可能浪费时间或欠拟合
4. **无 Best Model 选择**：只有定期 checkpoint，需手动选择最佳
5. **二分类用 CrossEntropyLoss**：二分类也使用 2 类 CE，而非 BCEWithLogitsLoss
6. **CBraMod 不做通道映射**：`get_supported_channels()` 返回 None，依赖数据预处理对齐通道
7. **LaBraM pos_embed 固定 128+1**：预训练使用 128 通道，微调时通过 `input_chans` 索引子集
