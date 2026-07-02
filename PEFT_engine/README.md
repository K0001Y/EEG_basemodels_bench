# PEFT Engine

EEG 癫痫发作检测的 LoRA 微调实验框架。支持 **CBraMod** 和 **LaBraM** 两个基础模型，在 **CHB-MIT**、**Siena** 和 **TUSZ** 三个数据集上进行二分类（发作/正常）微调实验。

> 完整规格说明见 [`docs/spec/LoRA.md`](../docs/spec/LoRA.md)

---

## 目录结构

```
PEFT_engine/
├── main.py                    # 统一入口：解析配置 → 构建数据/模型 → 训练
├── trainer.py                 # 训练循环：分组学习率、Cosine warmup、Early Stopping、Checkpoint
├── evaluator.py               # 评估器：PR AUC、ROC AUC、Sens/Spec/F1/Kappa、阈值优化
├── losses.py                  # 损失函数：Focal Loss、Weighted BCE
├── augmentation.py            # 数据增强：时间平移、通道 Dropout、高斯噪声
├── utils.py                   # 工具函数：种子、日志、配置加载、参数统计
├── __init__.py
│
├── configs/                   # 13 个 YAML 实验配置（见实验矩阵）
│   ├── cbramod_chbmit_schemeA.yaml    # P0 主实验
│   ├── cbramod_chbmit_schemeB.yaml    # P2 PEFT 扩展
│   ├── cbramod_chbmit_schemeC.yaml    # P1 消融
│   ├── cbramod_siena_schemeA.yaml     # P0 主实验
│   ├── cbramod_tusz_schemeA.yaml      # P0 TUSZ 主实验
│   ├── cbramod_chbmit_full.yaml       # 对照：全量微调
│   ├── cbramod_chbmit_frozen.yaml     # 对照：Linear Probing
│   ├── labram_chbmit_schemeA.yaml     # P1 消融
│   ├── labram_chbmit_schemeB.yaml     # P0 主实验
│   ├── labram_chbmit_schemeC.yaml     # P2 消融
│   ├── labram_siena_schemeA.yaml      # P1 消融
│   ├── labram_siena_schemeC.yaml      # P0 主实验
│   └── labram_tusz_schemeB.yaml       # P0 TUSZ 主实验
│
├── lora/                      # 自研 LoRA 实现
│   ├── lora_layer.py          #   LoRALayer（低秩分解 A/B）+ LoRALinear（nn.Linear 包装器）
│   ├── lora_mha.py            #   LoRAMultiheadAttention（nn.MHA 包装器，融合 in_proj 上挂 LoRA）
│   └── inject.py              #   inject_lora()（CBraMod 按模块 r）/ inject_lora_layerwise()（LaBraM 按层 r）
│
├── models/                    # 模型适配器
│   ├── base_model.py          #   BaseModelAdapter 抽象接口
│   ├── cbramod_adapter.py     #   CBraMod 适配：构建 backbone + 分类头、Scheme A/B/C
│   └── labram_adapter.py      #   LaBraM 适配：构建 NeuralTransformer + head、Scheme A/B/C
│
├── datasets/                  # 数据集加载器
│   ├── base_dataset.py        #   BaseDataset 抽象基类：pickle 加载、重采样、模型 reshape、采样器
│   ├── chbmit_dataset.py      #   CHB-MIT（chb01-20 训练 / chb21-22 验证 / chb23-24 测试）
│   ├── siena_dataset.py       #   Siena（pn01-10 训练 / pn11-12 验证 / pn13-14 测试）
│   └── tusz_dataset.py        #   TUSZ（train 训练 / dev 验证 / eval 测试，579+53+43 患者）
│
└── preprocessing/             # 离线预处理（EDF → pickle）
    ├── preprocess_chbmit.py   #   CHB-MIT：解析 summary → 10s 窗口切分 → 正类过采样
    ├── preprocess_siena.py    #   Siena：解析 EDF 标注/seizures 文件 → 同格式输出
    └── preprocess_tusz.py     #   TUSZ：解析 csv_bi 标注 → 参考电极计算双极导联 → 重采样 256Hz
```

---

## 快速开始

### 1. 数据预处理

将原始 EDF 文件转换为中间 pickle 格式（每个文件包含一个 10 秒窗口）：

```bash
# CHB-MIT
python PEFT_engine/preprocessing/preprocess_chbmit.py \
    --edf_dir datas/CHB-MIT \
    --output_dir datas/CHB-MIT/processed

# Siena
python PEFT_engine/preprocessing/preprocess_siena.py \
    --edf_dir "datas/Siena Scalp EEG Dataset/siena-scalp-eeg-database-1.0.0/physionet.org/files/siena-scalp-eeg/1.0.0" \
    --output_dir datas/Siena/processed

# TUSZ
python PEFT_engine/preprocessing/preprocess_tusz.py \
    --edf_dir datas/tusz_v2.0.6/edf \
    --output_dir datas/TUSZ/processed
```

输出格式（每个 pickle）：
```python
{'X': ndarray [16, 2560], 'y': int(0|1), 'patient': str}
#  16 通道 × 2560 采样点 (256Hz × 10s)
```

### 2. 训练

```bash
# CBraMod + CHB-MIT + Scheme A（P0 主实验）
unset LD_LIBRARY_PATH
python PEFT_engine/main.py --config PEFT_engine/configs/cbramod_chbmit_schemeA.yaml

# 断点续训
python PEFT_engine/main.py --config PEFT_engine/configs/cbramod_chbmit_schemeA.yaml \
    --resume results/cbramod_chbmit_schemeA/latest.pt
```

### 3. 查看训练进度

```bash
tensorboard --logdir results/cbramod_chbmit_schemeA/tensorboard
```

---

## 配置说明

每个 YAML 配置包含五个部分：

| 部分 | 关键字段 | 说明 |
|------|---------|------|
| `model` | `name`, `pretrained_weights` | 模型选择及预训练权重路径 |
| `lora` | `type`, `scheme`, `r_*` | LoRA 注入方式（`custom`/`peft`/`null`） |
| `dataset` | `name`, `processed_data_dir` | 数据集及预处理文件路径 |
| `train` | `epochs`, `learning_rate`, `loss_type` | 训练超参数 |
| `output` | `dir` | 输出目录（checkpoint、日志、TensorBoard） |

### LoRA 配置模式

| `lora` 值 | 模式 | 可训练参数 |
|-----------|------|-----------|
| `null` | 全量微调（Full Fine-tuning） | 全部参数 |
| `{frozen: true}` | 线性探测（Linear Probing） | 仅分类头 |
| `{type: custom}` | 自研 LoRA（Scheme A / B） | LoRA A/B + 分类头 |
| `{type: peft}` | HuggingFace PEFT（Scheme B / C） | PEFT adapter + 分类头 |

---

## LoRA 方案矩阵

### CBraMod（双路注意力，`nn.MultiheadAttention` 融合 QKV）

| 方案 | 实现方式 | 目标模块 | r 配置 | 说明 |
|------|---------|---------|--------|------|
| **A** (P0) | 自研 `LoRAMultiheadAttention` | `self_attn_t` / `self_attn_s` / `linear1` / `linear2` | t=16, s=8, ffn=8 | 主实验：时间路增强 |
| **B** (P2) | QKV 拆分 + PEFT 库 | `q/k/v_proj` + FFN | t=16, s=8, ffn=8 | PEFT 生态兼容 |
| **C** (P1) | PEFT 库 | `linear1` / `linear2` | ffn=16 | FFN-only 消融 |

### LaBraM（标准注意力，独立 QKV `nn.Linear`）

| 方案 | 实现方式 | 目标模块 | r 配置 | 说明 |
|------|---------|---------|--------|------|
| **A** (P1) | PEFT 库 | `qkv` / `proj` / `fc1` / `fc2` | 统一 r=8 | 标准 PEFT |
| **B** (P0) | 自研分层 LoRA | `qkv` / `proj` / `fc1` / `fc2` | 浅层 4 → 中层 8 → 深层 16 | 深层增强 |
| **C** (P0) | PEFT + LayerScale | `qkv` / `proj` + 解冻 `gamma_1/2` | attn=16 | 注意力 + 缩放 |

---

## 核心模块

### 数据处理管线

```
原始 EDF (256Hz)
    ↓ 预处理脚本（10s 窗口切分 + 正类过采样）
Pickle [16, 2560]
    ↓ Dataset.__getitem__
    ↓   数据增强（时间平移 ±1s / 通道 Dropout 10% / 高斯噪声 σ=0.01）
    ↓   重采样 256Hz → 200Hz: [16, 2000]
    ↓   Reshape: [16, 10, 200]
    ↓   CBraMod: ÷100 归一化 / LaBraM: 不归一化
模型输入 [B, 16, 10, 200]
```

### 训练策略

- **优化器**：AdamW，分组学习率（LoRA 参数 `1e-4`，分类头 `1e-3`）
- **调度器**：CosineAnnealingLR + 线性 warmup（step-level）
- **损失函数**：Focal Loss（α=0.25, γ=2.0），应对极端类别不平衡
- **采样器**：WeightedRandomSampler，按类别逆频率加权
- **早停**：基于验证集 PR AUC，默认 patience=10
- **阈值优化**：训练结束后在验证集上搜索最优 F1 阈值

### 评估指标

| 指标 | 角色 |
|------|------|
| **PR AUC** | 主选择指标（Early Stopping + 模型选择） |
| ROC AUC | 辅助参考 |
| Sensitivity / Specificity | 临床意义指标 |
| F1 / Balanced Accuracy / Kappa | 综合评估 |
| False Alarm Rate | 误报率 |

---

## 输出目录结构

```
results/{experiment_name}/
├── train.log              # 训练日志（控制台同步输出）
├── log.jsonl              # 结构化 JSON Lines（每 epoch 一条）
├── final_results.json     # 最终测试结果（best epoch + 最优阈值 + 全部指标）
├── latest.pt              # 最新 checkpoint（含 optimizer/scheduler/epoch 状态）
├── tensorboard/           # TensorBoard 事件文件
├── best_adapter/          # 最优 adapter 权重
├── latest_adapter/        # 最新 adapter 权重
└── checkpoint_epoch{N}/   # 周期性 checkpoint（每 save_ckpt_freq 个 epoch）
```

---

## 环境依赖

- Python ≥ 3.11
- PyTorch 2.0.1+cu118
- 完整依赖列表见 [`environment.yml`](environment.yml)（conda）或 [`requirements.txt`](requirements.txt)（pip）
- 本地 PEFT 库：`pip install -e external/Pefts/peft-main/`
- **使用前必须 `unset LD_LIBRARY_PATH`**，避免系统 CUDA 库覆盖 PyTorch 自带的 cu118 库
