# CBraMod 与 LaBraM 的 LoRA 微调方案设计（癫痫二分类优化版）

> 基于两个模型的架构特性，参考 EEG-FM-Bench 的实现方法，针对癫痫发作检测（二分类）任务优化

---

## 0. 癫痫检测任务特性分析

### 0.1 任务定义

| 属性 | 说明 |
|------|------|
| 任务类型 | 二分类（segment-level: 0=正常, 1=发作） |
| 数据来源 | CHB-MIT（24 患者）、Siena（14 患者） |
| 窗口长度 | 10 秒（256Hz → 2560 samples） |
| 通道数 | 16（标准双极导联） |

### 0.2 核心挑战

| 挑战 | 具体表现 | 对微调策略的影响 |
|------|---------|-----------------|
| **极端类别不平衡** | 发作段通常 <5% 总数据 | 必须使用 Focal Loss / 加权采样，否则模型退化为全预测负类 |
| **时序模式至关重要** | 发作起始/演化/终止有明显时序特征 | 需增强时间维度的适应能力（CBraMod 时间路、LaBraM 深层） |
| **假阴性代价高** | 漏检发作 → 临床风险 | 评估需关注灵敏度（sensitivity），而非仅看 accuracy |
| **空间模式有参考价值** | 发作可起源于特定脑区并扩散 | 仍需保留空间适应能力，但优先级低于时间 |
| **正类样本稀缺** | Siena 数据集更小，发作段更少 | 需更强的正则化与参数效率 |

### 0.3 优化策略总览

```
任务特性 → 模型特性 → 优化方向

类别不平衡 → Focal Loss + WeightedRandomSampler（通用）
时序关键   → CBraMod 时间路 r↑ / LaBraM 深层 r↑（模型特异）
假阴性高代价 → 评估指标改为 PR AUC + 灵敏度@特异度
正类稀缺   → 数据增强 + 更高 dropout
```

---

## 1. CBraMod LoRA 方案（癫痫优化）

### 1.1 架构约束 + 癫痫任务分析

CBraMod 的双路注意力架构天然契合癫痫检测的时序+空间双维度需求：

| 注意力路径 | 维度 | 功能 | 癫痫检测相关性 | 优化策略 |
|-----------|------|------|---------------|---------|
| `self_attn_t`（时间路） | d//2=100, head=4 | patch 间时间关系 | **高**：发作起始/演化/终止的时序模式 | r=16（增强） |
| `self_attn_s`（空间路） | d//2=100, head=4 | 通道间空间关系 | **中**：发作起源/扩散的空间模式 | r=8（标准） |
| FFN | 200→800→200 | 特征非线性变换 | **中**：特征重组与判别 | r=8（标准） |

**核心洞察**：癫痫发作的判别特征主要体现在时序演化模式上（如棘波、棘慢复合波的时序出现规律），而非单纯的通道空间分布。因此应给时间路分配更大的 LoRA 秩。

**PEFT 兼容性约束**：`nn.MultiheadAttention` 的 `in_proj_weight` 是融合 QKV 参数（shape `[3×100, 100]`），PEFT 库无法直接匹配。需要自研包装器或拆分 QKV。

---

### 方案 A：时间路径增强全量 LoRA（自研 MHA 包装器）

**核心思路**：自研 `LoRAMultiheadAttention` 包装类，对时间路分配 r=16、空间路分配 r=8，实现时序适应最大化。这是癫痫检测的**推荐主实验方案**。

**LoRA 作用位置**：

| 目标 | r 值 | A 矩阵 shape | B 矩阵 shape | 理由 |
|------|------|-------------|-------------|------|
| `self_attn_t.in_proj` | **16** | `[16, 100]` | `[300, 16]` | 时间路增强 |
| `self_attn_t.out_proj` | **16** | `[16, 100]` | `[100, 16]` | 时间路增强 |
| `self_attn_s.in_proj` | 8 | `[8, 100]` | `[300, 8]` | 空间路标准 |
| `self_attn_s.out_proj` | 8 | `[8, 100]` | `[100, 8]` | 空间路标准 |
| `linear1` | 8 | `[8, 200]` | `[800, 8]` | FFN 标准 |
| `linear2` | 8 | `[8, 800]` | `[200, 8]` | FFN 标准 |

**实现要点**：
- 自研 `LoRALayer`（`lora_A` + `lora_B` + scaling = alpha / r）
- `LoRAMultiheadAttention`：包装 `nn.MultiheadAttention`，冻结原始权重，前向时叠加 LoRA 贡献
- `inject_lora()`：通过模块路径正则匹配，原地替换为包装类，支持 per-module r 配置
- 关键：时间路和空间路使用不同的 r 值，需在注入时区分 `self_attn_t` 和 `self_attn_s`

**参数量估算**（12 层）：

| 模块 | r | 每层参数量 | 12 层合计 |
|------|---|-----------|----------|
| attn_t.in_proj (A+B) | 16 | 16×100 + 300×16 = 6,400 | 76,800 |
| attn_t.out_proj (A+B) | 16 | 16×100 + 100×16 = 3,200 | 38,400 |
| attn_s.in_proj (A+B) | 8 | 8×100 + 300×8 = 3,200 | 38,400 |
| attn_s.out_proj (A+B) | 8 | 8×100 + 100×8 = 1,600 | 19,200 |
| linear1 (A+B) | 8 | 8×200 + 800×8 = 8,000 | 96,000 |
| linear2 (A+B) | 8 | 8×800 + 200×8 = 8,000 | 96,000 |
| **LoRA 合计** | | **28,400/层** | **365,200** |
| classifier | - | - | ~30,000 |
| **总可训练** | | | **~395,200** |

总模型参数 ≈ 10.5M，**可训练比例 ≈ 3.8%**

**训练策略**：
- 损失：Focal Loss（gamma=2.0, alpha=0.25）
- 采样：WeightedRandomSampler（类别逆频率加权）
- 学习率：LoRA 参数 1e-4，分类头 1e-3
- 数据增强：时间平移 ±1s、通道随机丢弃 p=0.1

**优点**：
- 时间路 r=16 充分适应发作时序模式，空间路 r=8 控制参数量
- 完整覆盖 QKV + Output + FFN，适应能力最强
- 无需修改模型源码，包装器透明替换
- 已在 EEG-FM-Bench 中验证 MHA 包装器方案可行

**缺点**：
- 需自研 LoRA 代码（约 350 行），不依赖 PEFT 生态
- 需正确处理 per-path r 差异化注入逻辑
- checkpoint 格式为自定义

---

### 方案 B：QKV 拆分 + 时间增强（PEFT 库）

**核心思路**：修改 CBraMod 的 `TransformerEncoderLayer`，将 `nn.MultiheadAttention` 替换为使用独立 `q_proj`/`k_proj`/`v_proj` 的自定义注意力，使 PEFT 库能直接挂载，并对时间路使用更大 r。

**模型改造**：
```python
class SplitMHA(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.num_heads = num_heads

# 预训练权重迁移：将 in_proj_weight 拆分为 q/k/v
def remap_pretrained_weights(state_dict):
    for key in list(state_dict.keys()):
        if key.endswith('in_proj_weight'):
            base = key[:-len('in_proj_weight')]
            w = state_dict.pop(key)  # [3*d, d]
            d = w.shape[1]
            state_dict[base + 'q_proj.weight'] = w[:d]
            state_dict[base + 'k_proj.weight'] = w[d:2*d]
            state_dict[base + 'v_proj.weight'] = w[2*d:]
```

**LoRA 配置**（时间路 QKV 使用 r=16，空间路 QKV 使用 r=8）：

由于 PEFT 库的 `LoraConfig` 不直接支持 per-module r，采用两次 apply 策略：
```python
# 第一次：时间路模块（r=16）
config_t = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.1,
    target_modules=[r"blocks\.\d+\.self_attn_t\.(q|k|v)_proj",
                    r"blocks\.\d+\.self_attn_t\.out_proj"],
    bias="none",
)
model = get_peft_model(model, config_t)

# 第二次：空间路 + FFN（r=8）
config_s = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=[r"blocks\.\d+\.self_attn_s\.(q|k|v)_proj",
                    r"blocks\.\d+\.self_attn_s\.out_proj",
                    "linear1", "linear2"],
    bias="none",
    modules_to_save=["classifier"],
)
model = model.merge_and_unload()  # 合并第一次的 LoRA
model = get_peft_model(model, config_s)  # 再挂第二次
```

> 注：PEFT 多次 apply 的实现需谨慎处理，也可改用自研注入方式实现 per-module r。

**参数量估算**（12 层）：

| 模块 | r | 每层参数量 | 12 层合计 |
|------|---|-----------|----------|
| q/k/v_proj_t (A+B) | 16 | 3×(16×100+100×16) = 9,600 | 115,200 |
| out_proj_t (A+B) | 16 | 16×100+100×16 = 3,200 | 38,400 |
| q/k/v_proj_s (A+B) | 8 | 3×(8×100+100×8) = 4,800 | 57,600 |
| out_proj_s (A+B) | 8 | 8×100+100×8 = 1,600 | 19,200 |
| linear1 (A+B) | 8 | 8,000 | 96,000 |
| linear2 (A+B) | 8 | 8,000 | 96,000 |
| **合计** | | **35,200/层** | **422,400** |

+ classifier ~30K → 总可训练 ~452K，比例 ~4.3%

**训练策略**：同方案 A（Focal Loss + WeightedRandomSampler）

**优点**：
- 直接使用 PEFT 库，生态完整（支持 AdaLoRA 等扩展）
- Q/K/V 独立可挂载，消融实验粒度更细（可分别消融时间/空间 QKV）
- checkpoint 格式标准化

**缺点**：
- 需修改模型源码（`TransformerEncoderLayer`），侵入性高
- 需编写预训练权重重映射函数
- PEFT 多次 apply 实现复杂，或需改用自研注入实现 per-module r
- 改造后模型与原始预训练代码不兼容

---

### 方案 C：FFN-only + 增强训练策略（消融基线）

**核心思路**：完全避开 MHA 的 `in_proj_weight` 问题，只对 FFN 挂载 LoRA（r=16），但通过更强的训练策略补偿注意力未适应的不足。作为**消融对照基线**。

**LoRA 配置**：
```python
LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.1,
    target_modules=["linear1", "linear2"],
    bias="none",
    modules_to_save=["classifier"],
)
```

**参数量估算**（r=16，12 层）：

| 模块 | 每层参数量 | 12 层合计 |
|------|-----------|----------|
| linear1 (A+B) | 16×200 + 800×16 = 16,000 | 192,000 |
| linear2 (A+B) | 16×800 + 200×16 = 16,000 | 192,000 |
| classifier | - | ~30,000 |
| **合计** | **32,000/层** | **414,000** |

比例 ≈ 3.9%

**训练策略**（增强版，补偿注意力缺失）：
- 损失：Focal Loss（gamma=2.0, alpha=0.25）
- 采样：WeightedRandomSampler + 发作段 2× 过采样
- 数据增强：时间平移 ±2s、通道随机丢弃 p=0.15、高斯噪声 σ=0.01
- 学习率：LoRA 5e-5（降低，因 r 增大）、分类头 1e-3

**优点**：
- 实现最简单，PEFT 库一行配置
- 无需修改任何模型源码，无预训练权重迁移
- 通过更强训练策略补偿，可验证"注意力适应是否必要"

**缺点**：
- 注意力层完全未适应，发作时序模式学习能力受限
- 预期性能最低，仅作为消融对照

---

### CBraMod 三方案对比

| 维度 | 方案 A（时间增强自研） | 方案 B（QKV 拆分 PEFT） | 方案 C（FFN-only 消融） |
|------|---------------------|----------------------|---------------------|
| 注意力覆盖 | 时间路 r=16 + 空间路 r=8 | 同左 | 无 |
| FFN 覆盖 | r=8 | r=8 | r=16 |
| 可训练参数 | ~395K (3.8%) | ~452K (4.3%) | ~414K (3.9%) |
| 时间路适应能力 | 最强（r=16） | 最强（r=16） | 无 |
| Focal Loss | 是 | 是 | 是（增强版） |
| 加权采样 | 是 | 是 | 是（2× 过采样） |
| 实现复杂度 | 高 | 中 | 低 |
| 预期 PR AUC | 最优 | 最优 | 中等 |
| 角色 | **主实验** | PEFT 生态扩展 | 消融基线 |

---

## 2. LaBraM LoRA 方案（癫痫优化）

### 2.1 架构约束 + 癫痫任务分析

LaBraM 使用自研 `Attention` 类，QKV 投影为独立 `nn.Linear`，无 MHA 兼容性问题。关键特性：

| 组件 | 说明 | 癫痫检测相关性 |
|------|------|---------------|
| `attn.qkv` | 统一 QKV 投影 `[600, 200]` | 注意力模式适应（patch 间时序关系） |
| `attn.proj` | 输出投影 `[200, 200]` | 注意力输出变换 |
| `mlp.fc1/fc2` | FFN `[200→800→200]` | 特征非线性变换 |
| `gamma_1`/`gamma_2` | LayerScale `[200]` | 控制 attention/FFN 贡献比 |
| 深层 blocks (8-11) | 高层抽象特征 | **发作模式判别的核心层** |

**核心洞察**：LaBraM 的 12 层 Transformer 中，浅层学习通用 EEG 特征（波形、频率），深层学习任务特定抽象（发作模式判别）。癫痫检测应给深层分配更大 r，增强发作判别能力。

**通道对齐**：LaBraM 预训练使用 128 电极位置，CHB-MIT/Siena 仅 16 双极导联。通过 `input_chans` 索引子集或禁用 `pos_embed` 退化处理。

---

### 方案 A：PEFT 标准 + Focal Loss（推荐主实验）

**核心思路**：全量 LoRA（qkv + proj + fc1 + fc2），统一 r=8，配合 Focal Loss + 加权采样。与 EEG-FM-Bench default 配置直接对标，是**最直接的基线方案**。

**LoRA 配置**：
```python
LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.1,  # dropout 提高至 0.1（防过拟合）
    target_modules=["qkv", "proj", "fc1", "fc2"],
    bias="none",
    modules_to_save=["head"],
)
```

**参数量估算**（r=8，12 层）：

| 模块 | 每层参数量 | 12 层合计 |
|------|-----------|----------|
| qkv (A+B) | 8×200 + 600×8 = 6,400 | 76,800 |
| proj (A+B) | 8×200 + 200×8 = 3,200 | 38,400 |
| fc1 (A+B) | 8×200 + 800×8 = 8,000 | 96,000 |
| fc2 (A+B) | 8×800 + 200×8 = 8,000 | 96,000 |
| **LoRA 合计** | **25,600/层** | **307,200** |
| head | - | ~200 |
| **总可训练** | | **~307,400** |

总模型参数 ≈ 8.6M，**可训练比例 ≈ 3.6%**

**训练策略**：
- 损失：Focal Loss（gamma=2.0, alpha=0.25）
- 采样：WeightedRandomSampler（类别逆频率加权）
- 学习率：LoRA 1e-4，head 1e-3
- 数据增强：时间平移 ±1s、通道随机丢弃 p=0.1

**优点**：
- 实现最直接，PEFT 库标准用法
- 与 EEG-FM-Bench default 配置一致，可直接对标
- Attention + FFN 全覆盖，适应能力均衡

**缺点**：
- 所有层使用相同 r=8，未利用层间差异
- 未利用 LayerScale 参数的适应潜力

---

### 方案 B：深层增强分层 LoRA（大数据集最优）

**核心思路**：不同深度层分配不同 LoRA 秩——浅层小 r 学习通用特征，深层大 r 强化发作模式判别。

**分层策略**（癫痫检测优化）：

| 层范围 | 层索引 | r 值 | 理由 |
|--------|--------|------|------|
| 浅层 | 0–3 | 4 | 通用 EEG 特征（波形、频率），小幅适应 |
| 中层 | 4–7 | 8 | 过渡层，特征组合与初步抽象 |
| 深层 | 8–11 | **16** | **发作模式判别核心层，最大适应** |

**实现方式**（自研注入，参考 EEG-FM-Bench）：
```python
def inject_lora_layerwise(model, layer_r_config):
    """
    layer_r_config = {0: 4, 1: 4, 2: 4, 3: 4,
                      4: 8, 5: 8, 6: 8, 7: 8,
                      8: 16, 9: 16, 10: 16, 11: 16}
    """
    for name, module in model.named_modules():
        layer_idx = extract_layer_index(name)
        r = layer_r_config.get(layer_idx, 8)
        if is_target_linear(module, name):
            replace_with_lora(module, r=r, alpha=r*2)
```

**参数量估算**：

| 层范围 | r | 每层参数量 | 4 层合计 |
|--------|---|-----------|---------|
| 0–3 | 4 | 12,800 | 51,200 |
| 4–7 | 8 | 25,600 | 102,400 |
| 8–11 | 16 | 51,200 | 204,800 |
| head | - | - | ~200 |
| **合计** | | | **358,600** |

比例 ≈ 4.2%

**训练策略**：同方案 A（Focal Loss + WeightedRandomSampler）

**优点**：
- 深层 r=16 充分适应发作判别，浅层 r=4 避免过拟合
- 理论上参数效率最优：深层获得更大容量，总参数仅增加 ~50K
- 适合 CHB-MIT 大数据集（~4 万 segments）
- 可消融分析各层贡献

**缺点**：
- 实现复杂度高于方案 A
- 需自研注入逻辑支持 per-layer r
- 分层 r 值需实验调参

---

### 方案 C：注意力优先 + LayerScale + Focal Loss（小数据集最优）

**核心思路**：仅对注意力层（`qkv` + `proj`）挂载 r=16 的 LoRA，同时解冻 LayerScale 参数（`gamma_1`/`gamma_2`），通过 Focal Loss 应对类别不平衡。**适合 Siena 等小数据集**。

**理论依据**：
1. LoRA 原论文表明，注意力层适应通常比 FFN 更重要
2. LayerScale 的 `gamma_1`/`gamma_2` 控制各层 attention/FFN 输出的混合权重，解冻它们等价于让模型自适应调整 FFN 的贡献
3. 癫痫检测中，注意力模式（哪些 patch 是发作段）比特征变换更重要
4. r=16 补偿注意力缺失 FFN 的覆盖，LayerScale 间接适应 FFN

**LoRA 配置**：
```python
LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.1,
    target_modules=["qkv", "proj"],
    bias="none",
    modules_to_save=["head", "gamma_1", "gamma_2"],  # 解冻 LayerScale
)
```

**参数量估算**（r=16，12 层）：

| 模块 | 每层参数量 | 12 层合计 |
|------|-----------|----------|
| qkv (A+B) | 16×200 + 600×16 = 12,800 | 153,600 |
| proj (A+B) | 16×200 + 200×16 = 6,400 | 76,800 |
| gamma_1 | 200 | 2,400 |
| gamma_2 | 200 | 2,400 |
| head | - | ~200 |
| **合计** | **19,600/层** | **235,400** |

比例 ≈ 2.7%（所有方案中最低）

**训练策略**：
- 损失：Focal Loss（gamma=2.0, alpha=0.25）
- 采样：WeightedRandomSampler（类别逆频率加权）
- 学习率：LoRA 5e-5（降低，因 r 增大）、head 1e-3、LayerScale 1e-3
- 数据增强：时间平移 ±1s、通道随机丢弃 p=0.1

**优点**：
- 可训练参数量最小（仅 2.7%），极致参数效率
- 通过 LayerScale 间接适应 FFN，避免完全忽略 FFN
- r=16 的注意力 LoRA 可充分学习 patch 间发作时序模式
- 最适合 Siena 小数据集（14 患者，数据量少）

**缺点**：
- FFN 内部权重完全冻结，特征变换适应能力有限
- 对 CHB-MIT 大数据集可能欠拟合
- LayerScale 解冻效果取决于预训练模型是否使用 LayerScale（LaBraM `init_values=0.1`，已启用）

---

### LaBraM 三方案对比

| 维度 | 方案 A（PEFT 标准） | 方案 B（深层增强分层） | 方案 C（注意力+LayerScale） |
|------|-------------------|---------------------|--------------------------|
| 目标模块 | qkv, proj, fc1, fc2 | 同左，r 按层变化 | qkv, proj + gamma_1/2 |
| 深层 r | 8 | **16** | 16 |
| FFN 覆盖 | r=8 | 浅4/中8/深16 | 无（LayerScale 间接） |
| 可训练参数 | ~307K (3.6%) | ~359K (4.2%) | ~235K (2.7%) |
| Focal Loss | 是 | 是 | 是 |
| 加权采样 | 是 | 是 | 是 |
| 预期 PR AUC | 均衡 | **最优（大数据集）** | 节省（小数据集） |
| 适用数据集 | CHB-MIT + Siena | CHB-MIT（大数据集） | Siena（小数据集） |
| 角色 | **主实验** | 大数据集优化 | 小数据集优化 |

---

## 3. 训练优化策略（癫痫二分类通用）

### 3.1 损失函数

**默认使用 Focal Loss** 替代标准 BCEWithLogitsLoss：

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        """
        alpha: 正类权重（0.25 表示正类损失 ×0.25，因正类已通过采样增强）
        gamma: 聚焦参数（越大越关注难分类样本）
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        return (focal_weight * bce_loss).mean()
```

**参数选择依据**：

| 参数 | 值 | 理由 |
|------|---|------|
| gamma | 2.0 | 标准选择，对易分类的负类降权 |
| alpha | 0.25 | 配合 WeightedRandomSampler，双重平衡 |

**备选**：若 Focal Loss 效果不佳，退回 `BCEWithLogitsLoss(pos_weight=N_neg/N_pos)`。

### 3.2 类别平衡采样

```python
from torch.utils.data import WeightedRandomSampler

# 计算每个样本的权重（逆频率）
class_counts = [num_negative, num_positive]  # 如 [38000, 2000]
class_weights = [1.0 / class_counts[0], 1.0 / class_counts[1]]
sample_weights = [class_weights[label] for label in all_labels]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True,
)
# 传入 DataLoader: DataLoader(dataset, sampler=sampler, ...)
```

**效果**：每个 epoch 中正负类比例接近 1:1，模型不再偏向全预测负类。

### 3.3 数据增强

针对 EEG 信号特性的增强策略：

| 增强方式 | 参数 | 说明 |
|---------|------|------|
| 时间平移 | ±1s（±256 samples） | 随机裁剪窗口位置，模拟发作起始点的位置变化 |
| 通道丢弃 | p=0.1 | 随机置零 1-2 个通道，增强通道缺失鲁棒性 |
| 高斯噪声 | σ=0.01 | 添加微弱噪声，防止过拟合 |
| Mixup | alpha=0.2 | 仅在发作-正常对之间插值，生成过渡样本 |

> 增强仅在训练集应用，验证/测试集不增强。

### 3.4 阈值优化

训练完成后，在验证集上搜索最优决策阈值：

```python
from sklearn.metrics import precision_recall_curve

# 在验证集上计算各阈值下的 precision/recall
precision, recall, thresholds = precision_recall_curve(y_val, y_scores)
f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
best_threshold = thresholds[f1_scores.argmax()]

# 使用 best_threshold 在测试集上评估
y_pred = (y_test_scores > best_threshold).astype(int)
```

**备选策略**：若临床要求更高灵敏度，可选择 sensitivity ≥ 0.8 时的最优阈值。

---

## 4. 评估指标优化

### 4.1 Segment-level（开发阶段，模型选择）

| 指标 | 函数 | 说明 | 角色 |
|------|------|------|------|
| **PR AUC** | `auc(recall, precision)` | 正类精确率-召回率曲线下面积 | **主选择指标**（替代 ROC AUC） |
| ROC AUC | `roc_auc_score` | 阈值无关判别力 | 次要指标 |
| Sensitivity | `recall_score` | 发作段召回率（假阴性代价） | 关键指标 |
| Specificity | `tn / (tn + fp)` | 正常段特异度（假阳性代价） | 关键指标 |
| Balanced Accuracy | `balanced_accuracy_score` | 各类召回率均值 | 次要指标 |
| F1 (binary) | `f1_score` | 发作类 F1 | 次要指标 |
| Cohen's Kappa | `cohen_kappa_score` | 一致性系数 | 次要指标 |

> **主选择指标改为 PR AUC**：在极端类别不平衡场景下，ROC AUC 会高估模型性能（因负类基数大，假阳性率低）。PR AUC 更能反映模型对正类（发作）的实际判别能力。

**额外报告**：
- Sensitivity @ Specificity=0.8（固定特异度下的灵敏度）
- False Alarm Rate = FP / (FP + TN)（假阳性率，越低越好）

### 4.2 Event-level（最终阶段，SzCORE）

训练完成后，将最佳模型接入 SzCORE 评估流程：

1. 将 segment-level 模型包装为 EDF 推理接口（滑动窗口 → 事件聚合）
2. 按 SzCORE 数据流规范输出事件预测（`{onset, offset, confidence}`）
3. 评估指标：event-based F1, OVLP, epoch-based sensitivity/specificity

> 此阶段为训练后的独立评估步骤，不影响训练循环代码。

---

## 5. 方案选择决策

### 5.1 决策树

```
                    ┌─ CBraMod ─────────────────────────────────────┐
                    │                                                │
                    │  是否可修改模型源码？                            │
                    │    ├─ 是 → 方案 B（QKV 拆分 + 时间增强 PEFT）    │
                    │    │   （需 PEFT 生态 / 消融 QKV 粒度）           │
                    │    └─ 否 → 方案 A（时间增强自研 MHA 包装器）★    │
                    │        （推荐主实验，时间路 r=16）                │
                    │                                                │
                    │  消融对照 → 方案 C（FFN-only + 增强训练）          │
                    │                                                │
                    ├─ LaBraM ──────────────────────────────────────┐
                    │                                                │
                    │  数据集规模？                                   │
                    │    ├─ 大 (CHB-MIT) → 方案 B（深层增强分层）★     │
                    │    │   （深层 r=16，发作模式判别最优）             │
                    │    ├─ 中 → 方案 A（PEFT 标准 + Focal Loss）★     │
                    │    │   （均衡覆盖，对标 EEG-FM-Bench）             │
                    │    └─ 小 (Siena) → 方案 C（注意力+LayerScale）     │
                    │        （r=16 注意力 + 解冻 gamma，防过拟合）      │
                    └────────────────────────────────────────────────┘

  ★ = 推荐主实验方案
```

### 5.2 与 EEG-FM-Bench 的对标关系

| EEG-FM-Bench 配置 | 对应方案 | 差异 |
|-------------------|---------|------|
| `target_type: "default"` | CBraMod 方案 A / LaBraM 方案 A | 本项目增加时间路 r 差异化 + Focal Loss |
| `target_type: "attention"` | LaBraM 方案 C | 本项目增加 LayerScale 解冻 + r=16 |
| `target_type: "ffn"` | CBraMod 方案 C | 本项目增加增强训练策略 |
| EEG-FM-Bench 无 | CBraMod 方案 B | 本项目独创：QKV 拆分 + 时间增强 |
| EEG-FM-Bench 无 | LaBraM 方案 B | 本项目独创：深层增强分层 LoRA |
| EEG-FM-Bench 损失 | CrossEntropyLoss | 本项目改用 Focal Loss + WeightedRandomSampler |
| EEG-FM-Bench 选择指标 | ROC AUC | 本项目改用 PR AUC（更适合不平衡数据） |

---

## 6. 实验矩阵

### 6.1 主实验 + 消融

| 优先级 | 模型 | 数据集 | 方案 | r 配置 | 角色 |
|--------|------|--------|------|--------|------|
| P0 | CBraMod | CHB-MIT | A | t=16, s=8, ffn=8 | 主实验 |
| P0 | CBraMod | Siena | A | t=16, s=8, ffn=8 | 主实验 |
| P0 | LaBraM | CHB-MIT | B | 0-3:4, 4-7:8, 8-11:16 | 主实验 |
| P0 | LaBraM | Siena | C | attn=16 + LayerScale | 主实验 |
| P1 | CBraMod | CHB-MIT | C | ffn=16 | 消融：FFN-only 下限 |
| P1 | LaBraM | CHB-MIT | A | 8 | 消融：标准 vs 分层 |
| P1 | LaBraM | Siena | A | 8 | 消融：标准 vs Attention+LS |
| P2 | CBraMod | CHB-MIT | B | t=16, s=8, ffn=8 | PEFT 生态扩展 |
| P2 | LaBraM | CHB-MIT | C | attn=16 + LS | 消融：小数据策略 |

### 6.2 超参数消融（基于 P0 主实验）

| 消融变量 | 实验组 | 对照组 |
|---------|--------|--------|
| r 值 | CBraMod A: t=16, s=8 | t=8, s=8 / t=32, s=16 |
| r 值 | LaBraM B: deep=16 | deep=8 / deep=32 |
| Focal gamma | 2.0 | 1.0 / 3.0 |
| Focal alpha | 0.25 | 0.5 / 0.1 |
| 采样策略 | WeightedRandomSampler | 随机采样 / 过采样 2× |
| 数据增强 | 全部开启 | 关闭 / 仅时间平移 |

### 6.3 统一对照实验

使用同一代码框架（`PEFT_engine/`）运行以下对照：

| 策略 | 配置 | 说明 |
|------|------|------|
| Full Fine-tuning | `lora: null` | 所有参数可训练，上界参考 |
| Linear Probing | `lora: {frozen: true}` | 仅训练分类头，下界参考 |
| LoRA（本方案） | 各模型 P0 方案 | 主实验 |
