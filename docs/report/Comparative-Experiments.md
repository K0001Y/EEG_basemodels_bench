# 对比实验报告：多数据集多模型 LoRA 微调消融实验

> 基于 3 个癫痫 EEG 数据集（CHB-MIT、Siena、TUSZ）× 2 个基础模型（CBraMod、LaBraM）的 LoRA 微调对比实验设计与预期分析

---

## 1. 实验总览

### 1.1 实验目标

本项目旨在系统性评估 EEG 基础模型在癫痫发作检测（二分类）任务上的 LoRA 微调效果，核心研究问题包括：

1. **LoRA 方案对比**：不同 LoRA 注入策略（注意力增强 / 分层秩 / FFN-only）对癫痫检测性能的影响
2. **跨模型对比**：CBraMod（双路注意力）与 LaBraM（标准 ViT）在相同数据下的微调表现差异
3. **跨数据集泛化**：同一微调方案在不同规模、不同来源数据集上的稳定性
4. **参数效率分析**：LoRA 微调 vs 全量微调 vs Linear Probing 的性能-参数比

### 1.2 实验矩阵

共设计 **11 组主实验 + 消融** 和 **2 组对照实验**，覆盖 13 个配置：

| # | 模型 | 数据集 | 方案 | 可训练参数比 | 配置文件 | 角色 |
|---|------|--------|------|-------------|---------|------|
| 1 | CBraMod | CHB-MIT | A | 3.8% | `cbramod_chbmit_schemeA` | P0 主实验 |
| 2 | CBraMod | Siena | A | 3.8% | `cbramod_siena_schemeA` | P0 主实验 |
| 3 | CBraMod | TUSZ | A | 3.8% | `cbramod_tusz_schemeA` | P0 主实验 |
| 4 | LaBraM | CHB-MIT | B | 4.2% | `labram_chbmit_schemeB` | P0 主实验 |
| 5 | LaBraM | Siena | C | 2.7% | `labram_siena_schemeC` | P0 主实验 |
| 6 | LaBraM | TUSZ | B | 4.2% | `labram_tusz_schemeB` | P0 主实验 |
| 7 | CBraMod | CHB-MIT | C | 3.9% | `cbramod_chbmit_schemeC` | P1 消融 |
| 8 | LaBraM | CHB-MIT | A | 3.6% | `labram_chbmit_schemeA` | P1 消融 |
| 9 | LaBraM | Siena | A | 3.6% | `labram_siena_schemeA` | P1 消融 |
| 10 | CBraMod | CHB-MIT | B | 4.3% | `cbramod_chbmit_schemeB` | P2 消融 |
| 11 | LaBraM | CHB-MIT | C | 2.7% | `labram_chbmit_schemeC` | P2 消融 |
| 12 | CBraMod | CHB-MIT | Full FT | 100% | `cbramod_chbmit_full` | 上界对照 |
| 13 | CBraMod | CHB-MIT | Frozen | <0.5% | `cbramod_chbmit_frozen` | 下界对照 |

---

## 2. 数据集对比

### 2.1 数据集概况

| 属性 | CHB-MIT | Siena Scalp EEG | TUSZ v2.0.6 |
|------|---------|-----------------|-------------|
| 患者数 | 24 | 14 | 675 (579+53+43) |
| 记录数 | ~800+ EDF | ~50+ EDF | ~8140 EDF |
| 数据规模 | 中等 | 小 | 大 |
| 原始采样率 | 256 Hz | 512 Hz | 250 Hz (LE) / 400 Hz (AR) |
| 通道格式 | 双极导联（16 通道） | 双极导联（16 通道） | 参考电极（需计算双极导联） |
| 标注格式 | `.seizures` 文本文件 | `.seizures` 文本文件 | `.csv_bi`（二分类）/ `.csv`（多类） |
| 划分方式 | 按患者 (20/2/2) | 按患者 | 原始划分 (train/dev/eval) |
| 类别不平衡 | 发作段 <5% | 发作段 <5%（更严重） | 发作段占比低（大规模但分散） |
| 预处理特殊性 | 标准 | 标准 | 重采样 256Hz + 参考→双极计算 + 多 montage 去重 |

### 2.2 数据集差异对微调的影响

| 影响因素 | CHB-MIT | Siena | TUSZ |
|---------|---------|-------|------|
| 训练数据量 | 中（~4 万 segments） | 少（~1 万 segments） | 大（预计 >10 万 segments） |
| 过拟合风险 | 中等 | **高**（数据少 + 患者少） | 低（数据充足） |
| 泛化性测试 | 标准 | 最严苛（患者差异大） | 最可靠（患者群体多样） |
| 标注噪声 | 低（人工精确标注） | 低 | 中等（自动 + 人工混合标注） |
| 通道对齐难度 | 低（原生双极） | 低（原生双极） | 高（需从参考电极计算） |

---

## 3. 模型架构对比

### 3.1 架构特性

| 属性 | CBraMod | LaBraM |
|------|---------|--------|
| 架构类型 | 双路注意力（时间路 + 空间路） | 标准 ViT（自研 Attention） |
| 总参数量 | ~10.5M | ~8.6M |
| 注意力机制 | `nn.MultiheadAttention`（融合 QKV） | 自研 `Attention`（独立 QKV Linear） |
| 注意力路径 | `self_attn_t`（时间）+ `self_attn_s`（空间） | 统一 `attn.qkv` + `attn.proj` |
| FFN | `linear1` (200→800) + `linear2` (800→200) | `mlp.fc1` (200→800) + `mlp.fc2` (800→200) |
| 特殊组件 | 双路分治 | LayerScale (`gamma_1`/`gamma_2`) |
| 位置编码 | 可学习 | 无绝对位置编码 |
| 分类头 | `classifier` (all_patch_reps) | `head` |
| Transformer 深度 | 12 层 | 12 层 |
| 注意力头 | 8 头, d_head=25 (d=200, 分两路各 100) | 10 头, d_head=20 (d=200) |

### 3.2 架构对 LoRA 注入的影响

| 方面 | CBraMod | LaBraM |
|------|---------|--------|
| PEFT 库兼容性 | **不兼容**（`in_proj_weight` 融合 QKV） | **完全兼容**（独立 Linear） |
| 注入方式 | 需自研 `LoRAMultiheadAttention` 包装器 | PEFT 库直接挂载 |
| 差异化注入 | 支持：时间路 / 空间路可用不同 r | 支持：按层深度可用不同 r |
| 最优差异化维度 | **路径维度**（时间 vs 空间） | **深度维度**（浅层 vs 深层） |

---

## 4. LoRA 方案消融详解

### 4.1 CBraMod 三方案对比

CBraMod 的消融围绕 **双路注意力的差异化秩配置** 展开：

| 维度 | Scheme A（主实验） | Scheme B（P2 扩展） | Scheme C（消融基线） |
|------|-------------------|-------------------|-------------------|
| **核心思路** | 自研 MHA 包装器，per-path r | QKV 拆分 + PEFT 库 | 仅 FFN，注意力冻结 |
| **时间路 attn** | r=16 | r=16 | **冻结** |
| **空间路 attn** | r=8 | r=8 | **冻结** |
| **FFN** | r=8 | r=8 | r=16 |
| **LoRA 实现** | 自研（~350 行） | PEFT 库 | PEFT 库 |
| **模型源码修改** | 否（透明包装） | 是（拆分 MHA） | 否 |
| **可训练参数** | ~395K (3.8%) | ~452K (4.3%) | ~414K (3.9%) |
| **学习率** | 1e-4 | 1e-4 | 5e-5（降低） |
| **采样策略** | WeightedRandomSampler | WeightedRandomSampler | 过采样 2× |
| **数据增强** | 时间平移 ±1s, 通道丢弃 0.1 | 时间平移 ±1s, 通道丢弃 0.1 | **增强版**：平移 ±2s, 丢弃 0.15 |

**消融逻辑**：

- **A vs C**：验证注意力适应是否对癫痫检测必要。C 的注意力完全冻结，仅通过增强训练策略补偿 → 若 A 显著优于 C，证明时间路/空间路适应是关键
- **A vs B**：验证自研包装器 vs PEFT 库 QKV 拆分的性能差异。理论上两者等效（r 配置相同），差异来自实现细节
- **C 的增强训练**：增大时间平移范围（±2s vs ±1s）、通道丢弃概率（0.15 vs 0.1）、采用 2× 过采样、降低学习率（5e-5 vs 1e-4），补偿注意力适应缺失

### 4.2 LaBraM 三方案对比

LaBraM 的消融围绕 **层间差异化秩配置** 展开：

| 维度 | Scheme A（P1 消融） | Scheme B（主实验） | Scheme C（小数据优化） |
|------|-------------------|-------------------|---------------------|
| **核心思路** | PEFT 标准，统一 r=8 | 分层 LoRA（浅→深递增） | 仅注意力 + LayerScale |
| **浅层 (0-3) r** | 8 | **4** | 16 |
| **中层 (4-7) r** | 8 | **8** | 16 |
| **深层 (8-11) r** | 8 | **16** | 16 |
| **目标模块** | qkv, proj, fc1, fc2 | qkv, proj, fc1, fc2 | **仅** qkv, proj |
| **FFN 覆盖** | r=8 | 分层 4/8/16 | **冻结**（LayerScale 间接适应） |
| **LayerScale** | 冻结 | 冻结 | **解冻** (`gamma_1`/`gamma_2`) |
| **LoRA 实现** | PEFT 库 | 自研注入 | PEFT 库 |
| **可训练参数** | ~307K (3.6%) | ~359K (4.2%) | ~235K (**2.7%**, 最少) |
| **学习率** | 1e-4 | 1e-4 | 5e-5 |

**消融逻辑**：

- **B vs A**（CHB-MIT 上）：验证分层 r 是否优于统一 r。B 给深层更大 r，假设深层是发作判别核心 → 若 B 优于 A，证明分层策略有效
- **C vs A**（Siena 上）：验证注意力优先 + LayerScale 是否在小数据上优于全覆盖。C 参数最少（2.7%），通过 LayerScale 间接适应 FFN → 若 C 在 Siena 上不逊于 A，证明极致参数效率可行
- **C 在 CHB-MIT**（P2）：验证小数据策略在大数据上是否欠拟合 → 预期 C 在 CHB-MIT 上弱于 B

### 4.3 方案间核心差异总结

```
CBraMod 消融轴：路径维度（时间路 vs 空间路 vs 无注意力）
    A（时间增强） ←主实验→ C（FFN-only 消融）
    A（自研包装器） ←实现对比→ B（PEFT QKV 拆分）

LaBraM 消融轴：深度维度（统一 r vs 分层 r vs 注意力优先）
    B（分层增强） ←主实验→ A（统一 r 消融）
    C（注意力+LS） ←小数据主实验→ A（统一 r 消融）
```

---

## 5. 跨数据集对比实验

### 5.1 同一模型 + 同一方案 → 不同数据集

这组对比验证 **数据集特性** 对微调效果的影响：

| 实验组 | 模型 | 方案 | 数据集 | 核心变量 |
|--------|------|------|--------|---------|
| G1 | CBraMod | A | CHB-MIT | 数据量：中 |
| G2 | CBraMod | A | Siena | 数据量：小 |
| G3 | CBraMod | A | TUSZ | 数据量：大 |
| G4 | LaBraM | B | CHB-MIT | 数据量：中 |
| G5 | LaBraM | B | TUSZ | 数据量：大 |
| G6 | LaBraM | C/A | Siena vs CHB-MIT | 数据量：小 vs 中 |

**预期效果**：

- **G1 vs G2**（CBraMod A：CHB-MIT vs Siena）：Siena 数据少 → 预期性能低于 CHB-MIT，但 Scheme A 的时间路增强可能在 Siena 上仍有效；过拟合风险更高
- **G1 vs G3**（CBraMod A：CHB-MIT vs TUSZ）：TUSZ 数据充足 → 预期性能最优；TUSZ 患者群体更多样 → 泛化性可能更好，但也可能因标注噪声略降
- **G4 vs G5**（LaBraM B：CHB-MIT vs TUSZ）：Scheme B 的深层增强在大数据上更能发挥 → TUSZ 上预期性能最佳
- **G6**（LaBraM C vs A：Siena vs CHB-MIT）：Scheme C 为小数据优化 → 预期在 Siena 上相对优势更明显

### 5.2 数据集规模与方案匹配

| 数据集 | 规模 | CBraMod 最优方案（预期） | LaBraM 最优方案（预期） |
|--------|------|------------------------|------------------------|
| CHB-MIT | 中 | A（时间增强） | B（分层增强） |
| Siena | 小 | A（时间增强） | C（注意力+LS，防过拟合） |
| TUSZ | 大 | A（时间增强） | B（分层增强，充分利用数据） |

---

## 6. 跨模型对比实验

### 6.1 同一数据集 → 不同模型的主实验对比

| 数据集 | CBraMod 主实验 | LaBraM 主实验 | 对比焦点 |
|--------|---------------|--------------|---------|
| CHB-MIT | Scheme A (t=16, s=8, ffn=8) | Scheme B (分层 4/8/16) | 双路差异化 vs 分层差异化 |
| Siena | Scheme A (t=16, s=8, ffn=8) | Scheme C (attn=16 + LS) | 路径增强 vs 注意力+LayerScale |
| TUSZ | Scheme A (t=16, s=8, ffn=8) | Scheme B (分层 4/8/16) | 大数据上两种优化路径的极限 |

### 6.2 模型架构差异对消融的影响

**CBraMod 独有的消融维度——路径对比**：

CBraMod 的双路架构允许独立消融时间路和空间路。Scheme A 给时间路 r=16、空间路 r=8，而 Scheme C 完全冻结注意力。这组对比能回答：**癫痫检测中，注意力适应贡献多少？时间路 vs 空间路哪个更重要？**

**LaBraM 独有的消融维度——深度对比**：

LaBraM 的 12 层 Transformer 允许按深度分层消融。Scheme B 的浅层 r=4 vs 深层 r=16，而 Scheme A 统一 r=8。这组对比能回答：**癫痫检测的判别特征主要集中在浅层还是深层？**

### 6.3 可训练参数效率对比

| 实验 | 模型 | 可训练参数 | 参数比 | 预期性能排名 |
|------|------|-----------|--------|------------|
| CBraMod A | CBraMod | ~395K | 3.8% | 高 |
| CBraMod C | CBraMod | ~414K | 3.9% | 低（注意力冻结） |
| CBraMod B | CBraMod | ~452K | 4.3% | 与 A 接近 |
| LaBraM A | LaBraM | ~307K | 3.6% | 中 |
| LaBraM B | LaBraM | ~359K | 4.2% | 高（大数据集） |
| LaBraM C | LaBraM | ~235K | **2.7%** | 中（小数据集优） |

> **参数效率洞察**：LaBraM Scheme C 仅用 2.7% 参数（235K），是参数最少的方案，但在小数据集上预期不逊色于全量覆盖方案。CBraMod Scheme A 以 3.8% 参数（395K）实现路径差异化，预期为 CBraMod 最优方案。

---

## 7. 对照实验：LoRA vs Full FT vs Linear Probing

### 7.1 对照组设计

| 策略 | `lora` 配置 | 可训练参数 | 含义 |
|------|------------|-----------|------|
| **Full Fine-tuning** | `lora: null` | ~10.5M (100%) | 所有参数可训练，**上界参考** |
| **LoRA（主方案）** | 各 P0 方案 | ~300K–400K (3-4%) | 冻结 backbone + LoRA + 分类头 |
| **Linear Probing** | `lora: {frozen: true}` | ~30K (<0.5%) | 仅训练分类头，**下界参考** |

### 7.2 对照组差异要点

| 维度 | Full FT | LoRA | Linear Probing |
|------|---------|------|----------------|
| backbone 参数 | 可训练 | **冻结** | 冻结 |
| LoRA 参数 | 无 | **可训练** | 无 |
| 分类头 | 可训练 | 可训练 (`modules_to_save`) | 可训练 |
| 过拟合风险 | **最高**（参数多） | 低（参数少） | 最低 |
| 灾难性遗忘 | **可能** | 避免 | 避免 |
| 训练速度 | 最慢 | 中等 | 最快 |
| 显存占用 | 最高 | 中等 | 最低 |
| 预期性能 | 上界 | **接近上界** | 下界 |

### 7.3 对照实验回答的核心问题

- **LoRA vs Full FT**：LoRA 以 ~3-4% 参数能否接近全量微调性能？差距多大？
- **LoRA vs Linear Probing**：LoRA 的参数高效适应相比仅训练分类头，性能提升多少？
- **Full FT vs Linear Probing**：预训练 backbone 的特征质量如何？冻结特征是否已经足够好？

---

## 8. 训练策略差异分析

### 8.1 通用训练配置

所有实验共享以下通用配置：

| 参数 | 值 | 说明 |
|------|---|------|
| Epochs | 50 | 含 Early Stopping (patience=10) |
| Batch Size | 64 | |
| Weight Decay | 0.05 | |
| Warmup Epochs | 5 | |
| Grad Clip | 1.0 | |
| 损失函数 | Focal Loss (γ=2.0, α=0.25) | 应对类别不平衡 |
| 阈值优化 | 验证集最优 F1 阈值 | |
| 随机种子 | 3407 | 保证可复现 |

### 8.2 方案间训练策略差异

| 训练参数 | CBraMod A | CBraMod C | LaBraM A/B | LaBraM C |
|---------|-----------|-----------|------------|----------|
| LoRA 学习率 | 1e-4 | **5e-5** | 1e-4 | **5e-5** |
| 分类头学习率 | 1e-3 | 1e-3 | 1e-3 | 1e-3 |
| 采样策略 | Weighted | **Oversample 2×** | Weighted | Weighted |
| 时间平移增强 | ±1s | **±2s** | ±1s | ±1s |
| 通道丢弃增强 | 0.1 | **0.15** | 0.1 | 0.1 |
| 噪声增强 | σ=0.01 | σ=0.01 | σ=0.01 | σ=0.01 |

**差异原因**：

- **Scheme C（CBraMod FFN-only / LaBraM Attn-only）降低学习率**：r 值更大（16 vs 8），降低 lr 避免训练不稳定
- **CBraMod Scheme C 增强训练策略**：注意力未适应 → 通过更强增强（更大平移、更多通道丢弃、2× 过采样）弥补特征表达能力不足
- **LaBraM Scheme C 额外解冻 LayerScale**：通过 `gamma_1`/`gamma_2` 自适应调整 FFN 贡献，间接补偿 FFN 冻结

---

## 9. 评估指标体系

### 9.1 主选择指标

| 指标 | 角色 | 说明 |
|------|------|------|
| **PR AUC** | **主选择指标** | 极端不平衡下比 ROC AUC 更能反映正类判别能力 |
| ROC AUC | 次要 | 阈值无关判别力 |
| Sensitivity | 关键 | 发作召回率（假阴性代价高） |
| Specificity | 关键 | 正常段特异度 |
| Balanced Accuracy | 次要 | 各类召回率均值 |
| F1 (binary) | 次要 | 发作类 F1 |
| Cohen's Kappa | 次要 | 一致性系数 |
| Sensitivity @ Specificity=0.8 | 临床 | 固定特异度下的灵敏度 |

### 9.2 预期指标排序

基于方案设计和数据集特性，预期各实验在主指标（PR AUC）上的相对排序：

**CHB-MIT 数据集**（中等规模，标注质量高）：
```
Full FT > LoRA 方案 A/B ≈ LoRA 方案 B/A > LoRA 方案 C/C > Linear Probing
         (CBraMod/LaBraM)  (LaBraM/CBraMod)  (消融方案)
```

**Siena 数据集**（小规模，过拟合风险高）：
```
LoRA 主方案（A/C）> Full FT（可能过拟合） > LoRA 消融方案（A） > Linear Probing
```

**TUSZ 数据集**（大规模，患者多样）：
```
LoRA 主方案（A/B）≈ Full FT > Linear Probing
```

---

## 10. 预期结果分析

### 10.1 消融实验预期结论

| 对比 | 预期结果 | 理由 |
|------|---------|------|
| CBraMod A > C (CHB-MIT) | **显著优于** | 注意力适应（尤其时间路 r=16）对时序发作模式至关重要；C 仅 FFN 适应 + 增强训练不足以弥补 |
| CBraMod A ≈ B (CHB-MIT) | **性能接近** | 两者 r 配置等效，差异仅在实现方式（自研 vs PEFT 库）；微小差异来自 QKV 融合 vs 拆分的数值精度 |
| LaBraM B > A (CHB-MIT) | **优于** | 分层策略给深层 r=16 更多适应容量，而浅层 r=4 避免过拟合；统一 r=8 无法区分层间重要性 |
| LaBraM C vs A (Siena) | **C 不逊色或略优** | Siena 数据少 → C 参数最少（2.7%）+ LayerScale 解冻 → 过拟合风险最低；A 全覆盖在小数据上可能过拟合 |
| LaBraM C < B (CHB-MIT) | **弱于** | C 的 FFN 冻结在大数据上欠拟合；B 的全覆盖 + 分层适应更适合大数据 |
| CBraMod A vs LaBraM B (CHB-MIT) | **接近或 CBraMod 略优** | CBraMod 双路架构对时序+空间的双维度建模天然适配癫痫检测；LaBraM 标准 ViT 需要分层补偿 |
| Full FT > LoRA (CHB-MIT) | **Full FT 略优** | 全量微调参数容量最大；但 LoRA 预期差距在 2-5% PR AUC 以内 |
| LoRA > Linear Probing | **显著优于** | LoRA 适应 backbone 特征 vs 仅训练分类头 → 发作模式学习能力差异大 |

### 10.2 跨数据集泛化性预期

| 模型 + 方案 | CHB-MIT | Siena | TUSZ | 跨数据集稳定性 |
|------------|---------|-------|------|--------------|
| CBraMod A | 高 | 中高（数据少受限） | 高 | **较好**：时间路增强是通用策略 |
| LaBraM B | 高 | — | 最高 | **好**：大数据上分层增强充分发挥 |
| LaBraM C | — | 中高 | — | **受数据规模限制** |

### 10.3 参数效率预期

| 方案 | 参数比 | 性能/参数效率 | 适用场景 |
|------|--------|-------------|---------|
| CBraMod A | 3.8% | **高** | 通用，尤其时序关键任务 |
| LaBraM B | 4.2% | 高 | 大数据集最优 |
| LaBraM C | 2.7% | **最高**（小数据） | 小数据集参数效率最优 |
| CBraMod C | 3.9% | 低 | 仅消融对照 |
| Full FT | 100% | 低 | 上界参考 |

---

## 11. 实验运行指南

### 11.1 环境准备

```bash
unset LD_LIBRARY_PATH
source activate cbramod   # CBraMod 实验
# 或
source activate labram     # LaBraM 实验
```

### 11.2 数据预处理（按数据集）

```bash
# CHB-MIT
python PEFT_engine/preprocessing/preprocess_chbmit.py \
    --edf_dir datas/CHB-MIT --output_dir datas/CHB-MIT/processed

# Siena
python PEFT_engine/preprocessing/preprocess_siena.py \
    --edf_dir datas/Siena\ Scalp\ EEG\ Dataset --output_dir datas/Siena/processed

# TUSZ
python PEFT_engine/preprocessing/preprocess_tusz.py \
    --edf_dir datas/tusz_v2.0.6/edf --output_dir datas/TUSZ/processed
```

### 11.3 训练命令

```bash
# P0 主实验
python PEFT_engine/main.py --config PEFT_engine/configs/cbramod_chbmit_schemeA.yaml
python PEFT_engine/main.py --config PEFT_engine/configs/cbramod_siena_schemeA.yaml
python PEFT_engine/main.py --config PEFT_engine/configs/cbramod_tusz_schemeA.yaml
python PEFT_engine/main.py --config PEFT_engine/configs/labram_chbmit_schemeB.yaml
python PEFT_engine/main.py --config PEFT_engine/configs/labram_siena_schemeC.yaml
python PEFT_engine/main.py --config PEFT_engine/configs/labram_tusz_schemeB.yaml

# P1 消融
python PEFT_engine/main.py --config PEFT_engine/configs/cbramod_chbmit_schemeC.yaml
python PEFT_engine/main.py --config PEFT_engine/configs/labram_chbmit_schemeA.yaml
python PEFT_engine/main.py --config PEFT_engine/configs/labram_siena_schemeA.yaml

# P2 消融
python PEFT_engine/main.py --config PEFT_engine/configs/cbramod_chbmit_schemeB.yaml
python PEFT_engine/main.py --config PEFT_engine/configs/labram_chbmit_schemeC.yaml

# 对照组
python PEFT_engine/main.py --config PEFT_engine/configs/cbramod_chbmit_full.yaml
python PEFT_engine/main.py --config PEFT_engine/configs/cbramod_chbmit_frozen.yaml
```

### 11.4 断点续训

```bash
python PEFT_engine/main.py --config <config.yaml> --resume results/<output_dir>/latest.pt
```

### 11.5 结果目录

每个实验输出到 `results/<config_name>/`，包含：

```
results/cbramod_chbmit_schemeA/
├── tensorboard/            # TensorBoard 日志
├── best_adapter/           # 最优 checkpoint（验证集指标最优）
├── latest_adapter/         # 最新 checkpoint（断点续训用）
├── train_log.jsonl         # 训练日志（JSON Lines）
└── eval_results.json       # 测试集最终指标
```

---

## 12. 实验优先级与执行顺序

### 12.1 推荐执行顺序

| 批次 | 实验 | 理由 |
|------|------|------|
| **第 1 批**（P0） | CBraMod CHB-MIT A, LaBraM CHB-MIT B | 在中等规模数据集上验证两个主方案 |
| **第 2 批**（P0 扩展） | CBraMod Siena A, LaBraM Siena C | 验证小数据集上的主方案 |
| **第 3 批**（P0 TUSZ） | CBraMod TUSZ A, LaBraM TUSZ B | 验证大数据集上的主方案 |
| **第 4 批**（P1 消融） | CBraMod CHB-MIT C, LaBraM CHB-MIT/Siena A | 关键消融对比 |
| **第 5 批**（对照组） | CBraMod CHB-MIT Full/Frozen | 上下界参考 |
| **第 6 批**（P2 扩展） | CBraMod CHB-MIT B, LaBraM CHB-MIT C | 补充消融 |

### 12.2 超参数消融（基于 P0 结果后续追加）

在 P0 主实验完成后，基于最优方案进行超参数敏感性分析：

| 消融变量 | 实验组 | 对照组 |
|---------|--------|--------|
| r 值 | CBraMod A: t=16, s=8 | t=8, s=8 / t=32, s=16 |
| r 值 | LaBraM B: deep=16 | deep=8 / deep=32 |
| Focal γ | 2.0 | 1.0 / 3.0 |
| Focal α | 0.25 | 0.5 / 0.1 |
| 采样策略 | WeightedRandomSampler | 随机采样 / 过采样 2× |
| 数据增强 | 全部开启 | 关闭 / 仅时间平移 |

---

## 13. 总结

本实验设计通过 **3 数据集 × 2 模型 × 3 方案 + 2 对照** 的矩阵，系统性地覆盖了以下研究维度：

1. **LoRA 方案消融**：路径差异化（CBraMod）vs 深度分层（LaBraM）vs FFN-only 基线
2. **跨数据集泛化**：小数据（Siena）→ 中等数据（CHB-MIT）→ 大数据（TUSZ）
3. **跨模型对比**：双路注意力架构 vs 标准 ViT 架构
4. **参数效率**：2.7%–4.3% 可训练参数 vs 100% 全量微调 vs <0.5% Linear Probing
5. **训练策略**：Focal Loss + WeightedRandomSampler 应对类别不平衡

预期核心结论：
- **CBraMod Scheme A**（时间路增强）将是 CBraMod 最优方案，证明双路差异化 LoRA 的有效性
- **LaBraM Scheme B**（深层增强分层）在大数据集（CHB-MIT/TUSZ）上将是最优，**Scheme C**（注意力+LayerScale）在小数据集（Siena）上将更具优势
- LoRA 微调以 ~3-4% 参数预期能达到全量微调 95%+ 的性能
- 时间维度的适应能力对癫痫检测任务至关重要
