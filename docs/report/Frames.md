# EEG 基础模型评测框架汇总

---

## 1. Kastrati 等 (2025) — *EEG-Bench: A Benchmark for EEG Foundation Models in Clinical Applications*

> 聚焦**临床应用场景**的评测基准。

### 数据集（共 14 个）

| 类别 | 数据集 |
|------|--------|
| 常规与异常脑电 | TUAB（异常脑电检测）、TUAR（伪影）、TUEP（癫痫）、CHB-MIT（癫痫发作） |
| 睡眠 | Sleep-Telemetry（睡眠分期） |
| 帕金森病 | Cavanagh2018a、Cavanagh2018b、Singh2018、Singh2020、Singh2021、Brown2020 |
| 其他疾病 | Cavanagh2019（mTBI）、Albrecht2019（精神分裂症）、Gruendler2009（强迫症 OCD） |

### 任务（共 11 项）

- 正常 vs. 异常脑电二分类（Abnormal vs. Normal）
- 癫痫 vs. 非癫痫二分类（Epilepsy vs. No Epilepsy）
- 帕金森病分类 — 全数据集混合（PD All）
- 帕金森病分类 — 跨数据集测试（PD Held-Out）
- 强迫症检测（OCD）
- 轻度创伤性脑损伤检测（mTBI）
- 精神分裂症检测（Schizophrenia）
- 伪影检测 — 二分类（Binary Artifact）
- 伪影检测 — 多分类（Multiclass Artifact）
- 睡眠分期（Sleep Stages）
- 癫痫发作检测（Seizure）

### 模型

- **传统机器学习基线**：SVM、LDA（使用 Brainfeatures 工具箱提取时频域、统计和复杂度等特征）
- **脑电基础模型**：BENDR、Neuro-GPT、LaBraM

---

## 2. Lu 等 (2026) — *OmniEEG-Bench: A Standardized Evaluation Benchmark for EEG Foundation Models*

> 提出了一个**大一统基准**，覆盖面极广。

### 数据集（共 54 个）

涵盖开源社区极其广泛的数据集，包括：

- **异常/噪声**：EEGDenoiseNet、TUAB、TUEV、TUEP、TUSL
- **人口统计学**：MPI-LEMON（年龄/性别/性格）
- **临床疾病**：Siena EEG、ADHD、AD65、PD31、TDBRAIN、MDD、MODMA
- **睡眠**：ISRUC-Sleep 系列、Sleep-EDF、HMC
- **情感与认知**：SEED 系列（SEED、SEED-IV、SEED-V 等）、DEAP、FACED
- **BCI 与交互**：BCI-Speech、Things-EEG2、BCI Competition IV-2A & IV-1、PhysioNet-MI、BETA-SSVEP 等

### 任务（6 大家族，共 58 项）

1. **信号可靠性（Signal Reliability）** — 伪影/噪声识别、纵向重测信度
2. **生物特征与疾病（Biometrics & Disease）** — 年龄/性别预测、癫痫检测、ADHD、阿尔茨海默病、帕金森病、抑郁症检测等
3. **意识与状态（Consciousness & State）** — 意识水平检测、睡眠分期、认知任务状态识别
4. **认知与情感（Cognition & Emotion）** — 警觉性检测、多分类情绪识别、工作/认知负荷评估
5. **自然刺激解码（Naturalistic Stimulus Decoding）** — 自然语音感知与注意力解码、阅读解码、视觉语义分类
6. **运动与交互（Motor & Interaction）** — 运动想象、SSVEP 目标识别、错误相关电位反馈、闭环辅助控制

### 模型

- **脑电基础模型（10 种）**：BENDR、BIOT、LaBraM、CBraMod、BrainOmni、FEMBA、Neuro-GPT、NeuroLM、EEGMamba、REVE
- **特定任务神经网络基线**：EEGConformer、EEGNet

---

## 3. Yang 等 (2026) — *Are EEG Foundation Models Worth It?*

> 聚焦基础模型在 **BCI 及相关任务**中的实用性，重点对比非神经网络经典解码器。

### 数据集（主要 8 个 + 额外 2 个）

| 数据集 | 用途 |
|--------|------|
| Error-Related EEG Dataset | 人机交互错误相关负电位 |
| Alzheimer's Diagnosis EEG | 阿尔茨海默病与额颞叶痴呆 |
| Thinking Out Loud | 内在语音识别 |
| BCI Competition IV 2a | 经典运动想象 |
| Upper-Limb Motor Execution/Imagery | 上肢运动执行与想象 |
| Binocular Dual-Frequency SSVEP | 双目双频 SSVEP |
| DTU "Cocktail Party" | 鸡尾酒会效应 / 听觉注意力 |
| SEED-VIG | 模拟驾驶警觉性 |

> *额外 Zero-shot 评估：FACED（情绪）、TUEV（异常事件）*

### 任务（7 项分类 + 2 项回归）

**分类任务：**

- 2 分类 ERN 检测
- 3 分类阿尔茨海默病诊断
- 4 分类内在语音（Inner Speech）识别
- 4 分类运动想象
- 7 分类上肢运动执行
- 7 分类上肢运动想象
- 40 分类双目 SSVEP 目标识别
- *Zero-shot*：9 分类情绪、6 分类事件

**回归任务：**

- 听觉注意力解码（Auditory Attention Decoding，重建语音包络）
- 警觉性水平预测（Vigilance Level Prediction）

### 模型

- **脑电基础模型**：BENDR、BIOT、LaBraM、EEGPT、CBraMod，以及作者提出的基于掩码自编码器（MAE）预训练的 **ST-EEGFormer**
- **经典神经网络（Classic NN）**：DeepConvNet、EEGNet、EEG Conformer、CTNet
- **经典非神经网络（Classic Non-NN）**：
  - 基于 CSP/FBCSP 特征的 LDA/SVM
  - 黎曼几何分类器（MDM、FgMDM、TS-ElasticNet）
  - 相对频带功率（RBP）结合 RF/SVM/kNN/LightGBM
  - xDAWN-LDA/MDM
  - SSVEP 专用：FBCCA、TRCA
