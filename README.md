# A Systematic Evaluation of Parameter-Efficient Fine-Tuning for Clinical EEG Foundation Models

## 项目简介

本项目旨在使用一系列参数高效微调（Parameter-Efficient Fine-Tuning, PEFT）方法，对当前主流的脑电基础模型（EEG Foundation Models）在具体临床任务上进行微调，并使用领域统一的评估流程对微调结果进行系统性评估。

- **微调对象**：CBraMod、LaBraM 两个脑电基础模型
- **临床数据集**：CHB-MIT、Siena Scalp EEG
- **PEFT 方法**：LoRA 等（基于 [Hugging Face PEFT](https://github.com/huggingface/peft) 库）
- **评估框架**：[SzCORE](https://github.com/esl-epfl/szcore) — 开源癫痫发作检测基准平台

---

## 目录结构

```
EEG-FM-bench/
├── datas/                               # 临床 EEG 数据集（已 gitignore）
│   ├── CHB-MIT/                         #   CHB-MIT 头皮脑电数据库（24 名癫痫患者）
│   └── Siena Scalp EEG Dataset/         #   Siena 头皮脑电数据集
│
├── docs/                                # 项目文档
│   ├── essay/                           #   参考论文
│   ├── presentation/                    #   开题报告与展示材料
│   ├── report/                          #   技术报告
│   ├── spec/                            #   技术规格说明
│   └── todo                             #   PEFT 微调实验任务清单
│
├── external/                            # 外部依赖资产
│   ├── models/                          #   EEG 基础模型源码
│   │   ├── CBraMod-main/                #     CBraMod（ICLR 2025）
│   │   └── LaBraM-main/                 #     LaBraM（ICLR 2024）
│   ├── Frames/                          #   评估框架与基准
│   │   ├── EEG-FM-Bench-main/           #     EEG 基础模型综合评测基准
│   │   ├── EEG-Bench-main/              #     临床应用 EEG 基准
│   │   ├── omni-eegbench-main/          #     大一统 EEG 基准（54 数据集 / 58 任务）
│   │   ├── STEEGFormer-main/            #     ST-EEGFormer 基准
│   │   └── szcore-main/                 #     癫痫发作检测统一评估平台
│   └── Pefts/                           #   PEFT 方法库
│       └── peft-main/                   #     Hugging Face PEFT 库
│
├── .gitignore
└── README.md
```

## EEG 基础模型

| 模型 | 来源 | 论文 | 说明 |
|------|------|------|------|
| **CBraMod** | `external/models/CBraMod-main/` | [arXiv:2412.07236](https://arxiv.org/abs/2412.07236) (ICLR 2025) | Criss-Cross Brain Foundation Model，通过交叉注意力机制同时建模 EEG 的通道间与时间维关系 |
| **LaBraM** | `external/models/LaBraM-main/` | [ICLR 2024](https://openreview.net/forum?id=QzTpTRVtrP) | Large Brain Model，基于掩码自编码器在约 2500 小时多类型 EEG 数据上预训练 |

## 临床数据集

| 数据集 | 路径 | 任务 | 说明 |
|--------|------|------|------|
| **CHB-MIT** | `datas/CHB-MIT/` | 癫痫发作检测 | 儿童医院波士顿分院头皮脑电数据库，24 名难治性癫痫患者长时间多导联记录 |
| **Siena** | `datas/Siena Scalp EEG Dataset/` | 癫痫发作检测 | 锡耶纳大学头皮脑电数据库 |

> 数据集已通过 `.gitignore` 排除版本控制，需单独获取。

## 评估框架

本项目采用 **SzCORE**（`external/Frames/szcore-main/`）作为统一评估基准。SzCORE 是一个开源的癫痫发作检测基准平台，提供标准化的数据流和性能指标计算流程，支持容器化算法提交与自动化评测。

参考的其他评测框架详见 [`docs/report/Frames.md`](docs/report/Frames.md)。

## 实验任务

当前实验规划见 [`docs/todo`](docs/todo)：

- **CBraMod** × {CHB-MIT, Siena, TUSZ} × LoRA 微调
- **LaBraM** × {CHB-MIT, Siena, TUSZ} × LoRA 微调
