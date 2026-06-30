参数高效微调（Parameter-Efficient Fine-Tuning, **PEFT**）是目前大模型（包括 NLP、CV 以及你正在研究的 EEG 基础模型）下游适配的绝对主流技术。它的核心思想是：**冻结预训练模型的大部分权重，只微调或增加极少量的参数**，从而在大幅降低算力、显存消耗的同时，达到媲美甚至在特定场景下超越全量微调（Full Fine-Tuning）的性能。

以下是目前主流的 PEFT 方法分类总结，以及它们对应的开源仓库地址：

### 1. 基于重参数化（Reparameterization-based）

这是目前应用最广、效果最稳定的一类方法，尤以 LoRA 家族为代表。

* **LoRA (Low-Rank Adaptation)**
* **原理**：假设模型在微调时权重的更新量（$\Delta W$）具有“低秩”的内在特性。LoRA 冻结了原始的大矩阵权重，通过引入两个较小的矩阵相乘（降维矩阵 $A$ 和升维矩阵 $B$）来模拟权重的变化。微调结束后，可以把 $A \times B$ 的结果无缝加回到原权重中，推理时没有任何额外延迟。
* **适用场景**：几乎所有 Transformer 架构，是目前的首选 Baseline。


* **QLoRA (Quantized LoRA)**
* **原理**：在 LoRA 的基础上，将预训练模型的主干权重极度量化（比如 4-bit 精度），同时在 16-bit 精度下训练 LoRA 参数。这使得在单张消费级显卡上微调极大参数量的模型成为可能。


* **AdaLoRA (Adaptive LoRA)**
* **原理**：LoRA 会给所有层分配相同的“秩（Rank）”，但不同层的重要性不同。AdaLoRA 会根据参数的重要性自动动态分配预算，重要的层给更高的秩，不重要的层给更低的秩甚至裁剪掉。



### 2. 基于加性模块（Additive / Adapter-based）

这是最早被提出的一类 PEFT 方法。

* **Adapter Tuning**
* **原理**：在原有的网络层（通常是 Transformer 的 Attention 层和 FFN 层之后）插入一个轻量级的“瓶颈”网络（Bottleneck）。这个瓶颈网络先将高维特征降维，再升维回去。训练时只更新这个插入的 Adapter 模块。
* **特点**：虽然参数少，但因为增加了网络深度，推理时会有一点点延迟。



### 3. 基于软提示（Prompt-based）

这类方法受到 Prompt Engineering 的启发，将离散的文字 Prompt 变成了可求导的连续向量。

* **Prefix Tuning**
* **原理**：在 Transformer 的每一层的 Multi-Head Attention 计算中，在 Key 和 Value 序列的前面拼接上一段可训练的连续向量（Prefix）。


* **Prompt Tuning**
* **原理**：比 Prefix Tuning 更简单，只在最底层的输入 Embedding 序列前面加上可训练的“虚拟 Token”（Soft Prompt）。


* **P-Tuning (v1 / v2)**
* **原理**：P-Tuning 引入了一个小型的编码器（如 LSTM 或 MLP）来生成连续的 Prompt 向量；v2 版本则类似于 Prefix Tuning，在每一层都加入了可训练的 Prompt，极大提升了在小模型上的表现。



### 4. 选择性微调（Selective Fine-Tuning）

* **BitFit**
* **原理**：最简单粗暴的方法。什么新结构都不加，冻结模型所有的权重矩阵，**只更新网络中的 Bias（偏置项）参数**。虽然更新参数量极小（不到 0.1%），但在某些特定任务上能取得惊人的效果。



---

### 💻 核心代码仓库地址

在实际工程和学术研究中，你**不需要**去每个原论文的仓库里扒代码。开源社区已经为你准备了“大一统”的工具库：

#### 🥇 1. Hugging Face `PEFT` 库 (首选，绝对的行业标准)

* **仓库地址**: [https://github.com/huggingface/peft](https://github.com/huggingface/peft)
* **说明**: 只要你做大模型微调，这个库是绕不开的。它集成了上述提到的几乎所有主流方法（**LoRA, QLoRA, AdaLoRA, Prefix Tuning, Prompt Tuning, P-Tuning** 等）。它的 API 设计极其优雅，只需几行代码就能把一个庞大的模型转换成 PEFT 模型。
* **在 EEG 中的应用**：如果你打算在你的任务中微调 BENDR 或 LaBraM 等模型，强烈建议直接使用 Hugging Face 的 `peft` 库，把 EEG 基础模型包装成 `PeftModel` 即可。

#### 🥈 2. Adapter-Hub (专精 Adapter 的库)

* **仓库地址**: [https://github.com/Adapter-Hub/adapters](https://github.com/Adapter-Hub/adapters) (原名 adapter-transformers)
* **说明**: 如果你想重点研究 **Adapter** 及其变体（如 Bottleneck Adapters, Invertible Adapters），这个库比 HF 的 `peft` 提供了更丰富的 Adapter 架构支持，并且支持多个 Adapter 之间的组合和融合（Adapter Fusion）。

#### 🥉 3. 微软的官方 LoRA 仓库

* **仓库地址**: [https://github.com/microsoft/LoRA](https://github.com/microsoft/LoRA)
* **说明**: 这是原作者的实现代码。如果你想深入了解底层逻辑，或者在不依赖 Hugging Face 生态的自定义模型（比如 Yang 等人提出的纯视觉 Transformer `ST-EEGFormer`）上手动实现 LoRA 层，可以参考这里的源码。

### 💡 给你的研究建议

针对你的课题《针对临床脑电基础模型的参数高效微调系统性评估》：
你可以直接依赖 **Hugging Face `PEFT` 库**。选择 3-4 种具有代表性的方法（如 **LoRA** 代表重参数化，**Prompt Tuning** 代表提示类，**BitFit** 代表选择性），在相同的 EEG 基础模型和临床数据集上跑通对比实验，这就已经是一个非常扎实的“系统性评估”框架了。