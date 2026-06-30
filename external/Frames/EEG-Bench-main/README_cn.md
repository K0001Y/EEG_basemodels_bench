# 🧠 EEG-Bench

**一个标准化、可扩展的基准测试框架，用于在临床和 BCI 脑电解码任务中评估传统模型与基础模型。**

本基准测试支持跨 25 个数据集的严格跨被试和跨数据集评估。涵盖临床分类任务（如癫痫、帕金森、精神分裂症）和运动想象范式（如左/右手、五指解码），并提供从 CSP 到 BENDR、LaBraM、NeuroGPT 等基础模型的基线。

## 📦 安装

### 配置环境

```bash
conda env create -f environment.yml
conda activate eeg_bench
```

### 配置路径

如需覆盖默认设置，请更新以下内容：
- 在 `eeg_bench/config.json` 中修改 "data"、"cache" 和 "chkpt" 路径，指向你指定的目录。
- 在 MNE 配置中，相应调整 MOABB 下载路径。

## 📁 项目结构

```bash
eeg_bench/
├── datasets/         # EEG 数据集加载器（BCI 和临床）
├── models/           # 所有模型实现（CSP、LaBraM 等）
├── tasks/            # 基准测试任务（运动想象、临床诊断）
└── utils/            # 辅助工具
benchmark_console.py  # 运行实验的 CLI 接口
```


## 🚀 运行基准测试

通过 `benchmark_console.py` 脚本运行基准测试，使用 `--model` 和 `--task` 参数。

```bash
python benchmark_console.py --model labram --task lr
```

使用 `--all` 选项可运行所有任务对所有模型的测试。重复次数可通过 `--reps` 设置（默认：5）。

### 可用任务

| 任务代码 | 任务类 |
|-----------|--------------------------------|
| pd        | ParkinsonsClinicalTask         |
| sz        | SchizophreniaClinicalTask      |
| mtbi      | MTBIClinicalTask               |
| ocd       | OCDClinicalTask                |
| ep        | EpilepsyClinicalTask           |
| ab        | AbnormalClinicalTask           |
| lr        | LeftHandvRightHandMITask       |
| rf        | RightHandvFeetMITask           |
| lrft      | LeftHandvRightHandvFeetvTongueMITask |
| 5f        | FiveFingersMITask              |
| sleep_stages | SleepStagesClinicalTask    |
| seizure | SeizureClinicalTask             |
| binary_artifact | ArtifactBinaryClinicalTask|
| multiclass_artifact | ArtifactMulticlassClinicalTask |

### 可用模型

| 模型代码 | 模型类 |
|------------|-------------------------------|
| lda        | CSP 或 Brainfeatures + LDA |
| svm        | CSP 或 Brainfeatures + SVM |
| labram     | LaBraM                         |
| bendr      | BENDR                         |
| neurogpt   | NeuroGPT                      |

## 📋 结果

下表报告了每个任务在每个模型上的平衡准确率分数。

| **任务** | **类型** | **SVM** | **LDA** | **BENDR** | **Neuro-GPT** | **LaBraM** |
|---|---|---|---|---|---|---|
| 左手 vs 右手 | All | 0.665 | 0.660 | 0.665 ± .011 | 0.649 ± .005 | **0.672 ± .007** |
| 左手 vs 右手 | Held-Out | **0.785** | 0.762 | 0.722 ± .035 | 0.518 ± .021 | 0.735 ± .029 |
| 右手 vs 脚 | All | 0.580 | 0.569 | **0.746 ± .004** | 0.644 ± .007 | 0.738 ± .007 |
| 右手 vs 脚 | Held-Out | 0.506 | 0.714 | **0.745 ± .011** | 0.508 ± .024 | 0.718 ± .014 |
| 左手vs右手vs脚vs舌 | All | 0.287 | 0.291 | 0.625 ± .003 | 0.378 ± .010 | **0.638 ± .002** |
| 五指 | Single | 0.206 | 0.196 | 0.340 ± .008 | 0.2301 ± .004 | **0.354 ± .007** |
| 异常脑电 | Single | 0.722 | 0.677 | 0.717 ± .003 | 0.696 ± .005 | **0.838 ± .011** |
| 癫痫 | Single | 0.531 | 0.531 | **0.740 ± .015** | 0.734 ± .010 | 0.565 ± .017 |
| 帕金森 | All | 0.648 | 0.658 | 0.529 ± .009 | **0.687 ± .000** | 0.656 ± .025 |
| 帕金森 | Held-Out | 0.596 | 0.654 | 0.615 ± .038 | **0.673 ± .000** | 0.673 ± .038 |
| 强迫症 | Single | 0.633 | 0.717 | 0.513 ± .051 | 0.703 ± .082 | **0.740 ± .044** |
| 轻度创伤性脑损伤 | Single | 0.626 | **0.813** | 0.640 ± .093 | 0.646 ± .000 | 0.740 ± .173 |
| 精神分裂症 | Single | **0.679** | 0.547 | 0.471 ± .055 | 0.545 ± .042 | 0.543 ± .045 |
| 伪影二分类 | Single | 0.745 | 0.705 | 0.535 ± .003 | 0.711 ± .004 | **0.756 ± .007** |
| 伪影多分类 | Single | **0.437** | 0.325 | 0.192 ± .002 | 0.226 ± .006 | 0.430 ± .015 |
| 睡眠分期 | Single | 0.652 | **0.671** | 0.169 ± .001 | 0.166 ± .003 | 0.192 ± .001 |
| 癫痫发作 | Single | 0.572 | 0.529 | 0.501 ± .001 | 0.500 ± .000 | **0.588 ± .011** |


## ➕ 添加自定义数据集

目前本基准测试支持两种范式：临床和 BCI（运动想象）。临床范式中需要对整段录音进行分类，而 BCI 范式中对短序列（trial）进行分类。添加数据集步骤如下：

1. 将你的类放在 `datasets/bci/` 或 `datasets/clinical/` 中
2. 继承 `BaseBCIDataset` 或 `BaseClinicalDataset`
3. 实现以下方法：
    1. `_download`：自动下载数据集或提供手动下载说明。注意：如果数据集已存在于本地，`_download` 不应重复下载。
    2. `load_data`：此方法应填充以下属性：
        - `self.data`，类型为 `np.ndarray | List[BaseRaw]`，维度为 `(n_samples, n_channels, n_sample_length)`
        - `self.labels`，类型为 `np.ndarray | List[str]`，维度为 `(n_samples, )`，或多标签数据集为 `(n_samples, n_multi_labels)`
        - `self.meta`：字典，至少包含 `sampling_frequency`、`channel_names` 和 `name`
    4. 如果数据集包含尚未在枚举 `enums.BCIClasses` 或 `enums.ClinicalClasses` 中的类别，请相应添加。
    5. 对于多标签数据集，还需将数据集名称添加到

            elif dataset_name in [<MULTILABEL_DATASET_NAMES>]:
        语句中，位于 `eeg_bench/models/clinical/brainfeatures/feature_extraction_2.py:_prepare_data_cached()`。
    5. 为加速后续 `load_data` 调用，参照现有数据集类实现缓存。
    6. 所有 EEG 信号应标准化为微伏（µV）尺度。为减少内存和计算开销，采样率超过 250 Hz 的信号通常重采样至 250 Hz。

## 🧪 添加自定义任务

任务是基准测试的核心组织原则，封装了范式、数据集、预测类别、被试划分（即训练/测试集）和评估指标。每个任务类实现 `get_data()` 方法，返回训练或测试数据及对应标签和元数据。这些预定义划分确保评估一致性和可复现性。任务也分为临床和 BCI 两类。

每个任务定义：
- 使用的数据集
- 训练/测试被试划分
- 目标类别
- 评估指标

添加自定义任务：
- BCI 任务：将类放在 `tasks/bci/` 并继承 `AbstractBCITask`
- 临床任务：将类放在 `tasks/clinical/` 并继承 `AbstractClinicalTask`

实现 `get_data()` 方法返回训练/测试划分及数据、标签和元数据。

对于多标签任务，还需将其名称添加到 `eeg_bench/utils/utils.py` 的 `get_multilabel_tasks()` 方法中。此外，如果有特殊通道要求，还需在 `_prepare_data_cached()` 中添加：

    elif task_name == <YOUR_TASK_NAME>:
        t_channels = <YOUR_CHANNEL_LIST>
语句，位于 `eeg_bench/models/clinical/brainfeatures/feature_extraction_2.py`。

## 🤖 添加自定义模型

集成新模型时，实现 `AbstractModel` 接口并将代码放在：
- `models/bci/` — 运动想象（BCI）模型
- `models/clinical/` — 临床模型

### 模型必须实现：
```python
def fit(self, X: List[np.ndarray | List[BaseRaw]], y: List[np.ndarray | List[str]], meta: List[Dict]) -> None:
    # 每个 list 元素对应一个数据集
    pass

def predict(self, X: List[np.ndarray | List[BaseRaw]], meta: List[Dict]) -> np.ndarray:
    # 对每个数据集分别预测，返回拼接后的预测结果
    pass
```

### 运行你的模型
在 `benchmark_console.py` 中注册模型后运行：
```bash
python benchmark_console.py --model mymodel --task <你选择的任务>
```

## 📊 评估与可复现性

所有实验：
- 使用固定的被试级划分
- 支持留出数据集泛化评估
- 报告平衡准确率和加权 F1 分数
- 使用固定的随机种子（NumPy/PyTorch/random）

### 故障排除

由于包和模型数量众多，可能存在版本兼容问题。已知问题及解决方案如下：
- `RuntimeError: Failed to import transformers.training_args because of the following error (look up to see its traceback): No module named 'torch._six'` 或 `ModuleNotFoundError: No module named 'torch._six'`：需要删除
    - `conda_envs/eeg_bench/lib/python3.10/site-packages/deepspeed/runtime/utils.py` 第 18 行 `from torch._six import inf`
    - `conda_envs/eeg_bench/lib/python3.10/site-packages/deepspeed/runtime/zero/stage2.py` 第 9 行 `from torch._six import inf`

### 许可证

本作品采用 GNU GPL v3.0 或更高版本许可证。详情见 `LICENSE`。

注意：本仓库包含 [brainfeatures-toolbox](https://github.com/TNTLFreiburg/brainfeatures)（位于 `eeg_bench/models/clinical/brainfeatures/`），该工具箱同样采用 GNU GPL v3.0 许可证。
