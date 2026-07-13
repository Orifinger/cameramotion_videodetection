# DataB 自动解释的 DeepfakeJudge-7B 可靠性门

## 这轮具体测什么

这轮不直接清洗完整 DataB，而是先验证开源 DeepfakeJudge-7B 在 DataB 的有序视频帧上是否真的依据视觉证据评价自动 CoT。只有原始解释稳定高于同标签错配帧、错误 bbox、错误时间段和错误伪影类别时，才允许进入人工校准和全量筛选。

这里评审的是自动生成的解释、伪影类别、时间段和 bbox，不重新定义原始数据的 Real/Fake 身份。Judge 使用的 ground truth 必须从 DataB 原始图片路径中的 `real`/`fake` 目录独立取得，不能从候选 `<answer>` 读取。

## 模型、数据和存储

- Judge：`MBZUAI/Qwen-2.5-VL-Instruct-7B-Pointwise-DFJ`，即 DeepfakeJudge-7B pointwise 完整模型。
- 模型默认服务器路径：`/tmp/1res/models/Qwen-2.5-VL-Instruct-7B-Pointwise-DFJ`。
- DataB：`/input/workflow_58770161/workspace/test/camb/camerabenchdataB-main/detection/v4vif_2766busterall_trainall.json`。
- 项目部署根目录：`/input/workflow_58770161/workspace/test/cameramotion_det`。
- 门验收输出：`/tmp/1res/datab_deepfakejudge_gate`，属于一次性验证输出，当前不放 NAS、不上传 OSS。
- 代码和实验定义提交到 GitHub；模型权重由用户通过可用的内部模型下载渠道取得后放入上述 `/tmp` 目录。

本地结构审计确认 DataB 共 6766 条且 Real/Fake 各 3383 条。绝大多数记录为 16 帧，适配代码同时支持少量其他帧数，不会静默丢弃。

## 输入和控制条件

从 DataB 按 Real/Fake 与来源分层抽 200 条。每条原始解释都保留可信 Real/Fake 身份，并最多产生四种控制：

| 条件 | 保持不变 | 改变内容 | 测试目的 |
|---|---|---|---|
| 原始解释 | 原始帧、身份、CoT | 无 | 待评价对象 |
| 同标签错配帧 | 身份、候选 CoT | 换成另一条同标签视频的帧 | 检查 Judge 是否依赖当前画面 |
| 错误 bbox | 原始帧、身份、解释文本 | 把 bbox 移到尽可能远的合法位置 | 检查空间证据敏感性 |
| 错误时间段 | 原始帧、身份、解释文本 | 把时间段移到远离原区间的位置 | 检查时间证据敏感性 |
| 错误伪影类别 | 原始帧、身份、其他解释 | 把 V4+ 类别替换为另一合法类别 | 检查类别语义敏感性 |

所有条件均使用相同 DeepfakeJudge checkpoint、确定性解码和评分提示。模型输出必须为 `<reasoning>...</reasoning><score>1-5</score>`。

## 服务器需要覆盖的文件

从 GitHub 取得以下文件并复制到对应服务器位置：

```text
tools/build_datab_deepfakejudge_gate.py
tools/eval_datab_deepfakejudge_gate.py
scripts/datab_deepfakejudge/__init__.py
scripts/datab_deepfakejudge/infer_pointwise.py
scripts/datab_deepfakejudge/run_datab_deepfakejudge_gate.sh
```

例如最后一个文件应位于：

```text
/input/workflow_58770161/workspace/test/cameramotion_det/scripts/datab_deepfakejudge/run_datab_deepfakejudge_gate.sh
```

不需要复制或修改 DeepfakeJudge 官方仓库，也不需要安装 ms-swift。本项目用 `transformers + qwen-vl-utils + torchrun` 直接做 16 GPU 分片推理，避免依赖官方批量结果格式。

## 完整执行

先确认模型目录包含 `config.json` 和四个 `model-*.safetensors` 分片。若实际模型路径不同，每条命令都覆盖 `MODEL_PATH`。

```bash
ROOT=/input/workflow_58770161/workspace/test/cameramotion_det
RUN=${ROOT}/scripts/datab_deepfakejudge/run_datab_deepfakejudge_gate.sh
MODEL=/tmp/1res/models/Qwen-2.5-VL-Instruct-7B-Pointwise-DFJ
```

第一步，检查数据、模型、代码和 Python 依赖：

```bash
PROJECT_ROOT=${ROOT} MODEL_PATH=${MODEL} STAGE=preflight bash "${RUN}"
```

第二步，构造 200 条原始样本及控制输入，同时检查所有图片路径：

```bash
PROJECT_ROOT=${ROOT} MODEL_PATH=${MODEL} STAGE=build bash "${RUN}"
cat /tmp/1res/datab_deepfakejudge_gate/data/datab_deepfakejudge_gate_build_summary.json
```

第三步，使用 16 张 GPU 推理：

```bash
PROJECT_ROOT=${ROOT} MODEL_PATH=${MODEL} NPROC_PER_NODE=16 STAGE=infer bash "${RUN}"
```

第四步，汇总评分与控制对比：

```bash
PROJECT_ROOT=${ROOT} MODEL_PATH=${MODEL} STAGE=eval bash "${RUN}"
cat /tmp/1res/datab_deepfakejudge_gate/eval/datab_deepfakejudge_gate_summary.json
```

也可以一次执行，但第一次建议分步，以便先核对 build summary：

```bash
PROJECT_ROOT=${ROOT} MODEL_PATH=${MODEL} NPROC_PER_NODE=16 STAGE=all bash "${RUN}"
```

## 验收标准和结果解释

- 预测覆盖率至少 98%，评分格式有效率至少 95%。
- 原始解释分数严格高于同标签错配帧的比例至少 70%。
- bbox、时间、类别三个局部控制中，至少一个有不少于 30 对样本且原始解释严格胜出比例至少 65%。
- 通过只表示 Judge 对 DataB 的视觉和局部证据具有初步敏感性，不表示自动标注本身已经高质量。
- 未通过时不能用该 Judge 的分数筛选全量数据，应先查看 CSV 中的平分、反向样本和格式错误。

需要返回的核心结果文件是：

```text
/tmp/1res/datab_deepfakejudge_gate/eval/datab_deepfakejudge_gate_summary.json
```

逐条结果位于：

```text
/tmp/1res/datab_deepfakejudge_gate/eval/datab_deepfakejudge_gate_items.csv
```

若门验收通过，下一步才抽取 100 条原始解释做盲法人工评分，校准自动接受、复核和拒绝阈值；在人工校准前不运行完整 6766 条筛选。
