# DataB 显式 Camera labels+caption 检测 SFT 执行说明

## 实验问题

只检验一个变量：在训练和 ViF-Bench 推理时，都把匹配的 CameraBench `labels + caption` 追加到原检测 user prompt，是否能提高最终 `Real/Fake` 检测指标。

不使用 DataA，不使用旧 detection checkpoint 继续训练，不训练路由器，也不修改 system prompt、assistant CoT、`<answer>` 或图片输入。

## 两个训练分支

两个分支都从 `/home/admin/Qwen3-VL-8B-Instruct` 开始，使用同一批 DataB camera-covered 记录、相同顺序和原始 DataB full-SFT 5 epoch 参数。

| 分支 | LlamaFactory 数据集 | 唯一差异 |
|---|---|---|
| 无 Camera 条件检测模型 | `datab_explicit_camera_no_camera` | 原 user prompt 不变 |
| 显式 Camera 条件检测模型 | `datab_explicit_camera_labels_caption` | user prompt 末尾追加匹配的 `<labels>` 与 `<caption>` |

本地数据审计预期：原 detection 6766 行、camera sidecar 5639 行、匹配后两个分支各 5739 行。100 条重复 detection 解释仍保留，因此 5739 行对应 5639 个唯一 camera path。

## 服务器输入

```text
DataB detection:
/input/workflow_58770161/workspace/test/camb/camerabenchdataB-main/detection/v4vif_2766busterall_trainall.json

DataB camera:
/input/workflow_58770161/workspace/test/camb/camerabenchdataB-main/datab_cameramotion_labels_final/datab_cameramotion_labels_v2.jsonl

基础模型:
/home/admin/Qwen3-VL-8B-Instruct
```

LlamaFactory 与 DeepSpeed 路径由 `run.sh` 在已知服务器位置中自动发现，也允许通过 `LLAMAFACTORY_ROOT`、`LLAMAFACTORY_DATA_DIR` 和 `DEEPSPEED_CONFIG` 显式覆盖。

## 构建和预检

代码复制到服务器后执行：

```bash
cd /input/workflow_58770161/workspace/test/cameramotion_det
RUN=scripts/datab_explicit_camera_sft/run.sh

STAGE=preflight bash "$RUN"
STAGE=build bash "$RUN"
STAGE=smoke bash "$RUN"
```

`build` 必须输出 `matched_records: 5739`、`paired_integrity: true`、`no_camera_prompts_contain_camera_block: false`，并确认 Camera 分支每条记录恰好有一个 camera block。默认同时检查所有图片路径。

## 正式训练

两台服务器并行时分别执行：

```bash
cd /input/workflow_58770161/workspace/test/cameramotion_det
KEEP_ALIVE_AFTER_RUN=1 STAGE=train_no_camera bash scripts/datab_explicit_camera_sft/run.sh
```

```bash
cd /input/workflow_58770161/workspace/test/cameramotion_det
KEEP_ALIVE_AFTER_RUN=1 STAGE=train_with_camera bash scripts/datab_explicit_camera_sft/run.sh
```

只有一台服务器时可顺序执行：

```bash
cd /input/workflow_58770161/workspace/test/cameramotion_det
KEEP_ALIVE_AFTER_RUN=1 STAGE=train_both bash scripts/datab_explicit_camera_sft/run.sh
```

训练输出：

```text
/tmp/1res/datab_explicit_camera_sft/v1/train/no_camera
/tmp/1res/datab_explicit_camera_sft/v1/train/with_camera
```

两者均为 full SFT：5 epoch、学习率 `1e-5`、16 GPU、每卡 batch size 1、梯度累积 1、cosine scheduler、warmup 0.1，冻结视觉塔与多模态投影层，训练语言模型；每 500 step 保存一次，只保存模型权重，并允许覆盖已有输出目录。

## 存储

- 可重新构建的大 JSON 和训练模型位于 `/tmp`。
- 数据摘要、5739 条配对 manifest、最终运行配置和安装摘要复制到 NAS：`res/datab_explicit_camera_sft/v1`。
- 两个 full-SFT 模型属于可复用大文件，训练完成并检查正常后，在容器退出前分别上传 OSS：

```bash
ossutil64 cp -r /tmp/1res/datab_explicit_camera_sft/v1/train/no_camera/ oss://antsys-tamper/public/wong/skyra/selfcot/camerabench/ourexp/datab_explicit_camera_sft/v1/train/no_camera/
```

```bash
ossutil64 cp -r /tmp/1res/datab_explicit_camera_sft/v1/train/with_camera/ oss://antsys-tamper/public/wong/skyra/selfcot/camerabench/ourexp/datab_explicit_camera_sft/v1/train/with_camera/
```

ViF-Bench 的 labels+caption sidecar 生成完成后，再增加严格对应的两分支推理入口；本阶段不提前假设其文件路径或覆盖数。
