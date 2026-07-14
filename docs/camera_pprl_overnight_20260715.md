# 正确相机二元前置强化学习与检测恢复执行说明

## 这轮实验测什么

这轮实验检验：已经通过视觉依赖门的二元相机能力，经过一段短程、可验证奖励的前置强化学习后，能否在检测推理不提供任何相机文本时迁移到 ViF-Bench 的 Real/Fake 检测。

它不是把 camera caption 塞进检测 prompt，也不是继续做独立 Camera VQA/Detection 交错 SFT。训练分成两个不能混写结论的阶段：

1. 正确相机联合 SFT 模型进行 Camera-PPRL，然后立即评测相机能力和完整 ViF-Bench。
2. 保存直接 PPRL 结果后，再做 0.5 epoch 的 DataB 检测恢复，并重新评测相机能力和完整 ViF-Bench。

第二阶段只回答“检测回放能否恢复检测接口，同时保留相机能力”。如果只有恢复后模型提升，还必须补同等恢复计算、但没有 PPRL 的控制分支，才能声称提升来自相机预训练。

候选判定还必须接入第一台服务器已经运行的“仅检测回放模型”ViF 结果。只超过较弱的正确相机 SFT 起点、但仍低于仅检测回放模型，不算候选。

## 为什么现在值得做这一步

- 正确相机联合 SFT 已在 held-out DataA 二元问题上达到 74.44% Balanced ACC 和 86.28% Macro AP，并通过相反画面、无画面和翻转标签控制，证明相机能力确实依赖当前视觉输入。
- 现有 `pass@8` 采样中，82.78% 的题至少出现一次正确答案，12.56% 的题同时探索 Yes/No，允许短程 GRPO，但不支持无边界的大规模试参。
- VideoVeritas 报告的 PPRL 是先优化可验证的感知 pretext，再迁移到检测；其消融中约 1K 感知样本已经产生可测收益。因此本轮固定为 1024 条，而不是直接扩到全部数据。
- 联合 SFT 已证明“相机能力可学但不会自动迁移”。本轮只检验阶段式 PPRL 是否改变这个结论，不把先前 SFT 失败抹掉。

参考：

- [VideoVeritas: PPRL for AI-generated video detection](https://arxiv.org/abs/2602.08828)
- [ms-swift 多模态 GRPO 实践](https://github.com/modelscope/ms-swift/blob/main/docs/source/BestPractices/GRPO-Multi-Modal-Training.md)
- [ms-swift GRPO 使用说明](https://github.com/modelscope/ms-swift/blob/main/docs/source/Instruction/GRPO/GetStarted/GRPO.md)

## 固定实验设置

### 模型谱系

- 原始检测模型：`/tmp/1res/v4vif_2766busterall_trainall_5epoch/checkpoint-2115`
- 正确相机联合 SFT adapter：`/tmp/1res/camera_joint_sft_gate/train/correct_camera`
- Camera-PPRL 从“原始检测模型 + 正确相机联合 SFT adapter”的合并模型开始。
- 检测恢复从“正确相机联合 SFT + Camera-PPRL”的合并模型开始，不覆盖直接 PPRL 模型。

### Camera-PPRL

- 数据：`camera_train_correct.json` 中 1024 条完整 Yes/No 配对记录，只使用 DataA train cases。
- 选择：按 32 个相机 primitive 轮转采样，始终保留同一问题的 Yes/No 完整配对。
- 奖励：答案正确 0.9，严格短格式 0.1。
- GRPO：每题 8 个 rollout，temperature 1.0，top-p 1.0，beta 0.04，1 epoch。已有 `temperature=0.7` readiness 只有 12.56% 组内奖励方差，因此正式设置采用 ms-swift 多模态 GRPO 常用的 1.0 来增加二元动作探索。
- LoRA：rank 32，alpha 64，学习率 `1e-6`。
- 冻结视觉塔和多模态 projector；16 GPU，colocate vLLM，TP=4；当前 ms-swift 支持时启用 LoRA-only 权重同步以减少每轮 rollout 的同步开销。
- 输出允许裸 `Yes/No`、单一 `<answer>` 标签，或 Qwen 的空 `<think></think>` 包装后紧跟 Yes/No；非空 reasoning 和附加解释不计格式奖励。
- `dynamic_sample` 是本轮硬要求：零组内奖励方差的问题不进入梯度，并最多重采样 3 次。没有该 CLI 能力时预检直接停止，不降级成大部分样本零梯度的伪训练。

### 检测恢复

- 数据：完整 DataB detection JSON，性质是已见训练数据 replay，不是评测数据。
- 训练：0.5 epoch，LoRA rank 16、alpha 32、学习率 `5e-6`。
- 推理：原检测 system/user prompt，不提供 camera caption、camera label 或外部相机模型结果。

## 验收标准

直接 Camera-PPRL 相对正确相机联合 SFT：

- ViF-Bench coverage 和格式有效率均至少 99%。
- Balanced ACC 或 Fake F1 至少提高 1 点，另一项下降不超过 0.5 点。
- Real Recall 与 Fake Recall 均至少 45%。
- held-out 相机 Macro AP 下降不超过 2 点，matched 相对 opposite 的 Balanced ACC 至少高 10 点。
- 相对第一台服务器的仅检测回放模型也须满足相同的主指标增益/不退化标准；两项比较中的生成器级 Balanced ACC 胜率均至少 60%。

检测恢复相对直接 Camera-PPRL：

- 使用相同 ViF 标准。
- held-out 相机 Macro AP 下降不超过 5 点。
- 恢复后 matched 相对 opposite 的相机 Balanced ACC 仍须至少高 10 点，避免只保留标签先验而丢失视觉依赖。
- 同时必须超过仅检测回放模型；缺少该参照时只能标记为`等待检测对照`。
- 通过时状态只能写成“恢复候选，待等计算控制”，不能直接写成 Camera-PPRL 提升。

跨服务器接入的仅检测回放结果还必须与本轮 ViF 具有相同的预期样本数和完全相同的生成器集合；不兼容的 JSON 不参与候选判定。

ViF-Bench 已反复用于开发，只是开发 benchmark。GenBuster-Bench 与 MintVid 本轮不运行，继续留到方法冻结后。

## 存储分类

- 一次性大文件：合并模型、逐样本预测和 rollout 缓存放在 `/tmp/1res/camera_pprl/correct_camera_1024`，不上传 OSS。
- 持久小文件：固定 1024 条训练 JSON、smoke JSON、数据审计、评测 JSON、最终汇总和紧凑训练日志复制到 `${PROJECT_ROOT}/res/camera_pprl/correct_camera_1024`。
- 可复用大文件：正确相机联合 SFT compact adapter、Camera-PPRL compact adapter、检测恢复 compact adapter 从 `/tmp` 自动上传到 `oss://antsys-tamper/public/wong/skyra/selfcot/camerabench/ourexp/camera_pprl/correct_camera_1024/`。
- 流水线全部成功后执行 `/input/training/keep.sh`。失败时会先持久化小结果并上传已经生成的 compact adapter，但不会把失败判成方法失败。

默认读取的仅检测回放 ViF 参照为：

```text
/input/workflow_58770161/workspace/test/cameramotion_det/res/camera_joint_sft_gate/vif_four_model_compare/branches/detection-only/eval/camera_adapter_vifbench_eval.json
```

如果该文件在最终汇总时仍不存在，训练和两轮 ViF 不受影响，汇总状态会写成 `pending_detection_only_reference`。文件稍后到位后只需运行 `STAGE=summarize`，不需要重新推理或训练。

## 服务器部署文件

从 GitHub 手工复制以下文件到服务器项目的相同相对路径：

```text
rl/camera_detection_rewards.py
tools/build_camera_pprl_binary.py
tools/audit_camera_pprl_smoke.py
scripts/camera_pprl/run.sh
scripts/camera_pprl/summarize.py
```

服务器项目根目录固定为：

```text
/input/workflow_58770161/workspace/test/cameramotion_det
```

## 快速预检

预检检查文件、16 GPU、ms-swift/vLLM 导入、CLI 参数、奖励注册、DataB replay 全部图片路径，以及 ViF 的 16 份索引和帧目录。它只加载 processor 配置，不加载模型权重、不推理、不训练：

```bash
cd /input/workflow_58770161/workspace/test/cameramotion_det

RUN=scripts/camera_pprl/run.sh
STAGE=preflight bash "$RUN"
```

如果正确相机 adapter 不在当前容器，先恢复：

```bash
ossutil64 cp -r oss://antsys-tamper/public/wong/skyra/selfcot/camerabench/ourexp/camera_joint_sft_gate/train/correct_camera/ /tmp/1res/camera_joint_sft_gate/train/correct_camera/
```

若 `STAGE=all` 在构造数据时报告缺帧，先按容器启动流程恢复 DataA 统一抽帧和 DataB parsed frames；不要跳过 `--check-images`。

正式流程中的 32 条 smoke 会优先尝试 vLLM LoRA-only 权重同步；若该实验特性在当前 PPU/vLLM 上失败，脚本自动关闭它并用完整权重同步重试一次。smoke 随后读取 ms-swift 的 `frac_reward_zero_std`，要求平均零方差组比例不高于 80%，也就是至少 20% 的问题组提供有效相对优势；不通过就停在 smoke，不启动 1024 条正式训练。两种权重同步都失败也会停止。

## 无人值守完整执行

```bash
cd /input/workflow_58770161/workspace/test/cameramotion_det

ROOT=/tmp/1res/camera_pprl/correct_camera_1024
mkdir -p "$ROOT"

nohup env \
STAGE=all \
MS_SWIFT_ROOT=/input/workflow_58770161/workspace/test/ms_swift/ms-swift-main \
AUTO_UPLOAD_OSS=1 \
KEEP_ALIVE_AFTER_RUN=1 \
bash scripts/camera_pprl/run.sh \
> "$ROOT/launcher.log" 2>&1 &

echo "launcher pid: $!"
```

正常顺序为：预检、数据构造、合并 warm start、32 条 smoke、1024 条 Camera-PPRL、相机评测、直接 PPRL 完整 ViF、检测恢复、恢复后相机评测、恢复后完整 ViF、最终汇总、keep alive。

预计总时长约 7 至 12 小时，实际取决于动态重采样次数、PPU 上的 vLLM rollout 和 ViF 解码速度。脚本不会因为 GPU 利用率低于某个数值主动判失败或终止；并发完整 ViF 比较会让每张 GPU 同时运行两个模型进程。

## 查看进度与结果

```bash
tail -f /tmp/1res/camera_pprl/correct_camera_1024/launcher.log
```

训练日志位置可用下面命令发现：

```bash
find /tmp/1res/camera_pprl/correct_camera_1024/train -name trainer_log.jsonl -o -name logging.jsonl
```

全部完成后的核心结果：

```bash
cat /tmp/1res/camera_pprl/correct_camera_1024/camera_pprl_final_summary.json
```

如果最终只缺第一台服务器的检测对照，参照文件到位后运行：

```bash
cd /input/workflow_58770161/workspace/test/cameramotion_det
STAGE=summarize bash scripts/camera_pprl/run.sh
```

持久化副本：

```text
/input/workflow_58770161/workspace/test/cameramotion_det/res/camera_pprl/correct_camera_1024/
```

如果中途失败，先给出 `launcher.log` 最后 200 行、`pipeline.log` 最后 200 行，以及已经存在的 NAS 小结果；不需要下载整个 `/tmp` 目录。
