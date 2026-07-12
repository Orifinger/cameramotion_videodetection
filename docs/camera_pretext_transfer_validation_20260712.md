# 正确相机能力学习与检测迁移闭环验证

## 这轮实验测什么

先验证模型能否从视频帧真正学会正确的全局相机运动标签，再验证这项能力在推理时不提供相机文本的情况下，能否迁移为局部编辑 AIGC 视频检测增益。

这是一轮两阶段验证。阶段一未通过就停止；阶段一通过后才执行阶段二。它不使用 GRPO、外部 camera caption、检测时 camera prompt、bbox 或光流特征。

## 固定条件

- 初始模型：`/tmp/1res/v4vif_2766busterall_trainall_5epoch/checkpoint-2115`。
- DataA detection：`/input/workflow_58770161/workspace/test/cameramotion_det/res/dataA_v1/autolabel/dataa_vace_grounded_cot_40step_v3_sft_clean.json`。
- DataA camera labels：`/input/workflow_58770161/workspace/test/cameramotion_det/camera/camerajson/dataa_cameramotion_labels_40step_v3.jsonl`。
- 固定开发身份：`/input/workflow_58770161/workspace/test/cameramotion_det/tools/data/camera_motion_splits/dataA_test.json`。
- 训练身份：最终 1080 个完整 case 减固定 321 个开发身份；不读取少 13 个新 case 的旧 `dataA_train.json`。
- 阶段二沿用：`/tmp/1res/caspr_gate1/data/dataa_train_pairs_256.jsonl`、`datab_replay_512.jsonl`、`dataa_dev_pairs.jsonl`。
- 所有验证产物放在 `/tmp/1res/camera_pretext_transfer_gate`，容器退出后可丢失，本轮无需上传 OSS。

## 阶段一：正确相机能力学习

每个训练 case 只使用 real 视频一次，所有样本使用同一个 canonical prompt，目标为严格的：

```text
<camera_motion>["no-shaking", "complex-motion", ...]</camera_motion>
```

对照分支使用完全相同的视频、prompt、训练步数和每条目标标签数，对相机语义组执行固定合法置换：运动强度和速度循环置换，方向取反，tracking 类型循环置换。因此每条目标仍符合 taxonomy，但至少 motion bucket 与视频不匹配。基础模型不训练，作为第三个对照。

不能同时要求“整套 label set 的总体分布完全不变”和“每条视频的 set 都不同”，因为重复最多的 camera label set 超过样本半数，严格 derangement 在数学上不存在。固定语义置换牺牲标签名称的边缘频率相等性，换取零正确目标和明确的错误视觉语义对应；构建摘要会保存完整置换表。

- LoRA：rank 32、alpha 64、dropout 0.05，仅匹配语言模型常用投影层。
- 学习率：`1e-5`，cosine，warmup 3%。
- 16 GPU、每卡 1 个视频、48 optimizer steps；保存 step 24、48。
- 主评测使用同一个 canonical prompt；通过后再用一个未训练的同义改写 prompt 做鲁棒性诊断。
- 指标：多标签 micro-F1、支持标签 macro-F1、粗粒度 motion bucket accuracy、格式有效率和逐标签指标。

阶段一通过要求：正确标签分支的 macro-F1 同时比基础模型和错配标签分支至少高 10 个百分点，格式有效率至少 95%，粗粒度 motion bucket accuracy 至少 50%，预测覆盖率至少 99%。

## 阶段二：相机能力向检测迁移

从阶段一选定的 correct/shuffled LoRA 继续执行完全相同的 64 步配对检测训练：32 个 DataA pair steps 与 32 个 DataB replay steps。检测训练和开发推理均不提供 camera 文本。

比较三条分支：

1. 无相机前置学习：既有 `/tmp/1res/caspr_gate1/scores/pair_rank`。
2. 正确相机前置学习后检测训练。
3. 错配相机前置学习后检测训练。

正确相机与错配相机分支的总更新步数完全相同，它们是判断“正确相机监督是否有效”的因果对照。无相机前置学习分支少 48 个相机 SFT steps，只作为现有方法基线；因此论文不能把 correct 与 no-pretext 的差值单独归因于相机标签内容。

正确相机分支必须同时优于另外两条分支：整体视频 AUC 至少 `+2` 点、pair accuracy 至少 `+3` 点、complex-motion AUC 至少 `+2` 点，而且至少两个视频来源的 AUC 为正增益。通过后才运行 VIF-Bench，允许相对无相机前置学习分支最多下降 `1.5` 点。

## 服务器执行

先把 GitHub 上以下文件复制到服务器相同相对路径：

- `tools/build_camera_pretext_transfer_gate.py`
- `scripts/camera_pretext_transfer/` 整个目录
- `scripts/caspr_gate1/runtime.py`
- `scripts/caspr_gate1/train_verdict_rank.py`

进入项目目录：

```bash
cd /input/workflow_58770161/workspace/test/cameramotion_det
RUN=scripts/camera_pretext_transfer/run_camera_pretext_transfer_gate.sh
```

### 1. 构建和审计数据

```bash
STAGE=build bash "$RUN"
cat /tmp/1res/camera_pretext_transfer_gate/data/camera_pretext_transfer_data_summary.json
```

### 2. 单卡两步 smoke

```bash
STAGE=smoke_correct bash "$RUN"
cat /tmp/1res/camera_pretext_transfer_gate/camera_sft/correct_smoke/all_results.json
tail -n 2 /tmp/1res/camera_pretext_transfer_gate/camera_sft/correct_smoke/trainer_log.jsonl
```

### 3. 阶段一正式训练

同一套 16 GPU 顺序执行：

```bash
STAGE=train_correct bash "$RUN"
STAGE=train_shuffled bash "$RUN"
```

有两套 16 GPU 时，可分别在两个容器执行上面两条命令，但两边需要能看到相同代码、模型和数据。

### 4. 阶段一推理与验收

```bash
STAGE=infer_camera_base bash "$RUN"
STAGE=infer_camera_correct_24 bash "$RUN"
STAGE=infer_camera_correct_48 bash "$RUN"
STAGE=infer_camera_shuffled_24 bash "$RUN"
STAGE=infer_camera_shuffled_48 bash "$RUN"
STAGE=eval_stage1 bash "$RUN"

cat /tmp/1res/camera_pretext_transfer_gate/camera_eval/stage1_gate_step_24.json
cat /tmp/1res/camera_pretext_transfer_gate/camera_eval/stage1_gate_step_48.json
```

如果 step 48 通过且不劣于 step 24，再运行 prompt 改写诊断：

```bash
CAMERA_CHECKPOINT_STEP=48 STAGE=infer_camera_paraphrase bash "$RUN"
cat /tmp/1res/camera_pretext_transfer_gate/camera_eval/correct_48_paraphrased.json
```

如果 step 24 更好，后续命令把 `CAMERA_CHECKPOINT_STEP=48` 改为 `24`。如果 24 到 48 仍明显上升但尚未稳定，只在看完指标后决定是否补到 96 步，不预先盲跑第二轮。

### 4.1 24→48 仍上升时的唯一 96 步复核

2026-07-13 的实际结果满足“correct 语义指标继续上升、correct 与 shuffled 差距扩大”，但尚未达到阶段一门槛。因此只允许 correct/shuffled 从各自 step 48 再补相同 48 步。续训同时监督 chat template 的 assistant 结束标记，以修复首轮只监督 camera tag 内容造成的停止格式退化；标签内容和两分支公平性不变。

```bash
STAGE=continue_correct_96 bash "$RUN"
STAGE=continue_shuffled_96 bash "$RUN"

STAGE=infer_camera_correct_96 bash "$RUN"
STAGE=infer_camera_shuffled_96 bash "$RUN"
STAGE=eval_stage1_96 bash "$RUN"

cat /tmp/1res/camera_pretext_transfer_gate/camera_eval/stage1_gate_step_96.json
```

这里的 `96` 表示累计 camera SFT 更新步数约为 96；第二段重新初始化 optimizer 和 48 步 cosine scheduler，因此它是低成本续训 gate，不冒充一次连续 optimizer-state 的正式 96 步训练。若该 gate 通过，正式复现实验再从初始模型用修正后的结束标记监督连续训练 96 步；若仍未通过，停止本路线。

### 4.2 最终采用：干净四轮学习曲线

2026-07-13 进一步核对后明确：约 750 条训练视频、16 GPU、每卡 batch 1 时，每个 epoch 约为 `ceil(750/16)=47` steps。因此旧 step 48 已约等于一轮，而不是一次极短训练。考虑到 correct 在第一轮末仍未平台，最终不采用上面的分段 96 步作为正式判断，改为从原始 detection checkpoint 使用修正后的结束标记监督，连续训练固定最多 192 steps，约四个 epochs。

Correct 与 shuffled 都使用完全相同的 192 步预算，在累计 step `48/96/144/192` 保存。选择规则预先固定为：使用最早同时通过全部阶段一检查的 checkpoint；若四个 checkpoint 均未通过，则停止，不再增加 epoch。

```bash
STAGE=train_correct_clean_4epoch bash "$RUN"
STAGE=train_shuffled_clean_4epoch bash "$RUN"

STAGE=infer_camera_clean_4epoch bash "$RUN"
STAGE=eval_stage1_clean_4epoch bash "$RUN"

cat /tmp/1res/camera_pretext_transfer_gate/camera_eval/stage1_clean_4epoch_curve.json
```

旧的 `continue_*_96` 命令仅保留用于复现历史决策过程，不再作为当前正式执行路径。

### 5. 阶段二训练、推理与验收

只有阶段一通过才执行。以下示例选择 step 48：

```bash
CAMERA_CHECKPOINT_STEP=48 STAGE=train_transfer_correct bash "$RUN"
CAMERA_CHECKPOINT_STEP=48 STAGE=train_transfer_shuffled bash "$RUN"
STAGE=score_transfer_correct bash "$RUN"
STAGE=score_transfer_shuffled bash "$RUN"
STAGE=eval_stage2 bash "$RUN"

cat /tmp/1res/camera_pretext_transfer_gate/detection_transfer/stage2_transfer_gate.json
```

阶段二通过后再补 VIF-Bench 保留评测；当前脚本不会在 DataA 门失败时自动消耗这部分算力。
