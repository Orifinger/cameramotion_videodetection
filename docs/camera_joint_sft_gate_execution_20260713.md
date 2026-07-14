# 同 16 帧二元相机辅助与检测回放联合训练

## 这轮实验测什么

从现有 DataB 检测 checkpoint 出发，比较三个等记录数、等训练步数的 LoRA-SFT 分支：

1. **仅检测回放分支**：相机任务位置换成等量检测样本，控制继续训练和计算量。
2. **正确相机监督分支**：检测 replay 与类别均衡的二元相机 VQA 按 1:1 混合。
3. **翻转相机监督分支**：相机问题和 16 帧不变，每条 Yes/No 答案都翻转，且总体 Yes/No 数量不变。

相机任务对每个 primitive 只使用一个固定问题，不为同一视频复制 25 个提示词。Camera caption 仅保留在 split 审计中，不作为第一轮训练目标。检测 prompt 始终保持原样；检测推理也不提供 camera labels、caption 或其他外部相机文本。

这轮还不是 RL。它先回答三个前置问题：正确监督是否比错误监督更可学、相机回答是否依赖画面、联合训练是否保留原检测接口。满足条件后才运行短程 GRPO。

## 固定输入

```text
/tmp/1res/v4vif_2766busterall_trainall_5epoch/checkpoint-2115

/input/workflow_58770161/workspace/test/cameramotion_det/res/dataA_v1/autolabel/dataa_vace_grounded_cot_40step_v3_sft_clean.json
/input/workflow_58770161/workspace/test/cameramotion_det/camera/camerajson/dataa_cameramotion_labels_40step_v3.jsonl

/input/workflow_58770161/workspace/test/camb/camerabenchdataB-main/detection/v4vif_2766busterall_trainall.json
/input/workflow_58770161/workspace/test/camb/camerabenchdataB-main/datab_cameramotion_labels_final/datab_cameramotion_labels_v2.jsonl
```

正式构建要求 DataA 有 1080 个完整 real/fake case。视觉输入直接使用 detection JSON 已引用的 16 帧；旧 755/1076-case 文件和旧 split 不复用。

## 生成文件

默认目录为 `/tmp/1res/camera_joint_sft_gate/data`。

| 文件 | 用途 |
|---|---|
| `dataa_40step_v3_split_manifest.jsonl` | 按 case、VACE 来源和 coarse motion bucket 分层的 70:30 split |
| `dataa_train_detection.json` | DataA train 的完整 Real/Fake pair |
| `dataa_test_detection.json` | DataA 开发留出检测集 |
| `datab_detection_replay.json` | DataB 分层、Real/Fake 平衡的检测 replay |
| `camera_train_correct.json` | 全部可平衡 primitive 的正确 Yes/No 相机样本 |
| `camera_train_shuffled.json` | 输入不变、答案逐条翻转的错误监督控制 |
| `joint_sft_detection_only.json` | 仅检测回放训练分支 |
| `joint_sft_correct_camera.json` | 正确相机监督联合分支 |
| `joint_sft_shuffled_camera.json` | 翻转相机监督联合分支 |
| `camera_dev_matched_frames.jsonl` | 正确 16 帧的相机开发集 |
| `camera_dev_opposite_frames.jsonl` | 同问题下换成相反答案视频帧的视觉控制 |
| `camera_dev_no_frames.jsonl` | 移除全部帧的文本先验控制 |
| `camera_joint_sft_data_summary.json` | split、覆盖、平衡、分支数量和 SHA-256 审计 |

小型审计和评测 JSON 会同步到：

```text
/input/workflow_58770161/workspace/test/cameramotion_det/res/camera_joint_sft_gate
```

## 1. 文件与环境预检

预检只检查路径、命令、Python 依赖和 GPU 数量，不加载模型，也不等待 10 分钟。

```bash
cd /input/workflow_58770161/workspace/test/cameramotion_det

RUN=scripts/camera_joint_sft_gate/run.sh
STAGE=preflight bash "$RUN"
```

当前服务器的 LlamaFactory 代码根目录与数据目录分开放置，显式写法为：

```bash
LLAMAFACTORY_ROOT=/input/workflow_58770161/workspace/test/test_selfcot/LlamaFactory/LlamaFactory \
LLAMAFACTORY_DATA_DIR=/input/workflow_58770161/workspace/test/test_selfcot/Skyra/train/LLaMA-Factory/data \
STAGE=preflight bash "$RUN"
```

脚本分别通过 `examples/deepspeed/ds_z2_config.json` 和已有 `dataset_info.json` 自动识别代码根目录与数据目录，并在屏幕上打印最终选择。`build` 只允许更新已经存在的 `dataset_info.json`，不会在错误路径新建空目录。

当前环境只需要既有 LlamaFactory、PEFT、`qwen_vl_utils` 和 `transformers==4.57.3`。本轮不需要下载 RAFT、DINOv2、SEA-RAFT、额外 reward model 或新仓库。

## 2. 构建并注册数据

```bash
STAGE=build bash "$RUN"
```

查看紧凑审计：

```bash
python - <<'PY'
import json

p = "/tmp/1res/camera_joint_sft_gate/data/camera_joint_sft_data_summary.json"
x = json.load(open(p, encoding="utf-8"))
print("split:", x["split"]["total_cases"], x["split"]["train_cases"], x["split"]["test_cases"])
print("camera:", x["camera_supervision"]["train_records"], x["camera_supervision"]["train_answer_counts"])
print("labels:", x["camera_supervision"]["train_supported_labels"], x["camera_supervision"]["dev_supported_labels"])
print("task ratio:", x["joint_task_ratio"])
print("branches:", x["branch_counts"])
print("integrity:", x["integrity"])
PY
```

训练前必须确认：

- `total_cases=1080`，train/test 交集为空；
- 32 个左右的相机 primitive 在 train/dev 同时有正负支持；
- 相机训练 Yes/No 数量完全相同；
- 正确与翻转监督输入相同、每条答案相反、答案边际不变；
- camera 和 detection 记录约为 1:1；三个联合训练文件记录数完全相同；
- DataB replay 的 Real/Fake 数量相同；
- detection prompt 没有 camera 文本泄漏。

本地旧 1076-case 数据的结构干跑得到 5528 条相机记录、5528 条检测记录和每分支 11056 条；正式 1080-case 数字以服务器审计为准。

## 3. 两步训练冒烟

```bash
STAGE=smoke bash "$RUN"
```

它只验证 LlamaFactory 能读取混合图文任务并完成反向传播，不属于实验结果。

## 4. 三分支正式训练

每个分支使用一套 16 GPU。两台服务器可分别运行正确监督和翻转监督，任一结束后再运行仅检测回放。

```bash
STAGE=train_correct_camera bash "$RUN"
```

```bash
STAGE=train_shuffled_camera bash "$RUN"
```

```bash
STAGE=train_detection_only bash "$RUN"
```

共同设置：LoRA rank 64、alpha 128、dropout 0.05、学习率 `2e-4`、5 epochs、全局 batch 16、cosine scheduler、warmup 0.03、bf16，并按 epoch 保存 checkpoint。之所以不是 1 epoch，是此前同一二元相机协议已经实测第 1 epoch 尚未学稳，第 5 epoch 才通过视觉依赖门。

离开电脑前可按下面方式后台运行；训练正常结束后才会进入你已有的 `keep.sh`：

```bash
ROOT=/tmp/1res/camera_joint_sft_gate
mkdir -p "$ROOT"

nohup env \
STAGE=train_correct_camera \
KEEP_ALIVE_AFTER_RUN=1 \
bash "$RUN" \
> "$ROOT/train_correct_camera_launcher.log" 2>&1 &

echo "launcher pid: $!"
```

另两个分支只需替换 `STAGE` 和日志名。若要评测某个 epoch checkpoint，可通过 `CORRECT_ADAPTER`、`SHUFFLED_ADAPTER` 或 `DETECTION_ONLY_ADAPTER` 环境变量覆盖默认 final adapter 路径。

## 5. 先做正确监督与翻转监督的低成本相机门

正确相机和翻转相机两个分支完成后立即执行，不等待仅检测回放分支：

```bash
STAGE=eval_camera_pair bash "$RUN"
```

该步骤直接计算每个问题的 `Yes` 对 `No` logit，不依赖生成格式，分别评测正确帧、相反答案帧和无帧。核心指标是 Overall/Macro AP、Balanced ACC、ROC-AUC 和 paired question accuracy。逐样本分数留在 `/tmp`；评测 JSON 和汇总会自动复制到 NAS。

```text
/tmp/1res/camera_joint_sft_gate/camera_eval/correct_vs_flipped_camera_gate_summary.json
```

若中途退出，可分别恢复，不必重跑已完成模型：

```bash
STAGE=eval_camera_correct bash "$RUN"
STAGE=eval_camera_shuffled bash "$RUN"
STAGE=summarize_camera_pair bash "$RUN"
```

只有该门通过，才继续第 6 节采样检查和仅检测回放训练。未通过时先停止并检查 per-label 指标，不花第三次训练计算。

仅检测回放分支训练完成后，只补测该分支并复用已有正确/翻转结果：

```bash
STAGE=eval_camera_detection_only bash "$RUN"
STAGE=summarize bash "$RUN"
```

## 6. 短程 RL 前的采样检查

```bash
STAGE=readiness_correct bash "$RUN"
```

可选错误监督控制：

```bash
STAGE=readiness_shuffled bash "$RUN"
STAGE=summarize bash "$RUN"
```

可验证奖励固定为：输出严格 Yes/No 得 `0.1`，答案正确再得 `0.9`。`pass@8` 只检查采样是否覆盖正确动作以及组内奖励是否有方差，不把它当作检测提升。

该检查通过后再运行仅检测回放分支：

```bash
STAGE=train_detection_only bash "$RUN"
```

仅检测分支完成后，如需三个模型的完整相机汇总，再执行：

```bash
STAGE=eval_camera_all bash "$RUN"
STAGE=summarize bash "$RUN"
```

总验收文件：

```text
/tmp/1res/camera_joint_sft_gate/camera_eval/joint_sft_camera_gate_summary.json
```

## 7. 无相机文本的检测保留

先做较便宜的 DataA 三分支比较：

```bash
STAGE=dataa_detection_only bash "$RUN"
STAGE=dataa_correct_camera bash "$RUN"
STAGE=dataa_shuffled_camera bash "$RUN"
```

只有正确监督分支既学到视觉相机能力、又没有出现 Yes/No 接口接管时，再运行 VIF-Bench：

```bash
STAGE=vif_detection_only bash "$RUN"
STAGE=vif_correct_camera bash "$RUN"
STAGE=vif_shuffled_camera bash "$RUN"
```

DataA test 是本轮开发留出集，不包装为全新论文 test。起始 checkpoint 已见过 DataB，因此 DataB 只用于 replay，不能称 held-out。VIF-Bench 推理也严格使用原检测 prompt 和 `no_camera` 条件。

## 验收规则

- 正确监督的 Macro AP 比翻转监督高至少 3 点，或 Balanced ACC 高至少 5 点。
- 正确监督分支在相反答案帧上 Balanced ACC 至少下降 10 点，或无帧至少下降 8 点。
- train/dev 至少 20 个相机 primitive 有正负支持，开发集覆盖率至少 99%。
- `pass@8` 至少 20% 的样本具有非恒定组内奖励；不满足时不启动 GRPO。
- DataA/VIF 检测完整报告格式有效率、Balanced ACC、Fake F1 和 pair accuracy。严重接口接管，或正确监督明显差于两个控制，都不进入 RL。

这组门通过只说明联合训练配方值得进入短程 RL，不说明相机已经提高 AIGC 检测。最终方法结论仍必须由无相机文本的留出 DataA 和外部 VIF-Bench 检测结果建立。

## 存储

- 数据、adapter、原始预测和 rollouts 默认放 `/tmp/1res/camera_joint_sft_gate`。
- split、审计与评测小 JSON 自动复制到 NAS。
- 只有结果通过、确定要复用的正式 adapter 才需要上传 OSS；失败分支和中间 checkpoint 不上传。
- 若要跨服务器搬运一个确定保留的 adapter，使用一条简单命令，例如：

```bash
ossutil64 cp -r /tmp/1res/camera_joint_sft_gate/train/correct_camera/ oss://antsys-tamper/public/wong/skyra/selfcot/camerabench/ourexp/camera_joint_sft_gate/train/correct_camera/
```
