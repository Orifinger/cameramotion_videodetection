# VIF-Bench 相机适配器检测保留诊断

## 目的

比较完整 DataB 检测 SFT checkpoint 与“同一 checkpoint 合并 DataA 平衡二元相机问答 LoRA”后的模型，在 VIF-Bench 原检测协议中的能力变化。两者使用完全相同的 `no_camera` system/user prompt、测试分片和确定性解码；推理不提供 camera label、camera caption 或其他外部相机上下文。

这项实验只判断 camera-only adapter 是否损伤既有全生成视频检测能力，不是相机辅助检测收益实验。VIF-Bench 没有参加本轮 DataA camera adapter 训练，因此是适配器训练之外的外部分布；但项目此前已反复查看过 VIF-Bench 结果，所以不能称为全新的论文最终模型选择测试。

## 模型与路径

- 原检测 checkpoint：`/tmp/1res/v4vif_2766busterall_trainall_5epoch/checkpoint-2115`。
- 相机 LoRA：`/tmp/1res/dataa_camera_binary_vqa/detection_checkpoint_start/train/final`。
- V4Train 评测目录：`/input/workflow_58770161/workspace/test/cameramotion_det/eval/v4train-main/eval`。
- 当前服务器 VIF-Bench 分片目录：`/input/workflow_58770161/workspace/test/cameramotion_det/eval/v4train-main/test_index_splits/splits_16`；runner 同时兼容分片位于评测目录内部的旧布局。
- 原检测推理脚本：上述目录中的 `infer2_5_3.sh`。
- 原官方配对评测：上述目录中的 `eval.py`。
- 临时运行目录：`/tmp/1res/camera_detection_retention/vifbench_detection_checkpoint_start`。
- 持久化小结果：`/input/workflow_58770161/workspace/test/cameramotion_det/res/camera_detection_retention/vifbench_detection_checkpoint_start`。

合并后的完整 camera 模型、逐样本生成结果和组合预测留在 `/tmp`。它们属于可从 base checkpoint、LoRA 和 VIF-Bench 数据重建的诊断中间产物，不占 NAS，也不上传 OSS；评测 JSON、CSV、prompt hash、索引审计和 pipeline log 持久化到 NAS。

## 并行方式

默认同时启动原模型和 camera 模型的 16 个 VIF-Bench rank。每张 96G GPU 上各加载一个原模型进程和一个 camera 模型进程，共两个推理进程；这是与第一台服务器 `WORKERS_PER_GPU=2` 相同的显存量级，用来提高自回归推理阶段的 GPU 利用率。若设备侧出现显存或运行时问题，可设置 `PARALLEL_MODELS=0` 改为顺序运行，数据和评测协议不变。

## 预检

```bash
cd /input/workflow_58770161/workspace/test/cameramotion_det

STAGE=preflight \
MODEL_PATH=/tmp/1res/v4vif_2766busterall_trainall_5epoch/checkpoint-2115 \
ADAPTER_PATH=/tmp/1res/dataa_camera_binary_vqa/detection_checkpoint_start/train/final \
bash scripts/camera_detection_retention/run_vifbench.sh
```

预检会验证 16 个 index shard、索引中全部帧目录、模型和 adapter、V4Train 推理与评测脚本、16 张 GPU，并保存 system prompt 与 no-camera user suffix 的 SHA-256。`camera_context_provided` 必须为 `false`，user suffix 中不得含 camera placeholder。

## 正式后台执行

```bash
cd /input/workflow_58770161/workspace/test/cameramotion_det

ROOT=/tmp/1res/camera_detection_retention/vifbench_detection_checkpoint_start
mkdir -p "${ROOT}"

nohup env \
STAGE=all \
MODEL_PATH=/tmp/1res/v4vif_2766busterall_trainall_5epoch/checkpoint-2115 \
ADAPTER_PATH=/tmp/1res/dataa_camera_binary_vqa/detection_checkpoint_start/train/final \
PARALLEL_MODELS=1 \
KEEP_ALIVE_AFTER_RUN=1 \
bash scripts/camera_detection_retention/run_vifbench.sh \
> "${ROOT}/launcher.log" 2>&1 &

echo "launcher pid: $!"
```

完成后脚本自动执行 `/input/training/keep.sh`。查看运行状态：

```bash
tail -f /tmp/1res/camera_detection_retention/vifbench_detection_checkpoint_start/pipeline.log
```

两个模型的详细进度分别位于：

```text
/tmp/1res/camera_detection_retention/vifbench_detection_checkpoint_start/inference/base/inference.log
/tmp/1res/camera_detection_retention/vifbench_detection_checkpoint_start/inference/camera_adapter/inference.log
```

## 指标与验收

自带汇总严格复刻现有 `eval.py` 的每个生成模型 Real/Fake 配对指标和跨生成模型宏平均，同时额外报告覆盖率、格式有效率和 strict-valid pair 指标。原 `eval.py` 会把非 `Real` 的输出统一编码为 Fake，因此格式错误可能被隐式计为 Fake；新增汇总不会隐藏这个问题。

验收条件：

- 原模型和 camera 模型预测覆盖率均至少 99%。
- 两者 `<answer>` 格式有效率均至少 99%。
- 两者具有相同且非空的生成模型子集。
- camera 模型相对原模型的跨生成模型平均 Balanced ACC 降幅不超过 3 个百分点。
- camera 模型相对原模型的跨生成模型平均 Fake F1 降幅不超过 3 个百分点。

最终汇总：

```bash
cat /input/workflow_58770161/workspace/test/cameramotion_det/res/camera_detection_retention/vifbench_detection_checkpoint_start/eval/vifbench_camera_adapter_retention_summary.json
```

还会保留两个模型各自的详细评测 JSON、官方 CSV 和官方 eval log。需要解释结果时，优先提供上述 summary；如果未通过，再补充两个详细评测 JSON。
