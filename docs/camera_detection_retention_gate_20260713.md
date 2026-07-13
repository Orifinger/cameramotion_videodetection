# 相机二元问答适配器的检测保留诊断

## 这一步回答什么

检测模型起点已经通过平衡二元相机问答与视觉依赖门。下一步先判断：只训练 camera VQA 得到的 LoRA 挂回原检测模型后，在不提供 camera 文本的原始检测 prompt 下，是否保留 DataA 局部编辑检测与解释格式。

这不是联合训练，也不要求 camera-only adapter 提高检测。它只测能力注入的直接代价，为后续联合训练中的 detection replay 强度提供依据。

## 严格对照

- 对照：`/tmp/1res/v4vif_2766busterall_trainall_5epoch/checkpoint-2115`。
- 诊断模型：同一个 checkpoint 合并最终 binary-camera LoRA。
- 两者使用完全相同的 40step_v3 DataA 记录、321 个固定开发 case、16 帧、图像像素上限、system prompt、user prompt、生成参数和评测脚本。
- 检测推理不提供 camera caption、camera labels、bbox、mask 或外部光流特征。
- 评测直接解析完整生成结果中的 `<answer>Real/Fake</answer>`，并同时报告解释证据格式和 IoU；不使用新加的 `<verdict>` 候选分数代理原检测任务。

## 数据与路径

- 当前 detection JSON：`/input/workflow_58770161/workspace/test/cameramotion_det/res/dataA_v1/autolabel/dataa_vace_grounded_cot_40step_v3_sft_clean.json`。
- 固定开发身份来源：`/input/workflow_58770161/workspace/test/cameramotion_det/tools/data/camera_motion_splits/dataA_test.json`。
- 当前帧目录：`/tmp/cameramotion_det/dataA_v1/autolabel/dataa_vace_grounded_cot_frames_40step_v3`。
- camera adapter：`/tmp/1res/dataa_camera_binary_vqa/detection_checkpoint_start/train/final`。
- 临时运行目录：`/tmp/1res/camera_detection_retention/detection_checkpoint_start`。
- NAS 小结果：`/input/workflow_58770161/workspace/test/cameramotion_det/res/camera_detection_retention/detection_checkpoint_start`。

## 验收标准

- 两个模型的预测覆盖率都不低于 99%，camera 模型格式有效率不低于 95%。
- camera 模型相对原 checkpoint 的 Balanced ACC、Fake F1 和 pair accuracy 各下降不超过 3 个百分点。
- Evidence IoU 与 evidence 输出率作为解释保留诊断报告，但不单独决定当前门，因为原 checkpoint 在 DataA 上的证据指标本身较低。
- 通过只表示直接挂载 camera adapter 没有明显破坏 DataA 检测；不表示 camera 提高了检测。
- 未通过不停止 camera 方向，而是要求下一阶段从头进行 `detection replay + camera auxiliary` 联合训练，不能把 camera-only adapter 直接当检测模型。

## 服务器准备

只恢复最终 adapter：

```bash
ossutil64 cp -r oss://antsys-tamper/public/wong/skyra/selfcot/camerabench/ourexp/dataa_camera_binary_vqa/detection_checkpoint_start/train/final/ /tmp/1res/dataa_camera_binary_vqa/detection_checkpoint_start/train/final/
```

还需确认原 detection checkpoint、当前 40step_v3 图片帧和外部 `infer_dataa.py/eval_dataa.py` 已经恢复。合并后的完整模型只是可再生临时文件，不上传 OSS。

## 快速预检

```bash
cd /input/workflow_58770161/workspace/test/cameramotion_det

STAGE=preflight \
MODEL_PATH=/tmp/1res/v4vif_2766busterall_trainall_5epoch/checkpoint-2115 \
ADAPTER_PATH=/tmp/1res/dataa_camera_binary_vqa/detection_checkpoint_start/train/final \
bash scripts/camera_detection_retention/run.sh
```

预检只审计依赖、16 张 GPU、642 条当前 detection 记录和全部图片路径，不加载模型推理。

## 正式后台执行

```bash
cd /input/workflow_58770161/workspace/test/cameramotion_det

ROOT=/tmp/1res/camera_detection_retention/detection_checkpoint_start
mkdir -p "${ROOT}"

nohup env \
STAGE=all \
MODEL_PATH=/tmp/1res/v4vif_2766busterall_trainall_5epoch/checkpoint-2115 \
ADAPTER_PATH=/tmp/1res/dataa_camera_binary_vqa/detection_checkpoint_start/train/final \
KEEP_ALIVE_AFTER_RUN=1 \
bash scripts/camera_detection_retention/run.sh \
> "${ROOT}/launcher.log" 2>&1 &

echo "launcher pid: $!"
```

脚本依次构建数据、合并 LoRA、推理原 checkpoint、推理 camera 模型、评测并写入 NAS，最后才执行 `/input/training/keep.sh`。默认在 16 张 96G GPU 上每卡启动 2 个独立生成进程，共 32 个数据分片，以提高自回归生成阶段的利用率；每条样本仍只处理一次。如该服务器环境出现显存不足，在命令中设置 `WORKERS_PER_GPU=1` 可退回一卡一进程。如需重跑已有输出，默认会覆盖各 rank 的旧预测。

## 需要提供的结果

完成后提供以下三个小文件：

```text
/input/workflow_58770161/workspace/test/cameramotion_det/res/camera_detection_retention/detection_checkpoint_start/eval/camera_detection_retention_summary.json
/input/workflow_58770161/workspace/test/cameramotion_det/res/camera_detection_retention/detection_checkpoint_start/eval/base/dataa_detection_base_summary.json
/input/workflow_58770161/workspace/test/cameramotion_det/res/camera_detection_retention/detection_checkpoint_start/eval/camera_adapter/dataa_detection_camera_adapter_summary.json
```
