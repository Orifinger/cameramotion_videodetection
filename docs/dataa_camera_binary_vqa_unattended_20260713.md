# DataA 平衡二元相机问答无人值守实验

## 为什么今晚不直接跑一个最终大实验

上一轮失败的是“一个视频自由生成整套 camera labels”：它缺少逐标签负监督，结果塌缩到多数 motion bucket。今晚最需要判清的不是某组 detection 超参数，而是把每个 primitive 拆成平衡 Yes/No 后，模型是否会看视频学习相机运动。这个能力门未通过时，直接做联合 detection、DPO 或 GRPO 无法解释失败原因；能力门通过后，才有依据把 camera score vector 作为检测辅助目标。

## 两套 16 GPU 的分工

两套实验只改变起始 checkpoint，其余条件逐项相同：

1. 检测模型起点：判断当前计划实际要使用的 detection checkpoint 能否恢复相机能力。
2. 通用模型起点：判断相同监督对干净 Qwen3-VL-Instruct 是否可学，用于区分任务设计问题与 detection 专项微调造成的能力损失。

如果当晚只有一套机器，优先检测模型起点。它若通过，已经同时回答“任务可学”和“实际起点可用”；它若失败，再补通用起点来定位原因。

## 实验顺序

脚本按以下顺序自动执行：

1. 审计 1080-case manifest 和所有原视频路径，按固定 train/test identity 构建逐 primitive 平衡问答。
2. 未训练起点在正确视频开发集上打 candidate score。
3. 使用原始 real MP4、8 FPS、rank-64 LoRA 最多训练 5 轮；训练最多占 4.5 小时，至少完成一轮。
4. 评测第一轮 checkpoint 的正确视频结果。
5. 评测最终 checkpoint 的正确视频、对立标签视频置换、无视频三种条件。
6. 汇总 AP、AUC、balanced accuracy、paired question accuracy 和固定 gate。
7. 小结果复制到 NAS；整个运行目录连同两个 adapter 自动上传 OSS。失败退出也执行归档。

## 服务器文件

从 GitHub 复制以下新增文件到服务器相同相对路径：

- `scripts/camera_binary_vqa/__init__.py`
- `scripts/camera_binary_vqa/build_data.py`
- `scripts/camera_binary_vqa/runtime.py`
- `scripts/camera_binary_vqa/train.py`
- `scripts/camera_binary_vqa/score.py`
- `scripts/camera_binary_vqa/evaluate.py`
- `scripts/camera_binary_vqa/summarize_gate.py`
- `scripts/camera_binary_vqa/run_unattended.sh`

## 离开电脑前的短检查

两套机器分别先运行一次两样本 preflight。它会实际打开原 MP4、按 8 FPS 处理并完成 `Yes/No` candidate scoring，但不上传 OSS：

```bash
cd /input/workflow_58770161/workspace/test/cameramotion_det

STAGE=preflight \
RUN_NAME=detection_checkpoint_start \
MODEL_PATH=/tmp/1res/v4vif_2766busterall_trainall_5epoch/checkpoint-2115 \
bash scripts/camera_binary_vqa/run_unattended.sh
```

第二套机器：

```bash
cd /input/workflow_58770161/workspace/test/cameramotion_det

STAGE=preflight \
RUN_NAME=generic_instruct_start \
MODEL_PATH=/home/admin/Qwen3-VL-8B-Instruct \
bash scripts/camera_binary_vqa/run_unattended.sh
```

只有 preflight 正常结束后才启动 8 小时任务。若第二套机器缺少 manifest 所指向的 `/tmp` 原视频，脚本会在加载模型前明确失败；先恢复相同 DataA 视频缓存，不要让它空跑其他不等价实验。

## 无人值守完整命令

第一套机器运行检测模型起点：

```bash
cd /input/workflow_58770161/workspace/test/cameramotion_det

STAGE=all \
RUN_NAME=detection_checkpoint_start \
MODEL_PATH=/tmp/1res/v4vif_2766busterall_trainall_5epoch/checkpoint-2115 \
AUTO_UPLOAD_OSS=1 \
bash scripts/camera_binary_vqa/run_unattended.sh
```

第二套机器运行通用模型起点：

```bash
cd /input/workflow_58770161/workspace/test/cameramotion_det

STAGE=all \
RUN_NAME=generic_instruct_start \
MODEL_PATH=/home/admin/Qwen3-VL-8B-Instruct \
AUTO_UPLOAD_OSS=1 \
bash scripts/camera_binary_vqa/run_unattended.sh
```

脚本默认使用 16 GPU、每个 rank 4 个 CPU threads、8 FPS、`video_max_pixels=16384`、LoRA rank 64、学习率 `2e-4`、最多 5 epochs 和 16200 秒训练上限。不要为两套机器修改不同参数，否则起点对照失效。

## 产物位置

检测模型起点：

```text
/tmp/1res/dataa_camera_binary_vqa/detection_checkpoint_start/
/input/workflow_58770161/workspace/test/cameramotion_det/res/dataa_camera_binary_vqa/detection_checkpoint_start/
oss://antsys-tamper/public/wong/skyra/selfcot/camerabench/ourexp/dataa_camera_binary_vqa/detection_checkpoint_start/
```

通用模型起点：

```text
/tmp/1res/dataa_camera_binary_vqa/generic_instruct_start/
/input/workflow_58770161/workspace/test/cameramotion_det/res/dataa_camera_binary_vqa/generic_instruct_start/
oss://antsys-tamper/public/wong/skyra/selfcot/camerabench/ourexp/dataa_camera_binary_vqa/generic_instruct_start/
```

自动上传若因 OSS 客户端临时失败，分别补这一条即可：

```bash
ossutil64 cp -r /tmp/1res/dataa_camera_binary_vqa/detection_checkpoint_start/ oss://antsys-tamper/public/wong/skyra/selfcot/camerabench/ourexp/dataa_camera_binary_vqa/detection_checkpoint_start/
```

通用起点只需将命令中的两处 `detection_checkpoint_start` 改为 `generic_instruct_start`。

## 明天需要提供的结果

每套机器只需要提供以下两个小文件；若状态失败，再补 `pipeline.log` 和 `trainer_log.jsonl`：

```text
/input/workflow_58770161/workspace/test/cameramotion_det/res/dataa_camera_binary_vqa/<运行名>/eval/gate_summary.json
/input/workflow_58770161/workspace/test/cameramotion_det/res/dataa_camera_binary_vqa/<运行名>/data/data_summary.json
```

`gate_summary.json` 已包含未训练起点、第一轮、最终轮和三种视觉条件的核心指标。不要只发一个最终 accuracy；起点差值与视觉控制才决定这条路线是否真实可行。
