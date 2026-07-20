# 连续相机条件时序正常性专家：Gate 1 执行说明

## 这轮实验测什么

本轮不训练 Qwen3-VL，也不把 camera labels/caption 放进检测提示词。它只回答一个更靠前的问题：在真实视频上建模时，连续相机几何条件 `C` 是否能让时序取证证据 `Y` 的正常分布更容易建模，并在外部 AIGC 视频 benchmark 上稳定优于同容量无条件模型。

三个核心条件共用完全相同的实际有序帧、相邻 transition、DINOv2/RAFT 特征、scaler、PCA 和 flow 模型结构：

- `matched`：用当前视频的连续 RAFT camera context；
- `unconditional`：相同模型容量，但把 context 全部置零；
- `shuffled`：不重新训练，在评测时换入其他样本的连续 camera context，并插值到当前 transition 数。

DataB 不再默认为 16 帧。正式默认 `MAX_FRAMES=0`，表示使用 JSON 实际列出的全部帧；`n` 帧产生 `n-1` 条 transition。少于 3 帧的样本标为 `ctne_unavailable`，不补零伪造证据。

## 需要的模型和 Python 包

不需要下载新的大模型。继续使用：

- RAFT-Large：`/home/admin/raft_large_C_T_SKHT_V2-ff5fadd5.pth`
- DINOv2-Small：`/home/admin/dinov2-small`

条件 flow 从头训练。服务器只需额外确认小型 Python 包：

```bash
pip install nflows==0.14
```

现有 `torch 2.8.0`、`torchvision 0.23.0`、`transformers 4.57.3`、OpenCV 和 scikit-learn 继续复用。`MemorySlices` 的第三方 RAFT 权重不进入主 Gate 1。

## 输出与存储

| 内容 | 默认位置 | 类型 |
|---|---|---|
| manifest、审计、feature index、指标 CSV/JSON | `${PROJECT_ROOT}/res/camera_ctne_gate1/v1` | 持久化小文件，NAS |
| RAFT/DINO transition NPZ | `/tmp/1res/camera_ctne_gate1/v1/*_features` | 大文件，快速但易失的 `/tmp` |
| 六个 flow 与 preprocessor | `${META_ROOT}/model_bundle` | 小型复用模型，NAS |
| DataB validation 校准包 | `${META_ROOT}/calibration` | 小文件，NAS |

只有特征审计通过并确定后续复用时才上传 OSS。对应直接命令为：

```bash
ossutil64 cp -r /tmp/1res/camera_ctne_gate1/v1/datab_features/ oss://antsys-tamper/public/wong/skyra/selfcot/camerabench/ourexp/camera_flow/ctne_gate1_v1/datab_features/
```

```bash
ossutil64 cp -r /tmp/1res/camera_ctne_gate1/v1/vifbench_features/ oss://antsys-tamper/public/wong/skyra/selfcot/camerabench/ourexp/camera_flow/ctne_gate1_v1/vifbench_features/
```

GenBuster benchmark 特征若通过审计并需复用，同样放到 `ctne_gate1_v1/genbuster_benchmark_features/`。

## 两台服务器并行顺序

两台服务器都先把本目录中的全部代码复制到：

`/input/workflow_58770161/workspace/test/cameramotion_det/scripts/camera_ctne_gate1`

服务器 A 先构建 DataB manifest。这个步骤很快，而且服务器 B 的 overlap audit 会读取它：

```bash
cd /input/workflow_58770161/workspace/test/cameramotion_det
RUN=scripts/camera_ctne_gate1/run_server_a.sh
STAGE=preflight bash "$RUN"
STAGE=build bash "$RUN"
```

正式后台任务前，用 GPU 0 对 6 个样本跑一次 RAFT/DINO 运行时 smoke；它不是十分钟预检，也不产生论文指标：

```bash
STAGE=smoke bash "$RUN"
```

随后服务器 A 正式后台跑 DataB 提取、审计、六个 flow 和校准：

```bash
cd /input/workflow_58770161/workspace/test/cameramotion_det
ROOT=/tmp/1res/camera_ctne_gate1/v1
mkdir -p "$ROOT"

nohup env \
STAGE=all \
MAX_FRAMES=0 \
KEEP_ALIVE_AFTER_RUN=1 \
bash scripts/camera_ctne_gate1/run_server_a.sh \
> "$ROOT/server_a.log" 2>&1 &

echo "server A pid: $!"
```

`STAGE=all` 会重复执行可恢复的 build；已有且通过校验的 NPZ 默认复用。flow 的 3 个 seed × 2 个条件分给 GPU 0--5 并行，RAFT/DINO 提取使用全部 16 GPU。

服务器 B 可在 DataB manifest 已生成后并行提取 ViF-Bench：

```bash
cd /input/workflow_58770161/workspace/test/cameramotion_det
ROOT=/tmp/1res/camera_ctne_gate1/v1
mkdir -p "$ROOT"

nohup env \
STAGE=all_vif \
MAX_FRAMES=0 \
KEEP_ALIVE_AFTER_RUN=1 \
bash scripts/camera_ctne_gate1/run_server_b.sh \
> "$ROOT/server_b_vif.log" 2>&1 &

echo "server B pid: $!"
```

若后台提取完成后进入了 `/input/training/keep.sh`，正式评测前先停止这个仅用于保活的矩阵进程，再运行 ViF 评测：

```bash
pkill -f '/input/training/busy.py' || true
STAGE=eval_vif bash scripts/camera_ctne_gate1/run_server_b.sh
```

GenBuster 必须指向 Hugging Face 数据集中的 `benchmark` 帧根目录，不得用 DataB 已含的 train/test 目录冒充：

```bash
GENBUSTER_FRAME_ROOT=/实际的/GenBuster-200K/benchmark/parsed_frames \
STAGE=all_genbuster \
bash scripts/camera_ctne_gate1/run_server_b.sh

GENBUSTER_FRAME_ROOT=/实际的/GenBuster-200K/benchmark/parsed_frames \
STAGE=eval_genbuster \
bash scripts/camera_ctne_gate1/run_server_b.sh
```

两套外部结果都有后合并最终门：

```bash
STAGE=combine bash scripts/camera_ctne_gate1/run_server_b.sh
```

## 进度检查

提取进度按已有 NPZ 数查看：

```bash
find /tmp/1res/camera_ctne_gate1/v1/datab_features/features -name '*.npz' | wc -l
find /tmp/1res/camera_ctne_gate1/v1/vifbench_features/features -name '*.npz' | wc -l
```

训练进度：

```bash
tail -n 20 /tmp/1res/camera_ctne_gate1/v1/train_logs/job_0.log
```

正式小文件结果：

```bash
cat /input/workflow_58770161/workspace/test/cameramotion_det/res/camera_ctne_gate1/v1/features/datab_feature_audit.json
cat /input/workflow_58770161/workspace/test/cameramotion_det/res/camera_ctne_gate1/v1/eval/vifbench/ctne_gate1_summary.json
cat /input/workflow_58770161/workspace/test/cameramotion_det/res/camera_ctne_gate1/v1/eval/ctne_gate1_final_decision.json
```

## 验收边界

Gate 1 通过要求 camera 的增量来自 `matched > unconditional` 且 `matched > shuffled`，而不是 camera-only 来源捷径；ViF-Bench 与 GenBuster benchmark 至少一个有不低于 1 点的 AUROC 或跨生成器 Macro Balanced ACC 增益，另一个不得下降超过 0.5 点。Gate 1 通过后才做冻结 Qwen 检测分数融合；未通过则停止 camera 作为主贡献，不继续烧 pose token、caption SFT 或 RL。
