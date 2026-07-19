# 相机条件化几何残差最小验证

## 一句话目标

在不使用 DataA、检测 CoT、相机文本或复杂联合损失的前提下，验证“先估计并消除全局相机运动，再分析剩余时序几何异常”是否比原始运动特征更能泛化到 ViF-Bench 的 Real/Fake 检测。

这是一轮低成本方法门，不是最终 MLLM 实验。只有本门通过，才值得把相机几何特征接入 Qwen3-VL。

## 为什么先做这轮

- DataB 的 6766 条检测记录中，5739 条可以匹配 CameraBench sidecar；按唯一帧目录去重后为 5639 个视频。
- 5639 个唯一视频中，复杂运动 3126 个、轻微运动 1255 个、静止/无运动 1251 个、标签冲突 7 个。约 77.7% 含相机运动，因此“移动样本太少”不是主要瓶颈。
- 相机桶本身可以在 DataB 上得到约 57.08% Balanced ACC，说明相机运动与真假/来源存在数据偏置。实验必须按 `来源 x Real/Fake x 相机桶` 分层和加权，并设置错配几何控制。
- DataA 的生成质量、局部编辑分布与自动 CoT 质量会引入额外变量。本轮完全排除 DataA，避免无法判断提升来自相机几何还是 DataA/CoT。

## 特征与唯一训练目标

所有分支使用同一 16 帧、同一 DataB train/val 划分、同一冻结特征和同一小型 MLP；仅输入特征不同。

1. `DINO 外观`：冻结 DINOv2-Small 的帧级 CLS 时序统计。
2. `外观 + 原始运动`：DINO 外观加 RAFT 相邻帧稠密光流统计。
3. `外观 + 正确几何残差`：用 RAFT 对应点拟合每个相邻帧对的 homography 与 fundamental matrix，再统计去除该全局相机模型后的重投影/极线残差。
4. `外观 + 错配几何残差`：对当前帧对使用循环错位的几何模型，作为“仅增加同维数几何特征但相机条件不正确”的控制。

四个分支均只训练一个加权二分类 BCE。CameraBench labels/caption 只用于分层、加权和分桶报告，不作为分类器输入。阈值只在 held-out DataB validation 上选择；ViF-Bench 不参与模型选择或阈值选择。

## 数据与路径

- DataB detection：`/input/workflow_58770161/workspace/test/camb/camerabenchdataB-main/detection/v4vif_2766busterall_trainall.json`
- DataB camera sidecar：`/input/workflow_58770161/workspace/test/camb/camerabenchdataB-main/datab_cameramotion_labels_final/datab_cameramotion_labels_v2.jsonl`
- ViF-Bench 16 帧索引：`/input/workflow_58770161/workspace/test/cameramotion_det/eval/v4train-main/test_index_splits/splits_16`
- ViF-Bench camera sidecar：`/input/workflow_58770161/workspace/test/camb/camerabench_outputs/vifbench_cameramotion_labels_v2/datab_cameramotion_labels_v2.jsonl`
- RAFT 权重：`/home/admin/raft_large_C_T_SKHT_V2-ff5fadd5.pth`
- DINOv2-Small：`/home/admin/dinov2-small`

## 预注册验收

“正确几何残差”必须同时满足：

- DataB 与 ViF 特征覆盖率至少 99%；
- ViF AUROC 分别比“原始运动”和“错配几何”至少高 1.0 个百分点；
- 三个主要相机桶的 macro Balanced ACC 分别至少高 1.0 个百分点；
- 相对两个控制的配对 bootstrap AUROC 差值 95% CI 下界均大于 0；
- 静止/无运动桶的 Balanced ACC 相对原始运动下降不超过 1.0 个百分点；
- 当至少三个来源同时包含 Real/Fake 时，逐来源 Balanced ACC 胜率至少 60%。

若任一关键门失败，不进入 Qwen3-VL 融合，也不通过增加 DataA、CoT、RL 或损失项来补救。若通过，下一步只做一个小型冻结几何 projector/gate，并继续沿用原检测 SFT 损失。

## 输出与存储

- 可丢弃验证特征：`/tmp/1res/camera_geometric_residual_gate/v1/features/`
- 可丢弃完整运行目录：`/tmp/1res/camera_geometric_residual_gate/v1/`
- NAS 小型正式元数据与结果：`/input/workflow_58770161/workspace/test/cameramotion_det/res/camera_geometric_residual_gate/v1/`
- 本轮特征可以从现有帧和权重重建，不需要上传 OSS。只有门通过且后续融合要复用时，再决定是否归档。

## 服务器执行

先把本次 GitHub 更新的文件覆盖到服务器项目同名路径，然后执行：

```bash
cd /input/workflow_58770161/workspace/test/cameramotion_det
RUN=scripts/camera_geometric_residual_gate/run.sh

STAGE=preflight bash "$RUN"
STAGE=smoke NPROC_PER_NODE=1 bash "$RUN"
```

两步都通过后，用 16 张 GPU 后台运行全量验证：

```bash
cd /input/workflow_58770161/workspace/test/cameramotion_det
ROOT=/tmp/1res/camera_geometric_residual_gate/v1
mkdir -p "$ROOT"

nohup env \
STAGE=all \
NPROC_PER_NODE=16 \
KEEP_ALIVE_AFTER_RUN=1 \
bash scripts/camera_geometric_residual_gate/run.sh \
> "$ROOT/launcher.log" 2>&1 &

echo "launcher pid: $!"
```

查看进度：

```bash
find /tmp/1res/camera_geometric_residual_gate/v1/features/datab -name '*.npz' | wc -l
find /tmp/1res/camera_geometric_residual_gate/v1/features/vif_bench -name '*.npz' | wc -l
tail -n 80 /tmp/1res/camera_geometric_residual_gate/v1/launcher.log
```

预期数量分别为 5639 和 3160。最终只需提供：

```bash
cat /tmp/1res/camera_geometric_residual_gate/v1/eval/camera_geometric_residual_gate_summary.json
```

同一摘要也会持久化到：

`/input/workflow_58770161/workspace/test/cameramotion_det/res/camera_geometric_residual_gate/v1/eval/camera_geometric_residual_gate_summary.json`
