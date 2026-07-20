# 最终真假监督的连续相机交互判别门

这个实验只回答一个问题：在完全相同的 DataB Real/Fake 监督下，连续相机几何是否能给时序证据带来可迁移的检测增量。

它不是上一轮 `p(Y|C, Real)` 正常性 flow 的重复。上一轮只学习真实视频分布；本轮直接用 Real/Fake BCE 训练最终判别器，使 camera 与检测终点在同一损失中耦合。Qwen3-VL、camera caption、camera label 和外部 camera 文本都不参与本门。

## 输入与模型

- 复用 CTNE 已审核的 RAFT/DINO 特征，不重新抽取，也不下载新权重。
- `Y_t`：每个相邻帧 transition 的 DINO/flow 时序证据。
- `C_t`：同一 transition 的连续全局相机几何。
- 保留每个样本实际 transition 数，通过 padding mask 做变长 attention、mean 和 max pooling；不假设 DataB 有 16 帧。
- `matched`：`C_t` 通过 FiLM 调制 `Y_t`，直接预测 Real/Fake。
- `zero_camera`：等参数、同 seed、同初始化，但 camera 输入全为零。
- `camera_only`：等参数、同 seed、同初始化，但时序证据全为零。
- `shuffled_camera`：不另训模型，只在外部评测时把 matched 模型的 camera 换成其他样本的 camera。

三个训练条件各跑 seeds `13/37/73`。PCA/scaler 只在 DataB train 的 Real+Fake 上拟合；每个视频只在拟合统计时重采样为 16 个 transition 以实现视频等权，分类训练和评测始终使用完整变长序列。阈值只由 DataB validation 确定，ViF-Bench 禁止调参。

## 存储合同

- 可丢弃准备数组：`/tmp/1res/camera_discriminative_gate/v1/`。
- 持久小文件、九个小分类器和正式指标：`/input/workflow_58770161/workspace/test/cameramotion_det/res/camera_discriminative_gate/v1/`。
- 复用的大特征仍位于两台机器各自的 `/tmp/1res/camera_ctne_gate1/v1/*_features/`。它们现在成为可复用正式输入，应在容器回收前上传 OSS。

服务器 A 的 DataB 特征：

```bash
ossutil64 cp -r /tmp/1res/camera_ctne_gate1/v1/datab_features/ oss://antsys-tamper/public/wong/skyra/selfcot/camerabench/ourexp/camera_flow/ctne_gate1_v1/datab_features/
```

服务器 B 的 ViF-Bench 特征：

```bash
ossutil64 cp -r /tmp/1res/camera_ctne_gate1/v1/vifbench_features/ oss://antsys-tamper/public/wong/skyra/selfcot/camerabench/ourexp/camera_flow/ctne_gate1_v1/vifbench_features/
```

## 执行

先在服务器 A 快速检查。若当前是 `keep.sh` 在占卡，先结束自己的保活进程：

```bash
cd /input/workflow_58770161/workspace/test/cameramotion_det
pkill -f '/input/training/busy.py' || true
RUN=scripts/camera_discriminative_gate/run_server_a.sh
STAGE=preflight bash "$RUN"
STAGE=smoke bash "$RUN"
```

然后正式后台训练、DataB validation 校准并自动保活：

```bash
cd /input/workflow_58770161/workspace/test/cameramotion_det
ROOT=/tmp/1res/camera_discriminative_gate/v1
mkdir -p "$ROOT"
nohup env STAGE=all KEEP_ALIVE_AFTER_RUN=1 \
  bash scripts/camera_discriminative_gate/run_server_a.sh \
  > "$ROOT/server_a_launcher.log" 2>&1 &
echo "launcher pid: $!"
```

服务器 A 完成后，共享 NAS 中必须出现：

```bash
test -f /input/workflow_58770161/workspace/test/cameramotion_det/res/camera_discriminative_gate/v1/calibration/calibration.json && echo OK
find /input/workflow_58770161/workspace/test/cameramotion_det/res/camera_discriminative_gate/v1/model_bundle/models -name model.pt | wc -l
```

第二条应输出 `9`。随后在仍保存 ViF 特征的服务器 B 评测并保活：

```bash
cd /input/workflow_58770161/workspace/test/cameramotion_det
pkill -f '/input/training/busy.py' || true
ROOT=/tmp/1res/camera_discriminative_gate/v1
mkdir -p "$ROOT"
nohup env STAGE=eval_vif KEEP_ALIVE_AFTER_RUN=1 \
  bash scripts/camera_discriminative_gate/run_server_b.sh \
  > "$ROOT/server_b_launcher.log" 2>&1 &
echo "launcher pid: $!"
```

## 进度与结果

服务器 A：

```bash
watch -n 10 'find /tmp/1res/camera_discriminative_gate/v1/train/models -name model.pt 2>/dev/null | wc -l; for f in /tmp/1res/camera_discriminative_gate/v1/train_logs/job_*.log; do echo ===$f===; tail -n 1 "$f"; done'
```

服务器 B：

```bash
tail -f /tmp/1res/camera_discriminative_gate/v1/server_b_launcher.log
```

正式结果：

```bash
cat /input/workflow_58770161/workspace/test/cameramotion_det/res/camera_discriminative_gate/v1/eval/vifbench/camera_discriminative_gate_summary.json
```

## 预注册验收

相对 `zero_camera`，`matched` 必须满足：

1. ViF pooled AUROC 或跨生成器 Macro Balanced ACC 至少提高 1.0 点，另一项下降不超过 0.5 点。
2. matched 必须显著优于 shuffled camera，至少一项 paired bootstrap 95% CI 下界大于 0。
3. 至少两个 motion bucket 正向、过半受支持 generator 正向、至少 2/3 seed 的 AUROC 正向。
4. camera-only AUROC 不高于 0.65，且 matched 不低于 camera-only，避免把数据源捷径误报成相机交互收益。

通过只说明冻结相机特征对最终真假监督有独立增量，下一步才允许与冻结 Qwen 检测分数融合并在 GenBuster benchmark 复核。未通过则停止把 camera 作为论文主贡献，不继续 camera RL 或文本条件试参。
