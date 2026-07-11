# 相机补偿局部轨迹最小验证

## 1. 验证问题

这个实验只回答一个问题：在使用完全相同的密集原视频帧和局部监督时，显式估计并补偿全局相机运动，是否能比不补偿相机运动更可靠地检测 DataA 局部生成区域。

它不是 Qwen3-VL 训练，不使用外部 camera caption，不运行 DPO、GRPO 或其他强化学习，也暂时不使用 DataB。只有这个低成本视觉闸门通过，才讨论如何把局部证据分支接入 MLLM 和通用全生成视频训练。

## 2. 最终 DataA 数据契约

后续只接受 `40step_v3` 统一数据：

- case/证据记录：`res/dataA_v1/autolabel/dataa_vace_grounded_cot_v4_records_40step_v3.jsonl`；
- detection 数据：`res/dataA_v1/autolabel/dataa_vace_grounded_cot_40step_v3_sft_clean.json`；
- camera motion：`camera/camerajson/dataa_cameramotion_labels_40step_v3.jsonl`；
- 统一抽帧目录：`/tmp/cameramotion_det/dataA_v1/autolabel/dataa_vace_grounded_cot_frames_40step_v3`。

允许的原视频 run root 只有：

1. `/tmp/cameramotion_det/dataA_v1/vace14b/dataa_v1_subject_first_vace14b_finalv1`，198 cases；
2. `/tmp/cameramotion_det/dataA_v1/vace13b/dataa_v1_dataset_v2_vace13b_40step_v3`，714 cases；
3. `/tmp/cameramotion_det/dataA_v1/vace13b/dataa_v1_textedit_reserve_vace13b_40step_v3`，168 cases。

合计必须为 1080 cases。旧 VACE-1.3B 16-step run 一旦混入，manifest 构建器直接失败。

旧 held-out test 的 case 身份继续保留。对新 1080 cases，旧 test case id 仍归 test，其余 case 归 train；不能因为视频重新生成而把旧 test case 放回训练。

## 3. 为什么读取原视频

现有 detection 数据每个视频均匀抽取 16 帧，相邻间隔经常达到 0.24 至 1.0 秒。它们不是适合光流估计的真正相邻帧。

正式提取流程从 `real_video` 和 `fake_video` 读取完整视频：

- 最高按 8 FPS 密集采样；
- 16 帧组成约 2 秒窗口；
- 窗口步长 8 帧；
- 只在短窗口内累积相机变换；
- overlapping windows 复用一次提取的帧级特征。

原来的 16 帧只保留作稀疏输入对照，以及后续给 Qwen 选择证据帧。相机补偿与不补偿两组必须使用同一批密集帧。

## 4. 方法与三个对照

### 全局 ReStraV 基线

冻结 DINOv2 ViT-S/14，对每帧 CLS 特征计算步长、方向变化和汇总统计，形成 21 维视频窗口特征，再按 ReStraV 官方配置训练 `64→32→1` 的小 MLP。

### 局部轨迹但不补偿相机

保留 DINOv2 patch token，在固定图像坐标计算局部轨迹距离与转向变化；同时使用未分解的总光流幅值。DataA train 的 VACE mask 只作为 patch 监督。

### 相机补偿后的局部轨迹

TorchVision RAFT-Large 对密集相邻帧估计前后向光流。使用前后向一致点和 MAGSAC/RANSAC 拟合主导单应性，并在失败时依次退化到仿射和平移。Real 和 Fake 独立估计相机运动，不使用配对 Real 帮助 Fake，也不在拟合时使用 GT mask。

然后把每一帧 DINOv2 patch 特征变换到窗口首帧坐标，计算补偿后的局部轨迹；光流输入改为减去全局相机场后的残余幅值。DataA mask 通过 `case_manifest` 中 canonical-to-source frame mapping 精确映射到原视频时间，不能按 16 帧位置均匀近似。

## 5. 离线权重

- TorchVision RAFT-Large：`/home/admin/raft_large_C_T_SKHT_V2-ff5fadd5.pth`；
- DINOv2 Small：默认 `/home/admin/dinov2-small`；
- SEA-RAFT：`/home/admin/MemorySlices/Tartan-C-T-TSKH-spring540x960-M`。

第一轮只使用 TorchVision RAFT-Large。SEA-RAFT 权重先做文件预检，只有第一轮视觉闸门通过后才作为正式后端确认；这样不会把环境适配和方法有效性混在一起。

## 6. 服务器执行

```bash
ROOT=/input/workflow_58770161/workspace/test/cameramotion_det
cd "${ROOT}"

export TEST_SPLIT=${ROOT}/tools/data/camera_motion_splits/dataA_test.json
export OUT=/tmp/1res/camera_flow_probe_40step_v3

bash scripts/camera_flow_probe/run_camera_flow_probe.sh preflight
```

如果 `/home/admin/dinov2-small` 不是实际 Hugging Face 目录，显式指定内部模型目录：

```bash
export DINO_MODEL=/.aistudio/aistudio-modelhub/zeta/f94249_32800136/hugging_face/facebook__dinov2-small
bash scripts/camera_flow_probe/run_camera_flow_probe.sh preflight
```

预检必须满足：

- case 数量 1080；
- 三个来源分别为 198、714、168；
- held-out test 为 321 cases；
- camera Real/Fake 标签一致且无缺失；
- 原视频、mask NPZ 和 case manifest 全部可读；
- RAFT、DINOv2、SEA-RAFT 离线权重都可解析。

随后只跑每个 motion bucket 两个 train case：

```bash
bash scripts/camera_flow_probe/run_camera_flow_probe.sh smoke
```

查看：

```bash
python -m json.tool \
  /tmp/1res/camera_flow_probe_40step_v3/smoke/extraction_audit.json
```

同时人工检查：

```text
/tmp/1res/camera_flow_probe_40step_v3/smoke/visualizations/
```

每个面板依次显示原始帧、下一帧、RAFT 总光流、拟合的全局相机场、补偿后帧、像素残差、去除全局运动后的光流残差和仅用于审计的 GT mask。GT mask 不参与光流或相机拟合。

smoke 通过且 GPU 前向正常后，才进行完整 16 卡提取：

```bash
NPROC_PER_NODE=16 \
bash scripts/camera_flow_probe/run_camera_flow_probe.sh extract

bash scripts/camera_flow_probe/run_camera_flow_probe.sh audit
```

提取审计通过后训练三个轻量探针：

```bash
bash scripts/camera_flow_probe/run_camera_flow_probe.sh probe
```

核心结果：

```text
/tmp/1res/camera_flow_probe_40step_v3/probe/camera_aligned_local_probe_summary.json
/tmp/1res/camera_flow_probe_40step_v3/probe/camera_aligned_local_probe_predictions.csv
```

## 7. 分阶段验收

### 提取验收

- 有效 feature case 覆盖率至少 95%；
- 至少 90% case 能映射出正 mask patch；
- 相机拟合 inlier rate 中位数至少 50%；
- 同源 Real/Fake 独立估计后的角点运动差异中位数不超过图像对角线 2%；
- feature 文件不能含 NaN 或 Inf。

同源 Real/Fake 相机差异只用于质量诊断，不进入模型输入。

### 方法验收

以未补偿局部探针为唯一 camera 增量对照：

- 整体视频 AUC 至少提高 3 个百分点；
- `complex-motion` AUC 至少提高 3 个百分点；
- `no-motion` AUC 下降不超过 2 个百分点；
- case bootstrap 的 AUC 差值 95% 置信区间下界高于 0。

同时报告：视频 AUC、Balanced Accuracy、Fake Recall、成对 Fake>Real 准确率、patch AUC、patch IoU 和 pointing-game accuracy。

只有全部通过才进入 DataB 弱监督迁移和 Qwen 证据注入。若相机补偿不优于未补偿局部探针，则停止该方法，不改成反复试光流模型或调 RL。

## 8. 当前限制

- 第一轮的光流后端是标准 RAFT，不是最终方法贡献；
- DataA 是同源局部编辑数据，第一闸门只能建立局部可学习性和 camera 增量，不能建立 VIF-Bench 泛化；
- 新 40-step 视频重新生成后，旧 Gate 0、Gate 1 的数值不能直接当作新数据结果；
- smoke 为功能和几何质量检查，不是方法结果；
- 本轮不修改 `docs/final_experiment_plan_20260708.md`。
