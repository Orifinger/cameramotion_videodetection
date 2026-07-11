# 相机匹配局部反事实验证门

- 日期：2026-07-11
- 状态：代码完成，服务器 Gate 0 待执行
- 目标：先验证 DataA 是否具有真实、局部且不依赖相机运动捷径的生成信号，再决定是否训练局部反事实偏好模型。

## 1. 验证对象

这套代码不把 camera label/caption 输入检测模型。Camera 数据只承担三项作用：

1. 审计同源 DataA real/fake 是否保持相同全局相机运动；
2. 对 DataB Real/Fake 做 motion 分层配平，降低运动分布捷径；
3. 构造可选的 camera-aware 前置感知分支，与 local-only 分支做严格消融。

DataA 的真实 VACE `M_gen` mask 是正式局部监督。自动标注 CoT 中的 bbox 只允许用于诊断 fallback，不能代替正式 edit mask。

## 2. 新增代码

| 文件 | 用途 |
|---|---|
| `tools/build_dataa_counterfactual_gate_sets.py` | 对齐 detection、camera、grounded index 和真实 mask，生成无 bbox 泄漏的 A/B 双顺序 DPO/评测集 |
| `tools/dataa_counterfactual_signal_gate.py` | Gate 0：统计 mask 内外 real/fake 差异，检查数据是否真是局部编辑 |
| `tools/build_local_global_detection_replay.py` | 保持 DataA 完整 pair，并在相机粗签名内为 DataB 等量抽取 Real/Fake replay |
| `eval/eval_dataa_counterfactual_pair_gate.py` | Gate 1：评测局部编辑选择、A/B 偏置、swap consistency 和 bbox IoU |
| `eval/eval_counterfactual_transfer_gate.py` | Gate 2：联合判定 DataA 提升、VIF-Bench 保留和移动相机分桶提升 |
| `tests/test_counterfactual_validation_gates.py` | 合成同源视频、mask、偏置预测和迁移结果的端到端测试 |

## 3. 服务器预检

```bash
ROOT=/input/workflow_58770161/workspace/test/cameramotion_det
cd "${ROOT}"
git pull origin main

python -m unittest tests.test_counterfactual_validation_gates -v
```

先定位生成 grounded-CoT 时使用的输入索引：

```bash
find /input/workflow_58770161/workspace /tmp \
  -type f \
  \( -name '*grounded*cot*input*.jsonl' -o -name '*grounded*input*index*.jsonl' \) \
  2>/dev/null
```

该 JSONL 每条应至少含有：

```text
case_id
mask_npz
edit_bbox_xyxy
evidence_mask.mask_shape
real_video / fake_video
```

如果索引已丢失、但 VACE run root 和 `case_manifest.json` 仍在，可重建：

```bash
python scripts/dataa_v1/build_grounded_cot_input_index.py \
  --run-root /path/to/vace_run_root_1 \
  --run-root /path/to/vace_run_root_2 \
  --out-jsonl /tmp/1res/counterfactual_gate/data/dataa_grounded_input_index.jsonl \
  --out-summary /tmp/1res/counterfactual_gate/data/dataa_grounded_input_index_summary.json
```

## 4. 构建反事实数据

```bash
ROOT=/input/workflow_58770161/workspace/test/cameramotion_det
SPLIT_DIR=${ROOT}/tools/data/camera_motion_splits
OUT=/tmp/1res/counterfactual_gate
GROUNDED_INDEX=/path/to/dataa_grounded_input_index.jsonl

SPLIT_ARGS=(--dataa-test-json "${SPLIT_DIR}/dataA_test.json")
if test -f "${SPLIT_DIR}/dataA_train.json"; then
  SPLIT_ARGS+=(--dataa-train-json "${SPLIT_DIR}/dataA_train.json")
fi

python tools/build_dataa_counterfactual_gate_sets.py \
  --detection-json "${ROOT}/detection/dataa_vace_grounded_cot_instruct_tp8x2_sft_all.json" \
  --camera-jsonl "${ROOT}/camera/camerajson/dataa_cameramotion_labels_v2.jsonl" \
  --grounded-index-jsonl "${GROUNDED_INDEX}" \
  "${SPLIT_ARGS[@]}" \
  --out-dir "${OUT}/data" \
  --frames-per-video 8 \
  --require-true-mask \
  --check-mask-files \
  --seed 20260711
```

必须满足：

```text
formal_gate_eligible = true
train_test_case_overlap = []
user_prompts_contain_no_gt_bbox = true
camera_pair_mismatches = 0 或仅有可解释的极少数
```

主要输出：

```text
dataa_counterfactual_pair_manifest.jsonl
dataa_counterfactual_dpo_local_only.json
dataa_counterfactual_dpo_camera_aware.json
dataa_counterfactual_eval_local_only.json
dataa_counterfactual_eval_camera_aware.json
```

若暂时找不到真实 mask，可不传 `--grounded-index-jsonl` 和两个 mask 严格参数；生成的数据会标记为 `formal_gate_eligible=false`，只能检查格式，不能作为 Gate 0 结论。

## 5. Gate 0：局部信号有效性

先在 200 对上执行：

```bash
python tools/dataa_counterfactual_signal_gate.py \
  --pair-manifest-jsonl "${OUT}/data/dataa_counterfactual_pair_manifest.jsonl" \
  --out-dir "${OUT}/gate0_200" \
  --split train \
  --max-pairs 200 \
  --seed 20260711 \
  --workers 64 \
  --fail-on-gate
```

默认通过标准：

```text
有效 pair >= 90%
真实 mask 覆盖 >= 90%
camera 标签覆盖率 >= 90%
有 camera 标签的 pair 一致率 >= 98%
mask 内/外差异中位数比值 >= 2.0
mask 外平均绝对差异中位数 <= 0.03
至少 70% pair 的 mask 内差异高于 mask 外
```

通过 200 对后去掉 `--max-pairs 200` 跑全量。若 Gate 0 不通过，不进入任何 DPO/GRPO。

`unknown` 或缺失 camera 标签只降低标签覆盖率，不再被错误计为 Real/Fake camera 标签冲突。`--max-pairs` 使用固定 seed 随机抽样。

## 6. 构建局部/全生成 replay

```bash
python tools/build_local_global_detection_replay.py \
  --dataa-detection-json "${ROOT}/detection/dataa_vace_grounded_cot_instruct_tp8x2_sft_all.json" \
  --datab-detection-json /input/workflow_58770161/workspace/test/camb/camerabenchdataB-main/detection/v4vif_2766busterall_trainall.json \
  --datab-camera-jsonl /input/workflow_58770161/workspace/test/camb/camerabenchdataB-main/datab_cameramotion_labels_final/datab_cameramotion_labels_v2.jsonl \
  "${SPLIT_ARGS[@]}" \
  --out-dir "${OUT}/replay" \
  --require-target \
  --seed 20260711
```

该脚本在 `motion dynamics × speed × steadiness` 内分别等量抽取 DataB Real/Fake。Camera 文本不会进入任何 user/system prompt。

## 7. Gate 1：训练后局部配对验收

对 `dataa_counterfactual_eval_local_only.json` 跑现有 DataA pair inference，得到 prediction JSON 或 rank shard 目录后执行：

```bash
python eval/eval_dataa_counterfactual_pair_gate.py \
  --gt-json "${OUT}/data/dataa_counterfactual_eval_local_only.json" \
  --pred-json /path/to/pair_prediction_json_or_dir \
  --out-dir "${OUT}/gate1_pair_eval" \
  --fail-on-gate
```

默认通过标准：选择准确率不低于 70%、swap consistency 不低于 85%、预测 A 比例在 45% 至 55%、mean bbox IoU 不低于 0.30、两种格式正确率均不低于 95%。

## 8. Gate 2：检测迁移验收

每个分支先用既有 `eval_dataa.py` 生成 DataA summary/items，再用 `summarize_detection_by_camera_motion.py` 生成 motion summary。最后运行：

```bash
python eval/eval_counterfactual_transfer_gate.py \
  --control-dataa-summary /path/control_dataa_summary.json \
  --control-motion-summary /path/control_motion_summary.json \
  --control-vif-acc 0.8396 --control-vif-f1 0.8472 \
  --pair-dataa-summary /path/pair_dataa_summary.json \
  --pair-motion-summary /path/pair_motion_summary.json \
  --pair-vif-acc 0.00 --pair-vif-f1 0.00 \
  --camera-dataa-summary /path/camera_pair_dataa_summary.json \
  --camera-motion-summary /path/camera_pair_motion_summary.json \
  --camera-vif-acc 0.00 --camera-vif-f1 0.00 \
  --out "${OUT}/gate2_transfer_summary.json" \
  --fail-on-gate
```

将示例中的 `0.00` 替换为实际 VIF-Bench 指标。默认要求：pair-only 相对等步数 detection replay 的 DataA Balanced ACC 或 Fake F1 提升至少 3 点；VIF ACC/F1 各下降不超过 1 点；`minor-motion` 或 `complex-motion` 至少一个指标提升 1 点。Camera 只有在相对 pair-only 再提高至少 1 点时，才能作为论文核心贡献。

## 9. 当前停止点

本轮先执行数据构建和 Gate 0。Gate 0 结果未返回前，不写正式训练配置，也不把 fallback bbox 当作真实 edit mask。
