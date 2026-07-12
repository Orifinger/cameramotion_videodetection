# 相机感知同源配对排序第一项验证执行说明

## 这次测什么

这次只测试一个问题：同一 DataA case 的 Real/Fake 视频分别独立计算真假分数时，在普通二分类损失上增加 `Fake 分数高于 Real 分数` 的配对 margin，是否比等数据量、等步数的普通二分类续训更好。

它不测试 camera pretext 的收益，不训练长 CoT，不使用 camera caption、bbox、mask、RAFT 或 DINO 特征，也不运行 DPO/GRPO。

## 公平对照

两个模型都从 `/tmp/1res/v4vif_2766busterall_trainall_5epoch/checkpoint-2115` 开始，使用相同的：

- 按视频来源和 camera bucket 分层选择的 256 个 DataA train pairs；
- 512 条 Real/Fake 平衡的 DataB detection replay；
- 16 帧输入、verdict prompt、LoRA rank 32、学习率和 64 optimizer steps；
- DataA pair step 与 DataB replay step 交替的训练顺序。

唯一差异是：普通独立判别续训对照只使用二分类损失；相机分层同源配对排序方法额外使用权重 `0.2`、margin `0.5` 的 pair loss。

Real/Fake 从未放进同一个 A/B prompt。它们只是作为 batch 中两条独立序列前向，配对关系只进入 loss。

## 数据与泄漏限制

- DataA detection：`res/dataA_v1/autolabel/dataa_vace_grounded_cot_40step_v3_sft_clean.json`
- DataA camera：`camera/camerajson/dataa_cameramotion_labels_40step_v3.jsonl`
- 固定 dev 身份：`tools/data/camera_motion_splits/dataA_test.json`。若服务器存在 `dataA_train.json` 就直接读取；否则 train 严格取完整 case 减去 dev，结果等价且会写入泄漏审计。
- DataB replay：`/input/workflow_58770161/workspace/test/camb/camerabenchdataB-main/detection/v4vif_2766busterall_trainall.json`

当前 321 个 DataA test case 已被多次用于方案诊断，本实验明确把它们称为开发集，不把结果描述成未经使用的论文最终测试。初始 checkpoint 已看过完整 DataB，因此 DataB 也不能作为真正 held-out 测试。

## 验收线

相对普通对照，配对排序方法必须同时满足：

- DataA 开发集视频 AUC 提升至少 3 个百分点；
- pair accuracy 提升至少 5 个百分点；
- `complex-motion` AUC 提升至少 3 个百分点；
- 任一 VACE 来源的 AUC 下降不超过 2 个百分点；
- 后续 VIF-Bench 保留测试下降不超过 1.5 个百分点。

DataA 部分通过后状态只记为 `DataA 通过、VIF 保留待测`。VIF 保留也通过后才进入 camera pretext 正确标签/打乱标签对照。

## 服务器执行顺序

先确认依赖：

```bash
cd /input/workflow_58770161/workspace/test/cameramotion_det
python - <<'PY'
import peft, qwen_vl_utils, torch, transformers
print('torch', torch.__version__)
print('transformers', transformers.__version__)
print('peft', peft.__version__)
print('dependencies: OK')
PY
```

构建数据并执行单卡两步 smoke：

```bash
STAGE=build bash scripts/caspr_gate1/run_caspr_gate1.sh
STAGE=smoke bash scripts/caspr_gate1/run_caspr_gate1.sh
```

smoke 成功后依次训练和评分：

```bash
STAGE=train_control bash scripts/caspr_gate1/run_caspr_gate1.sh
STAGE=train_method bash scripts/caspr_gate1/run_caspr_gate1.sh
STAGE=score_base bash scripts/caspr_gate1/run_caspr_gate1.sh
STAGE=score_control bash scripts/caspr_gate1/run_caspr_gate1.sh
STAGE=score_method bash scripts/caspr_gate1/run_caspr_gate1.sh
STAGE=eval bash scripts/caspr_gate1/run_caspr_gate1.sh
```

最终先查看：

```bash
cat /tmp/1res/caspr_gate1/eval/caspr_gate1_dataa_summary.json
```

如果 DataA 门通过，再分别合并两个 adapter，使用既有 `infer2_5_3.sh` 对相同 VIF-Bench 条目和相同 prompt 做保留测试。合并模型约 17GB，属于本轮可丢弃验证文件，保存在 `/tmp`，不用上传 OSS。
