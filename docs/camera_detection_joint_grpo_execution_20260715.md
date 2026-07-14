# 检测主导的相机中间变量联合 SFT/GRPO 执行说明

## 一句话定义

模型在同一次生成中先从有序帧输出相机运动标签，再输出最终 `Real/Fake`；相机标签只是可验证的中间变量，实验是否通过只由无外部相机文本条件下的真假检测指标决定。

## 为什么这轮实验合理

1. 先前相机 VQA、相机-only PPRL 与检测是分开的目标，不能保证相机能力参与真假决策。本轮把 `<camera_motion>` 和 `<answer>` 放进同一条 rollout，序列级检测奖励会同时更新中间标签和最终答案对应的策略 token。
2. 三个 GRPO 分支共用同一检测模型、相机能力 adapter、联合输出 warm SFT、训练样本和推理提示词。唯一改变是相机奖励使用正确标签、打乱标签或完全不计相机正确性。
3. 正确/打乱分支的奖励是 `0.65 × Real/Fake + 0.30 × camera set F1 + 0.05 × format`。因为 `0.65 > 0.30 + 0.05`，检测错误无法靠相机与格式满分补偿。
4. 打乱相机标签在 `DataA/DataB × Real/Fake` 内保持边际分布；DataA 同一 real/fake pair 仍共享同一打乱标签。因此正确相机分支必须同时超过仅检测和打乱相机两个等算力对照，才能支持相机监督有增量。
5. 第一轮不生成自由文本 CoT。自由解释目前没有可靠逐句真值，加入它只会扩大输出空间和 reward hacking 面；检测主效应成立后再恢复解释，并继续以 Real/Fake 为主指标。

这不是成功保证。它是当前成本下最直接的因果门：最终以 ViF-Bench 的 Real/Fake 三对照为开发主门，不再用相机 VQA 分数解释检测失败。DataA 只诊断局部编辑迁移，不阻断通用检测复核。

## 数据与输出

- 公共训练集：1024 条，DataA 512 条（256 个完整 real/fake pair）、DataB 512 条，Real/Fake 各 512 条。
- DataA 开发门：324 个 case、648 条 Real/Fake 记录，和 DataA 训练 case 零重叠。
- 通用开发门：完整 ViF-Bench，不向模型提供 camera caption、label 或外部相机模型结果。
- 一次性大文件：merged model、rollout、逐样本预测放 `/tmp/1res/camera_detection_joint_grpo/v1`。
- 持久化小结果：摘要、评测 JSON/CSV、紧凑日志放 `/input/workflow_58770161/workspace/test/cameramotion_det/res/camera_detection_joint_grpo/v1`。
- 可复用大文件：公共 warm adapter 和三个 GRPO adapter 自动上传到 `oss://antsys-tamper/public/wong/skyra/selfcot/camerabench/ourexp/camera_detection_joint_grpo/v1/`。

## 服务器需更新的文件

从 GitHub 手动复制下列文件，并覆盖到相同的项目相对路径：

```text
tools/build_camera_detection_joint_grpo.py
tools/build_camera_joint_sft_gate.py
tools/audit_camera_pprl_smoke.py
rl/camera_detection_rewards.py
scripts/camera_detection_joint_grpo/__init__.py
scripts/camera_detection_joint_grpo/run.sh
scripts/camera_detection_joint_grpo/summarize.py
scripts/camera_joint_sft_gate/summarize_dataa.py
scripts/camera_joint_sft_gate/summarize_vif_four_model.py
scripts/camera_detection_retention/run_vifbench.sh
scripts/camera_detection_retention/vifbench_retention.py
scripts/caspr_gate1/merge_adapter.py
scripts/caspr_gate1/runtime.py
prompts/camera_detection_joint_grpo/system_prompt.txt
prompts/camera_detection_joint_grpo/user_suffix.txt
docs/camera_detection_joint_grpo_execution_20260715.md
docs/camera_conditioned_experiment_log.md
```

项目部署根目录固定为：

```bash
/input/workflow_58770161/workspace/test/cameramotion_det
```

## 第一层：无模型推理预检与数据构建

```bash
cd /input/workflow_58770161/workspace/test/cameramotion_det
RUN=scripts/camera_detection_joint_grpo/run.sh

STAGE=preflight bash "$RUN"
STAGE=build bash "$RUN"

cat /tmp/1res/camera_detection_joint_grpo/v1/data/camera_detection_joint_grpo_data_summary.json
```

必须看到：1024 条训练记录、DataA/DataB 各 512、Real/Fake 各 512、DataA train/eval overlap 为空、打乱标签改变率至少 80%，并且全部图片存在。预检只查环境、文件、ms-swift 参数、奖励注册和 16 张 GPU，不加载模型做十分钟试跑。

## 第二层：工程 smoke

若希望逐步观察，按顺序运行：

```bash
STAGE=smoke_sft bash "$RUN"
STAGE=train_warm_sft bash "$RUN"
BRANCH=correct_camera STAGE=smoke_grpo bash "$RUN"
```

正确相机 GRPO smoke 的 `mean_frac_reward_zero_std` 必须不高于 `0.80`。未通过就停止，不进入三个正式分支。

## 第三层：三个正式分支、DataA 诊断与 ViF-Bench 主门

推荐直接后台串行执行完整流程。若前面已单独完成 warm SFT，脚本会复用，不会重训正式 warm adapter；smoke 会重新跑一次但规模很小。DataA 无论通过与否都会继续 ViF-Bench，因为通用全生成检测才是这轮方法选择的主目标。

```bash
cd /input/workflow_58770161/workspace/test/cameramotion_det
RUN=scripts/camera_detection_joint_grpo/run.sh
ROOT=/tmp/1res/camera_detection_joint_grpo/v1
mkdir -p "$ROOT"

nohup env \
STAGE=all_full \
AUTO_UPLOAD_OSS=1 \
KEEP_ALIVE_AFTER_RUN=1 \
bash "$RUN" \
> "$ROOT/launcher.log" 2>&1 &

echo "launcher pid: $!"
```

查看训练状态：

```bash
tail -f /tmp/1res/camera_detection_joint_grpo/v1/launcher.log
```

DataA 诊断完成后可读取：

```bash
cat /tmp/1res/camera_detection_joint_grpo/v1/dataa_eval/camera_detection_joint_grpo_dataa_summary.json
```

DataA 汇总会标记 `passed` 或 `failed`，但两种状态都不会中止后续 ViF-Bench。它用于判断局部编辑迁移与通用检测是否一致，不承担方法总开关。

## 第四层：单独补跑 ViF-Bench

如果第三层使用的是 `STAGE=all_full`，无需再执行本节。若此前只运行了 `STAGE=all_dataa`，无论 DataA 状态如何，都用下面命令补跑通用检测主门：

```bash
cd /input/workflow_58770161/workspace/test/cameramotion_det
RUN=scripts/camera_detection_joint_grpo/run.sh
ROOT=/tmp/1res/camera_detection_joint_grpo/v1

nohup env \
STAGE=vif_all \
AUTO_UPLOAD_OSS=1 \
KEEP_ALIVE_AFTER_RUN=1 \
bash "$RUN" \
> "$ROOT/vif_launcher.log" 2>&1 &

echo "launcher pid: $!"
```

完成后读取：

```bash
cat /tmp/1res/camera_detection_joint_grpo/v1/vif_eval/camera_detection_joint_grpo_vif_summary.json
```

ViF 汇总只有 `status: camera_candidate` 才冻结训练配方，转到未用于选模的 GenBuster-200K `benchmark` 集；ViF-Bench 已用于开发，不作为最终未见测试。若 ViF 不通过，即使 DataA 通过也不把方法判为成功。

`camera_candidate` 和 `no_camera_gain` 都属于正常完成的实验结果，脚本都会在归档后执行 `/input/training/keep.sh`。只有环境、数据、训练、推理或 smoke 工程检查报错才以非零状态退出。

## 分支单独执行

三条训练命令分别是：

```bash
STAGE=train_correct bash "$RUN"
STAGE=train_detection_only bash "$RUN"
STAGE=train_shuffled bash "$RUN"
```

三条 ViF-Bench 命令必须先正确相机分支，因为后两个分支复用它生成的共同 warm-start 预测：

```bash
STAGE=vif_correct bash "$RUN"
STAGE=vif_detection_only bash "$RUN"
STAGE=vif_shuffled bash "$RUN"
STAGE=summarize_vif bash "$RUN"
```

## 手动 OSS 兜底

脚本默认自动上传 compact adapters。若自动上传失败，只对实际存在且通过文件审计的目录执行对应单条命令，例如：

```bash
ossutil64 cp -r /tmp/1res/camera_detection_joint_grpo/v1/artifacts/correct_camera_adapter/ oss://antsys-tamper/public/wong/skyra/selfcot/camerabench/ourexp/camera_detection_joint_grpo/v1/correct_camera_adapter/
```

不要上传 merged models、rollout 或逐样本 ViF 预测；这些都是可重建的验证产物。
